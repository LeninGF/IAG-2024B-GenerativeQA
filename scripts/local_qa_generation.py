"""Local-GPU utilities for building the SQuAD-style robbery QA dataset.

Reuses the prompt/schema design from dataset_build.ipynb, but runs inference
locally via transformers + bitsandbytes (4-bit) instead of the Hugging Face
Inference API, and uses `outlines` for schema-constrained JSON generation
instead of manual `json.loads` parsing (see dataset_xplore_and_upload.ipynb
for the hallucination/JSONDecodeError issues this avoids).

This module is shared by dataset_build_local_gpu.ipynb (pilot/model
comparison) and scripts/build_dataset_local_gpu.py (full-scale run).
"""
import json
import os
import time
from functools import wraps

import outlines
import torch
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

# Model configs verified in testing-pyt-ml-eqa-fge-*.ipynb: 1-GPU models use
# AutoModelForCausalLM/AutoTokenizer; gemma-3-4b-it is multimodal-capable and
# requires Gemma3ForConditionalGeneration/AutoProcessor instead.
MODEL_REGISTRY = {
    "qwen2.5-3b-instruct": {
        "hf_name": "Qwen/Qwen2.5-3B-Instruct",
        "model_class": AutoModelForCausalLM,
        "processor_class": AutoTokenizer,
        "torch_dtype": torch.float16,
        "num_gpus": 1,
    },
    "qwen2.5-7b-instruct": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_class": AutoModelForCausalLM,
        "processor_class": AutoTokenizer,
        "torch_dtype": torch.float16,
        "num_gpus": 2,
    },
    "gemma-3-1b-it": {
        "hf_name": "google/gemma-3-1b-it",
        "model_class": AutoModelForCausalLM,
        "processor_class": AutoTokenizer,
        "torch_dtype": torch.float16,
        "num_gpus": 1,
    },
    "gemma-3-4b-it": {
        "hf_name": "google/gemma-3-4b-it",
        "model_class": Gemma3ForConditionalGeneration,
        "processor_class": AutoProcessor,
        "torch_dtype": torch.bfloat16,
        "num_gpus": 2,
    },
}

# Same 5 questions used in dataset_build.ipynb, kept unchanged for comparability.
PREGUNTAS_COMUNES = [
    "¿Qué objetos fueron robados?",
    "¿En qué fecha ocurrió el incidente?",
    "¿A qué hora sucedió el robo?",
    "¿En qué dirección o entre qué calles sucedió el robo, suceso o incidente?",
    "¿Qué valor en dólares tenían los objetos sustraídos o robados?",
]


class AnswerSchema(BaseModel):
    answer_text: str
    is_impossible: str


def find_start_end_answer(context, answer):
    """Locate `answer` inside `context`; flags cases where it cannot be found."""
    impossible_flag = False
    start = context.find(answer)
    if start != -1:
        end = start + len(answer)
    else:
        start = 0
        end = 0
        impossible_flag = True
    return start, end, impossible_flag


def safe_json_loads(response_str):
    """Fallback parser, used only if constrained generation is bypassed."""
    try:
        return json.loads(response_str)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Response string:", response_str)
        return {"answer_text": "", "is_impossible": "imposible"}


def get_available_devices():
    """List CUDA devices (e.g. ["cuda:0", "cuda:1"]); GPU is required, this never falls back to CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Local inference for this project is designed "
            "for GPU (4-bit quantization + transformers); running on CPU is not "
            "supported. Please run on a CUDA-enabled machine."
        )
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def gpu_memory(device_id):
    """Print allocated/reserved/total VRAM for a logical GPU index; shared by the notebook and benchmark script."""
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    print(
        f"cuda:{device_id} | "
        f"allocated: {allocated:.2f} GiB | "
        f"reserved: {reserved:.2f} GiB | "
        f"total: {total:.2f} GiB"
    )


def load_local_model(model_key, gpu_ids, quantize_4bit=True, max_memory_gib=12):
    """Load a registered model onto the given logical GPU ids, wrapped by outlines for JSON-constrained generation.

    `gpu_ids` are logical indices (0-based) assumed already remapped by the
    caller via `CUDA_VISIBLE_DEVICES`, set *before* this module was imported
    (torch is imported at module load time, so setting the env var here would
    be too late). Single-GPU models take one id; 2-GPU models use
    `device_map="balanced"` across both ids given.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key '{model_key}'. Choices: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[model_key]
    if len(gpu_ids) != entry["num_gpus"]:
        raise ValueError(
            f"'{model_key}' requires {entry['num_gpus']} gpu id(s), got {gpu_ids}"
        )

    model_kwargs = {"torch_dtype": entry["torch_dtype"], "low_cpu_mem_usage": True}
    if entry["num_gpus"] == 1:
        model_kwargs["device_map"] = {"": gpu_ids[0]}
    else:
        model_kwargs["device_map"] = "balanced"
        model_kwargs["max_memory"] = {gid: f"{max_memory_gib}GiB" for gid in gpu_ids}
        model_kwargs["max_memory"]["cpu"] = "100GiB"
    if quantize_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=entry["torch_dtype"],
        )

    print(f"Loading {entry['hf_name']} on logical gpu ids {gpu_ids} ...")
    hf_model = entry["model_class"].from_pretrained(entry["hf_name"], **model_kwargs)
    processor = entry["processor_class"].from_pretrained(entry["hf_name"])
    return outlines.from_transformers(hf_model, processor)


def build_prompt(context, question):
    """Same prompt template used in dataset_build.ipynb's generate_squad_entry."""
    return f"""
    Eres un científico de datos que construye un dataset estilo SQuAD en español.
    Tu tarea es extraer del siguiente contexto la respuesta exacta para la pregunta.

    - Contexto: {context}

    - Pregunta: {question}

    ### Instrucciones:
    1. Responde SOLO con el fragmento exacto del contexto que corresponde a la pregunta.
    2. Si la pregunta no puede responderse con el contexto, escribe 'imposible' en el campo 'is_impossible'. Caso contrario coloca 'respondido'
    """


def generate_squad_entry_local(context, questions, model, context_id=None):
    """Local-GPU equivalent of generate_squad_entry from dataset_build.ipynb, using outlines for guaranteed-valid JSON."""
    answer_list = []
    for question in questions:
        prompt = build_prompt(context, question)
        result = model(prompt, AnswerSchema)
        response_json = AnswerSchema.model_validate_json(result).model_dump()
        start_position, end_position, impossible_flag = find_start_end_answer(
            context=context, answer=response_json["answer_text"]
        )
        dataset_details = {
            "context": context,
            "question": question,
            "answer_start": start_position,
            "answer_end": end_position,
            "impossible_find_answer": impossible_flag,
        }
        response_full = {**dataset_details, **response_json}
        if context_id is not None:
            response_full["context_id"] = context_id
        answer_list.append(response_full)
    return answer_list


def retry(max_retries=3, delay=5):
    """Retry decorator for local generation (no API rate limit, only transient GPU/parse errors)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_retries:
                        raise Exception(f"Fallo después de {max_retries} reintentos") from e
                    print(f"Error ({e}), reintentando ({attempts}/{max_retries})...")
                    time.sleep(delay * attempts)
        return wrapper
    return decorator


@retry(max_retries=2, delay=10)
def process_single_context_local(context, questions, model, context_id=None):
    return generate_squad_entry_local(context, questions, model, context_id=context_id)


def process_full_dataset_local(
    dataset,
    output_file,
    model,
    questions=None,
    checkpoint_interval=100,
    id_prefix="context",
    context_id_fn=None,
):
    """Process every context in `dataset` and append the generated QA entries to `output_file`.

    `context_id_fn(idx)` maps a row index in `dataset` to a global context_id.
    Required when `dataset` is a re-indexed subset (e.g. `Dataset.select(...)`)
    so ids stay aligned with the original filtered_ds.
    Defaults to `f"{id_prefix}_{idx}"` (matches the original notebook).
    """
    questions = questions if questions is not None else PREGUNTAS_COMUNES
    context_id_fn = context_id_fn or (lambda idx: f"{id_prefix}_{idx}")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        for idx in range(len(dataset)):
            context_id = context_id_fn(idx)

            try:
                context = dataset[idx]["relato"]
                results = process_single_context_local(context, questions, model, context_id=context_id)

                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")

                if idx % checkpoint_interval == 0:
                    f.flush()
                    print(f"Checkpoint guardado en contexto {idx}")

            except Exception as e:
                print(f"Error crítico en contexto {idx}: {str(e)}")
                with open("errores_v2.log", "a", encoding="utf-8") as err_log:
                    err_log.write(f"{context_id}\t{str(e)}\n")

    print("Procesamiento completo!")
