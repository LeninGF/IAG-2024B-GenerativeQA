"""Full-scale local-GPU dataset build for the robbery QA SQuAD-style dataset.

Reuses filtering logic from dataset_build.ipynb (82-150 word contexts) and the
JSON-schema generation from local_qa_generation.py. Always reprocesses the
full filtered dataset.

Usage:
    python scripts/build_dataset_local_gpu.py --model qwen2.5-7b-instruct \
        --gpu-ids 4,5 --output-file dataset/dataset_squad_v2_localgpu.json

NOTE: this script has not been executed in this environment (no local GPU
here). It is meant to be run and validated on a GPU-equipped machine.
"""
import argparse
import os
import sys

# Kept as a plain string map (not imported from local_qa_generation.MODEL_REGISTRY)
# so argparse can be set up before CUDA_VISIBLE_DEVICES is set and torch is imported.
MODEL_CHOICES = {
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
}

MIN_WORDS = 82
MAX_WORDS = 150


def parse_args():
    parser = argparse.ArgumentParser(description="Build the local-GPU robbery QA dataset")
    parser.add_argument("--model", required=True, choices=list(MODEL_CHOICES),
                         help="Local instruct model to use for generation")
    parser.add_argument("--gpu-ids", required=True,
                         help="Comma-separated physical GPU ids dedicated to this run, e.g. '0' or '4,5'")
    parser.add_argument("--dataset-path", default="LeninGF/autotrain-data-robberyclassification",
                         help="Source Hugging Face dataset with the raw 'relato' texts")
    parser.add_argument("--output-file", default=os.path.join("dataset", "dataset_squad_v2_localgpu.json"),
                         help="Output JSONL file for the new dataset")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of contexts to process (for smoke tests)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max-memory-gib", type=int, default=12,
                         help="Per-GPU memory cap (GiB), used only for 2-GPU balanced models")
    return parser.parse_args()


def build_filtered_dataset(dataset_path, limit=None):
    """Same word-count filter (82-150 words) as dataset_build.ipynb, not an API artifact."""
    from datasets import load_dataset

    ds = load_dataset(dataset_path)

    def count_words(sample):
        sample["word_count"] = len(sample["relato"].split())
        return sample

    ds = ds.map(count_words, batched=False)
    filtered = ds["train"].filter(
        lambda batch: [MIN_WORDS <= x <= MAX_WORDS for x in batch["word_count"]],
        batched=True,
        batch_size=1000,
    )
    if limit is not None:
        filtered = filtered.select(range(min(limit, len(filtered))))
    return filtered


def main():
    args = parse_args()
    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    # Must happen before any torch import in this process (local_qa_generation imports torch at module load).
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    sys.path.insert(0, os.path.dirname(__file__))
    from local_qa_generation import (  # noqa: E402
        MODEL_REGISTRY,
        PREGUNTAS_COMUNES,
        load_local_model,
        process_full_dataset_local,
    )
    from dotenv import load_dotenv  # noqa: E402
    from huggingface_hub import login  # noqa: E402

    entry = MODEL_REGISTRY[args.model]
    if len(gpu_ids) != entry["num_gpus"]:
        raise ValueError(
            f"'{args.model}' requires {entry['num_gpus']} gpu id(s), got {len(gpu_ids)} ({args.gpu_ids})"
        )

    # Reads HUGGINGFACE_TOKEN from a .env file at the repo root (never commit the token).
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN not found. Add it to a .env file at the repo root."
        )
    login(hf_token)

    filtered_ds = build_filtered_dataset(args.dataset_path, limit=args.limit)
    print(f"Contextos disponibles tras filtro {MIN_WORDS}-{MAX_WORDS} palabras: {len(filtered_ds)}")

    logical_gpu_ids = list(range(len(gpu_ids)))
    model = load_local_model(
        args.model,
        gpu_ids=logical_gpu_ids,
        quantize_4bit=not args.no_4bit,
        max_memory_gib=args.max_memory_gib,
    )

    process_full_dataset_local(
        dataset=filtered_ds,
        output_file=args.output_file,
        model=model,
        questions=PREGUNTAS_COMUNES,
        checkpoint_interval=args.checkpoint_interval,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
