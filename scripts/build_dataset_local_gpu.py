"""Full-scale local-GPU dataset build for the robbery QA SQuAD-style dataset.

Reuses filtering logic from dataset_build.ipynb (82-150 word contexts) and the
JSON-schema generation from local_qa_generation.py. Always reprocesses the
full filtered dataset.

Usage:
    python scripts/build_dataset_local_gpu.py --model qwen2.5-7b-instruct \
        --gpu-ids 4,5 --output-file dataset/dataset_squad_v2_localgpu.json

    # Build and push the result to Hugging Face in the same run:
    python scripts/build_dataset_local_gpu.py --model qwen2.5-7b-instruct \
        --gpu-ids 4,5 --push-to-hub --repo-id LeninGF/robos-question-answering

    # Push an already-generated JSONL without touching the GPU/model:
    python scripts/build_dataset_local_gpu.py --push-only \
        --input-file dataset/dataset_squad_v2_localgpu_20260827_120000.json \
        --repo-id LeninGF/robos-question-answering

NOTE: this script has not been executed in this environment (no local GPU
here). It is meant to be run and validated on a GPU-equipped machine.
"""
import argparse
import os
import sys
from datetime import datetime

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
    parser.add_argument("--model", choices=list(MODEL_CHOICES),
                         help="Local instruct model to use for generation (required unless --push-only)")
    parser.add_argument("--gpu-ids",
                         help="Comma-separated physical GPU ids dedicated to this run, e.g. '0' or '4,5' "
                              "(required unless --push-only)")
    parser.add_argument("--dataset-path", default="LeninGF/autotrain-data-robberyclassification",
                         help="Source Hugging Face dataset with the raw 'relato' texts")
    parser.add_argument("--output-file", default=None,
                         help="Output JSONL file for the new dataset (default: dataset/dataset_squad_v2_localgpu_<timestamp>.json)")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of contexts to process (for smoke tests)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max-memory-gib", type=int, default=12,
                         help="Per-GPU memory cap (GiB), used only for 2-GPU balanced models")
    parser.add_argument("--push-only", action="store_true",
                         help="Skip GPU/model loading and generation; just push --input-file to --repo-id")
    parser.add_argument("--push-to-hub", action="store_true",
                         help="After a full build, also push --output-file to --repo-id")
    parser.add_argument("--input-file", default=None,
                         help="JSONL dataset to push, used only with --push-only")
    parser.add_argument("--repo-id", default=None,
                         help="Destination Hugging Face dataset repo, e.g. 'LeninGF/robos-question-answering'")
    args = parser.parse_args()

    if args.push_only:
        if not args.input_file:
            parser.error("--push-only requires --input-file")
        if not args.repo_id:
            parser.error("--push-only requires --repo-id")
    else:
        if not args.model:
            parser.error("--model is required unless --push-only is set")
        if not args.gpu_ids:
            parser.error("--gpu-ids is required unless --push-only is set")
        if args.push_to_hub and not args.repo_id:
            parser.error("--push-to-hub requires --repo-id")
        if args.output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_file = os.path.join("dataset", f"dataset_squad_v2_localgpu_{timestamp}.json")

    return args


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


def push_dataset_to_hub(jsonl_path, repo_id):
    """Load a JSONL dataset and push it to a Hugging Face dataset repo, same pattern as dataset_xplore_and_upload.ipynb."""
    from datasets import load_dataset
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN not found. Add it to a .env file at the repo root."
        )
    login(hf_token)

    ds = load_dataset("json", data_files=jsonl_path, split="train")
    print(f"Subiendo {len(ds)} filas de '{jsonl_path}' a '{repo_id}' ...")
    ds.push_to_hub(repo_id)
    print("Subida completa!")


def main():
    args = parse_args()

    if args.push_only:
        push_dataset_to_hub(args.input_file, args.repo_id)
        return

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

    if args.push_to_hub:
        push_dataset_to_hub(args.output_file, args.repo_id)


if __name__ == "__main__":
    main()
