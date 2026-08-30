"""Full-scale local-GPU dataset build for the robbery QA SQuAD-style dataset.

Reuses filtering logic from dataset_build.ipynb (82-150 word contexts) and the
JSON-schema generation from local_qa_generation.py. Always reprocesses the
full filtered dataset unless --use-dataset-sample is set.

Usage:
    python scripts/build_dataset_local_gpu.py --model qwen2.5-7b-instruct \
        --gpu-ids 4,5 --output-file dataset/dataset_squad_v2_localgpu.json

    # Build and push the result to Hugging Face in the same run:
    python scripts/build_dataset_local_gpu.py --model qwen2.5-7b-instruct \
        --gpu-ids 4,5 --push-to-hub --repo-id LeninGF/robos-question-answering

    # Resume an interrupted build (skips contexts already fully written):
    python scripts/build_dataset_local_gpu.py --model qwen2.5-7b-instruct \
        --gpu-ids 4,5 --output-file dataset/dataset_squad_v2_localgpu.json \
        --resume

    # Parallel replica build: launch N independent workers, one per GPU, each
    # writing --output-file.worker-ID, then merge the shards:
    for i in 0 1 2 3; do
      python scripts/build_dataset_local_gpu.py --model gemma-3-1b-it \
          --gpu-ids $i --worker-id $i --num-workers 4 \
          --output-file dataset/squadv2_gemma.jsonl &
    done
    python scripts/build_dataset_local_gpu.py --merge \
        --output-file dataset/squadv2_gemma.jsonl

    # Resume an interrupted replica build while changing the number of GPUs/workers:
    # use scripts/plan_resume_shards.py to generate one --resume-manifest per new
    # GPU (see README.md), then each worker is launched like this:
    python scripts/build_dataset_local_gpu.py --model gemma-3-1b-it \
        --gpu-ids 0 --worker-id 0 \
        --resume-manifest resume_manifests/gemma-3-1b-it_20260829_120000/worker_0.json \
        --output-file dataset/squadv2_gemma.jsonl

    # Reproducible random sample (default 10000 contexts when --use-dataset-sample):
    python scripts/build_dataset_local_gpu.py --model qwen2.5-3b-instruct \
        --gpu-ids 4 --use-dataset-sample --sample-size 5000 --sample-seed 42 \
        --output-file dataset/squadv2_qwen_sample.jsonl

    # Use a larger token budget for long literal answers (e.g. lists of stolen
    # objects); default is 128 for backward compatibility:
    python scripts/build_dataset_local_gpu.py --model qwen2.5-3b-instruct \
        --gpu-ids 4 --use-dataset-sample --sample-size 5000 --max-new-tokens 256 \
        --output-file dataset/squadv2_qwen_sample.jsonl

    # Push an already-generated JSONL without touching the GPU/model:
    python scripts/build_dataset_local_gpu.py --push-only \
        --input-file dataset/dataset_squad_v2_localgpu_20260827_120000.json \
        --repo-id LeninGF/robos-question-answering

NOTE: this script has not been executed in this environment (no local GPU
here). It is meant to be run and validated on a GPU-equipped machine.
"""
import argparse
import glob
import json
import os
import random
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

# Number of questions per context (must match local_qa_generation.PREGUNTAS_COMUNES).
# Kept here so --merge can validate shards without importing torch.
QUESTIONS_PER_CONTEXT = 5

# Default sample size used when --use-dataset-sample is set without --sample-size.
DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_SAMPLE_SEED = 42


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
    parser.add_argument("--worker-id", type=int, default=None,
                         help="Index of this worker process (0-based). Requires --num-workers.")
    parser.add_argument("--num-workers", type=int, default=None,
                         help="Total number of worker processes launched for this model/GPU replica run.")
    parser.add_argument("--merge", action="store_true",
                         help="Merge --output-file.worker-* shards into --output-file and validate completeness.")
    parser.add_argument("--use-dataset-sample", action="store_true",
                         help="Process a random sample of the filtered dataset instead of all contexts.")
    parser.add_argument("--sample-size", type=int, default=None,
                         help="Number of contexts to sample (default 10000 when --use-dataset-sample is set).")
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED,
                         help="Seed for reproducible random sampling (default 42).")
    parser.add_argument("--resume", action="store_true",
                         help="Skip contexts already fully written in --output-file and clean partial contexts before continuing")
    parser.add_argument("--resume-manifest", default=None,
                         help="JSON file (list of {filtered_idx, context_id}) produced by "
                              "scripts/plan_resume_shards.py; bypasses --num-workers/--use-dataset-sample and "
                              "processes exactly the listed filtered-dataset indices. Requires --worker-id.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max-memory-gib", type=int, default=12,
                         help="Per-GPU memory cap (GiB), used only for 2-GPU balanced models")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                         help="Maximum number of new tokens per generated answer (default 128)")
    parser.add_argument("--max-retries", type=int, default=2,
                         help="Number of retries per question after the first attempt (default 2)")
    parser.add_argument("--retry-delay", type=int, default=3,
                         help="Seconds to wait between retries (default 3; increase to 10 when using cloud/API-backed generation)")
    parser.add_argument("--push-only", action="store_true",
                         help="Skip GPU/model loading and generation; just push --input-file to --repo-id")
    parser.add_argument("--push-to-hub", action="store_true",
                         help="After a full build, also push --output-file to --repo-id")
    parser.add_argument("--input-file", default=None,
                         help="JSONL dataset to push, used only with --push-only")
    parser.add_argument("--repo-id", default=None,
                         help="Destination Hugging Face dataset repo, e.g. 'LeninGF/robos-question-answering'")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be >= 0")

    if args.merge:
        if not args.output_file:
            parser.error("--merge requires --output-file")
        if args.push_only:
            parser.error("--merge and --push-only are mutually exclusive")
        if args.push_to_hub:
            parser.error("--merge cannot be combined with --push-to-hub; push after merging")
        if args.model or args.gpu_ids:
            parser.error("--merge cannot be combined with --model/--gpu-ids")
        return args

    if args.resume_manifest:
        if args.worker_id is None:
            parser.error("--resume-manifest requires --worker-id (used to name the output shard)")
        if args.num_workers is not None:
            parser.error("--resume-manifest cannot be combined with --num-workers (indices come from the manifest)")
        if args.use_dataset_sample:
            parser.error("--resume-manifest cannot be combined with --use-dataset-sample (the manifest already selects exact indices)")
        if args.push_to_hub:
            parser.error("--push-to-hub cannot be combined with --resume-manifest; merge shards first, then push")
    elif (args.worker_id is None) != (args.num_workers is None):
        parser.error("--worker-id and --num-workers must be used together")
    if args.num_workers is not None:
        if args.num_workers < 1:
            parser.error("--num-workers must be >= 1")
        if not (0 <= args.worker_id < args.num_workers):
            parser.error("--worker-id must be between 0 and --num-workers - 1")
        if args.push_to_hub:
            parser.error("--push-to-hub cannot be combined with --num-workers; merge shards first, then push")

    if args.sample_size is not None and not args.use_dataset_sample:
        parser.error("--sample-size requires --use-dataset-sample")
    if args.use_dataset_sample and args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be >= 1")

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


def sample_dataset_indices(dataset_len, sample_size, seed):
    """Return a sorted random subset of indices (0..dataset_len-1), reproducible via seed."""
    rng = random.Random(seed)
    size = min(sample_size, dataset_len)
    return sorted(rng.sample(range(dataset_len), size))


def merge_shards(output_file):
    """Merge --output-file.worker-* shards into output_file and validate completeness."""
    shard_files = sorted(glob.glob(f"{output_file}.worker-*"))
    if not shard_files:
        raise FileNotFoundError(
            f"No se encontraron shards para merge: {output_file}.worker-*"
        )

    lines = []
    counts = {}
    for shard_path in shard_files:
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                context_id = obj.get("context_id")
                if context_id is not None:
                    counts[context_id] = counts.get(context_id, 0) + 1
                lines.append(stripped + "\n")

    incomplete = [cid for cid, n in counts.items() if n != QUESTIONS_PER_CONTEXT]
    if incomplete:
        preview = ", ".join(map(str, incomplete[:20]))
        extra = f" ... y {len(incomplete) - 20} más" if len(incomplete) > 20 else ""
        raise RuntimeError(
            f"Merge falló: contextos incompletos (no tienen las {QUESTIONS_PER_CONTEXT} preguntas): {preview}{extra}"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Merge OK: {len(counts)} contextos, {sum(counts.values())} filas -> {output_file}")


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

    if args.merge:
        merge_shards(args.output_file)
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

    # Manifest mode (from scripts/plan_resume_shards.py) selects exact filtered_ds indices
    # directly, bypassing --use-dataset-sample/--num-workers entirely. Kept as a fully
    # separate branch so the stride-based path below is untouched.
    if args.resume_manifest:
        with open(args.resume_manifest, "r", encoding="utf-8") as mf:
            manifest_entries = json.load(mf)
        dataset_to_process = filtered_ds.select([entry["filtered_idx"] for entry in manifest_entries])
        manifest_context_ids = [entry["context_id"] for entry in manifest_entries]
        context_id_fn = lambda i, cids=manifest_context_ids: cids[i]  # noqa: E731
        output_file = f"{args.output_file}.worker-{args.worker_id}"
        print(
            f"Resume-manifest: {len(dataset_to_process)} contextos asignados desde "
            f"{args.resume_manifest} -> {output_file}"
        )
    else:
        sampled_indices = None
        dataset_to_process = filtered_ds
        if args.use_dataset_sample:
            sample_size = args.sample_size if args.sample_size is not None else DEFAULT_SAMPLE_SIZE
            sampled_indices = sample_dataset_indices(len(filtered_ds), sample_size, args.sample_seed)
            dataset_to_process = filtered_ds.select(sampled_indices)
            print(
                f"Muestra aleatoria: {len(sampled_indices)} contextos "
                f"(seed {args.sample_seed}, size {len(sampled_indices)})"
            )

        output_file = args.output_file
        if args.num_workers is not None:
            worker_indices = list(range(args.worker_id, len(dataset_to_process), args.num_workers))
            dataset_to_process = dataset_to_process.select(worker_indices)
            output_file = f"{args.output_file}.worker-{args.worker_id}"
            if sampled_indices is not None:
                context_id_fn = lambda i, wi=worker_indices, si=sampled_indices: f"context_{si[wi[i]]}"  # noqa: E731
            else:
                context_id_fn = lambda i, wi=worker_indices: f"context_{wi[i]}"  # noqa: E731
            print(
                f"Worker {args.worker_id}/{args.num_workers}: {len(dataset_to_process)} contextos -> {output_file}"
            )
        elif sampled_indices is not None:
            context_id_fn = lambda i, si=sampled_indices: f"context_{si[i]}"  # noqa: E731
        else:
            context_id_fn = lambda i: f"context_{i}"  # noqa: E731

    logical_gpu_ids = list(range(len(gpu_ids)))
    model = load_local_model(
        args.model,
        gpu_ids=logical_gpu_ids,
        quantize_4bit=not args.no_4bit,
        max_memory_gib=args.max_memory_gib,
    )

    process_full_dataset_local(
        dataset=dataset_to_process,
        output_file=output_file,
        model=model,
        questions=PREGUNTAS_COMUNES,
        checkpoint_interval=args.checkpoint_interval,
        model_name=args.model,
        resume=args.resume,
        context_id_fn=context_id_fn,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    if args.push_to_hub:
        push_dataset_to_hub(output_file, args.repo_id)


if __name__ == "__main__":
    main()
