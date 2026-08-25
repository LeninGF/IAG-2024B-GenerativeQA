"""Benchmark local-GPU load time and QA-generation throughput for a model config.

Standalone from the dataset-build pipeline: reports model load time, per-context
generation time, aggregate throughput, and GPU memory usage, so the 4 model
configs (Qwen 3B/7B, Gemma 1B/4B) can be compared before committing to a
full-scale dataset build.

Usage:
    python scripts/benchmark_local_gpu.py --model gemma-3-4b-it --gpu-ids 6,7 --num-samples 5

NOTE: this script has not been executed in this environment (no local GPU
here). It is meant to be run and validated on a GPU-equipped machine.
"""
import argparse
import json
import os
import sys
import time

# Kept as a plain list (not imported from local_qa_generation.MODEL_REGISTRY)
# so argparse can be set up before CUDA_VISIBLE_DEVICES is set and torch is imported.
MODEL_CHOICES = ["qwen2.5-3b-instruct", "qwen2.5-7b-instruct", "gemma-3-1b-it", "gemma-3-4b-it"]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark local-GPU QA generation")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES,
                         help="Local instruct model to benchmark")
    parser.add_argument("--gpu-ids", required=True,
                         help="Comma-separated physical GPU ids dedicated to this run, e.g. '0' or '4,5'")
    parser.add_argument("--dataset-path", default="LeninGF/autotrain-data-robberyclassification")
    parser.add_argument("--num-samples", type=int, default=10,
                         help="Number of contexts to benchmark generation over")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max-memory-gib", type=int, default=12,
                         help="Per-GPU memory cap (GiB), used only for 2-GPU balanced models")
    parser.add_argument("--results-file", default="benchmark_results.jsonl",
                         help="JSONL file to append this run's summary to")
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    # Must happen before any torch import in this process (local_qa_generation imports torch at module load).
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    sys.path.insert(0, os.path.dirname(__file__))
    from local_qa_generation import (  # noqa: E402
        MODEL_REGISTRY,
        PREGUNTAS_COMUNES,
        gpu_memory,
        load_local_model,
        process_single_context_local,
    )
    from build_dataset_local_gpu import build_filtered_dataset  # noqa: E402

    entry = MODEL_REGISTRY[args.model]
    if len(gpu_ids) != entry["num_gpus"]:
        raise ValueError(
            f"'{args.model}' requires {entry['num_gpus']} gpu id(s), got {len(gpu_ids)} ({args.gpu_ids})"
        )

    filtered_ds = build_filtered_dataset(args.dataset_path, limit=args.num_samples)
    print(f"Benchmarking over {len(filtered_ds)} contexts")

    logical_gpu_ids = list(range(len(gpu_ids)))
    for logical_id in logical_gpu_ids:
        gpu_memory(logical_id)

    load_start = time.perf_counter()
    model = load_local_model(
        args.model,
        gpu_ids=logical_gpu_ids,
        quantize_4bit=not args.no_4bit,
        max_memory_gib=args.max_memory_gib,
    )
    load_time_s = time.perf_counter() - load_start
    print(f"Model load time: {load_time_s:.2f}s")

    for logical_id in logical_gpu_ids:
        gpu_memory(logical_id)

    context_times = []
    for idx in range(len(filtered_ds)):
        context = filtered_ds[idx]["relato"]
        start = time.perf_counter()
        process_single_context_local(context, PREGUNTAS_COMUNES, model, context_id=f"bench_{idx}")
        context_times.append(time.perf_counter() - start)

    avg_context_time_s = sum(context_times) / len(context_times)
    contexts_per_min = 60.0 / avg_context_time_s

    summary = {
        "model": args.model,
        "gpu_ids": args.gpu_ids,
        "quantize_4bit": not args.no_4bit,
        "num_samples": len(filtered_ds),
        "load_time_s": round(load_time_s, 2),
        "avg_context_time_s": round(avg_context_time_s, 3),
        "contexts_per_min": round(contexts_per_min, 2),
    }
    print("Summary:", json.dumps(summary, indent=2))

    with open(args.results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
