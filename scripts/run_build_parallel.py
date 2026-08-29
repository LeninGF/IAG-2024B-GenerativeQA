"""Launch N parallel replica workers of build_dataset_local_gpu.py (one per GPU).

This orchestrator targets the 1-GPU model replicas used for the quality
comparison (qwen2.5-3b-instruct and gemma-3-1b-it): each physical GPU runs one
independent worker on a different slice of the (optionally sampled) filtered
dataset. After all workers finish, it merges the shards unless --no-merge is
passed.

Usage examples:
    # Launch 4 Gemma workers on GPUs 4-7, wait, and merge:
    python scripts/run_build_parallel.py --model gemma-3-1b-it \
        --gpus 4,5,6,7 \
        --use-dataset-sample --sample-size 17568 --sample-seed 42 \
        --output-file dataset/squadv2_gemma_sample.jsonl

    # Same, but with a larger per-answer token budget:
    python scripts/run_build_parallel.py --model qwen2.5-3b-instruct \
        --gpus 0,1,2,3 --max-new-tokens 256 \
        --output-file dataset/squadv2_qwen_sample.jsonl

    # Print the worker commands without launching them:
    python scripts/run_build_parallel.py --model qwen2.5-3b-instruct \
        --gpus 0,1,2,3 --output-file dataset/squadv2_qwen_sample.jsonl \
        --dry-run
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_SCRIPT = os.path.join(REPO_ROOT, "scripts", "build_dataset_local_gpu.py")

# Only 1-GPU models are supported by the replica worker mode. The 2-GPU models
# (qwen2.5-7b-instruct, gemma-3-4b-it) use device_map="balanced" and are not
# meant to be replicated one-per-GPU by this orchestrator.
SINGLE_GPU_MODELS = {"qwen2.5-3b-instruct", "gemma-3-1b-it"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch parallel worker replicas of build_dataset_local_gpu.py and merge shards"
    )
    parser.add_argument("--model", required=True, choices=sorted(SINGLE_GPU_MODELS),
                        help="1-GPU instruct model to use")
    parser.add_argument("--gpus", required=True,
                        help="Comma-separated physical GPU ids, one per worker, e.g. '4,5,6,7'")
    parser.add_argument("--output-file", required=True,
                        help="Final merged JSONL path (workers write --output-file.worker-ID)")
    parser.add_argument("--dataset-path", default="LeninGF/autotrain-data-robberyclassification",
                        help="Source Hugging Face dataset with the raw 'relato' texts")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of contexts to process (for smoke tests)")
    parser.add_argument("--use-dataset-sample", action="store_true",
                        help="Process a random sample of the filtered dataset instead of all contexts")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of contexts to sample (default 10000 when --use-dataset-sample is set)")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Seed for reproducible random sampling (default 42)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted workers (skips contexts already fully written)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max-memory-gib", type=int, default=12,
                        help="Per-GPU memory cap (GiB), only used by 2-GPU balanced models")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum number of new tokens per generated answer (default 128)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Number of retries per question after the first attempt (default 2)")
    parser.add_argument("--retry-delay", type=int, default=3,
                        help="Seconds to wait between retries (default 3; increase to 10 when using cloud/API-backed generation)")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory where worker logs are written (default: logs)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Do not run the merge step after workers finish")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the worker/merge commands without launching them")
    args = parser.parse_args()

    if args.sample_size is not None and not args.use_dataset_sample:
        parser.error("--sample-size requires --use-dataset-sample")
    if args.use_dataset_sample and args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be >= 0")

    return args


def build_worker_cmd(args, gpu, worker_id, num_workers):
    cmd = [
        sys.executable,
        BUILD_SCRIPT,
        "--model", args.model,
        "--gpu-ids", gpu,
        "--worker-id", str(worker_id),
        "--num-workers", str(num_workers),
        "--output-file", args.output_file,
        "--dataset-path", args.dataset_path,
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.use_dataset_sample:
        cmd += ["--use-dataset-sample"]
        if args.sample_size is not None:
            cmd += ["--sample-size", str(args.sample_size)]
        cmd += ["--sample-seed", str(args.sample_seed)]
    if args.resume:
        cmd += ["--resume"]
    if args.no_4bit:
        cmd += ["--no-4bit"]
    cmd += ["--max-memory-gib", str(args.max_memory_gib)]
    cmd += ["--max-new-tokens", str(args.max_new_tokens)]
    cmd += ["--max-retries", str(args.max_retries)]
    cmd += ["--retry-delay", str(args.retry_delay)]
    return cmd


def main():
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("ERROR: --gpus no puede estar vacío")
    num_workers = len(gpus)

    worker_cmds = [
        build_worker_cmd(args, gpu, worker_id, num_workers)
        for worker_id, gpu in enumerate(gpus)
    ]
    merge_cmd = [sys.executable, BUILD_SCRIPT, "--merge", "--output-file", args.output_file]

    if args.dry_run:
        print("Dry run: comandos que se ejecutarían")
        for cmd in worker_cmds:
            print("  " + " ".join(cmd))
        if not args.no_merge:
            print("Merge:")
            print("  " + " ".join(merge_cmd))
        return

    os.makedirs(args.log_dir, exist_ok=True)
    log_paths = [
        os.path.join(args.log_dir, f"{args.model}_worker_{worker_id}.log")
        for worker_id in range(num_workers)
    ]

    procs = []
    for worker_id, (cmd, log_path) in enumerate(zip(worker_cmds, log_paths)):
        print(f"Lanzando worker {worker_id}/{num_workers - 1} en GPU {gpus[worker_id]} ...")
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        procs.append((proc, log_file))

    failed = False
    for worker_id, (proc, log_file) in enumerate(procs):
        returncode = proc.wait()
        log_file.close()
        if returncode != 0:
            print(f"Worker {worker_id} falló con código {returncode}; log: {log_paths[worker_id]}")
            failed = True
        else:
            print(f"Worker {worker_id} OK; log: {log_paths[worker_id]}")

    if failed:
        raise SystemExit("ERROR: al menos un worker falló; no se ejecutó el merge.")

    if args.no_merge:
        print("Workers terminados. No se ejecutó merge (--no-merge).")
        return

    print("Merge:", " ".join(merge_cmd))
    result = subprocess.run(merge_cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"ERROR: el merge falló con código {result.returncode}")
    print("Proceso paralelo completado.")


if __name__ == "__main__":
    main()
