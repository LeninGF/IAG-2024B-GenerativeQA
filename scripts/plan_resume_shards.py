"""Plan a resume-with-worker-reassignment for an interrupted run_build_parallel.sh job.

Standalone, CPU-only helper: scans the existing per-worker shards of a
build_dataset_local_gpu.py replica run, figures out which sampled contexts are
still missing, and repartitions ONLY the missing ones across a (possibly
different) number of GPUs. Existing shards are backed up and then cleaned of
partial contexts, but are otherwise reused as-is (already-completed contexts
are never regenerated).

This script only imports and reuses build_dataset_local_gpu.py /
local_qa_generation.py; it does not modify their --num-workers stride logic or
resume-scanning code. It assumes the original run used --use-dataset-sample
(as the M2 commands in research.org did).

Usage:
    python scripts/plan_resume_shards.py \
        --model gemma-3-1b-it \
        --output-file ../../data/dataset_squadv2_M2/squadv2_gemma-3-1b.jsonl \
        --sample-size 20000 --sample-seed 42 \
        --new-gpus 0,1,2,3,4,5,6,7 \
        --max-new-tokens 512

This prints a summary (total/completed/remaining contexts) and writes, under
dataset/resume_manifests/<model>_<timestamp>/:
    worker_0.json ... worker_{N-1}.json   (one manifest per new GPU)
    launch_resume.sh                      (ready-to-run launcher, one process per GPU)
"""
import argparse
import glob
import json
import os
import shutil
import stat
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_SCRIPT = os.path.join(REPO_ROOT, "scripts", "build_dataset_local_gpu.py")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset_local_gpu import (  # noqa: E402
    build_filtered_dataset,
    sample_dataset_indices,
)
from local_qa_generation import PREGUNTAS_COMUNES, scan_resume_state  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan a resume with worker reassignment: reuse completed contexts, "
                     "repartition the rest across a new set of GPUs."
    )
    parser.add_argument("--model", required=True,
                         help="Model key used in the original run, e.g. gemma-3-1b-it")
    parser.add_argument("--output-file", required=True,
                         help="Base --output-file of the original run (shards are <output-file>.worker-*)")
    parser.add_argument("--dataset-path", default="LeninGF/autotrain-data-robberyclassification",
                         help="Must match --dataset-path of the original run")
    parser.add_argument("--sample-size", type=int, required=True,
                         help="Must match --sample-size of the original run")
    parser.add_argument("--sample-seed", type=int, default=42,
                         help="Must match --sample-seed of the original run (default 42)")
    parser.add_argument("--new-gpus", required=True,
                         help="Comma-separated physical GPU ids currently free, e.g. '0,1,2,3,4,5,6,7'. "
                              "Length can differ from the original run; pass exactly what is free.")
    parser.add_argument("--manifest-dir", default=None,
                         help="Where to write manifests/launcher (default: "
                              "dataset/resume_manifests/<model>_<timestamp>)")
    parser.add_argument("--log-dir", default=None,
                         help="Where the generated launcher writes worker logs (default: "
                              "logs/<model>_resume_<timestamp>; never reuses a previous --log-dir)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=int, default=3)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-memory-gib", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    new_gpus = [g.strip() for g in args.new_gpus.split(",") if g.strip()]
    if not new_gpus:
        raise SystemExit("ERROR: --new-gpus no puede estar vacío")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = args.manifest_dir or os.path.join(
        REPO_ROOT, "dataset", "resume_manifests", f"{args.model}_{timestamp}"
    )
    log_dir = args.log_dir or os.path.join(REPO_ROOT, "logs", f"{args.model}_resume_{timestamp}")
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Rebuild the exact same filtered + sampled dataset used by the original run.
    filtered_ds = build_filtered_dataset(args.dataset_path)
    sampled_indices = sample_dataset_indices(len(filtered_ds), args.sample_size, args.sample_seed)
    print(f"Muestra original reconstruida: {len(sampled_indices)} contextos (seed {args.sample_seed})")

    # 2. Back up and scan existing shards to find globally-completed context_ids.
    shard_pattern = f"{args.output_file}.worker-*"
    shard_files = sorted(glob.glob(shard_pattern))
    if not shard_files:
        print(f"AVISO: no se encontraron shards existentes para {shard_pattern}; "
              "se tratará como si nada estuviera completo todavía.")

    completed_ids = set()
    for shard_path in shard_files:
        backup_path = f"{shard_path}.bak_{timestamp}"
        shutil.copy2(shard_path, backup_path)
        shard_completed, cleaned = scan_resume_state(shard_path, PREGUNTAS_COMUNES)
        completed_ids |= shard_completed
        print(f"{shard_path}: {len(shard_completed)} contextos completos, {cleaned} parciales limpiados "
              f"(backup: {backup_path})")

    # 3. Figure out which sample positions are still missing.
    remaining_positions = [
        pos for pos, filtered_idx in enumerate(sampled_indices)
        if f"context_{filtered_idx}" not in completed_ids
    ]
    print(
        f"Total muestra: {len(sampled_indices)} | ya completos: "
        f"{len(sampled_indices) - len(remaining_positions)} | restantes: {len(remaining_positions)}"
    )
    if not remaining_positions:
        print("Nada pendiente: todos los contextos de la muestra ya están completos.")
        return

    # 4. Repartition the remaining positions across the new GPU count.
    num_new_workers = len(new_gpus)
    manifest_paths = []
    for worker_id in range(num_new_workers):
        worker_positions = remaining_positions[worker_id::num_new_workers]
        manifest_entries = [
            {"filtered_idx": sampled_indices[pos], "context_id": f"context_{sampled_indices[pos]}"}
            for pos in worker_positions
        ]
        manifest_path = os.path.join(manifest_dir, f"worker_{worker_id}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_entries, f, ensure_ascii=False)
        manifest_paths.append(manifest_path)
        print(f"worker {worker_id} (GPU {new_gpus[worker_id]}): {len(manifest_entries)} contextos -> {manifest_path}")

    # 5. Generate the ready-to-run launcher script.
    launcher_path = os.path.join(manifest_dir, "launch_resume.sh")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for worker_id, (gpu, manifest_path) in enumerate(zip(new_gpus, manifest_paths)):
        log_path = os.path.join(log_dir, f"{args.model}_worker_{worker_id}.log")
        cmd = (
            f'nohup python "{BUILD_SCRIPT}" --model {args.model} '
            f"--gpu-ids {gpu} --worker-id {worker_id} "
            f'--resume-manifest "{manifest_path}" '
            f'--output-file "{args.output_file}" '
            f"--dataset-path {args.dataset_path} "
            f"--max-new-tokens {args.max_new_tokens} --max-retries {args.max_retries} "
            f"--retry-delay {args.retry_delay} --max-memory-gib {args.max_memory_gib} "
            + ("--no-4bit " if args.no_4bit else "")
            + f'> "{log_path}" 2>&1 &'
        )
        lines.append(f"echo 'Lanzando worker {worker_id}/{num_new_workers - 1} en GPU {gpu} ...'")
        lines.append(cmd)
    lines.append("")
    lines.append("wait")
    lines.append(
        f'echo "Todos los workers terminaron. Corre ahora: '
        f'python \\"{BUILD_SCRIPT}\\" --merge --output-file \\"{args.output_file}\\""'
    )
    with open(launcher_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(launcher_path, os.stat(launcher_path).st_mode | stat.S_IEXEC)

    print(f"\nLauncher listo: bash {launcher_path}")
    print(f"Logs de esta corrida se escribirán en: {log_dir}")


if __name__ == "__main__":
    main()
