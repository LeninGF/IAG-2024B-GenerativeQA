#!/usr/bin/env bash
# Run a single fine-tuning/ablation job on multiple GPUs via torchrun (DDP).
#
# Use this when you want one job (e.g. the F1 plateau test) to use more than
# one GPU. For many independent experiments across GPUs, keep using
# scripts/run_ablation_parallel.sh instead.
#
# Usage:
#   bash scripts/run_ablation_multi_gpu.sh \
#       --gpus "4 5 6 7" \
#       --model mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es \
#       --dataset merged --mode ft --epochs 30 \
#       --hf-dataset LeninGF/question-answering-robbery-m2 \
#       --output-dir out_experiments/f1_plateau_4gpu
#
# All arguments except --gpus are passed through to scripts/run_qa_ablation.py.
set -euo pipefail

GPUS=""
MASTER_PORT="${MASTER_PORT:-29500}"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$GPUS" ]]; then
  echo "Error: --gpus is required, e.g. --gpus \"4 5 6 7\"" >&2
  exit 1
fi

# Accept both space-separated ("4 5 6 7") and comma-separated ("4,5,6,7").
GPU_LIST=$(echo "$GPUS" | tr ',' ' ')
read -r -a GPU_ARRAY <<< "$GPU_LIST"
N_GPUS=${#GPU_ARRAY[@]}
CUDA_LIST=$(IFS=,; echo "${GPU_ARRAY[*]}")

echo "Multi-GPU DDP job: $N_GPUS GPU(s) [$CUDA_LIST]"
echo "Command: torchrun --nproc_per_node=${N_GPUS} scripts/run_qa_ablation.py --gpus ${CUDA_LIST} ${PASSTHROUGH[*]}"

CUDA_VISIBLE_DEVICES="$CUDA_LIST" torchrun \
  --nproc_per_node="$N_GPUS" \
  --master_port="$MASTER_PORT" \
  scripts/run_qa_ablation.py \
  --gpus "$CUDA_LIST" \
  "${PASSTHROUGH[@]}"
