#!/usr/bin/env bash
# Launch the default QA ablation matrix (4 models x 3 datasets x zsl/ft) on up
# to N GPUs. Each experiment is an independent process, so jobs are distributed
# round-robin over the available GPUs and run concurrently in waves.
#
# Usage:
#   bash scripts/run_ablation_parallel.sh \
#       --output-dir out_experiments/run1 \
#       --gpus "0 1 2 3 4 5 6 7"
#
#   # Print what would be launched without running anything:
#   bash scripts/run_ablation_parallel.sh --dry-run --gpus "0 1"
#
# The launcher is a convenience wrapper around scripts/run_qa_ablation.py.
# To run a custom matrix, generate commands with:
#   python scripts/run_qa_ablation.py --plan-only --plan-gpus 8 \
#       --output-dir out_experiments/run1
set -euo pipefail

OUT_DIR="${OUT_DIR:-out_experiments/run_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

read -r -a GPU_ARRAY <<< "$GPUS"
N_GPUS=${#GPU_ARRAY[@]}
if [[ $N_GPUS -eq 0 ]]; then
  echo "No GPUs provided" >&2
  exit 1
fi

MODELS=(
  "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
  "MMG/bert-base-spanish-wwm-cased-finetuned-squad2-es"
  "deepset/xlm-roberta-base-squad2"
  "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
)
DATASETS=("merged" "strict_gemma" "strict_qwen")
MODES=("zsl" "ft")

mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"
echo "GPUs: ${GPU_ARRAY[*]} ($N_GPUS)"
echo "Experiments: $(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#MODES[@]} ))"

launch_experiment() {
  local model="$1"
  local dataset="$2"
  local mode="$3"
  local gpu="$4"
  local tag
  tag=$(echo "$model" | tr '/' '_')
  local log="$OUT_DIR/${tag}__${dataset}__${mode}.log"

  echo "GPU ${gpu}: ${model} | ${dataset} | ${mode}"
  local cmd="python scripts/run_qa_ablation.py --model ${model} --dataset ${dataset} --mode ${mode} --gpu ${gpu} --output-dir ${OUT_DIR}"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  CUDA_VISIBLE_DEVICES=${gpu} ${cmd}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" nohup ${cmd} > "${log}" 2>&1 &
  fi
}

counter=0
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
      gpu="${GPU_ARRAY[$((counter % N_GPUS))]}"
      launch_experiment "$model" "$dataset" "$mode" "$gpu"
      counter=$((counter + 1))
      if [[ $DRY_RUN -eq 0 && $((counter % N_GPUS)) -eq 0 ]]; then
        echo "Wave of ${N_GPUS} jobs done, waiting..."
        wait
      fi
    done
  done
done

if [[ $DRY_RUN -eq 0 ]]; then
  wait
  echo "All experiments finished. Results in ${OUT_DIR}"
else
  echo "Dry run complete (nothing executed)."
fi
