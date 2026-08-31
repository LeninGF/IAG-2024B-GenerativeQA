#!/usr/bin/env bash
# Launch the default QA ablation matrix on up to N GPUs.
#
# Two modes:
#   A) Dataset-variant ablation (default): reads the QC JSONL files on disk
#      (merged, strict_gemma, strict_qwen) and runs 4 models x 3 datasets x
#      {zsl, ft} = 24 experiments.
#   B) Final merged dataset (gold-safe): pass --data-dir or --hf-dataset to
#      use the prepared/HF splits; then the matrix becomes 4 models x merged x
#      {zsl, ft} = 8 experiments.
#
# Usage:
#   # Option A (default QC variant ablation):
#   bash scripts/run_ablation_parallel.sh \
#       --output-dir out_experiments/run1 \
#       --gpus "0 1 2 3 4 5 6 7"
#
#   # Option B (prepared local splits, gold-excluded):
#   bash scripts/run_ablation_parallel.sh \
#       --output-dir out_experiments/run1 \
#       --gpus "0 1 2 3 4 5 6 7" \
#       --data-dir dataset/prepared_m2
#
#   # Option B (splits from Hugging Face):
#   bash scripts/run_ablation_parallel.sh \
#       --output-dir out_experiments/run1 \
#       --gpus "0 1 2 3 4 5 6 7" \
#       --hf-dataset LeninGF/question-answering-robbery-m2
#
#   # Print what would be launched without running anything:
#   bash scripts/run_ablation_parallel.sh --dry-run --gpus "0 1"
set -euo pipefail

OUT_DIR="${OUT_DIR:-out_experiments/run_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
DRY_RUN=0
EPOCHS=""
EARLY_STOPPING=""
DATA_DIR=""
HF_DATASET=""
LIMIT_CONTEXTS=""

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
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --early-stopping-patience)
      EARLY_STOPPING="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --hf-dataset)
      HF_DATASET="$2"
      shift 2
      ;;
    --limit-contexts)
      LIMIT_CONTEXTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$DATA_DIR" && -n "$HF_DATASET" ]]; then
  echo "--data-dir and --hf-dataset are mutually exclusive" >&2
  exit 1
fi

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

if [[ -n "$DATA_DIR" || -n "$HF_DATASET" ]]; then
  # Prepared/HF splits are the final merged dataset; running the same source
  # under different dataset names would duplicate identical runs.
  DATASETS=("merged")
  echo "Data source: prepared/HF -> using dataset 'merged' only"
else
  DATASETS=("merged" "strict_gemma" "strict_qwen")
  echo "Data source: QC JSONL files on disk -> using merged/strict_gemma/strict_qwen"
fi

MODES=("zsl" "ft")

mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"
echo "GPUs: ${GPU_ARRAY[*]} ($N_GPUS)"
echo "Experiments: $(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#MODES[@]} ))"
if [[ -n "$EPOCHS" ]]; then echo "Epochs: $EPOCHS"; fi
if [[ -n "$EARLY_STOPPING" ]]; then echo "Early stopping patience: $EARLY_STOPPING"; fi
if [[ -n "$DATA_DIR" ]]; then echo "Data dir: $DATA_DIR"; fi
if [[ -n "$HF_DATASET" ]]; then echo "HF dataset: $HF_DATASET"; fi
if [[ -n "$LIMIT_CONTEXTS" ]]; then echo "Limit contexts: $LIMIT_CONTEXTS (smoke)"; fi

launch_experiment() {
  local model="$1"
  local dataset="$2"
  local mode="$3"
  local gpu="$4"
  local tag
  tag=$(echo "$model" | tr '/' '_')
  local log="$OUT_DIR/${tag}__${dataset}__${mode}.log"

  local cmd="python scripts/run_qa_ablation.py --model ${model} --dataset ${dataset} --mode ${mode} --gpu ${gpu} --output-dir ${OUT_DIR}"
  if [[ -n "$EPOCHS" ]]; then
    cmd+=" --epochs ${EPOCHS}"
  fi
  if [[ -n "$EARLY_STOPPING" ]]; then
    cmd+=" --early-stopping-patience ${EARLY_STOPPING}"
  fi
  if [[ -n "$DATA_DIR" ]]; then
    cmd+=" --data-dir ${DATA_DIR}"
  fi
  if [[ -n "$HF_DATASET" ]]; then
    cmd+=" --hf-dataset ${HF_DATASET}"
  fi
  if [[ -n "$LIMIT_CONTEXTS" ]]; then
    cmd+=" --limit-contexts ${LIMIT_CONTEXTS}"
  fi

  echo "GPU ${gpu}: ${model} | ${dataset} | ${mode}"
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
