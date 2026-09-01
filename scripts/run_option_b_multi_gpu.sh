#!/usr/bin/env bash
# Run the Option B matrix (final merged dataset, gold-safe) using multi-GPU for
# fine-tuning.
#
# Matrix: 4 models x merged x {zsl, ft} = 8 experiments.
#   - ZSL (zero-shot) is fast -> runs on a single GPU (default GPU 0).
#   - FT (fine-tuning) is expensive -> runs on all provided GPUs via torchrun.
#
# Usage:
#   bash scripts/run_option_b_multi_gpu.sh \
#       --gpus "0 1 2 3 4 5 6 7" \
#       --epochs 15 \
#       --hf-dataset LeninGF/question-answering-robbery-m2 \
#       --output-dir out_experiments/option_b
#
#   # Use local prepared splits instead of HF:
#   bash scripts/run_option_b_multi_gpu.sh \
#       --gpus "0 1 2 3 4 5 6 7" \
#       --epochs 15 \
#       --data-dir dataset/prepared_m2 \
#       --output-dir out_experiments/option_b
#
#   # Run only two models (e.g. split work across two terminals):
#   bash scripts/run_option_b_multi_gpu.sh --models "mrm8488/...,MMG/..." \
#       --gpus "0 1 2 3" --output-dir out_experiments/option_b
#
#   # Print what would run without executing:
#   bash scripts/run_option_b_multi_gpu.sh --dry-run --gpus "0 1 2 3 4 5 6 7"
#
#   # Resume: skip any zsl/ft experiment whose metrics_summary.json already
#   # exists under OUT_DIR/<model__name>/merged/<mode>/:
#   bash scripts/run_option_b_multi_gpu.sh --resume --gpus "0 1 2 3 4 5 6 7" \
#       --output-dir out_experiments/option_b_abblation
set -euo pipefail

# Ensure ZSL python and torchrun FT children find the conda env's newer
# libstdc++ (CXXABI_1.3.15) instead of the older system library.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

DEFAULT_MODELS=(
  "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
  "MMG/bert-base-spanish-wwm-cased-finetuned-squad2-es"
  "deepset/xlm-roberta-base-squad2"
  "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
)

OUT_DIR="${OUT_DIR:-out_experiments/option_b_multi}"
EPOCHS="${EPOCHS:-15}"
EARLY_STOPPING=""
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
HF_DATASET="${HF_DATASET:-LeninGF/question-answering-robbery-m2}"
DATA_DIR=""
ZSL_GPU="${ZSL_GPU:-0}"
LIMIT_CONTEXTS=""
DRY_RUN=0
RESUME=0
MODELS_STR=""
PUSH_MODEL=0
MODEL_REPO_ID=""
LR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --early-stopping-patience) EARLY_STOPPING="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --hf-dataset) HF_DATASET="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --zsl-gpu) ZSL_GPU="$2"; shift 2 ;;
    --limit-contexts) LIMIT_CONTEXTS="$2"; shift 2 ;;
    --models) MODELS_STR="$2"; shift 2 ;;
    --push-model) PUSH_MODEL=1; shift ;;
    --model-repo-id) MODEL_REPO_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume) RESUME=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$DATA_DIR" && -n "$HF_DATASET" ]]; then
  echo "--data-dir and --hf-dataset are mutually exclusive" >&2
  exit 1
fi

if [[ -n "$MODELS_STR" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_STR"
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

GPU_LIST=$(echo "$GPUS" | tr ',' ' ')
read -r -a GPU_ARR <<< "$GPU_LIST"
N_GPUS=${#GPU_ARR[@]}

# Completion marker written by run_qa_ablation.py for each (model, mode).
exp_dir_for() {
  local model="$1"
  local mode="$2"
  local tag="${model//\//__}"
  echo "${OUT_DIR}/${tag}/merged/${mode}"
}

is_completed() {
  [[ -f "$(exp_dir_for "$1" "$2")/metrics_summary.json" ]]
}

DONE_COUNT=0
PENDING_COUNT=0
for _model in "${MODELS[@]}"; do
  for _mode in zsl ft; do
    if is_completed "$_model" "$_mode"; then
      DONE_COUNT=$((DONE_COUNT + 1))
    else
      PENDING_COUNT=$((PENDING_COUNT + 1))
    fi
  done
done

echo "=== Option B (final merged dataset, gold-safe) ==="
echo "Models: ${#MODELS[@]}"
echo "Epochs: $EPOCHS"
echo "FT GPUs: $GPU_LIST ($N_GPUS)"
echo "ZSL GPU: $ZSL_GPU"
echo "Output dir: $OUT_DIR"
echo "Experiments: $(( DONE_COUNT + PENDING_COUNT )) total, $DONE_COUNT completed, $PENDING_COUNT pending"
if [[ $RESUME -eq 1 ]]; then echo "Resume: enabled (skip completed)"; fi
if [[ -n "$DATA_DIR" ]]; then echo "Data: $DATA_DIR"; else echo "Data: HF $HF_DATASET"; fi
if [[ -n "$EARLY_STOPPING" ]]; then echo "Early stopping patience: $EARLY_STOPPING"; fi
if [[ -n "$LIMIT_CONTEXTS" ]]; then echo "Limit contexts: $LIMIT_CONTEXTS (smoke)"; fi
if [[ $PUSH_MODEL -eq 1 ]]; then echo "Push fine-tuned models to HF: yes"; fi
if [[ $DRY_RUN -eq 1 ]]; then echo "(dry run)"; fi
echo ""

run_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  (would run) $*"
  else
    "$@"
  fi
}

for model in "${MODELS[@]}"; do
  echo "=== Model: $model ==="

  # ZSL (fast, single GPU)
  if [[ $RESUME -eq 1 ]] && is_completed "$model" "zsl"; then
    echo "  [ZSL] skip (already completed: $(exp_dir_for "$model" "zsl"))"
  else
    zsl_cmd=(python scripts/run_qa_ablation.py --model "$model" --dataset merged --mode zsl
             --gpu "$ZSL_GPU" --output-dir "$OUT_DIR")
    if [[ -n "$DATA_DIR" ]]; then
      zsl_cmd+=(--data-dir "$DATA_DIR")
    else
      zsl_cmd+=(--hf-dataset "$HF_DATASET")
    fi
    if [[ -n "$LIMIT_CONTEXTS" ]]; then
      zsl_cmd+=(--limit-contexts "$LIMIT_CONTEXTS")
    fi
    if [[ -n "$LR" ]]; then
      zsl_cmd+=(--lr "$LR")
    fi
    echo "  [ZSL] ${zsl_cmd[*]}"
    if [[ $DRY_RUN -eq 0 ]]; then
      CUDA_VISIBLE_DEVICES="$ZSL_GPU" "${zsl_cmd[@]}"
    fi
  fi

  # FT (multi-GPU via torchrun)
  if [[ $RESUME -eq 1 ]] && is_completed "$model" "ft"; then
    echo "  [FT] skip (already completed: $(exp_dir_for "$model" "ft"))"
  else
    ft_cmd=(bash scripts/run_ablation_multi_gpu.sh --gpus "$GPU_LIST"
            --model "$model" --dataset merged --mode ft
            --output-dir "$OUT_DIR" --epochs "$EPOCHS")
    if [[ -n "$DATA_DIR" ]]; then
      ft_cmd+=(--data-dir "$DATA_DIR")
    else
      ft_cmd+=(--hf-dataset "$HF_DATASET")
    fi
    if [[ -n "$EARLY_STOPPING" ]]; then
      ft_cmd+=(--early-stopping-patience "$EARLY_STOPPING")
    fi
    if [[ -n "$LIMIT_CONTEXTS" ]]; then
      ft_cmd+=(--limit-contexts "$LIMIT_CONTEXTS")
    fi
    if [[ -n "$LR" ]]; then
      ft_cmd+=(--lr "$LR")
    fi
    if [[ $PUSH_MODEL -eq 1 ]]; then
      ft_cmd+=(--push-model)
      if [[ -n "$MODEL_REPO_ID" ]]; then
        ft_cmd+=(--model-repo-id "$MODEL_REPO_ID")
      fi
    fi
    echo "  [FT] ${ft_cmd[*]}"
    run_cmd "${ft_cmd[@]}"
  fi
  echo ""
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run complete (nothing executed)."
else
  echo "All Option B experiments finished. Results in ${OUT_DIR}"
fi
