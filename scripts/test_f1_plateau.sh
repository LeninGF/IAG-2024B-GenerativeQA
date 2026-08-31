#!/usr/bin/env bash
# Run a single fine-tuning with many epochs on the final merged dataset to find
# where eval F1 plateaus, then analyze the training history.
#
# Defaults:
#   - Model: mrm8488 BETO SQuAD2-es (the paper baseline)
#   - Data: Hugging Face dataset LeninGF/question-answering-robbery-m2
#   - Epochs: 20
#   - GPU: 0
#
# Usage:
#   bash scripts/test_f1_plateau.sh [--epochs 20] [--gpu 0] [--output-dir out_experiments/f1_plateau_test]
#
#   # Use local prepared splits instead of HF:
#   bash scripts/test_f1_plateau.sh --data-dir dataset/prepared_m2
#
#   # Tune the plateau definition:
#   bash scripts/test_f1_plateau.sh --tolerance 0.2 --min-epochs 5
set -euo pipefail

MODEL="${MODEL:-mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es}"
EPOCHS="${EPOCHS:-20}"
GPU="${GPU:-0}"
OUT_DIR="${OUT_DIR:-out_experiments/f1_plateau_test}"
DATA_DIR=""
HF_DATASET="${HF_DATASET:-LeninGF/question-answering-robbery-m2}"
TOLERANCE=""
MIN_EPOCHS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --hf-dataset) HF_DATASET="$2"; shift 2 ;;
    --tolerance) TOLERANCE="$2"; shift 2 ;;
    --min-epochs) MIN_EPOCHS="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$DATA_DIR" && -n "$HF_DATASET" ]]; then
  echo "--data-dir and --hf-dataset are mutually exclusive" >&2
  exit 1
fi

TAG=$(echo "$MODEL" | tr '/' '_')

echo "=== F1 plateau test ==="
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "GPU: $GPU"
echo "Output dir: $OUT_DIR"
if [[ -n "$DATA_DIR" ]]; then echo "Data: $DATA_DIR"; else echo "Data: HF $HF_DATASET"; fi

CMD="python scripts/run_qa_ablation.py --model ${MODEL} --dataset merged --mode ft --gpu ${GPU} --output-dir ${OUT_DIR} --epochs ${EPOCHS}"
if [[ -n "$DATA_DIR" ]]; then
  CMD+=" --data-dir ${DATA_DIR}"
else
  CMD+=" --hf-dataset ${HF_DATASET}"
fi

echo "Running: ${CMD}"
CUDA_VISIBLE_DEVICES="${GPU}" ${CMD}

HISTORY="${OUT_DIR}/${TAG}/merged/ft/training_history.csv"
if [[ ! -f "$HISTORY" ]]; then
  echo "ERROR: training history not found at ${HISTORY}" >&2
  exit 1
fi

ANALYZE="python scripts/find_f1_plateau.py --history ${HISTORY}"
if [[ -n "$TOLERANCE" ]]; then ANALYZE+=" --tolerance ${TOLERANCE}"; fi
if [[ -n "$MIN_EPOCHS" ]]; then ANALYZE+=" --min-epochs ${MIN_EPOCHS}"; fi

echo ""
echo "=== Plateau analysis ==="
${ANALYZE}
