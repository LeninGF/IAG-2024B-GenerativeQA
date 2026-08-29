#!/usr/bin/env bash
# Convenience wrapper around run_build_parallel.py.
#
# Usage examples:
#   bash scripts/run_build_parallel.sh --model gemma-3-1b-it \
#       --gpus 4,5,6,7 --use-dataset-sample --sample-size 17568 \
#       --output-file dataset/squadv2_gemma_sample.jsonl
#
#   bash scripts/run_build_parallel.sh --model qwen2.5-3b-instruct \
#       --gpus 0,1,2,3 --output-file dataset/squadv2_qwen_sample.jsonl --dry-run
#
#   bash scripts/run_build_parallel.sh --model qwen2.5-3b-instruct \
#       --gpus 0,1,2,3 --max-new-tokens 512 --max-retries 2 --retry-delay 3 \
#       --output-file dataset/squadv2_qwen_sample.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/run_build_parallel.py" "$@"
