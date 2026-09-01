#!/usr/bin/env bash
# Smoke test: verify that the QA ablation environment has all the libraries
# needed to load the option-B models' tokenizers. This specifically guards
# against the protobuf/tiktoken ImportError that broke
# deepset/xlm-roberta-base-squad2 in transformers' slow-tokenizer conversion.
#
# Usage:
#   bash scripts/smoke_check_env.sh
#
# It only loads tokenizers (no model weights, no GPU required).
set -euo pipefail

# Same fix used by the run scripts: prefer the conda env's libstdc++.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

python - <<'PY'
import importlib
import sys

# 1) Libraries required by transformers' slow-tokenizer conversion.
#    protobuf's import name is google.protobuf (not protobuf).
REQUIRED = [
    ("google.protobuf", "protobuf"),
    ("sentencepiece", "sentencepiece"),
    ("tiktoken", "tiktoken"),
    ("transformers", "transformers"),
    ("torch", "torch"),
]
missing = []
for import_name, display_name in REQUIRED:
    try:
        importlib.import_module(import_name)
        print(f"[ok] import {display_name}")
    except Exception as exc:  # noqa: BLE001 - report and continue
        missing.append(display_name)
        print(f"[FAIL] import {display_name}: {exc}")

if missing:
    sys.exit("Missing libraries: " + ", ".join(missing))

# 2) Load every tokenizer used by the Option B matrix. This is exactly the
#    call that previously raised:
#    SentencePieceExtractor requires protobuf / ValueError: tiktoken is required
from transformers import AutoTokenizer

MODELS = [
    "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    "MMG/bert-base-spanish-wwm-cased-finetuned-squad2-es",
    "deepset/xlm-roberta-base-squad2",
    "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
]

for model_id in MODELS:
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"[ok] tokenizer {model_id}: {type(tok).__name__} vocab={tok.vocab_size}")

print("Smoke check passed.")
PY
