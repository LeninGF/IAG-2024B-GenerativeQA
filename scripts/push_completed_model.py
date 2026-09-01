#!/usr/bin/env python
"""Push an already-trained saved model to the Hugging Face Hub.

Use this to retry a failed `--push-model` upload without retraining. The training
pipeline saves the best model (including tokenizer files) to
`<exp_dir>/model`, so you can re-upload that directory with a valid short repo
id.

Usage:
  # Auto-generate a short, valid repo id from the model directory path:
  python scripts/push_completed_model.py \
      --model-dir out_experiments/option_b_abblation/\
mrm8488__distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es/merged/ft/model

  # Or specify the repo id explicitly:
  python scripts/push_completed_model.py \
      --model-dir <path/to/model> \
      --repo-id LeninGF/qa-merged-distill-bert-base-...-ft-20260901_042912

Requires HUGGINGFACE_TOKEN in the repo-root .env file (same as run_qa_ablation.py).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def infer_model_and_dataset(model_dir: Path) -> tuple[str, str]:
    """Reconstruct model id and dataset from an experiment model dir path.

    Expected layout: <root>/<model__name>/merged/<mode>/model
    """
    model_dir = model_dir.resolve()
    mode_dir = model_dir.parent          # .../merged/<mode>
    merged_dir = mode_dir.parent         # .../merged
    model_dir_name = merged_dir.parent.name  # <model__name>
    dataset = merged_dir.name
    model_id = model_dir_name.replace("__", "/")
    return model_id, dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True,
                   help="Path to a saved model directory (contains config.json, "
                        "pytorch_model.bin or model.safetensors, tokenizer files).")
    p.add_argument("--repo-id", default=None,
                   help="HF repo id. If omitted, auto-generated using the same "
                        "short-name logic as run_qa_ablation.py.")
    p.add_argument("--private", action="store_true", default=False,
                   help="Create the repo as private (default: public).")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").is_file():
        sys.exit(f"Not a model directory (missing config.json): {model_dir}")

    # Import helpers from run_qa_ablation.py without executing its main().
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_qa_ablation import auto_model_repo_id  # noqa: E402

    if args.repo_id:
        repo_id = args.repo_id
    else:
        model_id, dataset = infer_model_and_dataset(model_dir)
        repo_id = auto_model_repo_id(model_id, dataset)
        print(f"Inferred model={model_id} dataset={dataset}")

    print(f"Model dir: {model_dir}")
    print(f"Repo id  : {repo_id}")

    from dotenv import load_dotenv
    from huggingface_hub import login
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        sys.exit("HUGGINGFACE_TOKEN not found. Add it to a .env file at the repo root.")
    login(hf_token)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print("Loading model...")
    model = AutoModelForQuestionAnswering.from_pretrained(str(model_dir))

    print(f"Pushing model to {repo_id} ...")
    model.push_to_hub(repo_id, private=args.private)
    tokenizer.push_to_hub(repo_id)
    print(f"Model pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
