"""Demo: load a fine-tuned QA model and predict on a random example.

This is the "load model -> sample a random context/question -> predict" flow,
similar to the final part of question-answering-Bert.ipynb.

Usage examples:
    # Predict on a random row from an HF dataset using a locally saved model:
    python scripts/qa_demo.py \
        --model-dir out_experiments/paper_replication/.../ft/model \
        --hf-dataset LeninGF/question-answering-robbery-m2

    # Predict on the legacy paper dataset:
    python scripts/qa_demo.py \
        --model-dir out_experiments/paper_replication/.../ft/model \
        --hf-dataset LeninGF/robos-question-answering --hf-schema legacy

    # Predict on a local SQuAD v2 JSONL:
    python scripts/qa_demo.py \
        --model-repo-id LeninGF/qa-m2-beto-ft \
        --jsonl out_qc_M2/squadv2_final_merged.jsonl
"""
from __future__ import annotations

import argparse
import os
import random
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", help="Local saved model dir (e.g. <exp_dir>/model)")
    src.add_argument("--model-repo-id", help="Hugging Face model repo id")
    p.add_argument("--hf-dataset", default=None,
                   help="HF dataset repo to sample a random row from")
    p.add_argument("--jsonl", default=None,
                   help="Local SQuAD v2 JSONL to sample a random row from")
    p.add_argument("--hf-schema", choices=["auto", "legacy", "squad2"], default="auto",
                   help="HF dataset schema (default auto)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.hf_dataset and not args.jsonl:
        raise SystemExit("Provide --hf-dataset or --jsonl")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qa_dataset_utils as qa_utils
    import run_qa_ablation as rqa
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    model_id = args.model_dir or args.model_repo_id
    print(f"Loading model from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    model.eval()

    if args.jsonl:
        rows = qa_utils.load_squad2_variant(args.jsonl, "demo")
    else:
        from datasets import load_dataset

        ds = load_dataset(args.hf_dataset)
        split = next(s for s in ("test", "validation", "train") if s in ds)
        print(f"Loading dataset {args.hf_dataset} (split={split}) ...")
        rows = [qa_utils.normalize_hf_squad_row(dict(r), args.hf_schema) for r in ds[split]]

    if not rows:
        raise SystemExit("No rows found in dataset")

    rng = random.Random(args.seed)
    row = dict(rng.choice(rows))
    if row.get("answers") and row["answers"].get("text"):
        gold = row["answers"]["text"][0]
    else:
        gold = "(no answer / impossible)"

    args_ns = argparse.Namespace(
        max_length=args.max_length,
        stride=args.stride,
        per_device_eval_batch_size=args.batch_size,
    )
    pred = rqa.predict_examples(model, tokenizer, [row], args_ns, qa_utils)[0]

    print("\n" + "=" * 70)
    print(f"Context: {row['context']}")
    print("-" * 70)
    print(f"Question: {row['question']}")
    print(f"Gold answer: {gold}")
    print("-" * 70)
    print(f"Predicted: {pred['prediction_text']!r}")
    print(f"no_answer_probability: {pred['no_answer_probability']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
