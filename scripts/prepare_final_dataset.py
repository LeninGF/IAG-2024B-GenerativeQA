"""Prepare the final QC M2 SQuAD v2 dataset for training and optionally push it
to the Hugging Face Hub.

The QC pipeline already produced final, filtered datasets in ``out_qc_M2/``.
This script does NOT apply any additional quality filtering: it only validates
the SQuAD v2 schema, performs a deterministic context-level split
(train/dev/test), writes the split JSONL files locally, and optionally pushes
them to a Hugging Face dataset repo.

Usage examples:
    # Prepare locally (default input = merged dataset):
    python scripts/prepare_final_dataset.py --output-dir dataset/prepared_m2

    # Smoke test with only 20 contexts:
    python scripts/prepare_final_dataset.py --limit-contexts 20 \
        --output-dir /tmp/prepared_m2

    # Prepare and push to Hugging Face:
    python scripts/prepare_final_dataset.py \
        --input out_qc_M2/squadv2_final_merged.jsonl \
        --output-dir dataset/prepared_m2 \
        --repo-id LeninGF/robos-question-answering-m2 \
        --push

Dependencies: only the Python stdlib for local preparation. ``datasets`` and
``huggingface_hub`` are imported lazily only when ``--push`` is used.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from qa_dataset_utils import (
    filter_by_context,
    load_squad2_variant,
    save_jsonl,
    split_by_context,
)

DEFAULT_INPUT = "out_qc_M2/squadv2_final_merged.jsonl"
DEFAULT_OUTPUT_DIR = "dataset/prepared_m2"
DEFAULT_REPO_ID = "LeninGF/robos-question-answering-m2"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"QC SQuAD v2 JSONL to prepare (default: {DEFAULT_INPUT})")
    parser.add_argument("--output-dir", default=None,
                        help=f"Directory where train/dev/test JSONL are written (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--dev-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--push", action="store_true",
                        help="Push train/dev/test to the Hugging Face Hub")
    parser.add_argument("--limit-contexts", type=int, default=None,
                        help="Only use the first N contexts (smoke tests)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    records = load_squad2_variant(args.input, "input")
    print(f"Loaded {len(records)} records "
          f"({sum(1 for r in records if not r['is_impossible'])} answerable, "
          f"{sum(1 for r in records if r['is_impossible'])} unanswerable)")

    train_ids, dev_ids, test_ids = split_by_context(
        records,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        limit_contexts=args.limit_contexts,
    )
    print(f"Context split -> train: {len(train_ids)}, dev: {len(dev_ids)}, test: {len(test_ids)}")

    train = filter_by_context(records, set(train_ids))
    dev = filter_by_context(records, set(dev_ids))
    test = filter_by_context(records, set(test_ids))
    # Drop internal bookkeeping column before writing/pushing to the Hub.
    for rows in (train, dev, test):
        for r in rows:
            r.pop("_source", None)

    # SQuAD v2 readiness summary (informative; validation already happened in
    # load_squad2_variant, these counts are for the final written splits).
    all_split = train + dev + test
    imp_with_ans = sum(1 for r in all_split if r["is_impossible"] and (r["answers"]["text"] or r["answers"]["answer_start"]))
    ans_without_ans = sum(1 for r in all_split if not r["is_impossible"] and not r["answers"]["text"])
    bad_offsets = sum(
        1 for r in all_split
        if not r["is_impossible"] and not r["context"].startswith(r["answers"]["text"][0], r["answers"]["answer_start"][0])
    )
    print(f"SQuAD v2 readiness: {len(all_split)} rows | impossible_with_answers={imp_with_ans} | "
          f"answerable_without_answers={ans_without_ans} | invalid_offsets={bad_offsets}")

    paths = {
        "train": os.path.join(output_dir, "train.jsonl"),
        "dev": os.path.join(output_dir, "dev.jsonl"),
        "test": os.path.join(output_dir, "test.jsonl"),
    }
    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        save_jsonl(rows, paths[name])
        print(f"  {name}: {len(rows)} rows -> {paths[name]}")

    info = {
        "source": os.path.abspath(args.input),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_frac": args.train_frac,
        "dev_frac": args.dev_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "n_records": len(records),
        "n_train": len(train),
        "n_dev": len(dev),
        "n_test": len(test),
        "n_train_contexts": len(train_ids),
        "n_dev_contexts": len(dev_ids),
        "n_test_contexts": len(test_ids),
    }
    info_path = os.path.join(output_dir, "prepared_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"Saved {info_path}")

    if args.push:
        push_to_hub(paths, args.repo_id, args.train_frac, args.dev_frac, args.test_frac, args.seed)


def push_to_hub(paths, repo_id, train_frac, dev_frac, test_frac, seed):
    """Push the three JSONL files as a HF DatasetDict with train/dev/test splits."""
    from datasets import DatasetDict, load_dataset
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("HUGGINGFACE_TOKEN not found. Add it to a .env file at the repo root.")
    login(hf_token)

    print("Loading splits for HF push ...")
    train_ds = load_dataset("json", data_files=paths["train"], split="train")
    dev_ds = load_dataset("json", data_files=paths["dev"], split="train")
    test_ds = load_dataset("json", data_files=paths["test"], split="train")
    dataset_dict = DatasetDict({"train": train_ds, "validation": dev_ds, "test": test_ds})

    print(f"Pushing {len(dataset_dict['train'])}/{len(dataset_dict['validation'])}/"
          f"{len(dataset_dict['test'])} rows to {repo_id} ...")
    dataset_dict.push_to_hub(repo_id, private=False)
    print(f"Done. Load with: load_dataset('{repo_id}')")


if __name__ == "__main__":
    main()
