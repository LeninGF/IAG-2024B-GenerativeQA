"""Run zero-shot and/or fine-tuned extractive QA ablations on the M2 datasets.

Design: one process = one (model, dataset, mode) experiment, so you can launch
independent processes on different GPUs in parallel. Use ``--plan-only`` to
print the exact commands for the full matrix, or ``--all`` to run everything
sequentially in a single process.

Examples:
    # Single experiment on GPU 0:
    python scripts/run_qa_ablation.py \\
        --model mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es \\
        --dataset merged --mode ft --gpu 0 --output-dir out_experiments/run1

    # Print the full 4x3x2 command matrix with round-robin GPU assignment:
    python scripts/run_qa_ablation.py --plan-only --plan-gpus 8 \\
        --output-dir out_experiments/run1

    # Smoke test (small contexts, 1 epoch, one dataset/model):
    python scripts/run_qa_ablation.py \\
        --model mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es \\
        --dataset merged --mode both --gpu 0 --limit-contexts 100 --epochs 1 \\
        --output-dir /tmp/qa_smoke

    # Run the whole matrix sequentially on one GPU:
    python scripts/run_qa_ablation.py --all --gpu 0 --output-dir out_experiments/run1

    # Use a prepared local directory (from prepare_final_dataset.py, gold-excluded):
    python scripts/run_qa_ablation.py \\
        --model mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es \\
        --dataset merged --mode both --gpu 0 \\
        --data-dir dataset/prepared_m2 --output-dir out_experiments/run1

    # Load prepared splits directly from the Hugging Face Hub:
    python scripts/run_qa_ablation.py \\
        --model mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es \\
        --dataset merged --mode both --gpu 0 \\
        --hf-dataset LeninGF/robos-question-answering-m2 --output-dir out_experiments/run1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Default experiment matrix (can be overridden via --models/--datasets)
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    "MMG/bert-base-spanish-wwm-cased-finetuned-squad2-es",
    "deepset/xlm-roberta-base-squad2",
    "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
]

DEFAULT_DATASETS = ["merged", "strict_gemma", "strict_qwen"]

DATASET_PATHS = {
    "merged": "out_qc_M2/squadv2_final_merged.jsonl",
    "strict_gemma": "out_qc_M2/squadv2_strict_gemma.jsonl",
    "strict_qwen": "out_qc_M2/squadv2_strict_qwen.jsonl",
}

DEFAULT_GOLD_AUDIT = "out_qc_M2/audit_stratified_sample_labeled_v1.csv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="HF model id (required unless --all/--plan-only)")
    p.add_argument("--dataset", default=None, choices=list(DATASET_PATHS),
                   help="M2 dataset variant (required unless --all/--plan-only)")
    p.add_argument("--mode", default="both", choices=["zsl", "ft", "both"],
                   help="zsl = pre-trained only, ft = fine-tune, both = both (default both)")
    p.add_argument("--gpu", type=int, default=0, help="GPU id used for this process")
    p.add_argument("--output-dir", default=None,
                   help="Root output directory (default out_experiments/run_<timestamp>)")

    # Matrix control
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="Comma-separated HF model ids (default: 4-model set)")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                   help="Comma-separated M2 variants (default: merged,strict_gemma,strict_qwen)")
    p.add_argument("--all", action="store_true",
                   help="Run the full matrix sequentially in this process")
    p.add_argument("--plan-only", action="store_true",
                   help="Print the full command matrix with round-robin GPUs and exit")
    p.add_argument("--plan-gpus", type=int, default=8, help="GPUs used by --plan-only")

    # Data split / evaluation
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--dev-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gold-audit", default=DEFAULT_GOLD_AUDIT)
    p.add_argument("--limit-contexts", type=int, default=None,
                   help="Only use the first N contexts (smoke tests)")
    p.add_argument("--data-dir", default=None,
                   help="Use prepared local splits train.jsonl/dev.jsonl/test.jsonl from this directory")
    p.add_argument("--hf-dataset", default=None,
                   help="Load prepared splits from a Hugging Face dataset repo (train/validation/test)")

    # Training hyperparameters (paper defaults from Table 3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--per-device-train-batch-size", type=int, default=32)
    p.add_argument("--per-device-eval-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--save-models", action="store_true",
                   help="Keep fine-tuned checkpoints on disk (default: only best model artifacts)")
    p.add_argument("--skip-zero-shot", action="store_true")
    p.add_argument("--skip-fine-tune", action="store_true")
    args = p.parse_args()
    if args.data_dir and args.hf_dataset:
        p.error("--data-dir and --hf-dataset are mutually exclusive")
    return args


def sanitize_model(model: str) -> str:
    return model.replace("/", "__")


def model_short(model: str) -> str:
    """Short stable label, e.g. mrm8488__bert-base-spanish-wwm-..."""
    return sanitize_model(model).split("__")[-1][:50] if "__" in model else model


def default_output_dir() -> str:
    return f"out_experiments/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ---------------------------------------------------------------------------
# Plan printing (no torch import needed)
# ---------------------------------------------------------------------------
def print_plan(args):
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    modes = ["zsl", "ft"]
    combos = [(m, d, mode) for m in models for d in datasets for mode in modes]
    print(f"# {len(combos)} experiments, round-robin over {args.plan_gpus} GPU(s)")
    for i, (model, dataset, mode) in enumerate(combos):
        gpu = i % args.plan_gpus
        cmd = (
            f"python scripts/run_qa_ablation.py --model {model} --dataset {dataset} "
            f"--mode {mode} --gpu {gpu} --output-dir {args.output_dir} "
            f"--test-size {args.test_size} --dev-size {args.dev_size} --seed {args.seed} "
            f"--epochs {args.epochs} --lr {args.lr} --weight-decay {args.weight_decay} "
            f"--max-length {args.max_length} --stride {args.stride} "
            f"--per-device-train-batch-size {args.per_device_train_batch_size} "
            f"--per-device-eval-batch-size {args.per_device_eval_batch_size} "
            f"--grad-accum {args.grad_accum} --gold-audit {args.gold_audit} "
        )
        if args.data_dir:
            cmd += f"--data-dir {args.data_dir} "
        if args.hf_dataset:
            cmd += f"--hf-dataset {args.hf_dataset} "
        cmd += "--fp16" if args.fp16 else "--no-fp16"
        if args.save_models:
            cmd += " --save-models"
        if args.limit_contexts is not None:
            cmd += f" --limit-contexts {args.limit_contexts}"
        print(cmd)


# ---------------------------------------------------------------------------
# Data loading / tokenization helpers (imported after CUDA_VISIBLE_DEVICES set)
# ---------------------------------------------------------------------------
def load_experiment_splits(args, qa_utils, dataset_name):
    """Load the selected M2 variant and build a common context-level split.

    The context partition is always computed from the merged dataset so every
    experiment (even when launched in parallel) uses identical train/dev/test
    contexts. Test/dev references always come from the merged dataset; the
    training rows come from the selected variant.
    """
    merged = qa_utils.load_squad2_variant(DATASET_PATHS["merged"], "merged")
    variants = {}
    if dataset_name != "merged":
        variants[dataset_name] = qa_utils.load_squad2_variant(DATASET_PATHS[dataset_name], dataset_name)

    train_frac = 1.0 - args.dev_size - args.test_size
    train_ids, dev_ids, test_ids = qa_utils.split_by_context(
        merged, train_frac, args.dev_size, args.test_size,
        seed=args.seed, limit_contexts=args.limit_contexts,
    )
    dev = qa_utils.filter_by_context(merged, set(dev_ids))
    test = qa_utils.filter_by_context(merged, set(test_ids))
    if dataset_name == "merged":
        train = qa_utils.filter_by_context(merged, set(train_ids))
    else:
        train = qa_utils.filter_by_context(variants[dataset_name], set(train_ids))
    return train, dev, test


def _normalize_hf_row(row, split_name, qa_utils):
    rec = dict(row)
    rec["is_impossible"] = bool(rec.get("is_impossible", False))
    errors = qa_utils.validate_squad2_record(rec)
    if errors:
        raise ValueError(f"Invalid SQuAD v2 record in HF split '{split_name}': {errors} :: {rec}")
    rec["context_id"] = qa_utils.derive_context_id(rec)
    rec["_source"] = split_name
    return rec


def load_prepared_splits(args, qa_utils):
    """Load train/dev/test from a prepared directory or a HF dataset repo.

    Prepared files (from scripts/prepare_final_dataset.py) already exclude the
    gold-audit pairs and use the same SQuAD v2 schema. When ``--limit-contexts``
    is set, only the first N contexts (sorted) are kept for smoke tests.
    """
    if args.data_dir:
        train = qa_utils.load_squad2_variant(os.path.join(args.data_dir, "train.jsonl"), "train")
        dev = qa_utils.load_squad2_variant(os.path.join(args.data_dir, "dev.jsonl"), "dev")
        test = qa_utils.load_squad2_variant(os.path.join(args.data_dir, "test.jsonl"), "test")
    else:
        from datasets import load_dataset

        ds = load_dataset(args.hf_dataset)
        train = [_normalize_hf_row(dict(r), "train", qa_utils) for r in ds["train"]]
        dev = [_normalize_hf_row(dict(r), "validation", qa_utils) for r in ds["validation"]]
        test = [_normalize_hf_row(dict(r), "test", qa_utils) for r in ds["test"]]

    if args.limit_contexts is not None:
        all_ids = sorted({r["context_id"] for r in train})
        keep = set(all_ids[: max(0, args.limit_contexts)])
        train = [r for r in train if r["context_id"] in keep]
        dev = [r for r in dev if r["context_id"] in keep]
        test = [r for r in test if r["context_id"] in keep]
    return train, dev, test


def make_train_features(examples, tokenizer, max_length, stride):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    start_positions, end_positions = [], []
    for i, sample_idx in enumerate(sample_map):
        if examples["is_impossible"][sample_idx]:
            start_positions.append(0)
            end_positions.append(0)
            continue
        answer = examples["answers"][sample_idx]
        start_char = int(answer["answer_start"][0])
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        offsets = inputs["offset_mapping"][i]
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - list(reversed(sequence_ids)).index(1)
        if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
            continue
        start_idx = context_start
        while start_idx <= context_end and offsets[start_idx][0] <= start_char:
            start_idx += 1
        start_positions.append(start_idx - 1)
        end_idx = context_end
        while end_idx >= context_start and offsets[end_idx][1] >= end_char:
            end_idx -= 1
        end_positions.append(end_idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    # Offsets are not needed for training; removing them keeps the collator
    # simple (only tensor-friendly columns remain).
    inputs.pop("offset_mapping", None)
    return inputs


def make_eval_features(examples, tokenizer, max_length, stride):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    inputs["example_id"] = [examples["id"][i] for i in sample_map]
    return inputs


def predict_examples(model, tokenizer, examples, args, qa_utils):
    """Run extractive QA inference on raw examples and return SQuAD predictions."""
    import numpy as np
    import torch
    from datasets import Dataset

    feat_ds = Dataset.from_list(examples).map(
        lambda b: make_eval_features(b, tokenizer, args.max_length, args.stride),
        batched=True,
    )
    features = [dict(f) for f in feat_ds]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    start_logits, end_logits = [], []
    bs = args.per_device_eval_batch_size
    with torch.no_grad():
        for i in range(0, len(features), bs):
            batch = features[i : i + bs]
            input_ids = torch.tensor([b["input_ids"] for b in batch], device=device)
            attention_mask = torch.tensor([b["attention_mask"] for b in batch], device=device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits.append(out.start_logits.detach().cpu().numpy())
            end_logits.append(out.end_logits.detach().cpu().numpy())
    start = np.concatenate(start_logits, axis=0) if start_logits else np.zeros((0, args.max_length))
    end = np.concatenate(end_logits, axis=0) if end_logits else np.zeros((0, args.max_length))
    return qa_utils.postprocess_qa_predictions(examples, features, (start, end))


def make_compute_metrics(eval_examples, eval_features, qa_utils):
    """Trainer-compatible compute_metrics using our local SQuAD v2 implementation."""
    def compute_metrics(p):
        start_logits, end_logits = p.predictions
        preds = qa_utils.postprocess_qa_predictions(eval_examples, eval_features, (start_logits, end_logits))
        m = qa_utils.squad_v2_metrics(preds, eval_examples, compute_best=False)
        return {
            "exact": m["exact"],
            "f1": m["f1"],
            "HasAns_exact": m["HasAns_exact"],
            "HasAns_f1": m["HasAns_f1"],
            "NoAns_exact": m["NoAns_exact"],
            "NoAns_f1": m["NoAns_f1"],
        }
    return compute_metrics


def save_training_curves(exp_dir, log_history, best_epoch=None):
    """Extract per-epoch train/eval metrics from Trainer log history and plot them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    per_epoch = {}
    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        d = per_epoch.setdefault(epoch, {})
        if "loss" in entry:
            d["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            d["eval_loss"] = entry["eval_loss"]
        if "eval_exact" in entry:
            d["eval_exact"] = entry["eval_exact"]
        if "eval_f1" in entry:
            d["eval_f1"] = entry["eval_f1"]
    for epoch in sorted(per_epoch):
        d = per_epoch[epoch]
        rows.append({
            "epoch": epoch,
            "train_loss": d.get("train_loss"),
            "eval_loss": d.get("eval_loss"),
            "eval_exact": d.get("eval_exact"),
            "eval_f1": d.get("eval_f1"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "training_history.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="train loss")
    axes[0].plot(df["epoch"], df["eval_loss"], marker="o", label="eval loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / validation loss")
    axes[0].legend()
    axes[1].plot(df["epoch"], df["eval_exact"], marker="o", label="eval EM")
    axes[1].plot(df["epoch"], df["eval_f1"], marker="o", label="eval F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation EM / F1")
    axes[1].legend()
    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"best epoch ({best_epoch})")
            ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, "training_curves.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(exp_dir, "training_curves.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_metrics(exp_dir, model_id, dataset, mode, metrics, kind_metrics, gold_metrics, predictions, config):
    os.makedirs(exp_dir, exist_ok=True)
    summary = {
        "model": model_id,
        "dataset": dataset,
        "mode": mode,
        **{k: float(v) for k, v in metrics.items()},
    }
    with open(os.path.join(exp_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(exp_dir, "metrics_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    if kind_metrics:
        kind_rows = []
        for kind, m in kind_metrics.items():
            kind_rows.append({"kind": kind, **{k: float(v) for k, v in m.items()}})
        with open(os.path.join(exp_dir, "metrics_by_question_type.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(kind_rows[0].keys()))
            writer.writeheader()
            writer.writerows(kind_rows)

    if gold_metrics:
        with open(os.path.join(exp_dir, "gold_audit_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"model": model_id, "dataset": dataset, "mode": mode,
                       **{k: float(v) for k, v in gold_metrics.items()}}, f, indent=2, ensure_ascii=False)

    if predictions:
        with open(os.path.join(exp_dir, "predictions_test.jsonl"), "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[{model_id} | {dataset} | {mode}] "
          f"exact={metrics.get('exact', float('nan')):.4f} f1={metrics.get('f1', float('nan')):.4f} "
          f"HasAns_f1={metrics.get('HasAns_f1', float('nan')):.4f} "
          f"NoAns_f1={metrics.get('NoAns_f1', float('nan')):.4f} -> {exp_dir}")


def run_one_experiment(args, model_id, dataset, mode, output_root, qa_utils):
    exp_dir = os.path.join(output_root, sanitize_model(model_id), dataset, mode)
    os.makedirs(exp_dir, exist_ok=True)

    config = {
        "model": model_id, "dataset": dataset, "mode": mode, "gpu": args.gpu,
        "output_dir": exp_dir, "seed": args.seed,
        "test_size": args.test_size, "dev_size": args.dev_size,
        "limit_contexts": args.limit_contexts, "gold_audit": args.gold_audit,
        "data_dir": args.data_dir, "hf_dataset": args.hf_dataset,
        "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
        "max_length": args.max_length, "stride": args.stride,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "grad_accum": args.grad_accum, "fp16": args.fp16,
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Heavy imports after CUDA_VISIBLE_DEVICES is set (in main).
    from datasets import Dataset
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoTokenizer,
        DefaultDataCollator,
        Trainer,
        TrainingArguments,
    )

    if args.data_dir or args.hf_dataset:
        train, dev, test = load_prepared_splits(args, qa_utils)
    else:
        train, dev, test = load_experiment_splits(args, qa_utils, dataset)
    gold = qa_utils.build_gold_dataset(args.gold_audit) if os.path.exists(args.gold_audit) else []
    print(f"[{model_id} | {dataset} | {mode}] train={len(train)} dev={len(dev)} "
          f"test={len(test)} gold={len(gold)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Common per-question-kind map for test metrics.
    kind_by_id = {r["id"]: qa_utils.question_kind(r["question"]) for r in test}

    if mode in ("zsl", "both") and not args.skip_zero_shot:
        model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        preds = predict_examples(model, tokenizer, test, args, qa_utils)
        metrics = qa_utils.squad_v2_metrics(preds, test)
        kind_metrics = qa_utils.per_question_type_metrics(preds, test, kind_by_id)
        gold_metrics = None
        if gold:
            gold_preds = predict_examples(model, tokenizer, gold, args, qa_utils)
            gold_metrics = qa_utils.squad_v2_metrics(gold_preds, gold)
        write_metrics(exp_dir, model_id, dataset, "zsl", metrics, kind_metrics,
                      gold_metrics, preds, config)
        del model
        import torch
        torch.cuda.empty_cache()

    if mode in ("ft", "both") and not args.skip_fine_tune:
        model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        train_ds = Dataset.from_list(train)
        dev_ds = Dataset.from_list(dev)
        tokenized_train = train_ds.map(
            lambda b: make_train_features(b, tokenizer, args.max_length, args.stride),
            batched=True, remove_columns=train_ds.column_names,
        )
        eval_mapped = dev_ds.map(
            lambda b: make_eval_features(b, tokenizer, args.max_length, args.stride),
            batched=True,
        )
        eval_features = [dict(f) for f in eval_mapped]
        # Trainer eval dataset must contain only tensor-friendly columns; the
        # full feature list is used by compute_metrics for post-processing.
        eval_feat_ds = eval_mapped.remove_columns(
            [c for c in eval_mapped.column_names if c not in {"input_ids", "attention_mask"}]
        )

        training_args = TrainingArguments(
            output_dir=os.path.join(exp_dir, "checkpoints"),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            fp16=args.fp16,
            logging_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            seed=args.seed,
            save_total_limit=2,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=eval_feat_ds,
            processing_class=tokenizer,
            data_collator=DefaultDataCollator(),
            compute_metrics=make_compute_metrics(dev, eval_features, qa_utils),
        )
        trainer.train()

        log_history = trainer.state.log_history
        best_epoch = None
        best_f1 = -1.0
        for entry in log_history:
            if "eval_f1" in entry and entry.get("eval_f1", -1.0) > best_f1:
                best_f1 = entry["eval_f1"]
                best_epoch = entry.get("epoch")
        save_training_curves(exp_dir, log_history, best_epoch=best_epoch)

        # Evaluate the best model (load_best_model_at_end already loaded it).
        preds = predict_examples(trainer.model, tokenizer, test, args, qa_utils)
        metrics = qa_utils.squad_v2_metrics(preds, test)
        kind_metrics = qa_utils.per_question_type_metrics(preds, test, kind_by_id)
        gold_metrics = None
        if gold:
            gold_preds = predict_examples(trainer.model, tokenizer, gold, args, qa_utils)
            gold_metrics = qa_utils.squad_v2_metrics(gold_preds, gold)
        write_metrics(exp_dir, model_id, dataset, "ft", metrics, kind_metrics,
                      gold_metrics, preds, config)

        if not args.save_models:
            import shutil
            ckpt_dir = os.path.join(exp_dir, "checkpoints")
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir, ignore_errors=True)
            log_dir = os.path.join(exp_dir, "logs")
            if os.path.isdir(log_dir):
                shutil.rmtree(log_dir, ignore_errors=True)

        import torch
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    args.output_dir = args.output_dir or default_output_dir()

    if args.plan_only:
        print_plan(args)
        return

    if args.all:
        if args.skip_zero_shot and args.skip_fine_tune:
            raise SystemExit("--skip-zero-shot and --skip-fine-tune are mutually exclusive for --all")
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        modes = []
        if not args.skip_zero_shot:
            modes.append("zsl")
        if not args.skip_fine_tune:
            modes.append("ft")
        if not modes:
            raise SystemExit("No modes to run")
        # Set GPU before importing torch/transformers.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import qa_dataset_utils as qa_utils
        for model in models:
            for dataset in datasets:
                for mode in modes:
                    run_one_experiment(args, model, dataset, mode, args.output_dir, qa_utils)
        return

    if not args.model or not args.dataset:
        raise SystemExit("--model and --dataset are required (or use --all / --plan-only)")

    if args.skip_zero_shot and args.skip_fine_tune:
        raise SystemExit("Both --skip-zero-shot and --skip-fine-tune set; nothing to run")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import qa_dataset_utils as qa_utils

    modes = []
    if args.mode in ("zsl", "both") and not args.skip_zero_shot:
        modes.append("zsl")
    if args.mode in ("ft", "both") and not args.skip_fine_tune:
        modes.append("ft")
    if not modes:
        raise SystemExit("No modes selected")

    for mode in modes:
        run_one_experiment(args, args.model, args.dataset, mode, args.output_dir, qa_utils)


if __name__ == "__main__":
    main()
