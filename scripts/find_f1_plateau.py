"""Analyze a fine-tuning `training_history.csv` to find where eval F1 plateaus.

The CSV is produced by scripts/run_qa_ablation.py in each fine-tuned run
(out_experiments/<run_id>/<model>/<dataset>/ft/training_history.csv).

Usage:
    python scripts/find_f1_plateau.py --history <path> [--tolerance 0.1] [--min-epochs 3]

Definition used:
    plateau_epoch = the earliest epoch e (>= --min-epochs) such that the best
    eval F1 from e onward is within --tolerance F1 points of eval F1 at e.
    In other words, from that epoch on, training gains less than tolerance.
"""

from __future__ import annotations

import argparse

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", required=True, help="Path to training_history.csv")
    p.add_argument("--tolerance", type=float, default=0.1,
                   help="F1-point threshold for negligible improvement (default 0.1)")
    p.add_argument("--min-epochs", type=int, default=3,
                   help="Do not report a plateau before this epoch (default 3)")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.history)
    if "epoch" not in df or "eval_f1" not in df:
        raise SystemExit("training_history.csv must contain 'epoch' and 'eval_f1' columns")

    df = df.dropna(subset=["eval_f1"]).sort_values("epoch").reset_index(drop=True)
    if df.empty:
        raise SystemExit("No eval_f1 rows found in history")

    epochs = df["epoch"].astype(float)
    f1 = df["eval_f1"].astype(float)

    print(f"History: {args.history}")
    print(f"Tolerance: {args.tolerance} F1 points | min epochs before plateau: {args.min_epochs}\n")
    print(f"{'Epoch':>6} {'eval_f1':>10} {'delta':>10}")
    prev = None
    for ep, score in zip(epochs, f1):
        delta = "" if prev is None else f"{score - prev:+.3f}"
        print(f"{ep:6.1f} {score:10.4f} {delta:>10}")
        prev = score

    best_idx = int(f1.idxmax())
    best_epoch = float(epochs.iloc[best_idx])
    best_f1 = float(f1.iloc[best_idx])

    plateau_epoch = None
    plateau_f1 = None
    for i in range(len(df)):
        ep = float(epochs.iloc[i])
        if ep < args.min_epochs:
            continue
        future_best = float(f1.iloc[i:].max())
        if future_best - float(f1.iloc[i]) < args.tolerance:
            plateau_epoch = ep
            plateau_f1 = float(f1.iloc[i])
            break

    print(f"\nBest epoch: {best_epoch:.1f} (eval F1 = {best_f1:.4f})")
    if plateau_epoch is not None:
        print(f"Plateau detected from epoch {plateau_epoch:.1f} "
              f"(eval F1 = {plateau_f1:.4f}, within {args.tolerance} of future best)")
        print(f"Suggestion: {max(best_epoch, plateau_epoch):.0f} epochs is a safe budget; "
              f"use early stopping patience 2-3 if you want to save compute.")
    else:
        print(f"No plateau detected up to epoch {float(epochs.iloc[-1]):.1f}: "
              f"F1 is still improving by >= {args.tolerance} points. Run more epochs.")


if __name__ == "__main__":
    main()
