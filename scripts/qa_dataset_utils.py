"""Shared utilities for the M2 QA dataset preparation and ablation experiments.

This module is intentionally dependency-light: it uses only the Python standard
library (plus numpy for the metric/postprocessing helpers). Heavy libraries
(torch, transformers, datasets) are imported lazily by the scripts that need
them, so the dataset preparation can be smoke-tested without a GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Text normalization (same logic as qc_squadv2_datasets.ipynb)
# ---------------------------------------------------------------------------
def normalize_text(s: Optional[str]) -> str:
    """Unicode normalization + whitespace handling for comparison purposes."""
    if s is None:
        return ""
    s = str(s).lower()
    s = unicodedata.normalize("NFKC", s)
    s = "".join(
        ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn"
    )
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Question-kind mapping (same as qc_squadv2_datasets.ipynb)
# ---------------------------------------------------------------------------
QKIND_EN = {
    "hora": "time",
    "fecha": "date",
    "valor": "value",
    "lugar": "place",
    "objetos": "objects",
    "otro": "other",
}


def question_kind(question: str) -> str:
    q = normalize_text(question)
    if "hora" in q:
        return "hora"
    if "fecha" in q or "día" in q:
        return "fecha"
    if "valor" in q or "cuanto" in q or "cuánto" in q:
        return "valor"
    if "lugar" in q or "dónde" in q or "donde" in q or "dirección" in q or "direccion" in q or "calles" in q:
        return "lugar"
    if "objeto" in q or "robado" in q or "robaron" in q or "sustraid" in q or "sustraíd" in q:
        return "objetos"
    return "otro"


def question_kind_en(question: str) -> str:
    return QKIND_EN.get(question_kind(question), "other")


# ---------------------------------------------------------------------------
# JSONL loading / SQuAD v2 schema helpers
# ---------------------------------------------------------------------------
_CONTEXT_ID_RE = re.compile(r"^(?:merged|gemma|qwen)_(context_\d+)_([0-9a-f]+)$")


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records: Iterable[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def derive_context_id(record: dict) -> str:
    """Recover the original M2 context_id from a QC SQuAD v2 id.

    QC ids look like ``<model>_context_<n>_<hexhash>``, e.g.
    ``merged_context_2_63be43e3``. If the record already carries an explicit
    ``context_id`` field, that is returned instead.
    """
    if record.get("context_id") is not None:
        return str(record["context_id"])
    rid = str(record.get("id", ""))
    m = _CONTEXT_ID_RE.match(rid)
    if m:
        return m.group(1)
    # Fallback: reconstruct from id parts (robust to non-hex hashes).
    parts = rid.split("_")
    if len(parts) >= 3 and parts[1] == "context":
        return "_".join(parts[1:-1])
    raise ValueError(f"Cannot derive context_id from id: {rid!r}")


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize_hf_squad_row(row: dict, schema: str = "auto") -> dict:
    """Normalize a Hugging Face dataset row to the SQuAD v2 schema.

    ``schema`` is one of:
      - ``auto``: detect legacy rows by the presence of ``answer_text`` /
        ``impossible_find_answer`` columns (paper ``robos-question-answering``).
      - ``legacy``: force conversion from the paper's legacy schema.
      - ``squad2``: assume the row is already SQuAD v2.
    """
    rec = dict(row)
    if schema == "auto":
        schema = "legacy" if ("answer_text" in rec and "impossible_find_answer" in rec) else "squad2"
    if schema == "legacy":
        rec = _legacy_to_squad2(rec)
    return rec


def _legacy_to_squad2(rec: dict) -> dict:
    """Convert the paper's ``robos-question-answering`` schema to SQuAD v2.

    Legacy fields: ``index``, ``context``, ``question``, ``answer_text``,
    ``answer_start``, ``answer_end``, ``impossible_find_answer``.
    """
    context = str(rec.get("context") or "")
    question = str(rec.get("question") or "")

    if rec.get("index") is not None:
        rec["id"] = f"legacy_{rec['index']}"
    elif rec.get("id"):
        rec["id"] = str(rec["id"])
    else:
        rec["id"] = f"legacy_{hashlib.md5((context + question).encode('utf-8')).hexdigest()[:16]}"

    rec["context"] = context
    rec["question"] = question
    rec["is_impossible"] = _normalize_bool(rec.get("impossible_find_answer"))
    if rec["is_impossible"]:
        rec["answers"] = {"text": [], "answer_start": []}
    else:
        rec["answers"] = {
            "text": [str(rec.get("answer_text") or "").strip()],
            "answer_start": [-1 if rec.get("answer_start") is None else int(rec["answer_start"])],
        }

    if not rec.get("context_id"):
        rec["context_id"] = f"ctx_{hashlib.md5(context.encode('utf-8')).hexdigest()[:12]}"
    return rec


def validate_squad2_record(record: dict) -> List[str]:
    """Return a list of schema problems for one SQuAD v2 record (empty = ok)."""
    errors = []
    for field in ("id", "context", "question", "is_impossible", "answers"):
        if field not in record:
            errors.append(f"missing field: {field}")
            continue
    if errors:
        return errors
    answers = record["answers"]
    if not isinstance(answers, dict) or "text" not in answers or "answer_start" not in answers:
        errors.append("answers must be a dict with 'text' and 'answer_start'")
        return errors
    imp = _normalize_bool(record["is_impossible"])
    texts = answers.get("text") or []
    starts = answers.get("answer_start") or []
    if imp:
        if texts or starts:
            errors.append("is_impossible=True but answers are not empty")
    else:
        if len(texts) != 1 or len(starts) != 1:
            errors.append("is_impossible=False but answers must contain exactly one span")
        else:
            ctx = record["context"]
            start = starts[0]
            if not ctx.startswith(texts[0], start):
                errors.append("answer_start does not point to answer_text in context")
    return errors


def load_squad2_variant(path: str, name: str, strict: bool = True) -> List[dict]:
    """Load a QC SQuAD v2 JSONL and add ``context_id`` to each record.

    When ``strict=True`` (default), schema/offset problems raise an error.
    When ``strict=False``, problematic records are kept and their errors are
    stored in ``_schema_errors`` so callers can analyse readiness without
    aborting (used by the exploration notebook).
    """
    records = []
    for raw in load_jsonl(path):
        rec = dict(raw)
        errors = validate_squad2_record(rec)
        if errors:
            if strict:
                raise ValueError(f"Invalid SQuAD v2 record in {name} ({path}): {errors} :: {rec}")
            rec["_schema_errors"] = errors
        rec["context_id"] = derive_context_id(rec)
        rec["is_impossible"] = _normalize_bool(rec["is_impossible"])
        rec["_source"] = name
        records.append(rec)
    return records


def split_by_context(
    records: Sequence[dict],
    train_frac: float,
    dev_frac: float,
    test_frac: float,
    seed: int,
    limit_contexts: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (train_ids, dev_ids, test_ids) split at the context level.

    The split is deterministic for a given ``seed`` and context list, so all
    experiment variants share the same partition even when launched in parallel.
    """
    import random

    if abs((train_frac + dev_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("train_frac + dev_frac + test_frac must equal 1.0")
    context_ids = sorted({r["context_id"] for r in records})
    if limit_contexts is not None:
        context_ids = context_ids[: max(0, limit_contexts)]
    rng = random.Random(seed)
    shuffled = list(context_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_dev = int(round(n * dev_frac))
    test_ids = set(shuffled[:n_test])
    dev_ids = set(shuffled[n_test : n_test + n_dev])
    train_ids = set(shuffled[n_test + n_dev :])
    return sorted(train_ids), sorted(dev_ids), sorted(test_ids)


def filter_by_context(records: Sequence[dict], context_ids: set) -> List[dict]:
    return [r for r in records if r["context_id"] in context_ids]


# ---------------------------------------------------------------------------
# Gold audit reader
# ---------------------------------------------------------------------------
GOLD_REQUIRED_COLUMNS = ["row_id", "context", "question", "gold_answer", "gold_is_impossible"]


def build_gold_dataset(csv_path: str) -> List[dict]:
    """Convert the GLM-labeled audit CSV into SQuAD v2-style records.

    The reference answer is ``gold_answer`` and the reference answerability is
    ``gold_is_impossible``. If a gold answer is not a literal span of the
    context (should not happen), it is treated as unanswerable and logged.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Gold audit CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    missing = [c for c in GOLD_REQUIRED_COLUMNS if c not in rows[0]] if rows else GOLD_REQUIRED_COLUMNS
    if missing:
        raise ValueError(f"Gold audit CSV missing columns: {missing}")

    records = []
    warnings = 0
    for row in rows:
        context = str(row.get("context") or "")
        question = str(row.get("question") or "")
        gold_answer = str(row.get("gold_answer") or "").strip()
        imp = _normalize_bool(row.get("gold_is_impossible"))
        if not imp and gold_answer:
            start = context.find(gold_answer)
            if start < 0:
                # Gold answers are human/LLM-corrected references; they may be
                # paraphrases rather than literal spans. EM/F1 comparison only
                # uses the text, so keep the answer with a placeholder offset.
                warnings += 1
                start = -1
            answers = {"text": [gold_answer], "answer_start": [start]}
        else:
            answers = {"text": [], "answer_start": []}
        records.append(
            {
                "id": f"gold_{row.get('row_id', len(records))}",
                "context": context,
                "question": question,
                "is_impossible": imp,
                "answers": answers,
                "context_id": str(row.get("context_id") or ""),
                "model": str(row.get("model") or ""),
                "error_type": str(row.get("error_type") or "none"),
                "human_correct": row.get("human_correct", ""),
                "kind": question_kind(question),
            }
        )
    if warnings:
        print(f"[gold audit] {warnings} gold answers were not literal spans; "
              f"kept with placeholder offset (-1) for EM/F1 comparison.")
    return records


# ---------------------------------------------------------------------------
# SQuAD v2 metrics (official EM/F1 definitions, local implementation)
# ---------------------------------------------------------------------------
def _tokenize_for_metric(text: str) -> List[str]:
    """Official SQuAD v2 / HF evaluate-style normalization for EM/F1."""
    if text is None:
        return []
    return normalize_answer_official(text).split()


def normalize_answer_official(s: Optional[str]) -> str:
    """Lowercase, remove punctuation and collapse whitespace (like evaluate/squad_v2)."""
    import string

    if s is None:
        return ""
    text = str(s).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def _compute_f1(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return 0.0
    from collections import Counter

    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _best_em_f1(pred_text: str, gold_texts: Sequence[str]) -> Tuple[float, float]:
    pred_tokens = _tokenize_for_metric(pred_text)
    em = 0.0
    f1 = 0.0
    for gold in gold_texts:
        gold_tokens = _tokenize_for_metric(gold)
        if pred_tokens == gold_tokens:
            em = 1.0
        f1 = max(f1, _compute_f1(pred_tokens, gold_tokens))
    return em, f1


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def squad_v2_metrics(
    predictions: Sequence[dict],
    references: Sequence[dict],
    threshold: float = 0.0,
    compute_best: bool = True,
) -> dict:
    """Compute overall / HasAns / NoAns EM+F1 plus best-threshold variants.

    ``predictions``: list of {"id", "prediction_text", "no_answer_probability"}.
    ``references``: list of {"id", "is_impossible", "answers": {"text": [...],
    "answer_start": [...]}}.
    """
    ref_by_id = {r["id"]: r for r in references}

    def eval_at(th: float) -> dict:
        ems, f1s = [], []
        has_em, has_f1, no_em, no_f1 = [], [], [], []
        for pred in predictions:
            ref = ref_by_id.get(pred["id"])
            if ref is None:
                raise KeyError(f"Prediction id {pred['id']!r} has no reference")
            text = pred.get("prediction_text", "")
            if pred.get("no_answer_probability", float("-inf")) > th:
                text = ""
            if ref["is_impossible"]:
                em = 1.0 if text == "" else 0.0
                f1 = 1.0 if text == "" else 0.0
                no_em.append(em)
                no_f1.append(f1)
            else:
                if text == "":
                    em, f1 = 0.0, 0.0
                else:
                    em, f1 = _best_em_f1(text, ref["answers"].get("text") or [])
                has_em.append(em)
                has_f1.append(f1)
            ems.append(em)
            f1s.append(f1)
        return {
            "exact": _mean(ems),
            "f1": _mean(f1s),
            "total": len(ems),
            "HasAns_exact": _mean(has_em),
            "HasAns_f1": _mean(has_f1),
            "HasAns_total": len(has_em),
            "NoAns_exact": _mean(no_em),
            "NoAns_f1": _mean(no_f1),
            "NoAns_total": len(no_em),
        }

    result = eval_at(threshold)
    result["threshold"] = threshold

    if compute_best and predictions:
        scores = sorted({p.get("no_answer_probability", float("-inf")) for p in predictions})
        thresholds = [float("-inf"), *scores, float("inf")]
        best_exact, best_f1 = -1.0, -1.0
        best_exact_thresh, best_f1_thresh = 0.0, 0.0
        for th in thresholds:
            r = eval_at(th)
            if r["exact"] > best_exact:
                best_exact = r["exact"]
                best_exact_thresh = th
            if r["f1"] > best_f1:
                best_f1 = r["f1"]
                best_f1_thresh = th
        result["best_exact"] = best_exact
        result["best_exact_thresh"] = best_exact_thresh
        result["best_f1"] = best_f1
        result["best_f1_thresh"] = best_f1_thresh
    return result


def per_question_type_metrics(
    predictions: Sequence[dict],
    references: Sequence[dict],
    kind_by_id: Dict[str, str],
) -> Dict[str, dict]:
    kinds = sorted({k for k in kind_by_id.values()})
    out = {}
    for kind in kinds:
        ids = {rid for rid, k in kind_by_id.items() if k == kind}
        preds = [p for p in predictions if p["id"] in ids]
        refs = [r for r in references if r["id"] in ids]
        if preds:
            out[QKIND_EN.get(kind, kind)] = squad_v2_metrics(preds, refs, compute_best=False)
    return out


# ---------------------------------------------------------------------------
# QA prediction post-processing (HF-style, max-over-windows)
# ---------------------------------------------------------------------------
def postprocess_qa_predictions(
    examples: Sequence[dict],
    features: Sequence[dict],
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 100,
) -> List[dict]:
    """Convert per-feature start/end logits into one prediction per example.

    ``examples``: raw SQuAD v2 records (must contain "id", "context").
    ``features``: tokenized features with keys "example_id", "offset_mapping",
    "input_ids", "attention_mask" (the tokenizer output after mapping).
    ``raw_predictions``: (start_logits, end_logits) arrays of shape
    (n_features, seq_len).
    """
    start_logits, end_logits = raw_predictions
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    example_index_to_features: Dict[str, List[int]] = {}
    for feat_idx, feat in enumerate(features):
        ex_id = feat["example_id"]
        example_index_to_features.setdefault(ex_id, []).append(feat_idx)

    example_by_id = {ex["id"]: ex for ex in examples}
    out = []
    for ex in examples:
        ex_id = ex["id"]
        feat_idxs = example_index_to_features.get(ex_id, [])
        if not feat_idxs:
            out.append({"id": ex_id, "prediction_text": "", "no_answer_probability": float("inf")})
            continue

        # Aggregate the best span scores across windows, and take the minimum
        # null score (i.e. the strongest evidence that an answer exists).
        # Key candidates by (feature_idx, start, end) so that text is always
        # reconstructed from the same window that produced the score.
        best_scores: Dict[Tuple[int, int, int], float] = {}
        min_null_score = float("inf")
        for feat_idx in feat_idxs:
            offsets = features[feat_idx]["offset_mapping"]
            sl = start_logits[feat_idx]
            el = end_logits[feat_idx]
            null_score = float(sl[0] + el[0])
            min_null_score = min(min_null_score, null_score)

            # Keep only the top n_best start/end positions per window to avoid
            # an O(seq_len^2) scan (HF-style n_best decoding).
            start_idx = np.argsort(sl)[-n_best_size:]
            end_idx = np.argsort(el)[-n_best_size:]
            for start in start_idx:
                if offsets[start][0] == 0 and offsets[start][1] == 0:
                    continue
                for end in end_idx:
                    if end < start or end - start >= max_answer_length:
                        continue
                    if offsets[end][0] == 0 and offsets[end][1] == 0:
                        continue
                    score = float(sl[start] + el[end])
                    key = (int(feat_idx), int(start), int(end))
                    best_scores[key] = max(best_scores.get(key, float("-inf")), score)

        best_text = ""
        best_score = float("-inf")
        for (feat_idx, start, end), score in best_scores.items():
            if score > best_score:
                offsets = features[feat_idx]["offset_mapping"]
                if start < len(offsets) and end < len(offsets):
                    span = offsets[start][1] - offsets[start][0]
                    if span > 0:
                        context = example_by_id[ex_id]["context"]
                        text = context[offsets[start][0] : offsets[end][1]]
                        if text.strip():
                            best_text = text
                            best_score = score
        # SQuAD v2 official convention: no_answer_probability is the null score
        # minus the best non-null span score. A threshold of 0.0 then predicts
        # "no answer" only when the null hypothesis scores higher than the best
        # extracted span (the HF run_squad.py convention).
        if best_score == float("-inf"):
            no_answer_probability = float("inf")
        else:
            no_answer_probability = min_null_score - best_score
        out.append(
            {
                "id": ex_id,
                "prediction_text": best_text,
                "no_answer_probability": no_answer_probability,
            }
        )
    return out
