#!/usr/bin/env python3
"""Tune and evaluate LogisticRegression on the pre-split ICD-10 bag data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/xiangling/EMSAssist/sklearn_separated/dataset/icd10_code_bag"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/xiangling/EMSAssist/sklearn_separated/evaluation/logistic_regression_metrics.txt"),
    )
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(data_dir: Path, split: str):
    x = sparse.load_npz(data_dir / f"{split}_x.npz").tocsr()
    y = np.load(data_dir / f"{split}_y.npy", allow_pickle=False)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{split}: X has {x.shape[0]} rows but y has {y.shape[0]}")
    return x, y


def top_k_accuracy(model: LogisticRegression, x, y, k: int) -> float:
    scores = model.decision_function(x)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    k = min(k, scores.shape[1])
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_labels = model.classes_[top_indices]
    return float(np.mean(np.any(top_labels == y[:, None], axis=1)))


def make_model(c: float, args: argparse.Namespace) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        solver="saga",
        penalty="l2",
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
    )


def main() -> None:
    args = parse_args()
    if any(c <= 0 for c in args.c_values):
        raise ValueError("All C values must be positive")

    print("Loading train/validation/test data...", flush=True)
    train_x, train_y = load_split(args.data_dir, "train")
    val_x, val_y = load_split(args.data_dir, "val")
    test_x, test_y = load_split(args.data_dir, "test")

    tuning = []
    best_c = None
    best_score = -np.inf
    for c in args.c_values:
        started = time.time()
        model = make_model(c, args)
        model.fit(train_x, train_y)
        score = accuracy_score(val_y, model.predict(val_x))
        elapsed = time.time() - started
        tuning.append({"C": c, "validation_accuracy": float(score), "seconds": elapsed})
        print(f"C={c:g}: validation accuracy={score:.8f} ({elapsed:.1f}s)", flush=True)
        if score > best_score:
            best_score, best_c = score, c

    # Refit on all non-test data after hyperparameter selection.
    final_x = sparse.vstack((train_x, val_x), format="csr")
    final_y = np.concatenate((train_y, val_y))
    print(f"Refitting with C={best_c:g} on train + validation...", flush=True)
    final_model = make_model(best_c, args)
    final_model.fit(final_x, final_y)

    predictions = final_model.predict(test_x)
    accuracy = float(accuracy_score(test_y, predictions))
    metrics = {
        "top_1": top_k_accuracy(final_model, test_x, test_y, 1),
        "top_3": top_k_accuracy(final_model, test_x, test_y, 3),
        "top_5": top_k_accuracy(final_model, test_x, test_y, 5),
        "accuracy": accuracy,
    }
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y, predictions, average=average, zero_division=0
        )
        metrics[f"precision_{average}"] = float(precision)
        metrics[f"recall_{average}"] = float(recall)
        metrics[f"f1_{average}"] = float(f1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "LogisticRegression evaluation",
        "=================================",
        f"data_dir: {args.data_dir}",
        f"train_samples: {train_x.shape[0]}",
        f"validation_samples: {val_x.shape[0]}",
        f"test_samples: {test_x.shape[0]}",
        f"features: {train_x.shape[1]}",
        f"classes: {len(final_model.classes_)}",
        "model: LogisticRegression(solver=saga, penalty=l2)",
        f"max_iter: {args.max_iter}",
        f"tol: {args.tol}",
        "",
        "Validation tuning (selection metric: accuracy)",
    ]
    lines.extend(
        f"C={row['C']:g}: accuracy={row['validation_accuracy']:.8f}, seconds={row['seconds']:.1f}"
        for row in tuning
    )
    lines.extend(
        [
            f"best_C: {best_c:g}",
            f"best_validation_accuracy: {best_score:.8f}",
            "",
            "Test metrics (final model refit on train + validation)",
            f"Top-1: {metrics['top_1']:.8f}",
            f"Top-3: {metrics['top_3']:.8f}",
            f"Top-5: {metrics['top_5']:.8f}",
            f"Accuracy: {metrics['accuracy']:.8f}",
            f"Precision (macro): {metrics['precision_macro']:.8f}",
            f"Recall (macro): {metrics['recall_macro']:.8f}",
            f"F1 Score (macro): {metrics['f1_macro']:.8f}",
            f"Precision (weighted): {metrics['precision_weighted']:.8f}",
            f"Recall (weighted): {metrics['recall_weighted']:.8f}",
            f"F1 Score (weighted): {metrics['f1_weighted']:.8f}",
            "",
            "JSON",
            json.dumps({"best_C": best_c, "validation": tuning, "test": metrics}, indent=2),
        ]
    )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Metrics written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
