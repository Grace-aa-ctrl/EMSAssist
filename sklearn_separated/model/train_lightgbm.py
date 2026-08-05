#!/usr/bin/env python3
"""Tune and evaluate LightGBM on the pre-split ICD-10 bag dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy import sparse
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
        default=Path("/home/xiangling/EMSAssist/sklearn_separated/evaluation/lightgbm_metrics.txt"),
    )
    parser.add_argument("--max-rounds", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=0)
    return parser.parse_args()


def load_split(data_dir: Path, split: str):
    x = sparse.load_npz(data_dir / f"{split}_x.npz").tocsr()
    y = np.load(data_dir / f"{split}_y.npy", allow_pickle=False)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{split}: X has {x.shape[0]} rows but y has {y.shape[0]}")
    return x, y


def top_k_accuracy(probabilities: np.ndarray, y: np.ndarray, k: int) -> float:
    k = min(k, probabilities.shape[1])
    top_indices = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_indices == y[:, None], axis=1)))


def main() -> None:
    args = parse_args()
    print("Loading train/validation/test data...", flush=True)
    train_x, train_y = load_split(args.data_dir, "train")
    val_x, val_y = load_split(args.data_dir, "val")
    test_x, test_y = load_split(args.data_dir, "test")

    classes = np.unique(train_y)
    expected_classes = np.arange(len(classes))
    if not np.array_equal(classes, expected_classes):
        raise ValueError("LightGBM requires labels encoded contiguously from 0 to num_class-1")

    # User-provided baseline parameters. bagging_freq activates bagging_fraction.
    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "data_random_seed": args.seed,
        "num_threads": args.num_threads,
        "verbosity": -1,
    }

    print("Tuning boosting rounds with the validation set...", flush=True)
    started = time.time()
    train_data = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    val_data = lgb.Dataset(val_x, label=val_y, reference=train_data, free_raw_data=False)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.max_rounds,
        valid_sets=[val_data],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=10),
        ],
    )
    tuning_seconds = time.time() - started
    best_iteration = model.best_iteration
    best_validation_logloss = float(model.best_score["validation"]["multi_logloss"])
    val_probabilities = model.predict(val_x, num_iteration=best_iteration)
    validation_accuracy = float(
        accuracy_score(val_y, np.argmax(val_probabilities, axis=1))
    )
    print(
        f"best_iteration={best_iteration}, validation accuracy={validation_accuracy:.8f}, "
        f"validation logloss={best_validation_logloss:.8f}",
        flush=True,
    )

    print("Refitting on train + validation...", flush=True)
    final_x = sparse.vstack((train_x, val_x), format="csr")
    final_y = np.concatenate((train_y, val_y))
    final_data = lgb.Dataset(final_x, label=final_y)
    refit_started = time.time()
    final_model = lgb.train(params, final_data, num_boost_round=best_iteration)
    refit_seconds = time.time() - refit_started

    probabilities = final_model.predict(test_x, num_iteration=best_iteration)
    predictions = np.argmax(probabilities, axis=1)
    metrics = {
        "top_1": top_k_accuracy(probabilities, test_y, 1),
        "top_3": top_k_accuracy(probabilities, test_y, 3),
        "top_5": top_k_accuracy(probabilities, test_y, 5),
        "accuracy": float(accuracy_score(test_y, predictions)),
    }
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y, predictions, average=average, zero_division=0
        )
        metrics[f"precision_{average}"] = float(precision)
        metrics[f"recall_{average}"] = float(recall)
        metrics[f"f1_{average}"] = float(f1)

    report = {
        "parameters": params,
        "validation_tuning": {
            "maximum_boosting_rounds": args.max_rounds,
            "early_stopping_rounds": args.early_stopping_rounds,
            "best_iteration": best_iteration,
            "best_multi_logloss": best_validation_logloss,
            "accuracy_at_best_iteration": validation_accuracy,
            "seconds": tuning_seconds,
        },
        "test": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "LightGBM evaluation",
        "===================",
        f"data_dir: {args.data_dir}",
        f"train_samples: {train_x.shape[0]}",
        f"validation_samples: {val_x.shape[0]}",
        f"test_samples: {test_x.shape[0]}",
        f"features: {train_x.shape[1]}",
        f"classes: {len(classes)}",
        "",
        "Validation tuning (early stopping on multi_logloss)",
        f"maximum_boosting_rounds: {args.max_rounds}",
        f"early_stopping_rounds: {args.early_stopping_rounds}",
        f"best_iteration: {best_iteration}",
        f"best_validation_multi_logloss: {best_validation_logloss:.8f}",
        f"validation_accuracy_at_best_iteration: {validation_accuracy:.8f}",
        f"tuning_seconds: {tuning_seconds:.1f}",
        f"final_refit_seconds: {refit_seconds:.1f}",
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
        json.dumps(report, indent=2),
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Metrics written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
