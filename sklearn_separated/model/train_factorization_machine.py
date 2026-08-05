#!/usr/bin/env python3
"""Train and evaluate a multiclass Factorization Machine on sparse vectors."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn


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
        default=Path("/home/xiangling/EMSAssist/sklearn_separated/evaluation/factorization_machine_metrics.txt"),
    )
    parser.add_argument("--factor-dimensions", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_split(data_dir: Path, split: str):
    x = sparse.load_npz(data_dir / f"{split}_x.npz").tocsr().astype(np.float32)
    y = np.load(data_dir / f"{split}_y.npy", allow_pickle=False).astype(np.int64)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{split}: X has {x.shape[0]} rows but y has {y.shape[0]}")
    return x, y


class MulticlassFactorizationMachine(nn.Module):
    """One multiclass FM trained jointly with class-specific factor embeddings."""

    def __init__(self, n_features: int, n_classes: int, factor_dim: int):
        super().__init__()
        self.n_classes = n_classes
        self.factor_dim = factor_dim
        self.linear = nn.Embedding(n_features, n_classes, sparse=True)
        self.factors = nn.Embedding(
            n_features, n_classes * factor_dim, sparse=True
        )
        self.bias = nn.Parameter(torch.zeros(n_classes))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, mean=0.0, std=0.01)

    def forward(
        self,
        rows: torch.Tensor,
        columns: torch.Tensor,
        values: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        linear_terms = self.linear(columns) * values[:, None]
        linear_sum = torch.zeros(
            batch_size, self.n_classes, device=values.device, dtype=values.dtype
        )
        linear_sum.index_add_(0, rows, linear_terms)

        factors = self.factors(columns).view(-1, self.n_classes, self.factor_dim)
        weighted_factors = factors * values[:, None, None]
        factor_sum = torch.zeros(
            batch_size,
            self.n_classes,
            self.factor_dim,
            device=values.device,
            dtype=values.dtype,
        )
        squared_sum = torch.zeros_like(factor_sum)
        factor_sum.index_add_(0, rows, weighted_factors)
        squared_sum.index_add_(
            0, rows, factors.square() * values.square()[:, None, None]
        )
        interactions = 0.5 * (factor_sum.square() - squared_sum).sum(dim=2)
        return self.bias + linear_sum + interactions


def scipy_batch_to_tensors(x_batch: sparse.csr_matrix, device: torch.device):
    coo = x_batch.tocoo(copy=False)
    rows = torch.as_tensor(coo.row, dtype=torch.long, device=device)
    columns = torch.as_tensor(coo.col, dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return rows, columns, values


def make_optimizers(model: MulticlassFactorizationMachine, learning_rate: float):
    sparse_optimizer = torch.optim.SparseAdam(
        [model.linear.weight, model.factors.weight], lr=learning_rate
    )
    dense_optimizer = torch.optim.Adam([model.bias], lr=learning_rate)
    return sparse_optimizer, dense_optimizer


def train_epoch(model, x, y, batch_size, optimizers, device, rng) -> float:
    model.train()
    order = rng.permutation(x.shape[0])
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    sparse_optimizer, dense_optimizer = optimizers
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        x_batch = x[indices]
        rows, columns, values = scipy_batch_to_tensors(x_batch, device)
        targets = torch.as_tensor(y[indices], dtype=torch.long, device=device)
        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)
        logits = model(rows, columns, values, len(indices))
        loss = criterion(logits, targets)
        loss.backward()
        sparse_optimizer.step()
        dense_optimizer.step()
        loss_sum += loss.item() * len(indices)
    return loss_sum / x.shape[0]


@torch.no_grad()
def predict_scores(model, x, batch_size, device) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, x.shape[0], batch_size):
        x_batch = x[start : start + batch_size]
        rows, columns, values = scipy_batch_to_tensors(x_batch, device)
        logits = model(rows, columns, values, x_batch.shape[0])
        scores.append(logits.cpu().numpy())
    return np.concatenate(scores, axis=0)


def top_k_accuracy(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_indices == y[:, None], axis=1)))


def fit_candidate(args, factor_dim, train_x, train_y, val_x, val_y, device):
    torch.manual_seed(args.seed)
    model = MulticlassFactorizationMachine(
        train_x.shape[1], len(np.unique(train_y)), factor_dim
    ).to(device)
    optimizers = make_optimizers(model, args.learning_rate)
    rng = np.random.default_rng(args.seed)
    best_accuracy = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    best_state = None
    for epoch in range(1, args.max_epochs + 1):
        started = time.time()
        train_loss = train_epoch(
            model, train_x, train_y, args.batch_size, optimizers, device, rng
        )
        val_scores = predict_scores(model, val_x, args.batch_size, device)
        val_accuracy = float(accuracy_score(val_y, val_scores.argmax(axis=1)))
        seconds = time.time() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": val_accuracy,
                "seconds": seconds,
            }
        )
        print(
            f"factor_dim={factor_dim}, epoch={epoch}: train_loss={train_loss:.6f}, "
            f"validation_accuracy={val_accuracy:.8f} ({seconds:.1f}s)",
            flush=True,
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    return {
        "factor_dim": factor_dim,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "history": history,
        "state": best_state,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if any(d <= 0 for d in args.factor_dimensions):
        raise ValueError("All factor dimensions must be positive")

    print(f"Using device: {device}", flush=True)
    train_x, train_y = load_split(args.data_dir, "train")
    val_x, val_y = load_split(args.data_dir, "val")
    test_x, test_y = load_split(args.data_dir, "test")
    classes = np.unique(train_y)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Labels must be contiguous integers starting at zero")

    candidates = []
    best_candidate = None
    for factor_dim in args.factor_dimensions:
        result = fit_candidate(
            args, factor_dim, train_x, train_y, val_x, val_y, device
        )
        candidates.append({key: value for key, value in result.items() if key != "state"})
        if (
            best_candidate is None
            or result["best_validation_accuracy"]
            > best_candidate["best_validation_accuracy"]
        ):
            best_candidate = result
        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_factor_dim = best_candidate["factor_dim"]
    best_epoch = best_candidate["best_epoch"]
    print(
        f"Best factor_dim={best_factor_dim}, refitting for {best_epoch} epochs on train + validation...",
        flush=True,
    )
    final_x = sparse.vstack((train_x, val_x), format="csr")
    final_y = np.concatenate((train_y, val_y))
    torch.manual_seed(args.seed)
    final_model = MulticlassFactorizationMachine(
        final_x.shape[1], len(classes), best_factor_dim
    ).to(device)
    optimizers = make_optimizers(final_model, args.learning_rate)
    rng = np.random.default_rng(args.seed)
    refit_history = []
    for epoch in range(1, best_epoch + 1):
        started = time.time()
        loss = train_epoch(
            final_model, final_x, final_y, args.batch_size, optimizers, device, rng
        )
        seconds = time.time() - started
        refit_history.append({"epoch": epoch, "train_loss": loss, "seconds": seconds})
        print(f"refit epoch={epoch}: train_loss={loss:.6f} ({seconds:.1f}s)", flush=True)

    test_scores = predict_scores(final_model, test_x, args.batch_size, device)
    predictions = test_scores.argmax(axis=1)
    metrics = {
        "top_1": top_k_accuracy(test_scores, test_y, 1),
        "top_3": top_k_accuracy(test_scores, test_y, 3),
        "top_5": top_k_accuracy(test_scores, test_y, 5),
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
        "model": "multiclass_factorization_machine",
        "device": str(device),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_candidates": candidates,
        "best_factor_dim": best_factor_dim,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_candidate["best_validation_accuracy"],
        "refit_history": refit_history,
        "test": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Factorization Machine evaluation",
        "================================",
        f"data_dir: {args.data_dir}",
        f"train_samples: {train_x.shape[0]}",
        f"validation_samples: {val_x.shape[0]}",
        f"test_samples: {test_x.shape[0]}",
        f"features: {train_x.shape[1]}",
        f"classes: {len(classes)}",
        f"device: {device}",
        f"learning_rate: {args.learning_rate}",
        f"batch_size: {args.batch_size}",
        "",
        "Validation tuning (selection metric: accuracy)",
    ]
    lines.extend(
        f"factor_dim={item['factor_dim']}: best_epoch={item['best_epoch']}, "
        f"best_accuracy={item['best_validation_accuracy']:.8f}"
        for item in candidates
    )
    lines.extend(
        [
            f"best_factor_dim: {best_factor_dim}",
            f"best_epoch: {best_epoch}",
            f"best_validation_accuracy: {best_candidate['best_validation_accuracy']:.8f}",
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
    )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Metrics written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
