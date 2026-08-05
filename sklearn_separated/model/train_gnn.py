#!/usr/bin/env python3
"""Train a feature-cooccurrence GNN for multiclass sparse-vector classification."""

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
        default=Path("/home/xiangling/EMSAssist/sklearn_separated/evaluation/gnn_metrics.txt"),
    )
    parser.add_argument("--hidden-dimensions", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--graph-neighbors", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_split(data_dir: Path, split: str):
    x = sparse.load_npz(data_dir / f"{split}_x.npz").tocsr().astype(np.float32)
    y = np.load(data_dir / f"{split}_y.npy", allow_pickle=False).astype(np.int64)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{split}: X has {x.shape[0]} rows but y has {y.shape[0]}")
    return x, y


def row_normalize(x: sparse.csr_matrix) -> sparse.csr_matrix:
    row_sums = np.asarray(x.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sums, dtype=np.float32)
    nonzero = row_sums != 0
    inverse[nonzero] = 1.0 / row_sums[nonzero]
    return sparse.diags(inverse).dot(x).tocsr().astype(np.float32)


def keep_top_k_per_row(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    matrix = matrix.tocsr()
    new_indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int64)
    selected_indices = []
    selected_data = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        data = matrix.data[start:end]
        indices = matrix.indices[start:end]
        if len(data) > k:
            chosen = np.argpartition(data, -k)[-k:]
            data = data[chosen]
            indices = indices[chosen]
        selected_indices.append(indices)
        selected_data.append(data)
        new_indptr[row + 1] = new_indptr[row] + len(data)
    return sparse.csr_matrix(
        (np.concatenate(selected_data), np.concatenate(selected_indices), new_indptr),
        shape=matrix.shape,
    )


def build_normalized_graph(x: sparse.csr_matrix, neighbors: int) -> sparse.csr_matrix:
    print("Building feature co-occurrence graph from training data...", flush=True)
    binary = x.copy()
    binary.data = np.ones_like(binary.data, dtype=np.float32)
    cooccurrence = (binary.T @ binary).tocsr().astype(np.float32)
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    cooccurrence = keep_top_k_per_row(cooccurrence, neighbors)
    cooccurrence = cooccurrence.maximum(cooccurrence.T).tocsr()
    cooccurrence.setdiag(1.0)
    degrees = np.asarray(cooccurrence.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degrees, dtype=np.float32)
    nonzero = degrees > 0
    inverse_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    normalized = sparse.diags(inverse_sqrt) @ cooccurrence @ sparse.diags(inverse_sqrt)
    normalized = normalized.tocsr().astype(np.float32)
    print(
        f"Graph nodes={normalized.shape[0]}, undirected normalized nnz={normalized.nnz}",
        flush=True,
    )
    return normalized


def scipy_to_torch_sparse(matrix: sparse.csr_matrix, device: torch.device):
    matrix = matrix.tocsr()
    crow = torch.as_tensor(matrix.indptr, dtype=torch.int64, device=device)
    col = torch.as_tensor(matrix.indices, dtype=torch.int64, device=device)
    values = torch.as_tensor(matrix.data, dtype=torch.float32, device=device)
    return torch.sparse_csr_tensor(crow, col, values, size=matrix.shape, device=device)


class FeatureGraphGNN(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.node_embeddings = nn.Parameter(torch.empty(n_features, hidden_dim))
        self.graph_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.node_embeddings)

    def encode_nodes(self, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(torch.sparse.mm(adjacency, self.node_embeddings))
        hidden = nn.functional.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.graph_linear(hidden)
        hidden = torch.relu(torch.sparse.mm(adjacency, hidden))
        return hidden

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        node_states = self.encode_nodes(adjacency)
        sample_states = torch.sparse.mm(x, node_states)
        sample_states = nn.functional.dropout(
            sample_states, p=self.dropout, training=self.training
        )
        return self.classifier(sample_states)


@torch.no_grad()
def evaluate(model, x, adjacency, y) -> tuple[float, np.ndarray]:
    model.eval()
    logits = model(x, adjacency)
    predictions = logits.argmax(dim=1).cpu().numpy()
    return float(accuracy_score(y, predictions)), logits.cpu().numpy()


def fit_candidate(args, hidden_dim, train_x, train_y, val_x, val_y, adjacency, device):
    torch.manual_seed(args.seed)
    model = FeatureGraphGNN(
        train_x.shape[1], hidden_dim, len(np.unique(train_y)), args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    targets = torch.as_tensor(train_y, dtype=torch.long, device=device)
    best_accuracy = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x, adjacency)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        validation_accuracy, _ = evaluate(model, val_x, adjacency, val_y)
        seconds = time.time() - started
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "validation_accuracy": validation_accuracy,
                "seconds": seconds,
            }
        )
        print(
            f"hidden_dim={hidden_dim}, epoch={epoch}: loss={loss.item():.6f}, "
            f"validation_accuracy={validation_accuracy:.8f} ({seconds:.1f}s)",
            flush=True,
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    return {
        "hidden_dim": hidden_dim,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "history": history,
    }


def train_fixed_epochs(args, hidden_dim, epochs, x, y, adjacency, device):
    torch.manual_seed(args.seed)
    model = FeatureGraphGNN(
        x.shape[1], hidden_dim, len(np.unique(y)), args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    targets = torch.as_tensor(y, dtype=torch.long, device=device)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, adjacency)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        seconds = time.time() - started
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "seconds": seconds})
        print(f"refit epoch={epoch}: loss={loss.item():.6f} ({seconds:.1f}s)", flush=True)
    return model, history


def top_k_accuracy(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_indices == y[:, None], axis=1)))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    print(f"Using device: {device}", flush=True)
    train_x_raw, train_y = load_split(args.data_dir, "train")
    val_x_raw, val_y = load_split(args.data_dir, "val")
    test_x_raw, test_y = load_split(args.data_dir, "test")
    classes = np.unique(train_y)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Labels must be contiguous integers starting at zero")

    train_graph = build_normalized_graph(train_x_raw, args.graph_neighbors)
    train_x = scipy_to_torch_sparse(row_normalize(train_x_raw), device)
    val_x = scipy_to_torch_sparse(row_normalize(val_x_raw), device)
    train_adjacency = scipy_to_torch_sparse(train_graph, device)

    candidates = []
    best_candidate = None
    for hidden_dim in args.hidden_dimensions:
        result = fit_candidate(
            args, hidden_dim, train_x, train_y, val_x, val_y, train_adjacency, device
        )
        candidates.append(result)
        if (
            best_candidate is None
            or result["best_validation_accuracy"]
            > best_candidate["best_validation_accuracy"]
        ):
            best_candidate = result
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_hidden_dim = best_candidate["hidden_dim"]
    best_epoch = best_candidate["best_epoch"]
    print(
        f"Best hidden_dim={best_hidden_dim}; rebuilding graph and refitting for {best_epoch} epochs...",
        flush=True,
    )
    final_x_raw = sparse.vstack((train_x_raw, val_x_raw), format="csr")
    final_y = np.concatenate((train_y, val_y))
    final_graph = build_normalized_graph(final_x_raw, args.graph_neighbors)
    final_x = scipy_to_torch_sparse(row_normalize(final_x_raw), device)
    test_x = scipy_to_torch_sparse(row_normalize(test_x_raw), device)
    final_adjacency = scipy_to_torch_sparse(final_graph, device)
    final_model, refit_history = train_fixed_epochs(
        args,
        best_hidden_dim,
        best_epoch,
        final_x,
        final_y,
        final_adjacency,
        device,
    )
    _, test_scores = evaluate(final_model, test_x, final_adjacency, test_y)
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
        "model": "two_layer_feature_cooccurrence_gnn",
        "device": str(device),
        "graph_neighbors": args.graph_neighbors,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "validation_candidates": candidates,
        "best_hidden_dim": best_hidden_dim,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_candidate["best_validation_accuracy"],
        "refit_history": refit_history,
        "test": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "GNN evaluation",
        "==============",
        f"data_dir: {args.data_dir}",
        f"train_samples: {train_x_raw.shape[0]}",
        f"validation_samples: {val_x_raw.shape[0]}",
        f"test_samples: {test_x_raw.shape[0]}",
        f"features/graph_nodes: {train_x_raw.shape[1]}",
        f"classes: {len(classes)}",
        "graph: training-only feature co-occurrence graph",
        f"graph_neighbors: {args.graph_neighbors}",
        f"device: {device}",
        "",
        "Validation tuning (selection metric: accuracy)",
    ]
    lines.extend(
        f"hidden_dim={item['hidden_dim']}: best_epoch={item['best_epoch']}, "
        f"best_accuracy={item['best_validation_accuracy']:.8f}"
        for item in candidates
    )
    lines.extend(
        [
            f"best_hidden_dim: {best_hidden_dim}",
            f"best_epoch: {best_epoch}",
            f"best_validation_accuracy: {best_candidate['best_validation_accuracy']:.8f}",
            "",
            "Test metrics (final graph/model refit on train + validation)",
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
