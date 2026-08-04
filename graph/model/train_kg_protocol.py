#!/usr/bin/env python3
"""Train an inductive KG-assisted NEMSIS protocol classifier.

The persistent graph contains clinical codes, ICD-like parent categories, and
protocols. PCRs are queries rather than graph nodes, so unseen PCRs can be
classified without adding target edges at inference time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEP = "~|~"
FIELD_NAMES = (
    "PrimarySymptomCode",
    "PrimaryImpressionCode",
    "AdditionalSymptomCode",
    "SecondaryImpressionCode",
)


@dataclass
class Config:
    data: str
    output_dir: str
    cache: str | None = None
    raw_dir: str | None = None
    use_raw_context: bool = False
    seed: int = 42
    epochs: int = 20
    batch_size: int = 4096
    dim: int = 128
    rgcn_layers: int = 2
    dropout: float = 0.2
    lr: float = 2e-3
    weight_decay: float = 1e-5
    patience: int = 4
    num_workers: int = 0
    max_codes_per_field: int = 12
    class_weight: str = "sqrt"
    device: str = "auto"


def clean_cell(value: str) -> str:
    return value.strip().strip("'").strip('"')


def row_values(line: str) -> list[str]:
    return [clean_cell(x) for x in line.rstrip("\r\n").split(SEP)]


def stable_split(signature: str, seed: int) -> int:
    """Group-hash split: train=0, val=1, test=2, approximately 70/10/20."""
    digest = hashlib.blake2b(
        f"{seed}|{signature}".encode(), digest_size=8
    ).digest()
    bucket = int.from_bytes(digest, "little") % 10
    return 0 if bucket < 7 else (1 if bucket == 7 else 2)


def parent_code(code: str) -> str:
    """Coarse ICD-like category; namespace prevents collisions with leaf codes."""
    stem = code.split(".")[0]
    if len(stem) >= 3 and stem[0].isalpha() and stem[1:3].isdigit():
        return f"ICD_CATEGORY::{stem[:3]}"
    return f"CODE_FAMILY::{stem}"


def load_examples(path: Path, seed: int, max_codes: int):
    pcr_keys: list[str] = []
    raw_fields: list[list[list[str]]] = []
    labels_raw: list[str] = []
    splits: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = row_values(next(f))
        indexes = {name: header.index(name) for name in FIELD_NAMES}
        key_idx, label_idx = header.index("PcrKey"), header.index("ProtocolIds")
        for line in f:
            values = row_values(line)
            if len(values) != len(header):
                continue
            fields = []
            for name in FIELD_NAMES:
                codes = list(dict.fromkeys(values[indexes[name]].split()))
                fields.append(codes[:max_codes] or ["__MISSING__"])
            signature = "\x1e".join(" ".join(x) for x in fields)
            pcr_keys.append(values[key_idx])
            raw_fields.append(fields)
            labels_raw.append(values[label_idx])
            splits.append(stable_split(signature, seed))
    return pcr_keys, raw_fields, labels_raw, np.asarray(splits, dtype=np.int8)


def read_context(
    raw_dir: Path, target_keys: set[str]
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read age and earliest vital record for target PCRs only.

    This scans the two large ASCII tables once. Missing measurements remain NaN.
    """
    feature_names = [
        "ageinyear", "eVitals_06_SBP", "eVitals_10_HR", "eVitals_12_SpO2",
        "eVitals_14_RR", "eVitals_16_ETCO2", "eVitals_27_Pain",
        "eVitals_29", "eVitals_30", "eVitals_31",
        "eVitals_19_GCS_Eye", "eVitals_20_GCS_Verbal", "eVitals_21_GCS_Motor",
    ]
    context: dict[str, np.ndarray] = {
        k: np.full(len(feature_names), np.nan, dtype=np.float32)
        for k in target_keys
    }

    computed = raw_dir / "ComputedElements.txt"
    if computed.exists():
        with computed.open("r", encoding="utf-8", errors="replace") as f:
            header = row_values(next(f))
            ki, ai = header.index("PcrKey"), header.index("ageinyear")
            for line in f:
                v = row_values(line)
                if len(v) <= max(ki, ai) or v[ki] not in context:
                    continue
                try:
                    context[v[ki]][0] = float(v[ai])
                except ValueError:
                    pass

    vital_file = raw_dir / "FACTPCRVITAL.txt"
    vital_cols = [
        "eVitals_06", "eVitals_10", "eVitals_12", "eVitals_14",
        "eVitals_16", "eVitals_27", "eVitals_29", "eVitals_30",
        "eVitals_31", "eVitals_19", "eVitals_20", "eVitals_21",
    ]
    earliest: dict[str, str] = {}
    if vital_file.exists():
        with vital_file.open("r", encoding="utf-8", errors="replace") as f:
            header = row_values(next(f))
            pos = {x: header.index(x) for x in ["PcrKey", "eVitals_01", *vital_cols]}
            for line in f:
                v = row_values(line)
                if len(v) != len(header):
                    continue
                key, timestamp = v[pos["PcrKey"]], v[pos["eVitals_01"]]
                if key not in context or (key in earliest and timestamp >= earliest[key]):
                    continue
                earliest[key] = timestamp
                for j, col in enumerate(vital_cols, start=1):
                    try:
                        context[key][j] = float(v[pos[col]])
                    except ValueError:
                        context[key][j] = np.nan
    return context, feature_names


def build_arrays(config: Config):
    keys, raw_fields, raw_labels, splits = load_examples(
        Path(config.data), config.seed, config.max_codes_per_field
    )
    # Vocabularies intentionally come from train only. Unknown validation/test
    # codes map to UNK, giving a genuinely inductive evaluation.
    train_i = np.flatnonzero(splits == 0)
    code_set = {"__PAD__", "__UNK__", "__MISSING__"}
    label_set = set()
    for i in train_i:
        label_set.add(raw_labels[i])
        for field in raw_fields[i]:
            code_set.update(field)
    codes = sorted(code_set)
    # Protocol labels absent from train cannot be learned, but retaining all labels
    # keeps evaluation/reporting explicit.
    labels = sorted(set(raw_labels), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    code_to_id = {x: i for i, x in enumerate(codes)}
    label_to_id = {x: i for i, x in enumerate(labels)}

    n, f, width = len(keys), len(FIELD_NAMES), config.max_codes_per_field
    x = np.zeros((n, f, width), dtype=np.int32)
    mask = np.zeros((n, f, width), dtype=np.bool_)
    for i, fields in enumerate(raw_fields):
        for r, values in enumerate(fields):
            ids = [code_to_id.get(v, code_to_id["__UNK__"]) for v in values[:width]]
            x[i, r, :len(ids)] = ids
            mask[i, r, :len(ids)] = True
    y = np.asarray([label_to_id[v] for v in raw_labels], dtype=np.int64)

    context_names: list[str] = []
    context = np.empty((n, 0), dtype=np.float32)
    if config.use_raw_context:
        if not config.raw_dir:
            raise ValueError("--raw-dir is required with --use-raw-context")
        context_map, context_names = read_context(Path(config.raw_dir), set(keys))
        context = np.stack([context_map[k] for k in keys])
    return {
        "x": x, "mask": mask, "y": y, "split": splits, "context": context,
        "codes": np.asarray(codes, dtype=object),
        "labels": np.asarray(labels, dtype=object),
        "context_names": np.asarray(context_names, dtype=object),
    }


def load_or_prepare(config: Config):
    if config.cache and Path(config.cache).exists():
        with np.load(config.cache, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    arrays = build_arrays(config)
    if config.cache:
        Path(config.cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(config.cache, **arrays)
    return arrays


def build_relational_graph(arrays):
    """Build train-only typed edges: field-specific code↔protocol and code↔parent."""
    codes = arrays["codes"].tolist()
    n_codes = len(codes)
    labels = arrays["labels"].tolist()
    n_protocols = len(labels)
    code_to_id = {x: i for i, x in enumerate(codes)}
    parents = sorted({parent_code(x) for x in codes if not x.startswith("__")})
    parent_to_id = {x: n_codes + n_protocols + i for i, x in enumerate(parents)}
    protocol_offset = n_codes
    edge_counters = [Counter() for _ in range(10)]

    train = np.flatnonzero(arrays["split"] == 0)
    for i in train:
        protocol = protocol_offset + int(arrays["y"][i])
        for relation in range(4):
            ids = arrays["x"][i, relation][arrays["mask"][i, relation]]
            for code in set(map(int, ids)):
                edge_counters[relation][(code, protocol)] += 1
                edge_counters[relation + 4][(protocol, code)] += 1
    for code, idx in code_to_id.items():
        if code.startswith("__"):
            continue
        parent = parent_to_id[parent_code(code)]
        edge_counters[8][(idx, parent)] = 1
        edge_counters[9][(parent, idx)] = 1

    edge_index, edge_weight = [], []
    for counter in edge_counters:
        if not counter:
            edge_index.append(torch.empty((2, 0), dtype=torch.long))
            edge_weight.append(torch.empty(0))
            continue
        pairs = list(counter)
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        counts = torch.tensor([counter[p] for p in pairs], dtype=torch.float32)
        # Weighted mean aggregation normalized over incoming neighbors.
        degree = torch.zeros(n_codes + n_protocols + len(parents))
        degree.scatter_add_(0, dst, counts)
        weight = counts / degree[dst].clamp_min(1)
        edge_index.append(torch.stack([src, dst]))
        edge_weight.append(weight)
    metadata = {
        "n_codes": n_codes, "n_protocols": n_protocols,
        "n_parents": len(parents), "n_nodes": n_codes + n_protocols + len(parents),
        "protocol_offset": protocol_offset, "relations": 10,
    }
    return edge_index, edge_weight, metadata


class PCRDataset(Dataset):
    def __init__(self, arrays, split: int, context_mean=None, context_std=None):
        self.indices = np.flatnonzero(arrays["split"] == split)
        self.x, self.mask, self.y = arrays["x"], arrays["mask"], arrays["y"]
        raw = arrays["context"].astype(np.float32)
        if raw.shape[1]:
            missing = np.isnan(raw).astype(np.float32)
            filled = np.where(np.isnan(raw), context_mean, raw)
            normalized = (filled - context_mean) / context_std
            self.context = np.concatenate([normalized, missing], axis=1).astype(np.float32)
        else:
            self.context = raw

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        return (
            torch.from_numpy(self.x[i].astype(np.int64)),
            torch.from_numpy(self.mask[i]),
            torch.from_numpy(self.context[i]),
            torch.tensor(self.y[i], dtype=torch.long),
        )


class RelationalGraphEncoder(nn.Module):
    def __init__(self, n_nodes: int, dim: int, relations: int, layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(n_nodes, dim)
        self.self_linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(layers))
        self.rel_linears = nn.ModuleList(
            nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(relations))
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, edges, weights):
        h = self.embedding.weight
        for self_linear, rel_linears, norm in zip(
            self.self_linears, self.rel_linears, self.norms
        ):
            out = self_linear(h)
            for linear, edge, weight in zip(rel_linears, edges, weights):
                if edge.numel() == 0:
                    continue
                src, dst = edge
                message = linear(h[src]) * weight[:, None]
                out.index_add_(0, dst, message)
            h = norm(h + self.dropout(torch.relu(out)))
        return h


class KGProtocolModel(nn.Module):
    def __init__(self, graph_meta, dim, layers, dropout, context_dim):
        super().__init__()
        self.meta = graph_meta
        self.graph = RelationalGraphEncoder(
            graph_meta["n_nodes"], dim, graph_meta["relations"], layers, dropout
        )
        self.field_queries = nn.Parameter(torch.empty(4, dim))
        self.field_projection = nn.Sequential(
            nn.Linear(4 * dim, 2 * dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.context_projection = (
            nn.Sequential(nn.Linear(context_dim, dim), nn.ReLU(), nn.Dropout(dropout))
            if context_dim else None
        )
        fusion_dim = 3 * dim if context_dim else 2 * dim
        self.case_projection = nn.Linear(fusion_dim, dim)
        self.protocol_bias = nn.Parameter(torch.zeros(graph_meta["n_protocols"]))
        nn.init.xavier_uniform_(self.field_queries)

    def forward(self, code_ids, mask, context, edges, weights):
        h = self.graph(edges, weights)
        code_h = h[code_ids]  # [B, 4, K, D]
        scores = (code_h * self.field_queries[None, :, None, :]).sum(-1)
        scores = scores.masked_fill(~mask, -1e4)
        pooled = (torch.softmax(scores, dim=-1)[..., None] * code_h).sum(2)
        case_parts = [self.field_projection(pooled.flatten(1))]
        if self.context_projection is not None:
            case_parts.append(self.context_projection(context))
        case_h = nn.functional.normalize(
            self.case_projection(torch.cat(case_parts, dim=1)), dim=-1
        )
        start = self.meta["protocol_offset"]
        protocol_h = nn.functional.normalize(
            h[start:start + self.meta["n_protocols"]], dim=-1
        )
        return case_h @ protocol_h.T * math.sqrt(case_h.shape[-1]) + self.protocol_bias


def calculate_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    ranking = np.argsort(-logits, axis=1)
    top1 = float(np.mean(ranking[:, :1] == y_true[:, None]))
    top3 = float(np.mean(np.any(ranking[:, :min(3, logits.shape[1])] == y_true[:, None], axis=1)))
    top5 = float(np.mean(np.any(ranking[:, :min(5, logits.shape[1])] == y_true[:, None], axis=1)))
    pred = ranking[:, 0]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )
    return {
        "top1": top1, "top3": top3, "top5": top5, "accuracy": top1,
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1),
    }


@torch.no_grad()
def evaluate(model, loader, edges, weights, device):
    model.eval()
    ys, outputs = [], []
    for x, mask, context, y in loader:
        logits = model(
            x.to(device), mask.to(device), context.to(device), edges, weights
        )
        ys.append(y.numpy())
        outputs.append(logits.cpu().numpy())
    return calculate_metrics(np.concatenate(ys), np.concatenate(outputs))


def train(config: Config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available()
        else ("cpu" if config.device == "auto" else config.device)
    )
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = load_or_prepare(config)
    split_counts = np.bincount(arrays["split"], minlength=3)
    print(f"split counts train/val/test: {split_counts.tolist()}")

    raw_context = arrays["context"].astype(np.float32)
    train_rows = arrays["split"] == 0
    if raw_context.shape[1]:
        mean = np.nanmean(raw_context[train_rows], axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nanstd(raw_context[train_rows], axis=0)
        std = np.where((std < 1e-6) | np.isnan(std), 1.0, std)
    else:
        mean = std = np.empty(0, dtype=np.float32)
    datasets = [
        PCRDataset(arrays, s, mean, std) for s in range(3)
    ]
    loaders = [
        DataLoader(ds, batch_size=config.batch_size, shuffle=(s == 0),
                   num_workers=config.num_workers, pin_memory=device.type == "cuda")
        for s, ds in enumerate(datasets)
    ]
    edges, weights, meta = build_relational_graph(arrays)
    edges = [x.to(device) for x in edges]
    weights = [x.to(device) for x in weights]
    model = KGProtocolModel(
        meta, config.dim, config.rgcn_layers, config.dropout,
        datasets[0].context.shape[1],
    ).to(device)

    counts = np.bincount(
        arrays["y"][train_rows], minlength=meta["n_protocols"]
    ).astype(np.float32)
    if config.class_weight == "none":
        class_weights = np.ones_like(counts)
    elif config.class_weight == "inverse":
        class_weights = counts.sum() / np.maximum(counts, 1)
    else:
        class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    best, stale = -1.0, 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        for x, mask, context, y in loaders[0]:
            x, mask = x.to(device), mask.to(device)
            context, y = context.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, mask, context, edges, weights), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
        val = evaluate(model, loaders[1], edges, weights, device)
        record = {"epoch": epoch, "train_loss": loss_sum / seen, **val}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val["macro_f1"] > best:
            best, stale = val["macro_f1"], 0
            torch.save(
                {
                    "model_state": model.state_dict(), "config": asdict(config),
                    "graph_meta": meta, "codes": arrays["codes"].tolist(),
                    "labels": arrays["labels"].tolist(), "context_mean": mean,
                    "context_std": std,
                },
                out / "best_model.pt",
            )
        else:
            stale += 1
            if stale >= config.patience:
                break

    checkpoint = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, loaders[2], edges, weights, device)
    report = {
        "split_counts": {"train": int(split_counts[0]), "validation": int(split_counts[1]),
                         "test": int(split_counts[2])},
        "best_validation_macro_f1": best,
        "test": test_metrics,
        "history": history,
    }
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "vocabulary.json").write_text(
        json.dumps({"codes": arrays["codes"].tolist(), "labels": arrays["labels"].tolist(),
                    "context_features": arrays["context_names"].tolist()},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print("test:", json.dumps(test_metrics, ensure_ascii=False))


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", default="outputs/kg_protocol")
    p.add_argument("--cache")
    p.add_argument("--raw-dir")
    p.add_argument("--use-raw-context", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--rgcn-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-codes-per-field", type=int, default=12)
    p.add_argument("--class-weight", choices=["none", "sqrt", "inverse"], default="sqrt")
    p.add_argument("--device", default="auto")
    return Config(**vars(p.parse_args()))


if __name__ == "__main__":
    train(parse_args())

