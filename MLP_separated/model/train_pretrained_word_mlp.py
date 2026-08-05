"""Tune an MLP on validation macro-F1 and evaluate once on the test split."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], num_classes: int, dropout: float):
        super().__init__()
        layers = []
        width = input_size
        for hidden in hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches(x, y, batch_size, shuffle, rng):
    order = np.arange(len(y))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        # Fancy indexing yields a writable contiguous array and avoids mmap warnings.
        yield np.asarray(x[idx], dtype=np.float32), y[idx]


def predict(model, x, y, batch_size, device):
    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for xb, yb in batches(x, y, batch_size, False, np.random.default_rng(0)):
            tensor = torch.from_numpy(xb).to(device, non_blocking=True)
            logits.append(model(tensor).float().cpu().numpy())
            targets.append(yb)
    return np.concatenate(logits), np.concatenate(targets)


def calculate_metrics(logits, y):
    pred = logits.argmax(axis=1)
    num_classes = logits.shape[1]
    result = {"top_1": float(np.mean(pred == y))}
    for k in (3, 5):
        effective_k = min(k, num_classes)
        top = np.argpartition(logits, -effective_k, axis=1)[:, -effective_k:]
        result[f"top_{k}"] = float(np.mean(np.any(top == y[:, None], axis=1)))
    result["accuracy"] = float(accuracy_score(y, pred))
    # Macro scores give every protocol equal importance despite class imbalance.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=np.arange(num_classes), average="macro", zero_division=0
    )
    result.update(precision=float(precision), recall=float(recall), f1=float(f1))
    return result


def train_candidate(config, data, metadata, args, candidate_id):
    train_x, train_y, val_x, val_y = data
    model = MLP(metadata["input_size"], config["hidden_sizes"], metadata["num_classes"], config["dropout"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    rng = np.random.default_rng(args.seed + candidate_id)
    best, stale = None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in batches(train_x, train_y, args.batch_size, True, rng):
            xb = torch.from_numpy(xb).to(args.device, non_blocking=True)
            yb = torch.from_numpy(yb).to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.device.startswith("cuda")):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(yb)
        val_logits, targets = predict(model, val_x, val_y, args.batch_size, args.device)
        metrics = calculate_metrics(val_logits, targets)
        print(f"candidate={candidate_id} epoch={epoch} loss={loss_sum/len(train_y):.6f} val_f1={metrics['f1']:.6f} val_accuracy={metrics['accuracy']:.6f}", flush=True)
        if best is None or metrics["f1"] > best["metrics"]["f1"] + 1e-6:
            best = {"epoch": epoch, "metrics": metrics,
                    "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best["state"])
    return model, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=Path("/home/xiangling/EMSAssist/MLP_separated/dataset/pretrained_word"))
    parser.add_argument("--model_path", type=Path, default=Path("/home/xiangling/EMSAssist/MLP_separated/model/pretrained_word_mlp.pt"))
    parser.add_argument("--evaluation_path", type=Path, default=Path("/home/xiangling/EMSAssist/MLP_separated/evaluation/pretrained_word_mlp_metrics.txt"))
    parser.add_argument("--report_title", default="Pretrained Word MLP Evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_seed(args.seed)
    metadata = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    load_x = lambda name: np.load(args.dataset_dir / f"{name}_x.npy", mmap_mode="r")
    load_y = lambda name: np.load(args.dataset_dir / f"{name}_y.npy")
    train_x, train_y = load_x("train"), load_y("train")
    val_x, val_y = load_x("val"), load_y("val")
    test_x, test_y = load_x("test"), load_y("test")
    configs = [
        {"hidden_sizes": [256], "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4},
        {"hidden_sizes": [512, 128], "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
        {"hidden_sizes": [512, 256], "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-5},
    ]
    trials, selected = [], None
    started = time.time()
    for candidate_id, config in enumerate(configs, 1):
        model, best = train_candidate(config, (train_x, train_y, val_x, val_y), metadata, args, candidate_id)
        trial = {"candidate": candidate_id, "config": config, "best_epoch": best["epoch"], "validation_metrics": best["metrics"]}
        trials.append(trial)
        if selected is None or best["metrics"]["f1"] > selected["best"]["metrics"]["f1"]:
            selected = {"candidate": candidate_id, "config": config, "model": model, "best": best}

    test_logits, targets = predict(selected["model"], test_x, test_y, args.batch_size, args.device)
    test_metrics = calculate_metrics(test_logits, targets)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": selected["model"].state_dict(), "input_size": metadata["input_size"],
                "num_classes": metadata["num_classes"], "label_classes": metadata["label_classes"],
                "config": selected["config"], "selected_epoch": selected["best"]["epoch"], "seed": args.seed}, args.model_path)
    report = {"selection_metric": "validation_macro_f1", "selected_candidate": selected["candidate"],
              "selected_config": selected["config"], "selected_epoch": selected["best"]["epoch"],
              "validation_metrics": selected["best"]["metrics"], "test_metrics": test_metrics,
              "all_validation_trials": trials, "elapsed_seconds": time.time() - started,
              "model_path": str(args.model_path)}
    args.evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [args.report_title, "=" * len(args.report_title), "",
             "Test metrics (Precision/Recall/F1 are macro-averaged)", "-----------------------------------------------------"]
    lines += [f"{name}: {value:.6f}" for name, value in test_metrics.items()]
    lines += ["", "Full JSON report", "----------------", json.dumps(report, indent=2)]
    args.evaluation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved model: {args.model_path}\nSaved evaluation: {args.evaluation_path}")


if __name__ == "__main__":
    main()
