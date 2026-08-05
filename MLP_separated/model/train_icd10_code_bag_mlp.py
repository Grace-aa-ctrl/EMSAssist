import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch import nn


class MLP(nn.Module):
  def __init__(self, input_size, hidden_sizes, num_classes, dropout):
    super().__init__()
    layers = []
    current = input_size
    for hidden in hidden_sizes:
      layers.extend([nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)])
      current = hidden
    layers.append(nn.Linear(current, num_classes))
    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def batches(x, y, batch_size, shuffle, rng):
  indexes = np.arange(x.shape[0])
  if shuffle:
    rng.shuffle(indexes)
  for start in range(0, len(indexes), batch_size):
    if shuffle:
      # Sorted row access is much faster for CSR; batch order remains randomized.
      idx = np.sort(indexes[start:start + batch_size])
      yield x[idx].tocsr(), y[idx]
    else:
      end = min(start + batch_size, x.shape[0])
      yield x[start:end], y[start:end]


def scipy_csr_to_torch(x, device):
  return torch.sparse_csr_tensor(
    torch.from_numpy(x.indptr),
    torch.from_numpy(x.indices),
    torch.from_numpy(x.data),
    size=x.shape,
    device=device,
    check_invariants=False,
  )


def predict(model, x, y, batch_size, device):
  model.eval()
  logits_parts = []
  targets = []
  with torch.inference_mode():
    for xb, yb in batches(x, y, batch_size, False, np.random.default_rng(0)):
      xb = scipy_csr_to_torch(xb, device)
      logits_parts.append(model(xb).cpu().numpy())
      targets.append(yb)
  return np.concatenate(logits_parts), np.concatenate(targets)


def metrics_from_logits(logits, y):
  pred = logits.argmax(axis=1)
  result = {
    "top_1": float(np.mean(pred == y)),
    "accuracy": float(accuracy_score(y, pred)),
  }
  order = np.argsort(logits, axis=1)[:, -5:]
  result["top_3"] = float(np.mean(np.any(order[:, -3:] == y[:, None], axis=1)))
  result["top_5"] = float(np.mean(np.any(order == y[:, None], axis=1)))
  for average in ("macro", "weighted", "micro"):
    p, r, f1, _ = precision_recall_fscore_support(
      y, pred, average=average, zero_division=0
    )
    result[f"precision_{average}"] = float(p)
    result[f"recall_{average}"] = float(r)
    result[f"f1_{average}"] = float(f1)
  return result


def train_candidate(config, train_x, train_y, val_x, val_y, input_size, num_classes, args, candidate_id):
  model = MLP(input_size, config["hidden_sizes"], num_classes, config["dropout"]).to(args.device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
  criterion = nn.CrossEntropyLoss()
  scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
  rng = np.random.default_rng(args.seed + candidate_id)
  best = None
  stale = 0

  for epoch in range(1, args.max_epochs + 1):
    model.train()
    total_loss = 0.0
    seen = 0
    for xb, yb in batches(train_x, train_y, args.batch_size, True, rng):
      xb = scipy_csr_to_torch(xb, args.device)
      yb = torch.from_numpy(yb).to(args.device, non_blocking=True)
      optimizer.zero_grad(set_to_none=True)
      with torch.amp.autocast("cuda", enabled=args.device.startswith("cuda")):
        logits = model(xb)
        loss = criterion(logits, yb)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      total_loss += loss.item() * len(yb)
      seen += len(yb)

    val_logits, val_targets = predict(model, val_x, val_y, args.batch_size, args.device)
    val_metrics = metrics_from_logits(val_logits, val_targets)
    print(f"candidate={candidate_id} epoch={epoch} loss={total_loss/seen:.6f} "
          f"val_macro_f1={val_metrics['f1_macro']:.6f} val_accuracy={val_metrics['accuracy']:.6f}")
    if best is None or val_metrics["f1_macro"] > best["metrics"]["f1_macro"] + 1e-6:
      best = {
        "epoch": epoch,
        "metrics": val_metrics,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
      }
      stale = 0
    else:
      stale += 1
      if stale >= args.patience:
        break
  model.load_state_dict(best["state"])
  return model, best


def main():
  parser = argparse.ArgumentParser(description="Tune and evaluate an MLP on ICD-10 code-bag embeddings.")
  parser.add_argument("--dataset_dir", default="/home/xiangling/EMSAssist/MLP_separated/dataset/icd10_code_bag")
  parser.add_argument("--model_path", default="/home/xiangling/EMSAssist/MLP_separated/model/icd10_code_bag_mlp.pt")
  parser.add_argument("--evaluation_path", default="/home/xiangling/EMSAssist/MLP_separateds/evaluation/icd10_code_bag_mlp_metrics.txt")
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--batch_size", type=int, default=1024)
  parser.add_argument("--max_epochs", type=int, default=15)
  parser.add_argument("--patience", type=int, default=3)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but unavailable; use --device cpu or install compatible PyTorch")
  set_seed(args.seed)
  root = Path(args.dataset_dir)
  metadata = json.loads((root / "metadata.json").read_text())
  train_x = sparse.load_npz(root / "train_x.npz").tocsr().astype(np.float32)
  val_x = sparse.load_npz(root / "val_x.npz").tocsr().astype(np.float32)
  test_x = sparse.load_npz(root / "test_x.npz").tocsr().astype(np.float32)
  train_y = np.load(root / "train_y.npy")
  val_y = np.load(root / "val_y.npy")
  test_y = np.load(root / "test_y.npy")

  configs = [
    {"hidden_sizes": [256], "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4},
    {"hidden_sizes": [512, 128], "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4},
    {"hidden_sizes": [512, 256], "dropout": 0.2, "lr": 5e-4, "weight_decay": 1e-5},
  ]
  trials = []
  best_trial = None
  started = time.time()
  for i, config in enumerate(configs, 1):
    model, best = train_candidate(
      config, train_x, train_y, val_x, val_y,
      metadata["input_size"], metadata["num_classes"], args, i
    )
    trial = {"candidate": i, "config": config, "best_epoch": best["epoch"], "validation": best["metrics"]}
    trials.append(trial)
    if best_trial is None or best["metrics"]["f1_macro"] > best_trial["best"]["metrics"]["f1_macro"]:
      best_trial = {"model": model, "config": config, "best": best, "candidate": i}

  test_logits, test_targets = predict(best_trial["model"], test_x, test_y, args.batch_size, args.device)
  test_metrics = metrics_from_logits(test_logits, test_targets)
  model_path = Path(args.model_path)
  model_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
    "model_state_dict": best_trial["model"].state_dict(),
    "input_size": metadata["input_size"],
    "num_classes": metadata["num_classes"],
    "label_classes": metadata["label_classes"],
    "config": best_trial["config"],
    "selected_epoch": best_trial["best"]["epoch"],
    "seed": args.seed,
  }, model_path)

  report = {
    "selection_metric": "validation_macro_f1",
    "selected_candidate": best_trial["candidate"],
    "selected_config": best_trial["config"],
    "selected_epoch": best_trial["best"]["epoch"],
    "validation_metrics": best_trial["best"]["metrics"],
    "test_metrics": test_metrics,
    "all_validation_trials": trials,
    "elapsed_seconds": time.time() - started,
    "model_path": str(model_path),
  }
  evaluation_path = Path(args.evaluation_path)
  evaluation_path.parent.mkdir(parents=True, exist_ok=True)
  lines = [
    "ICD-10 Code Bag MLP Evaluation",
    "=======================",
    f"Selected candidate: {report['selected_candidate']}",
    f"Selected config: {json.dumps(report['selected_config'])}",
    f"Selected epoch: {report['selected_epoch']}",
    "",
    "Test metrics",
    "------------",
  ]
  lines.extend(f"{name}: {value:.6f}" for name, value in test_metrics.items())
  lines.extend(["", "Full JSON report", "----------------", json.dumps(report, indent=2)])
  evaluation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(json.dumps(test_metrics, indent=2))
  print(f"saved model: {model_path}")
  print(f"saved evaluation: {evaluation_path}")


if __name__ == "__main__":
  main()
