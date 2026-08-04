"""Fine-tune MobileBERT with an explicit MLP head on concatenated symptom text."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


DELIMITER = "~|~"
FIELD_NAMES = ["primary symptom", "primary impression", "additional symptom", "secondary impression"]


class TextDataset(Dataset):
    def __init__(self, path: Path, labels: np.ndarray):
        with path.open(encoding="utf-8") as handle:
            self.header = handle.readline().rstrip("\n").split(DELIMITER)
            self.rows = [line.rstrip("\n").split(DELIMITER) for line in handle]
        self.labels = labels
        if len(self.rows) != len(labels) or any(len(row) != 6 for row in self.rows):
            raise ValueError(f"Invalid dataset {path}: rows={len(self.rows)}, labels={len(labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.rows[index]
        text = " [SEP] ".join(f"{name}: {value}" for name, value in zip(FIELD_NAMES, row[1:5]))
        return text, int(self.labels[index])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class TextCollator:
    """Picklable tokenizer collator for Python 3.14 forkserver workers."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        texts, labels = zip(*samples)
        encoded = self.tokenizer(list(texts), padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded


def make_loader(dataset, tokenizer, args, shuffle):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
                      persistent_workers=args.num_workers > 0,
                      collate_fn=TextCollator(tokenizer, args.max_length))


class MobileBertWithMLP(nn.Module):
    """MobileBERT encoder followed by a trainable two-layer classification MLP."""

    def __init__(self, model_name, num_classes, local_files_only=True):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        self.mobilebert = AutoModel.from_pretrained(
            model_name, config=config, local_files_only=local_files_only)
        hidden_size = config.hidden_size
        dropout = getattr(config, "classifier_dropout", None)
        if dropout is None:
            dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.mobilebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = self.loss_function(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def evaluate(model, loader, device):
    model.eval(); logits_parts, target_parts = [], []
    with torch.inference_mode():
        for batch in loader:
            targets = batch.pop("labels")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                logits = model(**batch).logits
            logits_parts.append(logits.float().cpu().numpy()); target_parts.append(targets.numpy())
    return np.concatenate(logits_parts), np.concatenate(target_parts)


def metrics(logits, targets):
    predictions = logits.argmax(1); num_classes = logits.shape[1]
    result = {"top_1": float(np.mean(predictions == targets))}
    for k in (3, 5):
        top = np.argpartition(logits, -min(k, num_classes), axis=1)[:, -min(k, num_classes):]
        result[f"top_{k}"] = float(np.mean(np.any(top == targets[:, None], axis=1)))
    result["accuracy"] = float(accuracy_score(targets, predictions))
    p, r, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(num_classes), average="macro", zero_division=0)
    result.update(precision=float(p), recall=float(r), f1=float(f1))
    return result


def new_model(args, num_classes):
    return MobileBertWithMLP(
        args.model_name, num_classes, local_files_only=args.local_files_only)


def train_candidate(config, train_loader, val_loader, num_classes, args, candidate_id):
    set_seed(args.seed + candidate_id)
    model = new_model(args, num_classes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    best, stale = None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train(); total_loss = 0.0; seen = 0
        for batch in train_loader:
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.device.startswith("cuda")):
                output = model(**batch); loss = output.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * len(batch["labels"]); seen += len(batch["labels"])
        val_logits, val_targets = evaluate(model, val_loader, args.device)
        val_metrics = metrics(val_logits, val_targets)
        print(f"candidate={candidate_id} epoch={epoch} loss={total_loss/seen:.6f} "
              f"val_f1={val_metrics['f1']:.6f} val_accuracy={val_metrics['accuracy']:.6f}", flush=True)
        if best is None or val_metrics["f1"] > best["metrics"]["f1"] + 1e-6:
            best = {"epoch": epoch, "metrics": val_metrics,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience: break
    del model, optimizer; torch.cuda.empty_cache()
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, default=Path("/home/xiangling/EMSAssist/BERT_merged/dataset/symptom_text"))
    parser.add_argument("--model_name", default="google/mobilebert-uncased")
    parser.add_argument("--model_path", type=Path, default=Path("/home/xiangling/EMSAssist/BERT_merged/model/symptom_text_mobilebert_mlp.pt"))
    parser.add_argument("--evaluation_path", type=Path, default=Path("/home/xiangling/EMSAssist/BERT_merged/evaluation/symptom_text_mobilebert_mlp_metrics.txt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    set_seed(args.seed)
    metadata = json.loads((args.dataset_dir / "metadata.json").read_text())
    datasets = {name: TextDataset(args.dataset_dir / f"{name}.txt", np.load(args.dataset_dir / f"{name}_y.npy"))
                for name in ("train", "val", "test")}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    loaders = {name: make_loader(ds, tokenizer, args, name == "train") for name, ds in datasets.items()}
    configs = [{"lr": 2e-5, "weight_decay": 0.01}, {"lr": 3e-5, "weight_decay": 0.01},
               {"lr": 5e-5, "weight_decay": 0.01}]
    started = time.time(); trials = []; selected = None
    for candidate_id, config in enumerate(configs, 1):
        best = train_candidate(config, loaders["train"], loaders["val"], metadata["num_classes"], args, candidate_id)
        trial = {"candidate": candidate_id, "config": config, "best_epoch": best["epoch"],
                 "validation_metrics": best["metrics"]}; trials.append(trial)
        if selected is None or best["metrics"]["f1"] > selected["best"]["metrics"]["f1"]:
            selected = {"candidate": candidate_id, "config": config, "best": best}
    model = new_model(args, metadata["num_classes"]); model.load_state_dict(selected["best"]["state"]); model.to(args.device)
    test_logits, test_targets = evaluate(model, loaders["test"], args.device)
    test_metrics = metrics(test_logits, test_targets)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_name": args.model_name,
                "architecture": "MobileBertWithMLP", "num_classes": metadata["num_classes"],
                "label_classes": metadata["label_classes"], "config": selected["config"],
                "selected_epoch": selected["best"]["epoch"], "max_length": args.max_length,
                "seed": args.seed}, args.model_path)
    report = {"selection_metric": "validation_macro_f1", "selected_candidate": selected["candidate"],
              "selected_config": selected["config"], "selected_epoch": selected["best"]["epoch"],
              "validation_metrics": selected["best"]["metrics"], "test_metrics": test_metrics,
              "all_validation_trials": trials, "elapsed_seconds": time.time() - started,
              "model_path": str(args.model_path)}
    lines = ["Symptom Text MobileBERT + MLP Evaluation", "==========================================", "",
             "Test metrics (Precision/Recall/F1 are macro-averaged)", "-----------------------------------------------------"]
    lines += [f"{k}: {v:.6f}" for k, v in test_metrics.items()]
    lines += ["", "Full JSON report", "----------------", json.dumps(report, indent=2)]
    args.evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    args.evaluation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(test_metrics, indent=2)); print(f"Saved model: {args.model_path}\nSaved evaluation: {args.evaluation_path}")


if __name__ == "__main__":
    main()
