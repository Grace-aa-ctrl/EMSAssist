import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def read_lines(path):
  return Path(path).read_text(encoding="utf-8").splitlines()


def write_lines(path, values):
  Path(path).write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main():
  parser = argparse.ArgumentParser(description="Create stratified 70/10/20 ICD-10 code-bag train/validation/test splits.")
  parser.add_argument("--embedding_dir", default="/home/xiangling/EMSAssist/MLP_separated/embedding/encoding_one-hot")
  parser.add_argument("--output_dir", default="/home/xiangling/EMSAssist/MLP_separated/dataset/icd10_code_bag")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  source = Path(args.embedding_dir)
  output = Path(args.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  x = sparse.load_npz(source / "icd10_code_bag_embeddings.npz").tocsr().astype(np.float32)
  labels = np.asarray(read_lines(source / "icd10_code_bag_protocol_labels.txt"))
  keys = np.asarray(read_lines(source / "icd10_code_bag_pcr_keys.txt"))
  if not (x.shape[0] == len(labels) == len(keys)):
    raise ValueError(f"Row mismatch: X={x.shape[0]}, labels={len(labels)}, keys={len(keys)}")
  if len(np.unique(keys)) != len(keys):
    raise ValueError("PCR keys are not unique; grouped splitting is required to prevent leakage")

  encoder = LabelEncoder()
  y = encoder.fit_transform(labels).astype(np.int64)
  all_idx = np.arange(x.shape[0])
  train_idx, held_idx = train_test_split(
    all_idx, test_size=0.30, random_state=args.seed, stratify=y
  )
  val_idx, test_idx = train_test_split(
    held_idx, test_size=2.0 / 3.0, random_state=args.seed, stratify=y[held_idx]
  )

  splits = {"train": train_idx, "val": val_idx, "test": test_idx}
  for name, idx in splits.items():
    idx = np.sort(idx)
    sparse.save_npz(output / f"{name}_x.npz", x[idx], compressed=True)
    np.save(output / f"{name}_y.npy", y[idx])
    np.save(output / f"{name}_source_indices.npy", idx)
    write_lines(output / f"{name}_pcr_keys.txt", keys[idx])

  counts = {
    name: np.bincount(y[idx], minlength=len(encoder.classes_)).tolist()
    for name, idx in splits.items()
  }
  metadata = {
    "source_embedding": str(source / "icd10_code_bag_embeddings.npz"),
    "seed": args.seed,
    "split_method": "stratified_random_split",
    "ratios": {"train": 0.70, "val": 0.10, "test": 0.20},
    "sizes": {name: int(len(idx)) for name, idx in splits.items()},
    "input_size": int(x.shape[1]),
    "num_classes": int(len(encoder.classes_)),
    "label_classes": encoder.classes_.tolist(),
    "class_counts": counts,
  }
  (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
  print(json.dumps(metadata["sizes"], indent=2))


if __name__ == "__main__":
  main()
