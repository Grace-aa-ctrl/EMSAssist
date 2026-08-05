"""Create reproducible stratified 70/10/20 splits for dense word embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def read_lines(path: Path) -> np.ndarray:
    return np.asarray(path.read_text(encoding="utf-8").splitlines())


def write_lines(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def copy_rows(source: np.ndarray, indices: np.ndarray, destination: Path, chunk_size: int) -> None:
    """Copy selected rows in bounded-memory chunks into a standard .npy file."""
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(len(indices), source.shape[1])
    )
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        output[start:stop] = source[indices[start:stop]]
    output.flush()
    del output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding_dir", type=Path, default=Path("/home/xiangling/EMSAssist/MLP_merged/embedding/embedding_pretrained/word_pretrained_output"))
    parser.add_argument("--embedding_prefix", default="word_pretrained",
                        help="Common filename prefix for embeddings, labels, and PCR keys")
    parser.add_argument("--output_dir", type=Path, default=Path("/home/xiangling/EMSAssist/MLP_merged/dataset/pretrained_word"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy_chunk_size", type=int, default=4096)
    args = parser.parse_args()

    source_file = args.embedding_dir / f"{args.embedding_prefix}_embeddings.npy"
    x = np.load(source_file, mmap_mode="r")
    labels = read_lines(args.embedding_dir / f"{args.embedding_prefix}_protocol_labels.txt")
    keys = read_lines(args.embedding_dir / f"{args.embedding_prefix}_pcr_keys.txt")
    if x.ndim != 2 or not (len(x) == len(labels) == len(keys)):
        raise ValueError(f"Row mismatch: embeddings={x.shape}, labels={len(labels)}, keys={len(keys)}")
    if len(np.unique(keys)) != len(keys):
        raise ValueError("PCR keys are not unique; a grouped split is needed to prevent leakage")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int64)
    indices = np.arange(len(y))
    train_idx, held_idx = train_test_split(indices, test_size=0.30, random_state=args.seed, stratify=y)
    val_idx, test_idx = train_test_split(
        held_idx, test_size=2 / 3, random_state=args.seed, stratify=y[held_idx]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}
    for name, idx in splits.items():
        print(f"Writing {name}: {len(idx):,} rows")
        copy_rows(x, idx, args.output_dir / f"{name}_x.npy", args.copy_chunk_size)
        np.save(args.output_dir / f"{name}_y.npy", y[idx])
        np.save(args.output_dir / f"{name}_source_indices.npy", idx)
        write_lines(args.output_dir / f"{name}_pcr_keys.txt", keys[idx])

    metadata = {
        "source_embedding": str(source_file),
        "seed": args.seed,
        "split_method": "stratified_random_split",
        "ratios": {"train": 0.70, "val": 0.10, "test": 0.20},
        "sizes": {name: int(len(idx)) for name, idx in splits.items()},
        "input_size": int(x.shape[1]),
        "num_classes": int(len(encoder.classes_)),
        "label_classes": encoder.classes_.tolist(),
        "class_counts": {
            name: np.bincount(y[idx], minlength=len(encoder.classes_)).tolist()
            for name, idx in splits.items()
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["sizes"], indent=2))


if __name__ == "__main__":
    main()
