"""Create reproducible stratified 70/10/20 symptom-text dataset splits."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DELIMITER = "~|~"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, default=Path(
        "/home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label_symptom_text.txt"))
    parser.add_argument("--output_dir", type=Path, default=Path(
        "/home/xiangling/EMSAssist/dataset/symptom_text"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.input_file.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        rows = [line.rstrip("\n") for line in handle]
    columns = header.split(DELIMITER)
    if columns != ["PcrKey", "PrimarySymptomDescription", "PrimaryImpressionDescription",
                   "AdditionalSymptomDescription", "SecondaryImpressionDescription", "ProtocolIds"]:
        raise ValueError(f"Unexpected columns: {columns}")
    parsed = [row.split(DELIMITER) for row in rows]
    bad = [i for i, parts in enumerate(parsed) if len(parts) != 6]
    if bad:
        raise ValueError(f"Malformed rows at zero-based data indexes: {bad[:10]}")
    keys = np.asarray([parts[0] for parts in parsed])
    labels = np.asarray([parts[5] for parts in parsed])
    if len(np.unique(keys)) != len(keys):
        raise ValueError("PCR keys are not unique; grouped splitting is required")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int64)
    all_idx = np.arange(len(rows))
    train_idx, held_idx = train_test_split(all_idx, test_size=0.30, random_state=args.seed, stratify=y)
    val_idx, test_idx = train_test_split(
        held_idx, test_size=2 / 3, random_state=args.seed, stratify=y[held_idx])
    splits = {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in splits.items():
        with (args.output_dir / f"{name}.txt").open("w", encoding="utf-8") as handle:
            handle.write(header + "\n")
            handle.writelines(rows[i] + "\n" for i in idx)
        np.save(args.output_dir / f"{name}_y.npy", y[idx])
        np.save(args.output_dir / f"{name}_source_indices.npy", idx)

    metadata = {
        "source_file": str(args.input_file), "delimiter": DELIMITER, "seed": args.seed,
        "split_method": "stratified_random_split",
        "ratios": {"train": 0.70, "val": 0.10, "test": 0.20},
        "sizes": {name: int(len(idx)) for name, idx in splits.items()},
        "num_classes": int(len(encoder.classes_)), "label_classes": encoder.classes_.tolist(),
        "text_columns": columns[1:5], "label_column": columns[5],
        "class_counts": {name: np.bincount(y[idx], minlength=len(encoder.classes_)).tolist()
                         for name, idx in splits.items()},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata["sizes"], indent=2))


if __name__ == "__main__":
    main()
