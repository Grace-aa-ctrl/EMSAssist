#!/usr/bin/env python3
"""Encode four ICD-10 fields per PCR as a bag-of-codes sparse matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy import sparse


DEFAULT_INPUT = Path(
    "/home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt"
)
DELIMITER = "~|~"
CODE_COLUMNS = ("PrimarySymptomCode", "PrimaryImpressionCode", "AdditionalSymptomCode", "SecondaryImpressionCode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将每行的四个症状字段编码为 ICD-10 bag-of-codes multi-hot 稀疏向量。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 txt 文件")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("icd10_bag_output"), help="输出目录"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100_000, help="构造稀疏矩阵时每批样本数"
    )
    return parser.parse_args()


def iter_records(path: Path) -> Iterator[tuple[str, tuple[str, ...], str]]:
    """Yield (PcrKey, unique ICD-10 codes, ProtocolIds)."""
    with path.open("r", encoding="utf-8") as file:
        header_line = file.readline()
        if not header_line:
            raise ValueError(f"输入文件为空: {path}")
        header = header_line.rstrip("\r\n").split(DELIMITER)
        required = ("PcrKey", *CODE_COLUMNS, "ProtocolIds")
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"输入文件缺少列: {missing}")

        pcr_index = header.index("PcrKey")
        code_indices = tuple(header.index(name) for name in CODE_COLUMNS)
        label_index = header.index("ProtocolIds")

        for line_number, line in enumerate(file, start=2):
            fields = line.rstrip("\r\n").split(DELIMITER)
            if len(fields) != len(header):
                raise ValueError(
                    f"第 {line_number} 行字段数为 {len(fields)}，应为 {len(header)}"
                )
            # set implements bag-of-codes presence: repeated codes remain 1, not a count.
            codes = tuple(
                sorted(
                    {
                        code
                        for index in code_indices
                        for code in fields[index].split()
                        if code
                    }
                )
            )
            yield fields[pcr_index], codes, fields[label_index]


def build_vocabulary(path: Path) -> list[str]:
    vocabulary: set[str] = set()
    for _, codes, _ in iter_records(path):
        vocabulary.update(codes)
    return sorted(vocabulary)


def encode(path: Path, vocabulary: list[str], chunk_size: int):
    code_to_index = {code: index for index, code in enumerate(vocabulary)}
    chunks: list[sparse.csr_matrix] = []
    pcr_keys: list[str] = []
    labels: list[str] = []
    row_indices: list[int] = []
    column_indices: list[int] = []
    rows_in_chunk = 0

    def finish_chunk() -> None:
        nonlocal row_indices, column_indices, rows_in_chunk
        if rows_in_chunk == 0:
            return
        data = np.ones(len(row_indices), dtype=np.uint8)
        chunks.append(
            sparse.csr_matrix(
                (data, (row_indices, column_indices)),
                shape=(rows_in_chunk, len(vocabulary)),
                dtype=np.uint8,
            )
        )
        row_indices, column_indices, rows_in_chunk = [], [], 0

    for pcr_key, codes, label in iter_records(path):
        pcr_keys.append(pcr_key)
        labels.append(label)
        row_indices.extend([rows_in_chunk] * len(codes))
        column_indices.extend(code_to_index[code] for code in codes)
        rows_in_chunk += 1
        if rows_in_chunk == chunk_size:
            finish_chunk()
    finish_chunk()

    matrix = sparse.vstack(chunks, format="csr") if chunks else sparse.csr_matrix((0, len(vocabulary)), dtype=np.uint8)
    return matrix, pcr_keys, labels


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size 必须为正整数")
    if not args.input.is_file():
        raise FileNotFoundError(f"找不到输入文件: {args.input}")

    vocabulary = build_vocabulary(args.input)
    matrix, pcr_keys, labels = encode(args.input, vocabulary, args.chunk_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output_dir / "icd10_bag.npz", matrix, compressed=True)
    (args.output_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.save(args.output_dir / "pcr_keys.npy", np.asarray(pcr_keys, dtype=str))
    np.save(args.output_dir / "protocol_ids.npy", np.asarray(labels, dtype=str))
    metadata = {
        "input": str(args.input.resolve()),
        "shape": list(matrix.shape),
        "nonzero": int(matrix.nnz),
        "dtype": str(matrix.dtype),
        "code_columns": list(CODE_COLUMNS),
        "encoding": "binary multi-hot bag of ICD-10 codes",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
