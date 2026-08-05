import argparse
import json
import os

import numpy as np
from scipy import sparse


DEFAULT_INPUT_FILE = "/home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt"
DEFAULT_OUTPUT_DIR = "/home/xiangling/EMSAssist/MLP_seperated/embedding/encoding_one-hot"
CODE_COLUMN_INDEXES = (1, 2, 3, 4)
CODE_COLUMN_NAMES = (
  "PrimarySymptomCode",
  "PrimaryImpressionCode",
  "AdditionalSymptomCode",
  "SecondaryImpressionCode",
)


def iter_rows(input_file):
  with open(input_file, "r", encoding="utf-8", errors="replace") as r_f:
    header = r_f.readline().rstrip("\n").split("~|~")
    for line in r_f:
      columns = line.rstrip("\n").split("~|~")
      if len(columns) < 6:
        continue
      yield header, columns


def build_code_vocab(input_file):
  codes = set()
  for _, columns in iter_rows(input_file):
    for column_idx in CODE_COLUMN_INDEXES:
      codes.update(code for code in columns[column_idx].strip().split() if code)
  return {code: idx for idx, code in enumerate(sorted(codes))}


def write_lines(file_path, values):
  with open(file_path, "w", encoding="utf-8") as w_f:
    for value in values:
      w_f.write(str(value) + "\n")


def build_embeddings(input_file, output_dir, prefix):
  os.makedirs(output_dir, exist_ok=True)
  code_vocab = build_code_vocab(input_file)
  vocab_size = len(code_vocab)

  row_indexes = []
  column_indexes = []
  data = []
  pcr_keys = []
  protocol_labels = []

  for row_idx, (_, columns) in enumerate(iter_rows(input_file)):
    pcr_keys.append(columns[0].strip())
    protocol_labels.append(columns[5].strip())

    for segment_idx, column_idx in enumerate(CODE_COLUMN_INDEXES):
      seen_codes = set(code for code in columns[column_idx].strip().split() if code)
      for code in seen_codes:
        code_idx = code_vocab.get(code)
        if code_idx is None:
          continue
        row_indexes.append(row_idx)
        column_indexes.append(segment_idx * vocab_size + code_idx)
        data.append(1)

  embedding = sparse.csr_matrix(
    (np.asarray(data, dtype=np.float32), (row_indexes, column_indexes)),
    shape=(len(pcr_keys), len(CODE_COLUMN_INDEXES) * vocab_size),
  )

  sparse.save_npz(os.path.join(output_dir, prefix + "_embeddings.npz"), embedding)
  with open(os.path.join(output_dir, prefix + "_vocab.json"), "w", encoding="utf-8") as w_f:
    json.dump(
      {
        "encoding": "icd10_code_bag",
        "shape": list(embedding.shape),
        "segments": list(CODE_COLUMN_NAMES),
        "segment_vocab_size": vocab_size,
        "vocab": code_vocab,
      },
      w_f,
      indent=2,
      sort_keys=True,
    )
  write_lines(os.path.join(output_dir, prefix + "_pcr_keys.txt"), pcr_keys)
  write_lines(os.path.join(output_dir, prefix + "_protocol_labels.txt"), protocol_labels)

  print("wrote %s rows x %s columns to %s" % (embedding.shape[0], embedding.shape[1], output_dir))
  print("ICD-10 vocabulary size: %s" % vocab_size)


def parse_args():
  parser = argparse.ArgumentParser(description="Create concatenated ICD-10 code-bag embeddings for four NEMSIS symptom columns.")
  parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
  parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--prefix", type=str, default="icd10_code_bag")
  return parser.parse_args()


def main():
  args = parse_args()
  build_embeddings(args.input_file, args.output_dir, args.prefix)


if __name__ == "__main__":
  main()
