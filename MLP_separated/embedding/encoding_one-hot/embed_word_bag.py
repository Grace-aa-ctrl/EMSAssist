import argparse
import json
import os
import re
import sys

import numpy as np
from scipy import sparse


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from EMSAssist.preprocessing.nemsis_processor import (  # noqa: E402
  NEMSIS_Processor,
  add_sym_ref_name,
  pri_imp_ref_name,
  pri_sym_ref_name,
  sec_imp_ref_name,
)


DEFAULT_INPUT_FILE = "/home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt"
DEFAULT_OUTPUT_DIR = "/home/xiangling/EMSAssist/MLP_seperated/embedding/encoding_one-hot"
DEFAULT_NEMSIS_DIR = "/data_8TB_2/xiangling_workspace/NEMSIS_ASCII_25"
CODE_COLUMN_INDEXES = (1, 2, 3, 4)
CODE_COLUMN_NAMES = (
  "PrimarySymptomCode",
  "PrimaryImpressionCode",
  "AdditionalSymptomCode",
  "SecondaryImpressionCode",
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


def nemsis_file_path(nemsis_dir, nemsis_year, file_name):
  candidates = []
  if nemsis_year:
    candidates.append(os.path.join(nemsis_dir, nemsis_year, file_name))
  candidates.append(os.path.join(nemsis_dir, file_name))

  for file_path in candidates:
    if os.path.isfile(file_path):
      return file_path

  raise FileNotFoundError("Cannot find %s. Tried: %s" % (file_name, ", ".join(candidates)))


def load_global_dict(nemsis_dir, nemsis_year):
  processor = object.__new__(NEMSIS_Processor)
  processor.ref_files = [
    nemsis_file_path(nemsis_dir, nemsis_year, pri_sym_ref_name),
    nemsis_file_path(nemsis_dir, nemsis_year, pri_imp_ref_name),
    nemsis_file_path(nemsis_dir, nemsis_year, add_sym_ref_name),
    nemsis_file_path(nemsis_dir, nemsis_year, sec_imp_ref_name),
  ]
  _, global_d, _, _ = processor.get_dict()
  return global_d


def iter_rows(input_file):
  with open(input_file, "r", encoding="utf-8", errors="replace") as r_f:
    header = r_f.readline().rstrip("\n").split("~|~")
    for line in r_f:
      columns = line.rstrip("\n").split("~|~")
      if len(columns) < 6:
        continue
      yield header, columns


def tokenize(text):
  return TOKEN_RE.findall(text.lower())


def tokens_for_code_field(field, global_d, missing_codes):
  tokens = set()
  for code in field.strip().split():
    description = global_d.get(code)
    if description is None:
      missing_codes.add(code)
      continue
    tokens.update(tokenize(description))
  return tokens


def build_word_vocab(input_file, global_d):
  vocab = set()
  missing_codes = set()

  for _, columns in iter_rows(input_file):
    for column_idx in CODE_COLUMN_INDEXES:
      vocab.update(tokens_for_code_field(columns[column_idx], global_d, missing_codes))

  return {word: idx for idx, word in enumerate(sorted(vocab))}, missing_codes


def write_lines(file_path, values):
  with open(file_path, "w", encoding="utf-8") as w_f:
    for value in values:
      w_f.write(str(value) + "\n")


def build_embeddings(input_file, output_dir, prefix, global_d):
  os.makedirs(output_dir, exist_ok=True)
  word_vocab, missing_codes = build_word_vocab(input_file, global_d)
  vocab_size = len(word_vocab)

  row_indexes = []
  column_indexes = []
  data = []
  pcr_keys = []
  protocol_labels = []

  for row_idx, (_, columns) in enumerate(iter_rows(input_file)):
    pcr_keys.append(columns[0].strip())
    protocol_labels.append(columns[5].strip())

    for segment_idx, column_idx in enumerate(CODE_COLUMN_INDEXES):
      tokens = tokens_for_code_field(columns[column_idx], global_d, missing_codes)
      for token in tokens:
        token_idx = word_vocab.get(token)
        if token_idx is None:
          continue
        row_indexes.append(row_idx)
        column_indexes.append(segment_idx * vocab_size + token_idx)
        data.append(1)

  embedding = sparse.csr_matrix(
    (np.asarray(data, dtype=np.float32), (row_indexes, column_indexes)),
    shape=(len(pcr_keys), len(CODE_COLUMN_INDEXES) * vocab_size),
  )

  sparse.save_npz(os.path.join(output_dir, prefix + "_embeddings.npz"), embedding)
  with open(os.path.join(output_dir, prefix + "_vocab.json"), "w", encoding="utf-8") as w_f:
    json.dump(
      {
        "encoding": "word_bag",
        "shape": list(embedding.shape),
        "segments": list(CODE_COLUMN_NAMES),
        "segment_vocab_size": vocab_size,
        "vocab": word_vocab,
      },
      w_f,
      indent=2,
      sort_keys=True,
    )
  write_lines(os.path.join(output_dir, prefix + "_pcr_keys.txt"), pcr_keys)
  write_lines(os.path.join(output_dir, prefix + "_protocol_labels.txt"), protocol_labels)

  if missing_codes:
    write_lines(os.path.join(output_dir, prefix + "_missing_codes.txt"), sorted(missing_codes))

  print("wrote %s rows x %s columns to %s" % (embedding.shape[0], embedding.shape[1], output_dir))
  print("word vocabulary size: %s" % vocab_size)
  print("ICD-10 descriptions loaded: %s" % len(global_d))
  if missing_codes:
    print("left %s missing ICD-10 codes out of the word bag" % len(missing_codes))


def parse_args():
  parser = argparse.ArgumentParser(description="Create concatenated word-bag embeddings for four NEMSIS symptom columns.")
  parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
  parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--prefix", type=str, default="word_bag")
  parser.add_argument("--nemsis_dir", type=str, default=DEFAULT_NEMSIS_DIR)
  parser.add_argument("--nemsis_year", type=str, default="")
  return parser.parse_args()


def main():
  args = parse_args()
  global_d = load_global_dict(args.nemsis_dir, args.nemsis_year)
  build_embeddings(args.input_file, args.output_dir, args.prefix, global_d)


if __name__ == "__main__":
  main()
