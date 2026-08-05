import argparse
import os

import numpy as np

from pretrained_common import (
  CODE_COLUMN_INDEXES,
  CODE_COLUMN_NAMES,
  DEFAULT_INPUT_FILE,
  DEFAULT_MODEL_NAME,
  DEFAULT_NEMSIS_DIR,
  DEFAULT_OUTPUT_DIR,
  code_list,
  count_rows,
  encode_texts,
  iter_rows,
  load_global_dict,
  load_model,
  model_hidden_size,
  save_metadata,
  write_lines,
)


def collect_dataset_codes(input_file):
  codes = set()
  for _header, columns in iter_rows(input_file):
    for column_idx in CODE_COLUMN_INDEXES:
      codes.update(code_list(columns[column_idx]))
  return sorted(codes)


def code_text(code, global_d):
  description = global_d.get(code)
  if description:
    return "ICD-10 code %s: %s" % (code, description)
  return "ICD-10 code %s" % code


def build_code_embedding_table(codes, global_d, tokenizer, model, torch, device, max_length, batch_size):
  hidden_size = model_hidden_size(model)
  code_to_index = {code: idx for idx, code in enumerate(codes)}
  table = np.zeros((len(codes), hidden_size), dtype=np.float32)

  for start_idx in range(0, len(codes), batch_size):
    batch_codes = codes[start_idx:start_idx + batch_size]
    batch_texts = [code_text(code, global_d) for code in batch_codes]
    table[start_idx:start_idx + len(batch_codes)] = encode_texts(
      batch_texts,
      tokenizer,
      model,
      torch,
      device,
      max_length,
    )

  return code_to_index, table


def mean_code_embedding(codes, code_to_index, code_embedding_table, hidden_size):
  indexes = [code_to_index[code] for code in codes if code in code_to_index]
  if not indexes:
    return np.zeros(hidden_size, dtype=np.float32)
  return code_embedding_table[indexes].mean(axis=0)


def build_embeddings(args):
  os.makedirs(args.output_dir, exist_ok=True)
  global_d = load_global_dict(args.nemsis_dir, args.nemsis_year)
  torch, tokenizer, model = load_model(
    args.model_name,
    args.device,
    cache_dir=args.model_cache_dir,
    local_files_only=args.local_files_only,
  )
  hidden_size = model_hidden_size(model)
  row_count = count_rows(args.input_file)
  output_shape = (row_count, len(CODE_COLUMN_INDEXES) * hidden_size)
  output_path = os.path.join(args.output_dir, args.prefix + "_embeddings.npy")

  codes = collect_dataset_codes(args.input_file)
  code_to_index, code_embedding_table = build_code_embedding_table(
    codes,
    global_d,
    tokenizer,
    model,
    torch,
    args.device,
    args.max_length,
    args.batch_size,
  )
  np.save(os.path.join(args.output_dir, args.prefix + "_code_embedding_table.npy"), code_embedding_table)

  embeddings = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=output_shape)
  pcr_keys = []
  protocol_labels = []

  for row_idx, (_header, columns) in enumerate(iter_rows(args.input_file)):
    pcr_keys.append(columns[0].strip())
    protocol_labels.append(columns[5].strip())

    for segment_idx, column_idx in enumerate(CODE_COLUMN_INDEXES):
      codes_in_field = code_list(columns[column_idx])
      embedding = mean_code_embedding(codes_in_field, code_to_index, code_embedding_table, hidden_size)
      start = segment_idx * hidden_size
      embeddings[row_idx, start:start + hidden_size] = embedding

    if args.log_every and (row_idx + 1) % args.log_every == 0:
      print("processed %s / %s rows" % (row_idx + 1, row_count))

  embeddings.flush()
  write_lines(os.path.join(args.output_dir, args.prefix + "_pcr_keys.txt"), pcr_keys)
  write_lines(os.path.join(args.output_dir, args.prefix + "_protocol_labels.txt"), protocol_labels)
  save_metadata(
    args.output_dir,
    args.prefix,
    {
      "encoding": "pretrained_icd10_embedding",
      "input_file": args.input_file,
      "model_name": args.model_name,
      "shape": list(output_shape),
      "dtype": "float32",
      "pooling": "attention_mask_mean_pooling",
      "field_pooling": "mean_of_icd10_code_embeddings",
      "segments": list(CODE_COLUMN_NAMES),
      "segment_embedding_size": hidden_size,
      "code_count": len(codes),
      "code_embedding_table": args.prefix + "_code_embedding_table.npy",
      "output_file": output_path,
      "code_text_format": "ICD-10 code {code}: {description}",
    },
  )
  save_metadata(
    args.output_dir,
    args.prefix + "_code_vocab",
    {
      "vocab": code_to_index,
    },
  )
  print("wrote pretrained ICD-10 embeddings:", output_path)
  print("shape:", output_shape)
  print("ICD-10 code count:", len(codes))


def parse_args():
  parser = argparse.ArgumentParser(description="Create pretrained ICD-10 code embeddings for four NEMSIS symptom/impression fields.")
  parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
  parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--prefix", type=str, default="icd10_pretrained")
  parser.add_argument("--nemsis_dir", type=str, default=DEFAULT_NEMSIS_DIR)
  parser.add_argument("--nemsis_year", type=str, default="")
  parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
  parser.add_argument("--model_cache_dir", type=str, default=None)
  parser.add_argument("--local_files_only", action="store_true")
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--max_length", type=int, default=64)
  parser.add_argument("--log_every", type=int, default=10000)
  return parser.parse_args()


def main():
  args = parse_args()
  build_embeddings(args)


if __name__ == "__main__":
  main()
