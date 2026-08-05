import argparse
import os

import numpy as np

from pretrained_common import (
  CODE_COLUMN_INDEXES,
  CODE_COLUMN_NAMES,
  DEFAULT_INPUT_FILE,
  DEFAULT_NEMSIS_DIR,
  DEFAULT_OUTPUT_DIR,
  DEFAULT_WORD_MODEL_NAME,
  code_list,
  count_rows,
  description_for_codes,
  encode_texts,
  iter_rows,
  load_global_dict,
  load_model,
  model_hidden_size,
  save_metadata,
  write_lines,
)


def build_field_texts(columns, global_d):
  texts = []
  for column_idx in CODE_COLUMN_INDEXES:
    codes = code_list(columns[column_idx])
    text = description_for_codes(codes, global_d)
    texts.append(text if text else "[UNK]")
  return texts


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
  embeddings = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=output_shape)

  pcr_keys = []
  protocol_labels = []
  pending_texts = []
  pending_positions = []
  row_idx = 0

  for _header, columns in iter_rows(args.input_file):
    pcr_keys.append(columns[0].strip())
    protocol_labels.append(columns[5].strip())
    field_texts = build_field_texts(columns, global_d)

    for segment_idx, text in enumerate(field_texts):
      pending_texts.append(text)
      pending_positions.append((row_idx, segment_idx))

    if len(pending_texts) >= args.batch_size:
      encoded = encode_texts(
        pending_texts, tokenizer, model, torch, args.device, args.max_length, normalize=True
      )
      for embedding, (target_row, segment_idx) in zip(encoded, pending_positions):
        start = segment_idx * hidden_size
        embeddings[target_row, start:start + hidden_size] = embedding
      pending_texts = []
      pending_positions = []

    row_idx += 1
    if args.log_every and row_idx % args.log_every == 0:
      print("processed %s / %s rows" % (row_idx, row_count))

  if pending_texts:
    encoded = encode_texts(
      pending_texts, tokenizer, model, torch, args.device, args.max_length, normalize=True
    )
    for embedding, (target_row, segment_idx) in zip(encoded, pending_positions):
      start = segment_idx * hidden_size
      embeddings[target_row, start:start + hidden_size] = embedding

  embeddings.flush()
  write_lines(os.path.join(args.output_dir, args.prefix + "_pcr_keys.txt"), pcr_keys)
  write_lines(os.path.join(args.output_dir, args.prefix + "_protocol_labels.txt"), protocol_labels)
  save_metadata(
    args.output_dir,
    args.prefix,
    {
      "encoding": "pretrained_word_embedding",
      "input_file": args.input_file,
      "model_name": args.model_name,
      "shape": list(output_shape),
      "dtype": "float32",
      "pooling": "attention_mask_mean_pooling_with_l2_normalization",
      "segments": list(CODE_COLUMN_NAMES),
      "segment_embedding_size": hidden_size,
      "output_file": output_path,
      "text_source": "ICD-10 descriptions from nemsis_processor.get_dict global_d",
    },
  )
  print("wrote pretrained word embeddings:", output_path)
  print("shape:", output_shape)


def parse_args():
  parser = argparse.ArgumentParser(description="Create pretrained text embeddings for four NEMSIS symptom/impression fields.")
  parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
  parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--prefix", type=str, default="word_pretrained")
  parser.add_argument("--nemsis_dir", type=str, default=DEFAULT_NEMSIS_DIR)
  parser.add_argument("--nemsis_year", type=str, default="")
  parser.add_argument("--model_name", type=str, default=DEFAULT_WORD_MODEL_NAME)
  parser.add_argument("--model_cache_dir", type=str, default=None)
  parser.add_argument("--local_files_only", action="store_true")
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--max_length", type=int, default=64)
  parser.add_argument("--log_every", type=int, default=10000)
  return parser.parse_args()


def main():
  args = parse_args()
  build_embeddings(args)


if __name__ == "__main__":
  main()
