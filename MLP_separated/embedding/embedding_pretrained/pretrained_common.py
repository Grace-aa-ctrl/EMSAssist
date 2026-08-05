import json
import os
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

from nemsis_processor import (  # noqa: E402
  NEMSIS_Processor,
  add_sym_ref_name,
  pri_imp_ref_name,
  pri_sym_ref_name,
  sec_imp_ref_name,
)


DEFAULT_INPUT_FILE = "/home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt"
DEFAULT_OUTPUT_DIR = "/home/xiangling/EMSAssist/MLP_seperated/embedding/embedding_pretrained"
DEFAULT_NEMSIS_DIR = "/data_8TB_2/xiangling_workspace/NEMSIS_ASCII_25"
DEFAULT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_WORD_MODEL_NAME = "FremyCompany/BioLORD-2023"
CODE_COLUMN_INDEXES = (1, 2, 3, 4)
CODE_COLUMN_NAMES = (
  "PrimarySymptomCode",
  "PrimaryImpressionCode",
  "AdditionalSymptomCode",
  "SecondaryImpressionCode",
)


def require_transformers():
  try:
    import torch
    from transformers import AutoModel, AutoTokenizer
  except ImportError as exc:
    raise ImportError(
      "This script requires torch and transformers. Install them first, for example: "
      "pip install torch transformers"
    ) from exc
  return torch, AutoTokenizer, AutoModel


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


def count_rows(input_file):
  count = 0
  for _header, _columns in iter_rows(input_file):
    count += 1
  return count


def code_list(field):
  return [code for code in field.strip().split() if code]


def description_for_codes(codes, global_d):
  descriptions = []
  for code in codes:
    description = global_d.get(code)
    if description:
      descriptions.append(description)
  return " ; ".join(descriptions)


def write_lines(file_path, values):
  with open(file_path, "w", encoding="utf-8") as w_f:
    for value in values:
      w_f.write(str(value) + "\n")


def save_metadata(output_dir, prefix, metadata):
  with open(os.path.join(output_dir, prefix + "_metadata.json"), "w", encoding="utf-8") as w_f:
    json.dump(metadata, w_f, indent=2, sort_keys=True)


def load_model(model_name, device, cache_dir=None, local_files_only=False):
  torch, AutoTokenizer, AutoModel = require_transformers()
  tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    local_files_only=local_files_only,
  )
  model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    local_files_only=local_files_only,
  )
  model.to(device)
  model.eval()
  return torch, tokenizer, model


def mean_pool(last_hidden_state, attention_mask):
  mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
  summed = (last_hidden_state * mask).sum(dim=1)
  counts = mask.sum(dim=1).clamp(min=1e-9)
  return summed / counts


def encode_texts(texts, tokenizer, model, torch, device, max_length, normalize=False):
  encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
  )
  encoded = {key: value.to(device) for key, value in encoded.items()}

  with torch.no_grad():
    outputs = model(**encoded)
    pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    if normalize:
      pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
  return pooled.detach().cpu().numpy().astype(np.float32)


def model_hidden_size(model):
  hidden_size = getattr(model.config, "hidden_size", None)
  if hidden_size is None:
    hidden_size = getattr(model.config, "dim", None)
  if hidden_size is None:
    raise ValueError("Cannot determine model hidden size from model.config")
  return int(hidden_size)
