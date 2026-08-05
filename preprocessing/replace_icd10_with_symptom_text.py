import argparse
import os
from collections import Counter

from nemsis_processor import (
  NEMSIS_Processor,
  add_sym_ref_name,
  pri_imp_ref_name,
  pri_sym_ref_name,
  sec_imp_ref_name,
)


DEFAULT_INPUT_FILE = "/home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt"
DEFAULT_OUTPUT_FILE = "/home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label_symptom_text.txt"
CODE_COLUMN_INDEXES = (1, 2, 3, 4)


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


def replace_code_field(field, global_d, missing_codes):
  codes = field.strip().split()
  descriptions = []

  for code in codes:
    description = global_d.get(code)
    if description is None:
      missing_codes[code] += 1
      descriptions.append(code)
    else:
      descriptions.append(description)

  return " ; ".join(descriptions)


def replace_icd10_codes(input_file, output_file, global_d):
  missing_codes = Counter()
  row_count = 0

  with open(input_file, "r", encoding="utf-8", errors="replace") as r_f, open(output_file, "w", encoding="utf-8") as w_f:
    for row_idx, line in enumerate(r_f):
      line = line.rstrip("\n")
      columns = line.split("~|~")

      if row_idx == 0:
        for column_idx in CODE_COLUMN_INDEXES:
          if column_idx < len(columns):
            columns[column_idx] = columns[column_idx].replace("Code", "Description")
        w_f.write("~|~".join(columns) + "\n")
        continue

      for column_idx in CODE_COLUMN_INDEXES:
        if column_idx < len(columns):
          columns[column_idx] = replace_code_field(columns[column_idx], global_d, missing_codes)

      w_f.write("~|~".join(columns) + "\n")
      row_count += 1

  return row_count, missing_codes


def parse_args():
  parser = argparse.ArgumentParser(description="Replace NEMSIS ICD-10 codes with symptom/impression descriptions.")
  parser.add_argument("--nemsis_dir", type=str, default="/data_8TB_2/xiangling_workspace/NEMSIS_ASCII_25")
  parser.add_argument("--nemsis_year", type=str, default="")
  parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE)
  parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
  return parser.parse_args()


def main():
  args = parse_args()
  global_d = load_global_dict(args.nemsis_dir, args.nemsis_year)
  row_count, missing_codes = replace_icd10_codes(args.input_file, args.output_file, global_d)

  print("loaded %s ICD-10 descriptions" % len(global_d))
  print("wrote %s rows to %s" % (row_count, args.output_file))
  if missing_codes:
    print("left %s unique missing ICD-10 codes unchanged" % len(missing_codes))
    for code, count in missing_codes.most_common(20):
      print("%s\t%s" % (code, count))


if __name__ == "__main__":
  main()
