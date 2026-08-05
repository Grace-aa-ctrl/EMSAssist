import argparse
from datetime import datetime
import os
from collections import defaultdict
import utils.file_utils as fu
from utils.nemsis_processor import NEMSIS_Processor as NP
from utils.vitals_processor import Vital_Processor as VP
# from . import file_utils as util
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split


fact_pcr_procedure_file_name = "FACTPCRPROCEDURE.txt"
fact_pcr_procedure_reduced_file_name = "FACTPCRPROCEDURE_reduced.txt"

cached_pcr2si_protocol_med_vital_procedure_file_name = "nemsis_pcr2si_protocol_med_vital_procedure.txt"
cached_pcr2procedure_file_name = "nemsis_pcr2procedure.txt"
# cached_pcr2si_protocol_med_vital_noscene_file_name = "nemsis_pcr2si_protocol_med_vital_noscene.txt"

# cached_pcr2med_file_name = "nemsis_pcr2med.txt"
# cached_pcr2si_protocol_med_file_name = "nemsis_pcr2si_protocol_med.txt"                             # 4 si - med line
# cached_si2med_med_set_file_name = "nemsis_si2med_med_set_dedicated.txt"   # med code list
# dedicated_med_file = "tamu_medication.csv"
# # nemsis_med_code2unit_file_name = "nemsis_med_type2unit_code.txt"               # 
# nemsis_med_quant_unit_file = "nemsis_med_quant_units.csv"

class ProcedureProcessor(object):

  def __init__(self, args):

    self.current_dir = os.path.dirname(os.path.realpath(__file__))
    self.nemsis_dir = args.nemsis_dir
    self.nemsis_year = args.nemsis_year
    self.data_dir = os.path.join(self.current_dir, "..",  args.data_folder)
    self.cache_dir = os.path.join(self.data_dir, args.cache_folder)
    self.vp = VP(args)

    self.cached_pcr2si_protocol_med_vital_procedure_file_path = os.path.join(self.cache_dir, self.nemsis_year, cached_pcr2si_protocol_med_vital_procedure_file_name)
    self.cached_pcr2procedure_file_path = os.path.join(self.cache_dir, self.nemsis_year, cached_pcr2procedure_file_name)
    self.nemsis_procedure_file_path = os.path.join(self.nemsis_dir, self.nemsis_year, fact_pcr_procedure_file_name)
    self.procedure_subtype2supertype_file_path = os.path.join(self.data_dir, "procedure_subtype2supertype.csv")

    self.PcrKey_pcr_k_index = 1
    self.procedure_index = 2
    self.nemsis_procedure_reduced_file_path = os.path.join(self.nemsis_dir, self.nemsis_year, fact_pcr_procedure_reduced_file_name)
    self.subtype2supertype_id, self.supertype_id2desc = self.extract_subtype2supertype_id()

  def extract_subtype2supertype_id(self):
    """
      extract_subtype2supertype_id is to extract the procedure subtype and supertype from the procedure code
    """
    subtype2supertype_id = {}
    supertype_id2desc = {}
    # read the procedure code file
    df = pd.read_csv(self.procedure_subtype2supertype_file_path, delimiter=',', keep_default_na=False)
    # print("df size is %s" % df.shape)
    # print("df columns are %s" % df.columns)

    # create a dictionary to store the procedure subtype and supertype
    for i, row in df.iterrows():
      subtype_code = str((row['SNOMED'])).strip()
      supertype_desc = str(row['PARENT']).strip()
      supertype_id = row['PARENT_ID']
      if subtype_code in subtype2supertype_id:
        print("subtype_code %s is in subtype2supertype_id, index %s, row %s" % (subtype_code, i, row))
      assert subtype_code not in subtype2supertype_id
      subtype2supertype_id[subtype_code] = supertype_id
      supertype_id2desc[supertype_id] = supertype_desc

    print("subtype2supertype_id is %s" % subtype2supertype_id)
    print("supertype_id2desc is %s" % supertype_id2desc)

    return subtype2supertype_id, supertype_id2desc
  
  def reduce_nemsis_procedure_file(self):
    """
      This function is used to reduce the nemsis procedure file
    """

    pcr2si_protocol_med_vitals = self.vp.get_si_protocol_med_vitals()
    print("get pcr2si_protocol_med_vitals of size %s" % len(pcr2si_protocol_med_vitals))
    vitals_pcr_set = set(pcr2si_protocol_med_vitals.keys())
    print("vitals_pcr_set size: ", len(vitals_pcr_set))

    procedure_vitals_pcr_set = set()
    count_reduced_lines = 0
    with open(self.nemsis_procedure_file_path, "r") as r_f, open(self.nemsis_procedure_reduced_file_path, "w") as w_f:
      for nemsis_procedure_row, nemsis_procedure_line in enumerate(r_f):
        procedure_pcr = nemsis_procedure_line.split('~|~')[self.PcrKey_pcr_k_index].strip()
        if (nemsis_procedure_row == 0) or (procedure_pcr in vitals_pcr_set):
          count_reduced_lines += 1
          w_f.write(nemsis_procedure_line)
          if procedure_pcr in vitals_pcr_set:
            procedure_vitals_pcr_set.add(procedure_pcr)
      
    print("vitals_med_pcr_set size: ", len(procedure_vitals_pcr_set))
    print("write %s lines to %s" % (count_reduced_lines, self.nemsis_procedure_reduced_file_path))

  def get_pcr2procedure(self):
 
    """
      get_pcr2procedure is to get the pcr2procedure supertype id dictionary from the nemsis procedure file
    """
    self.pcr2procedure = defaultdict(set)            # there are multiple procedure supertype id for each pcr
    self.pcr2procedure_len = defaultdict(int)        # check the value length for each pcr key

    if not os.path.isfile(self.nemsis_procedure_reduced_file_path):
      print(f"{self.nemsis_procedure_reduced_file_path} does not exist, we need to create it")
      self.reduce_nemsis_procedure_file()
    else:
      print(f"{self.nemsis_procedure_reduced_file_path} exist, we directly return pcr2procedure")

    non_dedicated_procedure_lines = 0

    df = pd.read_csv(self.nemsis_procedure_reduced_file_path, delimiter='~\|~', engine='python', keep_default_na=False)
    df.columns = df.columns.str.strip("'")
    for i, row in df.iterrows():
      # print the keys of the row
      # print("row keys are %s" % row.keys())
      pcr_k = str(row["PcrKey"]).strip()
      procedure_subtype = str(row["eProcedures_03"]).strip()
      if procedure_subtype not in self.subtype2supertype_id:
        # print("procedure_subtype %s is not in subtype2supertype_id, index %s, row %s" % (procedure_subtype, i, row))
        non_dedicated_procedure_lines += 1
        continue
      assert procedure_subtype in self.subtype2supertype_id
      procedure_supertype_id = self.subtype2supertype_id[procedure_subtype]
      # if i == 0:
      #   header = row
      #   continue
      if pcr_k not in self.pcr2procedure:
        self.pcr2procedure[pcr_k] = set()
      self.pcr2procedure[pcr_k].add(procedure_supertype_id)
      self.pcr2procedure_len[pcr_k] = len(self.pcr2procedure[pcr_k])

    print("pcr2procedure size is %s" % len(self.pcr2procedure))
    print("pcr2procedure_len value set is %s" % set(self.pcr2procedure_len.values()))
    print("non_dedicated_procedure_lines is %s" % non_dedicated_procedure_lines)

    count_procedure_label_len_dict = defaultdict(int)
    for pcr_k, procedure_set in self.pcr2procedure.items():
      procedure_label_len = len(procedure_set)
      count_procedure_label_len_dict[procedure_label_len] += 1
    print("count_procedure_label_len_dict is %s" % count_procedure_label_len_dict)

    return self.pcr2procedure

  def get_pcr2si_protocol_med_vitals_procedure(self):
    """
      get_pcr2si_protocol_med_vital_procedure is to append the procedure information to the end of si_protocol_med_vitals.
    """

    if os.path.isfile(self.cached_pcr2si_protocol_med_vital_procedure_file_path):
      print(f"{self.cached_pcr2si_protocol_med_vital_procedure_file_path} exist, we directly return pcr2si_protocol_med_vitals")
      return fu.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_procedure_file_path, line_length=56, is_value_a_set=False, discard_header=True)
    else:
      print(f"{self.cached_pcr2si_protocol_med_vital_procedure_file_path} does not exist, we need to create it")
      # return util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_procedure_file_path, line_length=56, is_value_a_set=False, discard_header=True)

    # self.nemsis_procedure_file_path = os.path.join(self.nemsis_dir, self.nemsis_year, fact_pcr_procedure_file_name)
    # if not os.path.isfile(self.nemsis_procedure_reduced_file_path):
    #   print(f"{self.nemsis_procedure_reduced_file_path} does not exist, we need to create it")
      # self.reduce_nemsis_procedures_file()
      # assert os.path.isfile(self.cached_pcr2si_protocol_med_vital_noscene_file_path)
      # return util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_scene_file_path, line_length=56, is_value_a_set=False, discard_header=True)

    _ = self.get_pcr2procedure()  # returns pcr2procedure

    # pcr2si_protocol_med_vitals = self.vp.get_si_protocol_med_vitals()
    # print("get pcr2si_protocol_med_vitals of size %s" % len(pcr2si_protocol_med_vitals))

    # pcr2scene = defaultdict(set)            # there are multiple history events for each pcr
    # max_set_size = 0
    # with open(self.nemsis_alcohol_drug_history_file_path, "r") as r_f:
    #   for row, line in enumerate(r_f):
    #     if row == 0:
    #       continue
    #     pcr_k, history = line.strip().split('~|~')
    #     pcr_k = pcr_k.strip()
    #     history = history.strip()

    #     # only keep the history events that belong to scene
    #     if (pcr_k.strip() in pcr2si_protocol_med_vitals) and (history.strip() in self.scene_dict):
    #       scene = str(self.scene_dict[history.strip()])
    #       pcr2scene[pcr_k].add(scene)
    #       max_set_size = max(max_set_size, len(pcr2scene[pcr_k]))
    
    # print("max_set_size is %s" % max_set_size)
    # print("get pcr2scene of size %s" % len(pcr2scene))
        
    procedure_lines = 0
    # scene_lines = 0
    with open(self.vp.cached_pcr2si_protocol_med_vital_file_path, "r") as r_f, open(self.cached_pcr2si_protocol_med_vital_procedure_file_path, "w") as w_f:
      for row, line in enumerate(r_f):

        # write the header
        if row == 0:
          w_f.write(line.strip() + '~|~' + 'procedure' + '\n')
          continue

        pcr_k = line.strip().split('~|~')[0].strip()
        # assert pcr_k in self.pcr2procedure
        if pcr_k in self.pcr2procedure:
          # procedure_str = ' '.join(list(self.pcr2procedure[pcr_k]))
          procedure_str = ' '.join(str(item) for item in list(self.pcr2procedure[pcr_k]))
          procedure_lines += 1
          w_f.write(line.strip() + '~|~' + procedure_str + '\n')
        # else:
        #   print("pcr_k %s is not in pcr2procedure" % pcr_k)

    print("write %s procedure_lines to %s" % (procedure_lines, self.cached_pcr2si_protocol_med_vital_procedure_file_path))
    # print("nosttcene lines is %s" % noscene_lines)
    return fu.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_procedure_file_path, line_length=56, is_value_a_set=False, discard_header=True)
  
    # # make sure we get two files with disjoint pcr keys
    # pcr2si_protocol_med_vitals_scene = util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_scene_file_path, line_length=56, is_value_a_set=False, discard_header=True)
    # pcr2si_protocol_med_vitals_noscene = util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_noscene_file_path, line_length=55, is_value_a_set=False, discard_header=True)
    # for pcr_k in pcr2si_protocol_med_vitals_scene.keys():
    #   assert pcr_k not in pcr2si_protocol_med_vitals_noscene
    # for pcr_k in pcr2si_protocol_med_vitals_noscene.keys():
    #   assert pcr_k not in pcr2si_protocol_med_vitals_scene
    # print("there are %s pcr keys in scene file and %s pcr keys in noscene file, there are no overlapping pcr_k" % (len(pcr2si_protocol_med_vitals_scene), len(pcr2si_protocol_med_vitals_noscene)))

    # return pcr2si_protocol_med_vitals_scene, pcr2si_protocol_med_vitals_scene

  # def do_split(self, file_path_to_split, train_file_name, val_file_name, test_file_name):

  #   # lines = util.readFile(self.cached_pcr2si_protocol_med_vital_scene_file_path)
  #   lines = util.readFile(file_path_to_split)
  #   header = lines[0]
  #   lines = lines[1:]
  #   train_lines, val_test_lines = train_test_split(lines, test_size=0.4, random_state=util.global_seed)
  #   val_lines, test_lines = train_test_split(val_test_lines, test_size=0.5, random_state=util.global_seed)

  #   util.writeListFile(os.path.join(self.cache_dir, self.nemsis_year, train_file_name), [header] + train_lines)
  #   print("write train_file.txt of size %s" % len(train_lines))
  #   util.writeListFile(os.path.join(self.cache_dir, self.nemsis_year, val_file_name), [header] + val_lines)
  #   print("write val_file.txt of size %s" % len(val_lines))
  #   util.writeListFile(os.path.join(self.cache_dir, self.nemsis_year, test_file_name), [header] + test_lines)
  #   print("write test_file.txt of size %s" % len(test_lines))

  # def write_train_val_test_files(self):
  #   """
  #     write_train_val_test_files is to split the pcr2si_protocol_med_vital_history into train, val, and test files.
  #   """
  #   self.do_split(self.cached_pcr2si_protocol_med_vital_scene_file_path, "train_file_scene.txt", "val_file_scene.txt", "test_file_scene.txt")
  #   self.do_split(self.cached_pcr2si_protocol_med_vital_noscene_file_path, "train_file_noscene.txt", "val_file_noscene.txt", "test_file_noscene.txt")

if __name__ == "__main__":
    
  time_s = datetime.now()
  print("Start time: ", time_s)

  parser = argparse.ArgumentParser(description = "control the functions for NEMSIS Medication processing")
  parser.add_argument("--nemsis_dir", action='store', type=str, default = "/slot1/NEMSIS_Databases")
  parser.add_argument("--nemsis_year", action='store', type=str, default = "2023")
  parser.add_argument("--data_folder", action='store', type=str, default = "data")
  # parser.add_argument("--nemsis_med_quant_unit_file", action='store', type=str, default = "nemsis_med_quant_units.csv")
  parser.add_argument("--cache_folder", action='store', type=str, default = "nemsis_cache_files")
  # parser.add_argument("--dedicated_med_file", action='store', type=str, default = "tamu_medication.csv")
  # parser.add_argument("--dedicated_protocol_file", action='store', type=str, default = None)

  parser.add_argument("--train_file", action='store', type=str, default = "train_file.txt")
  parser.add_argument("--val_file", action='store', type=str, default = "val_file.txt")
  parser.add_argument("--test_file", action='store', type=str, default = "test_file.txt")
  # parser.add_argument("--train_file_multi_label", action='store', type=str, default = "train_med_multi_label.txt")
  # parser.add_argument("--val_file_multi_label", action='store', type=str, default = "val_med_multi_label.txt")
  # parser.add_argument("--test_file_multi_label", action='store', type=str, default = "test_med_multi_label.txt")

  args = parser.parse_args()
  pp = ProcedureProcessor(args)
  get_pcr2si_protocol_med_vital_procedure = pp.get_pcr2si_protocol_med_vitals_procedure()

  # hp.get_pcr2si_protocol_med_vital_history()
  # hp.write_train_val_test_files()


  time_e = datetime.now()
  print("End time: ", time_e)
  time_t = time_e - time_s
  print("This run takes %s" % time_t)

