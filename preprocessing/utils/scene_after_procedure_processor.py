import argparse
from datetime import datetime
import os
from collections import defaultdict
import utils.file_utils as util
# from utils.nemsis_processor import NEMSIS_Processor as NP
from utils.procedure_processor import ProcedureProcessor as PP
# from . import file_utils as util
# from .nemsis_processor import NEMSIS_Processor as NP
import numpy
from sklearn.model_selection import train_test_split


fact_pcr_history_file_name = "FACTPCRALCOHOLDRUGUSEINDICATOR.txt"
# cached_pcr2si_protocol_med_vital_history_file_name = "nemsis_pcr2si_protocol_med_vital_history.txt"                             
cached_pcr2si_protocol_med_vital_procedure_scene_file_name = "nemsis_pcr2si_protocol_med_vital_procedure_scene.txt"
cached_pcr2si_protocol_med_vital_procedure_noscene_file_name = "nemsis_pcr2si_protocol_med_vital_procedure_noscene.txt"

class SceneAfterProcedureProcessor(object):

  def __init__(self, args):

    self.current_dir = os.path.dirname(os.path.realpath(__file__))
    self.nemsis_dir = args.nemsis_dir
    self.nemsis_year = args.nemsis_year
    self.data_dir = os.path.join(self.current_dir, "..",  args.data_folder)
    self.cache_dir = os.path.join(self.data_dir, args.cache_folder)
    self.pp = PP(args)

    self.prearrival_model_data_dir = args.prearrival_model_data_dir
    self.prearrival_cache_dir = os.path.join(self.prearrival_model_data_dir, args.data_folder, args.cache_folder, self.nemsis_year)

    # self.alcohol_drug_history_dict = {
    #   	'3117001' : 1,
    #     '3117003' :	2,
    #     '3117005'	: 3,
    #     '3117007'	: 4,
    #     '3117009'	: 5,
    #     '3117013'	: 6,
    # }

    self.scene_dict = {
      	'3117001' : 1,
        '3117003' :	2,
    }

    self.nemsis_alcohol_drug_history_file_path = os.path.join(self.nemsis_dir, self.nemsis_year, fact_pcr_history_file_name)
    self.cached_pcr2si_protocol_med_vital_procedure_scene_file_path = os.path.join(self.prearrival_cache_dir, cached_pcr2si_protocol_med_vital_procedure_scene_file_name)
    self.cached_pcr2si_protocol_med_vital_procedure_noscene_file_path = os.path.join(self.prearrival_cache_dir, cached_pcr2si_protocol_med_vital_procedure_noscene_file_name)

  def get_pcr2si_protocol_med_vital_procedure_scene(self):
    """
      get_pcr2si_protocol_med_vital_procedure_scene is to append the eHistory.17 - Alcohol/Drug Use Indicators to the end of si_protocol_med_vitals_procedure.
    """

    if os.path.isfile(self.cached_pcr2si_protocol_med_vital_procedure_scene_file_path) and os.path.isfile(self.cached_pcr2si_protocol_med_vital_procedure_noscene_file_path):
      print("directly read pcr2si_protocol_med_vital_procedure_scene from %s and pcr2si_protocol_med_vital_procedure_noscene from %s, respectively" % (self.cached_pcr2si_protocol_med_vital_procedure_scene_file_path, self.cached_pcr2si_protocol_med_vital_procedure_noscene_file_path))

      # make sure we get two files with disjoint pcr keys
      pcr2si_protocol_med_vitals_procedure_scene = util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_procedure_scene_file_path, line_length=57, is_value_a_set=False, discard_header=True)
      pcr2si_protocol_med_vitals_procedure_noscene = util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_procedure_noscene_file_path, line_length=56, is_value_a_set=False, discard_header=True)
      for pcr_k in pcr2si_protocol_med_vitals_procedure_scene.keys():
        assert pcr_k not in pcr2si_protocol_med_vitals_procedure_noscene
      for pcr_k in pcr2si_protocol_med_vitals_procedure_noscene.keys():
        assert pcr_k not in pcr2si_protocol_med_vitals_procedure_scene
      print("there are %s pcr keys in scene file and %s pcr keys in noscene file, there are no overlapping pcr_k" % (len(pcr2si_protocol_med_vitals_procedure_scene), len(pcr2si_protocol_med_vitals_procedure_noscene)))

      return pcr2si_protocol_med_vitals_procedure_scene, pcr2si_protocol_med_vitals_procedure_noscene

    pcr2si_protocol_med_vitals_procedure = self.pp.get_pcr2si_protocol_med_vitals_procedure()
    print("[scene_after_procedure_processor] get pcr2si_protocol_med_vitals_procedure of size %s" % len(pcr2si_protocol_med_vitals_procedure))

    pcr2scene = defaultdict(set)            # there are multiple history events for each pcr
    max_set_size = 0
    with open(self.nemsis_alcohol_drug_history_file_path, "r") as r_f:
      for row, line in enumerate(r_f):
        if row == 0:
          continue
        pcr_k, scene = line.strip().split('~|~')
        pcr_k = pcr_k.strip()
        scene = scene.strip()

        # only keep the history events that belong to scene
        if (pcr_k.strip() in pcr2si_protocol_med_vitals_procedure) and (scene.strip() in self.scene_dict):
          scene = str(self.scene_dict[scene.strip()])
          pcr2scene[pcr_k].add(scene)
          max_set_size = max(max_set_size, len(pcr2scene[pcr_k]))
    
    print("max_set_size is %s" % max_set_size)
    print("get pcr2scene of size %s" % len(pcr2scene))
        
    noscene_lines = 0
    scene_lines = 0
    with open(self.pp.cached_pcr2si_protocol_med_vital_procedure_file_path, "r") as r_f, open(self.cached_pcr2si_protocol_med_vital_procedure_noscene_file_path, "w") as w_noscene_f, open(self.cached_pcr2si_protocol_med_vital_procedure_scene_file_path, "w") as w_scene_f:
      for row, line in enumerate(r_f):

        # write the header
        if row == 0:
          w_noscene_f.write(line.strip() + '\n')
          w_scene_f.write(line.strip() + '~|~' + 'scene' + '\n')
          continue

        pcr_k = line.strip().split('~|~')[0].strip()
        if pcr_k in pcr2scene:
          scene_str = ' '.join(list(pcr2scene[pcr_k]))
          scene_lines += 1
          w_scene_f.write(line.strip() + '~|~' + scene_str + '\n')
        else:
          noscene_lines += 1
          w_noscene_f.write(line.strip() + '\n')

    print("scene lines is %s" % scene_lines)
    print("noscene lines is %s" % noscene_lines)
  
    # make sure we get two files with disjoint pcr keys
    pcr2si_protocol_med_vitals_procedure_scene = util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_procedure_scene_file_path, line_length=57, is_value_a_set=False, discard_header=True)
    pcr2si_protocol_med_vitals_procedure_noscene = util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_procedure_noscene_file_path, line_length=56, is_value_a_set=False, discard_header=True)
    for pcr_k in pcr2si_protocol_med_vitals_procedure_scene.keys():
      assert pcr_k not in pcr2si_protocol_med_vitals_procedure_noscene
    for pcr_k in pcr2si_protocol_med_vitals_procedure_noscene.keys():
      assert pcr_k not in pcr2si_protocol_med_vitals_procedure_scene
    print("there are %s pcr keys in scene file and %s pcr keys in noscene file, there are no overlapping pcr_k" % (len(pcr2si_protocol_med_vitals_procedure_scene), len(pcr2si_protocol_med_vitals_procedure_noscene)))

    return pcr2si_protocol_med_vitals_procedure_scene, pcr2si_protocol_med_vitals_procedure_noscene

  def do_split(self, file_path_to_split, train_file_name, val_file_name, test_file_name):

    lines = util.readFile(file_path_to_split)
    header = lines[0]
    lines = lines[1:]
    train_lines, val_test_lines = train_test_split(lines, test_size=0.4, random_state=util.global_seed)
    val_lines, test_lines = train_test_split(val_test_lines, test_size=0.5, random_state=util.global_seed)

    util.writeListFile(os.path.join(self.prearrival_cache_dir, train_file_name), [header] + train_lines)
    print("write %s lines to %s" % (len(train_lines), os.path.join(self.prearrival_cache_dir, train_file_name)))

    util.writeListFile(os.path.join(self.prearrival_cache_dir, val_file_name), [header] + val_lines)
    print("write %s lines to %s" % (len(val_lines), os.path.join(self.prearrival_cache_dir, val_file_name)))

    util.writeListFile(os.path.join(self.prearrival_cache_dir, test_file_name), [header] + test_lines)
    print("write %s lines to %s" % (len(test_lines), os.path.join(self.prearrival_cache_dir, test_file_name)))

  def write_train_val_test_files(self):
    """
      write_train_val_test_files is to split the pcr2si_protocol_med_vital_history into train, val, and test files.
    """
    self.do_split(self.cached_pcr2si_protocol_med_vital_procedure_scene_file_path, "train_file_scene.txt", "val_file_scene.txt", "test_file_scene.txt")
    self.do_split(self.cached_pcr2si_protocol_med_vital_procedure_noscene_file_path, "train_file_noscene.txt", "val_file_noscene.txt", "test_file_noscene.txt")

if __name__ == "__main__":
    
  time_s = datetime.now()
  print("Start time: ", time_s)

  parser = argparse.ArgumentParser(description = "control the functions for NEMSIS Medication processing")
  parser.add_argument("--nemsis_dir", action='store', type=str, default = "/slot1/NEMSIS_Databases")
  parser.add_argument("--nemsis_year", action='store', type=str, default = "2023")
  parser.add_argument("--data_folder", action='store', type=str, default = "data")
  parser.add_argument("--prearrival_model_data_dir", action='store', type=str, default = "/slot1/prearrival_models_datasets")
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
  spp = SceneAfterProcedureProcessor(args)

  spp.get_pcr2si_protocol_med_vital_procedure_scene()
  spp.write_train_val_test_files()

  time_e = datetime.now()
  print("End time: ", time_e)
  time_t = time_e - time_s
  print("This run takes %s" % time_t)
