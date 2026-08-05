import argparse
from datetime import datetime
# from . import file_utils as util
# from .nemsis_processor import NEMSIS_Processor as NP
# from .medication_processor import Medication_Processor as MP
import utils.file_utils as util
from utils.nemsis_processor import NEMSIS_Processor as NP
from utils.medication_processor import Medication_Processor as MP
import os
from collections import defaultdict
import json

import numpy
# from sklearn.model_selection import train_test_split
import re

fact_pcr_vitals_file_name = "FACTPCRVITAL.txt"
fact_pcr_vitals_reduced_file_name = "FACTPCRVITAL_reduced.txt"
PcrVitalKey_index = 0                                     # PcrVitalKey == not used
PcrKey_pcr_k_index = 1                                    # PcrKey
eVitals_06_sbp_index = 2                                  # eVitals.06 - SBP (Systolic Blood Pressure)
eVitals_10_hr_index = 3                                   # eVitals.10 - Heart Rate
eVitals_12_pulse_oximetry_index = 4                       # eVitals.12 - Pulse Oximetry
eVitals_14_respiratory_rate_index = 5                     # eVitals.14 - Respiratory Rate
eVitals_16_etco2_index = 6                                # eVitals.16 - End Tidal Carbon Dioxide (ETCO2)
eVitals_18_blood_glucose_index = 7                        # eVitals.18 - Blood Glucose Level
eVitals_27_pain_scale_score_index = 8                     # eVitals.27 - Pain Scale Score == not used
eVitals_02_prior_to_EMS_index = 9                         # eVitals.02 - Obtained Prior to this Unit's EMS Care == not used
eVitals_04_ecg_type_index = 10                            # eVitals.04 - ECG Type  == not used
eVitals_08_blood_pressure_measurement_method_index = 11   # eVitals.08 - Method of Blood Pressure Measurement == not used
eVitals_26_avpu_index = 12                                # eVitals.26 - Level of Responsiveness (AVPU) == not used
eVitals_29_stroke_scale_score_index = 13                  # eVitals.29 - Stroke Scale Score   == not used
eVitals_30_stroke_scale_type_index = 14                   # eVitals.30 - Stroke Scale Type    == not used
eVitals_31_reperfusion_checklist_index = 15               # eVitals.31 - Reperfusion Checklist      == not used
eVitals_19_gcs_eye_index = 16                             # eVitals.19 - Glasgow Coma Score-Eye     == not used
eVitals_20_gcs_verbal_index = 17                          # eVitals.20 - Glasgow Coma Score-Verbal  == not used
eVitals_21_gcs_motor_index = 18                           # eVitals.21 - Glasgow Coma Score-Motor   == not used
eVitals_01_date_time_index = 19                           # eVitals.01 - Date/Time Vital Signals are measured

ecg_group_file_name = "PCRVITALECGGROUP.txt"
PcrVitalECGGroupKey_idx = 0                             # PcrVitalECGGroupKey
PcrVitalKey_idx = 1                                     # VitalKey
eVitals_03_ecg_rhythm_idx = 2                           # eVitals.03 - Cardiac rhythm / Electrocardiography (ECG)

cached_pcr2si_protocol_med_vital_file_name = "nemsis_pcr2si_protocol_med_vital.txt"                             
cached_vitals_normalization_params_file_name = "vitals_normalization_params.json"

class Vital_Processor(object):

  def __init__(self, args):

    self.current_dir = os.path.dirname(os.path.realpath(__file__))
    self.nemsis_dir = args.nemsis_dir
    self.nemsis_year = args.nemsis_year
    self.data_dir = os.path.join(self.current_dir, "..",  args.data_folder)
    self.cache_dir = os.path.join(self.data_dir, args.cache_folder)
    self.np = NP(self.nemsis_dir, self.data_dir, self.cache_dir, self.nemsis_year)
    self.mp = MP(args)

    self.PcrVitalKey_index = PcrVitalKey_index if int(self.nemsis_year) >= 2020 else PcrVitalKey_index + 1                                        # PcrVitalKey
    self.PcrKey_pcr_k_index = PcrKey_pcr_k_index if int(self.nemsis_year) >= 2020 else PcrKey_pcr_k_index + 1                                     # PcrKey
    self.eVitals_06_sbp_index = eVitals_06_sbp_index if int(self.nemsis_year) >= 2020 else eVitals_06_sbp_index + 1                                  # eVitals.06 - SBP (Systolic Blood Pressure)
    self.eVitals_10_hr_index = eVitals_10_hr_index if int(self.nemsis_year) >= 2020 else eVitals_10_hr_index + 1                                   # eVitals.10 - Heart Rate
    self.eVitals_12_pulse_oximetry_index = eVitals_12_pulse_oximetry_index if int(self.nemsis_year) >= 2020 else eVitals_12_pulse_oximetry_index + 1                       # eVitals.12 - Pulse Oximetry
    self.eVitals_14_respiratory_rate_index = eVitals_14_respiratory_rate_index if int(self.nemsis_year) >= 2020 else eVitals_14_respiratory_rate_index + 1                     # eVitals.14 - Respiratory Rate
    self.eVitals_16_etco2_index = eVitals_16_etco2_index if int(self.nemsis_year) >= 2020 else eVitals_16_etco2_index + 1                                # eVitals.16 - End Tidal Carbon Dioxide (ETCO2)
    self.eVitals_18_blood_glucose_index = eVitals_18_blood_glucose_index if int(self.nemsis_year) >= 2020 else eVitals_18_blood_glucose_index + 1                        # eVitals.18 - Blood Glucose Level
    self.eVitals_27_pain_scale_score_index = eVitals_27_pain_scale_score_index if int(self.nemsis_year) >= 2020 else eVitals_27_pain_scale_score_index + 1                     # eVitals.27 - Pain Scale Score == not used
    self.eVitals_02_prior_to_EMS_index = eVitals_02_prior_to_EMS_index if int(self.nemsis_year) >= 2020 else eVitals_02_prior_to_EMS_index + 1                         # eVitals.02 - Obtained Prior to this Unit's EMS Care == not used
    self.eVitals_04_ecg_type_index = eVitals_04_ecg_type_index if int(self.nemsis_year) >= 2020 else eVitals_04_ecg_type_index + 1                            # eVitals.04 - ECG Type  == not used
    self.eVitals_08_blood_pressure_measurement_method_index = eVitals_08_blood_pressure_measurement_method_index if int(self.nemsis_year) >= 2020 else eVitals_08_blood_pressure_measurement_method_index + 1   # eVitals.08 - Method of Blood Pressure Measurement == not used
    self.eVitals_26_avpu_index = eVitals_26_avpu_index if int(self.nemsis_year) >= 2020 else eVitals_26_avpu_index + 1                                # eVitals.26 - Level of Responsiveness (AVPU) == not used
    self.eVitals_29_stroke_scale_score_index = eVitals_29_stroke_scale_score_index if int(self.nemsis_year) >= 2020 else eVitals_29_stroke_scale_score_index + 1                  # eVitals.29 - Stroke Scale Score   == not used
    self.eVitals_30_stroke_scale_type_index = eVitals_30_stroke_scale_type_index if int(self.nemsis_year) >= 2020 else eVitals_30_stroke_scale_type_index + 1                   # eVitals.30 - Stroke Scale Type    == not used
    self.eVitals_31_reperfusion_checklist_index = eVitals_31_reperfusion_checklist_index if int(self.nemsis_year) >= 2020 else eVitals_31_reperfusion_checklist_index + 1               # eVitals.31 - Reperfusion Checklist      == not used
    self.eVitals_19_gcs_eye_index = eVitals_19_gcs_eye_index if int(self.nemsis_year) >= 2020 else eVitals_19_gcs_eye_index + 1                             # eVitals.19 - Glasgow Coma Score-Eye     == not used
    self.eVitals_20_gcs_verbal_index = eVitals_20_gcs_verbal_index if int(self.nemsis_year) >= 2020 else eVitals_20_gcs_verbal_index + 1                          # eVitals.20 - Glasgow Coma Score-Verbal  == not used
    self.eVitals_21_gcs_motor_index = eVitals_21_gcs_motor_index if int(self.nemsis_year) >= 2020 else eVitals_21_gcs_motor_index + 1                           # eVitals.21 - Glasgow Coma Score-Motor   == not used
    self.eVitals_01_date_time_index = eVitals_01_date_time_index if int(self.nemsis_year) >= 2020 else 0                           # eVitals.01 - Date/Time Vital Signals are measured

    # we only consider the vitals of our interests with clear format
    self.vitals_of_interest = [
      self.eVitals_06_sbp_index, 
      self.eVitals_10_hr_index, 
      self.eVitals_12_pulse_oximetry_index, 
      self.eVitals_14_respiratory_rate_index, 
      self.eVitals_16_etco2_index, 
      self.eVitals_18_blood_glucose_index, 
      # eVitals_29_stroke_scale_score_index, 
      # eVitals_30_stroke_scale_type_index, 
      # eVitals_31_reperfusion_checklist_index, 
      # eVitals_19_gcs_eye_index, 
      # eVitals_20_gcs_verbal_index, 
      # eVitals_21_gcs_motor_index
    ]

    self.cached_pcr2si_protocol_med_vital_file_path = os.path.join(self.cache_dir, self.nemsis_year, cached_pcr2si_protocol_med_vital_file_name)
    self.cached_vitals_normalization_params_file_path = os.path.join(self.cache_dir, self.nemsis_year, cached_vitals_normalization_params_file_name)

  # def read_cached_pcr2si_protocol_med_vitals(self, file_path):

  #   pcr2si_protocol_med_vitals = dict()
  #   with open(file_path, "r") as r_f:
  #     for row, line in enumerate(r_f):

  #       event = line.strip().split('~|~')
  #       assert len(event) == 46
  #       pcr_k = event[0].strip()
  #       si_protocol_med_vitals = ('~|~').join(event[1:])
  #       pcr2si_protocol_med_vitals[pcr_k] = si_protocol_med_vitals

  #   print("retrieved %s pcr2si_protocol_med_vitals from %s with %s lines of cached pcr2si_protocol_med, the key set size = %s, the value set size = %s" % 
  #         (len(pcr2si_protocol_med_vitals), file_path, row+1, len(set(pcr2si_protocol_med_vitals.keys())), len(set(pcr2si_protocol_med_vitals.values()))))
  #   return pcr2si_protocol_med_vitals

  def reduce_nemsis_vitals_file(self, nemsis_vitals_reduced_file_path):
    """
      This function is used to reduce the nemsis vitals file to only include the pcrs that exist in the si_protocol_med file
    """

    # in 2019, the date index is move to the first of the column
    # we want to first filter the pcrs that both exist for vitals and si_protocol_med

    nemsis_vitals_file_path = os.path.join(self.nemsis_dir, self.nemsis_year, fact_pcr_vitals_file_name)
    # nemsis_vitals_reduced_file_path = os.path.join(self.nemsis_dir, self.nemsis_year, fact_pcr_vitals_reduced_file_name)

    pcr_k_index = self.PcrKey_pcr_k_index
    # vitals_pcr_file_path = nemsis_vitals_file_path.split('.')[0] + str(pcr_k_index) + '.txt'
    # print("vitals_pcr_file_path: ", vitals_pcr_file_path)

    pcr2si_protocol_med = self.mp.get_pcr2si_protocols_meds()
    med_pcr_set = set(pcr2si_protocol_med.keys())
    print("med_pcr_set size: ", len(med_pcr_set))

    vitals_med_pcr_set = set()
    count_reduced_lines = 0
    with open(nemsis_vitals_file_path, "r") as r_f, open(nemsis_vitals_reduced_file_path, "w") as w_f:
      for nemsis_vitals_row, nemsis_vitals_line in enumerate(r_f):
        # if nemsis_vitals_row == 0:
        #   continue
        vitals_pcr = nemsis_vitals_line.split('~|~')[pcr_k_index].strip()
        if (nemsis_vitals_row == 0) or (vitals_pcr in med_pcr_set):
          count_reduced_lines += 1
          w_f.write(nemsis_vitals_line)
          vitals_med_pcr_set.add(vitals_pcr)
      
    print("vitals_med_pcr_set size: ", len(vitals_med_pcr_set))
    print("write %s lines to %s" % (count_reduced_lines, nemsis_vitals_reduced_file_path))


  def get_pcr2vitals(self):
    """
      This function is used to get the pcr2vitals dictionary
      1. first reduce the nemsis vitals file to only include the pcrs that exist in the si_protocol_med file
    """

    nemsis_vitals_reduced_file_path = os.path.join(self.nemsis_dir, self.nemsis_year, fact_pcr_vitals_reduced_file_name)
    if not os.path.isfile(nemsis_vitals_reduced_file_path):
      self.reduce_nemsis_vitals_file(nemsis_vitals_reduced_file_path)

    pcr_vitals_count = defaultdict(int)
    per_vital_dict = defaultdict(list)
    per_pcr_vitals_dict = defaultdict(lambda: defaultdict(list))
    valid_vitals_mean_dict = defaultdict(float)
    valid_vitals_std_dict = defaultdict(float)
    valid_vitals_min_dict = defaultdict(float)
    valid_vitals_max_dict = defaultdict(float)
    valid_vitals_max_abs_dict = defaultdict(float)
    valid_vitals_20_percentile_dict = defaultdict(float)    
    valid_vitals_80_percentile_dict = defaultdict(float)
    valid_vitals_2_percentile_dict = defaultdict(float)    
    valid_vitals_98_percentile_dict = defaultdict(float)

    ##### Pass 1: we do the first pass to get the min, max, mean and std of each vitals
    with open(nemsis_vitals_reduced_file_path, "r") as r_f:
      for row, line in enumerate(r_f):
        if row == 0:
          continue
        vitals_event = line.strip().split('~|~')
        assert len(vitals_event) == 20                # there are 20 columns in the nemsis vital file

        # ### pcr key ###
        pcr_k = vitals_event[self.PcrKey_pcr_k_index].strip()
        pcr_vitals_count[pcr_k] += 1
        # valid_line = False

        ### eVitals_01_date_time ###
        current_datetime = vitals_event[self.eVitals_01_date_time_index].strip()
        try:
          current_datetime_obj = datetime.strptime(current_datetime, "%d%b%Y:%H:%M:%S")       
          per_vital_dict[self.eVitals_01_date_time_index].append(current_datetime_obj)
          per_pcr_vitals_dict[pcr_k][self.eVitals_01_date_time_index].append(current_datetime_obj)
        except ValueError:
          # valid_line = False
          continue                                                                                # if the time info is not available, we will ignore this line
          # pass

        ### eVitals_06_sbp ###
        sbp = vitals_event[self.eVitals_06_sbp_index].strip()
        if sbp not in util.nemsis_value_blacklist():
          pattern = re.compile(r'^(?:[0-9]|[1-9][0-9]|[1-4][0-9]{2}|500)$')
          assert pattern.match(sbp)
          per_vital_dict[self.eVitals_06_sbp_index].append(float(sbp))
          per_pcr_vitals_dict[pcr_k][self.eVitals_06_sbp_index].append(float(sbp))
        else:
          per_pcr_vitals_dict[pcr_k][self.eVitals_06_sbp_index].append(None)
          # valid_line = True

        ### eVitals_10_hr ###
        hr = vitals_event[self.eVitals_10_hr_index].strip()
        if hr not in util.nemsis_value_blacklist():
          pattern = re.compile(r'^(?:[0-9]|[1-9][0-9]|[1-4][0-9]{2}|500)$')
          assert pattern.match(hr)
          per_vital_dict[self.eVitals_10_hr_index].append(float(hr))
          per_pcr_vitals_dict[pcr_k][self.eVitals_10_hr_index].append(float(hr))
        else:
          per_pcr_vitals_dict[pcr_k][self.eVitals_10_hr_index].append(None)
          # valid_line = True

        ### eVitals_12_pulse_oximetry ###
        pulse_oximetry = vitals_event[self.eVitals_12_pulse_oximetry_index].strip()
        if pulse_oximetry not in util.nemsis_value_blacklist():
          pattern = re.compile(r'^(?:[0-9]|[1-9][0-9]|100)$')
          assert pattern.match(pulse_oximetry)
          per_vital_dict[self.eVitals_12_pulse_oximetry_index].append(float(pulse_oximetry))
          per_pcr_vitals_dict[pcr_k][self.eVitals_12_pulse_oximetry_index].append(float(pulse_oximetry))
        else:
          per_pcr_vitals_dict[pcr_k][self.eVitals_12_pulse_oximetry_index].append(None)
          # valid_line = True

        ### eVitals_14_respiratory_rate ###
        respiratory_rate = vitals_event[self.eVitals_14_respiratory_rate_index].strip()
        if respiratory_rate not in util.nemsis_value_blacklist():
          pattern = re.compile(r'^(?:[0-9]|[1-9][0-9]|[1-2][0-9]{2}|300)$')
          assert pattern.match(respiratory_rate)
          per_vital_dict[self.eVitals_14_respiratory_rate_index].append(float(respiratory_rate))
          per_pcr_vitals_dict[pcr_k][self.eVitals_14_respiratory_rate_index].append(float(respiratory_rate))
        else:
          per_pcr_vitals_dict[pcr_k][self.eVitals_14_respiratory_rate_index].append(None)
          # valid_line = True

        ### eVitals_16_etco2 ###
        etco2 = vitals_event[self.eVitals_16_etco2_index].strip()
        if etco2 not in util.nemsis_value_blacklist():
          # pattern = re.compile(r'^(?:[0-9]|[1-9][0-9]|[1-6][0-9]{2}|7[0-5][0-9]|760)\.[0-9]$')
          pattern = re.compile(r'^(?:[0-9]|[1-9][0-9]|[1-6][0-9]{2}|7[0-5][0-9]|760)(?:\.[0-9])?$')
          # if not pattern.match(etco2):
          #   print(etco2)
          assert pattern.match(etco2)
          per_vital_dict[self.eVitals_16_etco2_index].append(float(etco2))
          per_pcr_vitals_dict[pcr_k][self.eVitals_16_etco2_index].append(float(etco2))
        else:
          per_pcr_vitals_dict[pcr_k][self.eVitals_16_etco2_index].append(None)
          # valid_line = True

        ### eVitals_16_blood_glucose ###
        blood_glucose = vitals_event[self.eVitals_18_blood_glucose_index].strip()
        if blood_glucose not in util.nemsis_value_blacklist(): 
          if blood_glucose != "High" and blood_glucose != "Low":
            pattern = re.compile(r'^(?:[2][0][0][0]|[0-1][0-9][0-9][0-9]|[0-9][0-9][0-9]|[0-9][0-9]|[0-9]|High|Low)$')
            assert pattern.match(blood_glucose)
            per_vital_dict[self.eVitals_18_blood_glucose_index].append(float(blood_glucose))
            per_pcr_vitals_dict[pcr_k][self.eVitals_18_blood_glucose_index].append(float(blood_glucose))
            # valid_line = True
          else:
            per_pcr_vitals_dict[pcr_k][self.eVitals_18_blood_glucose_index].append(blood_glucose)
        else:
          per_pcr_vitals_dict[pcr_k][self.eVitals_18_blood_glucose_index].append(None)

    # print(set(pcr_vitals_count.values()))
    print(f"Total rows {row}, total pcr {len(pcr_vitals_count)}, total valid vitals {len(per_vital_dict)}")
    # for key, value in valid_vitals_dict.items():
    #   print(key, "->", len(value), "->", numpy.(value), "->", numpy.std(value))

    for key, value in per_vital_dict.items():
      if key == self.eVitals_01_date_time_index:
        print(key, "->", len(value))
      else:
        valid_vitals_mean_dict[key] = numpy.mean(value)                                                           # for z_score normalization
        valid_vitals_std_dict[key] = numpy.std(value)                                                             # for z_score normalization
        z_score_normed_vitals = [(v - valid_vitals_mean_dict[key]) / valid_vitals_std_dict[key] for v in value]
        max_abs_vital = max(abs(numpy.min(z_score_normed_vitals)), abs(numpy.max(z_score_normed_vitals)))
        valid_vitals_max_abs_dict[key] = max_abs_vital                                                            # for min_max normalization after z_score normalization
        valid_vitals_min_dict[key] = numpy.min(value)                                                             # for min_max normalization
        valid_vitals_max_dict[key] = numpy.max(value)                                                             # for min_max normalization

        if key == self.eVitals_18_blood_glucose_index:
          valid_vitals_20_percentile_dict[key] = numpy.percentile(value, 20)
          valid_vitals_80_percentile_dict[key] = numpy.percentile(value, 80)

        print(f"{key} -> {len(value)} -> {valid_vitals_min_dict[key]} -> {numpy.percentile(value, 5)} -> {numpy.percentile(value, 20)} -> {numpy.percentile(value, 80)} -> {numpy.percentile(value, 95)} ->  {valid_vitals_max_dict[key]} -> {valid_vitals_mean_dict[key]} -> {valid_vitals_std_dict[key]} -> {valid_vitals_max_abs_dict[key]}")
    
    # dump the vitals_mean to a metadata file

    normalization_data = {
        "vitals_mean": valid_vitals_mean_dict,
        "vitals_std": valid_vitals_std_dict,
        "vitals_min": valid_vitals_min_dict,
        "vitals_max": valid_vitals_max_dict,
        "vitals_max_z_score": valid_vitals_max_abs_dict
    }

    # Write the dictionary to the JSON file
    with open(self.cached_vitals_normalization_params_file_path, "w") as file:
        json.dump(normalization_data, file, indent=4)

    # util.writeDictToFile(self.cached_vitals_normalization_params_file_path, valid_vitals_mean_dict)

    return self.sort_and_normalize_vitals(
                            per_pcr_vitals_dict,
                            valid_vitals_mean_dict,
                            valid_vitals_std_dict,
                            valid_vitals_max_abs_dict,
                            valid_vitals_min_dict,
                            valid_vitals_max_dict,
                            valid_vitals_20_percentile_dict,
                            valid_vitals_80_percentile_dict,
                          )

    # return pcr2vitals_dict

  def normalize_vitals(self, unnormed_vitals, min_vital, max_vital, mean_vital, std_vital, max_abs_vital):

    minmax_normed_vitals = []
    z_score_normed_vitals = []
    chained_normed_vitals = []

    for i, unnormed_vital in enumerate(unnormed_vitals):
      
      minmax_normed_vital = util.min_max_normalize(unnormed_vital, min_vital, max_vital)      # min-max normalization
      minmax_normed_vitals.append(minmax_normed_vital)
      z_score_normed_vital = util.z_score_normalize(unnormed_vital, mean_vital, std_vital)    # z-score normalization
      z_score_normed_vitals.append(z_score_normed_vital)
      chained_normed_vital = util.chained_normalize(z_score_normed_vital, max_abs_vital)      # min-max normalization after z-score normalization
      chained_normed_vitals.append(chained_normed_vital)
    
    return [minmax_normed_vitals, z_score_normed_vitals, chained_normed_vitals]

  def sort_and_normalize_vitals(self,
                                per_pcr_vitals_dict,
                                valid_vitals_mean_dict,
                                valid_vitals_std_dict,
                                valid_vitals_max_abs_dict,
                                valid_vitals_min_dict,
                                valid_vitals_max_dict,
                                valid_vitals_20_percentile_dict,
                                valid_vitals_80_percentile_dict):

    """
      For each pcr, for each vital, we sort by the date time and then normalize the vitals.
    """

    pcr2vitals_dict = defaultdict(lambda: defaultdict(list))

    for vital_index in self.vitals_of_interest:

      mean_vital = valid_vitals_mean_dict[vital_index]
      std_vital = valid_vitals_std_dict[vital_index]
      max_abs_vital = valid_vitals_max_abs_dict[vital_index]
      min_vital = valid_vitals_min_dict[vital_index]
      max_vital = valid_vitals_max_dict[vital_index]

      for pcr_k, vitals_dict in per_pcr_vitals_dict.items():  

        # chronologically sort the vitals based on the date time
        assert len(vitals_dict[vital_index]) == len(vitals_dict[self.eVitals_01_date_time_index])
        combined_list = list(zip(vitals_dict[self.eVitals_01_date_time_index], vitals_dict[vital_index]))
        try:
          sorted_combined = sorted(combined_list, key=lambda x: x[0])
        except TypeError:
          print(pcr_k)
          print(vitals_dict[self.eVitals_01_date_time_index])
          print(vitals_dict[vital_index])
          print(combined_list)
          print(sorted(combined_list))
          exit()
        chronologically_sorted_vitals = [item[1] for item in sorted_combined]                 # take out the sorted vitals

        for i, vital in enumerate(chronologically_sorted_vitals):
          if vital == "High" or vital == "Low":
            assert vital_index == self.eVitals_18_blood_glucose_index
            # glucose vitals contains "High" and "Low" values, replace them with the 20th and 80th percentiles, respectively
            chronologically_sorted_vitals[i] = valid_vitals_20_percentile_dict[vital_index] if vital == "Low" else valid_vitals_80_percentile_dict[vital_index]        

        mean_filled_vitals = util.mean_fill_missing_values(chronologically_sorted_vitals[:], mean_vital)
        forward_backward_filled_vitals = util.forward_backward_fill_missing_values(chronologically_sorted_vitals[:], mean_vital)


        mean_filled_normed_vitals = self.normalize_vitals(mean_filled_vitals, min_vital, max_vital, mean_vital, std_vital, max_abs_vital)
        forward_backward_filled_normed_vitals = self.normalize_vitals(forward_backward_filled_vitals, min_vital, max_vital, mean_vital, std_vital, max_abs_vital)

        # check all values in mean_filled_vitals and forward_backward_filled_vitals are not None
        # assert all(v is not None for v in mean_filled_normed_vitals)
        # assert all(v is not None for v in forward_backward_filled_normed_vitals)

            
        # if pcr_k == "233585628":
        #     print(f"pcr_k: {pcr_k}, vital_index: {vital_index}")
        #     print(f"original vitals: {vitals_dict[vital_index]}")
        #     print(f"chronologically_sorted_vitals: {chronologically_sorted_vitals}")
        #     print(f"mean_filled_vitals: {mean_filled_vitals}")
        #     print(f"forward_backward_filled_vitals: {forward_backward_filled_vitals}")
        #     print(f"mean_filled_normed_vitals: {mean_filled_normed_vitals}")
        #     print(f"forward_backward_filled_normed_vitals: {forward_backward_filled_normed_vitals}")

        pcr2vitals_dict[pcr_k][vital_index] = [chronologically_sorted_vitals] + mean_filled_normed_vitals + forward_backward_filled_normed_vitals
        # pcr2vitals_dict[pcr_k][vital_index] = [mean_filled_normed_vitals, forward_backward_filled_normed_vitals]
    return pcr2vitals_dict   

  def get_si_protocol_med_vitals(self):
  # def get_si_protocol_med_vitals(self, pcr2vitals, args):
    """
      input: texts of patient's symptoms and impressions, vitals
      labels: 1. protocol, 2. med_type, 3. med_quantity
      for each type of med, most of its units is only one unit
    """
    if os.path.isfile(self.cached_pcr2si_protocol_med_vital_file_path):
      return util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_file_path, line_length=55, is_value_a_set=False, discard_header=True)
    pcr2vitals = self.get_pcr2vitals()

    # begin to organize the input and labels
    vitals_headers = []
    vital_filling = ["mean_filled", "forward_backward_filled"]
    vital_norms = ["minmax", "z_score", "chained"]
    for i in range(len(self.vitals_of_interest)):
      vitals_headers.append("vital_%s" % i)
      for f in vital_filling:
        for n in vital_norms:
            vitals_headers.append("%s_%s_vital_%s" % (f, n, i))           # refer to the buttom of get_cached_pcr2si_protocol_med_vitals
    print("vitals_headers:", vitals_headers)

    write_line_count = 0
    with open(self.mp.cached_pcr2si_protocol_med_file_path, "r") as r_f, open(self.cached_pcr2si_protocol_med_vital_file_path, "w") as w_f:
      for row, line in enumerate(r_f):
        si_protocol_med_event = line.strip().split('~|~')
        if row == 0:
          w_f.write("~|~".join(si_protocol_med_event + vitals_headers) + "\n")
          write_line_count += 1
          continue
        pcr_k = line.split("~|~")[0]
        if pcr_k in pcr2vitals:
          vital_examples = self.get_vital_examples(pcr2vitals[pcr_k])
          assert len(vital_examples) == len(self.vitals_of_interest) * 7
          assert len(vital_examples) == len(vitals_headers)
          w_f.write("~|~".join(si_protocol_med_event + vital_examples) + "\n")
          write_line_count += 1
    print("write %s lines to %s" % (write_line_count, self.cached_pcr2si_protocol_med_vital_file_path))
    return util.read_dict_from_file(self.cached_pcr2si_protocol_med_vital_file_path, line_length=55, is_value_a_set=False, discard_header=True)

  # def write_vital_files(self, lines, pcr2vitals, vitals_headers, file_path):

  #   pcr_med_lines = ['~|~'.join(lines[0])]
  #   vital_lines = ['~|~'.join(vitals_headers)]
  #   for _, line in enumerate(lines[1:]):
  #     if len(line) != 9:
  #       print(line)
  #     assert len(line) == 9
  #     pcr_k = line[0]
  #     if pcr_k in pcr2vitals:
  #       vital_examples = self.get_vital_examples(pcr2vitals[pcr_k])
  #       assert len(vital_examples) == len(self.vitals_of_interest) * 7
  #       assert len(vital_examples) == len(vitals_headers)
  #       pcr_med_lines.append('~|~'.join(line))
  #       vital_lines.append('~|~'.join(vital_examples))
  #   util.write_train_eval_test_files(file_path, pcr_med_lines, vital_lines)

  def get_vital_examples(self, vitals_dict):

    vital_examples = []
    for vital_index in self.vitals_of_interest:
      unnormed_normed_vitals = vitals_dict[vital_index]
      assert len(unnormed_normed_vitals) == 7           # we have 1 original vital and 6 normed vitals, 6*7 = 42 columns
      # vital_example = []
      for idx, vitals in enumerate(unnormed_normed_vitals):
        assert len(vitals) == len(unnormed_normed_vitals[0])
        vital_examples.append(' '.join(map(str, vitals)))
      # vital_examples.append(vital_example)
    return vital_examples


if __name__ == "__main__":
    
  time_s = datetime.now()

  parser = argparse.ArgumentParser(description = "control the functions for NEMSIS Medication processing")
  parser.add_argument("--nemsis_dir", action='store', type=str, default = "/slot1/NEMSIS_Databases")
  parser.add_argument("--nemsis_year", action='store', type=str, default = "2023")
  parser.add_argument("--data_folder", action='store', type=str, default = "data")
  # parser.add_argument("--nemsis_med_quant_unit_file", action='store', type=str, default = "nemsis_med_quant_units.csv")
  parser.add_argument("--cache_folder", action='store', type=str, default = "nemsis_cache_files")
  # parser.add_argument("--dedicated_med_file", action='store', type=str, default = "tamu_medication.csv")
  # parser.add_argument("--dedicated_protocol_file", action='store', type=str, default = None)

  # parser.add_argument("--train_file", action='store', type=str, default = "train_med.txt")
  # parser.add_argument("--val_file", action='store', type=str, default = "val_med.txt")
  # parser.add_argument("--test_file", action='store', type=str, default = "test_med.txt")

  # parser.add_argument("--train_file", action='store', type=str, default = "train_si_protocol_med_vitals.txt")
  # parser.add_argument("--val_file", action='store', type=str, default = "val_si_protocol_med_vitals.txt")
  # parser.add_argument("--test_file", action='store', type=str, default = "test_si_protocol_med_vitals.txt")
  # parser.add_argument("--train_file_multi_label", action='store', type=str, default = "train_vitals_multi_label.txt")
  # parser.add_argument("--val_file_multi_label", action='store', type=str, default = "val_vitals_multi_label.txt")
  # parser.add_argument("--test_file_multi_label", action='store', type=str, default = "test_vitals_multi_label.txt")

  args = parser.parse_args()

  vp = Vital_Processor(args)
  # pcr2vitals = vp.get_pcr2vitals()
  # vp.write_train_validate_test_files(pcr2vitals, args)
  vp.get_si_protocol_med_vitals()
  # mp.write_med2unit()

  time_t = datetime.now() - time_s
  print("This run takes %s" % time_t)
