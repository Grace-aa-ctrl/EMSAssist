import argparse
from datetime import datetime
import os
from collections import defaultdict
import utils.file_utils as util
from utils.nemsis_processor import NEMSIS_Processor as NP
# from . import file_utils as util
# from .nemsis_processor import NEMSIS_Processor as NP
import numpy
import json

# global_seed = 1993
fact_pcr_medication_file_name = "FACTPCRMEDICATION.txt"
eMedications_01_date_time_index = 0
PcrMedicationKey_foreign_k_index = 1
PcrKey_pcr_k_index = 2
eMedications_03_medication_given_index = 3
eMedications_03Descr_index = 4
eMedications_05_medication_dosage_index = 5
eMedications_06_medication_dosage_units_index = 6
eMedications_07_response_to_medication_index = 7
eMedications_10_role_of_person_administering_medication_index = 8
eMedications_02_medication_administering_before_EMS_index = 9

cached_pcr2med_file_name = "nemsis_pcr2med.txt"
cached_pcr2si_protocol_med_file_name = "nemsis_pcr2si_protocol_med.txt"                             # 4 si - med line
cached_si2med_med_set_file_name = "nemsis_si2med_med_set_dedicated.txt"   # med code list
dedicated_med_file = "tamu_medication.csv"
# nemsis_med_code2unit_file_name = "nemsis_med_type2unit_code.txt"               # 
nemsis_med_quant_unit_file = "nemsis_med_quant_units.csv"

# cached_si2med_quantity_tamu_cardiac_file_name = "nemsis_si2med_quantity_dedicated.txt"
# cached_med2unit_tamu_cardiac_file_name = "nemsis_med2unit_dedicated.txt"
# cached_med2desc_file_name = "nemsis_med2desc_dedicated.txt"               # med code - desc
# cached_si2med_dedicated_file_name = "nemsis_si2med_dedicated.txt"         # 4 si - type_num - med_desc - dosage - dosage_unit
# cached_oxygen_gas_quantity_file_name = "nemsis_oxygen_gas_dedicated.txt"
# cached_oxygen_liquid_quantity_file_name = "nemsis_oxygen_liquid_dedicated.txt"

class Medication_Processor(object):

  def __init__(self, args):

    self.current_dir = os.path.dirname(os.path.realpath(__file__))
    self.nemsis_dir = args.nemsis_dir
    self.nemsis_year = args.nemsis_year
    self.data_dir = os.path.join(self.current_dir, "..",  args.data_folder)
    self.cache_dir = os.path.join(self.data_dir, args.cache_folder)
    self.np = NP(self.nemsis_dir, self.data_dir, self.cache_dir, self.nemsis_year)

    self.nemsis_med_quant_unit_file = os.path.join(self.data_dir, nemsis_med_quant_unit_file)
    self.nemsis_unitcode2unitdesc = self.get_nemsis_unitcode2unitdesc()
    print("nemsis_unitcode2unitdesc len:", len(self.nemsis_unitcode2unitdesc))

    self.dedicated_med_file_path = os.path.join(self.data_dir, dedicated_med_file)
    self.dedicated_medications, self.dedicated_med2id, self.dedicated_type2desc = self.get_dedicated_medication()    ## todo: get the type2med_desc with dedicated medication list

    self.cached_pcr2si_protocol_med_file_path = os.path.join(self.cache_dir, self.nemsis_year, cached_pcr2si_protocol_med_file_name)
    self.cached_pcr2med_file_path = os.path.join(self.cache_dir, self.nemsis_year, cached_pcr2med_file_name)
    # print(len(self.dedicated_medications), "self.dedicated_medications:", self.dedicated_medications, "self.dedicated_medication_set:", self.dedicated_medication_set)
    # print(len(self.unitcode2unitdesc), "self.unitcode2unitdesc:", self.unitcode2unitdesc)
    # self.cached_med_code2unit_file_path = os.path.join(self.cache_dir, self.nemsis_year, nemsis_med_code2unit_file_name)

  def get_nemsis_unitcode2unitdesc(self):
    unitcode2unitdesc = dict()
    with open(self.nemsis_med_quant_unit_file, "r") as r_f:
      for _, line in enumerate(r_f):
        # medications_in_one_line = []
        unitcode, unitdesc = line.strip().split(',')
        assert unitcode.strip() not in unitcode2unitdesc
        unitcode2unitdesc[unitcode.strip()] = unitdesc.strip()
    return unitcode2unitdesc

  def get_dedicated_medication(self):
    dedicated_med2id = defaultdict(int)
    dedicated_med_list = []
    dedicated_type2desc = defaultdict(str)
    with open(self.dedicated_med_file_path, 'r') as r_f:
      for med_id, line in enumerate(r_f):
        medications_in_one_line = []
        meds = line.strip().split(',')

        for med in meds:
          if med != "": 
            medications_in_one_line.append(med)
            dedicated_med2id[med] = str(med_id)
        # print(medications_in_one_line)
        dedicated_med_list.append(medications_in_one_line)
        dedicated_type2desc[str(med_id)] = " ".join(medications_in_one_line)

    return dedicated_med_list, dedicated_med2id, dedicated_type2desc  

  # def read_cached_pcr2si_protocol_med(self, file_path):

  #   pcr2si_protocol_med = defaultdict(list)
  #   with open(file_path, "r") as r_f:
  #     for row, line in enumerate(r_f):

  #       event = line.strip().split('~|~')
  #       assert len(event) == 10
  #       pcr_k = event[0].strip()
  #       si_protocol_med = ('~|~').join(event[1:])
  #       pcr2si_protocol_med[pcr_k].append(si_protocol_med)

  #   # print("[read_cached_pcr2si_protocol_med] retrieved %s pcr2si_protocol_med from %s with %s lines of cached pcr2si_protocol_med, the key set size = %s, the value set size = %s" % 
  #   #       (len(pcr2si_protocol_med), file_path, row+1, len(set(pcr2si_protocol_med.keys())), len(set(pcr2si_protocol_med.values()))))
  #   print("[read_cached_pcr2si_protocol_med] retrieved %s pcr2si_protocol_med from %s with %s lines of cached pcr2si_protocol_med, the key set size = %s" % 
  #         (len(pcr2si_protocol_med), file_path, row+1, len(set(pcr2si_protocol_med.keys()))))
  #   return pcr2si_protocol_med

  def get_pcr2med_set(self):
    """
      get_pcr2med is to write the pcr2med dict file. It accomplish the following: 
        1. pcr2med1:  1) remove the nemsis pcr2med records based on the pcr filter of pcr2si_protocols and invalid med lines
                      2) determine whether a medication is dedicated based on the medication description
        2. pcr2med2: remove the med lines that don't have primarily dominated units
        3. pcr2med3: remove the med lines whose quantity is out of the range of [2%, 98%] of the med quantity
        4. In the returned pcr2med, each med is a concatenation of multiple meds
        5. The returned metadata is the type2quantities, which is used to normalize the med quantity
    """

    if os.path.isfile(self.cached_pcr2med_file_path):
      return util.read_dict_from_file(self.cached_pcr2med_file_path, line_length=5, is_value_a_set=False, discard_header=True)

    # multiple protocol labels
    nemsis_pcr2si_protocols = self.np.get_pcr2si_protocols()
    print("get pcr2si_protocol of size %s" % len(nemsis_pcr2si_protocols))
    # return None

    # the date index is move to the end of the column before nemsis 2020
    pcr_k_index = PcrKey_pcr_k_index if int(self.nemsis_year) >= 2020 else PcrKey_pcr_k_index - 1
    med_index = eMedications_03_medication_given_index if int(self.nemsis_year) >= 2020 else eMedications_03_medication_given_index - 1
    med_desc_index = eMedications_03Descr_index if int(self.nemsis_year) >= 2020 else eMedications_03Descr_index - 1
    med_improve_index = eMedications_07_response_to_medication_index if int(self.nemsis_year) >= 2020 else eMedications_07_response_to_medication_index - 1
    med_quantity_index = eMedications_05_medication_dosage_index if int(self.nemsis_year) >= 2020 else eMedications_05_medication_dosage_index - 1 
    med_quantity_unit_index = eMedications_06_medication_dosage_units_index if int(self.nemsis_year) >= 2020 else eMedications_06_medication_dosage_units_index - 1 

    # med_code2desc = dict()
    # desc2med_code = dict()
    # dedicated_med_type2count = defaultdict(int)
    # dedicated_protocol2count = defaultdict(int)

    pcr2med1 = defaultdict(list)                                  # pcr: [[med_type, med_desc, med_quantity, med_quantity_unit]]
    pcr2med2 = defaultdict(list)                                  # pcr: [[med_type, med_desc, med_quantity, med_quantity_unit]]
    # pcr2med2 = defaultdict(lambda: defaultdict(float))            # pcr: med_type -> med_quantity
    # pcr2med3 = defaultdict(list)
    type2unit_count = defaultdict(lambda: defaultdict(int))
    type2quantities = defaultdict(list)

    write_line_count = 0
    nemsis_med_file_path = os.path.join(self.nemsis_dir, self.nemsis_year, fact_pcr_medication_file_name)
    with open(nemsis_med_file_path, "r") as r_f:

      ##### filter 1: remove invalid med lines
      for row, line in enumerate(r_f):
        if row == 0:
          continue
        med_event = line.strip().split('~|~')
        assert len(med_event) == 10

        pcr_k = med_event[pcr_k_index].strip()
        med = med_event[med_index].strip()
        med_desc = med_event[med_desc_index].strip()
        response_to_med = med_event[med_improve_index].strip()
        med_quantity = med_event[med_quantity_index].strip()
        # med_quantity_unit = med_event[med_quantity_unit_index].strip()
        unit_code = med_event[med_quantity_unit_index].strip()

        if pcr_k not in nemsis_pcr2si_protocols:
          continue

        # med is not administered due to various reasons, references below: 
        # https://nemsis.org/media/nemsis_v3/release-3.5.0/DataDictionary/PDFHTML/EMSDEMSTATE/index.html
        # https://nemsis.org/wp-content/uploads/2022/05/2021-NEMSIS-RDS-340-User-Manual_v1-FINAL_.pdf
        if med == '7701001' or med == '7701003' or med == '8801001' or med == '8801003' or med == '8801007' or med == '8801009' or med == '8801019' or med == '8801023' or med.lower() == "unknown":
          continue
        if med_desc == 'Not Applicable':
          continue
        if med_quantity == '7701001' or med_quantity == '7701003' or float(med_quantity) <= 0.0:
          continue

        if response_to_med != '9916001': # we only focus on improved case
          continue

        if unit_code == '7701001' or unit_code == '7701003' or unit_code.lower() == "unknown":
          continue
        # if med_quantity_unit not in self.unitcode2unitdesc:
        #   print("row = %s, med_quantity_unit = %s" % (row, med_quantity_unit))
        # assert unit_desc == self.nemsis_unitcode2unitdesc[unit_code]
        # assert unit_code in self.nemsis_unitcode2unitdesc
        # unit_desc = self.nemsis_unitcode2unitdesc[unit_code]

        # we determine whether a medication is dedicated based on the medication description
        is_dedicated = False
        candidate_med = med_desc.lower()
        med_type = None
        for type_num, meds_same_type in enumerate(self.dedicated_medications):
          # med_type_list = []
          for dedicated_med in meds_same_type:
            if (dedicated_med.lower() in candidate_med) or (candidate_med in dedicated_med.lower()):
              is_dedicated = True
              med_type = type_num
              break

        if not is_dedicated:
          continue

        # print("there are protocol and medication pcr")

        # if med not in med_code2desc:
        #   med_code2desc[med] = med_desc
        # else:
        #   # remove several mistaken/outlier cases in 2018 nemsis medication data
        #   if (med == "285059" and med_desc.lower() == "oxygen") or (med == "4850" and med_desc.lower() == "heparin") or (med == "5224" and med_desc.lower() == "duoneb"):
        #     # med_code2desc[med] = "DuoNeb"
        #   #   continue
        #   # if med == "4850" and med_desc.lower() == "heparin":
        #   #   # med_code2desc[med] = "Glucose"
        #   #   continue
        #   # if med == "5224" and med_desc.lower() == "duoneb":
        #     # med_code2desc[med] = "Heparin"
        #     continue
        #   if med_desc != med_code2desc[med]:
        #     print("row = %s, med2desc[%s] = %s, med_desc = %s" % (row, med, med_code2desc[med], med_desc))
        #   assert med_code2desc[med] == med_desc

        # if med_desc not in desc2med_code:
        #   desc2med_code[med_desc] = med
        # else:
        #   # several mistaken/outlier cases in 2018 nemsis medication data
        #   if (med_desc == "Glucose" and med == "7086") :
        #     # desc2med_code[med_desc] = "4850"
        #     continue
        #   if desc2med_code[med_desc] != med:
        #     print("row = %s, desc2med[%s] = %s, med = %s, response = %s" % (row, med_desc, desc2med_code[med_desc], med, response_to_med))
        #   assert desc2med_code[med_desc] == med

        type2unit_count[str(med_type)][unit_code] += 1          
        med_line = [str(med_type), med_desc, unit_code, med_quantity]
        pcr2med1[pcr_k].append(med_line)
      print("get pcr2med1 of size %s" % len(pcr2med1))

    ##### filter 2: remove the med lines that don't have primarily dominated units
    # type2unit_result_dict = defaultdict(str)
    type2unit_code = defaultdict(str)
    type2unit_desc = defaultdict(str)
    # med_type_keys = list(type2unit_count.keys())
    # med_type_keys.sort()
    # for w_row, med_type in enumerate(med_type_keys):
    finalized_unit_list = []
    for med_type in range(len(type2unit_count)):
      med_type = str(med_type)
      quant_unit_codes = type2unit_count[med_type]
      print(med_type, "->", {k: v for k, v in sorted(quant_unit_codes.items(), key=lambda item: item[1], reverse=True)})
      top_unit_code, top_count = sorted(quant_unit_codes.items(), key=lambda item: item[1], reverse=True)[0]
      # type2unit_result_dict[med_type] = (top_unit_code)
      type2unit_code[med_type] = top_unit_code
      finalized_unit_list.append(self.nemsis_unitcode2unitdesc[top_unit_code])
    
    med_unit_list_file = os.path.join(self.cache_dir, self.nemsis_year, "type_unit2med_list.txt")
    with open(med_unit_list_file, "w") as w_f:
      for unit_desc, med_desc in zip(finalized_unit_list, self.dedicated_medications):
        w_f.write("~|~".join([unit_desc] + med_desc) + "\n")

    # open(med_unit_list_file, "w").write("\n".join(finalized_unit_list) + "\n")                          # write this output for ocr


    # util.write_list_to_file(finalized_unit_list, med_unit_list_file)
    # Write the dictionary to the JSON file
    # with open(normalization_output_file, "w") as file:
    #     json.dump(normalization_data, file, indent=4)

    for pcr_k, med_lines in pcr2med1.items():
      for med_line in med_lines:
        med_type, med_desc, unit_code, med_quantity = med_line
        if unit_code == type2unit_code[med_type]:                            # filter 2
          # assert unit_desc == self.unitcode2unitdesc[med_quantity_unit]
          # assert unit_desc == self.nemsis_unitcode2unitdesc[unit_code]
          if med_type not in type2unit_desc:
            type2unit_desc[med_type] = self.nemsis_unitcode2unitdesc[unit_code]
          assert self.nemsis_unitcode2unitdesc[unit_code] == type2unit_desc[med_type]

          pcr2med2[pcr_k].append(med_line)
          # pcr2med2[pcr_k][med_type] += float(med_quantity)                                  
    print("get pcr2med2 of size %s" % len(pcr2med2))          

    ##### filter 3: remove the med lines whose quantity is out of the range of [2%, 98%] of the med quantity
    for pcr_k, med_lines in pcr2med2.items():
      # accumulated_type2quantity = defaultdict(float)                              # each med type's quantity will be accumulated
      for med_line in med_lines:
        med_type, med_desc, unit_code, med_quantity = med_line
        type2quantities[med_type].append(float(med_quantity))
        # accumulated_type2quantity[med_type] += float(med_quantity)
      # type2quantities[med_type].append(accumulated_type2quantity[med_type])

    for med_type in range(len(type2quantities)):
      med_type = str(med_type)
      quantity_list = type2quantities[med_type]
      print("med type: %s, min: %s, 1-pct: %s, 2-pct: %s, 5-pct: %s, 20-pct: %s, 80-pct: %s, 95-pct: %s, 98-pct: %s, 99-pct: %s, max: %s" % 
            (med_type, min(quantity_list), numpy.percentile(quantity_list, 1), numpy.percentile(quantity_list, 2), numpy.percentile(quantity_list, 5), numpy.percentile(quantity_list, 20), numpy.percentile(quantity_list, 80), numpy.percentile(quantity_list, 95), numpy.percentile(quantity_list, 98), numpy.percentile(quantity_list, 99), max(quantity_list)))
    type2quantities_2pct = {med_type: numpy.percentile(quantity_list, 2) for med_type, quantity_list in type2quantities.items()}
    type2quantities_98pct = {med_type: numpy.percentile(quantity_list, 98) for med_type, quantity_list in type2quantities.items()}

    # type2quantities is used to filter med lines, res_type2quantities is used for the actual normalization of med quantity in the next step
    res_type2quantities = defaultdict(list)
    with open(self.cached_pcr2med_file_path, "w") as w_f:
      w_f.write("pcr~|~med_type~|~med_desc~|~med_unit~|~med_quantity\n")
      write_line_count += 1
      for pcr_k, med_lines in pcr2med2.items():
        type2quantity = defaultdict(float)                              # each med type's quantity will be accumulated
        med_type_list = []
        med_desc_list = []
        unit_desc_list = []
        med_quantity_list = []

        filter_pcr_k = False
        for med_line in med_lines:
          med_type, med_desc, unit_code, med_quantity = med_line
          type2quantity[med_type] += float(med_quantity)
          # med_type, med_desc, med_quantity, med_quantity_unit, unit_desc = med_line
          if float(med_quantity) < type2quantities_2pct[med_type] or float(med_quantity) > type2quantities_98pct[med_type]:
            filter_pcr_k = True
            break      
        if filter_pcr_k:
          continue

        med_type_list = list(type2quantity.keys())
        med_type_list.sort()
        med_type_str = " ".join(med_type_list)
        med_desc_list = [self.dedicated_type2desc[med_type] for med_type in med_type_list]
        med_desc_str = ",".join(med_desc_list)
        unit_desc_list = [type2unit_desc[med_type] for med_type in med_type_list]
        unit_desc_str = ",".join(unit_desc_list)
        med_quantity_list = [str(type2quantity[med_type]) for med_type in med_type_list]
        med_quantity_str = " ".join(med_quantity_list)
        med_line_str = "~|~".join([med_type_str, med_desc_str, unit_desc_str, med_quantity_str])
        w_f.write(pcr_k + '~|~' + med_line_str + '\n')

        res_type2quantities[med_type].append(float(med_quantity))
        write_line_count += 1

    print("write %s lines pcr2med to %s" % (write_line_count, self.cached_pcr2med_file_path))
    return (util.read_dict_from_file(self.cached_pcr2med_file_path, line_length=5, is_value_a_set=False, discard_header=True), res_type2quantities)

  def get_pcr2si_protocols_meds(self):

    """
      get_pcr2si_protocols_meds is to write the pcr2si_protocol_med file. 
      1. the pcr2si_protocol_med file contains text inputs, and 3-label outputs: multiple protocols, mutliple med_type, multiple med_quantity
      2. use 3 ways to normalize the med quantity based on the med type: min-max normalization, z-score normalization, chained normalization
      3. return the dict of pcr2si_protocol_med, the key is pcr, the value is a set of si_protocol_med
    """

    if os.path.isfile(self.cached_pcr2si_protocol_med_file_path):
      return util.read_dict_from_file(self.cached_pcr2si_protocol_med_file_path, line_length=13, is_value_a_set=False, discard_header=True)

    # protocol_index = 4
    # med_type_index = 5
    # med_desc_index = 6
    # quantity_index = 7
    # unit_code_index = 8
    nemsis_pcr2meds, type2quantities = self.get_pcr2med_set()
    # return None

    # aggregate normalized meds
    type2quantities_mean = {med_type: numpy.mean(quantity_list) for med_type, quantity_list in type2quantities.items()}
    type2quantities_std = {med_type: numpy.std(quantity_list) for med_type, quantity_list in type2quantities.items()}
    type2quantities_min = {med_type: min(quantity_list) for med_type, quantity_list in type2quantities.items()}
    type2quantities_max = {med_type: max(quantity_list) for med_type, quantity_list in type2quantities.items()}
    type2z_score = {med_type: [(v - type2quantities_mean[med_type]) / type2quantities_std[med_type] for v in quantity_list] for med_type, quantity_list in type2quantities.items()}
    type2abs_max_z_score = {med_type: max(abs(numpy.min(z_score)), abs(numpy.max(z_score))) for med_type, z_score in type2z_score.items()}

    normalization_data = {
        "type2quantities_mean": type2quantities_mean,
        "type2quantities_std": type2quantities_std,
        "type2quantities_min": type2quantities_min,
        "type2quantities_max": type2quantities_max,
        "type2abs_max_z_score": type2abs_max_z_score
    }

    # Specify the path where you want to save the JSON file
    normalization_output_file = os.path.join(self.cache_dir, self.nemsis_year,  "medicine_normalization_params.json")

    # Write the dictionary to the JSON file
    with open(normalization_output_file, "w") as file:
        json.dump(normalization_data, file, indent=4)

    # print(f"Normalization parameters written to {output_file}")

    for t in range(len(type2quantities_mean)):
      t = str(t)
      print("med type: %s, min: %s, 1-pct: %s, 2-pct: %s, 5-pct: %s, 20-pct: %s, 80-pct: %s, 95-pct: %s, 98-pct: %s, 99-pct: %s, max: %s, mean: %s, std: %s" % 
            (t, type2quantities_min[t], numpy.percentile(type2quantities[t], 1), numpy.percentile(type2quantities[t], 2), numpy.percentile(type2quantities[t], 5), numpy.percentile(type2quantities[t], 20), numpy.percentile(type2quantities[t], 80), numpy.percentile(type2quantities[t], 95), numpy.percentile(type2quantities[t], 98), numpy.percentile(type2quantities[t], 99), type2quantities_max[t], type2quantities_mean[t], type2quantities_std[t]))

    write_line_count  = 0
    with open(self.np.cached_pcr2si_protocol_file_path, "r") as r_f, open(self.cached_pcr2si_protocol_med_file_path, "w") as w_f:
      w_f.write("pcr~|~ps~|~pi~|~as~|~si~|~protocol~|~med_type~|~med_desc~|~med_unit~|~quantity~|~minmax_quantity~|~z_score_quantity~|~chained_quantity\n")
      write_line_count += 1

      for row, line in enumerate(r_f):
        event = line.strip().split('~|~')
        assert len(event) == 6
        pcr_k = event[0].strip()
        if pcr_k not in nemsis_pcr2meds:
          continue
        si_protocol = ('~|~').join(event[1:6])

        med_type_str, med_desc_str, unit_desc_str, med_quantity_str = nemsis_pcr2meds[pcr_k].split('~|~')
        med_types = med_type_str.split(' ')
        med_quantities = med_quantity_str.split(' ')
        assert len(med_types) == len(med_quantities)
        minmax_normed_quantities = []
        z_score_normed_quantities = []
        chained_normed_quantities = []
        for med_type, med_quantity in zip(med_types, med_quantities):
          med_type = str(med_type)
          quantity = float(med_quantity)
          minmax_normed_quantity = util.min_max_normalize(quantity, type2quantities_min[med_type], type2quantities_max[med_type])
          minmax_normed_quantities.append(minmax_normed_quantity)
          z_score_normed_quantity = util.z_score_normalize(quantity, type2quantities_mean[med_type], type2quantities_std[med_type])
          z_score_normed_quantities.append(z_score_normed_quantity)
          chained_normed_quantity = util.chained_normalize(z_score_normed_quantity, type2abs_max_z_score[med_type])
          chained_normed_quantities.append(chained_normed_quantity)
        minmax_normed_quantity_str = ' '.join([str(q) for q in minmax_normed_quantities])
        z_score_normed_quantity_str = ' '.join([str(q) for q in z_score_normed_quantities])
        chained_normed_quantity_str = ' '.join([str(q) for q in chained_normed_quantities])

        si_protocol_med_line = ('~|~').join([si_protocol, med_type_str, med_desc_str, unit_desc_str, med_quantity_str, minmax_normed_quantity_str, z_score_normed_quantity_str, chained_normed_quantity_str])
        w_f.write(pcr_k + '~|~' + si_protocol_med_line + '\n')
        write_line_count += 1

    print("write %s lines pcr2si_protocol_med to %s" % (write_line_count, self.cached_pcr2si_protocol_med_file_path))
    return util.read_dict_from_file(self.cached_pcr2si_protocol_med_file_path, line_length=13, is_value_a_set=False, discard_header=True)

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

  args = parser.parse_args()

  mp = Medication_Processor(args)
  # mp.get_pcr2med()
  mp.get_pcr2si_protocols_meds()

  time_t = datetime.now() - time_s
  print("This run takes %s" % time_t)
