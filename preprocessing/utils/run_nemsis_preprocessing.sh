# clear
# python medication_processor.py --nemsis_year 2023 --data_folder data --cache_folder nemsis_cache_files --dedicated_med_file tamu_medication.csv --nemsis_med_quant_unit_file nemsis_med_quant_units.csv
# python vitals_processor.py --nemsis_year 2022
# python vitals_processor.py --nemsis_year 2021
# python vitals_processor.py --nemsis_year 2020
# python vitals_processor.py --nemsis_year 2019

# python nemsis_processor.py --nemsis_year 2023
# python medication_processor.py --nemsis_year 2023
# python vitals_processor.py --nemsis_year 2023
# python history_processor.py --nemsis_year 2023

# python procedure_processor.py --nemsis_year 2023
python scene_after_procedure_processor.py --nemsis_year 2023
# python ppg_nemsis_hooker.py --nemsis_year 2023
