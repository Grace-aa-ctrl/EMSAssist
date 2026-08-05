import os
import math
import json
from collections import Counter

import os
import math
import json
from collections import Counter

def merge_split_with_demographics(
        data_dir,
        split_filename,
        demo_filename="nemsis_pcr2demographics.txt",
        output_filename=None,
        include_payment=False):      

    SEP = "~|~"

    # -------------------------------
    # Race mapping → integers 1..6
    # -------------------------------
    race_map = {
        "2514001": 1, # American Indian or Alaska Native
        "2514003": 2, # Asian
        "2514005": 3, # Black or African American
        "2514007": 4, # Hispanic or Latino
        "2514009": 5, # Native Hawaiian or Other Pacific Islander
        "2514011": 6, # White
    }

    # -------------------------------
    # Payment mapping → integers 1..12
    # Only used if include_payment=True
    # -------------------------------
    payment_map = {
        "2601001": 1, # Insurance
        "2601003": 2, # Medicaid
        "2601005": 3, # Medicare
        "2601007": 4, # Not Billed (for any reason)
        "2601009": 5, # Other Government
        "2601011": 6, # Self Pay
        "2601013": 7, # Workers Compensation
        "2601015": 8, # Payment by Facility
        "2601017": 9,  # Contracted Payment
        "2601019": 10, # Community Network
        "2601021": 11, # No Insurance Identified
        "2601023": 12  # Other Payment Option
    }

    split_file = os.path.join(data_dir, split_filename)
    demo_file = os.path.join(data_dir, demo_filename)

    if output_filename is None:
        output_filename = split_filename.replace(".txt", "_demo.txt")

    out_file = os.path.join(data_dir, output_filename)

    # ---------------------------------------------------------
    # Load demographics → dict[PCR] = list(row)
    # ---------------------------------------------------------
    demo_dict = {}

    with open(demo_file, "r") as f:
        lines = f.read().strip().splitlines()
        full_demo_header = lines[0].split(SEP)

        age_idx_demo = full_demo_header.index("age_years")
        payment_idx_demo = full_demo_header.index("payment_method")
        race_idx_demo = full_demo_header.index("race_codes")

        for line in lines[1:]:
            parts = line.split(SEP)
            demo_dict[parts[0]] = parts

    # # ---------------------------------------------------------
    # # RAW MISSINGNESS (before any imputation)
    # # ---------------------------------------------------------
    # raw_missing_age = 0
    # raw_missing_race = 0
    # raw_missing_payment = 0 if include_payment else None

    # for pcr, row in demo_dict.items():
    #     age_raw = row[age_idx_demo]
    #     race_raw = row[race_idx_demo]
    #     pay_raw  = row[payment_idx_demo] if include_payment else None

    #     if age_raw in ("None", ""):
    #         raw_missing_age += 1
    #     if race_raw in ("None", ""):
    #         raw_missing_race += 1
    #     if include_payment and pay_raw in ("None", ""):
    #         raw_missing_payment += 1

    # print("\n===== RAW MISSINGNESS BEFORE IMPUTATION =====")
    # print(f" Total demographic records: {len(demo_dict)}")
    # print(f"  Missing age_years: {raw_missing_age} ({raw_missing_age/len(demo_dict):.2%})")
    # print(f"  Missing race_codes: {raw_missing_race} ({raw_missing_race/len(demo_dict):.2%})")
    # if include_payment:
    #     print(f"  Missing payment_method: {raw_missing_payment} ({raw_missing_payment/len(demo_dict):.2%})")
    # print("==============================================\n")

    # ---------------------------------------------------------
    # Compute / load age normalization stats
    # ---------------------------------------------------------
    norm_file = os.path.join(data_dir, "age_normalization.json")

    # Compute train statistics
    age_values = [
        float(row[age_idx_demo]) for row in demo_dict.values()
        if row[age_idx_demo] not in ("None", "")
    ]

    if not age_values:
        raise ValueError("No valid age values found.")

    age_mean = sum(age_values) / len(age_values)
    age_std = math.sqrt(sum((x - age_mean) ** 2 for x in age_values) / len(age_values))
    age_min = min(age_values)
    age_max = max(age_values)

    # Save only for train split
    if "train" in split_filename:
        stats = {
            "age_mean": age_mean,
            "age_std": age_std,
            "age_min": age_min,
            "age_max": age_max
        }
        with open(norm_file, "w") as jf:
            json.dump(stats, jf, indent=2)
        print(f"Saved age normalization stats → {norm_file}")

    else:
        # Load for val/test
        if not os.path.exists(norm_file):
            raise ValueError("Normalization file missing — run train split first.")
        with open(norm_file, "r") as jf:
            stats = json.load(jf)
        age_mean = stats["age_mean"]
        age_std = stats["age_std"]
        age_min = stats["age_min"]
        age_max = stats["age_max"]
        print(f"Loaded age normalization stats ← {norm_file}")

    # ---------------------------------------------------------
    # Build final demographics header
    # ---------------------------------------------------------
    demo_header = full_demo_header.copy()
    demo_body_indices = list(range(1, len(full_demo_header)))

    if not include_payment:
        demo_header.pop(payment_idx_demo)
        demo_body_indices.remove(payment_idx_demo)

    # Add z-score column
    merged_extra_cols = ["z_score_age"]

    # ---------------------------------------------------------
    # MERGING
    # ---------------------------------------------------------
    with open(split_file, "r") as f_in, open(out_file, "w") as f_out:

        split_lines = f_in.read().strip().splitlines()
        split_header = split_lines[0].split(SEP)

        merged_header = split_header + demo_header[1:] + merged_extra_cols
        f_out.write(SEP.join(merged_header) + "\n")

        for line in split_lines[1:]:
            parts = line.split(SEP)
            pcr = parts[0]

            if pcr in demo_dict:
                raw = demo_dict[pcr]
                demo_row = [raw[i] for i in demo_body_indices]
            else:
                demo_row = ["None"] * len(demo_body_indices)

            # ------- recode race -------
            race_col = demo_body_indices.index(race_idx_demo)
            race_raw = demo_row[race_col]

            if race_raw in ("None", ""):
                demo_row[race_col] = "None"
            else:
                codes = race_raw.split()
                mapped = [str(race_map[c]) for c in codes if c in race_map]
                demo_row[race_col] = " ".join(mapped) if mapped else "None"

            # ------- recode payment -------
            if include_payment:
                pay_col = demo_body_indices.index(payment_idx_demo)
                pay_raw = raw[payment_idx_demo]
                if pay_raw in ("None", ""):
                    demo_row[pay_col] = "None"
                else:
                    codes = pay_raw.split()
                    mapped = [str(payment_map[c]) for c in codes if c in payment_map]
                    demo_row[pay_col] = " ".join(mapped) if mapped else "None"

            # ------- z-score age -------
            age_col = demo_body_indices.index(age_idx_demo)
            age_raw = demo_row[age_col]

            if age_raw in ("None", ""):
                age_val = age_mean   # imputing
            else:
                age_val = float(age_raw)

            z_score_age = (age_val - age_mean) / age_std

            merged_row = parts + demo_row + [str(z_score_age)]
            f_out.write(SEP.join(merged_row) + "\n")

    print(f"Merged file saved to:\n{out_file}")

    # ---------------------------------------------------------
    # SUMMARY AFTER MERGING (unchanged)
    # ---------------------------------------------------------
    age_idx = merged_header.index("age_years")
    race_idx = merged_header.index("race_codes")
    payment_idx = merged_header.index("payment_method") if include_payment else None

    total_rows = 0
    missing_age = missing_race = missing_payment = 0
    race_counter = Counter()
    payment_counter = Counter()

    with open(out_file, "r") as f:
        next(f)
        for line in f:
            total_rows += 1
            parts = line.strip().split(SEP)

            if parts[age_idx] in ("None", ""):
                missing_age += 1

            if parts[race_idx] in ("None", ""):
                missing_race += 1
            else:
                race_counter[parts[race_idx]] += 1

            if include_payment:
                if parts[payment_idx] in ("None", ""):
                    missing_payment += 1
                else:
                    payment_counter[parts[payment_idx]] += 1

    print("\n===== SUMMARY FOR:", split_filename, "=====")
    print(f"Total rows: {total_rows}")
    print(f" Missing age (raw): {missing_age} ({missing_age/total_rows:.2%})")
    print(f" Missing race: {missing_race} ({missing_race/total_rows:.2%})")
    if include_payment:
        print(f" Missing payment: {missing_payment} ({missing_payment/total_rows:.2%})")

    print("\nRace counts:")
    for k, v in race_counter.most_common():
        print(f"  {k}: {v}")

    if include_payment:
        print("\nPayment counts:")
        for k, v in payment_counter.most_common():
            print(f"  {k}: {v}")

    return out_file



data_path = "/home/claire/conformal-prediction-ems/data/nemsis_cache_files/2023"

# for scene
merge_split_with_demographics(data_path, "train_file_scene.txt", "nemsis_pcr2demographics.txt", output_filename=None, include_payment=False)
merge_split_with_demographics(data_path, "val_file_scene.txt", "nemsis_pcr2demographics.txt", output_filename=None, include_payment=False)
merge_split_with_demographics(data_path, "test_file_scene.txt", "nemsis_pcr2demographics.txt", output_filename=None, include_payment=False)

# or for noscene
merge_split_with_demographics(data_path, "train_file_noscene.txt", "nemsis_pcr2demographics.txt", output_filename=None, include_payment=False)
merge_split_with_demographics(data_path, "val_file_noscene.txt", "nemsis_pcr2demographics.txt", output_filename=None, include_payment=False)
merge_split_with_demographics(data_path, "test_file_noscene.txt", "nemsis_pcr2demographics.txt", output_filename=None, include_payment=False)
