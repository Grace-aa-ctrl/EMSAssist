import pandas as pd
from io import open as io_open
import sys
import os
from datetime import datetime
import argparse
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import random
import math
import re
# from PIL import ExifTags, ImageDraw
from collections import OrderedDict, defaultdict
import jiwer

global_seed = 1993

def readFile(file_path):
  res = []
  with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
    for idx, line in enumerate(f):
      line = line.strip()
      res.append(line)
#   print("get %s lines from %s" % (len(res), file_path))
  return res

# If you're having issues reading the file back, here's a proper read function:
def readListFile(file_path, encoding=None):
    """Read the file back as a list, preserving empty lines"""
    with open(file_path, "r", encoding=encoding or "utf-8", errors="replace") as f:
        content = f.read()
    
    # Split by newlines to get the original list back
    lines = content.split('\n')
    return lines

def write2DListToLineFile(file_path, output_list, shard, out_dir):
    line_fns = []
    for idx, line in enumerate(output_list):

        cur_line = "line" + str(idx)
        line_file = file_path + "-" + cur_line + ".txt"

        # we concatenate the from 2:
        line_fns.append(line_file)
        writeListFile(line_file, [line])
    shard_text_fn_file = "shard" + str(shard) + "_fn_file.txt"
    shard_text_fn_file_path = os.path.join(out_dir, shard_text_fn_file)
    writeListFile(shard_text_fn_file_path, line_fns)

def list2id(list_data):
    list2id_dict = dict()
    for idx, e in enumerate(list_data):
        assert e not in list2id_dict
        list2id_dict[e] = idx
    return list2id_dict


def writeListFile(file_path, output_list, encoding = None):
    f = open(file_path, mode = "w")
    output_list = [str(e) for e in output_list]
    output_str = "\n".join(output_list)
    f.write(output_str)
    f.close()
    print("write %s lines to %s" % (len(output_list), file_path))

def write2DListFile(file_path, output_list, line_sep = " "):
    str_list = []
    for out_line in output_list:
        str_line = []
        for e in out_line:
            str_line.append(str(e))
        str_list.append(str_line)
    out_list = list(map(line_sep.join, str_list))
    writeListFile(file_path, out_list)

def writeSetFile(file_path, output_set, sort = True):
    output_list = list(output_set)
    if sort:
        output_list.sort()
    writeListFile(file_path, output_list)

# def read_train_eval_test_files(file_path, delimiter = '~|~'):
  
# #   read_lines = readFile(file_path)
#     read_lines = read_delimited_file(file_path, discard_header = False, delimiter = delimiter)
#     res_lines = []
#     for _, line in enumerate(read_lines):
#         sample_label_line = line.strip().split(delimiter)
#         sample_label_list = []
#         for line_element in sample_label_line:
#             sample_label_list.append(line_element.strip())
#         res_lines.append(sample_label_list)
#     return res_lines

def write_train_eval_test_files(file_path, sample_lines, labels):
  
  out_lines = []
  assert len(sample_lines) == len(labels)
  for sample, label in zip(sample_lines, labels):
    out_lines.append(sample + '~|~' + label)
  writeListFile(file_path, out_lines)

def read_delimited_file(file_path, discard_header = False, delimiter = None):
    lines = readFile(file_path)
    # print("reading %s lines from %s" % (len(lines), file_path))
    if discard_header:
        lines = lines[1:]
    no_join_lines = []
    for line in lines:
        if delimiter is None:
            t = line.strip().split()
        else:
            t = line.strip().split(delimiter)
        t = [e.strip() for e in t]
        no_join_lines.append(t)
    return no_join_lines

def read_dict_from_file(file_path, line_length=2, is_value_a_set=False, discard_header=False, delimiter = '~|~'):
    res_dict = defaultdict(str)
    if is_value_a_set:
        res_dict = defaultdict(set)

    with open(file_path, "r", encoding="utf-8", errors="replace") as r_f:
      for idx, line in enumerate(r_f):
        if idx == 0 and discard_header:
          continue

        event = line.strip().split(delimiter)
        if len(event) != line_length:
            print(f"line {idx} has {len(event)} elements, not equal to {line_length}")
            print(event)
            exit(1)
        assert len(event) == line_length
        
        k = event[0].strip()
        v = "~|~".join(list(map(str.strip, event[1:])))
        if is_value_a_set:
            res_dict[k].add(v)
        else:
            assert k not in res_dict
            res_dict[k] = v
    print("read %s lines from %s to get a dict size %s" % (idx+1, file_path, len(res_dict)))
    return res_dict

def writeDictToFile(file_path, d, sep = '~|~'):

    out_list = []
    for k, v in d.items():
        out_line = str(k) + sep + str(v)
        out_list.append(out_line)
    writeListFile(file_path, out_list)

def read2DArrayFromFile(file_path, line_sep = ' ', dtype = int):
    lines = readFile(file_path)
    arr = []
    for row in lines:
        cols = row.split()
        arr_cols = []
        for col in cols:
            arr_cols.append(dtype(col))
        arr.append(arr_cols)
    return arr

def whole_word_found(str, word):
    if re.search(r"\b" + re.escape(str) + r"\b", word):
        return True
    return False

def nemsis_value_blacklist():
    return set(["7701001", "Not Applicable", "7701003", "Not Recorded", "8801005", "8801019", "8801023"])

def min_max_normalize(unnormed_value, min_val, max_val):
    if type(unnormed_value) == str:
        print(f"unnormed_value is string {unnormed_value}")
        exit(1)
    return 2 * (unnormed_value - min_val) / (max_val - min_val) - 1

def z_score_normalize(unnormed_value, mean_val, std_val):
    return (unnormed_value - mean_val) / std_val

def chained_normalize(z_scored_normed_value, max_abs_val):
    return z_scored_normed_value / max_abs_val

def forward_backward_fill_missing_values(data, mean_value):
    """
    Fill missing values in the list with the previous and following values.
    
    Parameters:
        data (list of float or None): The data list with missing values represented by None.
        
    Returns:
        list of float: The data list with missing values filled.
    """
    
    n = len(data)
    
    # Forward fill
    for i in range(1, n):
        if data[i] is None:
            data[i] = data[i - 1]

    # Backward fill
    for i in range(n - 2, -1, -1):
        if data[i] is None:
            data[i] = data[i + 1]

    if all(val is None for val in data):
        return mean_fill_missing_values(data, mean_value)
    return data

def mean_fill_missing_values(data, mean_value):
    """
    Fill missing values in the list with mean values.
    
    Parameters:
        data (list of float or None): The data list with missing values represented by None.
        
    Returns:
        list of float: The data list with missing values filled.
    """
    for i, val in enumerate(data):
        if val is None:
            data[i] = mean_value  # if the vital is missing, we will fill it with the mean value
    return data

def split_data(samples, labels, seed=global_seed):

    """
    Split the samples and labels into training and test sets.
    Parameters:
        samples (list): The samples to split.
        labels (list): The labels to split.
        seed (int): The seed for random number generation.
        
    Returns:
        list: The training set.
        list: The test set.
    """
    train_lines, val_test_lines, train_labels, val_test_labels = train_test_split(samples, labels, random_state=seed, test_size=0.4)
    val_lines, test_lines, val_labels, test_labels = train_test_split(val_test_lines, val_test_labels, random_state=seed, test_size=0.5)
    print("total number of samples %s, train samples %s, eval samples %s, test samples %s" % (len(samples), len(train_lines), len(val_lines), len(test_lines)))
    return train_lines, val_lines, test_lines, train_labels, val_labels, test_labels

# Function to correct the image orientation based on EXIF data
def correct_image_orientation(image, save=False, save_path=None):
    """
    Correct the orientation of an image based on its EXIF data.
    """
    if os.path.isfile(save_path):
        return image, False
    
    try:
        exif = image._getexif()
        rotated = False
        if exif is not None:
            # print("exif is not None")
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                # print(tag_name)
                if tag_name == 'Orientation':
                    if value == 3:
                        # print("rotate 180")
                        image = image.rotate(180, expand=True)
                        # rotated = True
                    elif value == 6:
                        # print("rotate 270")
                        image = image.rotate(270, expand=True)
                        rotated = True
                    elif value == 8:
                        # print("rotate 90")
                        image = image.rotate(90, expand=True)
                        rotated = True
                # else:
                #     print("tag_name is not Orientation")
        if save:
            image.save(save_path)

    # except AttributeError:
    #     pass
    except Exception as e:
        print("Error in correct_image_orientation for image %s: ", save_path)

    return image, rotated

# Function to draw bounding boxes on the image
def add_bounding_boxes(image, annotations):
    draw = ImageDraw.Draw(image)
    for box in annotations:
        x_min, y_min, x_max, y_max = tuple(box)
        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=5)
        draw.text((x_min, y_min), "object", fill="white")
    return image

# Define your transformation
text_transformation = jiwer.Compose([
    jiwer.SubstituteRegexes({
        r"~\|~": r" ", 
        r"\[": r" ", 
        r"\]": r" ",
        r"\(": r" ", 
        r"\)": r" ",
        r"\,": r" "
    }),  # Remove the delimiter ~|~ and brackets [ ]
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")  # Reduce to list of words
])

if __name__ == "__main__":   
    
    t_start = datetime.now()

    parser = argparse.ArgumentParser(description = "control the functions to extract metamap concepts and select nemsis protocols")
    parser.add_argument("--raw_input", action='store', default="", type=str, help="raw input to give quick check of MetaMap or MetaMapLite")
    parser.add_argument("--plot_match_count", action='store_true', default=False, help="decide whether to plot the count statistics")
    

    args = parser.parse_args()
    if args.raw_input:
        concept_list = processRawInput(args.raw_input)
        print(";".join(concept_list))

    if args.plot_match_count:
        plotCountGraph()

    t_total = datetime.now() - t_start
    print("this run takes %s" % t_total)
