import os
import evaluate
import torch
from tqdm.auto import tqdm

def get_model_path(saved_model_dir, 
                   nemsis_year, 
                   text_encoder_name, 
                   vital_encoder_name, 
                   scene_encoder_name, 
                   do_protocol, 
                   do_med_type, 
                   do_quantity, 
                   debug_mode,
                   single_label = True):
    
    # create the dir and save the trained models and validation/test results
    saved_model_dir = os.path.join(saved_model_dir, nemsis_year)
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    best_model_name = ""
    # Append encoder names if they exist
    if text_encoder_name:
        best_model_name += text_encoder_name
    if vital_encoder_name:
        if best_model_name:  # If there's already a name part, add an underscore
            best_model_name += "_"
        best_model_name += vital_encoder_name
    if scene_encoder_name:
        if best_model_name:  # If there's already a name part, add an underscore
            best_model_name += "_"
        best_model_name += scene_encoder_name
    best_model_name += "_debug_mode" if debug_mode else ""
    best_model_name += "_single_label"

    task_name = ""
    if do_protocol:
        task_name += "_protocol" 
    if do_med_type:
        task_name += "_med_type" 
    if do_quantity:
        task_name += "_quantity"

    best_model_path = os.path.join(saved_model_dir, best_model_name + task_name + ".pt")
    print("[training_utils] according to your config, best_model_path:", best_model_path)

    return best_model_name, task_name, best_model_path, saved_model_dir

def get_progressive_model_path(saved_model_dir,
                               nemsis_year,
                               text_encoder_name,
                               vital_encoder_name,
                               scene_encoder_name,
                               do_protocol,
                               do_med_type,
                               do_quantity,
                               debug_mode,
                               single_label = True,
                               pretrained_text_model=None,
                               pretrained_vital_model=None,
                               pretrained_scene_model=None,
                               pretrained_textvital_model=None,
                               pretrained_textscene_model=None,
                               pretrained_vitalscene_model=None):
    
    # create the dir and save the trained models and validation/test results
    saved_model_dir = os.path.join(saved_model_dir, nemsis_year)
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)

    best_model_name = ""
    # Append encoder names if they exist
    if text_encoder_name:
        best_model_name += text_encoder_name
    if vital_encoder_name:
        if best_model_name:  # If there's already a name part, add an underscore
            best_model_name += "_"
        best_model_name += vital_encoder_name
    if scene_encoder_name:
        if best_model_name:  # If there's already a name part, add an underscore
            best_model_name += "_"
        best_model_name += scene_encoder_name
    best_model_name += "_debug_mode" if debug_mode else ""
    best_model_name += "_single_label"

    pretrain = ""
    if pretrained_textvital_model:
        pretrain += "tvp"
    elif pretrained_textscene_model:
        pretrain += "tsp"
    elif pretrained_vitalscene_model:
        pretrain += "vsp"
    else:
        if pretrained_text_model:
            pretrain += "tp"
        if pretrained_vital_model:
            pretrain += "vp"
        if pretrained_scene_model:
            pretrain += "sp"

    best_model_name += "_" + pretrain


    task_name = ""
    if do_protocol:
        task_name += "_protocol" 
    if do_med_type:
        task_name += "_med_type" 
    if do_quantity:
        task_name += "_quantity"

    best_model_path = os.path.join(saved_model_dir, best_model_name + task_name + ".pt")
    print("[training_utils] according to your config, best_model_path:", best_model_path)

    return best_model_name, task_name, best_model_path, saved_model_dir

def print_evaluate_results(args, results):

    for model_name in results.keys():

        if args.do_protocol:
            protocol_accuracy = results[model_name]["protocol"]["accuracy"]
            protocol_top1_accuracy = results[model_name]["protocol"]["top-1"]
            protocol_top3_accuracy = results[model_name]["protocol"]["top-3"]
            protocol_top5_accuracy = results[model_name]["protocol"]["top-5"]
            print(f"[training_utils] Protocol - Accuracy: {protocol_accuracy:.4f}, top-1: {protocol_top1_accuracy:.4f}, top-3: {protocol_top3_accuracy:.4f}, top-5: {protocol_top5_accuracy:.4f}")
        if args.do_med_type:
            med_type_accuracy = results[model_name]["med_type"]["accuracy"]
            med_type_top1_accuracy = results[model_name]["med_type"]["top-1"]
            med_type_top3_accuracy = results[model_name]["med_type"]["top-3"]
            med_type_top5_accuracy = results[model_name]["med_type"]["top-5"]
            print(f"[training_utils] Med Type - Accuracy: {med_type_accuracy:.4f}, top-1: {med_type_top1_accuracy:.4f}, top-3: {med_type_top3_accuracy:.4f}, top-5: {med_type_top5_accuracy:.4f}")
        if args.do_quantity:
            quantity_mse_value = results[model_name]["quantity"]["mse"]
            quantity_pearsonr_value = results[model_name]["quantity"]["pearsonr"]
            quantity_spearmanr_value = results[model_name]["quantity"]["spearmanr"]
            print(f"[training_utils] Quantity - mse: {quantity_mse_value:.4f}, pearsonr: {quantity_pearsonr_value:.4f}, spearmanr: {quantity_spearmanr_value:.4f}")

    # return protocol_top1_accuracy, protocol_top3_accuracy, protocol_top5_accuracy, med_type_top1_accuracy, med_type_top3_accuracy, med_type_top5_accuracy, quantity_mse_value, quantity_pearsonr_value, quantity_spearmanr_value

def compute_topk_accuracy(logits, labels, k):
    _, topk_indices = torch.topk(logits, k, dim=-1)
    topk_correct = topk_indices.eq(labels.view(-1, 1).expand_as(topk_indices))
    topk_accuracy = topk_correct.sum().item() / labels.size(0)
    return topk_accuracy