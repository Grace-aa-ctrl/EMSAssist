"""Tune and evaluate an MLP using the pretrained ICD-10 vector split."""

import sys

from train_pretrained_word_mlp import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--dataset_dir", "/home/xiangling/EMSAssist/MLP_merged/dataset/pretrained_icd10",
        "--model_path", "/home/xiangling/EMSAssist/MLP_merged/model/pretrained_icd10_mlp.pt",
        "--evaluation_path", "/home/xiangling/EMSAssist/MLP_merged/evaluation/pretrained_icd10_mlp_metrics.txt",
        "--report_title", "Pretrained ICD-10 MLP Evaluation",
    ]
    main()
