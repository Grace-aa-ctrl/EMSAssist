"""Create the stratified 70/10/20 split for pretrained ICD-10 embeddings."""

import sys

from split_pretrained_word_dataset import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--embedding_dir", "/home/xiangling/EMSAssist/MLP_merged/embedding/embedding_pretrained/icd10_pretrained_output",
        "--embedding_prefix", "icd10_pretrained",
        "--output_dir", "/home/xiangling/EMSAssist/MLP_merged/dataset/pretrained_icd10",
    ]
    main()
