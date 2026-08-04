"""Train the officially released MobileBERT on the symptom-text dataset.

This entry point reuses the same dataset loading, hyperparameter tuning,
training, and evaluation pipeline as ``train_symptom_text_bert.py`` while
selecting google/mobilebert-uncased and separate output files by default.
"""

import sys

from train_symptom_text_bert import main


DEFAULT_ARGUMENTS = {
    "--model_name": "google/mobilebert-uncased",
    "--model_path": "/home/xiangling/EMSAssist/BERT_merged/model/symptom_text_mobilebert.pt",
    "--evaluation_path": (
        "/home/xiangling/EMSAssist/BERT_merged/evaluation/symptom_text_mobilebert_metrics.txt"
    ),
}


def add_defaults(argv: list[str]) -> None:
    """Add MobileBERT defaults without overriding explicit CLI arguments."""
    supplied_options = {argument.split("=", 1)[0] for argument in argv[1:]}
    for option, value in DEFAULT_ARGUMENTS.items():
        if option not in supplied_options:
            argv.extend((option, value))


if __name__ == "__main__":
    add_defaults(sys.argv)
    main()
