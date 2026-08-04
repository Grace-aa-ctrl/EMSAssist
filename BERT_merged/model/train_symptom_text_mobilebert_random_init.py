"""Train a randomly initialized MobileBERT on the symptom-text dataset.

This entry point reuses the same dataset loading, tokenizer, hyperparameter
tuning, training, and evaluation pipeline as ``train_symptom_text_bert.py``.
Only the model weights are randomly initialized from the official MobileBERT
configuration instead of being loaded from a pretrained checkpoint.
"""

import sys

import train_symptom_text_bert
from transformers import AutoConfig, AutoModelForSequenceClassification


DEFAULT_ARGUMENTS = {
    "--model_name": "google/mobilebert-uncased",
    "--model_path": (
        "/home/xiangling/EMSAssist/BERT_merged/model/"
        "symptom_text_mobilebert_random_init.pt"
    ),
    "--evaluation_path": (
        "/home/xiangling/EMSAssist/BERT_merged/evaluation/"
        "symptom_text_mobilebert_random_init_metrics.txt"
    ),
}


def add_defaults(argv: list[str]) -> None:
    """Add MobileBERT defaults without overriding explicit CLI arguments."""
    supplied_options = {argument.split("=", 1)[0] for argument in argv[1:]}
    for option, value in DEFAULT_ARGUMENTS.items():
        if option not in supplied_options:
            argv.extend((option, value))


def new_random_model(args, num_classes):
    """Build a classifier with random weights from the model configuration."""
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        local_files_only=args.local_files_only,
    )
    return AutoModelForSequenceClassification.from_config(config)


if __name__ == "__main__":
    add_defaults(sys.argv)
    train_symptom_text_bert.new_model = new_random_model
    train_symptom_text_bert.main()
