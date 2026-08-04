"""Train a randomly initialized MobileBERT with four separate BoW segments."""

import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification


MERGED_MODEL_DIR = Path("/home/xiangling/EMSAssist/BERT_merged/model")
if str(MERGED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MERGED_MODEL_DIR))

import train_symptom_text_bert as training


class SeparateBagOfWordsTextDataset(training.TextDataset):
    """Return the four symptom/impression fields as separate text segments."""

    def __getitem__(self, index):
        row = self.rows[index]
        return tuple(row[1:5]), int(self.labels[index])


class SeparateBagOfWordsCollator:
    """Tokenize each of the four fields independently for count bags."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        segments, labels = zip(*samples)
        flat_segments = [segment for sample in segments for segment in sample]
        encoded = self.tokenizer(
            flat_segments,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_size = len(samples)
        result = {
            "input_ids": encoded["input_ids"].reshape(batch_size, 4, -1),
            "attention_mask": encoded["attention_mask"].reshape(batch_size, 4, -1),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return result


class SeparateBagOfWordsMobileBert(nn.Module):
    """Project four count bags and concatenate them as a four-vector sequence.

    Each segment computes ``bow @ E / bow.sum()`` independently. Gathering and
    masked-averaging embeddings is algebraically equivalent and avoids creating
    a dense ``batch_size x 4 x vocabulary_size`` one-hot tensor.
    """

    def __init__(self, model_name, num_labels, local_files_only):
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_config(config)

    def forward(self, input_ids, attention_mask=None, labels=None, **_unused):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        token_embeddings = self.model.get_input_embeddings()(input_ids)
        weights = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        segment_embeddings = (token_embeddings * weights).sum(dim=2)
        segment_embeddings = segment_embeddings / weights.sum(dim=2).clamp_min(1.0)

        segment_attention_mask = torch.ones(
            input_ids.shape[:2], dtype=attention_mask.dtype, device=input_ids.device
        )
        return self.model(
            inputs_embeds=segment_embeddings,
            attention_mask=segment_attention_mask,
            labels=labels,
        )


def new_model(args, num_classes):
    return SeparateBagOfWordsMobileBert(
        args.model_name,
        num_labels=num_classes,
        local_files_only=args.local_files_only,
    )


DEFAULT_ARGUMENTS = {
    "--dataset_dir": (
        "/home/xiangling/EMSAssist/BERT_separated/dataset/symptom_text"
    ),
    "--model_name": "google/mobilebert-uncased",
    "--model_path": (
        "/home/xiangling/EMSAssist/BERT_separated/model/"
        "symptom_text_mobilebert_separate_bow_random_init.pt"
    ),
    "--evaluation_path": (
        "/home/xiangling/EMSAssist/BERT_separated/evaluation/"
        "symptom_text_mobilebert_separate_bow_random_init_metrics.txt"
    ),
}


def add_defaults(argv):
    supplied_options = {argument.split("=", 1)[0] for argument in argv[1:]}
    for option, value in DEFAULT_ARGUMENTS.items():
        if option not in supplied_options:
            argv.extend((option, value))


if __name__ == "__main__":
    training.TextDataset = SeparateBagOfWordsTextDataset
    training.TextCollator = SeparateBagOfWordsCollator
    training.new_model = new_model
    add_defaults(sys.argv)
    training.main()
