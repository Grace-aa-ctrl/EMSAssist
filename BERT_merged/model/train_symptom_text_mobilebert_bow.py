"""Train MobileBERT using a one-hot bag-of-words-equivalent input encoding.

The four symptom/impression fields of each sample are treated as one sentence.
Token counts are projected through MobileBERT's pretrained word-embedding matrix
and mean-pooled before being passed to MobileBERT as a one-token sequence.
"""

import sys

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

import train_symptom_text_bert as training


class BagOfWordsTextDataset(training.TextDataset):
    """Return the four text fields as a single sentence."""

    def __getitem__(self, index):
        row = self.rows[index]
        return " ".join(row[1:5]), int(self.labels[index])


class BagOfWordsCollator:
    """Tokenize words for an order-independent count bag."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        texts, labels = zip(*samples)
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded


class BagOfWordsMobileBert(nn.Module):
    """Feed a normalized token-count bag through MobileBERT embeddings.

    For a count vector ``bow`` and embedding matrix ``E``, the pooled vector is
    ``bow @ E / bow.sum()``.  Gathering token embeddings and taking their masked
    mean computes exactly the same value without materializing a large dense
    one-hot tensor.
    """

    def __init__(self, model_name, num_labels, local_files_only):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            local_files_only=local_files_only,
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **_unused):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        token_embeddings = self.model.get_input_embeddings()(input_ids)
        weights = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        bag_embedding = (token_embeddings * weights).sum(dim=1)
        bag_embedding = bag_embedding / weights.sum(dim=1).clamp_min(1.0)

        bag_embedding = bag_embedding.unsqueeze(1)
        bag_attention_mask = torch.ones(
            (input_ids.shape[0], 1), dtype=attention_mask.dtype, device=input_ids.device
        )
        return self.model(
            inputs_embeds=bag_embedding,
            attention_mask=bag_attention_mask,
            labels=labels,
        )


def new_model(args, num_classes):
    return BagOfWordsMobileBert(
        args.model_name,
        num_labels=num_classes,
        local_files_only=args.local_files_only,
    )


DEFAULT_ARGUMENTS = {
    "--model_name": "google/mobilebert-uncased",
    "--model_path": "/home/xiangling/EMSAssist/BERT_merged/model/symptom_text_mobilebert_bow.pt",
    "--evaluation_path": (
        "/home/xiangling/EMSAssist/BERT_merged/evaluation/symptom_text_mobilebert_bow_metrics.txt"
    ),
}


def add_defaults(argv):
    supplied_options = {argument.split("=", 1)[0] for argument in argv[1:]}
    for option, value in DEFAULT_ARGUMENTS.items():
        if option not in supplied_options:
            argv.extend((option, value))


if __name__ == "__main__":
    training.TextDataset = BagOfWordsTextDataset
    training.TextCollator = BagOfWordsCollator
    training.new_model = new_model
    add_defaults(sys.argv)
    training.main()
