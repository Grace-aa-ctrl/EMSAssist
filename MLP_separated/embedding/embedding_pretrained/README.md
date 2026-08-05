# Pretrained Embedding

This directory contains scripts for dense pretrained embeddings on
`/home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt`.

Both scripts embed the same four fields and concatenate the four vectors:

```text
[PrimarySymptomCode, PrimaryImpressionCode, AdditionalSymptomCode, SecondaryImpressionCode]
```

The default models are:

```text
word_pretrained:  FremyCompany/BioLORD-2023
icd10_pretrained: emilyalsentzer/Bio_ClinicalBERT
```

You can replace either with any HuggingFace `AutoModel` model via `--model_name`.

## Dependencies

The current environment must have:

```bash
pip install torch transformers
```

If the model is already cached locally, pass `--local_files_only`.

## Word/Text Embedding

This script converts ICD-10 codes to text descriptions using
`nemsis_processor.get_dict()` / `global_d`, embeds each field's text, and
concatenates the four field embeddings. Its default, `FremyCompany/BioLORD-2023`,
is trained for clinical sentence and biomedical concept similarity. Following
the model's recommended Transformers inference recipe, embeddings use masked
mean pooling followed by L2 normalization.

```bash
cd /home/xiangling/EMSAssist
python embedding/embedding_pretrained/embed_word_pretrained.py
```

Output:

```text
word_pretrained_embeddings.npy
word_pretrained_metadata.json
word_pretrained_pcr_keys.txt
word_pretrained_protocol_labels.txt
```

For a model with hidden size `768`, the output shape is:

```text
(number_of_rows, 4 * 768)
```

## ICD-10 Embedding

This path is unchanged and continues to default to
`emilyalsentzer/Bio_ClinicalBERT` with masked mean pooling.

This script first creates one pretrained vector for every ICD-10 code in the
dataset. Each code is embedded as:

```text
ICD-10 code {code}: {description}
```

Then, for each PCR row and each symptom/impression field, if the field contains
multiple ICD-10 codes, their code vectors are averaged. The four field vectors
are concatenated.

```bash
cd /home/xiangling/EMSAssist
python embedding/embedding_pretrained/embed_icd10_pretrained.py
```

Output:

```text
icd10_pretrained_embeddings.npy
icd10_pretrained_metadata.json
icd10_pretrained_code_embedding_table.npy
icd10_pretrained_code_vocab_metadata.json
icd10_pretrained_pcr_keys.txt
icd10_pretrained_protocol_labels.txt
```

## Example With GPU

```bash
python embedding/embedding_pretrained/embed_word_pretrained.py --device cuda --batch_size 128
python embedding/embedding_pretrained/embed_icd10_pretrained.py --device cuda --batch_size 256
```

## Output Format

The `.npy` embedding files are dense `float32` arrays written with NumPy memmap.
They can be loaded with:

```python
import numpy as np

x = np.load("/home/xiangling/EMSAssist/embedding/embedding_pretrained/word_pretrained_embeddings.npy", mmap_mode="r")
print(x.shape)
```
