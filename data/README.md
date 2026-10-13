# Data

Datasets are **not** versioned in this repository. Download them and arrange them as follows:

```
data/
├── ner_dataset.csv              # Kaggle NER dataset
├── nyt/
│   └── train.json               # NYT relation extraction, JSON lines
└── docred/
    ├── train_annotated.json
    └── train_distant.json
```

## Kaggle NER dataset

https://www.kaggle.com/datasets/debasisdotcom/name-entity-recognition-ner-dataset

A latin-1 CSV with columns `Sentence #, Word, POS, Tag` (about 1M tokens, 47,959 sentences,
17 BIO tags: `geo, gpe, per, org, tim, art, nat, eve`). Used by Parts A, B and the rule-based
part of C.

## NYT relation extraction

One JSON object per line:

```json
{"sentText": "...", "relationMentions": [{"em1Text": "...", "em2Text": "...", "label": "/location/location/contains"}]}
```

This is the format of the NYT dataset distributed with CopyRE (Zeng et al., 2018) and several
joint extraction repositories. Point `data_path` in `configs/bert_re.yaml` to the file.

## DocRED

https://github.com/thunlp/DocRED (files on the project's Google Drive / Hugging Face mirror).
`train_distant.json` (~1.5M distantly supervised pairs) is large: `max_examples` in
`configs/bilstm_re.yaml` limits how many pairs are loaded.
