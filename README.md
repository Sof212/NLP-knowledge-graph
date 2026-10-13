# Named Entity Recognition & Relation Extraction

An end-to-end information extraction project in English: from raw text to entities, relations
and a knowledge graph. Classic and neural approaches are compared on the same pipeline:

| Step | Approaches |
|------|-----------|
| **A. Preprocessing & representations** | text cleaning, spaCy tokens / POS / lemmas, Word2Vec, BERT contextual embeddings |
| **B. Named entity recognition** | spaCy pretrained NER, **linear-chain CRF** with hand-crafted features (BIO tags) |
| **C. Relation extraction** | **rule-based** dependency patterns + heuristic coreference → knowledge graph; supervised **BERT** and **BiLSTM + entity-aware attention** classifiers |

## Results

> Numbers below come from the original exploratory notebook. The refactored code fixes several
> bugs and uses leakage-free splits (see [Changes from the notebook](#changes-from-the-original-notebook)),
> so re-run the scripts and update this table with your new figures.

| Task | Dataset | Model | Metric | Score |
|------|---------|-------|--------|------:|
| NER | Kaggle NER (47,959 sentences, 17 BIO tags) | CRF | entity-level F1 (P / R) | **0.851** (0.861 / 0.841) |
| Relation extraction | 30k words of the NER corpus | Rule-based (spaCy dependencies) | triples extracted | 22 → **40** with coreference |
| Relation classification | NYT (24 Freebase relations) | BERT-base (no entity information) | val. accuracy | 0.660 |
| Relation classification | DocRED distant (300k pairs, 96 Wikidata relations) | BiLSTM + attention | macro-F1 / micro-F1 | 0.468 / 0.516 |

<!-- Add your figures (outputs/*/confusion_matrix.png, knowledge_graph_*.png) to docs/ and show them here. -->

## Project structure

```
ner-relation-extraction/
├── configs/
│   ├── crf.yaml                 # NER with CRF
│   ├── bert_re.yaml             # BERT relation classifier
│   └── bilstm_re.yaml           # BiLSTM relation classifier
├── data/README.md               # where to download each dataset (data is git-ignored)
├── notebooks/demo.ipynb         # guided tour of the package (Colab-ready)
├── scripts/
│   ├── explore_preprocessing.py # Part A: cleaning, spaCy, Word2Vec, BERT embeddings
│   ├── train_crf.py             # Part B: CRF training + entity-level evaluation
│   ├── extract_knowledge_graph.py   # Part C: rule-based triples (+ coreference) → graph
│   ├── train_relation_classifier.py # Part C: BERT or BiLSTM relation classifier
│   └── predict_relation.py      # inference on a sentence and two entities
├── src/ner_re/
│   ├── preprocessing.py         # dataset loading, cleaning, spaCy utilities
│   ├── embeddings.py            # Word2Vec and BERT embeddings
│   ├── crf.py                   # CRF features, training, seqeval metrics
│   ├── rule_relations.py        # dependency-based triples, knowledge graph
│   ├── coreference.py           # heuristic pronoun resolution
│   ├── re_data.py               # NYT / DocRED loaders, entity markers, group splits
│   ├── bert_re.py               # BERT + entity markers
│   ├── bilstm_re.py             # BiLSTM + entity-aware attention
│   ├── labels.py                # NER tags and Wikidata property names
│   ├── visualization.py         # confusion matrices, graphs, curves
│   └── utils.py
├── tests/                       # pytest unit tests
├── outputs/                     # models, metrics, figures (git-ignored)
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/<your-username>/ner-relation-extraction.git
cd ner-relation-extraction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

A GPU is recommended for the BERT and BiLSTM classifiers.

## Usage

Download the datasets as described in [`data/README.md`](data/README.md), then:

```bash
# Part A - preprocessing and embeddings
python scripts/explore_preprocessing.py --bert

# Part B - NER with a CRF
python scripts/train_crf.py --config configs/crf.yaml

# Part C - rule-based relation extraction and knowledge graph
python scripts/extract_knowledge_graph.py
python scripts/extract_knowledge_graph.py --coref

# Part C - supervised relation classification
python scripts/train_relation_classifier.py --model bert
python scripts/train_relation_classifier.py --model bilstm

# Inference
python scripts/predict_relation.py --model bert --model-path outputs/bert_re/best_model \
    --sentence "Barack Obama was born in Honolulu ." --e1 "Barack Obama" --e2 "Honolulu"
```

Every script writes its metrics (JSON), model and figures to `outputs/`. Run the tests with `pytest`.

## Methods

**CRF NER.** Each token is described by its lowercased form, casing and digit flags, POS tag,
prefixes/suffixes (1–3 characters) and the same features for a ±2 token window. A linear-chain
CRF (L-BFGS) learns tag transitions so that BIO sequences stay consistent. Evaluation is done at
entity level with `seqeval`.

**Rule-based relations.** spaCy's dependency parse gives, for each frequent verb, its nominal
subject and its direct / prepositional objects. A triple *(subject, verb, object)* is kept when
the subject is a PERSON or ORG and the object a PERSON, ORG, GPE or LOC; both are expanded to the
full entity span. A heuristic coreference step first replaces pronouns by the most recent
compatible entity, which roughly doubles the number of extracted triples.

**Supervised relation classification.** Entities are wrapped with markers
(`[E1] Obama [/E1] was born in [E2] Hawaii [/E2]`).
- *BERT*: the markers are added as special tokens and the `[CLS]` representation is classified.
- *BiLSTM*: a 3-layer BiLSTM encodes the sentence; an attention layer (with a boost on entity
  markers and padding masked) pools a sentence vector, concatenated with the hidden states at
  `[E1]` and `[E2]`. Class weights compensate for the strong imbalance of DocRED (e.g. `P17`
  *country* dominates).

Splits are made **by sentence (NYT) or by document (DocRED)** so the same text never appears in
both training and validation.

## Changes from the original notebook

The exploratory notebook was refactored into a tested package. Along the way:

- **BiLSTM entity markers were never detected**: the regex tokenizer split `<E1>` into `<`, `e1`, `>`,
  so the entity representations were always zero vectors and the attention boost had no effect.
  Markers are now kept as single tokens (unit-tested).
- **BiLSTM inference used a different tokenizer and length** (`str.split`, 100 tokens) than training
  (regex, 120 tokens). Training and inference now share the same encoding.
- **BERT ignored the entities during training** (positions were computed but never used) while
  inference added markers the model had never seen. Markers are now used consistently, which is
  the main expected gain over the 0.66 accuracy.
- **Data leakage**: random row splits put the same sentence/document in train and validation.
  Splits are now grouped by sentence or document, and the vocabulary and class weights are fitted
  on the training split only.
- Padding is masked in the BiLSTM attention; long contexts are cropped around the entities.
- CRF sentences come from the dataset's sentence ids instead of punctuation splitting; the split
  is seeded.
- Rule-based triples use full entity spans (`New York` instead of `York`); verbs are computed from
  the text instead of a hard-coded list; coreference parses the text once instead of three times.

## Datasets and credits

- [Kaggle – Named Entity Recognition (NER) dataset](https://www.kaggle.com/datasets/debasisdotcom/name-entity-recognition-ner-dataset)
- NYT relation extraction dataset (Riedel et al., 2010; JSON version used in CopyRE, Zeng et al., 2018)
- [DocRED](https://github.com/thunlp/DocRED) (Yao et al., *DocRED: A Large-Scale Document-Level Relation Extraction Dataset*, ACL 2019)

Project carried out as part of the *Master IA – Vision et Machines Intelligentes* (Université Paris Cité).
