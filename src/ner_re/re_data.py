import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ENTITY_MARKERS = ("[E1]", "[/E1]", "[E2]", "[/E2]")


def load_nyt(path):
    """
    Load the NYT relation extraction dataset (JSON lines).

    Each line holds ``sentText`` and a list of ``relationMentions`` with
    ``em1Text``, ``em2Text`` and a Freebase ``label`` such as
    ``/people/person/place_of_birth``.

    Parameters
    ----------
    path : str
        Path to the JSON-lines file.

    Returns
    -------
    pandas.DataFrame
        Columns ``sentence``, ``entity1``, ``entity2``, ``label``; labels
        are normalised to ``people_person_place_of_birth``.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for rel in item.get("relationMentions", []):
                rows.append({
                    "sentence": item["sentText"],
                    "entity1": rel["em1Text"],
                    "entity2": rel["em2Text"],
                    "label": rel["label"].lstrip("/").replace("/", "_"),
                })
    return pd.DataFrame(rows, columns=["sentence", "entity1", "entity2", "label"]).astype(str)


def load_docred(path, fallback="mentions", max_examples=None):
    """
    Load a DocRED file as sentence-level relation examples.

    For each labelled pair, the context is the concatenation of the
    evidence sentences. When no evidence is given, ``fallback`` decides
    the context: the sentences mentioning the head or tail entity
    (``"mentions"``) or the whole document (``"document"``).

    Parameters
    ----------
    path : str
        Path to ``train_annotated.json``, ``train_distant.json`` or
        ``dev.json``.
    fallback : {"mentions", "document"}, default="mentions"
        Context used when a relation has no evidence.
    max_examples : int, optional
        Stop after this many examples.

    Returns
    -------
    pandas.DataFrame
        Columns ``doc_id``, ``sentence``, ``entity1``, ``entity2``,
        ``label`` (Wikidata property id).
    """
    with open(path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    rows = []
    for doc_id, doc in enumerate(documents):
        sents = [" ".join(s) for s in doc["sents"]]
        names = [cluster[0]["name"] for cluster in doc["vertexSet"]]
        for lab in doc.get("labels", []):
            head, tail = names[lab["h"]], names[lab["t"]]
            evidence = [i for i in lab.get("evidence", []) if i < len(sents)]
            if evidence:
                context = [sents[i] for i in evidence]
            elif fallback == "document":
                context = sents
            else:
                context = [s for s in sents if head in s or tail in s]
            if not context:
                continue
            rows.append({"doc_id": doc_id, "sentence": " ".join(context),
                         "entity1": head, "entity2": tail, "label": lab["r"]})
            if max_examples and len(rows) >= max_examples:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=["doc_id", "sentence", "entity1", "entity2", "label"])


def load_relation_dataset(name, path, max_examples=None, fallback="mentions"):
    """
    Load a relation dataset and return the column to group splits by.

    Parameters
    ----------
    name : {"nyt", "docred"}
        Dataset format.
    path : str
        Path to the data file.
    max_examples : int, optional
        Keep only the first examples (useful for quick experiments).
    fallback : {"mentions", "document"}, default="mentions"
        DocRED context when a relation has no evidence.

    Returns
    -------
    df : pandas.DataFrame
        Columns ``sentence``, ``entity1``, ``entity2``, ``label`` (plus
        ``doc_id`` for DocRED).
    group_col : str
        ``"doc_id"`` for DocRED, ``"sentence"`` for NYT.
    """
    if name == "docred":
        return load_docred(path, fallback=fallback, max_examples=max_examples), "doc_id"
    if name == "nyt":
        df = load_nyt(path)
        return (df.head(max_examples) if max_examples else df), "sentence"
    raise ValueError(f"Unknown dataset '{name}', expected 'nyt' or 'docred'.")


def mark_entities(sentence, entity1, entity2, markers=ENTITY_MARKERS):
    """
    Surround the first non-overlapping mentions of two entities with markers.

    Parameters
    ----------
    sentence : str
        Input sentence.
    entity1 : str
        Head entity.
    entity2 : str
        Tail entity.
    markers : tuple of str, default=ENTITY_MARKERS
        ``(e1_start, e1_end, e2_start, e2_end)``.

    Returns
    -------
    str
        Sentence such as ``"[E1] Obama [/E1] visited [E2] Paris [/E2]"``.
        An entity that cannot be found is left unmarked.
    """
    spans = []
    s1 = sentence.find(entity1) if entity1 else -1
    if s1 >= 0:
        spans.append((s1, s1 + len(entity1), markers[0], markers[1]))
    start = 0
    while entity2:
        s2 = sentence.find(entity2, start)
        if s2 < 0:
            break
        e2 = s2 + len(entity2)
        if not spans or e2 <= spans[0][0] or s2 >= spans[0][1]:
            spans.append((s2, e2, markers[2], markers[3]))
            break
        start = s2 + 1
    for s, e, open_m, close_m in sorted(spans, reverse=True):
        sentence = f"{sentence[:s]}{open_m} {sentence[s:e]} {close_m}{sentence[e:]}"
    return sentence


def add_marked_sentences(df):
    """
    Add a ``marked`` column with entity markers to a relation dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``sentence``, ``entity1`` and ``entity2``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the new column.
    """
    df = df.copy()
    df["marked"] = [mark_entities(s, e1, e2)
                    for s, e1, e2 in zip(df["sentence"], df["entity1"], df["entity2"])]
    return df


def group_split(df, group_col, test_size=0.2, seed=42):
    """
    Split a dataframe so that a group never appears in both sets.

    Splitting by sentence or document avoids leakage: the same text with
    another entity pair would otherwise be seen during training.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to split.
    group_col : str
        Column identifying groups (e.g. ``"sentence"`` or ``"doc_id"``).
    test_size : float, default=0.2
        Fraction of groups in the test set.
    seed : int, default=42
        Random seed.

    Returns
    -------
    train_df, test_df : pandas.DataFrame
        The two splits, with reset indices.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def build_label_maps(labels):
    """
    Build label/index lookup tables.

    Parameters
    ----------
    labels : iterable of str
        Labels (duplicates allowed).

    Returns
    -------
    label2id : dict
        ``label -> index`` (labels sorted alphabetically).
    id2label : dict
        ``index -> label``.
    """
    unique = sorted(set(labels))
    return {lab: i for i, lab in enumerate(unique)}, dict(enumerate(unique))
