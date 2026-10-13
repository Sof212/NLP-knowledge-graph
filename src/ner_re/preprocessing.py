import re
from collections import Counter

import pandas as pd


def load_spacy(model_name="en_core_web_sm", max_length=3_000_000, disable=None):
    """
    Load a spaCy pipeline.

    Parameters
    ----------
    model_name : str, default="en_core_web_sm"
        Installed spaCy model name.
    max_length : int, default=3_000_000
        Maximum number of characters spaCy accepts in one call.
    disable : list of str, optional
        Pipeline components to disable.

    Returns
    -------
    spacy.language.Language
        Loaded pipeline.
    """
    import spacy

    nlp = spacy.load(model_name, disable=disable or [])
    nlp.max_length = max_length
    return nlp


def load_ner_dataset(path):
    """
    Load the Kaggle NER dataset (``Sentence #, Word, POS, Tag``).

    The sentence id is forward-filled so every token row carries the id
    of its sentence, and rows with a missing word are dropped.

    Parameters
    ----------
    path : str
        Path to ``ner_dataset.csv`` (latin-1 encoded).

    Returns
    -------
    pandas.DataFrame
        Columns ``sentence_id``, ``word``, ``pos``, ``tag``.
    """
    df = pd.read_csv(path, encoding="latin1")
    df.columns = ["sentence_id", "word", "pos", "tag"]
    df["sentence_id"] = df["sentence_id"].ffill()
    df = df.dropna(subset=["word"]).reset_index(drop=True)
    df["word"] = df["word"].astype(str)
    return df


def build_text_sample(df, n_words=30_000):
    """
    Join the first words of the dataset into a raw text.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_ner_dataset`.
    n_words : int, default=30_000
        Number of words to keep.

    Returns
    -------
    str
        Space-joined text.
    """
    return " ".join(df["word"].iloc[:n_words].tolist())


def clean_text(text):
    """
    Lowercase, remove non-alphabetic characters and collapse whitespace.

    Parameters
    ----------
    text : str
        Raw text.

    Returns
    -------
    str
        Cleaned text.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def spacy_preprocess(text, nlp, clean=True):
    """
    Extract tokens, POS tags and lemmas (without stop words) with spaCy.

    Parameters
    ----------
    text : str
        Input text.
    nlp : spacy.language.Language
        Loaded spaCy pipeline.
    clean : bool, default=True
        Apply :func:`clean_text` first. Note that cleaning removes
        capitalisation and punctuation, which degrades POS tagging.

    Returns
    -------
    dict
        Keys ``tokens``, ``pos``, ``lemmas`` and ``clean_text``.
    """
    if clean:
        text = clean_text(text)
    doc = nlp(text)
    lemmas = [t.lemma_ for t in doc if not t.is_stop and not t.is_space]
    return {
        "tokens": [t.text for t in doc],
        "pos": [t.pos_ for t in doc],
        "lemmas": lemmas,
        "clean_text": " ".join(lemmas),
    }


def chunk_text(text, chunk_size=75_000):
    """
    Split a long text into fixed-size character chunks.

    Parameters
    ----------
    text : str
        Input text.
    chunk_size : int, default=75_000
        Chunk length in characters.

    Returns
    -------
    list of str
        Consecutive chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def extract_entities(text, nlp):
    """
    Run spaCy named entity recognition.

    Parameters
    ----------
    text : str
        Input text.
    nlp : spacy.language.Language
        Loaded spaCy pipeline with an ``ner`` component.

    Returns
    -------
    pandas.DataFrame
        One row per entity with columns ``text``, ``label``, ``start``
        and ``end`` (character offsets in the chunk).
    """
    rows = []
    for chunk in chunk_text(text):
        rows.extend({"text": e.text, "label": e.label_, "start": e.start_char,
                     "end": e.end_char} for e in nlp(chunk).ents)
    return pd.DataFrame(rows, columns=["text", "label", "start", "end"])


def top_verbs(text, nlp, top_n=150):
    """
    Return the most frequent verb lemmas of a text.

    Parameters
    ----------
    text : str
        Input text.
    nlp : spacy.language.Language
        Loaded spaCy pipeline.
    top_n : int, default=150
        Number of verbs to return.

    Returns
    -------
    list of tuple of (str, int)
        ``(lemma, count)`` pairs sorted by decreasing frequency.
    """
    counts = Counter()
    for chunk in chunk_text(text):
        counts.update(t.lemma_.lower() for t in nlp(chunk) if t.pos_ == "VERB")
    return counts.most_common(top_n)
