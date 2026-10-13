import pickle

from sklearn.model_selection import train_test_split


def group_sentences(df):
    """
    Group the token-level NER dataframe into sentences.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`ner_re.preprocessing.load_ner_dataset`.

    Returns
    -------
    sentences : list of list of tuple of (str, str)
        ``(word, pos)`` pairs for each sentence.
    tags : list of list of str
        BIO tags aligned with ``sentences``.
    """
    sentences, tags = [], []
    for _, group in df.groupby("sentence_id", sort=False):
        sentences.append(list(zip(group["word"], group["pos"])))
        tags.append(group["tag"].tolist())
    return sentences, tags


def word2features(sent, i):
    """
    Build the hand-crafted CRF features of one token.

    Features include the lowercased word, casing and digit flags, POS,
    prefixes and suffixes up to length 3, and the same information for a
    [-2, +2] context window (with sentence boundary markers).

    Parameters
    ----------
    sent : list of tuple of (str, str)
        ``(word, pos)`` pairs of the sentence.
    i : int
        Index of the token.

    Returns
    -------
    dict
        Feature name to value.
    """
    word, pos = sent[i]
    features = {
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "pos": pos,
        "prefix-1": word[:1], "prefix-2": word[:2], "prefix-3": word[:3],
        "suffix-1": word[-1:], "suffix-2": word[-2:], "suffix-3": word[-3:],
        "has_hyphen": "-" in word,
        "has_digit": any(c.isdigit() for c in word),
        "has_upper_inside": any(c.isupper() for c in word[1:]),
    }
    for offset in (-2, -1, 1, 2):
        j = i + offset
        if 0 <= j < len(sent):
            w, p = sent[j]
            features[f"{offset}:word.lower()"] = w.lower()
            features[f"{offset}:word.istitle()"] = w.istitle()
            features[f"{offset}:word.isupper()"] = w.isupper()
            features[f"{offset}:pos"] = p
        elif offset < 0:
            features[f"BOS_{-offset}"] = True
        else:
            features[f"EOS_{offset}"] = True
    return features


def sent2features(sent):
    """
    Build the features of every token of a sentence.

    Parameters
    ----------
    sent : list of tuple of (str, str)
        ``(word, pos)`` pairs.

    Returns
    -------
    list of dict
        One feature dict per token.
    """
    return [word2features(sent, i) for i in range(len(sent))]


def split_dataset(sentences, tags, test_size=0.2, seed=42):
    """
    Featurise sentences and split them into train and test sets.

    Parameters
    ----------
    sentences : list of list of tuple of (str, str)
        Sentences from :func:`group_sentences`.
    tags : list of list of str
        Aligned BIO tags.
    test_size : float, default=0.2
        Fraction of sentences kept for testing.
    seed : int, default=42
        Random seed of the split.

    Returns
    -------
    X_train, X_test : list of list of dict
        Featurised sentences.
    y_train, y_test : list of list of str
        Aligned tags.
    """
    X = [sent2features(s) for s in sentences]
    return train_test_split(X, tags, test_size=test_size, random_state=seed)


def train_crf(X_train, y_train, c1=0.0, c2=1.0, max_iterations=100):
    """
    Train a linear-chain CRF with L-BFGS.

    Parameters
    ----------
    X_train : list of list of dict
        Featurised training sentences.
    y_train : list of list of str
        Training tags.
    c1 : float, default=0.0
        L1 regularisation coefficient (CRFsuite default).
    c2 : float, default=1.0
        L2 regularisation coefficient (CRFsuite default).
    max_iterations : int, default=100
        Maximum number of optimisation iterations.

    Returns
    -------
    sklearn_crfsuite.CRF
        Fitted model.
    """
    import sklearn_crfsuite

    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2,
                               max_iterations=max_iterations,
                               all_possible_transitions=True)
    crf.fit(X_train, y_train)
    return crf


def evaluate_ner(y_true, y_pred):
    """
    Compute entity-level NER metrics with seqeval.

    Parameters
    ----------
    y_true : list of list of str
        Gold BIO tags.
    y_pred : list of list of str
        Predicted BIO tags.

    Returns
    -------
    dict
        ``precision``, ``recall``, ``f1`` (micro, entity-level) and the
        full text ``report``.
    """
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, digits=3),
    }


def save_crf(crf, path):
    """
    Save a fitted CRF with pickle.

    Parameters
    ----------
    crf : sklearn_crfsuite.CRF
        Fitted model.
    path : str
        Output file path.

    Returns
    -------
    None
    """
    with open(path, "wb") as f:
        pickle.dump(crf, f)


def load_crf(path):
    """
    Load a CRF saved with :func:`save_crf`.

    Parameters
    ----------
    path : str
        Model file path.

    Returns
    -------
    sklearn_crfsuite.CRF
        Fitted model.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
