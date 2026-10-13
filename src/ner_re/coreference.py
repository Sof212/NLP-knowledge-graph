import re

PERSON_PRONOUNS = ("he", "she", "him", "her", "they", "them")
PERSON_POSSESSIVES = ("his", "their")
ORG_PRONOUNS = ("it", "they", "them")
ORG_POSSESSIVES = ("its", "their")
LOC_PRONOUNS = ("it",)
LOC_POSSESSIVES = ("its",)


def _case_variants(mapping):
    """
    Add capitalised variants to a pronoun replacement mapping.

    Parameters
    ----------
    mapping : dict
        ``pronoun -> replacement`` with lowercase pronouns.

    Returns
    -------
    dict
        Mapping containing both ``"he"`` and ``"He"`` style keys.
    """
    return {**mapping, **{k.capitalize(): v for k, v in mapping.items()}}


def pronoun_replacements(entities):
    """
    Choose which pronouns to replace and by which entity.

    The priority is PERSON > ORG > GPE/LOC: the first entity of the
    highest-priority type present becomes the antecedent (two persons
    are joined with ``"and"``).

    Parameters
    ----------
    entities : list of tuple of (str, str)
        ``(text, label)`` of the candidate antecedents, oldest first.

    Returns
    -------
    dict
        ``pronoun -> replacement``; empty if no antecedent is found.
    """
    persons = [t for t, lab in entities if lab == "PERSON"]
    orgs = [t for t, lab in entities if lab == "ORG"]
    locations = [t for t, lab in entities if lab in ("GPE", "LOC")]
    if persons:
        ref = " and ".join(persons[:2])
        pronouns, possessives = PERSON_PRONOUNS, PERSON_POSSESSIVES
    elif orgs:
        ref, pronouns, possessives = orgs[0], ORG_PRONOUNS, ORG_POSSESSIVES
    elif locations:
        ref, pronouns, possessives = locations[0], LOC_PRONOUNS, LOC_POSSESSIVES
    else:
        return {}
    mapping = {p: ref for p in pronouns}
    mapping.update({p: f"{ref}'s" for p in possessives})
    return _case_variants(mapping)


def replace_pronouns(sentence, replacements):
    """
    Replace whole-word pronouns in a sentence.

    Parameters
    ----------
    sentence : str
        Input sentence.
    replacements : dict
        ``pronoun -> replacement``.

    Returns
    -------
    str
        Sentence with pronouns substituted.
    """
    if not replacements:
        return sentence
    pattern = r"\b(" + "|".join(re.escape(p) for p in replacements) + r")\b"
    return re.sub(pattern, lambda m: replacements[m.group(0)], sentence)


def resolve_document(text, nlp, context_window=3, entities_per_sentence=5):
    """
    Heuristic, rule-based coreference resolution.

    The text is parsed once; each sentence is processed with the named
    entities of the previous sentences (sliding window) plus its own, and
    personal / possessive pronouns are replaced by the chosen antecedent.
    Example: ``"Obama gave a speech. He named Clinton."`` becomes
    ``"Obama gave a speech. Obama named Clinton."``.

    This is a lightweight baseline: it ignores gender and number
    agreement and should not be used where precision matters.

    Parameters
    ----------
    text : str
        Input document.
    nlp : spacy.language.Language
        spaCy pipeline with ``parser`` (or ``senter``) and ``ner``.
    context_window : int, default=3
        Approximate number of previous sentences kept as context.
    entities_per_sentence : int, default=5
        Used to convert ``context_window`` into a number of entities.

    Returns
    -------
    str
        Resolved document.
    """
    doc = nlp(text)
    recent, resolved = [], []
    for sent in doc.sents:
        current = [(e.text, e.label_) for e in sent.ents]
        resolved.append(replace_pronouns(sent.text.strip(),
                                         pronoun_replacements(recent + current)))
        if current:
            recent = (recent + current)[-context_window * entities_per_sentence:]
    return " ".join(resolved)
