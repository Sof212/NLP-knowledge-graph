from ner_re.preprocessing import chunk_text, top_verbs

SUBJECT_DEPS = ("nsubj", "nsubjpass")
OBJECT_DEPS = ("dobj", "attr", "pobj", "dative")
DEFAULT_SUBJECT_TYPES = ("PERSON", "ORG")
DEFAULT_OBJECT_TYPES = ("PERSON", "ORG", "GPE", "LOC")


def token_to_entity_map(doc):
    """
    Map every token index to the named entity span that contains it.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed document.

    Returns
    -------
    dict
        ``token_index -> spacy.tokens.Span``.
    """
    return {i: ent for ent in doc.ents for i in range(ent.start, ent.end)}


def extract_relations_doc(doc, verbs, subject_types=DEFAULT_SUBJECT_TYPES,
                          object_types=DEFAULT_OBJECT_TYPES):
    """
    Extract ``(subject, verb, object)`` triples from a parsed document.

    A triple is kept when a verb from ``verbs`` has a nominal subject that
    is a named entity of ``subject_types`` and a direct, attribute,
    dative or prepositional object that is a named entity of
    ``object_types``. Subjects and objects are expanded to their full
    entity span (``"New York"`` instead of ``"York"``).

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Document parsed with a pipeline that includes ``parser`` and
        ``ner``.
    verbs : set of str
        Verb lemmas to consider.
    subject_types : tuple of str
        Allowed entity labels for subjects.
    object_types : tuple of str
        Allowed entity labels for objects.

    Returns
    -------
    list of tuple of (str, str, str)
        Extracted triples.
    """
    ent_of = token_to_entity_map(doc)
    triples = []
    for token in doc:
        if token.pos_ != "VERB" or token.lemma_.lower() not in verbs:
            continue
        subjects = [c for c in token.children
                    if c.dep_ in SUBJECT_DEPS and c.ent_type_ in subject_types]
        if not subjects:
            continue
        objects = [c for c in token.children
                   if c.dep_ in OBJECT_DEPS and c.ent_type_ in object_types]
        for prep in (c for c in token.children if c.dep_ == "prep"):
            objects.extend(o for o in prep.children if o.ent_type_ in object_types)
        for subj in subjects:
            for obj in objects:
                s_text = ent_of[subj.i].text if subj.i in ent_of else subj.text
                o_text = ent_of[obj.i].text if obj.i in ent_of else obj.text
                if s_text != o_text:
                    triples.append((s_text, token.lemma_.lower(), o_text))
    return triples


def extract_relations(text, nlp, verbs=None, n_verbs=150, chunk_size=75_000, **kwargs):
    """
    Extract entity-verb-entity triples from a long text.

    Parameters
    ----------
    text : str
        Input text.
    nlp : spacy.language.Language
        spaCy pipeline with ``parser`` and ``ner``.
    verbs : set of str, optional
        Verb lemmas to consider. Defaults to the ``n_verbs`` most frequent
        verbs of ``text``.
    n_verbs : int, default=150
        Number of frequent verbs used when ``verbs`` is None.
    chunk_size : int, default=75_000
        Text is processed in chunks of this many characters.
    **kwargs
        Passed to :func:`extract_relations_doc`.

    Returns
    -------
    list of tuple of (str, str, str)
        Extracted triples.
    """
    if verbs is None:
        verbs = {v for v, _ in top_verbs(text, nlp, top_n=n_verbs)}
    triples = []
    for chunk in chunk_text(text, chunk_size):
        triples.extend(extract_relations_doc(nlp(chunk), verbs, **kwargs))
    return triples


def build_knowledge_graph(triples):
    """
    Build a directed knowledge graph from triples.

    Parameters
    ----------
    triples : list of tuple of (str, str, str)
        ``(subject, verb, object)`` triples.

    Returns
    -------
    networkx.DiGraph
        Nodes are entities; each edge has a ``label`` attribute joining
        the verbs that link the two entities.
    """
    import networkx as nx

    graph = nx.DiGraph()
    for subj, verb, obj in triples:
        if graph.has_edge(subj, obj):
            labels = set(graph[subj][obj]["label"].split(", ")) | {verb}
            graph[subj][obj]["label"] = ", ".join(sorted(labels))
        else:
            graph.add_edge(subj, obj, label=verb)
    return graph
