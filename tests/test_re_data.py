import json

import pandas as pd

from ner_re.coreference import pronoun_replacements, replace_pronouns
from ner_re.labels import property_name
from ner_re.re_data import build_label_maps, group_split, load_docred, load_nyt, mark_entities


def test_mark_entities_basic():
    """
    Both entities are wrapped with their markers.
    """
    out = mark_entities("Obama visited Paris", "Obama", "Paris")
    assert out == "[E1] Obama [/E1] visited [E2] Paris [/E2]"


def test_mark_entities_overlapping_names():
    """
    The tail is not marked inside the head when one name contains the other.
    """
    out = mark_entities("North Korea is in Asia. Korea is divided.", "North Korea", "Korea")
    assert out == "[E1] North Korea [/E1] is in Asia. [E2] Korea [/E2] is divided."


def test_mark_entities_missing_entity():
    """
    A missing entity is simply left unmarked.
    """
    assert mark_entities("Obama spoke", "Obama", "Paris") == "[E1] Obama [/E1] spoke"


def test_load_nyt(tmp_path):
    """
    Freebase labels are normalised.
    """
    path = tmp_path / "nyt.json"
    path.write_text(json.dumps({"sentText": "Bard College is in Annandale .",
                                "relationMentions": [{"em1Text": "Annandale",
                                                      "em2Text": "Bard College",
                                                      "label": "/location/location/contains"}]})
                    + "\n", encoding="utf-8")
    df = load_nyt(path)
    assert df.loc[0, "label"] == "location_location_contains"


def test_load_docred_evidence_and_fallback(tmp_path):
    """
    Evidence sentences are used, with the mention-based fallback otherwise.
    """
    doc = {
        "sents": [["Obama", "was", "born", "in", "Hawaii", "."], ["He", "likes", "Paris", "."]],
        "vertexSet": [[{"name": "Obama"}], [{"name": "Hawaii"}], [{"name": "Paris"}]],
        "labels": [{"h": 0, "t": 1, "r": "P19", "evidence": [0]},
                   {"h": 0, "t": 2, "r": "P551", "evidence": []}],
    }
    path = tmp_path / "docred.json"
    path.write_text(json.dumps([doc]), encoding="utf-8")
    df = load_docred(path)
    assert df.loc[0, "sentence"] == "Obama was born in Hawaii ."
    assert df.loc[1, "sentence"] == "Obama was born in Hawaii . He likes Paris ."
    assert property_name(df.loc[0, "label"]) == "place_of_birth"


def test_group_split_has_no_leakage():
    """
    No group appears in both splits.
    """
    df = pd.DataFrame({"doc_id": [i // 3 for i in range(30)], "label": ["a"] * 30})
    train, test = group_split(df, "doc_id", test_size=0.3, seed=0)
    assert not set(train["doc_id"]) & set(test["doc_id"])
    assert len(train) + len(test) == 30


def test_build_label_maps():
    """
    Label maps are sorted and inverse of each other.
    """
    label2id, id2label = build_label_maps(["b", "a", "b"])
    assert label2id == {"a": 0, "b": 1}
    assert id2label == {0: "a", 1: "b"}


def test_coreference_rules():
    """
    Person pronouns are replaced by the person antecedent.
    """
    repl = pronoun_replacements([("Obama", "PERSON"), ("Paris", "GPE")])
    assert replace_pronouns("He named his team.", repl) == "Obama named Obama's team."
    assert pronoun_replacements([("Monday", "DATE")]) == {}
