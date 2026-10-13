import pandas as pd

from ner_re.crf import group_sentences, sent2features, word2features
from ner_re.preprocessing import clean_text, load_ner_dataset


def test_clean_text():
    """
    Lowercase, punctuation/digits removed and whitespace collapsed.
    """
    assert clean_text("  Hello,   World! 2024 ") == "hello world"


def test_load_and_group_sentences(tmp_path):
    """
    Sentence ids are forward-filled and tokens grouped per sentence.
    """
    csv = tmp_path / "ner.csv"
    pd.DataFrame({
        "Sentence #": ["Sentence: 1", None, None, "Sentence: 2", None],
        "Word": ["Obama", "visited", "Paris", "Hello", None],
        "POS": ["NNP", "VBD", "NNP", "UH", "."],
        "Tag": ["B-per", "O", "B-geo", "O", "O"],
    }).to_csv(csv, index=False, encoding="latin1")

    df = load_ner_dataset(csv)
    sentences, tags = group_sentences(df)
    assert sentences == [[("Obama", "NNP"), ("visited", "VBD"), ("Paris", "NNP")],
                         [("Hello", "UH")]]
    assert tags == [["B-per", "O", "B-geo"], ["O"]]


def test_word2features_context_and_boundaries():
    """
    Context features and sentence boundary markers are produced.
    """
    sent = [("Obama", "NNP"), ("visited", "VBD"), ("Paris", "NNP")]
    first = word2features(sent, 0)
    assert first["word.istitle()"] and first["BOS_1"] and first["BOS_2"]
    assert first["1:word.lower()"] == "visited"
    assert first["2:pos"] == "NNP"
    assert word2features(sent, 2)["EOS_1"]
    assert len(sent2features(sent)) == 3
