import torch

from ner_re.bilstm_re import (E1_ID, E2_ID, PAD_ID, bilstm_forward, build_bilstm, build_vocab,
                              encode, tokenize)


def test_tokenize_keeps_markers():
    """
    Entity markers survive tokenisation (they were split in the notebook).
    """
    tokens = tokenize("[E1] Obama [/E1] visited [E2] Paris [/E2].")
    assert tokens == ["[E1]", "obama", "[/E1]", "visited", "[E2]", "paris", "[/E2]", "."]


def test_encode_marker_ids_and_padding():
    """
    Markers map to their reserved ids and sequences are padded.
    """
    vocab = build_vocab(["[E1] Obama [/E1] visited [E2] Paris [/E2]"])
    ids = encode("[E1] Obama [/E1] visited [E2] Paris [/E2]", vocab, max_len=10)
    assert ids[0] == E1_ID and ids[4] == E2_ID
    assert ids[-1] == PAD_ID and len(ids) == 10


def test_encode_crops_around_entities():
    """
    Long sequences keep the entity markers inside the window.
    """
    text = " ".join(["filler"] * 200) + " [E1] Obama [/E1] met [E2] Putin [/E2]"
    vocab = build_vocab([text])
    ids = encode(text, vocab, max_len=20)
    assert E1_ID in ids and E2_ID in ids


def test_forward_shape():
    """
    The functional model returns one logit vector per example.
    """
    model = build_bilstm(vocab_size=50, num_classes=4, embed_dim=8, hidden_dim=6, num_layers=1)
    x = torch.tensor([[E1_ID, 7, 3, E2_ID, 9, 5, PAD_ID, PAD_ID]] * 2)
    assert bilstm_forward(model, x).shape == (2, 4)
