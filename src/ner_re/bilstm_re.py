import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ner_re.re_data import ENTITY_MARKERS, add_marked_sentences, build_label_maps, mark_entities
from ner_re.utils import ensure_dir, get_device, save_json, set_seed

PAD, UNK = "<PAD>", "<UNK>"
SPECIAL_TOKENS = (PAD, UNK) + ENTITY_MARKERS
PAD_ID, UNK_ID, E1_ID, E1_END_ID, E2_ID, E2_END_ID = range(len(SPECIAL_TOKENS))
TOKEN_PATTERN = re.compile(r"\[/?E[12]\]|\w+|[^\w\s]")


def tokenize(text):
    """
    Tokenise text, keeping entity markers as single tokens.

    Parameters
    ----------
    text : str
        Input text, possibly containing ``[E1] ... [/E2]`` markers.

    Returns
    -------
    list of str
        Lowercased tokens; markers are kept verbatim.
    """
    return [t if t in ENTITY_MARKERS else t.lower() for t in TOKEN_PATTERN.findall(text)]


def build_vocab(texts, min_freq=1, max_size=None):
    """
    Build a token vocabulary from training texts.

    Parameters
    ----------
    texts : iterable of str
        Training texts (with markers).
    min_freq : int, default=1
        Minimum frequency for a token to be kept.
    max_size : int, optional
        Maximum number of regular tokens.

    Returns
    -------
    dict
        ``token -> index``; special tokens occupy the first indices.
    """
    counts = Counter(t for text in texts for t in tokenize(text) if t not in ENTITY_MARKERS)
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tok, freq in counts.most_common(max_size):
        if freq >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(text, vocab, max_len=120):
    """
    Convert a marked text into a fixed-length id sequence.

    Long sequences are cropped around the first entity marker so that the
    entities are kept inside the window.

    Parameters
    ----------
    text : str
        Text with entity markers.
    vocab : dict
        Vocabulary from :func:`build_vocab`.
    max_len : int, default=120
        Output length.

    Returns
    -------
    list of int
        Token ids, right-padded with ``PAD_ID``.
    """
    ids = [vocab.get(t, UNK_ID) for t in tokenize(text)]
    if len(ids) > max_len:
        markers = [i for i, t in enumerate(ids) if t in (E1_ID, E2_ID)]
        start = max(0, min(min(markers, default=0) - max_len // 4, len(ids) - max_len))
        ids = ids[start:start + max_len]
    return ids + [PAD_ID] * (max_len - len(ids))


def build_bilstm(vocab_size, num_classes, embed_dim=300, hidden_dim=512,
                 num_layers=3, dropout=0.3):
    """
    Create the layers of the BiLSTM entity-aware attention model.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    num_classes : int
        Number of relation labels.
    embed_dim : int, default=300
        Embedding dimension.
    hidden_dim : int, default=512
        LSTM hidden size per direction.
    num_layers : int, default=3
        Number of stacked LSTM layers.
    dropout : float, default=0.3
        Dropout rate.

    Returns
    -------
    torch.nn.ModuleDict
        Layers used by :func:`bilstm_forward`.
    """
    return nn.ModuleDict({
        "embedding": nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID),
        "dropout": nn.Dropout(dropout),
        "lstm": nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                        bidirectional=True, dropout=dropout if num_layers > 1 else 0.0),
        "attention": nn.Linear(2 * hidden_dim, hidden_dim),
        "context": nn.Linear(hidden_dim, 1, bias=False),
        "classifier": nn.Sequential(
            nn.Linear(6 * hidden_dim, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        ),
    })


def bilstm_forward(model, x, entity_boost=3.0):
    """
    Forward pass of the BiLSTM entity-aware attention model.

    The sentence representation is an attention-pooled sum of the BiLSTM
    states (padding is masked, entity start markers get a score boost).
    It is concatenated with the states at the ``[E1]`` and ``[E2]``
    markers and fed to an MLP classifier.

    Parameters
    ----------
    model : torch.nn.ModuleDict
        Layers from :func:`build_bilstm`.
    x : torch.Tensor
        Token ids of shape ``(batch, seq_len)``.
    entity_boost : float, default=3.0
        Value added to the attention scores of entity markers.

    Returns
    -------
    torch.Tensor
        Logits of shape ``(batch, num_classes)``.
    """
    states, _ = model["lstm"](model["dropout"](model["embedding"](x)))
    e1_mask = (x == E1_ID).unsqueeze(-1).float()
    e2_mask = (x == E2_ID).unsqueeze(-1).float()
    e1_repr = (states * e1_mask).sum(dim=1)
    e2_repr = (states * e2_mask).sum(dim=1)

    scores = model["context"](torch.tanh(model["attention"](states)))
    scores = scores + (e1_mask + e2_mask) * entity_boost
    scores = scores.masked_fill((x == PAD_ID).unsqueeze(-1), float("-inf"))
    sent_repr = (F.softmax(scores, dim=1) * states).sum(dim=1)
    return model["classifier"](torch.cat([sent_repr, e1_repr, e2_repr], dim=1))


def make_loader(df, vocab, label2id, max_len, batch_size, shuffle):
    """
    Encode a relation dataframe into a DataLoader.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``marked`` and ``label`` columns.
    vocab : dict
        Vocabulary.
    label2id : dict
        ``label -> index``.
    max_len : int
        Sequence length.
    batch_size : int
        Batch size.
    shuffle : bool
        Shuffle every epoch.

    Returns
    -------
    torch.utils.data.DataLoader
        Batches of ``(token_ids, labels)``.
    """
    x = torch.tensor([encode(t, vocab, max_len) for t in df["marked"]], dtype=torch.long)
    y = torch.tensor(df["label"].map(label2id).to_numpy(), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_loader(model, loader, device):
    """
    Predict label indices for a loader.

    Parameters
    ----------
    model : torch.nn.ModuleDict
        Trained model.
    loader : torch.utils.data.DataLoader
        Batches of ``(token_ids, labels)``.
    device : str
        Device.

    Returns
    -------
    y_true, y_pred : numpy.ndarray
        Gold and predicted label indices.
    """
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        y_pred.append(bilstm_forward(model, xb.to(device)).argmax(dim=1).cpu().numpy())
        y_true.append(yb.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def fit(train_df, val_df, config):
    """
    Train the BiLSTM entity-aware attention relation classifier.

    Parameters
    ----------
    train_df, val_df : pandas.DataFrame
        Columns ``sentence``, ``entity1``, ``entity2``, ``label``.
    config : dict
        Keys ``max_len``, ``min_freq``, ``embed_dim``, ``hidden_dim``,
        ``num_layers``, ``dropout``, ``batch_size``, ``epochs``,
        ``learning_rate``, ``class_weights``, ``seed`` and ``output_dir``.
        The best model (validation macro-F1) is saved with its vocabulary
        and label map.

    Returns
    -------
    history : list of dict
        Per-epoch ``train_loss``, ``val_macro_f1`` and ``val_micro_f1``.
    y_true, y_pred : numpy.ndarray
        Validation labels and predictions of the best epoch.
    id2label : dict
        ``index -> label``.
    """
    set_seed(config["seed"])
    device = get_device()
    output_dir = ensure_dir(config["output_dir"])

    label2id, id2label = build_label_maps(train_df["label"])
    train_df = add_marked_sentences(train_df)
    val_df = add_marked_sentences(val_df[val_df["label"].isin(label2id)])
    vocab = build_vocab(train_df["marked"], min_freq=config.get("min_freq", 1))

    train_loader = make_loader(train_df, vocab, label2id, config["max_len"],
                               config["batch_size"], shuffle=True)
    val_loader = make_loader(val_df, vocab, label2id, config["max_len"],
                             config["batch_size"] * 2, shuffle=False)

    model = build_bilstm(len(vocab), len(label2id), config["embed_dim"], config["hidden_dim"],
                         config["num_layers"], config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    weight = None
    if config.get("class_weights", True):
        y_train = train_df["label"].map(label2id).to_numpy()
        present = np.unique(y_train)
        w = np.ones(len(label2id), dtype=np.float32)
        w[present] = compute_class_weight("balanced", classes=present, y=y_train)
        weight = torch.tensor(w, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    history, best_f1, best_pred, y_true = [], -1.0, None, None
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            optimizer.zero_grad()
            loss = criterion(bilstm_forward(model, xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
        y_true, y_pred = predict_loader(model, val_loader, device)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total / len(train_loader),
                        "val_macro_f1": macro, "val_micro_f1": micro})
        print(f"epoch {epoch} | loss {history[-1]['train_loss']:.4f} | "
              f"val macro-F1 {macro:.4f} | micro-F1 {micro:.4f}")
        if macro > best_f1:
            best_f1, best_pred = macro, y_pred
            torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                        "id2label": id2label, "config": config},
                       os.path.join(output_dir, "best_model.pt"))
        save_json(history, os.path.join(output_dir, "history.json"))
    return history, y_true, best_pred, id2label


def load_bilstm(path, device=None):
    """
    Load a model saved by :func:`fit`.

    Parameters
    ----------
    path : str
        Path to ``best_model.pt``.
    device : str, optional
        Device.

    Returns
    -------
    model : torch.nn.ModuleDict
        Model in evaluation mode.
    vocab : dict
        Vocabulary.
    id2label : dict
        ``index -> label``.
    config : dict
        Training configuration.
    """
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_bilstm(len(ckpt["vocab"]), len(ckpt["id2label"]), cfg["embed_dim"],
                         cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["vocab"], ckpt["id2label"], cfg


@torch.no_grad()
def predict_relation(sentence, entity1, entity2, model, vocab, id2label, max_len=120, device=None):
    """
    Predict the relation between two entities with the BiLSTM model.

    The sentence goes through exactly the same marking, tokenisation and
    encoding as during training.

    Parameters
    ----------
    sentence : str
        Input sentence.
    entity1, entity2 : str
        Head and tail entities.
    model : torch.nn.ModuleDict
        Trained model.
    vocab : dict
        Training vocabulary.
    id2label : dict
        ``index -> label``.
    max_len : int, default=120
        Sequence length used during training.
    device : str, optional
        Device.

    Returns
    -------
    label : str
        Predicted relation.
    confidence : float
        Softmax probability of the prediction.
    """
    device = device or get_device()
    model.to(device).eval()
    x = torch.tensor([encode(mark_entities(sentence, entity1, entity2), vocab, max_len)],
                     device=device)
    probs = bilstm_forward(model, x).softmax(dim=-1)[0]
    idx = int(probs.argmax())
    return id2label[idx], float(probs[idx])
