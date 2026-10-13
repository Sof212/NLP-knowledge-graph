import os
from itertools import combinations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ner_re.re_data import ENTITY_MARKERS, add_marked_sentences, build_label_maps, mark_entities
from ner_re.utils import ensure_dir, get_device, save_json, set_seed


def load_bert_re(model_name, label2id, id2label):
    """
    Load a BERT sequence classifier extended with entity marker tokens.

    Parameters
    ----------
    model_name : str
        Hugging Face model id or local directory.
    label2id : dict
        ``label -> index``.
    id2label : dict
        ``index -> label``.

    Returns
    -------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer that knows the ``[E1] [/E1] [E2] [/E2]`` tokens.
    model : transformers.PreTrainedModel
        Classifier with resized embeddings.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": list(ENTITY_MARKERS)})
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label2id), label2id=label2id,
        id2label={int(k): v for k, v in id2label.items()},
    )
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def encode_examples(texts, tokenizer, max_length=128, labels=None):
    """
    Tokenise marked sentences into a TensorDataset.

    Parameters
    ----------
    texts : list of str
        Sentences with entity markers.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer from :func:`load_bert_re`.
    max_length : int, default=128
        Padding / truncation length.
    labels : list of int, optional
        Label indices.

    Returns
    -------
    torch.utils.data.TensorDataset
        ``(input_ids, attention_mask[, labels])``.
    """
    enc = tokenizer(list(texts), max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    tensors = [enc["input_ids"], enc["attention_mask"]]
    if labels is not None:
        tensors.append(torch.as_tensor(labels, dtype=torch.long))
    return TensorDataset(*tensors)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """
    Run one fine-tuning epoch.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Sequence classifier.
    loader : torch.utils.data.DataLoader
        Batches of ``(input_ids, attention_mask, labels)``.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Learning-rate scheduler stepped every batch.
    device : str
        Device.

    Returns
    -------
    float
        Mean training loss.
    """
    model.train()
    total = 0.0
    for input_ids, attention_mask, labels in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                    labels=labels.to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += out.loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def predict_loader(model, loader, device):
    """
    Predict label indices for every batch of a loader.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Sequence classifier.
    loader : torch.utils.data.DataLoader
        Batches whose first two tensors are ``input_ids`` and
        ``attention_mask``.
    device : str
        Device.

    Returns
    -------
    numpy.ndarray
        Predicted label indices.
    """
    model.eval()
    preds = []
    for batch in tqdm(loader, desc="predict", leave=False):
        logits = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device)).logits
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def fit(train_df, val_df, config):
    """
    Fine-tune BERT for relation classification with entity markers.

    Parameters
    ----------
    train_df, val_df : pandas.DataFrame
        Columns ``sentence``, ``entity1``, ``entity2``, ``label``.
    config : dict
        Keys ``model_name``, ``max_length``, ``batch_size``, ``epochs``,
        ``learning_rate``, ``warmup_ratio``, ``seed`` and ``output_dir``.
        The best model (validation macro-F1) is saved to
        ``<output_dir>/best_model``.

    Returns
    -------
    history : list of dict
        Per-epoch ``train_loss``, ``val_accuracy`` and ``val_macro_f1``.
    y_true, y_pred : numpy.ndarray
        Validation labels and predictions of the last epoch.
    id2label : dict
        ``index -> label``.
    """
    from transformers import get_linear_schedule_with_warmup

    set_seed(config["seed"])
    device = get_device()
    output_dir = ensure_dir(config["output_dir"])

    label2id, id2label = build_label_maps(train_df["label"])
    val_df = val_df[val_df["label"].isin(label2id)]
    tokenizer, model = load_bert_re(config["model_name"], label2id, id2label)
    model.to(device)

    train_df, val_df = add_marked_sentences(train_df), add_marked_sentences(val_df)
    y_true = val_df["label"].map(label2id).to_numpy()
    train_loader = DataLoader(
        encode_examples(train_df["marked"], tokenizer, config["max_length"],
                        train_df["label"].map(label2id).tolist()),
        batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(
        encode_examples(val_df["marked"], tokenizer, config["max_length"]),
        batch_size=config["batch_size"] * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.get("warmup_ratio", 0.1) * total_steps), total_steps)

    history, best_f1, y_pred = [], -1.0, None
    for epoch in range(1, config["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        y_pred = predict_loader(model, val_loader, device)
        acc = accuracy_score(y_true, y_pred)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": loss,
                        "val_accuracy": acc, "val_macro_f1": macro})
        print(f"epoch {epoch} | loss {loss:.4f} | val acc {acc:.4f} | val macro-F1 {macro:.4f}")
        if macro > best_f1:
            best_f1 = macro
            model.save_pretrained(os.path.join(output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
        save_json(history, os.path.join(output_dir, "history.json"))
    return history, y_true, y_pred, id2label


@torch.no_grad()
def predict_relation(sentence, entity1, entity2, model, tokenizer, max_length=128, device=None):
    """
    Predict the relation between two entities of a sentence.

    Parameters
    ----------
    sentence : str
        Input sentence.
    entity1, entity2 : str
        Head and tail entities (must appear in ``sentence``).
    model : transformers.PreTrainedModel
        Fine-tuned classifier.
    tokenizer : transformers.PreTrainedTokenizer
        Matching tokenizer.
    max_length : int, default=128
        Truncation length.
    device : str, optional
        Device.

    Returns
    -------
    label : str
        Predicted relation.
    confidence : float
        Softmax probability of the predicted relation.
    """
    device = device or get_device()
    model.to(device).eval()
    enc = tokenizer(mark_entities(sentence, entity1, entity2), max_length=max_length,
                    truncation=True, return_tensors="pt").to(device)
    probs = model(**enc).logits.softmax(dim=-1)[0]
    idx = int(probs.argmax())
    return model.config.id2label[idx], float(probs[idx])


def predict_text_relations(text, nlp, model, tokenizer, entity_types=None, min_confidence=0.0):
    """
    Detect entities with spaCy and classify every entity pair per sentence.

    Parameters
    ----------
    text : str
        Input text.
    nlp : spacy.language.Language
        spaCy pipeline with ``ner``.
    model : transformers.PreTrainedModel
        Fine-tuned classifier.
    tokenizer : transformers.PreTrainedTokenizer
        Matching tokenizer.
    entity_types : set of str, optional
        Keep only these spaCy labels (e.g. ``{"PERSON", "ORG", "GPE"}``).
    min_confidence : float, default=0.0
        Drop predictions below this probability.

    Returns
    -------
    list of dict
        ``sentence``, ``entity1``, ``entity2``, ``relation``,
        ``confidence``.
    """
    results = []
    for sent in nlp(text).sents:
        ents = [e.text for e in sent.ents if entity_types is None or e.label_ in entity_types]
        for e1, e2 in combinations(dict.fromkeys(ents), 2):
            label, conf = predict_relation(sent.text, e1, e2, model, tokenizer)
            if conf >= min_confidence:
                results.append({"sentence": sent.text, "entity1": e1, "entity2": e2,
                                "relation": label, "confidence": conf})
    return results
