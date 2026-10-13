def train_word2vec(text, vector_size=100, window=5, min_count=1, workers=4, seed=42):
    """
    Train a Word2Vec model on a raw text.

    Parameters
    ----------
    text : str
        Training corpus.
    vector_size : int, default=100
        Embedding dimension.
    window : int, default=5
        Context window size.
    min_count : int, default=1
        Ignore words with a lower total frequency.
    workers : int, default=4
        Number of training threads.
    seed : int, default=42
        Random seed.

    Returns
    -------
    gensim.models.Word2Vec
        Trained model; vectors are in ``model.wv``.
    """
    import re

    import nltk
    from gensim.models import Word2Vec

    try:
        nltk.download("punkt_tab", quiet=True)
        sentences = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(text)]
    except LookupError:
        sentences = [re.findall(r"\w+|[^\w\s]", s) for s in re.split(r"(?<=[.!?])\s+", text)]
    return Word2Vec(sentences=sentences, vector_size=vector_size, window=window,
                    min_count=min_count, workers=workers, seed=seed)


def bert_token_embeddings(text, model_name="bert-base-uncased", max_length=512, device=None):
    """
    Compute contextual token embeddings with a BERT encoder.

    Parameters
    ----------
    text : str
        Input text; truncated to ``max_length`` tokens.
    model_name : str, default="bert-base-uncased"
        Hugging Face model id.
    max_length : int, default=512
        Maximum number of tokens.
    device : str, optional
        Device on which to run the model.

    Returns
    -------
    tokens : list of str
        Word-piece tokens, including ``[CLS]`` and ``[SEP]``.
    embeddings : torch.Tensor
        Tensor of shape ``(n_tokens, hidden_size)`` on CPU.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, outputs.last_hidden_state[0].cpu()
