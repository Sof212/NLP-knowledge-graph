import argparse
import os

from ner_re.embeddings import bert_token_embeddings, train_word2vec
from ner_re.preprocessing import (build_text_sample, extract_entities, load_ner_dataset,
                                  load_spacy, spacy_preprocess, top_verbs)
from ner_re.utils import ensure_dir, save_json


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Part A: preprocessing, spaCy NER and embeddings.")
    parser.add_argument("--data", default="data/ner_dataset.csv")
    parser.add_argument("--n-words", type=int, default=30_000)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--probe-word", default="London",
                        help="Word whose Word2Vec neighbours are printed.")
    parser.add_argument("--bert", action="store_true", help="Also compute BERT embeddings.")
    parser.add_argument("--output-dir", default="outputs/preprocessing")
    return parser.parse_args()


def main():
    """
    Run the exploratory preprocessing pipeline and save a JSON summary.

    Returns
    -------
    None
    """
    args = parse_args()
    out = ensure_dir(args.output_dir)
    df = load_ner_dataset(args.data)
    text = build_text_sample(df, args.n_words)
    nlp = load_spacy(args.spacy_model)

    processed = spacy_preprocess(text, nlp)
    entities = extract_entities(text, nlp)
    entities.to_csv(os.path.join(out, "spacy_entities.csv"), index=False)
    verbs = top_verbs(text, nlp, top_n=30)

    w2v = train_word2vec(text)
    neighbours = []
    if args.probe_word in w2v.wv:
        neighbours = [(w, float(s)) for w, s in w2v.wv.most_similar(args.probe_word, topn=10)]

    summary = {
        "n_sentences": int(df["sentence_id"].nunique()),
        "n_tokens_dataset": int(len(df)),
        "n_tags": int(df["tag"].nunique()),
        "tag_counts": df["tag"].value_counts().to_dict(),
        "sample_spacy_tokens": len(processed["tokens"]),
        "sample_lemmas_without_stopwords": len(processed["lemmas"]),
        "spacy_entity_counts": entities["label"].value_counts().to_dict(),
        "top_verbs": verbs,
        "word2vec_vocab_size": len(w2v.wv),
        f"word2vec_neighbours_{args.probe_word}": neighbours,
    }
    if args.bert:
        tokens, emb = bert_token_embeddings(text)
        summary["bert_embeddings_shape"] = list(emb.shape)

    save_json(summary, os.path.join(out, "summary.json"))
    for key in ("n_sentences", "n_tokens_dataset", "sample_spacy_tokens", "word2vec_vocab_size"):
        print(f"{key}: {summary[key]}")
    print("top verbs:", verbs[:10])
    print(f"neighbours of '{args.probe_word}':", neighbours[:5])


if __name__ == "__main__":
    main()
