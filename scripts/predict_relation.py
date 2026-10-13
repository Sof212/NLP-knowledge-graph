import argparse

from ner_re.labels import property_name


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Predict the relation between two entities.")
    parser.add_argument("--model", choices=["bert", "bilstm"], required=True)
    parser.add_argument("--model-path", required=True,
                        help="BERT: directory of best_model. BiLSTM: path to best_model.pt.")
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--e1", required=True, help="Head entity.")
    parser.add_argument("--e2", required=True, help="Tail entity.")
    return parser.parse_args()


def main():
    """
    Load a trained classifier and print its prediction.

    Returns
    -------
    None
    """
    args = parse_args()
    if args.model == "bert":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from ner_re.bert_re import predict_relation

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        label, conf = predict_relation(args.sentence, args.e1, args.e2, model, tokenizer)
    else:
        from ner_re.bilstm_re import load_bilstm, predict_relation

        model, vocab, id2label, cfg = load_bilstm(args.model_path)
        label, conf = predict_relation(args.sentence, args.e1, args.e2, model, vocab,
                                       id2label, cfg["max_len"])
    print(f"({args.e1}) --[{property_name(label)}]--> ({args.e2})  confidence={conf:.2f}")


if __name__ == "__main__":
    main()
