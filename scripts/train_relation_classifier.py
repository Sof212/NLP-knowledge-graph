import argparse
import os

from sklearn.metrics import classification_report

from ner_re.labels import property_name
from ner_re.re_data import group_split, load_relation_dataset
from ner_re.utils import load_config, save_json
from ner_re.visualization import (plot_confusion_matrix, plot_label_distribution,
                                  plot_training_history)


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Part C: supervised relation classification.")
    parser.add_argument("--model", choices=["bert", "bilstm"], required=True)
    parser.add_argument("--config", help="Defaults to configs/<model>_re.yaml.")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs.")
    parser.add_argument("--max-examples", type=int, help="Override max_examples.")
    return parser.parse_args()


def main():
    """
    Load the data, split it by sentence/document, train and evaluate.

    Returns
    -------
    None
    """
    args = parse_args()
    config = load_config(args.config or f"configs/{args.model}_re.yaml")
    if args.epochs:
        config["epochs"] = args.epochs
    if args.max_examples:
        config["max_examples"] = args.max_examples

    if args.model == "bert":
        from ner_re.bert_re import fit
    else:
        from ner_re.bilstm_re import fit

    df, group_col = load_relation_dataset(config["dataset"], config["data_path"],
                                          config.get("max_examples"),
                                          config.get("docred_fallback", "mentions"))
    train_df, val_df = group_split(df, group_col, config["test_size"], config["seed"])
    print(f"{len(df)} examples, {df['label'].nunique()} labels | "
          f"train {len(train_df)} / val {len(val_df)} (split by {group_col})")

    out = config["output_dir"]
    os.makedirs(out, exist_ok=True)
    plot_label_distribution(df["label"], save_path=os.path.join(out, "label_distribution.png"),
                            show=False)

    history, y_true, y_pred, id2label = fit(train_df, val_df, config)

    names = [property_name(id2label[i]) for i in range(len(id2label))]
    report = classification_report(y_true, y_pred, labels=list(range(len(names))),
                                   target_names=names, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, labels=list(range(len(names))),
                                target_names=names, zero_division=0))
    save_json(report, os.path.join(out, "classification_report.json"))

    metrics = [k for k in history[0] if k.startswith("val_")]
    plot_training_history(history, metrics, save_path=os.path.join(out, "history.png"),
                          show=False)
    plot_confusion_matrix([names[i] for i in y_true], [names[i] for i in y_pred], names,
                          title=f"{args.model} relation classifier (row-normalised)",
                          save_path=os.path.join(out, "confusion_matrix.png"), show=False)


if __name__ == "__main__":
    main()
