import argparse
import os

from ner_re.crf import evaluate_ner, group_sentences, save_crf, split_dataset, train_crf
from ner_re.preprocessing import load_ner_dataset
from ner_re.utils import ensure_dir, load_config, save_json
from ner_re.visualization import plot_confusion_matrix


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Part B: train a CRF for named entity recognition.")
    parser.add_argument("--config", default="configs/crf.yaml")
    return parser.parse_args()


def main():
    """
    Train the CRF, evaluate it at entity level and save model and figures.

    Returns
    -------
    None
    """
    config = load_config(parse_args().config)
    out = ensure_dir(config["output_dir"])

    sentences, tags = group_sentences(load_ner_dataset(config["data_path"]))
    if config.get("max_sentences"):
        sentences, tags = sentences[:config["max_sentences"]], tags[:config["max_sentences"]]
    print(f"{len(sentences)} sentences")

    X_train, X_test, y_train, y_test = split_dataset(sentences, tags, config["test_size"],
                                                     config["seed"])
    crf = train_crf(X_train, y_train, config["c1"], config["c2"], config["max_iterations"])
    y_pred = crf.predict(X_test)

    metrics = evaluate_ner(y_test, y_pred)
    print(metrics["report"])
    print(f"precision {metrics['precision']:.4f} | recall {metrics['recall']:.4f} | "
          f"F1 {metrics['f1']:.4f}")
    save_json(metrics, os.path.join(out, "metrics.json"))
    save_crf(crf, os.path.join(out, "crf.pkl"))

    labels = sorted({t for seq in y_test for t in seq} - {"O"}) + ["O"]
    plot_confusion_matrix([t for s in y_test for t in s], [t for s in y_pred for t in s],
                          labels, title="CRF - token-level confusion (row-normalised)",
                          save_path=os.path.join(out, "confusion_matrix.png"), show=False)


if __name__ == "__main__":
    main()
