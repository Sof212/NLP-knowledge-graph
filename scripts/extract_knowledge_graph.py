import argparse
import os

import pandas as pd

from ner_re.coreference import resolve_document
from ner_re.preprocessing import build_text_sample, load_ner_dataset, load_spacy
from ner_re.rule_relations import build_knowledge_graph, extract_relations
from ner_re.utils import ensure_dir
from ner_re.visualization import plot_knowledge_graph


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Part C: rule-based relation extraction and knowledge graph.")
    parser.add_argument("--data", default="data/ner_dataset.csv")
    parser.add_argument("--n-words", type=int, default=30_000)
    parser.add_argument("--spacy-model", default="en_core_web_md")
    parser.add_argument("--n-verbs", type=int, default=150,
                        help="Number of most frequent verbs used as relation triggers.")
    parser.add_argument("--coref", action="store_true",
                        help="Resolve pronouns before extraction (heuristic).")
    parser.add_argument("--output-dir", default="outputs/knowledge_graph")
    return parser.parse_args()


def main():
    """
    Extract triples, save them as CSV and draw the knowledge graph.

    Returns
    -------
    None
    """
    args = parse_args()
    out = ensure_dir(args.output_dir)
    nlp = load_spacy(args.spacy_model)
    text = build_text_sample(load_ner_dataset(args.data), args.n_words)
    if args.coref:
        text = resolve_document(text, nlp)

    triples = extract_relations(text, nlp, n_verbs=args.n_verbs)
    tag = "coref" if args.coref else "raw"
    pd.DataFrame(triples, columns=["subject", "verb", "object"]).to_csv(
        os.path.join(out, f"triples_{tag}.csv"), index=False)
    print(f"{len(triples)} triples extracted")
    for t in triples[:20]:
        print("  ", t)

    graph = build_knowledge_graph(triples)
    plot_knowledge_graph(graph, title=f"Knowledge graph ({tag}, {len(triples)} triples)",
                         save_path=os.path.join(out, f"knowledge_graph_{tag}.png"), show=False)


if __name__ == "__main__":
    main()
