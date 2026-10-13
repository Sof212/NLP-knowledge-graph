import matplotlib.pyplot as plt
import numpy as np


def _finish(fig, save_path, show):
    """
    Save and/or display a figure, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalise.
    save_path : str or None
        Output path; nothing is saved if None.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure.
    """
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_confusion_matrix(y_true, y_pred, labels, normalize=True, title="Confusion matrix",
                          save_path=None, show=True):
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : sequence
        Gold and predicted labels.
    labels : list
        Label order of the axes.
    normalize : bool, default=True
        Normalise each row (recall per class). Recommended with many or
        imbalanced classes.
    title : str, default="Confusion matrix"
        Figure title.
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    if normalize:
        cm = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm),
                       where=cm.sum(axis=1, keepdims=True) > 0)
    size = max(8, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=len(labels) <= 20, fmt=".2f" if normalize else ".0f",
                cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _finish(fig, save_path, show)


def plot_label_distribution(labels, top_n=30, title="Label distribution",
                            save_path=None, show=True):
    """
    Plot the frequency of the most common labels.

    Parameters
    ----------
    labels : pandas.Series
        Label column.
    top_n : int, default=30
        Number of labels shown.
    title : str, default="Label distribution"
        Figure title.
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    counts = labels.value_counts().head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values)
    ax.set_xlabel("Count")
    ax.set_title(title)
    return _finish(fig, save_path, show)


def plot_knowledge_graph(graph, title="Knowledge graph", figsize=(14, 10), seed=42,
                         save_path=None, show=True):
    """
    Draw a knowledge graph with verb labels on the edges.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph from :func:`ner_re.rule_relations.build_knowledge_graph`.
    title : str, default="Knowledge graph"
        Figure title.
    figsize : tuple of int, default=(14, 10)
        Figure size.
    seed : int, default=42
        Seed of the spring layout, for reproducible drawings.
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    import networkx as nx

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(graph, k=0.8, seed=seed)
    nx.draw(graph, pos, ax=ax, with_labels=True, node_color="lightskyblue",
            node_size=2000, font_size=9, font_weight="bold", arrows=True)
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, font_color="darkred", font_size=8,
                                 edge_labels=nx.get_edge_attributes(graph, "label"))
    ax.set_title(title, fontsize=16)
    return _finish(fig, save_path, show)


def plot_training_history(history, metrics, title="Training history",
                          save_path=None, show=True):
    """
    Plot metrics recorded at each epoch.

    Parameters
    ----------
    history : list of dict
        One dict per epoch containing ``epoch`` and the metric keys.
    metrics : list of str
        Keys to plot.
    title : str, default="Training history"
        Figure title.
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in metrics:
        ax.plot(epochs, [h[m] for h in history], marker="o", label=m)
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show)
