"""Retrieval quality and retrieval bias, computed from the expert stages of neighbours.

The expert hypnogram is the reference for library epochs only. These metrics ask two
questions on held-out queries: do neighbours carry the query's expert stage, and when the
two models disagree, does a space side with the model that shaped it?
"""

import numpy as np

from ..ml.registry import CLASS_NAMES
from ..services.metrics import confusion_matrix, summarize

N_CLASSES = len(CLASS_NAMES)


def majority_vote(neighbour_stages: np.ndarray) -> np.ndarray:
    """Most frequent stage per row, nearest first; a tie goes to the nearest tied neighbour.

    ``-1`` marks a missing neighbour; a row without any neighbour votes ``-1``.
    """
    rows = np.arange(len(neighbour_stages))
    counts = np.zeros((len(neighbour_stages), N_CLASSES), dtype=np.int64)
    for column in neighbour_stages.T:
        valid = column >= 0
        np.add.at(counts, (rows[valid], column[valid]), 1)
    best = counts.max(axis=1)
    vote = np.full(len(neighbour_stages), -1, dtype=np.int64)
    for column in neighbour_stages.T[::-1]:  # nearer columns overwrite farther ones
        tied = (column >= 0) & (counts[rows, np.maximum(column, 0)] == best) & (best > 0)
        vote[tied] = column[tied]
    return vote


def precision_at_k(neighbour_stages: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per query: the share of its neighbours whose expert stage equals the query's."""
    valid = neighbour_stages >= 0
    hits = (neighbour_stages == truth[:, None]) & valid
    return hits.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)


def by_class(values: np.ndarray, truth: np.ndarray) -> dict[str, float | None]:
    return {name: (float(values[truth == index].mean()) if (truth == index).any() else None)
            for index, name in enumerate(CLASS_NAMES)}


def classification(truth: np.ndarray, predicted: np.ndarray) -> dict:
    """``summarize`` over the rows that have a prediction, with named per-class entries."""
    keep = predicted >= 0
    report = summarize(confusion_matrix(truth[keep], predicted[keep]))
    report["per_class"] = dict(zip(CLASS_NAMES, report["per_class"], strict=True))
    report["coverage"] = float(keep.mean())
    return report


def _share(mask: np.ndarray) -> float | None:
    return float(mask.mean()) if mask.size else None


def disagreement(truth: np.ndarray, predictions: dict[str, np.ndarray],
                 votes: dict[str, np.ndarray]) -> dict:
    """On epochs where exactly two models disagree, whom does each space's vote side with?

    ``preference`` is P(vote = first model) - P(vote = second model). A space shaped by one
    model leans towards it; the model-free space shows the lean the signal itself supports.
    """
    (first, a), (second, b) = predictions.items()
    mask = a != b
    t, a, b = truth[mask], a[mask], b[mask]
    report = {"n_epochs": int(mask.sum()), "share_of_queries": float(mask.mean()),
              "model_correct": {first: _share(a == t), second: _share(b == t)}, "spaces": {}}
    for space, vote in votes.items():
        v = vote[mask]
        report["spaces"][space] = {
            "vote_matches_expert": _share(v == t),
            f"sides_with_{first}": _share(v == a),
            f"sides_with_{second}": _share(v == b),
            "sides_with_neither": _share((v != a) & (v != b)),
            "preference": float((v == a).mean() - (v == b).mean()) if v.size else None,
            f"{first}_correct_when_sided_with": _share(a[v == a] == t[v == a]),
            f"{second}_correct_when_sided_with": _share(b[v == b] == t[v == b]),
        }
    return report


def cross_space(truth: np.ndarray, votes: dict[str, np.ndarray]) -> dict:
    """Coverage and accuracy when every space's vote agrees, and when they do not."""
    stacked = np.stack(list(votes.values()))
    unanimous = (stacked == stacked[0]).all(axis=0) & (stacked[0] >= 0)
    return {
        "spaces": list(votes),
        "unanimous": {"coverage": float(unanimous.mean()),
                      "accuracy": _share(stacked[0][unanimous] == truth[unanimous])},
        "split": {"coverage": float((~unanimous).mean())},
    }
