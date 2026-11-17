"""Phase 2: how far can similarity search be trusted, and does it lean towards one model?

    python scripts/evaluate_library.py [--k 5] [--recall-sample 1000]

Queries are the epochs the embedding fold never trained on: its validation participants
and the held-out test participants. Each query searches the whole library in every space
(Transformer, AttnSleep, model-free signal features), never returning its own participant.
The query's expert stage is not used by the search; it only scores the neighbours.

Writes ``library_eval.json`` (all metrics) and ``eval_neighbours.npz`` (every query's
neighbour ids and scores) to the library directory.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.core.config import get_settings  # noqa: E402
from app.library import metrics, store  # noqa: E402
from app.ml.registry import CLASS_NAMES  # noqa: E402


def neighbour_stages(ids: np.ndarray, stages: np.ndarray) -> np.ndarray:
    return np.where(ids >= 0, stages[np.maximum(ids, 0)], -1)


def exact_top_k(space: str, library: np.ndarray, queries: np.ndarray,
                query_participants: np.ndarray, participants: np.ndarray, k: int) -> np.ndarray:
    """Brute-force neighbours with the same participant filter, to measure ANN recall."""
    cosine = store.SPACES[space].distance.value == "Cosine"
    if cosine:
        library = library / np.linalg.norm(library, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    else:
        library_norms = (library ** 2).sum(axis=1)
    result = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), 100):
        chunk = queries[start:start + 100]
        closeness = chunk @ library.T
        if not cosine:  # negative squared Euclidean distance; the query norm is constant
            closeness = 2 * closeness - library_norms
        closeness[participants[None, :] == query_participants[start:start + 100, None]] = -np.inf
        top = np.argpartition(-closeness, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(closeness, top, axis=1), axis=1)
        result[start:start + 100] = np.take_along_axis(top, order, axis=1)
    return result


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=settings.LIBRARY_DIR)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--recall-sample", type=int, default=1000,
                        help="queries checked against brute-force search")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    manifest = store.load_manifest(args.dir)
    epochs = store.load_epochs(args.dir)
    fold = manifest["fold"]
    stages = epochs["stage"]
    seen = store.embedding_training_mask(epochs, fold)
    queries = np.flatnonzero(~seen)
    truth = stages[queries]
    query_participants = epochs["participant"][queries]
    is_test = epochs["split"][queries] == "test"
    print(f"{len(queries)} queries (unseen by fold {fold}) against {len(stages)} epochs, "
          f"k = {args.k}", flush=True)

    with np.load(args.dir / "probabilities.npz") as stored:
        probabilities = {space: stored[space] for space in store.MODEL_SPACES}
    fold_predictions = {space: probabilities[space][fold - 1].argmax(axis=1)[queries]
                        for space in store.MODEL_SPACES}
    oof_predictions = {space: store.out_of_fold(probabilities[space], epochs["split"],
                                                epochs["validation_fold"]).argmax(axis=1)[queries]
                       for space in store.MODEL_SPACES}
    ensemble_predictions = {space: probabilities[space].mean(axis=0).argmax(axis=1)[queries]
                            for space in store.MODEL_SPACES}

    client = store.connect(settings.QDRANT_URL, settings.QDRANT_GRPC_PORT)
    name = store.collection_name(manifest)
    vectors = {space: np.load(args.dir / store.SPACES[space].file, mmap_mode="r")
               for space in store.SPACES}
    query_vectors, spaces, votes, raw = {}, {}, {}, {"queries": queries}
    for space in store.SPACES:
        matrix = np.asarray(vectors[space][queries], dtype=np.float32)
        if space == "signal":
            matrix = store.standardize_signal(matrix, manifest)
        query_vectors[space] = matrix
        started = time.perf_counter()
        ids, scores = store.search_batch(client, name, space, matrix, query_participants.tolist(),
                                         limit=args.k, batch_size=args.batch_size)
        elapsed = time.perf_counter() - started
        raw[f"{space}_ids"], raw[f"{space}_scores"] = ids, scores
        found = neighbour_stages(ids, stages)
        votes[space] = metrics.majority_vote(found)
        precision = metrics.precision_at_k(found, truth)
        spaces[space] = {
            "release": store.SPACES[space].release,
            "precision_at_k": float(precision.mean()),
            "precision_at_k_by_query_stage": metrics.by_class(precision, truth),
            "knn_vote": metrics.classification(truth, votes[space]),
            "neighbours_in_embedding_training": float(seen[ids[ids >= 0]].mean()),
            "queries_per_second": round(len(queries) / elapsed, 1),
        }
        print(f"  {space}: P@{args.k} {precision.mean():.3f}, "
              f"vote macro-F1 {spaces[space]['knn_vote']['macro_f1']:.3f} "
              f"({len(queries) / elapsed:.0f} queries/s)", flush=True)

    sample = np.sort(np.random.default_rng(0).choice(
        len(queries), size=min(args.recall_sample, len(queries)), replace=False))
    participants = epochs["participant"]
    for space in store.SPACES:
        library = np.asarray(vectors[space], dtype=np.float32)
        if space == "signal":
            library = store.standardize_signal(library, manifest)
        exact = exact_top_k(space, library, query_vectors[space][sample],
                            query_participants[sample], participants, args.k)
        approximate = raw[f"{space}_ids"][sample]
        overlap = [len(set(a) & set(e)) / args.k for a, e in zip(approximate, exact, strict=True)]
        spaces[space]["ann_recall_at_k"] = float(np.mean(overlap))
        del library

    first, second = store.MODEL_SPACES
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "library": {"library_id": manifest["library_id"], "collection": name, "fold": fold,
                    "n_epochs": manifest["n_epochs"], "created_at": manifest["created_at"]},
        "k": args.k,
        "queries": {
            "definition": f"epochs unseen by fold {fold}: its validation participants and the "
                          "test participants; neighbours from the query's participant excluded",
            "n": int(len(queries)),
            "test": int(is_test.sum()),
            f"fold{fold}_validation": int((~is_test).sum()),
            "class_counts": {CLASS_NAMES[c]: int((truth == c).sum())
                             for c in range(len(CLASS_NAMES))},
        },
        "baselines": {
            **{f"{space}_fold{fold}": metrics.classification(truth, fold_predictions[space])
               for space in store.MODEL_SPACES},
            **{f"{space}_out_of_fold": metrics.classification(truth, oof_predictions[space])
               for space in store.MODEL_SPACES},
        },
        "spaces": spaces,
        "disagreement": {
            f"fold{fold}_models": metrics.disagreement(
                truth, {first: fold_predictions[first], second: fold_predictions[second]}, votes),
            "ensembles_on_test": metrics.disagreement(
                truth[is_test],
                {first: ensemble_predictions[first][is_test],
                 second: ensemble_predictions[second][is_test]},
                {space: vote[is_test] for space, vote in votes.items()}),
        },
        "agreement_across_spaces": {
            "all": metrics.cross_space(truth, votes),
            "model_spaces": metrics.cross_space(truth, {s: votes[s] for s in store.MODEL_SPACES}),
        },
    }
    (args.dir / "library_eval.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.dir / "eval_neighbours.npz", **raw)
    print_summary(report, first, second)
    return 0


def print_summary(report: dict, first: str, second: str) -> None:
    k = report["k"]
    print(f"\n{'space':<12} P@{k}   vote acc  macro-F1  kappa  N1 prec  N1 recall  ANN recall")
    for space, entry in report["spaces"].items():
        vote = entry["knn_vote"]
        n1 = vote["per_class"]["N1"]
        print(f"{space:<12} {entry['precision_at_k']:.3f}  {vote['accuracy']:.3f}     "
              f"{vote['macro_f1']:.3f}     {vote['kappa']:.3f}  {n1['precision'] or 0:.3f}    "
              f"{n1['recall'] or 0:.3f}      {entry['ann_recall_at_k']:.3f}")
    for name, baseline in report["baselines"].items():
        print(f"{name:<24} acc {baseline['accuracy']:.3f}  macro-F1 {baseline['macro_f1']:.3f}")
    for name, block in report["disagreement"].items():
        print(f"\nDisagreement ({name}): {block['n_epochs']} epochs; correct: "
              + ", ".join(f"{m} {v:.3f}" for m, v in block["model_correct"].items()))
        print(f"{'space':<12} matches expert  sides {first}  sides {second}  neither  preference")
        for space, entry in block["spaces"].items():
            print(f"{space:<12} {entry['vote_matches_expert']:.3f}           "
                  f"{entry[f'sides_with_{first}']:.3f}              "
                  f"{entry[f'sides_with_{second}']:.3f}            "
                  f"{entry['sides_with_neither']:.3f}    {entry['preference']:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
