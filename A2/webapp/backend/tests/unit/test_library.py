"""Similarity library: embeddings, model-free features, retrieval metrics and the index."""

import numpy as np
import pytest
import torch
from qdrant_client import QdrantClient

from app.library import metrics, store
from app.ml.embeddings import EMBEDDING_DIMS, embed_and_classify
from app.ml.networks import NETWORKS
from app.ml.signal_features import FEATURE_NAMES, signal_features

KWARGS = {"Transformer": {"in_channels": 2, "emb_dim": 128, "num_heads": 4, "num_layers": 3,
                          "mlp_ratio": 3, "num_classes": 5, "dropout": 0.1},
          "AttnSleep": {"channel_index": 0}}


def epochs(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.3, (n, 2, 30, 200)).astype(np.float32)


@pytest.mark.parametrize("network", list(NETWORKS))
def test_classifying_the_embedding_reproduces_forward(network):
    torch.manual_seed(0)
    model = NETWORKS[network](**KWARGS[network]).eval()
    batch = torch.from_numpy(epochs(8))
    with torch.inference_mode():
        features = model.embed(batch)
        np.testing.assert_allclose(model.classify(features).numpy(), model(batch).numpy(),
                                   atol=1e-6)
    assert features.shape == (8, EMBEDDING_DIMS[network])


def test_embed_and_classify_returns_softmax_and_is_batch_independent():
    torch.manual_seed(0)
    model = NETWORKS["Transformer"](**KWARGS["Transformer"]).eval()
    data = epochs(10)
    features, probabilities = embed_and_classify(model, data, batch_size=3)
    whole, _ = embed_and_classify(model, data, batch_size=10)
    np.testing.assert_allclose(features, whole, atol=1e-5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_signal_features_put_a_10_hz_rhythm_in_the_alpha_band():
    t = np.arange(6000) / 200
    data = np.zeros((1, 2, 6000), dtype=np.float32)
    data[0, 0] = 0.2 * np.sin(2 * np.pi * 10 * t)  # 20 uV alpha on Fpz-Cz
    data[0, 1] = 0.2 * np.sin(2 * np.pi * 1 * t)   # 20 uV slow wave on Pz-Oz
    features = signal_features(data.reshape(1, 2, 30, 200))
    assert features.shape == (1, len(FEATURE_NAMES)) and np.isfinite(features).all()
    names = np.array(FEATURE_NAMES)
    assert names[features[0, :6].argmax()] == "EEG Fpz-Cz:alpha"
    assert names[6 + features[0, 6:].argmax()] == "EEG Pz-Oz:slow"


def test_majority_vote_breaks_ties_towards_the_nearest_neighbour():
    stages = np.array([[2, 2, 1, 1, 3],    # 2 and 1 tie: 2 is nearer
                       [1, 2, 2, 1, 3],    # tie again: 1 is nearer
                       [4, 4, 4, 0, 0],    # clear majority
                       [0, -1, -1, -1, -1],
                       [-1, -1, -1, -1, -1]])
    assert metrics.majority_vote(stages).tolist() == [2, 1, 4, 0, -1]


def test_precision_at_k_ignores_missing_neighbours():
    stages = np.array([[2, 2, 1, -1, -1], [0, 0, 0, 0, 0]])
    np.testing.assert_allclose(metrics.precision_at_k(stages, np.array([2, 1])), [2 / 3, 0.0])


def test_disagreement_report_measures_which_model_a_space_sides_with():
    truth = np.array([1, 2, 2, 3, 0])
    first = np.array([1, 2, 1, 3, 0])   # agrees with second on the last two epochs
    second = np.array([2, 1, 2, 3, 0])
    votes = {"leans_first": np.array([1, 2, 1, 3, 0]), "expert": truth.copy()}
    report = metrics.disagreement(truth, {"a": first, "b": second}, votes)
    assert report["n_epochs"] == 3
    leans = report["spaces"]["leans_first"]
    assert leans["sides_with_a"] == 1.0 and leans["preference"] == 1.0
    assert leans["a_correct_when_sided_with"] == pytest.approx(2 / 3)
    assert report["spaces"]["expert"]["vote_matches_expert"] == 1.0


def test_out_of_fold_never_uses_a_model_that_trained_on_the_epoch():
    probabilities = np.zeros((5, 3, 5), dtype=np.float32)
    for fold in range(5):
        probabilities[fold, :, fold] = 1.0  # fold k always predicts class k-1
    split = np.array(["development", "development", "test"])
    folds = np.array([2, 5, store.TEST_FOLD])
    result = store.out_of_fold(probabilities, split, folds)
    assert result[0].argmax() == 1 and result[1].argmax() == 4
    np.testing.assert_allclose(result[2], 0.2)


@pytest.mark.filterwarnings("ignore:.*local Qdrant:UserWarning",
                            "ignore:Local mode performs exact:UserWarning")
def test_search_excludes_the_query_participant_and_returns_nearest_first():
    client = QdrantClient(":memory:")
    store.create_collection(client, "library")
    rng = np.random.default_rng(0)
    base = {name: rng.normal(size=(6, space.dim)).astype(np.float32)
            for name, space in store.SPACES.items()}
    base["transformer"][1] = base["transformer"][0] + 0.01  # participant P0's twin
    base["transformer"][2] = base["transformer"][0] + 0.05  # the nearest other person
    epochs_ = {"participant": np.array(["P0", "P0", "P1", "P1", "P2", "P2"]),
               "recording": np.array(["R0", "R0", "R1", "R1", "R2", "R2"]),
               "stage": np.array([0, 1, 2, 3, 4, 0]), "split": np.array(["test"] * 6)}
    from qdrant_client import models

    client.upsert("library", points=models.Batch(
        ids=list(range(6)), vectors={name: array.tolist() for name, array in base.items()},
        payloads=[store.payload(epochs_, row) for row in range(6)]))
    ids, scores = store.search_batch(client, "library", "transformer", base["transformer"][:1],
                                     ["P0"], limit=3)
    assert ids[0, 0] == 2 and not {0, 1} & set(ids[0].tolist())
    assert scores[0, 0] >= scores[0, 1] >= scores[0, 2]
    unfiltered, _ = store.search_batch(client, "library", "transformer",
                                       base["transformer"][:1], [None], limit=2)
    assert unfiltered[0].tolist() == [0, 1]


def test_signal_space_is_standardised_with_library_statistics():
    manifest = {"signal_features": {"mean": [1.0] * 12, "std": [2.0] * 12}}
    np.testing.assert_allclose(store.standardize_signal(np.full((1, 12), 5.0), manifest), 2.0)
