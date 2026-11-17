"""Library files and the Qdrant index built from them.

Files under ``LIBRARY_DIR`` are the source of truth, written once by
``scripts/build_library.py``. The Qdrant collection is a rebuildable index over them: a
search returns point ids, and a point id is the row number in every file below.

    manifest.json            provenance, counts, signal-feature statistics
    epochs.csv               id, participant, recording, epoch_index, stage, split, fold
    waveforms_100hz.npy      (N, 2, 3000) float16 microvolts: the epoch as displayed
    vectors_transformer.npy  (N, 128)  float32 Transformer CLS features, one fold
    vectors_attnsleep.npy    (N, 2400) float32 AttnSleep features, the same fold
    features_signal.npy      (N, 12)   float32 log band power, unstandardised
    probabilities.npz        transformer, attnsleep: (5 folds, N, 5) softmax outputs

All three spaces live in one collection as named vectors, so a point carries the same id
and payload whichever space a researcher searches in.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

LIBRARY_ID = "a2-sleep-edf-v1"
SYNTHETIC_LIBRARY_ID = "synthetic-sample-v1"  # scripts/build_library.py synthetic
EPOCH_COLUMNS = ("id", "participant", "recording", "epoch_index", "stage", "split",
                 "validation_fold")
TEST_FOLD = -1  # ``validation_fold`` of held-out test participants


@dataclass(frozen=True)
class Space:
    name: str
    dim: int
    distance: models.Distance
    release: str | None  # model release whose features span the space; None = model-free
    file: str


SPACES = {
    "transformer": Space("transformer", 128, models.Distance.COSINE, "transformer-a2-v1",
                         "vectors_transformer.npy"),
    "attnsleep": Space("attnsleep", 2400, models.Distance.COSINE, "attnsleep-a2-v1",
                       "vectors_attnsleep.npy"),
    "signal": Space("signal", 12, models.Distance.EUCLID, None, "features_signal.npy"),
}
MODEL_SPACES = tuple(name for name, space in SPACES.items() if space.release)
PAYLOAD_INDEXES = {
    "participant": models.PayloadSchemaType.KEYWORD,
    "stage": models.PayloadSchemaType.INTEGER,
    "split": models.PayloadSchemaType.KEYWORD,
}


def collection_name(manifest: dict) -> str:
    return f"{manifest['library_id']}-fold{manifest['fold']}"


def load_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run scripts/build_library.py extract.")
    return json.loads(path.read_text())


def load_epochs(directory: Path) -> dict[str, np.ndarray]:
    """``epochs.csv`` as one array per column, indexed by point id."""
    with (directory / "epochs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = {name: np.array([row[name] for row in rows]) for name in EPOCH_COLUMNS}
    for name in ("id", "epoch_index", "stage", "validation_fold"):
        columns[name] = columns[name].astype(np.int64)
    if not np.array_equal(columns["id"], np.arange(len(rows))):
        raise ValueError("epochs.csv ids must be 0..N-1 in order.")
    return columns


def standardize_signal(features: np.ndarray, manifest: dict) -> np.ndarray:
    """Z-score band powers with the library's statistics so every band weighs the same."""
    stats = manifest["signal_features"]
    return ((features - np.asarray(stats["mean"])) / np.asarray(stats["std"])).astype(np.float32)


def out_of_fold(probabilities: np.ndarray, split: np.ndarray,
                validation_fold: np.ndarray) -> np.ndarray:
    """``(5, N, 5)`` fold outputs -> ``(N, 5)`` predictions from models that never trained on
    the epoch: the fold that validated on it (development) or the ensemble (test)."""
    result = probabilities.mean(axis=0)
    development = split == "development"
    folds = validation_fold[development] - 1
    result[development] = probabilities[folds, np.flatnonzero(development)]
    return result


def embedding_training_mask(epochs: dict[str, np.ndarray], fold: int) -> np.ndarray:
    """True where the fold that produced the vectors trained on the epoch."""
    return (epochs["split"] == "development") & (epochs["validation_fold"] != fold)


# --- Qdrant ---

def connect(url: str, grpc_port: int, timeout: int = 300) -> QdrantClient:
    return QdrantClient(url=url, grpc_port=grpc_port, prefer_grpc=True, timeout=timeout)


def create_collection(client: QdrantClient, name: str) -> None:
    """HNSW is disabled (``m=0``) so the bulk upload is fast; ``build_hnsw`` enables it."""
    vectors = {}
    for space in SPACES.values():
        params = {"size": space.dim, "distance": space.distance}
        if space.dim > 1000:  # 2400-d floats: keep originals on disk, int8 copies in RAM
            params.update(on_disk=True, quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True)))
        vectors[space.name] = models.VectorParams(**params)
    client.create_collection(name, vectors_config=vectors,
                             hnsw_config=models.HnswConfigDiff(m=0))
    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field, field_schema=schema)


def build_hnsw(client: QdrantClient, name: str) -> None:
    client.update_collection(name, hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128))


def payload(epochs: dict[str, np.ndarray], index: int) -> dict:
    return {"participant": str(epochs["participant"][index]),
            "recording": str(epochs["recording"][index]),
            "stage": int(epochs["stage"][index]),
            "split": str(epochs["split"][index])}


def search_params(space: str) -> models.SearchParams:
    quantization = None
    if SPACES[space].dim > 1000:
        quantization = models.QuantizationSearchParams(rescore=True, oversampling=2.0)
    return models.SearchParams(hnsw_ef=128, quantization=quantization)


def excluding_participant(participant: str) -> models.Filter:
    """Neighbours from the same person are near-duplicates, not independent evidence."""
    return models.Filter(must_not=[models.FieldCondition(
        key="participant", match=models.MatchValue(value=participant))])


def search_batch(client: QdrantClient, name: str, space: str, vectors: np.ndarray,
                 exclude_participants: list[str | None], *, limit: int = 5,
                 batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Top-``limit`` point ids and scores per query vector, nearest first.

    Cosine scores are similarities (higher is closer); Euclid scores are distances (lower is
    closer). Missing neighbours are ``-1`` / NaN.
    """
    ids = np.full((len(vectors), limit), -1, dtype=np.int64)
    scores = np.full((len(vectors), limit), np.nan, dtype=np.float32)
    params = search_params(space)
    for start in range(0, len(vectors), batch_size):
        requests = [
            models.QueryRequest(
                query=np.asarray(vector, dtype=np.float32).tolist(), using=space, limit=limit,
                filter=None if participant is None else excluding_participant(participant),
                params=params, with_payload=False)
            for vector, participant in zip(vectors[start:start + batch_size],
                                           exclude_participants[start:start + batch_size],
                                           strict=True)
        ]
        for row, response in enumerate(client.query_batch_points(name, requests=requests),
                                       start=start):
            for column, point in enumerate(response.points):
                ids[row, column] = int(point.id)
                scores[row, column] = point.score
    return ids, scores
