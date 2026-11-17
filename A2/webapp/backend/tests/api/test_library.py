"""Similarity search API against a synthetic library and an in-process Qdrant.

The library comes from ``scripts/build_library.py synthetic``: the bundled sample stored as
night SC4031 from epoch 980 (as in the real library), a rescaled noisy copy (SX001) and a
copy shifted by seven epochs (SX002), all embedded with the suite's synthetic checkpoints.
``QdrantClient(":memory:")`` stands in for the server, so CI needs no Qdrant service.
"""

import argparse
import dataclasses
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient
from sqlalchemy import insert, text

from app.api.deps import get_releases, get_similarity
from app.core.config import Settings
from app.core.db import get_engine
from app.library import store
from app.library.service import SimilarityService
from app.ml.embeddings import embed_and_classify
from app.ml.preprocessing import preprocess_recording
from app.ml.registry import load_releases
from app.ml.signal_features import signal_features
from app.models import LibraryEpoch
from app.services import storage
from scripts import build_library
from tests.conftest import ATTNSLEEP, SAMPLE_PSG, TRANSFORMER, edf_variant, upload

pytestmark = pytest.mark.filterwarnings("ignore:.*local Qdrant:UserWarning",
                                        "ignore:Local mode:UserWarning")

SPACES = ("transformer", "attnsleep", "signal")


def error(response) -> str:
    return response.json()["detail"]["code"]


def rescaled_edf(factor: float) -> bytes:
    """The sample with every physical range scaled: a valid EDF that no library night equals."""
    content = bytearray(edf_variant(f"rescaled {factor}"))
    count = int(content[252:256])
    for offset in (256 + count * 104, 256 + count * 112):  # physical minimum, maximum
        for signal in range(count):
            field = slice(offset + 8 * signal, offset + 8 * (signal + 1))
            content[field] = f"{float(content[field]) * factor:g}".ljust(8).encode("ascii")
    return bytes(content)


@pytest.fixture(scope="session")
def library_dir(environment, tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("library")
    build_library.synthetic(argparse.Namespace(
        dir=directory, checkpoints=Path(os.environ["CHECKPOINTS_DIR"]), fold=5))
    return directory


@pytest.fixture(scope="session")
def qdrant(library_dir) -> QdrantClient:
    client = QdrantClient(":memory:")
    name = store.collection_name(store.load_manifest(library_dir))
    store.create_collection(client, name)
    build_library.upload_points(client, library_dir, name, wait=True)
    return client


def load_table(directory: Path | None) -> None:
    with get_engine().begin() as connection:
        connection.execute(text("TRUNCATE library_epochs"))
        if directory is not None:
            connection.execute(insert(LibraryEpoch), build_library.library_rows(directory))


def serve(directory: Path, qdrant_client) -> SimilarityService:
    from app.main import app

    service = SimilarityService(Settings(LIBRARY_DIR=directory), client=qdrant_client)
    app.dependency_overrides[get_similarity] = lambda: service
    return service


@pytest.fixture
def library(library_dir, qdrant):
    """The synthetic library, indexed and loaded, served by the API."""
    from app.main import app

    load_table(library_dir)
    yield serve(library_dir, qdrant)
    app.dependency_overrides.clear()
    load_table(None)


@pytest.fixture
def sample(client, worker, library):
    """The bundled sample with its hypnogram and both models run."""
    recording = upload(client, hypnogram=True, model_ids=(TRANSFORMER, ATTNSLEEP)).json()
    assert worker.run_pending() == 2
    return recording


@pytest.fixture
def transformer_sample(client, worker, library):
    """The bundled sample with the Transformer run: enough for most checks, and faster."""
    recording = upload(client, model_ids=(TRANSFORMER,)).json()
    assert worker.run_pending() == 1
    return recording


def similar(client, recording_id: str, epoch: int, space: str, **params):
    return client.get(f"/api/v1/recordings/{recording_id}/epochs/{epoch}/similar",
                      params={"space": space, **params})


def test_status_without_a_library(client):
    status = client.get("/api/v1/library").json()
    assert (status["available"], status["code"]) == (False, "LIBRARY_NOT_INSTALLED")
    assert status["spaces"] == [] and status["library_id"] is None


def test_status_describes_the_library_and_its_spaces(client, library):
    status = client.get("/api/v1/library").json()
    assert status["available"] is True and status["code"] is None
    assert (status["library_id"], status["synthetic"], status["fold"]) == \
        ("synthetic-sample-v1", True, 5)
    assert (status["n_epochs"], status["n_participants"], status["n_recordings"]) == (360, 3, 3)
    spaces = {space["name"]: space for space in status["spaces"]}
    assert list(spaces) == list(SPACES)
    assert all(space["available"] and space["reason"] is None for space in spaces.values())
    assert spaces["transformer"]["model_id"] == TRANSFORMER
    assert spaces["signal"]["model_id"] is None
    assert spaces["signal"]["score"] == "euclidean_distance"
    assert spaces["attnsleep"]["score"] == "cosine_similarity"
    assert spaces["transformer"]["evaluation"] is None  # no library_eval.json


def test_a_job_saves_the_query_features(sample, worker):
    recording_id = sample["id"]
    epochs = preprocess_recording(SAMPLE_PSG).epochs
    band_power = np.load(storage.signal_features_path(recording_id))
    np.testing.assert_allclose(band_power, signal_features(epochs), atol=1e-5)
    for model_id, dim in ((TRANSFORMER, 128), (ATTNSLEEP, 2400)):
        features = np.load(storage.embeddings_path(recording_id, model_id, 5))
        expected, _ = embed_and_classify(worker.ensemble(model_id)[4], epochs[:8])
        assert features.shape == (120, dim)
        np.testing.assert_allclose(features[:8], expected, atol=1e-4)


def test_a_library_night_is_recognised_and_its_participant_excluded(client, sample):
    recording_id = sample["id"]
    for space in SPACES:
        body = similar(client, recording_id, 10, space).json()
        assert body["library_match"] == {"recording": "SC4031", "participant": "SC403",
                                         "epoch_offset": 980}
        assert body["excluded_participant"] == "SC403"
        assert (body["k"], body["library_id"], body["space"]) == \
            (5, "synthetic-sample-v1", space)
        neighbours = body["neighbours"]
        assert [item["rank"] for item in neighbours] == [1, 2, 3, 4, 5]
        assert {item["epoch"]["participant"] for item in neighbours} <= {"SX001", "SX002"}
        scores = [item["score"] for item in neighbours]
        assert scores == sorted(scores, reverse=body["score"] == "cosine_similarity")

        stages = [item["epoch"]["stage"] for item in neighbours]
        assert body["vote"]["count"] == stages.count(body["vote"]["stage"])
        assert body["vote"]["count"] == max(stages.count(stage) for stage in stages)

    # The copies of an epoch are its nearest neighbours: query vectors (worker) and library
    # vectors (build script) share one space.
    for epoch in (10, 60, 100):
        body = similar(client, recording_id, epoch, "attnsleep", k=2).json()
        found = {(item["epoch"]["recording"], item["epoch"]["epoch_index"])
                 for item in body["neighbours"]}
        assert found == {("SX0011", epoch), ("SX0021", (epoch - 7) % 120)}


def test_neighbour_details_come_from_the_library(client, transformer_sample, library_dir):
    recording_id = transformer_sample["id"]
    body = similar(client, recording_id, 42, "transformer").json()
    rows = build_library.library_rows(library_dir)
    for item in body["neighbours"]:
        epoch = item["epoch"]
        row = rows[epoch["id"]]
        assert epoch["onset_seconds"] == 30 * row["epoch_index"]
        for name in ("participant", "recording", "epoch_index", "stage", "split",
                     "in_embedding_training", "transformer_stage", "attnsleep_stage"):
            assert epoch[name] == row[name], name

        detail = client.get(f"/api/v1/library/epochs/{epoch['id']}").json()
        assert {name: detail[name] for name in epoch} == epoch
        waveforms = np.load(library_dir / "waveforms_100hz.npy", mmap_mode="r")
        served = np.array([channel["samples"] for channel in detail["channels"]])
        assert served.shape == (2, 3000) and detail["sampling_rate"] == 100
        np.testing.assert_allclose(served, waveforms[epoch["id"]].astype(np.float32), atol=0.006)


def test_an_unrelated_recording_excludes_nobody(client, worker, library):
    recording = upload(client, rescaled_edf(0.5), model_ids=(TRANSFORMER,)).json()
    worker.run_pending()
    body = similar(client, recording["id"], 10, "signal").json()
    assert body["library_match"] is None and body["excluded_participant"] is None
    assert len(body["neighbours"]) == 5


def test_a_model_space_needs_that_model(client, worker, library):
    recording = upload(client, edf_variant("transformer only"), model_ids=(TRANSFORMER,)).json()
    worker.run_pending()
    assert similar(client, recording["id"], 3, "transformer").status_code == 200
    assert similar(client, recording["id"], 3, "signal").status_code == 200
    response = similar(client, recording["id"], 3, "attnsleep")
    assert response.status_code == 409 and error(response) == "MODEL_NOT_RUN"


def test_search_needs_a_preprocessed_recording_and_valid_parameters(client,
                                                                    transformer_sample):
    recording_id = transformer_sample["id"]
    pending = upload(client, edf_variant("no jobs")).json()
    response = similar(client, pending["id"], 0, "signal")
    assert response.status_code == 409 and error(response) == "WAVEFORM_NOT_READY"

    response = similar(client, recording_id, 120, "signal")
    assert response.status_code == 404 and error(response) == "EPOCH_NOT_FOUND"
    assert similar(client, recording_id, 0, "pixels").status_code == 422
    assert similar(client, recording_id, 0, "signal", k=0).status_code == 422
    assert similar(client, recording_id, 0, "signal", k=21).status_code == 422
    assert len(similar(client, recording_id, 0, "signal", k=20).json()["neighbours"]) == 20
    response = client.get("/api/v1/library/epochs/360")
    assert response.status_code == 404 and error(response) == "LIBRARY_EPOCH_NOT_FOUND"


def test_older_results_get_their_features_when_the_worker_starts(client, transformer_sample,
                                                                 worker):
    recording_id = transformer_sample["id"]
    paths = [storage.signal_features_path(recording_id),
             storage.embeddings_path(recording_id, TRANSFORMER, 5)]
    saved = [np.load(path) for path in paths]
    for path in paths:
        path.unlink()
    response = similar(client, recording_id, 5, "transformer")
    assert response.status_code == 409 and error(response) == "FEATURES_NOT_READY"

    assert worker.backfill_features() == 2
    assert worker.backfill_features() == 0
    for path, before in zip(paths, saved, strict=True):
        np.testing.assert_allclose(np.load(path), before, atol=1e-4)
    assert similar(client, recording_id, 5, "transformer").status_code == 200


def test_checkpoints_other_than_the_library_s_disable_their_space(client, transformer_sample):
    from app.main import app

    recording_id = transformer_sample["id"]

    releases = load_releases(os.environ["CHECKPOINTS_DIR"])
    other = releases[TRANSFORMER]
    folds = list(other.folds)
    folds[4] = {**folds[4], "sha256": "0" * 64}
    app.dependency_overrides[get_releases] = lambda: {
        **releases, TRANSFORMER: dataclasses.replace(other, folds=tuple(folds))}

    spaces = {space["name"]: space for space in client.get("/api/v1/library").json()["spaces"]}
    assert spaces["transformer"]["available"] is False
    assert "other Transformer weights" in spaces["transformer"]["reason"]
    assert spaces["attnsleep"]["available"] and spaces["signal"]["available"]
    response = similar(client, recording_id, 5, "transformer")
    assert response.status_code == 409 and error(response) == "SPACE_UNAVAILABLE"


class Unreachable:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise ConnectionError("connection refused")
        return fail


def test_the_library_is_unavailable_until_qdrant_and_postgresql_hold_it(
        client, transformer_sample, library_dir, qdrant):
    recording_id = transformer_sample["id"]
    serve(library_dir, Unreachable())
    status = client.get("/api/v1/library").json()
    assert (status["available"], status["code"]) == (False, "SIMILARITY_UNAVAILABLE")
    assert all(not space["available"] for space in status["spaces"])
    response = similar(client, recording_id, 5, "signal")
    assert response.status_code == 503 and error(response) == "SIMILARITY_UNAVAILABLE"

    serve(library_dir, QdrantClient(":memory:"))  # running, but nothing indexed
    assert client.get("/api/v1/library").json()["code"] == "LIBRARY_NOT_INDEXED"

    service = serve(library_dir, qdrant)
    load_table(None)
    assert client.get("/api/v1/library").json()["code"] == "LIBRARY_NOT_LOADED"
    load_table(library_dir)
    assert client.get("/api/v1/library").json()["available"] is True

    # Qdrant stopping later is noticed on the next request.
    service._client = Unreachable()
    assert client.get("/api/v1/library").json()["code"] == "SIMILARITY_UNAVAILABLE"


def test_a_rebuilt_library_is_reloaded(client, library, library_dir, tmp_path, qdrant):
    copy = tmp_path / "library"
    shutil.copytree(library_dir, copy)
    service = serve(copy, qdrant)
    assert client.get("/api/v1/library").json()["n_epochs"] == 360
    first = service.library()
    manifest = copy / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"n_epochs": 360', '"n_epochs": 361'))
    os.utime(manifest, ns=(first.version + 10**9, first.version + 10**9))
    status = client.get("/api/v1/library").json()
    assert status["code"] == "LIBRARY_NOT_INSTALLED"  # files no longer match the manifest
