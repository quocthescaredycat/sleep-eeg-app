"""Similarity search: epochs of an uploaded recording against the expert-scored library.

Neighbours are evidence for a researcher to inspect, not a prediction: the expert stage is
the reference for library epochs only. Qdrant returns ids; everything shown about a
neighbour comes from the ``library_epochs`` table and the library files.
"""

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...library import store
from ...library.metrics import majority_vote
from ...library.service import SPACE_LABELS, Library, SimilarityError, space_evaluation
from ...models import Job, JobStatus, LibraryEpoch
from ...schemas import (
    LibraryEpochPublic,
    LibraryEpochWaveform,
    LibraryPublic,
    ScoreKind,
    SimilarPublic,
    SpaceName,
)
from ...services import storage
from ...services.edf import CHANNELS, EPOCH_SECONDS
from ..deps import ApiError, ReleasesDep, SessionDep, SimilarityDep
from .recordings import DISPLAY_SAMPLING_RATE, get_recording

router = APIRouter(tags=["library"])


def api_error(error: SimilarityError) -> ApiError:
    return ApiError(error.status, error.code, str(error))


def score_kind(space: str) -> ScoreKind:
    cosine = store.SPACES[space].distance.value == "Cosine"
    return "cosine_similarity" if cosine else "euclidean_distance"


def epoch_public(row: LibraryEpoch) -> dict:
    fields = {name: getattr(row, name)
              for name in LibraryEpochPublic.model_fields if name != "onset_seconds"}
    return {**fields, "onset_seconds": row.epoch_index * EPOCH_SECONDS}


def ready_library(session: Session, similarity) -> Library:
    """The library, once its files, Qdrant collection and table all hold every epoch."""
    library = similarity.library()
    similarity.check_index(library)
    similarity.check_table(
        library, lambda: session.scalar(select(func.count()).select_from(LibraryEpoch)))
    return library


def load_features(path: Path, n_epochs: int, dim: int) -> np.ndarray:
    if not path.is_file():
        raise ApiError(409, "FEATURES_NOT_READY",
                       "This result has no similarity features yet. The worker saves them "
                       "with every new result and adds them to older ones when it starts.")
    features = np.load(path, mmap_mode="r")
    if features.shape != (n_epochs, dim):
        raise ApiError(409, "FEATURES_NOT_READY",
                       f"Saved features are shaped {features.shape}, not ({n_epochs}, {dim}).")
    return features


def describe(session: Session, ids: list[int]) -> dict[int, tuple[str, str, int]]:
    rows = session.execute(
        select(LibraryEpoch.id, LibraryEpoch.recording, LibraryEpoch.participant,
               LibraryEpoch.epoch_index).where(LibraryEpoch.id.in_(ids))).all()
    return {row.id: (row.recording, row.participant, row.epoch_index) for row in rows}


@router.get("/library", response_model=LibraryPublic)
def read_library(session: SessionDep, releases: ReleasesDep, similarity: SimilarityDep):
    """The reference library, whether this server can search it, and how each space scored
    in the library evaluation. Always ``200``: ``available`` and ``code`` say what is missing.
    """
    try:
        library = similarity.library()
    except SimilarityError as error:
        return LibraryPublic(available=False, code=error.code, message=str(error),
                             library_id=None, synthetic=False, fold=None, n_epochs=None,
                             n_participants=None, n_recordings=None, spaces=[], unanimous=None)
    problem = None
    try:
        ready_library(session, similarity)
    except SimilarityError as error:
        problem = error
    spaces = []
    for name, space in store.SPACES.items():
        reason = str(problem) if problem else similarity.space_problem(library, name, releases)
        spaces.append({"name": name, "label": SPACE_LABELS[name], "model_id": space.release,
                       "score": score_kind(name), "available": reason is None,
                       "reason": reason, "evaluation": space_evaluation(library.evaluation, name)})
    evaluation = library.evaluation or {}
    unanimous = evaluation.get("agreement_across_spaces", {}).get("all", {}).get("unanimous")
    manifest = library.manifest
    return LibraryPublic(
        available=problem is None, code=problem.code if problem else None,
        message=str(problem) if problem else None, library_id=manifest["library_id"],
        synthetic=bool(manifest.get("synthetic", False)), fold=library.fold,
        n_epochs=library.n_epochs, n_participants=manifest.get("n_participants"),
        n_recordings=manifest.get("n_recordings"), spaces=spaces, unanimous=unanimous)


@router.get("/library/epochs/{library_epoch_id}", response_model=LibraryEpochWaveform)
def read_library_epoch(library_epoch_id: int, session: SessionDep, similarity: SimilarityDep):
    """One library epoch: expert stage, out-of-fold model stages and 30 s of filtered EEG at
    100 Hz in microvolts, exactly as the models saw it."""
    try:
        library = similarity.library()
    except SimilarityError as error:
        raise api_error(error) from error
    row = session.get(LibraryEpoch, library_epoch_id) \
        if 0 <= library_epoch_id < library.n_epochs else None
    if row is None:
        raise ApiError(404, "LIBRARY_EPOCH_NOT_FOUND", "No library epoch has this id.")
    window = np.asarray(library.waveforms[library_epoch_id], dtype=np.float32)
    return LibraryEpochWaveform(
        **epoch_public(row), sampling_rate=DISPLAY_SAMPLING_RATE, unit="uV",
        channels=[{"label": label, "samples": np.round(samples, 2).tolist()}
                  for label, samples in zip(CHANNELS, window, strict=True)])


@router.get("/recordings/{recording_id}/epochs/{epoch_index}/similar",
            response_model=SimilarPublic)
def find_similar(recording_id: uuid.UUID, epoch_index: int, space: SpaceName,
                 session: SessionDep, releases: ReleasesDep, similarity: SimilarityDep,
                 k: Annotated[int, Query(ge=1, le=20)] = 5):
    """The ``k`` library epochs nearest to one epoch of this recording, in one space.

    A model space compares the features that model's fold saved with this recording's
    result, so the model must have been run here; the signal space needs only the
    preprocessing done by the first job. When the recording is itself a library night,
    neighbours from its participant are left out: they would include the epoch itself.
    """
    recording = get_recording(session, recording_id)
    if not 0 <= epoch_index < recording.n_epochs:
        raise ApiError(404, "EPOCH_NOT_FOUND",
                       f"Epoch index must be between 0 and {recording.n_epochs - 1}.")
    if not recording.waveform_ready:
        raise ApiError(409, "WAVEFORM_NOT_READY",
                       "Similarity search is available after the first job has finished.")
    try:
        library = ready_library(session, similarity)
    except SimilarityError as error:
        raise api_error(error) from error
    reason = similarity.space_problem(library, space, releases)
    if reason:
        raise ApiError(409, "SPACE_UNAVAILABLE", reason)

    band_power = load_features(storage.signal_features_path(recording_id), recording.n_epochs,
                               store.SPACES["signal"].dim)
    if space == "signal":
        vector = store.standardize_signal(np.asarray(band_power[epoch_index]), library.manifest)
    else:
        model_id = store.SPACES[space].release
        job = session.scalars(select(Job.id).where(
            Job.recording_id == recording_id, Job.model_id == model_id,
            Job.status == JobStatus.SUCCEEDED)).first()
        if job is None:
            raise ApiError(409, "MODEL_NOT_RUN",
                           f"Run {releases[model_id].name} on this recording first: "
                           "the features of its fold span this space.")
        features = load_features(storage.embeddings_path(recording_id, model_id, library.fold),
                                 recording.n_epochs, store.SPACES[space].dim)
        vector = np.asarray(features[epoch_index], dtype=np.float32)

    match = similarity.library_match(library, recording_id, band_power,
                                     lambda ids: describe(session, ids))
    excluded = match.participant if match else None
    try:
        hits = similarity.search(library, space, vector, excluded, k)
    except SimilarityError as error:
        raise api_error(error) from error
    rows = {row.id: row for row in session.scalars(
        select(LibraryEpoch).where(LibraryEpoch.id.in_([point for point, _ in hits])))}
    if len(rows) != len(hits):
        raise ApiError(503, "LIBRARY_NOT_LOADED",
                       "Qdrant returned epochs the library_epochs table does not hold.")
    stages = np.array([[rows[point].stage for point, _ in hits]], dtype=np.int64)
    vote = None
    if hits:
        stage = int(majority_vote(stages)[0])
        vote = {"stage": stage, "count": int((stages == stage).sum())}
    return SimilarPublic(
        recording_id=recording_id, epoch_index=epoch_index, space=space, k=k,
        score=score_kind(space), library_id=library.manifest["library_id"],
        library_match=None if match is None else asdict(match), excluded_participant=excluded,
        neighbours=[{"rank": rank, "score": round(score, 6), "epoch": epoch_public(rows[point])}
                    for rank, (point, score) in enumerate(hits, start=1)],
        vote=vote)
