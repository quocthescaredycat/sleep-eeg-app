"""Pydantic request and response models."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPublic(BaseModel):
    id: str
    name: str
    network: str
    input_channels: list[str]
    preprocessing_version: str
    available: bool
    synthetic: bool
    published_metrics: dict | None


class ChannelPublic(BaseModel):
    label: str
    sampling_rate: float
    unit: str


class JobCreate(BaseModel):
    recording_id: uuid.UUID
    model_id: str


class JobPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    recording_id: uuid.UUID
    model_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str | None
    progress: float
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class RecordingPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    sha256: str
    size_bytes: int
    duration_seconds: float
    n_epochs: int
    dropped_seconds: float
    channels: list[ChannelPublic]
    has_hypnogram: bool
    is_sample: bool
    waveform_ready: bool
    created_at: datetime
    expires_at: datetime | None
    jobs: list[JobPublic]


class RecordingsPublic(BaseModel):
    data: list[RecordingPublic]
    count: int


class EpochChannel(BaseModel):
    label: str
    samples: list[float]


class EpochPublic(BaseModel):
    recording_id: uuid.UUID
    epoch_index: int
    onset_seconds: float
    sampling_rate: float
    unit: str
    channels: list[EpochChannel]
    reference_stage: int | None


class PredictionsPublic(BaseModel):
    job_id: uuid.UUID
    recording_id: uuid.UUID
    model_id: str
    class_names: list[str]
    epoch_seconds: int
    stages: list[int]
    probabilities: list[list[float]]
    reference_stages: list[int | None] | None


class ClassMetrics(BaseModel):
    name: str
    precision: float | None
    recall: float | None
    f1: float | None
    support: int


class MetricsPublic(BaseModel):
    job_id: uuid.UUID
    scope: Literal["all", "benchmark"]
    n_epochs: int
    n_scored_epochs: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    kappa: float
    class_names: list[str]
    per_class: list[ClassMetrics]
    confusion_matrix: list[list[int]]


class HealthPublic(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]
    models_available: int


# --- Similarity search ---

SpaceName = Literal["transformer", "attnsleep", "signal"]
ScoreKind = Literal["cosine_similarity", "euclidean_distance"]


class SpaceDisagreement(BaseModel):
    """Library evaluation, on query epochs where the two fold models disagree."""

    n_epochs: int
    matches_expert: float
    sides_with: dict[str, float] = Field(
        description="Model space name -> share of 5-NN votes equal to that model's stage.")


class SpaceEvaluation(BaseModel):
    """What ``scripts/evaluate_library.py`` measured for a space on epochs its fold never
    trained on, neighbours from the query's participant excluded."""

    k: int
    n_queries: int
    precision_at_k: float
    vote_accuracy: float
    vote_macro_f1: float
    vote_precision: list[float | None] = Field(
        description="By stage index: how often a vote for that stage matches the expert.")
    disagreement: SpaceDisagreement | None


class SimilaritySpacePublic(BaseModel):
    name: SpaceName
    label: str
    model_id: str | None = Field(description="Model whose features span the space; "
                                             "null for the model-free signal space.")
    score: ScoreKind
    available: bool
    reason: str | None
    evaluation: SpaceEvaluation | None


class Consensus(BaseModel):
    coverage: float
    accuracy: float


class LibraryPublic(BaseModel):
    """The reference library, and whether this server can search it."""

    available: bool
    code: str | None
    message: str | None
    library_id: str | None
    synthetic: bool
    fold: int | None
    n_epochs: int | None
    n_participants: int | None
    n_recordings: int | None
    spaces: list[SimilaritySpacePublic]
    unanimous: Consensus | None = Field(
        description="Library evaluation: queries where every space's vote agrees.")


class LibraryEpochPublic(BaseModel):
    """One expert-scored library epoch. Model stages are out of fold: predicted by a fold
    that did not train on the epoch's participant (the five-fold mean for test ones)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    participant: str
    recording: str
    epoch_index: int
    onset_seconds: int
    stage: int
    split: Literal["development", "test"]
    in_embedding_training: bool = Field(
        description="The fold that produced the model-space vectors trained on this epoch.")
    transformer_stage: int
    transformer_confidence: float
    attnsleep_stage: int
    attnsleep_confidence: float


class LibraryEpochWaveform(LibraryEpochPublic):
    sampling_rate: float
    unit: str
    channels: list[EpochChannel]


class SimilarNeighbour(BaseModel):
    rank: int
    score: float
    epoch: LibraryEpochPublic


class LibraryMatchPublic(BaseModel):
    recording: str
    participant: str
    epoch_offset: int


class Vote(BaseModel):
    stage: int
    count: int


class SimilarPublic(BaseModel):
    recording_id: uuid.UUID
    epoch_index: int
    space: SpaceName
    k: int
    score: ScoreKind = Field(description="cosine_similarity: higher is closer; "
                                         "euclidean_distance: lower is closer.")
    library_id: str
    library_match: LibraryMatchPublic | None = Field(
        description="Set when this recording is itself a library night.")
    excluded_participant: str | None
    neighbours: list[SimilarNeighbour]
    vote: Vote | None = Field(
        description="Majority expert stage of the neighbours; ties go to the nearest.")
