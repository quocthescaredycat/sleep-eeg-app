"""Similarity search as the API serves it.

The API never imports PyTorch. Query vectors are files the worker saves next to each
result: the fold features of every finished model (``storage.embeddings_path``) and the
recording's band powers (``storage.signal_features_path``). The library side is the files
written by ``scripts/build_library.py``, their Qdrant collection and the ``library_epochs``
table; all three must describe the same library before a search runs.
"""

import json
import threading
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from ..core.config import Settings
from ..ml.registry import CLASS_NAMES, ModelRelease
from . import store

SPACE_LABELS = {"transformer": "Transformer", "attnsleep": "AttnSleep",
                "signal": "Signal (model-free)"}
# A night that is already in the library has epochs whose band powers equal library rows
# (measured: 0 in the interior of the bundled sample, at most 0.0024 next to its cut), while
# the nearest epoch of any other night lies at least 0.2 away. The library holds only the
# benchmark window of each night, so a whole uploaded night has twins for about a third of
# its epochs: sample widely, and require a few twins that agree on night and offset.
MATCH_TOLERANCE = 0.01
TABLE_CHECK_SECONDS = 60  # how long a passed library_epochs row count is trusted
MATCH_SAMPLES = 64
MATCH_MIN_TWINS = 3


class SimilarityError(Exception):
    """Search cannot run; ``code`` is the stable identifier the API returns."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class LibraryMatch:
    """The query recording is library night ``recording``: its epoch ``i`` is library
    epoch ``i + epoch_offset`` of that night."""

    recording: str
    participant: str
    epoch_offset: int


@dataclass(frozen=True)
class Library:
    directory: Path
    manifest: dict
    evaluation: dict | None
    waveforms: np.ndarray  # memory-mapped (N, 2, 3000) float16 microvolts
    signal: np.ndarray     # (N, 12) float64 unstandardised band powers, for matching
    version: int           # see library_version: a rebuilt library is reloaded

    @property
    def fold(self) -> int:
        return self.manifest["fold"]

    @property
    def n_epochs(self) -> int:
        return self.manifest["n_epochs"]

    @property
    def collection(self) -> str:
        return store.collection_name(self.manifest)


def library_version(directory: Path) -> int:
    """Changes when the library is rebuilt or (re-)evaluated."""
    evaluation = directory / "library_eval.json"
    return max((directory / "manifest.json").stat().st_mtime_ns,
               evaluation.stat().st_mtime_ns if evaluation.is_file() else 0)


def load_library(directory: Path) -> Library:
    manifest = store.load_manifest(directory)
    version = library_version(directory)
    count = manifest["n_epochs"]
    waveforms = np.load(directory / "waveforms_100hz.npy", mmap_mode="r")
    signal = np.load(directory / store.SPACES["signal"].file).astype(np.float64)
    if waveforms.shape[0] != count or signal.shape != (count, store.SPACES["signal"].dim):
        raise ValueError(f"{directory}: files do not match manifest.json.")
    evaluation_path = directory / "library_eval.json"
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.is_file() else None
    return Library(directory, manifest, evaluation, waveforms, signal, version)


def match_epochs(features: np.ndarray, library: np.ndarray) -> list[tuple[int, int]]:
    """``(query epoch, library id)`` for sampled query epochs that have an exact library twin.

    Sampling spans the recording but skips its first and last epoch, where a cut recording
    is filtered differently.
    """
    n = len(features)
    if n == 0 or len(library) == 0:
        return []
    picks = np.unique(np.linspace(min(1, n - 1), max(n - 2, 0), min(MATCH_SAMPLES, n))
                      .round().astype(int))
    rows = np.asarray(library, dtype=np.float64)
    norms = (rows ** 2).sum(axis=1)
    twins = []
    for chunk in np.array_split(picks, -(-len(picks) // 16)):  # bounds the distance matrix
        queries = np.asarray(features[chunk], dtype=np.float64)
        squared = norms[None, :] - 2 * queries @ rows.T + (queries ** 2).sum(axis=1)[:, None]
        nearest = squared.argmin(axis=1)
        gap = np.sqrt(np.maximum(squared[np.arange(len(chunk)), nearest], 0))
        twins += [(int(epoch), int(row)) for epoch, row, distance in zip(chunk, nearest, gap,
                                                                          strict=True)
                  if distance < MATCH_TOLERANCE]
    return twins


def find_library_match(features: np.ndarray, library: np.ndarray,
                       describe: Callable[[list[int]], dict[int, tuple[str, str, int]]],
                       ) -> LibraryMatch | None:
    """Is the query recording a library night? Then its participant must be excluded:
    otherwise the nearest neighbour is the epoch itself, expert stage included.

    ``describe`` maps library ids to ``(recording, participant, epoch_index)``. A match
    needs ``MATCH_MIN_TWINS`` twins in one night at one epoch offset, and those must be
    most of the twins found: identical band powers at a consistent offset do not happen
    by chance.
    """
    twins = match_epochs(features, library)
    if len(twins) < MATCH_MIN_TWINS:
        return None
    described = describe(sorted({row for _, row in twins}))
    votes = Counter()
    for epoch, row in twins:
        recording, participant, epoch_index = described[row]
        votes[(recording, participant, epoch_index - epoch)] += 1
    (recording, participant, offset), count = votes.most_common(1)[0]
    if count < MATCH_MIN_TWINS or 2 * count < len(twins):
        return None
    return LibraryMatch(recording, participant, offset)


def space_evaluation(evaluation: dict | None, space: str) -> dict | None:
    """What ``evaluate_library.py`` measured for one space, in the shape the UI shows."""
    if not evaluation or space not in evaluation.get("spaces", {}):
        return None
    entry = evaluation["spaces"][space]
    vote = entry["knn_vote"]
    fold = evaluation["library"]["fold"]
    block = evaluation.get("disagreement", {}).get(f"fold{fold}_models")
    disagreement = None
    if block and space in block["spaces"]:
        sided = block["spaces"][space]
        disagreement = {
            "n_epochs": block["n_epochs"],
            "matches_expert": sided["vote_matches_expert"],
            "sides_with": {name: sided[f"sides_with_{name}"] for name in store.MODEL_SPACES},
        }
    return {
        "k": evaluation["k"],
        "n_queries": evaluation["queries"]["n"],
        "precision_at_k": entry["precision_at_k"],
        "vote_accuracy": vote["accuracy"],
        "vote_macro_f1": vote["macro_f1"],
        "vote_precision": [vote["per_class"][name]["precision"] for name in CLASS_NAMES],
        "disagreement": disagreement,
    }


class SimilarityService:
    """One per API process: the loaded library, a Qdrant client and cached matches."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None):
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        self._library: Library | None = None
        self._table_checked: tuple[int, float] | None = None  # (library version, monotonic)
        self._matches: OrderedDict[tuple[uuid.UUID, int], LibraryMatch | None] = OrderedDict()
        self._lock = threading.Lock()

    def library(self) -> Library:
        directory = self.settings.LIBRARY_DIR
        try:
            version = library_version(directory)
            with self._lock:
                if self._library is None or self._library.version != version:
                    self._library = load_library(directory)
                    self._table_checked = None
                    self._matches.clear()
                return self._library
        except (OSError, ValueError, KeyError) as error:
            raise SimilarityError(
                503, "LIBRARY_NOT_INSTALLED",
                "No similarity library is installed on this server "
                "(see scripts/build_library.py).") from error

    def unreachable(self, error: Exception) -> SimilarityError:
        """After a failure, reconnect on the next request instead of waiting out gRPC's
        reconnection backoff (up to two minutes after a long outage)."""
        if self._owns_client:
            self._client = None
        return SimilarityError(503, "SIMILARITY_UNAVAILABLE",
                               "The similarity index (Qdrant) is not reachable.")

    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = store.connect(self.settings.QDRANT_URL, self.settings.QDRANT_GRPC_PORT,
                                         timeout=self.settings.QDRANT_TIMEOUT_SECONDS)
        return self._client

    def check_index(self, library: Library) -> None:
        """Asked on every request (about 1 ms): Qdrant can stop at any time."""
        try:
            client = self.client()
            points = (client.get_collection(library.collection).points_count
                      if client.collection_exists(library.collection) else 0)
        except Exception as error:  # gRPC and HTTP transports raise unrelated types
            raise self.unreachable(error) from error
        if points != library.n_epochs:
            raise SimilarityError(
                503, "LIBRARY_NOT_INDEXED",
                f"Qdrant collection {library.collection} holds {points} of "
                f"{library.n_epochs} epochs: run scripts/build_library.py index.")

    def check_table(self, library: Library, count_rows: Callable[[], int]) -> None:
        checked = self._table_checked
        if checked and checked[0] == library.version \
                and time.monotonic() - checked[1] < TABLE_CHECK_SECONDS:
            return
        rows = count_rows()
        if rows != library.n_epochs:
            raise SimilarityError(
                503, "LIBRARY_NOT_LOADED",
                f"Table library_epochs holds {rows} of {library.n_epochs} epochs: "
                "run scripts/build_library.py load-db.")
        self._table_checked = (library.version, time.monotonic())

    @staticmethod
    def space_problem(library: Library, space: str,
                      releases: dict[str, ModelRelease]) -> str | None:
        """Why a model space cannot be searched with this server's checkpoints, if it can't."""
        release_id = store.SPACES[space].release
        if release_id is None:
            return None
        release = releases.get(release_id)
        if release is None or not release.available:
            return f"Model {release_id} is not installed on this server."
        built = library.manifest.get("releases", {}).get(release_id, {}).get("folds_sha256", [])
        fold = library.fold
        if len(built) < fold or len(release.folds) < fold \
                or release.folds[fold - 1]["sha256"] != built[fold - 1]:
            return (f"The library was built with other {release.name} weights, so its vectors "
                    "and this server's do not share a space.")
        return None

    def library_match(self, library: Library, recording_id: uuid.UUID, features: np.ndarray,
                      describe: Callable[[list[int]], dict[int, tuple[str, str, int]]],
                      ) -> LibraryMatch | None:
        key = (recording_id, library.version)
        with self._lock:
            if key in self._matches:
                self._matches.move_to_end(key)
                return self._matches[key]
        match = find_library_match(features, library.signal, describe)
        with self._lock:
            self._matches[key] = match
            while len(self._matches) > 1024:
                self._matches.popitem(last=False)
        return match

    def search(self, library: Library, space: str, vector: np.ndarray,
               exclude_participant: str | None, k: int) -> list[tuple[int, float]]:
        try:
            ids, scores = store.search_batch(self.client(), library.collection, space,
                                             vector[None, :], [exclude_participant], limit=k)
        except Exception as error:
            raise self.unreachable(error) from error
        return [(int(point), float(score)) for point, score in zip(ids[0], scores[0], strict=True)
                if point >= 0]
