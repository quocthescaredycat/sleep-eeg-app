"""Locations of files written at runtime under ``DATA_DIR``."""

import hashlib
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ..core.config import get_settings

CHUNK = 1 << 20


class UploadTooLarge(Exception):
    pass


def recording_dir(recording_id: uuid.UUID) -> Path:
    return get_settings().DATA_DIR / "recordings" / str(recording_id)


def source_path(recording_id: uuid.UUID) -> Path:
    return recording_dir(recording_id) / "source.edf"


def hypnogram_path(recording_id: uuid.UUID) -> Path:
    return recording_dir(recording_id) / "hypnogram.edf"


def waveform_path(recording_id: uuid.UUID) -> Path:
    return recording_dir(recording_id) / "display_100hz.npy"


def signal_features_path(recording_id: uuid.UUID) -> Path:
    """``(n_epochs, 12)`` log band power: the model-free similarity space."""
    return recording_dir(recording_id) / "signal_features.npy"


def embeddings_path(recording_id: uuid.UUID, model_id: str, fold: int) -> Path:
    """``(n_epochs, d)`` features of one model fold, saved with the job's predictions."""
    return recording_dir(recording_id) / f"features_{model_id}_fold{fold}.npy"


def save_array(path: Path, array: np.ndarray) -> None:
    """Write atomically, so a reader never sees half a file. The directory must exist: a
    recording deleted meanwhile is not recreated."""
    partial = path.with_name(f"{path.stem}.partial.npy")
    np.save(partial, array)
    partial.replace(path)


def temp_dir() -> Path:
    path = get_settings().DATA_DIR / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_stream(stream: BinaryIO, max_bytes: int) -> tuple[Path, str, int]:
    """Copy an upload to a temporary file; return ``(path, sha256, size)``."""
    path = temp_dir() / f"{uuid.uuid4()}.edf"  # MNE picks its reader by extension
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("wb") as target:
            while chunk := stream.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLarge
                digest.update(chunk)
                target.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path, digest.hexdigest(), size


def move_into_place(temporary: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(temporary, target)


def delete_recording_files(recording_id: uuid.UUID) -> None:
    shutil.rmtree(recording_dir(recording_id), ignore_errors=True)
