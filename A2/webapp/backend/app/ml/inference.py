"""Five-fold soft-voting inference on CPU. Imported by the worker and the CLI only."""

from collections.abc import Callable

import numpy as np
import torch

from .networks import NETWORKS
from .preprocessing import PREPROCESSING_VERSION
from .registry import CheckpointError, ModelRelease


def load_ensemble(release: ModelRelease) -> list[torch.nn.Module]:
    """Verify hashes, then load every fold with ``strict=True`` in eval mode."""
    if release.preprocessing.get("version") != PREPROCESSING_VERSION:
        raise CheckpointError(f"{release.id}: trained with a different preprocessing version.")
    release.verify()
    models = []
    for path in release.fold_paths:
        model = NETWORKS[release.network](**release.network_kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        models.append(model.eval())
    return models


@torch.inference_mode()
def _ensemble(models: list[torch.nn.Module], epochs: np.ndarray, batch_size: int,
              on_progress: Callable[[int, int], None] | None,
              embed_fold: int | None) -> tuple[np.ndarray, np.ndarray | None]:
    total = np.zeros((len(epochs), 5), dtype=np.float64)
    features = []
    steps = len(models) * -(-len(epochs) // batch_size)
    done = 0
    for number, model in enumerate(models, start=1):
        for start in range(0, len(epochs), batch_size):
            batch = torch.from_numpy(epochs[start:start + batch_size])
            if number == embed_fold:  # forward == classify(embed(x)): same logits, plus features
                embedded = model.embed(batch)
                features.append(embedded.float().numpy())
                logits = model.classify(embedded)
            else:
                logits = model(batch)
            total[start:start + batch_size] += torch.softmax(logits.double(), dim=1).numpy()
            done += 1
            if on_progress is not None:
                on_progress(done, steps)
    probabilities = (total / len(models)).astype(np.float32)
    return probabilities, np.concatenate(features).astype(np.float32) if features else None


def predict_proba(models: list[torch.nn.Module], epochs: np.ndarray, *, batch_size: int = 64,
                  on_progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Return ``(n_epochs, 5)`` float32: the mean of each fold's softmax probabilities."""
    return _ensemble(models, epochs, batch_size, on_progress, None)[0]


def predict_and_embed(models: list[torch.nn.Module], epochs: np.ndarray, *, embed_fold: int,
                      batch_size: int = 64,
                      on_progress: Callable[[int, int], None] | None = None,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """``predict_proba`` plus fold ``embed_fold``'s features (``embed``), in the same pass.

    The features are what that fold's classifier head sees; similarity search compares them
    with library vectors from the same fold (``app.library``).
    """
    if not 1 <= embed_fold <= len(models):
        raise ValueError(f"embed_fold must be between 1 and {len(models)}.")
    probabilities, features = _ensemble(models, epochs, batch_size, on_progress, embed_fold)
    return probabilities, features
