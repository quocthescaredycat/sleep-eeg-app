"""Per-epoch feature vectors for similarity search, from the same pass as the predictions.

Every served network exposes ``embed`` (the features its last layer sees) and ``classify``
(that last layer), with ``forward == classify(embed(x))``. Each fold of a release is trained
independently, so each fold has its own feature space: vectors from different folds must
never be compared, averaged or mixed in one index.
"""

import numpy as np
import torch

EMBEDDING_DIMS = {"Transformer": 128, "AttnSleep": 2400}


@torch.inference_mode()
def embed_and_classify(model: torch.nn.Module, epochs: np.ndarray, *,
                       batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features [n, d], softmax probabilities [n, 5])``, both float32."""
    features, probabilities = [], []
    for start in range(0, len(epochs), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(epochs[start:start + batch_size]))
        embedded = model.embed(batch)
        features.append(embedded.float().numpy())
        probabilities.append(torch.softmax(model.classify(embedded).double(), dim=1).numpy())
    return (np.concatenate(features).astype(np.float32),
            np.concatenate(probabilities).astype(np.float32))
