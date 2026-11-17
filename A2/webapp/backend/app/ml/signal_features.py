"""Model-free spectral features: a similarity space that no classifier has shaped.

Learned embeddings inherit the decision boundaries of the model that produced them, so
neighbours found with them tend to agree with that model. Band power is computed from the
filtered EEG alone and serves as the neutral reference when comparing retrieval spaces.
"""

import numpy as np
from scipy.signal import welch

from .preprocessing import CHANNELS, TARGET_SFREQ

SIGNAL_FEATURES_VERSION = "log-band-power-v1"
BANDS = (
    ("slow", 0.5, 2.0),
    ("delta", 2.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 15.0),
    ("beta", 15.0, 30.0),
)
FEATURE_NAMES = tuple(f"{channel}:{band}" for channel in CHANNELS for band, _, _ in BANDS)
WINDOW_SECONDS = 2  # 0.5 Hz resolution: the narrowest band still spans several bins
CHUNK = 512


def signal_features(epochs: np.ndarray) -> np.ndarray:
    """``(n, 2, 30, 200)`` epochs in microvolts / 100 -> ``(n, 12)`` log10 band power (uV^2).

    Order follows ``FEATURE_NAMES``: every band of Fpz-Cz, then every band of Pz-Oz.
    """
    if epochs.ndim != 4 or epochs.shape[1:] != (len(CHANNELS), 30, TARGET_SFREQ):
        raise ValueError("Expected epochs shaped (n, 2, 30, 200).")
    nperseg = WINDOW_SECONDS * TARGET_SFREQ
    rows = []
    for start in range(0, len(epochs), CHUNK):
        chunk = epochs[start:start + CHUNK].reshape(-1, len(CHANNELS), 30 * TARGET_SFREQ)
        freqs, psd = welch(chunk.astype(np.float64) * 100.0, fs=TARGET_SFREQ,
                           nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
        step = freqs[1] - freqs[0]
        bands = [psd[..., (freqs >= low) & (freqs < high)].sum(axis=-1) * step
                 for _, low, high in BANDS]
        rows.append(np.log10(np.stack(bands, axis=-1) + 1e-6).reshape(len(chunk), -1))
    return np.concatenate(rows).astype(np.float32)
