# utils/learner.py
# Lean rolling learner for research/simulation.
# Computes feature→outcome correlations over a sliding window,
# produces bounded deltas, and returns structured "lessons".
# No file I/O. No global mutation. No stealth games.

from collections import deque
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# ---- Rolling buffer to accumulate evidence ----
class RollingBuffer:
    def __init__(self, maxlen: int = 256, feature_keys: List[str] = None):
        self.maxlen = maxlen
        self.feature_keys = feature_keys or [
            "entry_price", "exit_price", "position_size", "duration_min",
            "leverage", "confidence", "chf_spike", "vol", "regime",
            "volume", "source"
        ]
        self.X = deque(maxlen=maxlen)   # list of np.array features
        self.y = deque(maxlen=maxlen)   # list of floats (outcomes)

    def push(self, features: np.ndarray, outcome: float) -> None:
        self.X.append(features.astype(float))
        self.y.append(float(outcome))

    def matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.X:
            return np.empty((0, len(self.feature_keys))), np.empty((0,))
        return np.vstack(self.X), np.asarray(self.y)


# ---- Core learner (no state outside args) ----
def extract_features(trade: Dict[str, Any], vol: float, regime: float) -> np.ndarray:
    # Keep the ordering in sync with RollingBuffer.feature_keys
    return np.array([
        trade.get("entry_price", 0.0),
        trade.get("exit_price", 0.0),
        trade.get("position_size", 1.0),
        trade.get("duration_min", 0.0),
        trade.get("leverage", 1.0),
        trade.get("confidence", 0.5),
        trade.get("chf_spike", 0.0),
        float(vol),
        float(regime),
        trade.get("volume", 0.0),
        trade.get("source", 0.0),
    ], dtype=float)

def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    # NaN-safe Pearson over a window
    if x.size < 2 or y.size < 2:
        return 0.0
    xv = x - x.mean()
    yv = y - y.mean()
    denom = (np.sqrt((xv**2).sum()) * np.sqrt((yv**2).sum()))
    if denom == 0:
        return 0.0
    return float((xv * yv).sum() / denom)

def compute_deltas(buffer: RollingBuffer, k: int = 6, scale: float = 0.25) -> np.ndarray:
    """
    Map top-k absolute correlations to k deltas.
    Bounded to [-1, 1] then scaled.
    """
    X, y = buffer.matrix()
    if X.shape[0] < 8:  # need some data before making claims
        return np.zeros(k, dtype=float)

    cors = np.array([safe_pearson(X[:, i], y) for i in range(X.shape[1])])
    # pick strongest absolute correlations
    idx = np.argsort(np.abs(cors))[::-1][:k]
    strongest = np.clip(cors[idx], -1.0, 1.0)
    deltas = strongest * scale  # small, bounded adjustment suggestions
    return deltas

def analyze_trade(
    trade_data: Dict[str, Any],
    outcome: float,
    vol: float,
    regime: float,
    buffer: RollingBuffer,
    timestamp: str,
    mode: str,
    feed_label: str
) -> Dict[str, Any]:
    """
    - Push new sample
    - Compute bounded deltas from rolling correlation
    - Return structured lessons for logging
    """
    feats = extract_features(trade_data, vol, regime)
    buffer.push(feats, outcome)
    deltas = compute_deltas(buffer, k=6, scale=0.25)

    lessons = {
        "what": f"Decision: {trade_data.get('side','unknown')} @ {trade_data.get('entry_price')} size {trade_data.get('position_size')}",
        "why": (
            f"Signals: vol={vol:.5f}, regime={regime}, "
            f"feature-strength≈{float(np.abs(deltas).mean()):.3f}"
        ),
        "when": f"{timestamp}",
        "how": "rolling-corr over last {} trades; bounded deltas".format(len(buffer.y)),
        "where": f"mode={mode}; feed={feed_label}",
        "learned": {
            "strengths": "features with consistent sign over window",
            "weaknesses": "features with unstable or near-zero correlation",
            "improvements": {
                # These deltas are *suggestions* to upstream components.
                # Do not mutate config files here; return values instead.
                "delta_1": float(deltas[0]) if deltas.size > 0 else 0.0,
                "delta_2": float(deltas[1]) if deltas.size > 1 else 0.0,
                "delta_3": float(deltas[2]) if deltas.size > 2 else 0.0,
                "delta_4": float(deltas[3]) if deltas.size > 3 else 0.0,
                "delta_5": float(deltas[4]) if deltas.size > 4 else 0.0,
                "delta_6": float(deltas[5]) if deltas.size > 5 else 0.0,
            }
        }
    }
    return lessons
