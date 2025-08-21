import time
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, Any
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config/dynamic_genesis.yaml'

def get_timestamp(utc: bool = True) -> Dict[str, float]:
    """Return both float epoch and formatted string for true deception flexibility."""
    now = time.time()
    fmt = '%Y-%m-%d %H:%M:%S.%f' if utc else '%Y-%m-%d %H:%M:%S.%f %Z'
    ts_str = time.strftime(fmt, time.gmtime() if utc else time.localtime())
    return {"epoch": now, "string": ts_str}

def calculate_volatility(df: pd.DataFrame, column: str = 'close', window: int = 20) -> float:
    """Cheap, safe vol calc with entropy tweak."""
    if df.empty or column not in df:
        return 0.01
    returns = np.log(df[column].pct_change().dropna().clip(lower=1e-10) + 1)
    if returns.empty:
        return 0.01
    # Adaptive window based on entropy
    entropy = -np.sum(np.log(np.abs(returns) + 1e-10)) / len(returns)
    w = max(5, int(window + entropy * len(returns) * 0.05))
    sample = returns.tail(w)
    return float(np.std(sample) * np.sqrt(252 * 24 * 60))

def detect_regime(df: pd.DataFrame, features: list = ['close','volume'], n_clusters: int = 3) -> Dict[str, Any]:
    """Fast regime detection via MiniBatchKMeans with cheap confidence proxy."""
    if df.empty or len(df) < n_clusters * 2:
        return {"regime": -1, "confidence": 0.0, "centers": None}
    data = df[features].fillna(0).values
    model = MiniBatchKMeans(n_clusters=min(n_clusters, len(data)//2), batch_size=32, n_init=5)
    labels = model.fit_predict(data)
    # Cheap confidence proxy: inverse inertia normalized
    inertia = model.inertia_ / max(len(data), 1)
    conf = float(1.0 / (1.0 + inertia))
    return {"regime": int(labels[-1]), "confidence": conf, "centers": model.cluster_centers_.tolist()}
