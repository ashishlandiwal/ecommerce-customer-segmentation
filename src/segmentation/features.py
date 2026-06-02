"""Feature scaling and PCA."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(df.to_numpy(dtype=float)), scaler


def pca_transform(X: np.ndarray, n_components: int = 2, random_state: int = 42):
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)
    return coords, pca
