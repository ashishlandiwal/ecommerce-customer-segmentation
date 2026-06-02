"""KMeans (with k-selection) and DBSCAN."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


@dataclass
class KSelection:
    k_values: list[int]
    inertias: list[float]
    silhouettes: list[float]
    best_k: int


def choose_k(X: np.ndarray, k_range=None, random_state: int = 42) -> KSelection:
    """Sweep k, recording inertia (for the elbow) and silhouette; pick best silhouette."""
    if k_range is None:
        k_range = range(2, 9)
    ks, inertias, sils = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        ks.append(int(k))
        inertias.append(float(km.inertia_))
        sils.append(float(silhouette_score(X, labels)))
    best_k = ks[int(np.argmax(sils))]
    return KSelection(ks, inertias, sils, best_k)


def kmeans_fit(X: np.ndarray, k: int, random_state: int = 42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km, km.fit_predict(X)


def suggest_eps(X: np.ndarray, min_samples: int = 15, percentile: float = 90.0) -> float:
    """Heuristic DBSCAN ``eps`` from the k-distance graph.

    Standard practice: sort each point's distance to its ``min_samples``-th nearest
    neighbour and read the "knee". We approximate the knee with a high percentile,
    which is robust and dependency-free.
    """
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    distances, _ = nn.kneighbors(X)
    kth = np.sort(distances[:, -1])
    return float(np.percentile(kth, percentile))


def dbscan_fit(X: np.ndarray, eps: float = 1.5, min_samples: int = 15):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    return db, labels, n_clusters, n_noise
