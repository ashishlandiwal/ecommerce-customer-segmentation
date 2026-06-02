"""Plots for k-selection and the PCA cluster map."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cluster import KSelection


def plot_k_selection(ksel: KSelection, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(ksel.k_values, ksel.inertias, "o-")
    ax1.set_title("Elbow (inertia)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia")
    ax2.plot(ksel.k_values, ksel.silhouettes, "o-", color="#c0392b")
    ax2.axvline(ksel.best_k, ls="--", color="gray")
    ax2.set_title(f"Silhouette (best k = {ksel.best_k})")
    ax2.set_xlabel("k")
    ax2.set_ylabel("silhouette")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(coords: np.ndarray, labels, pca, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=8, alpha=0.6)
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0] * 100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.0f}% var)")
    ax.set_title("Customer segments (PCA projection)")
    legend = ax.legend(*scatter.legend_elements(), title="cluster", loc="best", fontsize=8)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
