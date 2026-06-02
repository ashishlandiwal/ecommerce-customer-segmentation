"""End-to-end pipeline: generate -> scale -> select k -> cluster -> profile -> save."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import evaluate
from .cluster import choose_k, dbscan_fit, kmeans_fit, suggest_eps
from .data import SEGMENT_NAMES, generate_customers
from .features import pca_transform, scale_features
from .profile import profile_segments


def run(
    n: int = 4000,
    random_state: int = 42,
    output: str | Path = "reports",
    eps: float | None = None,
    min_samples: int = 15,
):
    df, true_segments = generate_customers(n, random_state)
    X, _ = scale_features(df)

    ksel = choose_k(X, range(2, 9), random_state)
    _, labels = kmeans_fit(X, ksel.best_k, random_state)
    silhouette = float(silhouette_score(X, labels))
    ari = float(adjusted_rand_score(true_segments, labels))

    # Pick DBSCAN eps from the k-distance knee unless the caller overrides it.
    if eps is None:
        eps = suggest_eps(X, min_samples=min_samples)
    _, _, db_clusters, db_noise = dbscan_fit(X, eps=eps, min_samples=min_samples)
    coords, pca = pca_transform(X, 2, random_state)
    profiles = profile_segments(df, labels)

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_customers": int(n),
        "true_segments": len(SEGMENT_NAMES),
        "best_k": int(ksel.best_k),
        "silhouette": round(silhouette, 4),
        "adjusted_rand_index": round(ari, 4),
        "pca_explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "pca_cumulative_2components": round(float(pca.explained_variance_ratio_[:2].sum()), 4),
        "k_selection": {
            "k": ksel.k_values,
            "silhouette": [round(s, 4) for s in ksel.silhouettes],
        },
        "dbscan": {
            "eps": round(float(eps), 3),
            "min_samples": min_samples,
            "n_clusters": db_clusters,
            "n_noise": db_noise,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    profiles.to_csv(out / "segment_profiles.csv", index=False)
    evaluate.plot_k_selection(ksel, out / "k_selection.png")
    evaluate.plot_pca_scatter(coords, labels, pca, out / "pca_segments.png")
    return metrics, profiles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--customers", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="reports")
    ap.add_argument("--eps", type=float, default=None, help="DBSCAN eps (default: auto from k-distance)")
    ap.add_argument("--min-samples", type=int, default=15)
    args = ap.parse_args()

    metrics, profiles = run(args.customers, args.seed, args.output, args.eps, args.min_samples)
    print(json.dumps(metrics, indent=2))
    print("\nSegment profiles:")
    print(profiles.to_string(index=False))
    print(f"\nArtifacts -> {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
