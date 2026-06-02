import numpy as np

from segmentation import FEATURES, SEGMENT_NAMES, choose_k, generate_customers
from segmentation.features import scale_features
from segmentation.run import run


def test_generate_customers_shape():
    df, labels = generate_customers(n=500, random_state=1)
    assert list(df.columns) == FEATURES
    assert len(df) == 500 == len(labels)
    assert set(np.unique(labels)).issubset(set(range(len(SEGMENT_NAMES))))
    assert (df["recency_days"] > 0).all()
    assert (df["monetary"] > 0).all()


def test_choose_k_returns_k_in_range():
    df, _ = generate_customers(n=600, random_state=2)
    X, _ = scale_features(df)
    ksel = choose_k(X, range(2, 7), random_state=2)
    assert ksel.best_k in range(2, 7)
    assert len(ksel.silhouettes) == len(ksel.k_values)


def test_pipeline_recovers_structure(tmp_path):
    metrics, profiles = run(n=2000, random_state=42, output=tmp_path)
    # silhouette indicates real cluster structure
    assert metrics["silhouette"] > 0.3
    # clustering recovers the latent segments reasonably well (chance ARI ~ 0)
    assert metrics["adjusted_rand_index"] > 0.5
    # named, non-empty profiles for every discovered cluster
    assert len(profiles) == metrics["best_k"]
    assert profiles["segment_name"].notna().all()
