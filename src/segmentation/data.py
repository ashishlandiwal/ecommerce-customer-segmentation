"""Generate a realistic *synthetic* e-commerce customer base.

Real transactional datasets (e.g. UCI Online Retail) require a download and licensing
caveats. Instead we synthesize customers from five interpretable latent segments with
segment-specific RFM distributions plus noise. Because the latent segment is known, we
can quantitatively validate the unsupervised clustering with Adjusted Rand Index — a
luxury you don't get on real unlabeled data, and an honest way to prove the pipeline works.

To run on real data instead, load a CSV with the same feature columns (see ``FEATURES``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Recency (days since last order), Frequency (orders/year), Monetary (annual spend),
# average order value, tenure (days as a customer).
FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value", "tenure_days"]

SEGMENT_NAMES = ["champions", "loyal", "new_customers", "at_risk", "bargain_hunters"]

# (recency, frequency, monetary, avg_order_value, tenure) means per latent segment.
_SEGMENT_PROFILES = {
    "champions":       dict(recency=12,  freq=22, monetary=4200, aov=190, tenure=900, w=0.18),
    "loyal":           dict(recency=35,  freq=12, monetary=1900, aov=160, tenure=720, w=0.24),
    "new_customers":   dict(recency=20,  freq=3,  monetary=300,  aov=110, tenure=70,  w=0.20),
    "at_risk":         dict(recency=160, freq=9,  monetary=1500, aov=170, tenure=800, w=0.18),
    "bargain_hunters": dict(recency=70,  freq=5,  monetary=350,  aov=70,  tenure=400, w=0.20),
}


def generate_customers(n: int = 4000, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (features DataFrame, latent segment index per customer)."""
    rng = np.random.RandomState(random_state)
    weights = np.array([_SEGMENT_PROFILES[s]["w"] for s in SEGMENT_NAMES])
    weights = weights / weights.sum()
    counts = rng.multinomial(n, weights)

    rows, labels = [], []
    for seg_idx, (seg, count) in enumerate(zip(SEGMENT_NAMES, counts, strict=True)):
        p = _SEGMENT_PROFILES[seg]
        recency = np.clip(rng.normal(p["recency"], p["recency"] * 0.35, count), 1, 400)
        frequency = np.clip(rng.normal(p["freq"], max(1.0, p["freq"] * 0.30), count), 1, None)
        # monetary is lognormal-ish around the segment mean (spend is right-skewed)
        monetary = np.clip(rng.normal(p["monetary"], p["monetary"] * 0.30, count), 50, None)
        aov = np.clip(rng.normal(p["aov"], p["aov"] * 0.25, count), 20, None)
        tenure = np.clip(rng.normal(p["tenure"], p["tenure"] * 0.30, count), 7, 1500)
        for r, f, m, a, t in zip(recency, frequency, monetary, aov, tenure, strict=True):
            rows.append((r, f, m, a, t))
            labels.append(seg_idx)

    df = pd.DataFrame(rows, columns=FEATURES)
    labels = np.array(labels)
    # shuffle so rows aren't ordered by segment
    order = rng.permutation(len(df))
    return df.iloc[order].reset_index(drop=True), labels[order]
