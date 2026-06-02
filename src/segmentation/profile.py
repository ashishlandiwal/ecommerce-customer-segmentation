"""Turn cluster labels into interpretable, named segment profiles."""
from __future__ import annotations

import pandas as pd

from .data import FEATURES


def _label_clusters(prof: pd.DataFrame) -> list[str]:
    """Heuristic, human-readable names from each cluster's mean RFM profile.

    Names are derived relative to the other clusters (quantiles), so they adapt to the
    data rather than relying on hard-coded thresholds.
    """
    rec, freq, mon, ten = (
        prof["recency_days"],
        prof["frequency"],
        prof["monetary"],
        prof["tenure_days"],
    )
    names = []
    for i in prof.index:
        if rec[i] >= rec.quantile(0.8):
            names.append("at_risk")
        elif ten[i] <= ten.quantile(0.25):
            names.append("new_customers")
        elif (mon[i] >= mon.median()) and (freq[i] >= freq.median()):
            names.append("champions")
        elif freq[i] >= freq.median():
            names.append("loyal")
        else:
            names.append("bargain_hunters")
    return names


def profile_segments(df: pd.DataFrame, labels) -> pd.DataFrame:
    tmp = df.copy()
    tmp["cluster"] = labels
    prof = tmp.groupby("cluster")[FEATURES].mean().round(1)
    prof["size"] = tmp.groupby("cluster").size()
    prof["segment_name"] = _label_clusters(prof)
    return prof.reset_index()
