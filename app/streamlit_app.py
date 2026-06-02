"""Interactive customer-segment explorer.

Run with:  streamlit run app/streamlit_app.py
(Generates segments on first load if reports/ is empty.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segmentation.run import run  # noqa: E402

REPORTS = ROOT / "reports"

st.set_page_config(page_title="Customer Segmentation", page_icon="🛒", layout="wide")
st.title("🛒 E-Commerce Customer Segmentation")

if not (REPORTS / "metrics.json").exists():
    with st.spinner("Generating segments…"):
        run(output=REPORTS)

metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
profiles = pd.read_csv(REPORTS / "segment_profiles.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{metrics['n_customers']:,}")
c2.metric("Segments (best k)", metrics["best_k"])
c3.metric("Silhouette", metrics["silhouette"])
c4.metric("Adjusted Rand Index", metrics["adjusted_rand_index"])

st.subheader("Segment profiles")
st.dataframe(profiles, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Choosing k")
    if (REPORTS / "k_selection.png").exists():
        st.image(str(REPORTS / "k_selection.png"))
with right:
    st.subheader("Segments in PCA space")
    if (REPORTS / "pca_segments.png").exists():
        st.image(str(REPORTS / "pca_segments.png"))

st.caption(
    "Data is synthetic (five known latent segments); ARI measures how well the "
    "unsupervised clustering recovered them."
)
