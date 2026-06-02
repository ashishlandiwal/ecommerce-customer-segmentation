"""Customer segmentation: synthetic RFM data, KMeans/DBSCAN, PCA, profiling.

Public API:
    generate_customers(...) -> (DataFrame features, ndarray latent_segment)
    choose_k(...)           -> KSelection (elbow inertia + silhouette per k)
    profile_segments(...)   -> DataFrame of per-cluster feature means + names
"""
from .cluster import choose_k
from .data import FEATURES, SEGMENT_NAMES, generate_customers
from .profile import profile_segments

__all__ = ["generate_customers", "choose_k", "profile_segments", "FEATURES", "SEGMENT_NAMES"]
__version__ = "0.1.0"
