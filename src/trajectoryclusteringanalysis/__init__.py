"""
Trajectory Clustering Analysis (TCA) Package
"""
from .__version__ import __version__
from .tca import TCA

# Optional: make other important classes or functions available at the top level
# For example, if users frequently use clustering functions directly:
# from .unidimensional.clustering import compute_distance_matrix, hierarchical_clustering, k_medoids_clustering_faster
# from .plotting import plot_dendrogram, plot_clustermap

__all__ = ['TCA', '__version__']