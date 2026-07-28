"""SAND: Spatially Adaptive Network Depth — reproduction pipeline.

Reproduction of arXiv:2604.25936 (SAND) with an optional color implicit field.
Pipeline: glb model -> train T-MLP (SAND) / plain MLP (baseline) -> octree
network-depth map -> adaptive inference -> Marching Cubes -> renders.
"""

__all__ = ["config", "data", "tmlp", "train", "octree", "infer", "render"]
