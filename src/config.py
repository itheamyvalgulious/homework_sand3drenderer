"""Shared configuration for the SAND reproduction pipeline."""

from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    # io
    model: str = "assets/sample.glb"   # input .glb mesh
    out_dir: str = "output"           # all artifacts go here
    device: str = "cuda"               # "cuda" or "cpu"
    seed: int = 0

    # network (paper: 8 hidden layers x 256 units, SIREN-style)
    hidden: int = 256
    num_layers: int = 8
    w0: float = 30.0                   # SIREN omega_0

    # training (paper: 100k iters, batch 100k, lr 1e-4, 60% surface / 40% gauss sigma=0.01)
    iters: int = 20000                 # reduced for homework-scale GPUs; paper used 100000
    batch: int = 100_000
    lr: float = 1e-4
    sigma: float = 0.01                # gaussian perturbation, normalized units
    surf_frac: float = 0.6             # fraction of on-surface points
    # fraction of uniform-domain points supervising the far field. Applied to
    # the BASELINE pool only; SAND stays paper-faithful (octree filters far
    # points, Sec. 3.3), which is exactly the paper's selling point.
    domain_frac: float = 0.2
    rgb_lambda: float = 1.0            # weight of color loss when color=True
    log_every: int = 1000

    # octree / depth map (paper: max depth 9, r = 1.5e-4)
    octree_depth: int = 9
    err_thresh: float = 1.5e-4         # r in Eq. (5)
    depth_samples_per_leaf: int = 32   # points sampled per near leaf for Eq. (5)

    # inference / mesh extraction (paper: 512^3; reduced default for 4GB GPUs)
    res: int = 256
    chunk: int = 262_144               # query batch size during grid evaluation

    # data pool: training points are pre-sampled once (GT SDF queries are costly,
    # esp. winding-number signs for non-watertight meshes); each iteration draws
    # a random minibatch from the pool. Pool size = pool_batches * batch.
    pool_batches: int = 40

    # solid-union preprocessing: when set (and the input is not watertight),
    # build a watertight proxy mesh on an N^3 grid before training (cleans up
    # winding-number sign flicker in cluttered cavities of multi-part assets).
    solid_union_res: int | None = None

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))
