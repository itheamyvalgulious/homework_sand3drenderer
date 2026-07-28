"""Volumetric network-depth map: octree build, depth assignment, queries.

CONTRACT (paper Sec. 3.2, Eqs. 5-7; unit cube [0,1]^3):

class Octree:
    max_depth: int

    @classmethod
    def build(cls, mesh_data, max_depth: int) -> "Octree"
        # Root cell = [0,1]^3. A node is subdivided while its cell intersects the
        # surface AND its level < max_depth. Intersection test (paper Sec. 4.1):
        # distance(node center, mesh) <= 0.5 * cell diagonal.
        # Near-surface leaves: cells that still intersect the surface at max_depth.
        # Far leaves: everything else; each stores the GT signed distance at its
        # center (paper Eq. 7, depth-0 approximation).

    def is_far(self, pts: np.ndarray) -> np.ndarray
        # (N,3) -> (N,) bool; True when the point falls in a far (depth-0) leaf.

    def assign_depths(self, model, device: str, err_thresh: float,
                      samples_per_leaf: int, chunk: int) -> None
        # For every near leaf: sample `samples_per_leaf` points inside the cell
        # (+ cell center), evaluate model.forward_all in chunks, and set
        #   d(x) = min { i | |y_L(x) - y_i(x)| < err_thresh AND sign(y_i)==sign(y_L) }
        # on the SDF channel (channel 0); fall back to num_layers when no i qualifies.
        # Leaf depth = max over its sample points (Eq. 6). Far leaves keep depth 0.
        # `model` is a TMLP (needs forward_all); out_dim may be 1 or 4 — only SDF
        # channel is used for the criterion.

    def query(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]
        # (N,3) -> (net_depth (N,) uint8 in [0..num_layers], stored_sdf (N,) float32).
        # net_depth == 0 means: do not run the network, use stored_sdf (far region).
        # stored_sdf is only meaningful where net_depth == 0.

    def stats(self) -> dict
        # {"n_leaves", "n_near", "n_far", "depth_hist": {d: count}, "mean_depth_near"}

    def save(self, path: str) -> None          # npz
    @classmethod
    def load(cls, path: str) -> "Octree"
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch


class Octree:
    """Octree partition of [0,1]^3 with per-leaf network depth / far-region SDF.

    Leaves are stored in flat arrays. Queries descend the tree with per-level
    sorted key arrays; the key of cell (ix, iy, iz) at level l is the flat index
    ix + (iy << l) + (iz << 2l), so lookup is a vectorized searchsorted.
    """

    def __init__(self) -> None:
        self.max_depth: int = 0
        self.leaf_level = np.zeros(0, dtype=np.uint8)      # level of each leaf
        self.leaf_idx = np.zeros((0, 3), dtype=np.int32)   # integer cell coords
        self.leaf_near = np.zeros(0, dtype=bool)           # True: near-surface leaf
        self.leaf_depth = np.zeros(0, dtype=np.uint8)      # network depth (0 = far)
        self.leaf_sdf = np.zeros(0, dtype=np.float32)      # signed dist at center
        self._level_keys: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, mesh_data, max_depth: int) -> "Octree":
        self = cls()
        self.max_depth = int(max_depth)
        # The intersection test only needs |distance|; when the mesh provides a
        # cheap unsigned-distance oracle (non-watertight meshes whose signed
        # query also evaluates a winding number), use it and fix the signs of
        # far-leaf stored values in one final batched call below.
        dist_fn = getattr(mesh_data, "unsigned_sdf_at", None) or mesh_data.sdf_at
        leaves = []  # (level, ix, iy, iz, dist_at_center, near)
        nodes = [(0, 0, 0, 0)]
        for level in range(self.max_depth + 1):
            if not nodes:
                break
            size = 1.0 / (1 << level)
            half_diag = 0.5 * math.sqrt(3.0) * size
            centers = np.array(
                [((ix + 0.5) * size, (iy + 0.5) * size, (iz + 0.5) * size)
                 for (_, ix, iy, iz) in nodes], dtype=np.float64)
            sdf = np.asarray(dist_fn(centers), dtype=np.float64).reshape(-1)
            children = []
            for (lv, ix, iy, iz), s in zip(nodes, sdf):
                if abs(s) <= half_diag:  # cell intersects the surface
                    if level < self.max_depth:
                        for dx in (0, 1):
                            for dy in (0, 1):
                                for dz in (0, 1):
                                    children.append((level + 1, 2 * ix + dx,
                                                     2 * iy + dy, 2 * iz + dz))
                    else:
                        leaves.append((lv, ix, iy, iz, s, True))
                else:
                    leaves.append((lv, ix, iy, iz, s, False))
            nodes = children
        self.leaf_level = np.array([l[0] for l in leaves], dtype=np.uint8)
        self.leaf_idx = np.array([[l[1], l[2], l[3]] for l in leaves], dtype=np.int32)
        self.leaf_sdf = np.array([l[4] for l in leaves], dtype=np.float32)
        self.leaf_near = np.array([l[5] for l in leaves], dtype=bool)
        self.leaf_depth = np.zeros(len(leaves), dtype=np.uint8)
        # Far leaves must store the SIGNED distance at their center (paper Eq. 7).
        # If the build loop above used unsigned distances, recompute signed SDF
        # for far-leaf centers only (a single batched call).
        if dist_fn is not mesh_data.sdf_at:
            far_ids = np.nonzero(~self.leaf_near)[0]
            if far_ids.size:
                lv = self.leaf_level[far_ids].astype(np.int64)
                ijk = self.leaf_idx[far_ids].astype(np.float64)
                sizes = 1.0 / (1 << lv)
                centers = (ijk + 0.5) * sizes[:, None]
                self.leaf_sdf[far_ids] = mesh_data.sdf_at(centers).astype(np.float32)
        self._rebuild_lookup()
        return self

    # ----------------------------------------------------------------- lookup
    def _rebuild_lookup(self) -> None:
        """Per-level sorted (flat_key, leaf_id) arrays for vectorized descent."""
        self._level_keys = {}
        for level in np.unique(self.leaf_level):
            l = int(level)
            ids = np.nonzero(self.leaf_level == level)[0]
            ijk = self.leaf_idx[ids].astype(np.int64)
            flat = ijk[:, 0] + (ijk[:, 1] << l) + (ijk[:, 2] << (2 * l))
            order = np.argsort(flat, kind="stable")
            self._level_keys[l] = (flat[order], ids[order])

    def _locate(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) -> (N,) int64 leaf ids. Every point resolves to exactly one leaf."""
        pts = np.clip(np.asarray(pts, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
        n = pts.shape[0]
        leaf_ids = np.full(n, -1, dtype=np.int64)
        pending = np.arange(n)
        for level in range(self.max_depth + 1):
            if pending.size == 0:
                break
            entry = self._level_keys.get(level)
            if entry is None:
                continue
            keys, ids = entry
            scale = 1 << level
            ijk = np.floor(pts[pending] * scale).astype(np.int64)
            np.minimum(ijk, scale - 1, out=ijk)
            flat = ijk[:, 0] + (ijk[:, 1] << level) + (ijk[:, 2] << (2 * level))
            pos = np.searchsorted(keys, flat)
            pos = np.minimum(pos, keys.size - 1)
            hit = keys[pos] == flat
            leaf_ids[pending[hit]] = ids[pos[hit]]
            pending = pending[~hit]
        if np.any(leaf_ids < 0):
            raise RuntimeError("octree does not fully cover [0,1]^3")
        return leaf_ids

    # ---------------------------------------------------------------- queries
    def is_far(self, pts: np.ndarray) -> np.ndarray:
        return ~self.leaf_near[self._locate(pts)]

    def query(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = self._locate(pts)
        return self.leaf_depth[ids], self.leaf_sdf[ids]

    # ---------------------------------------------------------- depth assign
    def assign_depths(self, model, device: str, err_thresh: float,
                      samples_per_leaf: int, chunk: int,
                      rgb_thresh: float | None = None) -> None:
        # rgb_thresh: optional extension for color models. The paper's Eq. (5)
        # gates on the SDF channel only; with a color field, shallow depths that
        # suffice for geometry can still have unconverged RGB. When rgb_thresh
        # is set (and the model outputs 4 channels), a depth only qualifies if
        # max|Δrgb| < rgb_thresh holds as well, pushing textured regions deeper.
        near_ids = np.nonzero(self.leaf_near)[0]
        if near_ids.size == 0:
            return
        rng = np.random.default_rng(0)
        levels = self.leaf_level[near_ids].astype(np.int64)
        sizes = 1.0 / (1 << levels)                                    # (K,)
        origins = self.leaf_idx[near_ids].astype(np.float64) * sizes[:, None]
        centers = origins + 0.5 * sizes[:, None]                       # (K,3)
        k = near_ids.size
        u = rng.random((k, samples_per_leaf, 3))
        pts = origins[:, None, :] + u * sizes[:, None, None]           # (K,S,3)
        pts = np.concatenate([pts, centers[:, None, :]], axis=1)       # + center
        flat = pts.reshape(-1, 3).astype(np.float32)
        n = flat.shape[0]
        chunk = max(1, int(chunk))
        # Compute the qualifying depth per chunk and keep only uint8 depths;
        # accumulating all (N, L[, 4]) outputs would need gigabytes of RAM.
        d_all = np.empty(n, dtype=np.uint8)
        num_layers = 0
        with torch.no_grad():
            for start in range(0, n, chunk):
                x = torch.from_numpy(flat[start:start + chunk]).to(device)
                ys = model.forward_all(x)
                num_layers = len(ys)
                use_rgb = rgb_thresh is not None and ys[0].shape[1] == 4
                y = (torch.stack(list(ys), dim=1) if use_rgb
                     else torch.stack([y_[:, 0] for y_ in ys], dim=1))
                y = y.float().cpu().numpy()                        # (B,L) or (B,L,4)
                if y.ndim == 3:
                    sdf, rgb = y[..., 0], y[..., 1:4]
                    sdf_last = sdf[:, -1:]
                    ok = ((np.abs(sdf - sdf_last) < err_thresh)
                          & (np.sign(sdf) == np.sign(sdf_last))
                          & (np.abs(rgb - rgb[:, -1:, :]).max(axis=2) < rgb_thresh))
                else:
                    y_last = y[:, -1:]
                    ok = ((np.abs(y - y_last) < err_thresh)
                          & (np.sign(y) == np.sign(y_last)))
                found = ok.any(axis=1)
                # 1-based index of first qualifying output; fallback num_layers
                d_all[start:start + y.shape[0]] = np.where(
                    found, np.argmax(ok, axis=1) + 1, num_layers).astype(np.uint8)
        d = d_all.reshape(k, samples_per_leaf + 1)
        self.leaf_depth[near_ids] = d.max(axis=1)                      # Eq. 6 max-pool

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        depths, counts = np.unique(self.leaf_depth.astype(np.int64),
                                   return_counts=True)
        hist = {int(d): int(c) for d, c in zip(depths, counts)}
        near = self.leaf_near
        mean_depth = float(self.leaf_depth[near].mean()) if near.any() else 0.0
        return {
            "n_leaves": int(self.leaf_level.size),
            "n_near": int(near.sum()),
            "n_far": int((~near).sum()),
            "depth_hist": hist,
            "mean_depth_near": mean_depth,
        }

    # ------------------------------------------------------------- save/load
    def save(self, path: str) -> None:
        np.savez(path,
                 max_depth=np.int64(self.max_depth),
                 leaf_level=self.leaf_level,
                 leaf_idx=self.leaf_idx,
                 leaf_near=self.leaf_near,
                 leaf_depth=self.leaf_depth,
                 leaf_sdf=self.leaf_sdf)

    @classmethod
    def load(cls, path: str) -> "Octree":
        p = path if os.path.exists(path) else path + ".npz"
        self = cls()
        with np.load(p) as data:
            self.max_depth = int(data["max_depth"])
            self.leaf_level = np.array(data["leaf_level"])
            self.leaf_idx = np.array(data["leaf_idx"])
            self.leaf_near = np.array(data["leaf_near"])
            self.leaf_depth = np.array(data["leaf_depth"])
            self.leaf_sdf = np.array(data["leaf_sdf"])
        self._rebuild_lookup()
        return self
