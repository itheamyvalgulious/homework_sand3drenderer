"""Tests for sand.octree.Octree using an analytic sphere SDF and fake T-MLPs.

Run: .venv/bin/python tests/test_octree.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.octree import Octree

CENTER = np.array([0.5, 0.5, 0.5], dtype=np.float64)
RADIUS = 0.3
MAX_DEPTH = 4
ERR = 1.5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def analytic_sdf(pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return np.linalg.norm(pts - CENTER, axis=1) - RADIUS


def half_diag(level):
    return 0.5 * np.sqrt(3.0) / (1 << level)


class SphereMeshData:
    """Minimal MeshData stand-in (avoids importing sand.data)."""
    def __init__(self):
        self.mesh = trimesh.creation.icosphere(subdivisions=3, radius=RADIUS)
        self.mesh.apply_translation(CENTER)

    def sdf_at(self, pts):
        return analytic_sdf(pts).astype(np.float32)


class FakeTMLP(torch.nn.Module):
    """3-layer fake T-MLP. y1 is mode-dependent, y2 = y3 = exact sdf."""
    def __init__(self, mode="exact", delta=0.0, out_dim=1):
        super().__init__()
        self.mode = mode
        self.delta = delta
        self.out_dim = out_dim

    def _emit(self, sdf):
        if self.out_dim == 1:
            return sdf.unsqueeze(1)
        rgb = torch.stack([torch.full_like(sdf, 0.5), sdf.abs(),
                           torch.zeros_like(sdf)], dim=1)
        return torch.cat([sdf.unsqueeze(1), rgb], dim=1)

    def forward_all(self, x):
        c = torch.as_tensor(CENTER, dtype=x.dtype, device=x.device)
        sdf = torch.linalg.norm(x - c, dim=1) - RADIUS
        if self.mode == "exact":
            y1 = sdf
        elif self.mode == "abs":
            y1 = sdf.abs()                       # right magnitude, wrong sign inside
        elif self.mode == "sign_shift":
            y1 = sdf + self.delta * torch.sign(sdf)  # |y_L - y1| == |delta| everywhere
        elif self.mode == "late":
            y1 = sdf + 5 * ERR * torch.sign(sdf)
        else:
            raise ValueError(self.mode)
        if self.mode == "late":
            return [self._emit(y1), self._emit(y1), self._emit(sdf)]
        return [self._emit(y1), self._emit(sdf), self._emit(sdf)]


def build_tree():
    return Octree.build(SphereMeshData(), MAX_DEPTH)


def leaf_centers(tree, mask=None):
    if mask is None:
        mask = np.ones(tree.leaf_level.size, dtype=bool)
    levels = tree.leaf_level[mask].astype(np.int64)
    sizes = 1.0 / (1 << levels)
    return (tree.leaf_idx[mask].astype(np.float64) + 0.5) * sizes[:, None], levels


def test_build_covers_surface():
    tree = build_tree()
    st = tree.stats()
    assert st["n_near"] > 0 and st["n_far"] > 0
    assert st["n_leaves"] == st["n_near"] + st["n_far"]
    assert st["n_leaves"] % 7 == 1  # each subdivision replaces 1 leaf by 8
    assert set(np.unique(tree.leaf_level)).issubset(set(range(MAX_DEPTH + 1)))

    rng = np.random.default_rng(0)
    d = rng.normal(size=(3000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    surf = CENTER + RADIUS * d
    # every surface point must land in a near (surface-intersecting) leaf
    assert not tree.is_far(surf).any()
    # and within half a cell diagonal of some near-leaf center
    centers, _ = leaf_centers(tree, tree.leaf_near)
    dists = np.linalg.norm(surf[:, None, :] - centers[None, :, :], axis=2)
    assert (dists.min(axis=1) <= half_diag(MAX_DEPTH) + 1e-9).all()
    print("PASS test_build_covers_surface  (leaves=%d, near=%d, far=%d)"
          % (st["n_leaves"], st["n_near"], st["n_far"]))


def test_far_stored_sdf():
    tree = build_tree()
    centers, levels = leaf_centers(tree, ~tree.leaf_near)
    stored = tree.leaf_sdf[~tree.leaf_near].astype(np.float64)
    exact = analytic_sdf(centers)
    assert np.all(np.abs(stored - exact) < 1e-5)          # same oracle, ~exact
    assert np.all(np.abs(stored) > half_diag(levels) - 1e-6)  # far leaves are far
    print("PASS test_far_stored_sdf  (max err %.2e)" % np.abs(stored - exact).max())


def test_assign_depths_sign_and_thresh():
    tree = build_tree()
    near = tree.leaf_near
    run = lambda m: tree.assign_depths(m.to(DEVICE), DEVICE, ERR,
                                       samples_per_leaf=8, chunk=97)
    # exact first output -> every near leaf qualifies at i = 1
    run(FakeTMLP("exact"))
    assert (tree.leaf_depth[near] == 1).all()
    assert (tree.leaf_depth[~near] == 0).all()
    assert abs(tree.stats()["mean_depth_near"] - 1.0) < 1e-9

    # y1 = |sdf|: sign criterion blocks i=1 for inside samples -> depth 2 there
    run(FakeTMLP("abs"))
    centers, _ = leaf_centers(tree, near)
    neg = analytic_sdf(centers) < 0
    assert neg.any()
    assert (tree.leaf_depth[near][neg] == 2).all()       # center sample is inside
    assert set(np.unique(tree.leaf_depth[near])).issubset({1, 2})

    # sign-preserving shift isolates the err_thresh criterion
    run(FakeTMLP("sign_shift", delta=0.5 * ERR))
    assert (tree.leaf_depth[near] == 1).all()            # 0.5r < r qualifies
    run(FakeTMLP("sign_shift", delta=5 * ERR))
    assert (tree.leaf_depth[near] == 2).all()            # 5r >= r does not

    # no early output qualifies -> fall back to the last layer (num_layers = 3)
    run(FakeTMLP("late"))
    assert (tree.leaf_depth[near] == 3).all()

    # out_dim = 4 must give identical depths (only channel 0 is used)
    run(FakeTMLP("abs", out_dim=4))
    d4 = tree.leaf_depth.copy()
    run(FakeTMLP("abs", out_dim=1))
    assert np.array_equal(d4, tree.leaf_depth)
    print("PASS test_assign_depths_sign_and_thresh")


def test_query_roundtrip():
    tree = build_tree()
    tree.assign_depths(FakeTMLP("exact").to(DEVICE), DEVICE, ERR, 8, 100000)
    # leaf centers map back to their own records
    centers, _ = leaf_centers(tree)
    d, s = tree.query(centers)
    assert d.dtype == np.uint8 and s.dtype == np.float32
    assert np.array_equal(d, tree.leaf_depth)
    assert np.array_equal(s, tree.leaf_sdf)
    # boundary points of the unit cube resolve without error
    d, s = tree.query(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert d.shape == (2,) and s.shape == (2,)
    # random points: depth/is_far consistency, far SDF within half diagonal
    rng = np.random.default_rng(1)
    pts = rng.random((5000, 3))
    d, s = tree.query(pts)
    far = d == 0
    assert np.array_equal(far, tree.is_far(pts))
    assert (d[~far] == 1).all()                          # exact model -> depth 1
    ids = tree._locate(pts)
    hd = half_diag(tree.leaf_level[ids].astype(np.int64))
    err = np.abs(s[far].astype(np.float64) - analytic_sdf(pts[far]))
    assert (err <= hd[far] + 1e-6).all()                 # 1-Lipschitz within cell
    print("PASS test_query_roundtrip  (far frac %.3f, max far err %.2e)"
          % (far.mean(), err.max()))


def test_stats():
    tree = build_tree()
    tree.assign_depths(FakeTMLP("late").to(DEVICE), DEVICE, ERR, 4, 512)
    st = tree.stats()
    assert st["n_leaves"] == st["n_near"] + st["n_far"]
    assert sum(st["depth_hist"].values()) == st["n_leaves"]
    assert st["depth_hist"].get(0, 0) == st["n_far"]
    assert st["depth_hist"].get(3, 0) == st["n_near"]
    assert abs(st["mean_depth_near"] - 3.0) < 1e-9
    print("PASS test_stats  %s" % st)


def test_save_load():
    tree = build_tree()
    tree.assign_depths(FakeTMLP("abs").to(DEVICE), DEVICE, ERR, 8, 1000)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "octree.npz")
        tree.save(path)
        t2 = Octree.load(path)
    assert t2.max_depth == tree.max_depth
    for attr in ("leaf_level", "leaf_idx", "leaf_near", "leaf_depth", "leaf_sdf"):
        a, b = getattr(tree, attr), getattr(t2, attr)
        assert a.dtype == b.dtype and np.array_equal(a, b), attr
    assert t2.stats() == tree.stats()
    rng = np.random.default_rng(2)
    pts = rng.random((2000, 3))
    d1, s1 = tree.query(pts)
    d2, s2 = t2.query(pts)
    assert np.array_equal(d1, d2) and np.array_equal(s1, s2)
    # loaded tree still supports depth assignment
    t2.assign_depths(FakeTMLP("exact").to(DEVICE), DEVICE, ERR, 4, 512)
    assert (t2.leaf_depth[t2.leaf_near] == 1).all()
    print("PASS test_save_load")


if __name__ == "__main__":
    print("device:", DEVICE)
    test_build_covers_surface()
    test_far_stored_sdf()
    test_assign_depths_sign_and_thresh()
    test_query_roundtrip()
    test_stats()
    test_save_load()
    print("ALL OCTREE TESTS PASSED")
