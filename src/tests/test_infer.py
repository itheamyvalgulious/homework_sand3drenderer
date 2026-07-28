"""Tests for sand/infer.py. Plain script (no pytest):

    .venv/bin/python tests/test_infer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.infer import eval_grid_sand, eval_grid_baseline, extract_mesh

CENTER = np.array([0.5, 0.5, 0.5], dtype=np.float64)
RADIUS = 0.3
RAW_RGB = (0.0, 2.0, -2.0)  # constant raw (pre-clamp) color of the fake models


def sphere_sdf(pts):
    """Analytic SDF of a sphere centered in the unit cube."""
    pts = np.asarray(pts, dtype=np.float64)
    return (np.linalg.norm(pts - CENTER, axis=-1) - RADIUS).astype(np.float32)


def grid_pts(res):
    axis = np.linspace(0.5 / res, 1.0 - 0.5 / res, res)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)  # (res,res,res,3)


class FakeTMLP(torch.nn.Module):
    """Analytic stand-in honoring the TMLP contract (sand/tmlp.py).

    Tail 1 is coarse (offset by 0.01), tails >= 2 reproduce the exact sphere
    SDF on channel 0; channels 1:4 carry a constant raw RGB.
    """

    def __init__(self, num_layers=3, out_dim=4):
        super().__init__()
        self.hidden = 64
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.w0 = 30.0

    def forward_all(self, x):
        sdf = torch.from_numpy(sphere_sdf(x.detach().cpu().numpy())).to(x.device)
        ys = []
        for i in range(1, self.num_layers + 1):
            y = torch.zeros(x.shape[0], self.out_dim, device=x.device)
            y[:, 0] = sdf + (0.01 if i == 1 else 0.0)
            if self.out_dim == 4:
                y[:, 1], y[:, 2], y[:, 3] = RAW_RGB
            ys.append(y)
        return ys

    def forward_final(self, x):
        return self.forward_all(x)[-1]

    def forward_adaptive(self, x, depths):
        ys = torch.stack(self.forward_all(x), dim=1)  # (N, L, out_dim)
        idx = (depths.long() - 1).clamp(0, self.num_layers - 1)
        return ys[torch.arange(x.shape[0], device=x.device), idx]


class FakeBaseline(torch.nn.Module):
    """Analytic stand-in honoring the BaselineMLP contract (forward only)."""

    def __init__(self, num_layers=3, out_dim=4):
        super().__init__()
        self.hidden = 64
        self.num_layers = num_layers
        self.out_dim = out_dim

    def forward(self, x):
        sdf = torch.from_numpy(sphere_sdf(x.detach().cpu().numpy())).to(x.device)
        y = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        y[:, 0] = sdf
        if self.out_dim == 4:
            y[:, 1], y[:, 2], y[:, 3] = RAW_RGB
        return y


class FakeOctree:
    """Plane split: x < 0.5 -> far leaf (depth 0, stored analytic SDF), else depth 2."""

    max_depth = 9

    def query(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        net_depth = np.where(pts[:, 0] < 0.5, 0, 2).astype(np.uint8)
        return net_depth, sphere_sdf(pts)


def test_extract_mesh_sphere():
    res = 32
    sdf = sphere_sdf(grid_pts(res)).reshape(res, res, res)
    out = extract_mesh(sdf)
    verts, faces = out["verts"], out["faces"]
    assert verts.dtype == np.float32 and faces.dtype == np.int64
    assert verts.shape[0] > 100 and faces.shape[0] > 100
    assert verts.min() >= 0.0 and verts.max() <= 1.0
    assert out["vert_rgb"] is None and out["vert_depth"] is None

    # Closed mesh: every undirected edge is shared by exactly two faces.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), f"mesh not closed: {(counts != 2).sum()} bad edges"

    r = np.linalg.norm(verts.astype(np.float64) - CENTER, axis=1)
    assert abs(r.mean() - RADIUS) < 0.02, f"mean radius {r.mean()}"
    assert r.std() < 0.02, f"radius std {r.std()}"
    c = verts.mean(axis=0)
    assert np.all(np.abs(c - CENTER) < 0.01), f"center {c}"

    # Trilinear interpolation of rgb / depth grids at the vertices.
    rgb_grid = np.full((res, res, res, 3), [0.2, 0.4, 0.6], dtype=np.float32)
    depth_grid = np.full((res, res, res), 3, dtype=np.uint8)
    out2 = extract_mesh(sdf, rgb_grid=rgb_grid, depth_grid=depth_grid)
    assert out2["vert_rgb"].shape == (verts.shape[0], 3)
    assert out2["vert_rgb"].dtype == np.float32
    assert np.allclose(out2["vert_rgb"], [0.2, 0.4, 0.6], atol=1e-5)
    assert out2["vert_depth"].shape == (verts.shape[0],)
    assert np.allclose(out2["vert_depth"], 3.0, atol=1e-5)

    # No level set -> clear error.
    try:
        extract_mesh(np.ones((8, 8, 8), dtype=np.float32))
        raise AssertionError("expected ValueError for all-positive SDF grid")
    except ValueError:
        pass


def test_eval_grid_sand():
    res = 24
    model = FakeTMLP(num_layers=3, out_dim=4)
    octree = FakeOctree()
    out = eval_grid_sand(model, octree, res=res, device="cpu", chunk=5000,
                         with_rgb=True)
    pts = grid_pts(res)
    xs = pts[..., 0]
    sdf_gt = sphere_sdf(pts).reshape(res, res, res)

    assert out["sdf"].shape == (res, res, res) and out["sdf"].dtype == np.float32
    assert np.allclose(out["sdf"], sdf_gt, atol=2e-3), \
        f"max sdf err {np.abs(out['sdf'] - sdf_gt).max()}"

    depth = out["depth"]
    assert depth.shape == (res, res, res) and depth.dtype == np.uint8
    assert np.all(depth[xs < 0.5] == 0), "depth-0 plane violated"
    assert np.all(depth[xs >= 0.5] == 2), "depth-2 region violated"

    rgb = out["rgb"]
    assert rgb is not None and rgb.shape == (res, res, res, 3)
    assert rgb.dtype == np.float32 and rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert np.allclose(rgb[xs < 0.5], 0.5, atol=1e-6), "far points not gray"
    exp_rgb = np.clip(np.array(RAW_RGB, dtype=np.float64), 0.0, 1.0)
    assert np.allclose(rgb[xs >= 0.5], exp_rgb, atol=1e-5), "network rgb wrong"

    t = out["timing"]
    for k in ("octree_query_s", "network_s", "total_s", "mean_depth",
              "frac_depth0", "n_points"):
        assert k in t, f"missing timing key {k}"
    assert t["n_points"] == res ** 3
    assert abs(t["frac_depth0"] - float((xs < 0.5).mean())) < 1e-6
    assert 0.0 < t["mean_depth"] <= 2.0
    assert t["total_s"] >= t["network_s"] >= 0.0

    # with_rgb=False -> no rgb grid at all.
    out2 = eval_grid_sand(model, octree, res=res, device="cpu", chunk=100_000,
                          with_rgb=False)
    assert out2["rgb"] is None
    assert np.allclose(out2["sdf"], sdf_gt, atol=2e-3)


def test_eval_grid_baseline():
    res = 16
    pts = grid_pts(res)
    sdf_gt = sphere_sdf(pts).reshape(res, res, res)

    model = FakeBaseline(num_layers=3, out_dim=4)
    out = eval_grid_baseline(model, res=res, device="cpu", chunk=2000, with_rgb=True)
    assert out["sdf"].shape == (res, res, res) and out["sdf"].dtype == np.float32
    assert np.allclose(out["sdf"], sdf_gt, atol=2e-3)
    assert out["depth"].shape == (res, res, res) and out["depth"].dtype == np.uint8
    assert np.all(out["depth"] == 3), "baseline depth must be num_layers everywhere"
    assert out["rgb"] is not None and out["rgb"].shape == (res, res, res, 3)
    assert out["rgb"].min() >= 0.0 and out["rgb"].max() <= 1.0
    exp_rgb = np.clip(np.array(RAW_RGB, dtype=np.float64), 0.0, 1.0)
    assert np.allclose(out["rgb"], exp_rgb, atol=1e-5)
    t = out["timing"]
    for k in ("network_s", "total_s", "n_points"):
        assert k in t, f"missing timing key {k}"
    assert t["n_points"] == res ** 3

    # with_rgb=False -> None; TMLP-style model goes through forward_final.
    out2 = eval_grid_baseline(model, res=res, device="cpu", chunk=10 ** 9,
                              with_rgb=False)
    assert out2["rgb"] is None
    out3 = eval_grid_baseline(FakeTMLP(num_layers=3, out_dim=4), res=res,
                              device="cpu", chunk=2000, with_rgb=True)
    assert np.allclose(out3["sdf"], sdf_gt, atol=2e-3)
    assert np.all(out3["depth"] == 3)


if __name__ == "__main__":
    test_extract_mesh_sphere()
    print("ok: extract_mesh (sphere, closed, interp, empty-guard)")
    test_eval_grid_sand()
    print("ok: eval_grid_sand (plane split, sdf/rgb/depth, timings)")
    test_eval_grid_baseline()
    print("ok: eval_grid_baseline (shapes, depth fill, timings)")
    print("ALL TESTS PASSED")
