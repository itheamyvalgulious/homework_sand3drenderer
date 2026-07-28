"""Plain-assert tests for src/raymarch.py (no pytest).

Run:  PYTHONPATH=. .venv/bin/python src/tests/test_raymarch.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.octree import Octree
from src.raymarch import (BaselineField, SandField, _TorchOctree, _cell_leap,
                          _make_rays, render_raymarch)
from src.tmlp import TMLP

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)

DEV = "cpu"


def make_fake_octree():
    """8 leaves at level 1 covering [0,1]^3; leaves with ix==0 are 'near'."""
    oc = Octree()
    oc.max_depth = 1
    idx = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                   dtype=np.int32)
    oc.leaf_idx = idx
    oc.leaf_level = np.ones(8, dtype=np.uint8)
    oc.leaf_near = idx[:, 0] == 0
    oc.leaf_depth = np.where(oc.leaf_near, 3, 0).astype(np.uint8)
    # signed distance to the plane x=0.5 at each leaf center
    centers = (idx.astype(np.float64) + 0.5) * 0.5
    oc.leaf_sdf = (np.abs(centers[:, 0] - 0.5) - 0.1).astype(np.float32)
    oc._rebuild_lookup()
    return oc


def test_ttorch_octree_parity():
    oc = make_fake_octree()
    toc = _TorchOctree(oc, torch.device(DEV))
    rng = np.random.default_rng(0)
    pts = rng.random((10_000, 3)).astype(np.float32)
    d_np, s_np = oc.query(pts)
    d_t, s_t, lv_t = toc.query(torch.from_numpy(pts))
    assert np.array_equal(d_t.cpu().numpy(), d_np), "depth mismatch"
    assert np.allclose(s_t.cpu().numpy(), s_np), "stored sdf mismatch"
    assert np.array_equal(lv_t.cpu().numpy(),
                          oc.leaf_level[oc._locate(pts)]), "level mismatch"
    print("torch octree parity OK")


def test_cell_leap():
    # level-1 cell [0,0.5]^3, point (0.1,0.2,0.3), ray along +x:
    # exit face x=0.5 -> distance 0.4 (+ nudge)
    pts = torch.tensor([[0.1, 0.2, 0.3]])
    dirs = torch.tensor([[1.0, 0.0, 0.0]])
    level = torch.tensor([1])
    leap = _cell_leap(pts, dirs, level)
    assert abs(float(leap[0]) - 0.4) < 1e-4, float(leap[0])
    # ray along -y: exit face y=0 -> distance 0.2
    leap = _cell_leap(pts, torch.tensor([[0.0, -1.0, 0.0]]), level)
    assert abs(float(leap[0]) - 0.2) < 1e-4, float(leap[0])
    # diagonal ray: nearest exit face is z=0.5 at 0.2 per component
    d = torch.tensor([[1.0, 1.0, 1.0]]) / np.sqrt(3.0)
    leap = _cell_leap(pts, d, level)
    assert abs(float(leap[0]) - 0.2 * np.sqrt(3.0)) < 1e-3, float(leap[0])
    print("cell leap OK")


class SphereField(BaselineField):
    """Analytic sphere SDF at (0.5,0.5,0.5) r=0.3; depth always `num_layers`."""

    def __init__(self, device):
        super().__init__(None, device, chunk=8192, with_rgb=False)

    def _query_chunk(self, pts):
        n = pts.shape[0]
        sdf = (pts - 0.5).norm(dim=1) - 0.3
        self.stats["queries"] += n
        self.stats["net_depth_sum"] += n * 8
        return (sdf, None,
                torch.full((n,), 8, dtype=torch.long, device=pts.device),
                torch.zeros(n, dtype=torch.long, device=pts.device))


def non_bg_fraction(png_path, bg=(1.0, 1.0, 1.0)):
    from PIL import Image
    arr = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.int16)
    bg8 = (np.asarray(bg) * 255.0 + 0.5).astype(np.int16)
    return float(np.mean(np.any(arr != bg8, axis=-1))), arr


def test_trace_sphere():
    field = SphereField(DEV)
    png = os.path.join(OUT_DIR, "rm_sphere.png")
    dpng = os.path.join(OUT_DIR, "rm_sphere_depth.png")
    stats = render_raymarch(field, png, img_size=128, elev=25.0, azim=45.0,
                            title="analytic sphere", depth_out_path=dpng)
    assert os.path.isfile(png) and os.path.isfile(dpng)
    frac, arr = non_bg_fraction(png)
    assert 0.02 < frac < 0.5, f"implausible hit fraction {frac:.3%}"
    # image center ray must hit the sphere
    center = arr[64, 64]
    assert np.any(center != 255), f"center pixel missed: {center}"
    # shading variation across the sphere
    nb = arr[np.any(arr != 255, axis=-1)]
    assert nb.std() > 1.0, "no shading variation"
    # stats contract
    for k in ("n_rays", "n_hit", "march_queries", "normal_queries",
              "zero_network_frac", "mean_net_depth", "network_s", "trace_s",
              "total_s", "rays_per_s"):
        assert k in stats, k
    assert stats["n_hit"] > 0
    assert stats["march_queries"] > stats["n_rays"]  # multiple steps per ray
    assert stats["zero_network_frac"] == 0.0
    assert stats["mean_net_depth"] == 8.0
    print(f"analytic sphere trace OK ({frac:.1%} hits, "
          f"{stats['march_queries']} queries, {stats['total_s']:.2f}s)")


def test_sand_field_plumbing():
    oc = make_fake_octree()
    torch.manual_seed(0)
    model = TMLP(hidden=16, num_layers=4, out_dim=1, w0=5.0).to(DEV).eval()
    field = SandField(model, oc, DEV, chunk=4096)
    rng = np.random.default_rng(1)
    pts = torch.from_numpy(rng.random((20_000, 3)).astype(np.float32))
    sdf, rgb, depth, level = field.query(pts)
    assert sdf.shape == (20_000,) and rgb is None
    far = depth == 0
    # far points: stored sdf straight from the octree, zero network evals
    d_np, s_np = oc.query(pts.numpy())
    assert np.array_equal((depth > 0).cpu().numpy(), d_np > 0)
    assert np.allclose(sdf[far].cpu().numpy(), s_np[far.cpu().numpy()])
    # near points: genuine network output at their assigned depth
    near_idx = torch.nonzero(~far, as_tuple=True)[0]
    ref = model.forward_adaptive(pts[near_idx], depth[near_idx])[:, 0]
    assert torch.allclose(sdf[near_idx], ref, atol=1e-6)
    assert field.stats["zero_network"] == int(far.sum())
    # full render through the SAND field must terminate and stay white
    # (network is untrained; the far region leaps, nothing should hang)
    png = os.path.join(OUT_DIR, "rm_sand_plumbing.png")
    stats = render_raymarch(field, png, img_size=64, max_steps=64)
    assert os.path.isfile(png)
    assert stats["march_queries"] > 0
    print(f"sand field plumbing OK (zero-network "
          f"{field.stats['zero_network']}/{field.stats['queries']})")


def test_make_rays_box():
    eye, dirs, t0, t1, valid = _make_rays(64, 25.0, 45.0)
    assert dirs.shape == (64, 64, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0, atol=1e-5)
    assert valid[32, 32] and not valid[0, 0]  # center hits the box, corner not
    assert np.all(t1[valid] > t0[valid])
    print("ray generation OK")


if __name__ == "__main__":
    test_make_rays_box()
    test_ttorch_octree_parity()
    test_cell_leap()
    test_trace_sphere()
    test_sand_field_plumbing()
    print("ALL RAYMARCH TESTS PASSED")
