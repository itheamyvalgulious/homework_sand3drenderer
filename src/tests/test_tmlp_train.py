"""Tests for sand.tmlp and sand.train. Run: .venv/bin/python tests/test_tmlp_train.py"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tmlp import TMLP, BaselineMLP, count_params
from src.train import train_model, load_model
from src.config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SphereData:
    """Fake mesh_data: analytic SDF of a sphere r=0.3 centered at (0.5,0.5,0.5)."""
    has_color = False
    center = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    radius = 0.3

    def sample_training_points(self, n, surf_frac, sigma, rng, domain_frac=0.0):
        n_surf = int(n * surf_frac)
        n_pert = n - n_surf

        d = rng.normal(size=(n_surf, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        surf = self.center + self.radius * d
        sdf_surf = np.zeros(n_surf)

        d = rng.normal(size=(n_pert, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pert = self.center + self.radius * d + rng.normal(scale=sigma, size=(n_pert, 3))
        sdf_pert = np.linalg.norm(pert - self.center, axis=1) - self.radius

        pts = np.concatenate([surf, pert], axis=0).astype(np.float32)
        sdf = np.concatenate([sdf_surf, sdf_pert], axis=0).astype(np.float32)
        return pts, sdf, None


def test_shapes_and_adaptive():
    torch.manual_seed(0)
    model = TMLP(hidden=64, num_layers=8, out_dim=4)
    x = torch.rand(257, 3)

    ys = model.forward_all(x)
    assert isinstance(ys, list) and len(ys) == 8, "forward_all must return 8 tensors"
    assert all(y.shape == (257, 4) for y in ys), "each tail output must be (N,4)"
    torch.testing.assert_close(model.forward_final(x), ys[-1])

    # forward_adaptive with random depths == per-point indexing into forward_all.
    depths = torch.randint(1, 9, (257,), dtype=torch.int64)
    y_ad = model.forward_adaptive(x, depths)
    y_ref = torch.stack([ys[d - 1][i] for i, d in enumerate(depths.tolist())])
    torch.testing.assert_close(y_ad, y_ref)

    # Edge cases: all-min and all-max depths.
    torch.testing.assert_close(model.forward_adaptive(x, torch.ones(257, dtype=torch.int64)),
                               ys[0])
    torch.testing.assert_close(model.forward_adaptive(x, torch.full((257,), 8)),
                               ys[-1])

    # Baseline smoke.
    base = BaselineMLP(hidden=64, num_layers=8, out_dim=1)
    assert base(x).shape == (257, 1)
    assert count_params(model) > 0 and count_params(base) > 0
    print(f"[a] shapes/adaptive OK (TMLP params={count_params(model)}, "
          f"baseline params={count_params(base)})")


def test_overfit_and_roundtrip():
    cfg = Config(hidden=64, num_layers=3, iters=300, batch=4096, lr=1e-3,
                 sigma=0.01, surf_frac=0.6, log_every=50, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        meta = train_model("sand", False, SphereData(), cfg, DEVICE, tmp, tag="test_sand")

        # meta contract + artifacts.
        for k in ("tag", "kind", "color", "iters", "train_time_s", "ms_per_iter",
                  "final_loss", "n_params", "ckpt", "log"):
            assert k in meta, f"missing meta key {k}"
        assert os.path.isfile(meta["ckpt"])
        assert os.path.isfile(os.path.join(tmp, "meta_test_sand.json"))
        with open(os.path.join(tmp, "meta_test_sand.json")) as f:
            assert json.load(f)["tag"] == "test_sand"
        assert meta["train_time_s"] > 0 and meta["ms_per_iter"] > 0
        assert len(meta["log"]) >= 6

        # (b) loss decreases by >50% over 300 iters.
        first = meta["log"][0]["loss"]
        last = float(np.mean([e["loss"] for e in meta["log"][-3:]]))
        assert last < 0.5 * first, f"loss did not drop enough: {first:.4f} -> {last:.4f}"
        print(f"[b] overfit OK on {DEVICE}: loss {first:.4f} -> {last:.4f}")

        # (c) save/load roundtrip via load_model reproduces outputs.
        model = load_model("sand", meta["ckpt"], DEVICE)
        assert isinstance(model, TMLP) and not model.training
        raw = torch.load(meta["ckpt"], map_location=DEVICE, weights_only=True)
        ref = TMLP(**raw["kwargs"]).to(DEVICE)
        ref.load_state_dict(raw["state_dict"])
        ref.eval()
        torch.manual_seed(123)
        x = torch.rand(512, 3, device=DEVICE)
        with torch.no_grad():
            torch.testing.assert_close(model.forward_final(x), ref.forward_final(x))
        print("[c] save/load roundtrip OK")

        # baseline branch smoke (100 tiny iters).
        cfg_b = Config(hidden=64, num_layers=3, iters=100, batch=4096, lr=1e-3,
                       sigma=0.01, surf_frac=0.6, log_every=50, seed=0)
        meta_b = train_model("baseline", False, SphereData(), cfg_b, DEVICE, tmp,
                             tag="test_base")
        model_b = load_model("baseline", meta_b["ckpt"], DEVICE)
        assert isinstance(model_b, BaselineMLP)
        assert meta_b["log"][-1]["loss"] < meta_b["log"][0]["loss"]
        print("[+] baseline branch OK")

        # color=True without colors in mesh_data must raise ValueError.
        try:
            train_model("sand", True, SphereData(), cfg_b, DEVICE, tmp, tag="boom")
        except ValueError:
            print("[+] color-without-data ValueError OK")
        else:
            raise AssertionError("expected ValueError for color=True without colors")


if __name__ == "__main__":
    test_shapes_and_adaptive()
    test_overfit_and_roundtrip()
    print("ALL TESTS PASSED")
