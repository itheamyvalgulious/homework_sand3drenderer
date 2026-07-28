"""Tests for sand/data.py and scripts/make_sample_model.py.

Run with: .venv/bin/python tests/test_data.py
"""
import os
import subprocess
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data import MeshData  # noqa: E402

SAMPLE = os.path.join(ROOT, "assets", "sample.glb")


def ensure_sample() -> None:
    """Generate assets/sample.glb via scripts/make_sample_model.py if missing."""
    if not os.path.exists(SAMPLE):
        script = os.path.join(ROOT, "scripts", "make_sample_model.py")
        subprocess.run([sys.executable, script, "--out", SAMPLE], check=True)
    assert os.path.exists(SAMPLE)


def test_normalization(md: MeshData) -> None:
    b = md.mesh.bounds
    center = (b[0] + b[1]) / 2
    extent = (b[1] - b[0]).max()
    assert np.allclose(center, 0.5, atol=1e-4), f"bbox center {center}"
    assert abs(extent - 0.9) < 1e-4, f"longest bbox edge {extent}"
    assert b.min() >= -1e-6 and b.max() <= 1.0 + 1e-6, "mesh escapes unit cube"
    assert md.is_watertight(), "sample model must be watertight"
    assert md.source_path == SAMPLE


def test_training_points(md: MeshData) -> None:
    rng = np.random.default_rng(0)
    pts, sdf, rgb = md.sample_training_points(4096, 0.6, 0.01, rng)
    assert pts.shape == (4096, 3) and pts.dtype == np.float32
    assert sdf.shape == (4096,) and sdf.dtype == np.float32
    assert rgb is not None and rgb.shape == (4096, 3) and rgb.dtype == np.float32

    n_surf = int(round(4096 * 0.6))
    med = float(np.median(np.abs(sdf[:n_surf])))
    assert med < 1e-3, f"on-surface median |sdf| {med}"
    # Gaussian perturbation was actually applied to the off-surface part.
    spread = float(np.std(sdf[n_surf:]))
    assert spread > 1e-4, f"perturbed sdf std {spread} too small"
    # GT SDF is finite everywhere.
    assert np.isfinite(sdf).all()


def test_colors(md: MeshData) -> None:
    assert md.has_color, "sample model must carry color"
    rng = np.random.default_rng(1)
    pts, rgb = md.surface_points(2048, rng)
    assert pts.shape == (2048, 3) and pts.dtype == np.float32
    assert rgb is not None and rgb.shape == (2048, 3) and rgb.dtype == np.float32
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0, "rgb outside [0,1]"
    n_unique = len(np.unique(np.round(rgb, 3), axis=0))
    assert n_unique > 16, f"colors look constant ({n_unique} unique)"


def test_sphere_sdf_sign() -> None:
    """sdf_at sign agrees with the analytic SDF of a known sphere."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
    mesh.apply_translation([0.5, 0.5, 0.5])
    md = MeshData(mesh, has_color=False, source_path="<in-memory sphere>")

    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 1, size=(20000, 3)).astype(np.float32)
    analytic = np.linalg.norm(pts - 0.5, axis=1) - 0.4
    mask = np.abs(analytic) > 0.02  # stay clear of the tessellated boundary
    sdf = md.sdf_at(pts)
    assert sdf.dtype == np.float32 and sdf.shape == (20000,)
    agree = float(np.mean(np.sign(sdf[mask]) == np.sign(analytic[mask])))
    assert agree > 0.99, f"sign agreement {agree}"
    # Sphere center is deep inside -> clearly negative.
    assert md.sdf_at(np.array([[0.5, 0.5, 0.5]]))[0] < 0
    # No color on this mesh -> rgb must be None.
    _, rgb = md.surface_points(64, np.random.default_rng(3))
    assert rgb is None and not md.has_color


def main() -> None:
    ensure_sample()
    md = MeshData.load(SAMPLE)
    test_normalization(md)
    test_training_points(md)
    test_colors(md)
    test_sphere_sdf_sign()
    print("all tests passed")


if __name__ == "__main__":
    main()
