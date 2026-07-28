"""Plain-assert tests for sand/render.py (no pytest).

Run:  .venv/bin/python tests/test_render.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.render as sr
from src.render import depth_colormap, render_mesh, solid_colors

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT_DIR, exist_ok=True)


def make_sphere(subdivisions, radius, center):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    verts = np.asarray(m.vertices, dtype=np.float64) + np.asarray(center, dtype=np.float64)
    return verts, np.asarray(m.faces, dtype=np.int64)


def radial_depth_colors(verts, max_depth=9):
    """Fake radial network-depth pattern -> turbo colors."""
    p0 = np.array([0.5, 0.5, 0.0])
    r = np.linalg.norm(verts - p0, axis=1)
    d = np.clip(r / 0.8, 0.0, 1.0) * max_depth
    return depth_colormap(d, max_depth)


def non_bg_fraction(png_path, bg=(1.0, 1.0, 1.0)):
    from PIL import Image
    arr = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.int16)
    bg8 = (np.asarray(bg) * 255.0 + 0.5).astype(np.int16)
    return float(np.mean(np.any(arr != bg8, axis=-1)))


def test_colormaps():
    # solid_colors
    sc = solid_colors(100)
    assert sc.shape == (100, 3), sc.shape
    assert np.allclose(sc, (0.65, 0.75, 0.9))
    assert sc.dtype == np.float32
    assert sc.min() >= 0.0 and sc.max() <= 1.0
    sc2 = solid_colors(7, rgb=(0.1, 0.2, 0.3))
    assert sc2.shape == (7, 3) and np.allclose(sc2, (0.1, 0.2, 0.3))
    assert solid_colors(0).shape == (0, 3)

    # depth_colormap: shape, range, discrete LUT
    depths = np.arange(10)
    dc = depth_colormap(depths, 9)
    assert dc.shape == (10, 3), dc.shape
    assert dc.min() >= 0.0 and dc.max() <= 1.0
    # index 0 overridden to dark gray 0.15
    assert np.allclose(dc[0], 0.15), dc[0]
    # turbo is not constant: distinct colors along the range
    assert len(np.unique(dc, axis=0)) == 10
    assert not np.allclose(dc[9], dc[1])

    # float inputs: rounding + clipping
    dcf = depth_colormap(np.array([0.4, 0.6, 7.6, -2.0, 100.0, 3.2]), 9)
    assert dcf.shape == (6, 3)
    assert np.allclose(dcf[0], dc[0])    # 0.4 -> 0
    assert np.allclose(dcf[1], dc[1])    # 0.6 -> 1
    assert np.allclose(dcf[2], dc[8])    # 7.6 -> 8
    assert np.allclose(dcf[3], dc[0])    # -2  -> clip to 0 (gray)
    assert np.allclose(dcf[4], dc[9])    # 100 -> clip to max_depth
    assert np.allclose(dcf[5], dc[3])    # 3.2 -> 3
    # max_depth = 0 edge case: everything gray
    dc0 = depth_colormap(np.array([0, 1, 2]), 0)
    assert dc0.shape == (3, 3) and np.allclose(dc0, 0.15)
    print("colormaps OK")


def test_render_basic():
    verts, faces = make_sphere(subdivisions=5, radius=0.3, center=(0.5, 0.5, 0.5))
    assert len(faces) == 20480  # ~20k faces
    colors = radial_depth_colors(verts, max_depth=9)
    assert colors.shape == (len(verts), 3)

    png = os.path.join(OUT_DIR, "sphere_depth.png")
    t0 = time.time()
    render_mesh(verts, faces, colors, png, img_size=400, elev=25.0, azim=45.0,
                title="icosphere depth")
    dt = time.time() - t0
    assert os.path.isfile(png)

    from PIL import Image
    im = Image.open(png)
    assert im.size == (400, 400), im.size
    arr = np.asarray(im.convert("RGB"))
    assert arr.shape == (400, 400, 3)

    frac = non_bg_fraction(png)
    assert frac > 0.05, f"only {frac:.3%} non-background pixels"
    # Lambert shading must produce brightness variation across the sphere
    nb = arr[np.any(arr != 255, axis=-1)]
    assert nb.std() > 1.0, "render looks flat (no shading variation)"
    print(f"basic render OK ({dt:.2f}s, {frac:.1%} non-bg, {len(faces)} faces)")


def test_render_decimation():
    # two spheres, 2 x 20480 = 40960 faces > max_faces=30000
    v1, f1 = make_sphere(subdivisions=5, radius=0.25, center=(0.35, 0.5, 0.5))
    v2, f2 = make_sphere(subdivisions=5, radius=0.25, center=(0.65, 0.5, 0.5))
    verts = np.vstack([v1, v2])
    faces = np.vstack([f1, f2 + len(v1)])
    assert len(faces) == 40960
    colors = radial_depth_colors(verts, max_depth=9)

    # exercise the decimators directly
    nv, nf = sr._vertex_cluster_decimate(verts, faces, 30_000)
    assert 0 < len(nf) <= 30_000, len(nf)
    assert int(nf.max()) < len(nv)
    dv, df = sr._decimate(verts, faces, 30_000)
    assert 0 < len(df) <= 30_000, len(df)
    assert int(df.max()) < len(dv)
    print(f"decimation OK (cluster: {len(nf)} faces, _decimate: {len(df)} faces)")

    png = os.path.join(OUT_DIR, "two_spheres_decimated.png")
    t0 = time.time()
    render_mesh(verts, faces, colors, png, img_size=400, elev=20.0, azim=60.0,
                max_faces=30_000, title="decimated")
    dt = time.time() - t0
    assert os.path.isfile(png)
    from PIL import Image
    assert Image.open(png).size == (400, 400)
    frac = non_bg_fraction(png)
    assert frac > 0.05, f"only {frac:.3%} non-background pixels"
    print(f"decimated render OK ({dt:.2f}s, {frac:.1%} non-bg)")


def test_render_misc():
    # solid colors + dark background + no title; empty mesh must not crash
    verts, faces = make_sphere(subdivisions=3, radius=0.3, center=(0.5, 0.5, 0.5))
    png = os.path.join(OUT_DIR, "sphere_solid_dark.png")
    render_mesh(verts, faces, solid_colors(len(verts)), png, img_size=200,
                elev=60.0, azim=120.0, bg=(0.05, 0.05, 0.08))
    from PIL import Image
    assert Image.open(png).size == (200, 200)
    frac = non_bg_fraction(png, bg=(0.05, 0.05, 0.08))
    assert frac > 0.05

    empty_png = os.path.join(OUT_DIR, "empty.png")
    render_mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
              np.zeros((0, 3)), empty_png, img_size=64)
    assert Image.open(empty_png).size == (64, 64)
    print("misc render OK")


if __name__ == "__main__":
    test_colormaps()
    test_render_basic()
    test_render_decimation()
    test_render_misc()
    print("ALL RENDER TESTS PASSED")
