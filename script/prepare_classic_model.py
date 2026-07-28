"""Download Blender's Suzanne (monkey head) and export a textured glb.

Suzanne ships without UVs, so we generate a spherical UV projection and a
colorful procedural checker/gradient texture — same approach as the sample
asset — giving the color-field training something to learn.

Usage: .venv/bin/python scripts/prepare_classic_model.py [--out assets/suzanne.glb]
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np
import trimesh
from PIL import Image

URL = ("https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/"
       "master/data/suzanne.obj")


def make_texture(size: int = 512, checks: int = 16) -> Image.Image:
    """Colorful checkerboard with a smooth hue gradient (clearly directional)."""
    yy, xx = np.mgrid[0:size, 0:size]
    u = xx / (size - 1)
    v = yy / (size - 1)
    check = ((xx * checks // size) + (yy * checks // size)) % 2
    r = np.clip(0.25 + 0.75 * u, 0, 1)
    g = np.clip(0.25 + 0.75 * v, 0, 1)
    b = np.clip(0.9 - 0.6 * u * v, 0, 1)
    img = np.stack([r, g, b], axis=-1)
    img = np.where(check[..., None], img, img * 0.25 + 0.05)
    return Image.fromarray((img * 255).astype(np.uint8))


def spherical_uv(verts: np.ndarray) -> np.ndarray:
    c = verts.mean(axis=0)
    d = verts - c
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    u = (np.arctan2(d[:, 2], d[:, 0]) / (2 * np.pi)) + 0.5
    v = np.arccos(np.clip(d[:, 1], -1, 1)) / np.pi
    return np.stack([u, v], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/suzanne.glb")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = os.path.join(os.path.dirname(args.out), "suzanne.obj")
    if not os.path.exists(tmp):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, tmp)
    mesh = trimesh.load(tmp)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    print(f"suzanne raw: verts={len(mesh.vertices)} faces={len(mesh.faces)} "
          f"watertight={mesh.is_watertight}")
    # Drop stray sliver components (this OBJ ships a 3-vertex needle) while
    # keeping the head and the two separate eye shells.
    comps = [c for c in mesh.split(only_watertight=False) if len(c.faces) >= 10]
    mesh = trimesh.util.concatenate(comps)
    # Face points +Z; rotate 45 deg about Y so it faces the default pipeline
    # camera (azim=45 looks from the +X/+Z diagonal).
    mesh.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(45.0), [0.0, 1.0, 0.0], point=mesh.centroid))
    print(f"suzanne cleaned: {len(comps)} components kept")
    # The classic OBJ is the unsubdivided Blender primitive (507 v / 968 f).
    # Midpoint-subdivide x3 then lightly Laplacian-smooth for a showcase-grade
    # surface; UVs are generated afterwards, so nothing is lost.
    for _ in range(3):
        mesh = mesh.subdivide()
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.4, iterations=5,
                                       volume_constraint=True)
    print(f"suzanne subdivided: verts={len(mesh.vertices)} "
          f"faces={len(mesh.faces)} watertight={mesh.is_watertight}")
    uv = spherical_uv(np.asarray(mesh.vertices))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=make_texture())
    mesh.export(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
