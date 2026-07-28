"""Create a guaranteed-watertight textured sample model for the SAND pipeline.

Usage: python scripts/make_sample_model.py [--out assets/sample.glb]

Tries to download the Khronos Duck first; if that fails, or the Duck is not
watertight / lacks a texture image with UVs, falls back to a procedural
textured icosphere. The exported model is raw (unnormalized) — normalization
into [0,1]^3 happens in sand.data.MeshData.load.
"""
import argparse
import io
import os
import urllib.request

import numpy as np
import trimesh
from PIL import Image

DUCK_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/"
    "main/Models/Duck/glTF-Binary/Duck.glb"
)


def _as_single_mesh(loaded) -> trimesh.Trimesh:
    """Reduce a trimesh.load result to a single Trimesh."""
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 1:
            return geoms[0]
        return trimesh.util.concatenate(geoms)
    return loaded


def _has_texture(mesh: trimesh.Trimesh) -> bool:
    """True if the mesh has both UVs and a texture image."""
    vis = mesh.visual
    uv = getattr(vis, "uv", None)
    mat = getattr(vis, "material", None)
    img = getattr(mat, "image", None) if mat is not None else None
    if img is None and mat is not None:
        img = getattr(mat, "baseColorTexture", None)
    return uv is not None and len(uv) == len(mesh.vertices) and img is not None


def load_duck() -> trimesh.Trimesh:
    """Download the Khronos Duck; accept only if watertight and textured."""
    with urllib.request.urlopen(DUCK_URL, timeout=20) as resp:
        data = resp.read()
    mesh = _as_single_mesh(trimesh.load(io.BytesIO(data), file_type="glb"))
    if not mesh.is_watertight:
        raise ValueError("Duck mesh is not watertight")
    if not _has_texture(mesh):
        raise ValueError("Duck mesh has no texture image + UVs")
    return mesh


def make_checker_texture(size: int = 256, squares: int = 16) -> Image.Image:
    """Colorful checkerboard with RGB gradients and a sine-wave blue channel."""
    grid = np.mgrid[0:size, 0:size].astype(np.float32)
    xx, yy = grid[0], grid[1]
    check = (np.floor(xx * squares / size) + np.floor(yy * squares / size)) % 2
    r = xx / (size - 1)
    g = yy / (size - 1)
    b = 0.5 + 0.5 * np.sin((xx + yy) / size * 4 * np.pi)
    on = check == 0
    arr = np.stack(
        [
            np.where(on, r, 1.0 - r),
            np.where(on, g, 1.0 - g),
            np.where(on, b, 1.0 - b),
        ],
        axis=-1,
    )
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


def load_procedural() -> trimesh.Trimesh:
    """Watertight icosphere with spherical UVs and a checker/gradient texture."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # Spherical UV projection: u from longitude, v from polar angle.
    vn = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    u = (np.arctan2(vn[:, 1], vn[:, 0]) / (2 * np.pi)) % 1.0
    v = np.arccos(np.clip(vn[:, 2], -1.0, 1.0)) / np.pi
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.column_stack([u, v]), image=make_checker_texture()
    )
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="assets/sample.glb",
        help="output .glb path (default: assets/sample.glb)",
    )
    args = parser.parse_args()

    try:
        mesh = load_duck()
        which = f"downloaded Khronos Duck ({DUCK_URL})"
    except Exception as exc:
        mesh = load_procedural()
        which = f"procedural textured icosphere (Duck unusable: {exc})"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mesh.export(args.out)
    print(f"sample model written to {args.out} via {which}")
    print(
        f"  watertight={mesh.is_watertight}, textured={_has_texture(mesh)}, "
        f"vertices={len(mesh.vertices)}, faces={len(mesh.faces)}"
    )


if __name__ == "__main__":
    main()
