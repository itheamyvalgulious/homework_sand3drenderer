"""Re-render the robot showcase images from existing checkpoints.

Applies the post-run fixes (RGB clamp instead of sigmoid, full-resolution
renders without decimation) to the already-trained SAND models, and adds a
reference render of the ORIGINAL mesh with its GT texture for comparison.

Usage: .venv/bin/python scripts/rerender_showcase.py [--out outputs]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

from src.config import Config
from src.data import MeshData
from src.octree import Octree
from src import infer, train as train_mod
from src.render import render_mesh, depth_colormap, solid_colors

_MF = 5_000_000  # no decimation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = args.out

    cfg = Config.load(os.path.join(out, "config.json"))
    model = train_mod.load_model("sand", os.path.join(out, "ckpt_sand_color.pt"),
                                 cfg.device)
    octree = Octree.load(os.path.join(out, "octree_sand_color.npz"))
    r = infer.eval_grid_sand(model, octree, cfg.res, cfg.device, cfg.chunk, True)
    m = infer.extract_mesh(r["sdf"], r["rgb"], r["depth"])
    print(f"mesh: verts={len(m['verts'])} faces={len(m['faces'])}")

    vc = (m["vert_rgb"] * 255).astype(np.uint8) if m["vert_rgb"] is not None else None
    trimesh.Trimesh(m["verts"], m["faces"], vertex_colors=vc,
                    process=False).export(os.path.join(out, "mesh_sand_color.ply"))

    render_mesh(m["verts"], m["faces"], solid_colors(len(m["verts"])),
                os.path.join(out, "render_solid.png"), max_faces=_MF,
                title="SAND solid (sand_color)")
    print("saved render_solid.png", flush=True)
    render_mesh(m["verts"], m["faces"],
                depth_colormap(np.round(m["vert_depth"]), cfg.num_layers),
                os.path.join(out, "render_depth.png"), max_faces=_MF,
                title="SAND network depth (sand_color)")
    print("saved render_depth.png", flush=True)
    render_mesh(m["verts"], m["faces"], m["vert_rgb"],
                os.path.join(out, "render_textured.png"), max_faces=_MF,
                title="SAND textured (sand_color)")
    print("saved render_textured.png", flush=True)

    # Reference: original mesh with GT texture sampled at its vertices.
    md = MeshData.load(cfg.model)
    if md._tex is not None:
        uv = np.asarray(md._uv)
        h, w = md._tex.shape[:2]
        x = np.round(uv[:, 0] * (w - 1)).astype(np.int64) % w
        y = np.round((1.0 - uv[:, 1]) * (h - 1)).astype(np.int64) % h
        vc_gt = md._tex[y, x]
    else:
        vc_gt = md._vcolors
    render_mesh(md.mesh.vertices, md.mesh.faces, vc_gt,
                os.path.join(out, "render_reference.png"), max_faces=_MF,
                title="Original model (GT texture)")
    print("saved render_reference.png")


if __name__ == "__main__":
    main()
