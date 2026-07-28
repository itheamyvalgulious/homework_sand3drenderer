"""Render multi-angle 4-panel composites (solid | depth | textured | reference).

For each model: load the best extracted mesh (PLY with network RGB), the octree
(for depth coloring), and the original model (for GT-texture reference). Render
6 camera angles, each as a 4-panel side-by-side composite.

Usage: .venv/bin/python scripts/multi_angle.py [--mesh PLY] [--octree NPZ] \
       [--orig GLB] [--out DIR] [--solid-union-res N]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from src.config import Config
from src.data import MeshData
from src.octree import Octree
from src.render import render_mesh, depth_colormap, solid_colors

_ANGLES = [
    ("front", 25.0, 45.0),
    ("back", 25.0, 225.0),
    ("left", 25.0, 135.0),
    ("right", 25.0, 315.0),
    ("top", 65.0, 45.0),
    ("low", 8.0, 45.0),
]
_MF = 8_000_000
_IS = 800


def vert_colors_from_ply(ply_path):
    """Load PLY and return (verts, faces, vert_rgb)."""
    m = trimesh.load(ply_path, process=False)
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    vc = np.asarray(m.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0 \
        if m.visual.vertex_colors is not None else None
    return v, f, vc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="PLY with network vertex colors")
    ap.add_argument("--octree", default=None, help="NPZ octree for depth coloring")
    ap.add_argument("--orig", required=True, help="original .glb for GT reference")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--solid-union-res", type=int, default=None)
    ap.add_argument("--name", default="model", help="label for composites")
    ap.add_argument("--num-layers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    v, f, vrgb = vert_colors_from_ply(args.mesh)

    # depth coloring at mesh vertices
    vdepth = None
    if args.octree and os.path.exists(args.octree):
        oc = Octree.load(args.octree)
        d, _ = oc.query(v)
        vdepth = d.astype(np.float32)

    # GT reference: load the ORIGINAL mesh (no proxy) for true appearance
    md = MeshData.load(args.orig)  # no solid_union_res -> original mesh + its UVs
    if md._tex is not None:
        uv = np.asarray(md._uv)
        h, w = md._tex.shape[:2]
        x = np.round(uv[:, 0] * (w - 1)).astype(np.int64) % w
        y = np.round((1.0 - uv[:, 1]) * (h - 1)).astype(np.int64) % h
        ref_rgb = md._tex[y, x].astype(np.float32)
    else:
        ref_rgb = md._vcolors
    ref_v, ref_f = np.asarray(md.mesh.vertices), np.asarray(md.mesh.faces)

    sc = solid_colors(len(v))
    dc = depth_colormap(np.round(vdepth), args.num_layers) if vdepth is not None else sc

    for tag, elev, azim in _ANGLES:
        panels = []
        for kind, colors, title in [
            ("solid", sc, f"{args.name} solid {tag}"),
            ("depth", dc, f"{args.name} depth {tag}"),
            ("textured", vrgb, f"{args.name} textured {tag}"),
            ("reference", ref_rgb, f"{args.name} ref {tag}"),
        ]:
            if colors is None:
                colors = sc
            # reference panel uses the original mesh; others use the SAND PLY mesh
            rv, rf = (ref_v, ref_f) if kind == "reference" else (v, f)
            path = os.path.join(args.out, f"_tmp_{args.name}_{tag}_{kind}.png")
            render_mesh(rv, rf, colors, path, img_size=_IS, elev=elev, azim=azim,
                        max_faces=_MF, title=title)
            panels.append(Image.open(path).convert("RGB"))
            os.unlink(path)

        # composite
        w = sum(p.width for p in panels) + 6 * (len(panels) - 1)
        h = max(p.height for p in panels)
        canvas = Image.new("RGB", (w, h + 24), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        d.text((4, h + 4), f"{args.name} — {tag} (elev={elev} azim={azim})",
               fill=(0, 0, 0))
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width + 6
        outp = os.path.join(args.out, f"multiangle_{tag}.png")
        canvas.save(outp)
        print(f"saved {outp}", flush=True)


if __name__ == "__main__":
    main()
