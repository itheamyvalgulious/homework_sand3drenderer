"""Re-extract and re-render at 512^3 (paper's MC resolution).

The front body of the robot is a sandwich of thin plates ~0.008 normalized
units apart = ~2 voxels at 256^3, which Marching Cubes merges into torn
geometry. At 512^3 the gap is ~4 voxels and separable. SAND adaptive eval
makes this cheap (98% of points are depth-0). Baseline is also re-evaluated
at 512^3 for a fair visual comparison. Writes render_{solid,depth,textured,
baseline_textured}_res512.png and mesh_*_res512.ply alongside existing files
(nothing is overwritten).

Usage: .venv/bin/python scripts/rerender_res512.py [--out outputs_robot_union]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import trimesh

from src.config import Config
from src.octree import Octree
from src import infer, train as train_mod
from src.render import render_mesh, depth_colormap, solid_colors

_MF = 8_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_robot_union")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()
    out = args.out
    res = args.res

    cfg = Config.load(os.path.join(out, "config.json"))
    octree_path = os.path.join(out, "octree_sand_color_rgbcrit.npz")
    if not os.path.exists(octree_path):
        octree_path = os.path.join(out, "octree_sand_color.npz")

    t0 = time.perf_counter()
    model = train_mod.load_model("sand", os.path.join(out, "ckpt_sand_color.pt"),
                                 cfg.device)
    oc = Octree.load(octree_path)
    r = infer.eval_grid_sand(model, oc, res, cfg.device, cfg.chunk, True)
    print(f"[sand 512] {r['timing']}", flush=True)
    m = infer.extract_mesh(r["sdf"], r["rgb"], r["depth"])
    print(f"[sand 512] mesh verts={len(m['verts'])} faces={len(m['faces'])}",
          flush=True)
    vc = (m["vert_rgb"] * 255).astype(np.uint8)
    trimesh.Trimesh(m["verts"], m["faces"], vertex_colors=vc,
                    process=False).export(os.path.join(out, "mesh_sand_color_res512.ply"))
    render_mesh(m["verts"], m["faces"], solid_colors(len(m["verts"])),
                os.path.join(out, "render_solid_res512.png"), max_faces=_MF,
                title="SAND solid @512")
    print("saved render_solid_res512.png", flush=True)
    render_mesh(m["verts"], m["faces"],
                depth_colormap(np.round(m["vert_depth"]), cfg.num_layers),
                os.path.join(out, "render_depth_res512.png"), max_faces=_MF,
                title="SAND network depth @512")
    print("saved render_depth_res512.png", flush=True)
    render_mesh(m["verts"], m["faces"], m["vert_rgb"],
                os.path.join(out, "render_textured_res512.png"), max_faces=_MF,
                title="SAND textured @512")
    print("saved render_textured_res512.png", flush=True)
    del model, oc, r, m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    bckpt = os.path.join(out, "ckpt_base_color.pt")
    if os.path.exists(bckpt):
        bmodel = train_mod.load_model("baseline", bckpt, cfg.device)
        rb = infer.eval_grid_baseline(bmodel, res, cfg.device, cfg.chunk, True)
        print(f"[base 512] {rb['timing']}", flush=True)
        mb = infer.extract_mesh(rb["sdf"], rb["rgb"], None)
        vcb = (mb["vert_rgb"] * 255).astype(np.uint8)
        trimesh.Trimesh(mb["verts"], mb["faces"], vertex_colors=vcb,
                        process=False).export(os.path.join(out, "mesh_base_color_res512.ply"))
        render_mesh(mb["verts"], mb["faces"], mb["vert_rgb"],
                    os.path.join(out, "render_baseline_textured_res512.png"),
                    max_faces=_MF, title="Baseline textured @512")
        print("saved render_baseline_textured_res512.png", flush=True)
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
