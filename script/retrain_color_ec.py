"""Retrain the SAND color model with EXACT color transfer (post-proxy).

The first union run trained colors from a nearest-sample KDTree transfer,
which visibly errs in high-frequency texture areas (display, decals, eyes:
p99 error 0.28). The exact closest-point + barycentric transfer has ~0 error.
This script retrains only the color model (tag sand_color_ec), rebuilds the
joint SDF+RGB depth map, re-evals at 512^3 and renders. Existing files are
kept (new suffixes _ec).

Usage: .venv/bin/python scripts/retrain_color_ec.py [--out outputs_robot_union]
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
from src.data import MeshData
from src.octree import Octree
from src import infer, train as train_mod
from src.render import render_mesh, depth_colormap, solid_colors

_MF = 8_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_robot_union")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--rgb-thresh", type=float, default=0.01)
    args = ap.parse_args()
    out = args.out

    cfg = Config.load(os.path.join(out, "config.json"))
    md = MeshData.load(cfg.model, solid_union_res=cfg.solid_union_res)
    octree = Octree.load(os.path.join(out, "octree_base.npz"))

    t0 = time.perf_counter()
    meta = train_mod.train_model("sand", True, md, cfg, cfg.device, out,
                                 "sand_color_ec", octree=octree)
    print(f"[sand_color_ec] train done: {meta['train_time_s']:.0f}s "
          f"final_loss={meta['final_loss']:.4f}", flush=True)

    model = train_mod.load_model("sand", meta["ckpt"], cfg.device)
    oc = Octree.load(os.path.join(out, "octree_base.npz"))
    oc.assign_depths(model, cfg.device, cfg.err_thresh,
                     cfg.depth_samples_per_leaf, cfg.chunk,
                     rgb_thresh=args.rgb_thresh)
    oc.save(os.path.join(out, "octree_sand_color_ec_rgbcrit.npz"))
    print("stats:", oc.stats(), flush=True)

    r = infer.eval_grid_sand(model, oc, args.res, cfg.device, cfg.chunk, True)
    print(f"[ec 512] {r['timing']}", flush=True)
    m = infer.extract_mesh(r["sdf"], r["rgb"], r["depth"])
    print(f"[ec 512] mesh verts={len(m['verts'])} faces={len(m['faces'])}",
          flush=True)
    vc = (m["vert_rgb"] * 255).astype(np.uint8)
    trimesh.Trimesh(m["verts"], m["faces"], vertex_colors=vc,
                    process=False).export(os.path.join(out, "mesh_sand_color_ec_res512.ply"))
    render_mesh(m["verts"], m["faces"], solid_colors(len(m["verts"])),
                os.path.join(out, "render_solid_res512_ec.png"), max_faces=_MF,
                title="SAND solid @512 exact-color")
    render_mesh(m["verts"], m["faces"],
                depth_colormap(np.round(m["vert_depth"]), cfg.num_layers),
                os.path.join(out, "render_depth_res512_ec.png"), max_faces=_MF,
                title="SAND depth @512 exact-color")
    render_mesh(m["verts"], m["faces"], m["vert_rgb"],
                os.path.join(out, "render_textured_res512_ec.png"), max_faces=_MF,
                title="SAND textured @512 exact-color")
    print("saved render_*_res512_ec.png", flush=True)
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
