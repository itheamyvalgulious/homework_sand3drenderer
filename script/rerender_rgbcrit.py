"""Re-render the textured image with a joint SDF+RGB depth criterion.

The paper's depth map (Eq. 5) gates on SDF error only, so early-exit tails can
have unconverged RGB in textured regions (smearing artifacts). This script
rebuilds the depth map for the color model with the additional requirement
max|dRGB| < rgb_thresh (no retraining needed), then re-evals, re-extracts and
re-renders. Outputs: octree_sand_color_rgbcrit.npz, render_textured_rgbcrit.png,
render_depth_rgbcrit.png, and prints the new eval timing.

Usage: .venv/bin/python scripts/rerender_rgbcrit.py [--out outputs_suzanne] [--rgb-thresh 0.01]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import Config
from src.octree import Octree
from src import infer, train as train_mod
from src.render import render_mesh, depth_colormap

_MF = 5_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_suzanne")
    ap.add_argument("--rgb-thresh", type=float, default=0.01)
    args = ap.parse_args()
    out = args.out

    cfg = Config.load(os.path.join(out, "config.json"))
    model = train_mod.load_model("sand", os.path.join(out, "ckpt_sand_color.pt"),
                                 cfg.device)
    oc = Octree.load(os.path.join(out, "octree_base.npz"))
    oc.assign_depths(model, cfg.device, cfg.err_thresh,
                     cfg.depth_samples_per_leaf, cfg.chunk,
                     rgb_thresh=args.rgb_thresh)
    npz = os.path.join(out, "octree_sand_color_rgbcrit.npz")
    oc.save(npz)
    print("stats:", oc.stats(), flush=True)

    r = infer.eval_grid_sand(model, oc, cfg.res, cfg.device, cfg.chunk, True)
    print("eval timing:", r["timing"], flush=True)
    m = infer.extract_mesh(r["sdf"], r["rgb"], r["depth"])
    render_mesh(m["verts"], m["faces"], m["vert_rgb"],
                os.path.join(out, "render_textured_rgbcrit.png"), max_faces=_MF,
                title=f"SAND textured, joint RGB crit (thresh={args.rgb_thresh})")
    print("saved render_textured_rgbcrit.png", flush=True)
    render_mesh(m["verts"], m["faces"],
                depth_colormap(np.round(m["vert_depth"]), cfg.num_layers),
                os.path.join(out, "render_depth_rgbcrit.png"), max_faces=_MF,
                title=f"SAND depth, joint RGB crit (thresh={args.rgb_thresh})")
    print("saved render_depth_rgbcrit.png")


if __name__ == "__main__":
    main()
