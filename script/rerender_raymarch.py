"""Re-render existing runs with BOTH render modes — no retraining.

For every run directory and every kept network (base_geo, sand_geo,
sand_color), this script produces two renders of the same trained field:

- Marching Cubes mode (mc_<tag>.png): warm dense-grid eval (2nd of 2 runs)
  -> timed mesh extraction -> timed software rasterization. End-to-end time =
  grid eval + extraction + rasterization.
- Ray marching mode (rm_<tag>.png, plus rm_depth_<tag>.png for SAND):
  sphere tracing through the adaptive/full-depth field, 1 warmup + 1 timed
  run. SAND far (depth-0) leaves skip the network entirely (cell leap).

All timings land in report.json under "render_compare". The removed
baseline+texture model (base_color) and the ply-mesh Chamfer statistic are
scrubbed from report.json, and report.md is regenerated.

Usage: .venv/bin/python script/rerender_raymarch.py [--dirs D1 D2 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import Config
from src.octree import Octree
from src import infer, train as train_mod
from src.render import render_mesh, solid_colors
from src.raymarch import make_sand_field, make_baseline_field, render_raymarch
from run_pipeline import render_report_md

_MF = 2_000_000  # decimate MC meshes beyond this for rasterization; at 800x800
                 # output 2M triangles is already ~3x the pixel density, so the
                 # images are visually identical while the pure-numpy rasterizer
                 # stays tractable. raster_s below includes the decimation time.
NETWORKS = (("base_geo", "baseline", False, "baseline"),
            ("sand_geo", "sand", False, "SAND"),
            ("sand_color", "sand", True, "SAND+texture"))
SCRUB_KEYS = ("training", "eval", "meshes", "renders")


def pick_octree(out: str, tag: str) -> str:
    """Prefer the refined rgbcrit depth map when present (final renders did)."""
    if tag == "sand_color":
        rgbcrit = os.path.join(out, "octree_sand_color_rgbcrit.npz")
        if os.path.exists(rgbcrit):
            return rgbcrit
    return os.path.join(out, f"octree_{tag}.npz")


def run_marching_cubes(model, kind, octree, cfg, with_rgb, out, tag,
                       label) -> dict:
    """Grid eval (warm) + timed extraction + timed rasterization."""
    timing = None
    r = None
    for rep in range(2):  # warmup, then the timed run we keep
        if kind == "sand":
            r = infer.eval_grid_sand(model, octree, cfg.res, cfg.device,
                                     cfg.chunk, with_rgb)
        else:
            r = infer.eval_grid_baseline(model, cfg.res, cfg.device,
                                         cfg.chunk, with_rgb)
        timing = r["timing"]
        print(f"  [mc:{tag}] grid eval rep{rep}: {timing}", flush=True)

    t0 = time.perf_counter()
    m = infer.extract_mesh(r["sdf"], r["rgb"], None)
    extract_s = time.perf_counter() - t0
    print(f"  [mc:{tag}] mesh verts={len(m['verts'])} faces={len(m['faces'])} "
          f"extract={extract_s:.2f}s", flush=True)

    colors = m["vert_rgb"] if m["vert_rgb"] is not None \
        else solid_colors(len(m["verts"]))
    png = os.path.join(out, f"mc_{tag}.png")
    t0 = time.perf_counter()
    render_mesh(m["verts"], m["faces"], colors, png, max_faces=_MF,
                title=f"Marching Cubes - {label}")
    raster_s = time.perf_counter() - t0
    print(f"  [mc:{tag}] raster={raster_s:.2f}s -> {png}", flush=True)

    total = timing["total_s"] + extract_s + raster_s
    return {"res": int(cfg.res), "grid_eval": timing, "extract_s": extract_s,
            "raster_s": raster_s, "total_s": total,
            "n_verts": int(len(m["verts"])), "n_faces": int(len(m["faces"])),
            "image": os.path.basename(png)}


def run_raymarch(model, kind, octree, cfg, with_rgb, out, tag, label,
                 img_size: int) -> dict:
    """Sphere-traced render: 1 warmup + 1 timed run."""
    if kind == "sand":
        field = make_sand_field(model, octree, cfg.device, cfg.chunk, with_rgb)
    else:
        field = make_baseline_field(model, cfg.device, cfg.chunk, with_rgb)
    png = os.path.join(out, f"rm_{tag}.png")
    depth_png = os.path.join(out, f"rm_depth_{tag}.png") \
        if kind == "sand" else None
    stats = None
    for rep in range(2):  # warmup, then the timed run we keep
        stats = render_raymarch(field, png, img_size=img_size,
                                title=f"Ray marching - {label}",
                                depth_out_path=depth_png)
        print(f"  [rm:{tag}] rep{rep}: total={stats['total_s']:.2f}s "
              f"network={stats['network_s']:.2f}s "
              f"octree={stats['octree_query_s']:.2f}s "
              f"zero_net={stats['zero_network_frac']:.3f} "
              f"mean_depth={stats['mean_net_depth']:.2f}", flush=True)
    stats["image"] = os.path.basename(png)
    if kind == "sand":
        stats["octree_npz"] = os.path.basename(
            pick_octree(out, tag))
        stats["depth_image"] = os.path.basename(depth_png)
    return stats


def scrub_report(report: dict) -> None:
    """Drop the removed ply-mesh Chamfer statistic and base_color entries."""
    report.pop("chamfer_l1_x1e3", None)
    for key in SCRUB_KEYS:
        if key in report and isinstance(report[key], dict):
            report[key].pop("base_color", None)


def process_run(out: str, img_size: int) -> None:
    print(f"\n===== {out} =====", flush=True)
    cfg = Config.load(os.path.join(out, "config.json"))
    with open(os.path.join(out, "report.json")) as f:
        report = json.load(f)

    compare = {"img_size": int(img_size),
               "marching_cubes": {}, "raymarch": {}}
    for tag, kind, with_rgb, label in NETWORKS:
        ckpt = os.path.join(out, f"ckpt_{tag}.pt")
        if not os.path.exists(ckpt):
            print(f"  skip {tag}: no checkpoint", flush=True)
            continue
        print(f"--- {tag} ({label}) ---", flush=True)
        model = train_mod.load_model(kind, ckpt, cfg.device)
        octree = Octree.load(pick_octree(out, tag)) if kind == "sand" else None
        compare["marching_cubes"][tag] = run_marching_cubes(
            model, kind, octree, cfg, with_rgb, out, tag, label)
        compare["raymarch"][tag] = run_raymarch(
            model, kind, octree, cfg, with_rgb, out, tag, label, img_size)
        del model, octree
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report["render_compare"] = compare
    scrub_report(report)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(render_report_md(report))
    print(f"rewrote {out}/report.json + report.md", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dirs", nargs="+",
                    default=["report/suzanne", "report/robot_union2",
                             "report/r2"])
    ap.add_argument("--size", type=int, default=800)
    args = ap.parse_args()
    for d in args.dirs:
        process_run(d, args.size)


if __name__ == "__main__":
    main()
