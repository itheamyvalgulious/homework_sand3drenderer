"""Re-measure grid-eval timings (warm, steady-state) and regenerate the report.

The in-pipeline eval timings can be skewed by first-call warmup (CUDA context,
octree lookup caches). This script reloads checkpoints + octrees, runs each
grid eval twice, keeps the second (warm) timing, injects data_prep_s from the
training meta files, and rewrites outputs/report.json + report.md.

Usage: .venv/bin/python scripts/remeasure_eval.py [--out outputs]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import Config
from src.octree import Octree
from src import infer, train as train_mod
from run_pipeline import render_report_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = args.out

    cfg = Config.load(os.path.join(out, "config.json"))
    with open(os.path.join(out, "report.json")) as f:
        report = json.load(f)

    combos = (("sand_geo", "sand", False), ("sand_color", "sand", True),
              ("base_geo", "baseline", False), ("base_color", "baseline", True))
    for tag, kind, with_rgb in combos:
        ckpt = os.path.join(out, f"ckpt_{tag}.pt")
        if not os.path.exists(ckpt):
            print(f"skip {tag}: no checkpoint")
            continue
        model = train_mod.load_model(kind, ckpt, cfg.device)
        octree = None
        if kind == "sand":
            octree = Octree.load(os.path.join(out, f"octree_{tag}.npz"))
        # warmup (result discarded), then the timed run we keep
        timing = None
        for rep in range(2):
            if kind == "sand":
                r = infer.eval_grid_sand(model, octree, cfg.res, cfg.device,
                                         cfg.chunk, with_rgb)
            else:
                r = infer.eval_grid_baseline(model, cfg.res, cfg.device,
                                             cfg.chunk, with_rgb)
            timing = r["timing"]
            print(f"[{tag}] rep{rep}: {timing}", flush=True)
            del r
        report["eval"][tag] = json.loads(json.dumps(timing))
        del model, octree
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # inject per-run data-prep time from the training meta files
    for tag, _, _ in combos:
        meta_path = os.path.join(out, f"meta_{tag}.json")
        if os.path.exists(meta_path) and tag in report.get("training", {}):
            with open(meta_path) as f:
                meta = json.load(f)
            report["training"][tag]["data_prep_s"] = meta.get("data_prep_s", 0.0)

    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(render_report_md(report))
    print("rewrote report.json / report.md")


if __name__ == "__main__":
    main()
