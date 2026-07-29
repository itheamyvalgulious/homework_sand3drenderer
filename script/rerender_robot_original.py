"""Re-render report/robot_original with the browser visualizer's parameters.

Uses script.viz_render.FieldCache.render with its default arguments (the exact
values the web UI sends on a fresh page load): img_size=384, elev=25, azim=45,
ambient=0.15, point light at (2,2,2), white, intensity=1.0. This is the same
single-sided Lambert + stable-normal path users see in the browser, applied to
the offline run directory so the result is directly comparable to
report/robot_union2/rm_*.png.

Usage: .venv/bin/python script/rerender_robot_original.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.viz_render import FieldCache

ROOT = "report/robot_original"


def main() -> None:
    cache = FieldCache(device="cuda", roots=[ROOT])
    models = cache.list_models()
    if not models:
        print(f"no checkpoints under {ROOT}", file=sys.stderr)
        sys.exit(1)
    print(f"found {len(models)} field(s) under {ROOT}:")
    for m in models:
        print(f"  {m['tag']}  ({m['label']})")

    for m in models:
        key = m["tag"]
        short = key.rsplit(":", 1)[1]
        out_png = os.path.join(ROOT, f"rm_{short}.png")
        r = cache.render(key)  # all defaults == browser defaults
        with open(out_png, "wb") as f:
            f.write(r["png"])
        s = r["stats"]
        print(f"  {short}: {s['n_hit']}/{s['n_rays']} hits "
              f"({s['hit_frac']*100:.1f}%) in {s['total_s']:.2f}s -> {out_png}",
              flush=True)


if __name__ == "__main__":
    main()
