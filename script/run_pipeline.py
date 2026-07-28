#!/usr/bin/env python
"""End-to-end SAND (arXiv:2604.25936) reproduction pipeline orchestrator.

Steps: load mesh -> build octree -> train 3 models (SAND geo/color + baseline
geo) -> assign per-leaf network depths -> grid evaluation -> mesh extraction
-> software raster renders -> ray-marching renders -> report.json/report.md.

All heavy lifting lives in the `sand` package; this file only orchestrates.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
import time

import numpy as np
import torch
import trimesh

from src.config import Config
from src.data import MeshData
from src.octree import Octree
from src import train as train_mod
from src import infer
from src.render import render_mesh, depth_colormap, solid_colors
from src.raymarch import make_sand_field, make_baseline_field, render_raymarch


# --------------------------------------------------------------------- utils
class Timer:
    """Context manager printing per-step wall time."""

    def __init__(self, label: str):
        self.label = label
        self.seconds = 0.0

    def __enter__(self):
        print(f"\n=== {self.label} ===", flush=True)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        print(f"[time] {self.label}: {self.seconds:.2f} s", flush=True)
        return False


# ---------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="assets/sample.glb", help="input .glb mesh")
    p.add_argument("--out", default="output", help="output directory")
    p.add_argument("--iters", type=int, default=None,
                   help=f"training iterations (default {d.iters})")
    p.add_argument("--res", type=int, default=None,
                   help=f"grid resolution for eval/extraction (default {d.res})")
    p.add_argument("--octree-depth", type=int, default=None,
                   help=f"octree max depth (default {d.octree_depth})")
    p.add_argument("--device", default=d.device, help="cuda or cpu")
    p.add_argument("--batch", type=int, default=None, help="training batch size")
    p.add_argument("--lr", type=float, default=None, help="learning rate")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--solid-union-res", type=int, default=None,
                   help="build a watertight solid-union proxy on an N^3 grid "
                        "before training (for non-watertight multi-part assets)")
    p.add_argument("--quick", action="store_true",
                   help="smoke run: iters=1500, res=96, octree-depth=6")
    return p.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.model = args.model
    cfg.out_dir = args.out
    cfg.device = args.device
    if args.quick:
        cfg.iters, cfg.res, cfg.octree_depth = 1500, 96, 6
    for field, val in (("iters", args.iters), ("res", args.res),
                       ("octree_depth", args.octree_depth), ("batch", args.batch),
                       ("lr", args.lr), ("seed", args.seed),
                       ("solid_union_res", args.solid_union_res)):
        if val is not None:
            setattr(cfg, field, val)
    return cfg


# ---------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    cfg.save(os.path.join(out, "config.json"))

    pipeline_t0 = time.perf_counter()
    report: dict = {"config": json.loads(json.dumps(cfg.__dict__))}

    # ---------------------------------------------------------------- 1. mesh
    with Timer("1. load mesh") as t:
        if args.quick and not os.path.exists(args.model):
            print(f"[warn] {args.model} missing (expected assets/sample.glb to exist)")
        mesh_data = MeshData.load(args.model, solid_union_res=cfg.solid_union_res)
        print(f"mesh: {args.model}  verts={len(mesh_data.mesh.vertices)} "
              f"faces={len(mesh_data.mesh.faces)} watertight={mesh_data.is_watertight()} "
              f"has_color={mesh_data.has_color}")
    report["mesh"] = {"path": args.model, "n_verts": int(len(mesh_data.mesh.vertices)),
                      "n_faces": int(len(mesh_data.mesh.faces)),
                      "watertight": mesh_data.is_watertight(),
                      "has_color": mesh_data.has_color, "load_s": t.seconds}

    # ---------------------------------------------------------------- 2. octree
    base_npz = os.path.join(out, "octree_base.npz")
    with Timer(f"2. octree build (max_depth={cfg.octree_depth})") as t:
        octree = Octree.build(mesh_data, cfg.octree_depth)
        octree.save(base_npz)
        base_stats = octree.stats()
        print(f"octree stats: {base_stats}")
        print(f"saved {base_npz}")
    report["octree"] = {"max_depth": cfg.octree_depth, "build_s": t.seconds,
                        "base_stats": base_stats, "per_model": {}}

    # ---------------------------------------------------------------- 3. train
    runs = [("sand_geo", "sand", False), ("sand_color", "sand", True),
            ("base_geo", "baseline", False)]
    metas: dict[str, dict] = {}
    with Timer("3. train 4 models"):
        for tag, kind, color in runs:
            if color and not mesh_data.has_color:
                print(f"[warn] mesh has no color/texture; skipping color run {tag}")
                continue
            print(f"\n--- training {tag} (kind={kind}, color={color}) ---", flush=True)
            metas[tag] = train_mod.train_model(
                kind=kind, color=color, mesh_data=mesh_data, cfg=cfg,
                device=cfg.device, out_dir=out, tag=tag,
                octree=octree if kind == "sand" else None)
            m = metas[tag]
            print(f"[{tag}] done: {m['train_time_s']:.2f} s "
                  f"({m['ms_per_iter']:.2f} ms/iter) final_loss={m['final_loss']:.6f} "
                  f"n_params={m['n_params']}")
    report["training"] = {tag: {**{k: m[k] for k in
                                ("kind", "color", "iters", "train_time_s",
                                 "ms_per_iter", "final_loss", "n_params", "ckpt")},
                                "data_prep_s": m.get("data_prep_s", 0.0)}
                          for tag, m in metas.items()}

    # ------------------------------------------- 4. assign depths per SAND model
    octrees: dict[str, Octree] = {}
    for tag in ("sand_geo", "sand_color"):
        if tag not in metas:
            continue
        with Timer(f"4. assign_depths ({tag})") as t:
            model = train_mod.load_model("sand", metas[tag]["ckpt"], cfg.device)
            oc = Octree.load(base_npz)  # fresh copy; assignments stay separate
            oc.assign_depths(model, cfg.device, cfg.err_thresh,
                             cfg.depth_samples_per_leaf, cfg.chunk)
            npz = os.path.join(out, f"octree_{tag}.npz")
            oc.save(npz)
            stats = oc.stats()
            print(f"[{tag}] octree stats: mean_depth_near="
                  f"{stats['mean_depth_near']:.3f} depth_hist={stats['depth_hist']}")
            print(f"saved {npz}")
            # Reload from npz for the matching eval (as required).
            octrees[tag] = Octree.load(npz)
            report["octree"]["per_model"][tag] = {**stats, "assign_s": t.seconds,
                                                  "npz": npz}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------- 5. grid evaluation
    evals: dict[str, dict] = {}
    grids: dict[str, dict] = {}
    with Timer(f"5. grid evaluation (res={cfg.res})"):
        for tag, kind, with_rgb in (("sand_geo", "sand", False),
                                    ("sand_color", "sand", True),
                                    ("base_geo", "baseline", False)):
            if tag not in metas:
                continue
            model = train_mod.load_model(kind, metas[tag]["ckpt"], cfg.device)
            if kind == "sand":
                r = infer.eval_grid_sand(model, octrees[tag], cfg.res,
                                         cfg.device, cfg.chunk, with_rgb)
            else:
                r = infer.eval_grid_baseline(model, cfg.res, cfg.device,
                                             cfg.chunk, with_rgb)
            grids[tag] = r
            evals[tag] = r["timing"]
            print(f"[eval:{tag}] {r['timing']}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report["eval"] = evals

    # ---------------------------------------------------- 6. mesh extraction
    meshes: dict[str, dict] = {}
    with Timer("6. marching-cubes mesh extraction"):
        for tag in list(grids):
            g = grids[tag]
            depth = g["depth"] if tag.startswith("sand") else None
            meshes[tag] = infer.extract_mesh(g["sdf"], g["rgb"], depth)
            print(f"[mesh:{tag}] verts={len(meshes[tag]['verts'])} "
                  f"faces={len(meshes[tag]['faces'])}")
            # persist as PLY for external inspection (Meshlab/Blender)
            import trimesh as _tm
            vc = (meshes[tag]["vert_rgb"] * 255).astype(np.uint8) \
                if meshes[tag]["vert_rgb"] is not None else None
            ply = os.path.join(out, f"mesh_{tag}.ply")
            _tm.Trimesh(meshes[tag]["verts"], meshes[tag]["faces"],
                        vertex_colors=vc, process=False).export(ply)
    report["meshes"] = {tag: {"n_verts": int(len(m["verts"])),
                              "n_faces": int(len(m["faces"]))}
                        for tag, m in meshes.items()}

    # ------------------------------------------------------------ 7. renders
    renders: dict[str, str] = {}
    hero = "sand_color" if "sand_color" in meshes else "sand_geo"
    base = "base_geo"
    # render at full mesh resolution (no decimation) for maximum sharpness
    _MF = 5_000_000
    with Timer("7. software renders (800x800)"):
        m = meshes[hero]
        path = os.path.join(out, "render_solid.png")
        render_mesh(m["verts"], m["faces"], solid_colors(len(m["verts"])), path,
                    max_faces=_MF, title=f"SAND solid ({hero})")
        renders["render_solid"] = path
        if m["vert_depth"] is not None:
            path = os.path.join(out, "render_depth.png")
            render_mesh(m["verts"], m["faces"],
                        depth_colormap(np.round(m["vert_depth"]), cfg.num_layers),
                        path, max_faces=_MF, title=f"SAND network depth ({hero})")
            renders["render_depth"] = path
        if m["vert_rgb"] is not None:
            path = os.path.join(out, "render_textured.png")
            render_mesh(m["verts"], m["faces"], m["vert_rgb"], path,
                        max_faces=_MF, title=f"SAND textured ({hero})")
            renders["render_textured"] = path
        mb = meshes[base]
        path = os.path.join(out, "render_baseline_textured.png")
        colors = mb["vert_rgb"] if mb["vert_rgb"] is not None \
            else solid_colors(len(mb["verts"]))
        render_mesh(mb["verts"], mb["faces"], colors, path,
                    max_faces=_MF, title=f"Baseline ({base})")
        renders["render_baseline_textured"] = path
        for name, p in renders.items():
            print(f"saved {name}: {p}")
    report["renders"] = renders

    # ------------------------------------------------ 8. ray-marching renders
    raymarch_stats: dict[str, dict] = {}
    with Timer("8. ray-marching renders (800x800)"):
        for tag, kind, with_rgb in (("base_geo", "baseline", False),
                                    ("sand_geo", "sand", False),
                                    ("sand_color", "sand", True)):
            if tag not in metas:
                continue
            model = train_mod.load_model(kind, metas[tag]["ckpt"], cfg.device)
            if kind == "sand":
                field = make_sand_field(model, octrees[tag], cfg.device,
                                        cfg.chunk, with_rgb)
            else:
                field = make_baseline_field(model, cfg.device, cfg.chunk,
                                            with_rgb)
            path = os.path.join(out, f"rm_{tag}.png")
            depth_path = os.path.join(out, f"rm_depth_{tag}.png") \
                if kind == "sand" else None
            stats = render_raymarch(field, path, title=f"ray marching ({tag})",
                                    depth_out_path=depth_path)
            raymarch_stats[tag] = stats
            print(f"[raymarch:{tag}] {stats}")
            renders[f"rm_{tag}"] = path
            del model, field
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report["raymarch"] = raymarch_stats

    # ------------------------------------------------------------ 9. reports
    report["hardware"] = {
        "device": cfg.device,
        "gpu": (torch.cuda.get_device_name(0)
                if cfg.device.startswith("cuda") and torch.cuda.is_available()
                else None),
        "torch": torch.__version__,
    }
    report["total_time_s"] = time.perf_counter() - pipeline_t0

    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    md = render_report_md(report)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(md)
    print(f"\nwrote {os.path.join(out, 'report.json')} and report.md")
    print(f"[time] total pipeline: {report['total_time_s']:.2f} s")


def render_report_md(report: dict) -> str:
    cfg = report["config"]
    metas = report["training"]
    L = []
    L.append("# SAND 复现流水线报告\n")
    L.append(f"- 输入模型: `{report['mesh']['path']}` "
             f"(顶点 {report['mesh']['n_verts']}, 面 {report['mesh']['n_faces']}, "
             f"水密: {report['mesh']['watertight']}, 有纹理: {report['mesh']['has_color']})")
    hw = report["hardware"]
    L.append(f"- 硬件: {hw['gpu'] or hw['device']}  |  torch {hw['torch']}")
    L.append(f"- 备注: 论文使用 100k 迭代 / 512^3 网格; 本运行在 4GB GPU 上采用 "
             f"{cfg['iters']} 迭代 / {cfg['res']}^3 网格 (缩小规模)。\n")

    # (a) training table
    L.append("## (a) 训练时长对比\n")
    L.append("| 模型 | 参数量 | 迭代数 | 训练时长 (s) | 数据准备 (s) | ms/iter | 最终 loss |")
    L.append("|---|---|---|---|---|---|---|")
    names = {"sand_geo": "SAND 无纹理", "sand_color": "SAND 有纹理",
             "base_geo": "基线 无纹理"}
    for tag in ("sand_geo", "sand_color", "base_geo"):
        if tag not in metas:
            continue
        m = metas[tag]
        L.append(f"| {names[tag]} | {m['n_params']} | {m['iters']} "
                 f"| {m['train_time_s']:.2f} | {m.get('data_prep_s', 0.0):.1f} "
                 f"| {m['ms_per_iter']:.2f} "
                 f"| {m['final_loss']:.6f} |")
    L.append("")

    # (b) query/render timing table
    L.append("## (b) 渲染/查询时长对比 (网格分辨率 "
             f"{cfg['res']}^3)\n")
    L.append("| 场景 | SAND octree 查询 (s) | SAND 网络 (s) | SAND 总计 (s) "
             "| 基线总计 (s) | 加速比 | 平均深度 | depth0 占比 |")
    L.append("|---|---|---|---|---|---|---|---|")
    ev = report["eval"]
    for sand_tag, base_tag, label in (("sand_geo", "base_geo", "无纹理"),
                                      ("sand_color", "base_color", "有纹理")):
        if sand_tag not in ev or base_tag not in ev:
            continue
        s, b = ev[sand_tag], ev[base_tag]
        speedup = b["total_s"] / s["total_s"] if s["total_s"] > 0 else float("nan")
        L.append(f"| {label} | {s['octree_query_s']:.3f} | {s['network_s']:.3f} "
                 f"| {s['total_s']:.3f} | {b['total_s']:.3f} | {speedup:.2f}x "
                 f"| {s['mean_depth']:.2f} | {s['frac_depth0']:.3f} |")
    L.append("")

    # (c) octree stats
    oc = report["octree"]
    bs = oc["base_stats"]
    L.append("## (c) Octree 统计\n")
    L.append(f"- 最大深度: {oc['max_depth']}, 构建耗时: {oc['build_s']:.2f} s, "
             f"叶节点总数: {bs['n_leaves']} (近表面 {bs['n_near']} / 远场 {bs['n_far']})")
    L.append("")
    L.append("| 模型 | 近表面平均深度 | 深度分布 {深度: 叶子数} |")
    L.append("|---|---|---|")
    for tag, st in oc["per_model"].items():
        hist = ", ".join(f"{d}: {c}" for d, c in sorted(st["depth_hist"].items(),
                                                        key=lambda kv: int(kv[0])))
        L.append(f"| {tag} | {st['mean_depth_near']:.3f} | {hist} |")
    L.append("")

    # (d) ray-marching render timings
    rm = report.get("raymarch", {})
    if rm:
        L.append("## (d) Ray Marching 渲染时长 (800x800)\n")
        L.append("| 模型 | 总时长 (s) | 光线数 | 命中比例 | 场查询数 | "
                 "零网络查询占比 | 平均网络深度 | 网络 (s) | octree (s) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for tag in ("base_geo", "sand_geo", "sand_color"):
            if tag not in rm:
                continue
            s = rm[tag]
            L.append(f"| {names[tag]} | {s['total_s']:.2f} | {s['n_rays']} "
                     f"| {s['hit_frac']:.3f} | {s['march_queries']} "
                     f"| {s['zero_network_frac']:.3f} | {s['mean_net_depth']:.2f} "
                     f"| {s['network_s']:.3f} | {s['octree_query_s']:.3f} |")
        L.append("")

    # (e) config dump
    L.append("## (e) 配置\n")
    L.append("```json")
    L.append(json.dumps(cfg, indent=2, ensure_ascii=False))
    L.append("```")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
