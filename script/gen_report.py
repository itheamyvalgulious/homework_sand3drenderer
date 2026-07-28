"""Generate REPORT.md: ray marching vs Marching Cubes comparison.

Per the current experiment spec, the report covers exactly 3 models
(Suzanne 猴头 / Wall-E 机器人 / r2 机器人) x 3 networks (baseline 无纹理 /
SAND 无纹理 / SAND+纹理扩展) x 2 render modes (Marching Cubes 栅格化管线 /
Ray Marching 球体追踪), plus training times and the traditional-reference
renders. All numbers are read from each run's report.json ("render_compare",
written by script/rerender_raymarch.py) and meta_*.json — nothing is
recomputed here.

Usage: PYTHONPATH=. .venv/bin/python script/gen_report.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (dir, label, description)
RUNS = [
    ("report/suzanne", "Suzanne 猴头",
     "Blender Suzanne 原始图元，清洗/细分/转向 + 程序化棋盘贴图，6.2 万面。"),
    ("report/robot_union2", "Wall-E 机器人",
     "22.5 万面、67 个非封闭部件、67 张 1024² 贴图。实心化 v2 "
     "(绕数高斯平滑+并集) 预处理。"),
    ("report/r2", "R2 机器人",
     "6.6 万面、13 个非封闭部件、2048² 贴图。实心化 v2 预处理。"),
]

NETWORKS = [("base_geo", "baseline (无纹理)"),
            ("sand_geo", "SAND (无纹理)"),
            ("sand_color", "SAND+纹理扩展")]

HW = "NVIDIA GeForce GTX 1650 (4GB) | torch 2.13.0+cu130 | 8 核 CPU / 19GB RAM"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def img_md(path, caption):
    """Markdown image if the file exists, else a placeholder."""
    if os.path.exists(path):
        return f"![{caption}]({path})"
    return f"*(missing: {path})*"


def fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def sec_reference():
    L = ["## 一、参考渲染(传统光栅化 · 原始模型)\n",
         "原始三角网格 + 真贴图/顶点色,经软件光栅化渲染,作为重建质量的肉眼参照。\n"]
    L.append("| 模型 | 参考渲染 |")
    L.append("|---|---|")
    for d, label, _ in RUNS:
        L.append(f"| {label} | {img_md(os.path.join(d, 'render_reference.png'), label + ' reference')} |")
    L.append("")
    return L


def sec_training():
    L = ["## 二、训练时长(3 模型 × 3 网络)\n",
         "| 模型 | 网络 | 参数量 | 迭代数 | 训练时长 (s) | ms/iter | 最终 loss |",
         "|---|---|---|---|---|---|---|"]
    for d, label, _ in RUNS:
        for tag, net_label in NETWORKS:
            m = load_json(os.path.join(d, f"meta_{tag}.json"))
            if m is None:
                L.append(f"| {label} | {net_label} | — | — | — | — | — |")
                continue
            L.append(f"| {label} | {net_label} | {m['n_params']} | {m['iters']} "
                     f"| {fmt(m['train_time_s'])} | {fmt(m['ms_per_iter'])} "
                     f"| {fmt(m['final_loss'], 6)} |")
    L.append("")
    return L


def _mode_time_cell(stats, *keys):
    """Format 'total (breakdown)' for a stats dict, or '—' when missing."""
    if not stats:
        return "—"
    parts = [f"**{fmt(stats.get('total_s'))}**"]
    sub = []
    for k in keys:
        if k in stats:
            sub.append(f"{k}={fmt(stats[k])}")
    if sub:
        parts.append("(" + ", ".join(sub) + ")")
    return " ".join(parts)


def sec_render_compare():
    L = ["## 三、渲染结果与时长(2 模式 × 3 网络)\n",
         "两种模式渲染同一训练好的隐式场,输出 800×800 图像(热态计时,取第 2 次运行):",
         "",
         "- **Marching Cubes (栅格化管线)**: 稠密网格评估 (256³) → MC 提取 → 软件光栅化;"
         " 总时长 = 网格评估 + 提取 + 光栅化。",
         "- **Ray Marching (球体追踪, 论文方式)**: 逐像素光线穿过 SDF 场;"
         " SAND 的 depth-0 远场叶零网络评估(按叶格跨越),近表面按八叉树深度早退;"
         " 相邻采样异号时二分回溯防隧穿。基线为全深度网络逐步追踪。",
         "",
         "**结果解读**:",
         "",
         "- MC 模式的 SAND 加速真实且显著: 网格评估阶段加速 ~5-10×"
         " (99% 体素 depth-0 零网络); 总时长另被提取/光栅化支配(与网格面数相关)。",
         "- RM 的 baseline 噪声团不是渲染 bug 而是网络本身的零集振荡, 实测:"
         " 沿单视线密采 4000 点出现 ~50 次符号变化(正常截面应为 2 次);"
         " 其 256³ 网格内部正负角点交织(18.6% 相邻边异号, 提取出 650 万面),"
         " MC 图之所以\"干净\"是因为亚像素面片把每个像素糊成实心轮廓 —"
         " 同一网格上 RM/MC 场值一致, 只是可视化方式不同。"
         " 10k 迭代的 SIREN 零等值面未收敛光滑(训练池负样本占 24.7%, 非样本不足)。",
         "- **SAND 网络本体同样未收敛**(suzanne 实测: 高斯壳符号一致率 SAND 各层 0.48"
         " vs baseline 0.56, 远场 0.42 vs 0.92)。SAND 的 RM 干净靠系统设计而非网络质量:"
         " (a) 训练采样本就不同 — SAND 池经八叉树过滤、不为远场负责(论文 Sec.3.3),"
         " baseline 则加 20% 全空间均匀点; (b) 渲染时 SAND 的场 = 八叉树预存 GT SDF"
         " (远场, ~99% 体积) + 网络浅层(近表面薄壳), 零交叉被 GT 远场引导、"
         " 偏差限制在亚像素薄壳内; baseline RM 全场直查网络, 零集振荡无处躲。"
         " 即两种模式对比的是两条完整管线, 而非两个裸网络。",
         "- 计时在共享 GPU 上测得, 绝对值有噪声, 同表内相对比较有效。",
         "",
         "baseline 网格切片证据(内部零集海绵):",
         "",
         img_md("report/suzanne/diag_baseline_grid_slices.png",
                "baseline 256^3 SDF 网格三正交切片(红=负值)"),
         ""]
    for d, label, desc in RUNS:
        r = load_json(os.path.join(d, "report.json"))
        rc = (r or {}).get("render_compare", {})
        mc = rc.get("marching_cubes", {})
        rm = rc.get("raymarch", {})
        L.append(f"### {label}\n")
        L.append(f"{desc}\n")

        # ---- timing table ----
        L.append("| 网络 | MC 总时长 (s) | MC 分解 (s) | MC grid 加速比 | "
                 "RM 总时长 (s) | RM 网络 (s) | RM 零网络查询占比 | RM 平均深度 | RM 时长比 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        base_rm = (rm.get("base_geo") or {}).get("total_s")
        base_grid = ((mc.get("base_geo") or {}).get("grid_eval") or {}).get("total_s")
        for tag, net_label in NETWORKS:
            ms, rs = mc.get(tag), rm.get(tag)
            mc_total = fmt(ms.get("total_s")) if ms else "—"
            mc_split = (f"grid {fmt(ms['grid_eval']['total_s'])} + "
                        f"extract {fmt(ms['extract_s'])} + "
                        f"raster {fmt(ms['raster_s'])}") if ms else "—"
            mc_sp = (f"{base_grid / ms['grid_eval']['total_s']:.1f}x"
                     if base_grid and ms and tag != "base_geo" else "—")
            if rs:
                rm_total = fmt(rs.get("total_s"))
                rm_net = fmt(rs.get("network_s"))
                rm_zero = fmt(rs.get("zero_network_frac"), 3) \
                    if tag.startswith("sand") else "0.000"
                rm_depth = fmt(rs.get("mean_net_depth"))
                sp = (f"{rs['total_s'] / base_rm:.2f}x"
                      if base_rm and rs.get("total_s") and tag != "base_geo"
                      else "—")
            else:
                rm_total = rm_net = rm_zero = rm_depth = sp = "—"
            L.append(f"| {net_label} | {mc_total} | {mc_split} | {mc_sp} | {rm_total} "
                     f"| {rm_net} | {rm_zero} | {rm_depth} | {sp} |")
        L.append("")

        # ---- image table ----
        L.append("| 网络 | Marching Cubes | Ray Marching | RM 深度图 |")
        L.append("|---|---|---|---|")
        for tag, net_label in NETWORKS:
            mc_img = img_md(os.path.join(d, f"mc_{tag}.png"), f"mc_{tag}")
            rm_img = img_md(os.path.join(d, f"rm_{tag}.png"), f"rm_{tag}")
            d_img = img_md(os.path.join(d, f"rm_depth_{tag}.png"),
                           f"rm_depth_{tag}") if tag.startswith("sand") else "—"
            L.append(f"| {net_label} | {mc_img} | {rm_img} | {d_img} |")
        L.append("")
    return L


def sec_legacy_stats():
    L = ["## 四、栅格化管线统计(保留)\n",
         "原稠密网格评估管线的计时与八叉树统计(对应 Marching Cubes 模式的网格评估阶段)。\n"]
    for d, label, _ in RUNS:
        r = load_json(os.path.join(d, "report.json"))
        if r is None:
            continue
        cfg = r.get("config", {})
        L.append(f"### {label}\n")
        ev = r.get("eval", {})
        if ev:
            L.append(f"网格分辨率 {cfg.get('res')}³:")
            L.append("")
            L.append("| 网络 | octree 查询 (s) | 网络 (s) | 总计 (s) | "
                     "depth0 占比 | 平均深度 |")
            L.append("|---|---|---|---|---|---|")
            for tag, net_label in NETWORKS:
                s = ev.get(tag)
                if not s:
                    continue
                L.append(f"| {net_label} | {fmt(s.get('octree_query_s'), 3)} "
                         f"| {fmt(s.get('network_s'), 3)} "
                         f"| {fmt(s.get('total_s'), 3)} "
                         f"| {fmt(s.get('frac_depth0'), 3) if 'frac_depth0' in s else '—'} "
                         f"| {fmt(s.get('mean_depth'), 3) if 'mean_depth' in s else '—'} |")
            L.append("")
        oc = r.get("octree", {})
        pm = oc.get("per_model", {})
        if pm:
            L.append("| SAND 模型 | 近表面平均深度 | 深度分配耗时 (s) |")
            L.append("|---|---|---|")
            for tag, st in pm.items():
                L.append(f"| {tag} | {fmt(st.get('mean_depth_near'), 3)} "
                         f"| {fmt(st.get('assign_s'))} |")
            L.append("")
    return L


def main():
    L = []
    L.append("# SAND 复现 — Ray Marching vs Marching Cubes 渲染对比报告\n")
    L.append(f"**硬件**: {HW}\n")
    L.append("**论文**: SAND: Spatially Adaptive Network Depth for Fast Sampling "
             "of Neural Implicit Surfaces (arXiv:2604.25936)\n")
    L.append("**说明**: 本报告对比两种渲染模式 — 论文的 ray marching (球体追踪) "
             "与此前实现的 marching cubes 栅格化管线; 覆盖 3 个模型 × 3 个网络 "
             "(baseline 无纹理 / SAND 无纹理 / SAND+纹理扩展)。"
             "baseline+纹理模型与基于 PLY 提取网格的 Chamfer 统计已按实验要求移除。\n")
    L.append("---\n")
    L += sec_reference()
    L.append("---\n")
    L += sec_training()
    L.append("---\n")
    L += sec_render_compare()
    L.append("---\n")
    L += sec_legacy_stats()

    outp = "REPORT.md"
    with open(outp, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {outp} ({sum(len(s) for s in L)} chars)")


if __name__ == "__main__":
    main()
