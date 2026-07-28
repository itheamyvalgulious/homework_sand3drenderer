"""Grid evaluation (adaptive SAND vs full-depth baseline) and mesh extraction.

CONTRACT:

def eval_grid_sand(model, octree, res: int, device: str, chunk: int,
                   with_rgb: bool) -> dict
    # Dense res^3 grid over [0,1]^3 (cell centers). For every point: octree.query ->
    # net_depth; depth-0 points take stored_sdf (rgb = neutral gray 0.5 if with_rgb);
    # the rest are evaluated with model.forward_adaptive in chunks of `chunk`.
    # Returns {"sdf": (res,res,res) float32,
    #          "depth": (res,res,res) uint8,
    #          "rgb": (res,res,res,3) float32 in [0,1] or None,
    #          "timing": {"octree_query_s", "network_s", "total_s",
    #                     "mean_depth", "frac_depth0", "n_points"}}
    # Network outputs: channel 0 = SDF, channels 1:4 = RGB (clamped to [0,1];
    # RGB is L1-trained against [0,1] targets on raw outputs, so no sigmoid).

def eval_grid_baseline(model, res: int, device: str, chunk: int,
                       with_rgb: bool) -> dict
    # Same grid, full-depth model.forward for ALL points.
    # Returns {"sdf", "rgb"|None, "timing": {"network_s", "total_s", "n_points"}}
    # ("depth" is implicitly num_layers everywhere; include a "depth" array filled
    #  with model depth for API symmetry.)

def extract_mesh(sdf_grid: np.ndarray, rgb_grid=None, depth_grid=None) -> dict
    # skimage.measure.marching_cubes at level 0.0, verts remapped to [0,1]^3.
    # Per-vertex rgb / depth trilinear-interpolated from the grids when given.
    # Returns {"verts": (V,3) f32, "faces": (F,3) int64,
    #          "vert_rgb": (V,3) f32|None, "vert_depth": (V,) f32|None}

Timing values are wall-clock seconds (time.perf_counter with cuda sync).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from skimage.measure import marching_cubes


def _grid_points(res: int) -> np.ndarray:
    """(res^3, 3) float32 cell-center coordinates of the dense [0,1]^3 grid.

    Flattened in 'ij' (C) order, so index = (i * res + j) * res + k.
    """
    axis = np.linspace(0.5 / res, 1.0 - 0.5 / res, res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def _cuda_sync(device: torch.device) -> None:
    """Synchronize before/after timed GPU sections (no-op on CPU)."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def eval_grid_sand(model, octree, res: int, device: str, chunk: int,
                   with_rgb: bool) -> dict:
    """Adaptive-depth grid evaluation driven by the octree depth map."""
    t_total0 = time.perf_counter()
    dev = torch.device(device)
    pts = _grid_points(res)
    n_points = pts.shape[0]
    chunk = max(1, int(chunk))

    # 1) Octree queries for every grid point (batched to bound memory).
    depths = np.empty(n_points, dtype=np.uint8)
    stored = np.empty(n_points, dtype=np.float32)
    octree_query_s = 0.0
    for s in range(0, n_points, chunk):
        t0 = time.perf_counter()
        d, sd = octree.query(pts[s:s + chunk])
        octree_query_s += time.perf_counter() - t0
        depths[s:s + chunk] = d
        stored[s:s + chunk] = sd

    # 2) Depth-0 (far) points take the stored SDF; the rest go through the network.
    sdf = np.empty(n_points, dtype=np.float32)
    far = depths == 0
    sdf[far] = stored[far]

    eval_idx = np.nonzero(~far)[0]
    outs = []
    _cuda_sync(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, eval_idx.size, chunk):
            idx = eval_idx[s:s + chunk]
            x = torch.from_numpy(pts[idx]).to(dev)
            d = torch.from_numpy(depths[idx].astype(np.int64)).to(dev)
            y = model.forward_adaptive(x, d)
            outs.append(y.detach().to("cpu", torch.float32).numpy())
    _cuda_sync(dev)
    network_s = time.perf_counter() - t0

    if outs:
        out = np.concatenate(outs, axis=0)
        sdf[eval_idx] = out[:, 0]
        has_rgb = bool(with_rgb) and out.shape[1] == 4
    else:  # every point is far; fall back to the declared output dim
        has_rgb = bool(with_rgb) and getattr(model, "out_dim", 1) == 4

    rgb = None
    if has_rgb:
        rgb = np.full((n_points, 3), 0.5, dtype=np.float32)  # far points: gray
        if outs:
            # RGB is trained against [0,1] targets with L1 on raw outputs,
            # so the eval-side transform is a plain clamp (NOT sigmoid).
            rgb[eval_idx] = np.clip(out[:, 1:4], 0.0, 1.0)

    depth_grid = depths.reshape(res, res, res)
    timing = {
        "octree_query_s": float(octree_query_s),
        "network_s": float(network_s),
        "total_s": float(time.perf_counter() - t_total0),
        "mean_depth": float(depth_grid.mean()),
        "frac_depth0": float(far.mean()),
        "n_points": int(n_points),
    }
    return {
        "sdf": sdf.reshape(res, res, res),
        "depth": depth_grid,
        "rgb": rgb.reshape(res, res, res, 3) if rgb is not None else None,
        "timing": timing,
    }


def eval_grid_baseline(model, res: int, device: str, chunk: int,
                       with_rgb: bool) -> dict:
    """Full-depth evaluation of every grid point (no octree, no early exit)."""
    t_total0 = time.perf_counter()
    dev = torch.device(device)
    pts = _grid_points(res)
    n_points = pts.shape[0]
    chunk = max(1, int(chunk))

    # TMLP exposes forward_final; BaselineMLP exposes forward.
    forward = getattr(model, "forward_final", None) or model.forward

    outs = []
    _cuda_sync(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, n_points, chunk):
            x = torch.from_numpy(pts[s:s + chunk]).to(dev)
            y = forward(x)
            outs.append(y.detach().to("cpu", torch.float32).numpy())
    _cuda_sync(dev)
    network_s = time.perf_counter() - t0

    out = np.concatenate(outs, axis=0)
    sdf = out[:, 0].astype(np.float32)
    rgb = None
    if with_rgb and out.shape[1] == 4:
        # Clamp, not sigmoid: RGB is L1-trained against [0,1] targets on raw outputs.
        rgb = np.clip(out[:, 1:4], 0.0, 1.0).astype(np.float32)

    num_layers = getattr(model, "num_layers", None)
    if num_layers is None:
        raise ValueError("eval_grid_baseline: model must expose a 'num_layers' attribute")
    depth = np.full((res, res, res), int(num_layers), dtype=np.uint8)

    return {
        "sdf": sdf.reshape(res, res, res),
        "depth": depth,
        "rgb": rgb.reshape(res, res, res, 3) if rgb is not None else None,
        "timing": {
            "network_s": float(network_s),
            "total_s": float(time.perf_counter() - t_total0),
            "n_points": int(n_points),
        },
    }


def _trilinear(grid: np.ndarray, pts: np.ndarray, res: int) -> np.ndarray:
    """Trilinearly sample `grid` at world-space points.

    grid: (res,res,res) or (res,res,res,C); pts: (V,3) in [0,1]^3 where grid
    node (i,j,k) sits at ((i,j,k) + 0.5) / res. Returns (V,) or (V,C) float32.
    """
    g = np.clip(pts.astype(np.float64) * res - 0.5, 0.0, res - 1.0)
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    i1 = np.minimum(i0 + 1, res - 1)
    xs = ((i0[:, 0], 1.0 - f[:, 0]), (i1[:, 0], f[:, 0]))
    ys = ((i0[:, 1], 1.0 - f[:, 1]), (i1[:, 1], f[:, 1]))
    zs = ((i0[:, 2], 1.0 - f[:, 2]), (i1[:, 2], f[:, 2]))
    acc = 0.0
    for ix, wx in xs:
        for iy, wy in ys:
            for iz, wz in zs:
                w = wx * wy * wz
                if grid.ndim == 4:
                    w = w[:, None]
                acc = acc + grid[ix, iy, iz] * w
    return np.asarray(acc, dtype=np.float32)


def extract_mesh(sdf_grid: np.ndarray, rgb_grid=None, depth_grid=None) -> dict:
    """Marching Cubes at the zero level set, verts remapped to [0,1]^3."""
    sdf_grid = np.asarray(sdf_grid, dtype=np.float32)
    if sdf_grid.ndim != 3 or len(set(sdf_grid.shape)) != 1:
        raise ValueError(
            f"extract_mesh: sdf_grid must be cubic (res,res,res), got {sdf_grid.shape}")
    res = sdf_grid.shape[0]
    lo, hi = float(sdf_grid.min()), float(sdf_grid.max())
    if not (lo < 0.0 < hi):
        raise ValueError(
            f"extract_mesh: SDF grid has no zero level set (min={lo:.6g}, "
            f"max={hi:.6g}); cannot extract a mesh")

    # Node (i,j,k) is at world ((i,j,k)+0.5)/res; marching_cubes with this
    # spacing returns index/res, so shift by half a cell to land in [0,1]^3.
    verts, faces, _, _ = marching_cubes(sdf_grid, level=0.0, spacing=(1.0 / res,) * 3)
    verts = (verts + 0.5 / res).astype(np.float32)
    faces = faces.astype(np.int64)

    vert_rgb = None
    if rgb_grid is not None:
        rgb_grid = np.asarray(rgb_grid, dtype=np.float32)
        if rgb_grid.shape[:3] != (res, res, res):
            raise ValueError(f"extract_mesh: rgb_grid shape {rgb_grid.shape} "
                             f"does not match sdf res {res}")
        vert_rgb = _trilinear(rgb_grid, verts, res)

    vert_depth = None
    if depth_grid is not None:
        depth_grid = np.asarray(depth_grid, dtype=np.float32)
        if depth_grid.shape != (res, res, res):
            raise ValueError(f"extract_mesh: depth_grid shape {depth_grid.shape} "
                             f"does not match sdf res {res}")
        vert_depth = _trilinear(depth_grid, verts, res)

    return {"verts": verts, "faces": faces,
            "vert_rgb": vert_rgb, "vert_depth": vert_depth}
