"""Ray-marching (sphere tracing) renderer for the neural SDF fields.

This is the paper's rendering model: one ray per pixel, marched through the
SDF field. Two field backends are provided:

- ``SandField``: the SAND adaptive field. Every query first asks the octree
  for the point's network depth. Depth-0 (far) leaves skip the network
  entirely; the ray instead leaps to the leaf boundary (the leaf provably
  contains no surface, see Octree.build), so far-region traversal costs one
  octree lookup per cell crossing and zero network evaluations. Near-leaf
  points evaluate ``model.forward_adaptive`` at their assigned depth.
- ``BaselineField``: plain sphere tracing with the full-depth network at
  every query (no octree, no early exit).

Both share the same camera geometry as ``render.render_mesh`` (perspective
orbit around (0.5,0.5,0.5) at radius 2.2, y-up) so ray-marched and rasterized
images are directly comparable.

CONTRACT:

class SandField(model, octree, device, chunk=262144, with_rgb=False)
class BaselineField(model, device, chunk=262144, with_rgb=False)
    # Both expose:
    #   .device  (torch.device)  .stats  (dict of cumulative counters/timings)
    #   .reset_stats() -> None
    #   .query(pts) -> (sdf (N,) f32, rgb (N,3) f32|None,
    #                   depth (N,) int64, level (N,) int64)
    #     pts: (N,3) float tensor on .device. depth == 0 means the answer came
    #     from the octree (no network ran for that point); level is the
    #     octree leaf level (used by the tracer for far-leaf cell leaps).

def render_raymarch(field, out_path, img_size=800, elev=25.0, azim=45.0,
                    bg=(1,1,1), title=None, hit_eps=1e-3, max_steps=1024,
                    base_rgb=(0.65,0.75,0.9), depth_out_path=None) -> dict
    # Sphere-traces one ray per pixel, shades hits (two-sided Lambert
    # headlight; network RGB when the field provides it, else base_rgb),
    # writes the PNG, and optionally a turbo network-depth map of the hit
    # points to depth_out_path. Returns a stats dict (see bottom of file).

Timing values are wall-clock seconds (time.perf_counter with cuda sync).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from .render import _camera, _RADIUS, _HALF_DIAG, _FIT_MARGIN, _AMBIENT, \
    depth_colormap

_SQRT3_HALF = math.sqrt(3.0) / 2.0
_LEAP_NUDGE = 1e-6      # push past the cell boundary after a far-leaf leap
# Central-difference offset for normal estimation. Larger than the original
# 2e-3 because the SIREN field oscillates on sub-voxel scales; a 2e-3 step
# samples that noise, producing jittered normals that show up as isolated black
# speckle under single-sided Lambert. 8e-3 averages over the jitter while still
# being well inside a cell (~1/256 of the unit cube).
_NORMAL_EPS = 8e-3
# SDF-gradient magnitude below this is treated as a degenerate normal
# (thin features / silhouettes where the 6-neighbor gradient is unreliable).
# Such points fall back to the ray-direction normal so they don't spike black
# under single-sided Lambert.
_NORMAL_GMIN = 0.1


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

class _TorchOctree:
    """GPU port of Octree's per-level sorted-key lookup (see Octree._locate).

    Mirrors the numpy implementation exactly (same flat key
    ix + (iy << l) + (iz << 2l), same searchsorted semantics) so marching
    queries never leave the device.
    """

    def __init__(self, octree, device: torch.device) -> None:
        self.max_depth = int(octree.max_depth)
        self.device = device
        self.leaf_depth = torch.from_numpy(
            octree.leaf_depth.astype(np.int64)).to(device)
        self.leaf_sdf = torch.from_numpy(
            octree.leaf_sdf.astype(np.float32)).to(device)
        self.leaf_level = torch.from_numpy(
            octree.leaf_level.astype(np.int64)).to(device)
        self.level_keys: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for level, (keys, ids) in octree._level_keys.items():
            self.level_keys[int(level)] = (
                torch.from_numpy(np.ascontiguousarray(keys, dtype=np.int64)).to(device),
                torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int64)).to(device))

    def query(self, pts: torch.Tensor):
        """(N,3) tensor -> (net_depth (N,) i64, stored_sdf (N,) f32, level (N,) i64)."""
        pts = pts.clamp(0.0, 1.0)
        n = pts.shape[0]
        leaf_ids = torch.full((n,), -1, dtype=torch.long, device=pts.device)
        pending = torch.arange(n, device=pts.device)
        for level in range(self.max_depth + 1):
            if pending.numel() == 0:
                break
            entry = self.level_keys.get(level)
            if entry is None:
                continue
            keys, ids = entry
            scale = 1 << level
            ijk = torch.floor(pts[pending] * scale).long().clamp_(max=scale - 1)
            flat = ijk[:, 0] + (ijk[:, 1] << level) + (ijk[:, 2] << (2 * level))
            pos = torch.searchsorted(keys, flat).clamp_(max=keys.numel() - 1)
            hit = keys[pos] == flat
            leaf_ids[pending[hit]] = ids[pos[hit]]
            pending = pending[~hit]
        if bool((leaf_ids < 0).any()):
            raise RuntimeError("octree does not fully cover [0,1]^3")
        return (self.leaf_depth[leaf_ids], self.leaf_sdf[leaf_ids],
                self.leaf_level[leaf_ids])


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


class _FieldBase:
    """Shared chunking / stats plumbing for field backends."""

    kind = "base"

    def __init__(self, model, device, chunk: int, with_rgb: bool) -> None:
        self.model = model
        self.device = torch.device(device)
        self.chunk = max(1, int(chunk))
        self.with_rgb = bool(with_rgb)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {
            "queries": 0,          # field queries served (marching + normals)
            "zero_network": 0,     # of which answered by the octree (depth 0)
            "net_depth_sum": 0,    # sum of network depths (network-served only)
            "network_s": 0.0,
            "octree_query_s": 0.0,
        }

    def query(self, pts: torch.Tensor):
        outs = [self._query_chunk(c) for c in pts.split(self.chunk, dim=0)]
        sdf = torch.cat([o[0] for o in outs], dim=0)
        rgb = None
        if outs[0][1] is not None:
            rgb = torch.cat([o[1] for o in outs], dim=0)
        depth = torch.cat([o[2] for o in outs], dim=0)
        level = torch.cat([o[3] for o in outs], dim=0)
        return sdf, rgb, depth, level

    def _query_chunk(self, pts: torch.Tensor):
        raise NotImplementedError


class SandField(_FieldBase):
    """SAND adaptive field: octree depth map + forward_adaptive early exit.

    Depth-0 (far) leaves never touch the network: the tracer leaps across the
    leaf cell (a far leaf provably contains no surface — |sdf(center)| > half
    the cell diagonal — so the whole cell is empty space).
    """

    kind = "sand"

    def __init__(self, model, octree, device, chunk=262_144, with_rgb=False):
        super().__init__(model, device, chunk, with_rgb)
        self.octree = _TorchOctree(octree, self.device)

    @torch.no_grad()
    def _query_chunk(self, pts: torch.Tensor):
        n = pts.shape[0]
        _sync(self.device)
        t0 = time.perf_counter()
        depth, stored, level = self.octree.query(pts)
        _sync(self.device)
        self.stats["octree_query_s"] += time.perf_counter() - t0

        sdf = torch.empty(n, dtype=torch.float32, device=self.device)
        rgb = None
        far = depth == 0
        sdf[far] = stored[far]
        near = ~far
        n_net = int(near.sum().item())
        self.stats["queries"] += n
        self.stats["zero_network"] += n - n_net
        if n_net:
            idx = torch.nonzero(near, as_tuple=True)[0]
            _sync(self.device)
            t0 = time.perf_counter()
            y = self.model.forward_adaptive(pts[idx], depth[idx])
            _sync(self.device)
            self.stats["network_s"] += time.perf_counter() - t0
            sdf[idx] = y[:, 0].float()
            self.stats["net_depth_sum"] += int(depth[idx].sum().item())
            if self.with_rgb and y.shape[1] == 4:
                rgb = torch.full((n, 3), 0.5, dtype=torch.float32,
                                 device=self.device)
                rgb[idx] = y[:, 1:4].clamp(0.0, 1.0)
        return sdf, rgb, depth, level


class BaselineField(_FieldBase):
    """Full-depth network at every query point (plain sphere tracing)."""

    kind = "baseline"

    @torch.no_grad()
    def _query_chunk(self, pts: torch.Tensor):
        n = pts.shape[0]
        forward = getattr(self.model, "forward_final", None) or self.model.forward
        _sync(self.device)
        t0 = time.perf_counter()
        y = forward(pts)
        _sync(self.device)
        self.stats["network_s"] += time.perf_counter() - t0
        self.stats["queries"] += n
        num_layers = int(getattr(self.model, "num_layers", 8))
        self.stats["net_depth_sum"] += n * num_layers
        sdf = y[:, 0].float()
        rgb = y[:, 1:4].clamp(0.0, 1.0) if (self.with_rgb and y.shape[1] == 4) \
            else None
        depth = torch.full((n,), num_layers, dtype=torch.long, device=self.device)
        level = torch.zeros(n, dtype=torch.long, device=self.device)
        return sdf, rgb, depth, level


def make_sand_field(model, octree, device, chunk=262_144, with_rgb=False):
    return SandField(model, octree, device, chunk, with_rgb)


def make_baseline_field(model, device, chunk=262_144, with_rgb=False):
    return BaselineField(model, device, chunk, with_rgb)


# ---------------------------------------------------------------------------
# camera / rays
# ---------------------------------------------------------------------------

def _make_rays(img_size: int, elev: float, azim: float):
    """Per-pixel ray directions + ray/[0,1]^3-box clip range.

    Returns (eye (3,) f32, dirs (H,W,3) f32 unit, t0 (H,W) f32, t1 (H,W) f32,
    valid (H,W) bool). Rays with valid=False miss the unit cube entirely.
    """
    eye, right, up, fwd = _camera(elev, azim)
    tan_half = (_HALF_DIAG * _FIT_MARGIN) / _RADIUS
    W = H = int(img_size)
    xs = (np.arange(W, dtype=np.float64) + 0.5) / W * 2.0 - 1.0
    ys = 1.0 - (np.arange(H, dtype=np.float64) + 0.5) / H * 2.0
    d = (fwd[None, None, :]
         + xs[None, :, None] * tan_half * right[None, None, :]
         + ys[:, None, None] * tan_half * up[None, None, :])
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
    ta = (0.0 - eye[None, None, :]) * inv
    tb = (1.0 - eye[None, None, :]) * inv
    tmin = np.minimum(ta, tb).max(axis=-1)
    tmax = np.maximum(ta, tb).min(axis=-1)
    t0 = np.maximum(tmin, 0.0)
    valid = t0 < tmax
    return (eye.astype(np.float32), d.astype(np.float32),
            t0.astype(np.float32), tmax.astype(np.float32), valid)


def _cell_leap(pts: torch.Tensor, dirs: torch.Tensor,
               level: torch.Tensor) -> torch.Tensor:
    """Distance from pts to the exit face of their level-`level` octree cell.

    Cell indexing matches Octree._locate (floor(p*scale) clamped to scale-1),
    so the leap stays inside the located leaf. Degenerate (axis-parallel)
    components contribute +inf; the result is clamped to >= _LEAP_NUDGE.
    """
    scale = (1 << level).double()
    size = 1.0 / scale
    p = pts.clamp(0.0, 1.0).double()
    ijk = torch.floor(p * scale[:, None]).clamp(max=(scale - 1)[:, None])
    origin = ijk * size[:, None]
    bnd = torch.where(dirs > 0, origin + size[:, None], origin)
    dist = torch.where(dirs.abs() > 1e-12,
                       (bnd - p) / dirs.double(),
                       torch.full_like(p, float("inf")))
    leap = dist.min(dim=1).values.clamp(min=0.0) + _LEAP_NUDGE
    return leap.float()


# ---------------------------------------------------------------------------
# tracer
# ---------------------------------------------------------------------------

@torch.no_grad()
def _secant_refine(field, eye: torch.Tensor, dirs: torch.Tensor,
                   t_lo: torch.Tensor, t_hi: torch.Tensor,
                   f_lo: torch.Tensor, f_hi: torch.Tensor,
                   iters: int = 8) -> torch.Tensor:
    """Root-finding in [t_lo, t_hi]: secant steps with a bisection fallback.

    The field is a SIREN network output and oscillates on sub-voxel scales,
    so plain bisection converges to an arbitrary root once both sides wiggle;
    the secant follows the smooth local sign change much better. The bracket
    [lo, hi] is maintained by sign when possible and always converges.
    Invariant at entry: f_lo > 0 (outside); f_hi arbitrary.
    """
    lo, hi = t_lo.clone(), t_hi.clone()
    flo, fhi = f_lo.clone(), f_hi.clone()
    for _ in range(int(iters)):
        prev_span = hi - lo
        denom = fhi - flo
        t_new = torch.where(denom.abs() > 1e-12,
                            hi - fhi * (hi - lo) / denom, 0.5 * (lo + hi))
        eps = 0.02 * prev_span
        t_new = t_new.clamp(min=lo + eps, max=hi - eps)
        p = eye + t_new[:, None] * dirs
        f_new, _, _, _ = field.query(p)
        move_lo = f_new > 0
        lo = torch.where(move_lo, t_new, lo)
        flo = torch.where(move_lo, f_new, flo)
        hi = torch.where(move_lo, hi, t_new)
        fhi = torch.where(move_lo, fhi, f_new)
        # secant stalled (bracket barely moved): force a bisection step
        stalled = (hi - lo) > 0.9 * prev_span
        if bool(stalled.any()):
            mid = 0.5 * (lo + hi)
            p = eye + mid[:, None] * dirs
            f_mid, _, _, _ = field.query(p)
            lo = torch.where(stalled & (f_mid > 0), mid, lo)
            flo = torch.where(stalled & (f_mid > 0), f_mid, flo)
            hi = torch.where(stalled & (f_mid <= 0), mid, hi)
            fhi = torch.where(stalled & (f_mid <= 0), f_mid, fhi)
    return 0.5 * (lo + hi)


@torch.no_grad()
def _trace(field, eye: torch.Tensor, dirs: torch.Tensor, t0: torch.Tensor,
           t1: torch.Tensor, hit_eps: float, max_steps: int):
    """Sphere-trace `dirs` rays through `field`.

    Robust variant with sign-change bracketing: the tracer maintains the
    invariant that every live ray's last sampled field value is positive.
    When the new sample flips sign (the step overshot the surface — possible
    because the learned SDF is not perfectly 1-Lipschitz, and far-leaf leaps
    land past thin near bands), the crossing is refined with the secant
    method instead of letting the ray tunnel through the interior (far
    leaves never trigger hits by design).

    depth/rgb bookkeeping uses the bracket's near-leaf evaluations
    (interpolated at the refined t): re-querying AT the refined point could
    land it back in a depth-0 leaf, which would report gray/depth-0 for a
    genuine surface hit (this caused gray speckle in textured renders).

    Returns (hit (N,) bool, t_hit (N,) f32, depth_hit (N,) i64,
    rgb_hit (N,3) f32|None). Marching queries accumulate into field.stats.
    """
    dev = field.device
    n = dirs.shape[0]
    t = t0.clone()
    hit = torch.zeros(n, dtype=torch.bool, device=dev)
    t_hit = torch.zeros(n, dtype=torch.float32, device=dev)
    depth_hit = torch.zeros(n, dtype=torch.long, device=dev)
    rgb_hit = None

    prev_t = torch.full((n,), float("nan"), dtype=torch.float32, device=dev)
    prev_f = torch.zeros(n, dtype=torch.float32, device=dev)
    prev_depth = torch.zeros(n, dtype=torch.long, device=dev)
    prev_rgb = torch.zeros((n, 3), dtype=torch.float32, device=dev)  # dummy init

    live = torch.arange(n, device=dev)
    for _ in range(int(max_steps)):
        if live.numel() == 0:
            break
        p = eye + t[live, None] * dirs[live]
        sdf, rgb, depth, level = field.query(p)
        # SandField: sdf holds the stored (GT) value in far leaves, so it is
        # meaningful on both sides of a sign change.
        fval = sdf

        has_prev = ~torch.isnan(prev_t[live])
        crossed = has_prev & (fval * prev_f[live] < 0.0)
        far = depth == 0
        # A negative first sample means the ray starts inside geometry
        # (or, for the baseline, inside its unconstrained far field): hit now.
        starts_inside = ~has_prev & (fval < 0.0)
        direct = ~far & (fval.abs() < hit_eps) & ~crossed

        done = crossed | direct | starts_inside
        if bool(done.any()):
            didx = live[done]
            done_f = fval[done]                # current samples of done rays
            is_cross = crossed[done]
            t_new = t[live][done].clone()
            d_new = depth[done]
            rgb_new = rgb[done] if rgb is not None else None
            if bool(is_cross.any()):
                cidx = didx[is_cross]
                crow = torch.nonzero(is_cross, as_tuple=True)[0]
                t_ref = _secant_refine(field, eye, dirs[cidx],
                                       prev_t[cidx], t[cidx],
                                       prev_f[cidx], done_f[is_cross])
                t_new[is_cross] = t_ref
                # depth/rgb must come from a NEAR-leaf (network) side of the
                # bracket. Interpolating with a far-leaf side washes the
                # network color toward the far-leaf default gray (0.5) — this
                # caused gray contamination in textured renders. Pick the
                # near side; interpolate only when both sides are near.
                pd_near = prev_depth[cidx] > 0
                cd_near = depth[done][is_cross] > 0
                denom = (prev_f[cidx] - done_f[is_cross]).clamp(min=1e-12)
                w = (prev_f[cidx] / denom).clamp(0.0, 1.0)  # 1 -> prev sample
                pd = prev_depth[cidx].float()
                cd = depth[done][is_cross].float()
                interp_d = pd * w + cd * (1.0 - w)
                d_pick = torch.where(cd_near, cd,
                         torch.where(pd_near, pd, interp_d))
                d_new[is_cross] = torch.round(d_pick).long()
                if rgb is not None and rgb_new is not None:
                    prgb = prev_rgb[cidx]
                    crgb = rgb[done][is_cross]
                    interp_c = prgb * w[:, None] + crgb * (1.0 - w[:, None])
                    rgb_new[is_cross] = torch.where(
                        cd_near[:, None], crgb,
                        torch.where(pd_near[:, None], prgb, interp_c))
            hit[didx] = True
            t_hit[didx] = t_new
            depth_hit[didx] = d_new
            if rgb_new is not None:
                if rgb_hit is None:
                    rgb_hit = torch.full((n, 3), 0.5, dtype=torch.float32,
                                         device=dev)
                rgb_hit[didx] = rgb_new

        # survivors: remember this sample (position/depth/rgb), then advance
        survive = ~done
        prev_t[live] = t[live]
        prev_f[live] = fval
        prev_depth[live] = depth
        if rgb is not None:
            prev_rgb[live] = rgb
        step = torch.empty_like(fval)
        if bool(far.any()):
            step[far] = _cell_leap(p[far], dirs[live][far], level[far])
        near = ~far
        if bool(near.any()):
            # survivors all have fval > 0, so near steps move forward
            step[near] = fval[near]
        t_next = t[live] + step
        exited = t_next > t1[live]
        t[live] = t_next
        live = live[survive & ~exited]
    return hit, t_hit, depth_hit, rgb_hit


@torch.no_grad()
def _shade_normals(field, pts: torch.Tensor,
                   view_dirs: torch.Tensor | None = None) -> torch.Tensor:
    """Outward normals at hit points via central differences (6 queries).

    With ``view_dirs`` (the hit ray directions) supplied, applies the same
    stabilization as ``script.viz_render._stable_normals``: degenerate-gradient
    points fall back to ``-view_dirs``, and any normal pointing away from the
    camera is flipped, so single-sided Lambert does not spike black on the
    ~1% of hits whose SDF gradient is sign-inverted or near-zero. Without
    ``view_dirs`` the legacy raw gradient is returned (callers that only need
    a magnitude-agnostic orientation still work).
    """
    offs = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                         [0, 0, 1], [0, 0, -1]], dtype=torch.float32,
                        device=pts.device) * _NORMAL_EPS
    q = (pts[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    sdf, _, _, _ = field.query(q)
    g = torch.stack([sdf[0::6] - sdf[1::6],
                     sdf[2::6] - sdf[3::6],
                     sdf[4::6] - sdf[5::6]], dim=1) / (2.0 * _NORMAL_EPS)
    gmag = torch.linalg.norm(g, dim=1, keepdim=True).clamp(min=1e-12)
    n = (g / gmag).float()
    if view_dirs is None:
        return n
    degenerate = (gmag.squeeze(1) < _NORMAL_GMIN).unsqueeze(1)
    fallback = (-view_dirs).float()
    n = torch.where(degenerate, fallback, n)
    facing = (n * (-view_dirs).float()).sum(dim=1, keepdim=True)
    inward = facing < 0
    n = torch.where(inward, -n, n)
    return n


def _save_png(img: np.ndarray, out_path: str, bg, title) -> None:
    img8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(img8)
    if title:
        draw = ImageDraw.Draw(pil)
        lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        text_col = (30, 30, 30) if lum > 0.5 else (235, 235, 235)
        draw.text((10, 8), str(title), fill=text_col)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pil.save(out_path)


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------

def render_raymarch(field, out_path, img_size=800, elev=25.0, azim=45.0,
                    bg=(1.0, 1.0, 1.0), title=None, hit_eps=1e-3,
                    max_steps=1024, base_rgb=(0.65, 0.75, 0.9),
                    depth_out_path=None) -> dict:
    """Ray-march one image through `field` and save it as a PNG.

    field: SandField / BaselineField (or any object with the .query/.stats/
    .reset_stats/.device interface). Stats accumulate into field.stats and
    are also returned as a dict:

      {"img_size", "n_rays", "n_hit", "hit_frac",
       "march_queries", "normal_queries", "zero_network_frac",
       "mean_net_depth", "octree_query_s", "network_s",
       "trace_s", "shade_s", "total_s", "rays_per_s"}
    """
    t_total0 = time.perf_counter()
    dev = field.device
    eye_np, dirs_np, t0_np, t1_np, valid_np = _make_rays(img_size, elev, azim)
    H = W = int(img_size)

    eye = torch.from_numpy(eye_np).to(dev)
    dirs_v = torch.from_numpy(dirs_np[valid_np]).to(dev)
    t0_v = torch.from_numpy(t0_np[valid_np]).to(dev)
    t1_v = torch.from_numpy(t1_np[valid_np]).to(dev)
    n_rays = dirs_v.shape[0]

    field.reset_stats()
    _sync(dev)
    t_trace0 = time.perf_counter()
    hit, t_hit, depth_hit, rgb_hit = _trace(field, eye, dirs_v, t0_v, t1_v,
                                            hit_eps, max_steps)
    _sync(dev)
    trace_s = time.perf_counter() - t_trace0
    march_stats = dict(field.stats)

    # ---- shading: normals only at hit points (6 extra queries each) ----
    t_shade0 = time.perf_counter()
    hit_ids = torch.nonzero(hit, as_tuple=True)[0]
    img = np.empty((H, W, 3), dtype=np.float64)
    img[:] = np.asarray(bg, dtype=np.float64)
    depth_map = np.zeros((H, W), dtype=np.float64)
    if hit_ids.numel():
        p_hit = eye + t_hit[hit_ids, None] * dirs_v[hit_ids]
        view_dirs = dirs_v[hit_ids]
        normals = _shade_normals(field, p_hit, view_dirs)
        # Single-sided Lambert: only faces turned toward the light are lit; back
        # faces go dark (clamped to 0). Matches the browser visualizer's
        # shading; the earlier abs() lit back faces, which hid the
        # inward-facing surfaces that indicate self-intersection artifacts.
        ldir = -view_dirs  # headlight: light travels along the view ray
        diffuse = torch.clamp((normals * ldir).sum(dim=1), min=0.0)
        lam = _AMBIENT + (1.0 - _AMBIENT) * diffuse
        if rgb_hit is not None:
            base = rgb_hit[hit_ids]
        else:
            base = torch.tensor(base_rgb, dtype=torch.float32,
                                device=dev).expand_as(normals)
        col = (base * lam[:, None]).clamp(0.0, 1.0)
        flat = np.asarray(valid_np, dtype=bool).reshape(-1)
        img = img.reshape(-1, 3)
        img[np.nonzero(flat)[0][hit_ids.cpu().numpy()]] = \
            col.double().cpu().numpy()
        img = img.reshape(H, W, 3)
        depth_map.reshape(-1)[np.nonzero(flat)[0][hit_ids.cpu().numpy()]] = \
            depth_hit[hit_ids].double().cpu().numpy()
    _sync(dev)
    shade_s = time.perf_counter() - t_shade0
    shade_stats = dict(field.stats)

    _save_png(img, out_path, bg, title)
    if depth_out_path is not None:
        max_d = int(getattr(field.model, "num_layers", 8))
        dimg = np.ones((H, W, 3), dtype=np.float32)
        m = depth_map > 0
        dimg[m] = depth_colormap(depth_map[m], max_d)
        _save_png(dimg, depth_out_path, bg,
                  (title + " depth") if title else None)

    total_s = time.perf_counter() - t_total0
    n_hit = int(hit_ids.numel())
    march_q = march_stats["queries"]
    return {
        "img_size": int(img_size),
        "n_rays": int(n_rays),
        "n_hit": n_hit,
        "hit_frac": float(n_hit / max(n_rays, 1)),
        "march_queries": int(march_q),
        "normal_queries": int(shade_stats["queries"] - march_q),
        "zero_network_frac": float(march_stats["zero_network"]
                                   / max(march_q, 1)),
        "mean_net_depth": float(
            march_stats["net_depth_sum"]
            / max(march_q - march_stats["zero_network"], 1)),
        "octree_query_s": float(shade_stats["octree_query_s"]),
        "network_s": float(shade_stats["network_s"]),
        "trace_s": float(trace_s),
        "shade_s": float(shade_s),
        "total_s": float(total_s),
        "rays_per_s": float(n_rays / max(total_s, 1e-9)),
    }
