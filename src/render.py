"""Headless software mesh renderer + color maps. No OpenGL / display required.

CONTRACT:

def render_mesh(verts: np.ndarray, faces: np.ndarray, vert_colors: np.ndarray,
                out_path: str, img_size: int = 800, elev: float = 25.0,
                azim: float = 45.0, max_faces: int = 150_000,
                bg: tuple = (1.0, 1.0, 1.0), title: str | None = None) -> None
    # verts (V,3) in [0,1]^3, faces (F,3) int, vert_colors (V,3) float in [0,1].
    # Software rasterizer (numpy/PIL z-buffer, painter order or per-pixel depth):
    # perspective-or-orthographic view from elev/azim looking at (0.5,0.5,0.5),
    # per-face Lambert shading (face normal . headlight) modulating the mean of
    # the triangle's vertex colors. If F > max_faces, decimate with
    # trimesh.simplify_quadric_decimation first (re-sample colors at the decimated
    # vertices by nearest original vertex). Saves PNG to out_path.

def depth_colormap(depths: np.ndarray, max_depth: int) -> np.ndarray
    # (N,) numeric depths in [0..max_depth] -> (N,3) float RGB in [0,1],
    # discrete "turbo" colormap (matplotlib), 0 = dark gray (no evaluation).

def solid_colors(n: int, rgb=(0.65, 0.75, 0.9)) -> np.ndarray
    # Constant per-vertex color helper for the untextured render.

Implementation notes (not part of the contract):
- Perspective pinhole camera orbiting the target at radius 2.2, y-up world:
  elevation is the angle above the x-z plane, azimuth rotates around +y.
  The vertical FOV is auto-fitted to the unit cube's half-diagonal with a
  small margin, so any geometry inside [0,1]^3 stays in frame.
- Headlight = view direction; two-sided Lambert shading abs(n . l) with
  ambient 0.35, so meshes with inconsistent winding still read fine.
- Visibility: per-pixel inverse-depth (1/z) z-buffer. 1/z is linear in
  screen space under a pinhole projection, so plain barycentric
  interpolation of it is exact (no perspective correction needed for
  depth-only tests with flat per-face colors).
- Decimation: tries trimesh.simplify_quadric_decimation; if its optional
  backend (fast_simplification) is unavailable or it fails, falls back to an
  internal pure-numpy vertex-clustering decimator that brackets/bisects a
  grid resolution until the face count fits max_faces.
"""
from __future__ import annotations

import math
import os

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw

# camera / shading constants
_TARGET = np.array([0.5, 0.5, 0.5])
_RADIUS = 2.2                    # orbit radius around the target
_NEAR = 0.05                     # triangles touching this plane are skipped
_AMBIENT = 0.35                  # ambient term of the headlight Lambert
_FIT_MARGIN = 1.12               # FOV margin around the unit cube
_HALF_DIAG = math.sqrt(3.0) / 2.0  # half diagonal of the [0,1]^3 cube
_EPS = 1e-5                      # barycentric inclusion tolerance (avoids cracks)


def solid_colors(n: int, rgb=(0.65, 0.75, 0.9)) -> np.ndarray:
    """Constant per-vertex color helper for the untextured render."""
    c = np.asarray(rgb, dtype=np.float32).reshape(1, 3)
    return np.repeat(c, int(n), axis=0)


def depth_colormap(depths: np.ndarray, max_depth: int) -> np.ndarray:
    """(N,) numeric depths in [0..max_depth] -> (N,3) float RGB in [0,1].

    Discrete "turbo" colormap (matplotlib) sampled at integers 0..max_depth;
    index 0 is overridden to dark gray 0.15 (no network evaluation). Float
    inputs are rounded, everything is clipped into [0, max_depth].
    """
    max_depth = int(max_depth)
    d = np.rint(np.asarray(depths, dtype=np.float64)).astype(np.int64)
    d = np.clip(d, 0, max_depth)
    lut = colormaps["turbo"](np.linspace(0.0, 1.0, max_depth + 1))[:, :3]
    lut[0] = 0.15  # depth 0 = no evaluation -> dark gray
    return lut[d].astype(np.float32)


def render_mesh(verts: np.ndarray, faces: np.ndarray, vert_colors: np.ndarray,
                out_path: str, img_size: int = 800, elev: float = 25.0,
                azim: float = 45.0, max_faces: int = 150_000,
                bg: tuple = (1.0, 1.0, 1.0), title: str | None = None) -> None:
    """Software-render a colored triangle mesh and save it as a PNG."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    colors = np.clip(np.asarray(vert_colors, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    W = H = int(img_size)

    if len(faces) > int(max_faces):
        orig_v, orig_c = verts, colors
        verts, faces = _decimate(verts, faces, int(max_faces))
        # re-sample colors at the decimated vertices from the nearest original vertex
        from scipy.spatial import cKDTree
        _, nn = cKDTree(orig_v).query(verts)
        colors = orig_c[nn]

    if len(verts) == 0 or len(faces) == 0:
        img = np.empty((H, W, 3), dtype=np.float64)
        img[:] = np.asarray(bg, dtype=np.float64)
    else:
        # --- camera: perspective orbit, y-up, looking at the cube center ---
        eye, right, up, fwd = _camera(elev, azim)
        rel = verts - eye
        xc = rel @ right
        yc = rel @ up
        zc = rel @ fwd                       # depth, positive in front
        invz = 1.0 / np.maximum(zc, 1e-12)   # linear in screen space
        tan_half = (_HALF_DIAG * _FIT_MARGIN) / _RADIUS
        px = (0.5 + 0.5 * (xc * invz) / tan_half) * (W - 1)
        py = (0.5 - 0.5 * (yc * invz) / tan_half) * (H - 1)

        # --- per-face flat color: mean vertex color x two-sided Lambert ---
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        normals = np.cross(verts[i1] - verts[i0], verts[i2] - verts[i0])
        nlen = np.linalg.norm(normals, axis=1)
        nonzero = nlen > 0
        normals[nonzero] /= nlen[nonzero, None]
        lam = _AMBIENT + (1.0 - _AMBIENT) * np.abs(normals @ fwd)
        mean_rgb = (colors[i0] + colors[i1] + colors[i2]) / 3.0
        face_rgb = np.clip(mean_rgb * lam[:, None], 0.0, 1.0)

        img = _rasterize(px, py, invz, zc, faces, face_rgb, W, H, bg)

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
# internals
# ---------------------------------------------------------------------------

def _camera(elev_deg: float, azim_deg: float):
    """Perspective orbit camera. Returns (eye, right, up, forward)."""
    el = math.radians(elev_deg)
    az = math.radians(azim_deg)
    eye = _TARGET + _RADIUS * np.array([
        math.cos(el) * math.cos(az),
        math.sin(el),
        math.cos(el) * math.sin(az),
    ])
    fwd = _TARGET - eye
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(fwd @ world_up)) > 0.999:  # looking straight up/down
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, right, up, fwd


def _rasterize(px, py, invz, zc, faces, face_rgb, W, H, bg):
    """Per-pixel inverse-depth z-buffer with a per-triangle bbox loop."""
    img = np.empty((H, W, 3), dtype=np.float64)
    img[:] = np.asarray(bg, dtype=np.float64)
    zbuf = np.zeros((H, W), dtype=np.float64)  # inverse depth, larger = closer

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    x0, x1, x2 = px[i0], px[i1], px[i2]
    y0, y1, y2 = py[i0], py[i1], py[i2]

    # normalized barycentric edge functions: w0 = a0*x + b0*y + c0 etc.
    denom = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    xlo = np.minimum(np.minimum(x0, x1), x2)
    xhi = np.maximum(np.maximum(x0, x1), x2)
    ylo = np.minimum(np.minimum(y0, y1), y2)
    yhi = np.maximum(np.maximum(y0, y1), y2)
    ok = (np.abs(denom) > 1e-9)
    ok &= (zc[i0] > _NEAR) & (zc[i1] > _NEAR) & (zc[i2] > _NEAR)
    ok &= (xhi >= 0) & (xlo <= W - 1) & (yhi >= 0) & (ylo <= H - 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        a0 = (y1 - y2) / denom
        b0 = (x2 - x1) / denom
        a1 = (y2 - y0) / denom
        b1 = (x0 - x2) / denom
        c0 = -(a0 * x2 + b0 * y2)  # inf/NaN where denom==0; masked out by `ok`
        c1 = -(a1 * x2 + b1 * y2)
    iz2 = invz[i2]
    dz0 = invz[i0] - iz2
    dz1 = invz[i1] - iz2

    xg = np.arange(W)
    yg = np.arange(H)
    for f in np.nonzero(ok)[0].tolist():
        ix0 = int(math.floor(xlo[f]))
        if ix0 < 0:
            ix0 = 0
        ix1 = int(math.ceil(xhi[f]))
        if ix1 > W - 1:
            ix1 = W - 1
        iy0 = int(math.floor(ylo[f]))
        if iy0 < 0:
            iy0 = 0
        iy1 = int(math.ceil(yhi[f]))
        if iy1 > H - 1:
            iy1 = H - 1
        if ix1 < ix0 or iy1 < iy0:
            continue
        xs = xg[ix0:ix1 + 1]
        ys = yg[iy0:iy1 + 1]
        w0 = a0[f] * xs[None, :] + b0[f] * ys[:, None] + c0[f]
        w1 = a1[f] * xs[None, :] + b1[f] * ys[:, None] + c1[f]
        mask = (w0 >= -_EPS) & (w1 >= -_EPS) & (w0 + w1 <= 1.0 + _EPS)
        if not mask.any():
            continue
        z = iz2[f] + w0 * dz0[f] + w1 * dz1[f]
        zs = zbuf[iy0:iy1 + 1, ix0:ix1 + 1]
        upd = mask & (z > zs)
        if upd.any():
            zs[upd] = z[upd]
            img[iy0:iy1 + 1, ix0:ix1 + 1][upd] = face_rgb[f]
    return img


def _decimate(verts, faces, max_faces):
    """Reduce a mesh to at most `max_faces` triangles.

    Prefers trimesh.simplify_quadric_decimation; falls back to pure-numpy
    vertex clustering when the fast_simplification backend is unavailable.
    """
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        dec = mesh.simplify_quadric_decimation(face_count=max_faces)
        nv = np.asarray(dec.vertices, dtype=np.float64)
        nf = np.asarray(dec.faces, dtype=np.int64)
        if len(nv) == 0 or len(nf) == 0 or len(nf) > max_faces:
            raise RuntimeError("quadric decimation returned an invalid mesh")
        return nv, nf
    except Exception:
        return _vertex_cluster_decimate(verts, faces, max_faces)


def _vertex_cluster_decimate(verts, faces, target_faces):
    """Vertex-clustering decimation fallback (pure numpy).

    Vertices are merged into representatives of a uniform grid; faces that
    become degenerate or duplicated are dropped. The grid resolution is
    bracketed and bisected so the result has <= target_faces faces when
    possible (best effort otherwise).
    """
    vmin = verts.min(axis=0)
    span = np.maximum(verts.max(axis=0) - vmin, 1e-12)

    def cluster(g):
        g = int(max(2, g))
        cell = np.floor((verts - vmin) / span * g).astype(np.int64)
        np.clip(cell, 0, g - 1, out=cell)
        key = (cell[:, 0] * g + cell[:, 1]) * g + cell[:, 2]
        uniq, inverse = np.unique(key, return_inverse=True)
        nv = np.zeros((len(uniq), 3), dtype=np.float64)
        np.add.at(nv, inverse, verts)
        nv /= np.bincount(inverse, minlength=len(uniq))[:, None]
        nf = inverse[faces]
        keep = ((nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2])
                & (nf[:, 2] != nf[:, 0]))
        nf = nf[keep]
        if len(nf):
            nf = np.unique(np.sort(nf, axis=1), axis=0)
        return nv, np.ascontiguousarray(nf)

    # coarsest grid
    nv, nf = cluster(2)
    if len(nf) > target_faces:
        return nv, nf  # cannot satisfy the target even at the coarsest grid
    best = (nv, nf)
    lo = 2
    hi = 2
    # grow hi until it no longer fits
    while hi <= 2048:
        hi *= 2
        nv, nf = cluster(hi)
        if len(nf) <= target_faces:
            lo, best = hi, (nv, nf)
        else:
            break
    # bisect for the finest grid that still fits
    while hi - lo > 1:
        mid = (lo + hi) // 2
        nv, nf = cluster(mid)
        if len(nf) <= target_faces:
            lo, best = mid, (nv, nf)
        else:
            hi = mid
    return best
