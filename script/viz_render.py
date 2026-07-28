"""Interactive ray-marching renderer backend for the SAND SDF visualizer.

Scans every trained field (baseline / SAND / SAND+texture) found under the
output/ and report/ trees, then serves on-demand point-lit sphere-traced
renders as in-memory PNGs. The shading is the same two-sided Lambertian model
as ``src.raymarch.render_raymarch``, but the light is a movable point light
(exposed via ``light_pos`` / ``light_color`` / ``intensity``) instead of the
fixed headlight, so the web UI can relight the surface interactively.

Each (run-directory, checkpoint-tag) pair is a distinct selectable model, so
multiple assets (e.g. ``report/r2`` and ``report/suzanne``) show up as
separate entries in the UI. Fields are built lazily on first use, so startup
stays cheap.

CONTRACT:

class FieldCache:
    def __init__(self, device="cuda", roots=None)
    def list_models(self) -> list[dict]
    def render(self, tag, light_pos=..., light_color=..., intensity=1.0,
               elev=25.0, azim=45.0, img_size=384, ambient=0.15,
               base_rgb=(0.65,0.75,0.9), hit_eps=1e-3, max_steps=1024) -> dict
"""
from __future__ import annotations

import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import Config
from src.octree import Octree
from src.raymarch import (_make_rays, _shade_normals, _trace,
                          make_baseline_field, make_sand_field)
from src.train import load_model

# Central-difference offset for normal estimation. Larger than src.raymarch's
# 2e-3 because the SIREN field oscillates on sub-voxel scales; a 2e-3 step
# samples that noise, producing jittered normals that show up as isolated black
# speckle under single-sided Lambert. 8e-3 averages over the jitter while still
# being well inside a cell (~1/256 of the unit cube).
_NORMAL_EPS = 8e-3
# SDF-gradient magnitude below this is treated as a degenerate normal
# (thin features / silhouettes where the 6-neighbor gradient is unreliable).
# Such points fall back to the ray-direction normal so they don't spike black
# under single-sided Lambert (render_raymarch hides this with abs(), which we
# can't use because it also lights back faces — see the shading comment below).
_NORMAL_GMIN = 0.1

# tag -> (kind, with_rgb, label); with_rgb is also implied by out_dim == 4.
_TAG_MAP = {
    "base_geo":   ("baseline", False, "Baseline"),
    "sand_geo":   ("sand",     False, "SAND"),
    "sand_color": ("sand",     True,  "SAND+texture"),
}
# list_models() presentation order within a run (most informative field first).
_LIST_ORDER = ("sand_color", "sand_geo", "base_geo")
_CHUNK = 262_144


def _stable_normals(field, pts: torch.Tensor,
                    view_dirs: torch.Tensor) -> torch.Tensor:
    """Outward normals at hit points with degenerate-gradient + sign fallback.

    Same 6-neighbor central difference as ``src.raymarch._shade_normals``, but
    (a) uses a larger step (``_NORMAL_EPS``) to average over SIREN sub-voxel
    oscillation that otherwise jittered normals into isolated black speckle
    under single-sided Lambert, and (b) flips any normal pointing away from the
    camera: a surface hit is by construction front-facing, so its outward
    normal must lie in the camera hemisphere (``n . -view > 0``). Sign-flipped
    gradients (the learned SDF is not a perfect signed field) otherwise make
    ``n . l`` spuriously negative at lit points — the main speckle source on R2.
    Points with near-zero gradient magnitude fall back to ``-view`` first.
    """
    dev = pts.device
    offs = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                         [0, 0, 1], [0, 0, -1]], dtype=torch.float32,
                        device=dev) * _NORMAL_EPS
    q = (pts[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    sdf, _, _, _ = field.query(q)
    g = torch.stack([sdf[0::6] - sdf[1::6],
                     sdf[2::6] - sdf[3::6],
                     sdf[4::6] - sdf[5::6]], dim=1) / (2.0 * _NORMAL_EPS)
    gmag = torch.linalg.norm(g, dim=1, keepdim=True).clamp(min=1e-12)
    nrm = (g / gmag).float()
    degenerate = (gmag.squeeze(1) < _NORMAL_GMIN).unsqueeze(1)
    # Fallback: outward normal ≈ toward the camera (negated ray direction).
    fallback = (-view_dirs).float()
    n = torch.where(degenerate, fallback, nrm)
    # Sign correction: a front-facing hit's outward normal points toward the
    # camera; flip the ~1% whose gradient points inward.
    facing = (n * (-view_dirs).float()).sum(dim=1, keepdim=True)
    inward = facing < 0
    n = torch.where(inward, -n, n)
    return n


def _octree_path(root: str, tag: str) -> str | None:
    """Pick the octree npz for a tag under ``root`` (None for baseline)."""
    if tag == "sand_color":
        rgbcrit = os.path.join(root, "octree_sand_color_rgbcrit.npz")
        if os.path.exists(rgbcrit):
            return rgbcrit
        return os.path.join(root, "octree_sand_color.npz")
    if tag == "sand_geo":
        return os.path.join(root, "octree_sand_geo.npz")
    if tag == "base_geo":
        return os.path.join(root, "octree_base.npz")  # baseline needs no octree
    return None


def _run_label(root: str) -> str:
    """Short human name for a run directory (last path segment)."""
    return os.path.basename(os.path.normpath(root)) or root


class FieldCache:
    """Lazily-built cache of trained fields keyed by ``"<root>:<tag>"``.

    ``__init__`` only scans roots and records which (root, tag) checkpoints
    exist; the actual model+field for a pair is built on first ``render`` and
    cached, so adding many runs to the scan does not slow startup.
    """

    def __init__(self, device: str = "cuda",
                 roots: list[str] | None = None) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        if roots is None:
            roots = ["output"]
            report_dir = "report"
            if os.path.isdir(report_dir):
                for sub in sorted(os.listdir(report_dir)):
                    p = os.path.join(report_dir, sub)
                    if os.path.isdir(p):
                        roots.append(p)

        # All discovered (root, tag) pairs, in presentation order:
        # output/ first, then report subdirs (sorted); within each run, the
        # _LIST_ORDER ordering applies (sand_color before sand_geo before base).
        self._available: list[tuple[str, str]] = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for tag in _LIST_ORDER:
                ckpt = os.path.join(root, f"ckpt_{tag}.pt")
                if os.path.exists(ckpt):
                    self._available.append((root, tag))

        self._models: dict[str, dict] = {}  # key = f"{root}:{tag}" -> built field

    @staticmethod
    def _key(root: str, tag: str) -> str:
        return f"{root}:{tag}"

    def list_models(self) -> list[dict]:
        """All selectable models, grouped by run then by _LIST_ORDER.

        Each entry: ``{"tag", "label", "kind", "with_rgb", "root"}`` where
        ``tag`` is the opaque key to pass back to ``render`` and ``label`` is
        ``"<run> · <Field>"`` (e.g. ``"r2 · SAND+texture"``).
        """
        out = []
        for root, tag in self._available:
            kind, with_rgb, field_label = _TAG_MAP[tag]
            out.append({
                "tag": self._key(root, tag),
                "label": f"{_run_label(root)} · {field_label}",
                "kind": kind,
                "with_rgb": with_rgb,
                "root": root,
            })
        return out

    def _ensure_loaded(self, key: str) -> dict:
        m = self._models.get(key)
        if m is not None:
            return m
        # Find the (root, tag) this key refers to.
        root, tag = key.rsplit(":", 1)
        if (root, tag) not in self._available:
            raise KeyError(f"unknown model key {key!r}")
        kind, with_rgb, label = _TAG_MAP[tag]
        ckpt = os.path.join(root, f"ckpt_{tag}.pt")
        model = load_model(kind, ckpt, self.device)
        octree = None
        if kind == "sand":
            op = _octree_path(root, tag)
            if op and os.path.exists(op):
                octree = Octree.load(op)
            field = make_sand_field(model, octree, self.device,
                                    _CHUNK, with_rgb)
        else:
            field = make_baseline_field(model, self.device, _CHUNK, with_rgb)
        cfg = None
        cfg_path = os.path.join(root, "config.json")
        if os.path.exists(cfg_path):
            try:
                cfg = Config.load(cfg_path)
            except Exception:
                cfg = None
        m = {
            "model": model, "field": field, "cfg": cfg,
            "with_rgb": with_rgb, "kind": kind, "label": label,
            "root": root, "tag": tag,
        }
        self._models[key] = m
        return m

    def render(self, tag: str,
               light_pos=(2.0, 2.0, 2.0), light_color=(1.0, 1.0, 1.0),
               intensity: float = 1.0, elev: float = 25.0, azim: float = 45.0,
               img_size: int = 384, ambient: float = 0.15,
               base_rgb=(0.65, 0.75, 0.9), hit_eps=1e-3, max_steps=1024) -> dict:
        """Sphere-trace one image of ``tag`` under a point light; return PNG bytes.

        ``tag`` is the opaque key from ``list_models`` (``"<root>:<ckpt_tag>"``).
        Mirrors ``render_raymarch`` (valid-ray masking, _trace, _shade_normals,
        flat-pixel composition) but replaces the headlight with a movable point
        light at ``light_pos`` of color ``light_color`` scaled by ``intensity``.
        Returns ``{"png": <bytes>, "stats": {...}}``.
        """
        m = self._ensure_loaded(tag)
        field = m["field"]
        dev = field.device

        # The point-light shading below is ambient + diffuse, so the background
        # ("empty space" pixels, i.e. rays that miss the surface) should be lit
        # by ambient alone too — otherwise a black lit object on a white
        # background looks like a silhouette, not a darkened scene. With
        # ambient=0 the whole frame (object + bg) is black, which is the
        # intuitive "lights off" behavior the UI's environment-light slider
        # implies. The light color tints the background as well, so a red
        # ambient reads as a red void.
        lc_np = np.asarray(light_color, dtype=np.float64)
        amb = float(max(0.0, min(1.0, ambient)))
        bg_rgb = (amb * lc_np).clip(0.0, 1.0)

        t_total0 = time.perf_counter()
        eye_np, dirs_np, t0_np, t1_np, valid_np = _make_rays(
            img_size, elev, azim)
        H = W = int(img_size)

        eye = torch.from_numpy(eye_np).to(dev)
        dirs_v = torch.from_numpy(dirs_np[valid_np]).to(dev)
        t0_v = torch.from_numpy(t0_np[valid_np]).to(dev)
        t1_v = torch.from_numpy(t1_np[valid_np]).to(dev)
        n_rays = int(dirs_v.shape[0])

        field.reset_stats()
        hit, t_hit, depth_hit, rgb_hit = _trace(
            field, eye, dirs_v, t0_v, t1_v, hit_eps, max_steps)

        hit_ids = torch.nonzero(hit, as_tuple=True)[0]
        img = np.empty((H, W, 3), dtype=np.float64)
        img[:] = bg_rgb  # empty space lit by ambient * light color
        if hit_ids.numel():
            p_hit = eye + t_hit[hit_ids, None] * dirs_v[hit_ids]
            view_dirs = dirs_v[hit_ids]
            normals = _stable_normals(field, p_hit, view_dirs)
            L = torch.tensor(light_pos, device=dev, dtype=torch.float32)
            ldir = F.normalize(L - p_hit, dim=1)
            # Single-sided Lambert: only faces turned toward the light are lit;
            # back faces go dark (clamped to 0). The earlier abs() was inherited
            # from render_raymarch's headlight, where it compensates for ragged
            # MC mesh winding — but here normals come from the SDF gradient
            # (stable outward orientation), so abs() wrongly lit back faces,
            # turning a top-lit sphere into two bright hemispheres with a dark
            # equator instead of one bright hemisphere + shadowed bottom.
            diffuse = torch.clamp((normals * ldir).sum(dim=1), min=0.0)
            lam = ambient + (1.0 - ambient) * float(intensity) * diffuse
            if rgb_hit is not None:
                base = rgb_hit[hit_ids]
            else:
                base = torch.tensor(base_rgb, device=dev,
                                    dtype=torch.float32).expand_as(normals)
            light_c = torch.tensor(light_color, device=dev,
                                   dtype=torch.float32)
            col = (base * lam[:, None] * light_c[None, :]).clamp(0.0, 1.0)
            flat = np.asarray(valid_np, dtype=bool).reshape(-1)
            img = img.reshape(-1, 3)
            img[np.nonzero(flat)[0][hit_ids.cpu().numpy()]] = \
                col.double().cpu().numpy()
            img = img.reshape(H, W, 3)

        img8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img8).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        total_s = time.perf_counter() - t_total0
        n_hit = int(hit_ids.numel())
        return {
            "png": png_bytes,
            "stats": {
                "img_size": int(img_size),
                "n_rays": n_rays,
                "n_hit": n_hit,
                "hit_frac": float(n_hit / max(n_rays, 1)),
                "total_s": float(total_s),
                "light_pos": list(light_pos),
                "light_color": list(light_color),
                "intensity": float(intensity),
                "ambient": float(ambient),
                "elev": float(elev),
                "azim": float(azim),
                "tag": tag,
            },
        }


if __name__ == "__main__":
    # Tiny CLI smoke: list models and render one small image to a temp png.
    c = FieldCache()
    print("models:")
    for m in c.list_models():
        print(" ", m)
    if c.list_models():
        r = c.render(c.list_models()[0]["tag"], img_size=256)
        out = "/tmp/viz_render_smoke.png"
        with open(out, "wb") as f:
            f.write(r["png"])
        print(f"wrote {out} ({len(r['png'])} bytes)")
        print({k: v for k, v in r["stats"].items() if k != "png"})
