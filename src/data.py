"""Mesh loading, normalization, and ground-truth SDF / color sampling.

CONTRACT (must be kept stable — other modules are implemented against it):

class MeshData:
    mesh: trimesh.Trimesh          # normalized: centered at (0.5,0.5,0.5), longest bbox edge == 0.9
    has_color: bool                # True if texture image + UVs, or vertex colors, are available
    source_path: str

    @classmethod
    def load(cls, path: str) -> "MeshData"
        # Load .glb (merge multi-mesh scenes via trimesh.util.concatenate),
        # normalize into the unit cube as above, warn (not fail) if not watertight.

    def surface_points(self, n: int, rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray | None]
        # Uniform surface samples. Returns (pts (n,3) float32, rgb (n,3) float32 in [0,1] or None).
        # rgb from texture (barycentric UV interpolation -> image sample) or vertex colors.

    def sample_training_points(self, n: int, surf_frac: float, sigma: float,
                               rng: np.random.Generator
                               ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]
        # Paper Sec. 4.1: surf_frac fraction on-surface points, the rest are surface
        # points perturbed by gaussian noise (std sigma, per coordinate, normalized units).
        # Returns (pts (n,3) f32, sdf_gt (n,) f32, rgb_gt (n,3) f32|None).
        # rgb_gt is the surface color at the *underlying* surface point (also for perturbed pts).

    def sdf_at(self, pts: np.ndarray) -> np.ndarray
        # Batched GT signed distance via Open3D RaycastingScene.compute_signed_distance.

    def is_watertight(self) -> bool

Unit cube convention: everything (training, octree, grids) lives in [0,1]^3.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import open3d as o3d
import trimesh

# Normalization target: bbox center and longest bbox edge (paper unit-cube convention).
_NORM_CENTER = np.array([0.5, 0.5, 0.5], dtype=np.float64)
_NORM_EXTENT = 0.9
# Max points per RaycastingScene call (keeps memory bounded on big grids).
_SDF_CHUNK = 1_000_000


def _merge_loaded(loaded) -> trimesh.Trimesh:
    """Reduce a trimesh.load result to a single Trimesh.

    Scene.dump() applies node transforms; trimesh.util.concatenate merges the
    resulting meshes into one.
    """
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(loaded.dump())
    return loaded


def _texture_image(mesh: trimesh.Trimesh):
    """PIL texture image of the mesh visual, or None."""
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return None
    img = getattr(mat, "image", None)  # SimpleMaterial
    if img is None:
        img = getattr(mat, "baseColorTexture", None)  # PBRMaterial (glTF)
    return img


def _has_uv_texture(mesh: trimesh.Trimesh) -> bool:
    uv = getattr(mesh.visual, "uv", None)
    return (
        uv is not None
        and len(uv) == len(mesh.vertices)
        and _texture_image(mesh) is not None
    )


def _detect_color(mesh: trimesh.Trimesh) -> bool:
    if _has_uv_texture(mesh):
        return True
    return getattr(mesh.visual, "kind", None) == "vertex"


def _build_raycast(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    """Open3D raycasting scene from a trimesh mesh."""
    legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.ascontiguousarray(mesh.vertices, dtype=np.float32)),
        o3d.utility.Vector3iVector(np.ascontiguousarray(mesh.faces, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


class MeshData:
    mesh: trimesh.Trimesh
    has_color: bool
    source_path: str

    def __init__(self, mesh: trimesh.Trimesh, has_color: bool, source_path: str):
        self.mesh = mesh
        self.has_color = bool(has_color)
        self.source_path = str(source_path)
        # Set when a solid-union proxy replaced the geometry: exact colors then
        # come from the ORIGINAL mesh (see _colors_from_original).
        self._orig_mesh = None
        self._orig_raycast = None

        # Resolve the color source once: texture (float image + UVs) preferred,
        # per-vertex colors as fallback.
        self._tex = None  # (h, w, 3) float32 in [0, 1]
        self._uv = None  # (n_vertices, 2) float32
        self._vcolors = None  # (n_vertices, 3) float32 in [0, 1]
        if _has_uv_texture(mesh):
            self._uv = np.asarray(mesh.visual.uv, dtype=np.float32)
            self._tex = (
                np.asarray(_texture_image(mesh).convert("RGB"), dtype=np.float32) / 255.0
            )
        elif getattr(mesh.visual, "kind", None) == "vertex":
            vc = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)
            self._vcolors = vc[:, :3] / 255.0

        self._raycast = _build_raycast(mesh)

        # Sign strategy for sdf_at. Watertight meshes use Open3D's signed
        # distance directly; open / multi-part meshes use unsigned distance
        # combined with a generalized winding-number inside test (libigl),
        # which is robust for triangle soups. `self._fwn` is None or
        # (V float64, F int64, orient) where orient flips the inside test
        # when the authored normals point inward.
        self._fwn = None
        if not self.is_watertight():
            try:
                import igl  # noqa: F401
            except ImportError:
                warnings.warn(
                    "libigl not installed; falling back to Open3D parity sign, "
                    "which is unreliable for non-watertight meshes")
            else:
                V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
                F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
                orient = self._detect_winding_orient(V, F)
                self._fwn = (V, F, orient)

    @classmethod
    def load(cls, path: str, solid_union_res: int | None = None) -> "MeshData":
        """Load .glb, normalize into the unit cube, warn if not watertight.

        When `solid_union_res` is set and the mesh is NOT watertight, a
        watertight solid-union proxy is built on a res^3 grid (see
        `_solid_union_proxy`) and replaces the geometry; colors keep coming
        from the original mesh via nearest-surface lookup.
        """
        mesh = _merge_loaded(trimesh.load(path))

        bounds = mesh.bounds
        center = bounds.mean(axis=0)
        extent = float((bounds[1] - bounds[0]).max())
        if not np.isfinite(extent) or extent <= 0:
            raise ValueError(f"degenerate mesh bounds in {path}")
        scale = _NORM_EXTENT / extent
        mat = np.eye(4)
        mat[:3, :3] = np.eye(3) * scale
        mat[:3, 3] = _NORM_CENTER - center * scale
        mesh.apply_transform(mat)

        if not mesh.is_watertight:
            warnings.warn(f"{path} is not watertight; SDF signs may be unreliable")
        obj = cls(mesh, has_color=_detect_color(mesh), source_path=path)
        if solid_union_res and not obj.is_watertight():
            import time
            cache = f"temp/{os.path.basename(path)}.union{int(solid_union_res)}.ply"
            if os.path.exists(cache):
                proxy = trimesh.load(cache)
                print(f"[solid-union] loaded cached proxy {cache} "
                      f"(verts={len(proxy.vertices)} faces={len(proxy.faces)})",
                      flush=True)
            else:
                t0 = time.time()
                proxy = obj._solid_union_proxy(int(solid_union_res))
                print(f"[solid-union] proxy: verts={len(proxy.vertices)} "
                      f"faces={len(proxy.faces)} watertight={proxy.is_watertight} "
                      f"in {time.time() - t0:.1f}s", flush=True)
                proxy.export(cache)
                print(f"[solid-union] cached to {cache}", flush=True)
            obj._adopt_proxy(proxy)
        return obj

    def _solid_union_proxy(self, res: int, fill_cavities: bool = True,
                           sigma: float = 4.0) -> trimesh.Trimesh:
        """Watertight solid-union proxy of a non-watertight asset.

        Method (v2, winding-field smoothing + union):
          1. Sample the generalized winding number W and the unsigned distance
             on a res^3 grid (both cached next to the model).
          2. Occupancy = (W > 0.5) OR (gaussian_smooth(W, sigma) > 0.5).
             The raw term keeps every thin panel exactly; the smoothed term
             fills interior voids where winding leaks through shell gaps
             (w ~ 0.25-0.37) but whose neighbours are deep inside (w >= 1).
             The union is monotone: smoothing can only ADD solidity, never
             erase a genuine shell.
          3. Connected-component filtering drops phantom dust; the clean sign
             is combined with the true distance, and Marching Cubes yields a
             watertight surface. Enclosed residual shells stay invisible.
          With fill_cavities=False only the raw term is used.
        """
        from scipy import ndimage
        from skimage.measure import marching_cubes

        lin = np.linspace(0.5 / res, 1.0 - 0.5 / res, res)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        wcache = f"temp/{os.path.basename(self.source_path)}.winding{res}.npy"
        dcache = f"temp/{os.path.basename(self.source_path)}.dist{res}.npy"
        if os.path.exists(wcache):
            W = np.load(wcache)
            print(f"[solid-union] loaded cached winding {wcache}", flush=True)
        else:
            import igl
            if self._fwn is None:
                raise RuntimeError("winding field requires a non-watertight mesh")
            V, F, orient = self._fwn
            print(f"[solid-union] sampling winding field on {res}^3 grid...",
                  flush=True)
            W = np.empty(len(pts), dtype=np.float32)
            ch = 4_000_000
            for i in range(0, len(pts), ch):
                q = np.ascontiguousarray(pts[i:i + ch], dtype=np.float64)
                W[i:i + ch] = igl.fast_winding_number(V, F, q).astype(np.float32)
            W = (W * orient).reshape(res, res, res)
            np.save(wcache, W.astype(np.float32))
            print(f"[solid-union] cached winding to {wcache}", flush=True)
        if os.path.exists(dcache):
            dist = np.load(dcache)
        else:
            dist = self.unsigned_sdf_at(pts).reshape(res, res, res)
            np.save(dcache, dist.astype(np.float32))
            print(f"[solid-union] cached dist to {dcache}", flush=True)

        occ = W > 0.5
        if fill_cavities:
            occ |= ndimage.gaussian_filter(W, sigma=sigma, mode="nearest") > 0.5
        labels, n_comp = ndimage.label(occ)
        if n_comp > 1:
            sizes = ndimage.sum(occ, labels, index=np.arange(1, n_comp + 1))
            # keep genuine parts (arms/treads are >= thousands of voxels),
            # drop phantom dust from sign flicker
            keep = (sizes >= 512) | (sizes >= 0.001 * sizes.max())
            occ = np.isin(labels, np.nonzero(keep)[0] + 1)
            print(f"[solid-union] components {n_comp} -> kept "
                  f"{int(keep.sum())}", flush=True)
        sign = np.where(occ, -1.0, 1.0).astype(np.float32)
        field = (sign * dist).astype(np.float32)
        verts, faces, _, _ = marching_cubes(field, level=0.0,
                                            spacing=(1.0 / res,) * 3)
        verts = (verts + 0.5 / res).astype(np.float32)
        return trimesh.Trimesh(verts, faces, process=True)

    def _adopt_proxy(self, proxy: trimesh.Trimesh) -> None:
        """Replace geometry with the watertight proxy; keep colors from the
        original mesh via exact closest-point lookup (barycentric texture
        interpolation on the nearest original triangle)."""
        # Keep the ORIGINAL mesh + a raycast scene for exact color transfer.
        self._orig_mesh = self.mesh
        self._orig_raycast = _build_raycast(self._orig_mesh)
        self.mesh = proxy
        self._fwn = None  # proxy is watertight: parity sign is well-defined
        self._raycast = _build_raycast(proxy)

    def _colors_from_original(self, pts: np.ndarray) -> np.ndarray | None:
        """Exact GT color at (proxy) surface points: closest point on the
        original mesh -> barycentric UV / vertex-color interpolation."""
        if self._orig_mesh is None:
            return None
        if self._tex is None and self._vcolors is None:
            return None
        q = o3d.core.Tensor(np.ascontiguousarray(pts, dtype=np.float32))
        ans = self._orig_raycast.compute_closest_points(q)
        prim = ans["primitive_ids"].numpy().astype(np.int64)
        uv2 = ans["primitive_uvs"].numpy().astype(np.float32)  # (n,2) barycentric
        valid = prim >= 0
        prim = np.clip(prim, 0, len(self._orig_mesh.faces) - 1)
        # open3d/Embree convention: p = (1-u-v)*v0 + u*v1 + v*v2
        bary = np.stack([1.0 - uv2[:, 0] - uv2[:, 1], uv2[:, 0], uv2[:, 1]],
                        axis=1).astype(np.float32)
        tri_vert = self._orig_mesh.faces[prim]
        if self._tex is not None:
            uv = (bary[:, :, None] * self._uv[tri_vert]).sum(axis=1)
            h, w = self._tex.shape[:2]
            x = np.round(uv[:, 0] * (w - 1)).astype(np.int64) % w
            y = np.round((1.0 - uv[:, 1]) * (h - 1)).astype(np.int64) % h
            rgb = self._tex[y, x].astype(np.float32)
        else:
            rgb = (bary[:, :, None] * self._vcolors[tri_vert]).sum(axis=1)
            rgb = rgb.astype(np.float32)
        rgb[~valid] = 0.5
        return rgb

    def is_watertight(self) -> bool:
        return bool(self.mesh.is_watertight)

    def _detect_winding_orient(self, V: np.ndarray, F: np.ndarray) -> float:
        """+1.0 when authored normals point outward (inside iff w > 0.5), else -1.0.

        Probes the winding number at small +/- face-normal offsets from surface
        samples: the inward side must look more 'inside' than the outward side.
        """
        import igl

        sp, fidx = trimesh.sample.sample_surface(self.mesh, 5000, seed=0)
        fn = self.mesh.face_normals[fidx]
        eps = 0.004
        wp = igl.fast_winding_number(V, F, np.ascontiguousarray(sp + eps * fn, dtype=np.float64))
        wm = igl.fast_winding_number(V, F, np.ascontiguousarray(sp - eps * fn, dtype=np.float64))
        return 1.0 if float(np.median(wm)) > float(np.median(wp)) else -1.0

    def unsigned_sdf_at(self, pts: np.ndarray) -> np.ndarray:
        """Batched unsigned distance to the surface, (n,3) -> (n,) float32."""
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        out = []
        for i in range(0, len(pts), _SDF_CHUNK):
            chunk = o3d.core.Tensor(pts[i : i + _SDF_CHUNK])
            out.append(self._raycast.compute_distance(chunk).numpy())
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def sdf_at(self, pts: np.ndarray) -> np.ndarray:
        """Batched GT signed distance, (n,3) -> (n,) float32."""
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if self._fwn is None:
            out = []
            for i in range(0, len(pts), _SDF_CHUNK):
                chunk = o3d.core.Tensor(pts[i : i + _SDF_CHUNK])
                out.append(self._raycast.compute_signed_distance(chunk).numpy())
            if not out:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(out).astype(np.float32)
        # Non-watertight: unsigned distance + winding-number sign.
        import igl

        V, F, orient = self._fwn
        out = []
        for i in range(0, len(pts), _SDF_CHUNK):
            chunk32 = pts[i : i + _SDF_CHUNK]
            d = self._raycast.compute_distance(o3d.core.Tensor(chunk32)).numpy()
            w = igl.fast_winding_number(V, F, np.ascontiguousarray(chunk32, dtype=np.float64))
            inside = (orient * w) > 0.5
            out.append(np.where(inside, -d, d))
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def surface_points(self, n: int, rng: np.random.Generator):
        """Uniform (area-weighted) surface samples with optional color."""
        seed = int(rng.integers(0, 2**31 - 1))
        pts, face_idx = trimesh.sample.sample_surface(self.mesh, n, seed=seed)
        pts = pts.astype(np.float32)
        if self._orig_raycast is not None:
            # Geometry is a solid-union proxy: exact colors from the original
            # textured mesh via closest-point + barycentric interpolation.
            return pts, self._colors_from_original(pts)
        return pts, self._colors_at(face_idx, pts)

    def sample_training_points(self, n: int, surf_frac: float, sigma: float,
                               rng: np.random.Generator, domain_frac: float = 0.0):
        """Paper Sec. 4.1 sampling, plus optional uniform-domain points.

        Fraction surf_frac: on-surface points. The remainder is split between
        gaussian-perturbed surface points (std sigma) and, when domain_frac > 0,
        points sampled uniformly in [0,1]^3 (fraction domain_frac of n) which
        supervise the far field. Returns (pts, sdf_gt, rgb_gt|None); rgb_gt for
        uniform points is the color of the nearest surface sample.
        """
        n_surf = int(round(n * surf_frac))
        n_uniform = int(round(n * domain_frac))
        n_uniform = min(max(n_uniform, 0), n - n_surf)
        surf, rgb = self.surface_points(n, rng)
        pts = surf.copy()
        n_pert = n - n_surf - n_uniform
        if n_pert > 0:
            noise = rng.normal(0.0, sigma, size=(n_pert, 3)).astype(np.float32)
            pts[n_surf:n_surf + n_pert] += noise
        if n_uniform > 0:
            pts[n - n_uniform:] = rng.uniform(0.0, 1.0, size=(n_uniform, 3))
            if rgb is not None:
                # Extend the color field off-surface by nearest-surface color.
                from scipy.spatial import cKDTree
                tree = cKDTree(surf.astype(np.float64))
                _, nn = tree.query(pts[n - n_uniform:].astype(np.float64))
                rgb = rgb.copy()
                rgb[n - n_uniform:] = rgb[nn]
        sdf = self.sdf_at(pts)
        return pts.astype(np.float32), sdf, rgb

    def _colors_at(self, face_idx: np.ndarray, pts: np.ndarray):
        """(n,3) float32 RGB in [0,1] at surface points, or None if no color."""
        if self._tex is None and self._vcolors is None:
            return None
        bary = trimesh.triangles.points_to_barycentric(
            self.mesh.triangles[face_idx], pts
        )
        tri_vert = self.mesh.faces[face_idx]  # (n, 3) vertex indices
        if self._tex is not None:
            # Barycentric UV interpolation, then nearest texture sample.
            uv = (bary[:, :, None] * self._uv[tri_vert]).sum(axis=1)  # (n, 2)
            h, w = self._tex.shape[:2]
            # trimesh flips the glTF v axis on load, so UVs here have their
            # origin at the bottom-left: image row = (1 - v) * (h - 1).
            x = np.round(uv[:, 0] * (w - 1)).astype(np.int64) % w
            y = np.round((1.0 - uv[:, 1]) * (h - 1)).astype(np.int64) % h
            return self._tex[y, x].astype(np.float32)
        # Vertex colors, barycentrically interpolated.
        rgb = (bary[:, :, None] * self._vcolors[tri_vert]).sum(axis=1)
        return rgb.astype(np.float32)
