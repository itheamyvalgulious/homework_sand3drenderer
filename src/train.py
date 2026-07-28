"""Training loop for T-MLP (SAND) and BaselineMLP.

CONTRACT:

def train_model(kind: str, color: bool, mesh_data, cfg: Config, device: str,
                out_dir: str, tag: str, octree=None) -> dict
    # kind: "sand" -> TMLP trained with sum over tails of L1 losses (paper Eq. 4/8);
    #       when `octree` is given, far-from-surface training points are filtered out
    #       via octree.is_far() with rejection resampling (paper Sec. 3.3).
    #       "baseline" -> BaselineMLP, standard L1 on final output only, no filtering.
    # color: False -> out_dim=1, SDF L1 loss.
    #        True  -> out_dim=4, loss = sdf_loss + cfg.rgb_lambda * rgb_loss (L1 on RGB).
    #        If mesh_data.has_color is False and color=True, raise ValueError.
    # Uses cfg.{iters,batch,lr,sigma,surf_frac,log_every,hidden,num_layers,w0,seed}.
    # Saves {out_dir}/ckpt_{tag}.pt (state_dict + constructor kwargs) and
    #       {out_dir}/meta_{tag}.json.
    # Returns meta dict with at least:
    #   {"tag", "kind", "color", "iters", "train_time_s", "ms_per_iter",
    #    "final_loss", "n_params", "ckpt": path}

def load_model(kind: str, ckpt_path: str, device: str):
    # Rebuild TMLP/BaselineMLP from checkpoint, eval mode.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .tmlp import TMLP, BaselineMLP, count_params

_KINDS = {"sand": TMLP, "baseline": BaselineMLP}


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _sample_near(mesh_data, octree, n: int, cfg: Config,
                 rng: np.random.Generator, rounds: int = 4):
    """Sample n training points, rejecting those in far (depth-0) octree leaves.

    Rejection-resamples the deficit for a few rounds; whatever is still missing is
    topped up with unfiltered points so the batch size stays exactly n.
    """
    pts_l: list[np.ndarray] = []
    sdf_l: list[np.ndarray] = []
    rgb_l: list[np.ndarray] = []
    have = 0
    for _ in range(rounds):
        if have >= n:
            break
        p, s, c = mesh_data.sample_training_points(n - have, cfg.surf_frac, cfg.sigma, rng)
        keep = ~octree.is_far(p)
        if keep.any():
            pts_l.append(p[keep])
            sdf_l.append(s[keep])
            if c is not None:
                rgb_l.append(c[keep])
            have += int(keep.sum())
    if have < n:
        # Top up with unfiltered points (paper-pragmatic fallback).
        p, s, c = mesh_data.sample_training_points(n - have, cfg.surf_frac, cfg.sigma, rng)
        pts_l.append(p)
        sdf_l.append(s)
        if c is not None:
            rgb_l.append(c)
    pts = np.concatenate(pts_l, axis=0)
    sdf = np.concatenate(sdf_l, axis=0)
    rgb = np.concatenate(rgb_l, axis=0) if rgb_l else None
    return pts, sdf, rgb


def train_model(kind: str, color: bool, mesh_data, cfg: Config, device: str,
                out_dir: str, tag: str, octree=None) -> dict:
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {sorted(_KINDS)}")
    if color and not getattr(mesh_data, "has_color", False):
        raise ValueError("color=True requested but mesh_data.has_color is False")

    torch.manual_seed(cfg.seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    out_dim = 4 if color else 1
    ctor_kwargs = dict(hidden=cfg.hidden, num_layers=cfg.num_layers,
                       out_dim=out_dim, w0=cfg.w0)
    model = _KINDS[kind](**ctor_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    os.makedirs(out_dir, exist_ok=True)

    # Pre-sample a pool of training points once (GT SDF evaluation dominates
    # per-iteration cost otherwise); minibatches are drawn from it each iter.
    # For kind=="sand" with an octree, far-region points are filtered out of
    # the pool (paper Sec. 3.3), once, instead of per iteration.
    pool_n = max(int(getattr(cfg, "pool_batches", 40)) * cfg.batch, cfg.batch)
    _sync(device)
    tp0 = time.perf_counter()
    if kind == "sand" and octree is not None:
        pts_p, sdf_p, rgb_p = _sample_near(mesh_data, octree, pool_n, cfg, rng)
    else:
        pts_p, sdf_p, rgb_p = mesh_data.sample_training_points(
            pool_n, cfg.surf_frac, cfg.sigma, rng,
            domain_frac=getattr(cfg, "domain_frac", 0.0))
    if color and rgb_p is None:
        raise ValueError("color=True but mesh_data returned rgb_gt=None")
    x_p = torch.from_numpy(np.ascontiguousarray(pts_p))
    s_p = torch.from_numpy(np.ascontiguousarray(sdf_p))
    r_p = torch.from_numpy(np.ascontiguousarray(rgb_p)) if color else None
    _sync(device)
    data_prep = time.perf_counter() - tp0
    print(f"[{tag}] data pool: {pool_n} points in {data_prep:.1f}s", flush=True)

    log: list[dict] = []
    last_loss = float("nan")
    _sync(device)
    t0 = time.perf_counter()
    for it in range(1, cfg.iters + 1):
        idx = torch.randint(pool_n, (cfg.batch,))
        x = x_p[idx].to(device, non_blocking=True)
        sdf_t = s_p[idx].to(device, non_blocking=True)
        rgb_t = r_p[idx].to(device, non_blocking=True) if color else None

        if kind == "sand":
            ys = model.forward_all(x)
            # Paper Eq. 4/8: L1 on the SDF channel summed over all tails.
            sdf_loss = sum(F.l1_loss(y[:, 0], sdf_t) for y in ys)
            loss = sdf_loss
            if color:
                rgb_loss = sum(F.l1_loss(y[:, 1:4], rgb_t) for y in ys)
                loss = loss + cfg.rgb_lambda * rgb_loss
        else:
            y = model(x)
            sdf_loss = F.l1_loss(y[:, 0], sdf_t)
            loss = sdf_loss
            if color:
                loss = loss + cfg.rgb_lambda * F.l1_loss(y[:, 1:4], rgb_t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

        if it == 1 or it % cfg.log_every == 0 or it == cfg.iters:
            log.append({"iteration": it, "loss": last_loss})
            print(f"[{tag}] iter {it}/{cfg.iters} loss {last_loss:.6f}", flush=True)
    _sync(device)
    train_time = time.perf_counter() - t0

    ckpt_path = os.path.join(out_dir, f"ckpt_{tag}.pt")
    torch.save({"kind": kind, "kwargs": ctor_kwargs,
                "state_dict": model.state_dict()}, ckpt_path)

    meta = {
        "tag": tag,
        "kind": kind,
        "color": color,
        "iters": cfg.iters,
        "train_time_s": train_time,
        "ms_per_iter": train_time * 1000.0 / cfg.iters,
        "data_prep_s": data_prep,
        "pool_size": pool_n,
        "final_loss": last_loss,
        "n_params": count_params(model),
        "ckpt": ckpt_path,
        "log": log,
    }
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_model(kind: str, ckpt_path: str, device: str):
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {sorted(_KINDS)}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except torch.AcceleratorError:
        # GPU temporarily full (shared device): build on CPU, weights are
        # moved to the device below anyway.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = _KINDS[kind](**ckpt["kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model
