"""T-MLP (tailed MLP, SAND) and BaselineMLP (plain full-depth MLP).

CONTRACT (must be kept stable):

class TMLP(torch.nn.Module):
    def __init__(self, hidden: int = 256, num_layers: int = 8, out_dim: int = 1,
                 w0: float = 30.0, in_dim: int = 3)
        # SIREN backbone (sine activations, Sitzmann et al. init; first layer omega w0,
        # hidden layers omega w0 is fine). Each hidden layer i has output tail(s):
        #   i == 1:      t_1 = W h_1 + b                      (coarse prediction)
        #   i >= 2:      t_i = (W0 h_i + b0) * (W1 h_i + b1)  (Hadamard product, paper Eq. 3)
        # Cumulative output y_i = y_{i-1} + t_i, y_0 = 0      (paper Eq. 2)
        # out_dim == 1 (SDF only) or 4 (SDF + RGB).

    def forward_all(self, x: Tensor) -> list[Tensor]
        # x (N,in_dim) -> list of num_layers tensors y_i, each (N,out_dim).

    def forward_final(self, x: Tensor) -> Tensor
        # == forward_all(x)[-1], but may be implemented directly.

    def forward_adaptive(self, x: Tensor, depths: Tensor) -> Tensor
        # depths (N,) int64 with values in [1, num_layers]; returns y_{depths[n]}(x[n])
        # computed layer-wise, only evaluating layers <= each point's depth.
        # (Points with depth 0 are NOT passed here; the caller substitutes stored SDF.)

class BaselineMLP(torch.nn.Module):
    def __init__(self, hidden: int = 256, num_layers: int = 8, out_dim: int = 1,
                 w0: float = 30.0, in_dim: int = 3)
        # Same SIREN backbone, single output head at the last layer. No tails.
    def forward(self, x: Tensor) -> Tensor  # (N,out_dim)

def count_params(model: torch.nn.Module) -> int
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch import Tensor


class SineLayer(nn.Module):
    """SIREN layer: y = sin(w0 * (W x + b)) with Sitzmann et al. initialization."""

    def __init__(self, in_features: int, out_features: int, w0: float = 30.0,
                 is_first: bool = False):
        super().__init__()
        self.in_features = in_features
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                # Official SIREN first-layer init: uniform(-1/fan_in, 1/fan_in).
                self.linear.weight.uniform_(-1.0 / self.in_features,
                                            1.0 / self.in_features)
            else:
                # Official SIREN hidden-layer init: uniform(-sqrt(6/fan_in)/w0, ...).
                bound = math.sqrt(6.0 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * self.linear(x))


def _make_backbone(in_dim: int, hidden: int, num_layers: int, w0: float) -> nn.ModuleList:
    layers = [SineLayer(in_dim, hidden, w0=w0, is_first=True)]
    for _ in range(num_layers - 1):
        layers.append(SineLayer(hidden, hidden, w0=w0, is_first=False))
    return nn.ModuleList(layers)


class TMLP(nn.Module):
    def __init__(self, hidden: int = 256, num_layers: int = 8, out_dim: int = 1,
                 w0: float = 30.0, in_dim: int = 3):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.w0 = w0
        self.in_dim = in_dim

        self.layers = _make_backbone(in_dim, hidden, num_layers, w0)

        # Tail 1: plain linear head (coarse prediction).
        self.tail_first = nn.Linear(hidden, out_dim)
        # Tails i >= 2: residual via Hadamard product of two linear heads (Eq. 3).
        self.tail_res_a = nn.ModuleList(
            nn.Linear(hidden, out_dim) for _ in range(num_layers - 1))
        self.tail_res_b = nn.ModuleList(
            nn.Linear(hidden, out_dim) for _ in range(num_layers - 1))
        # Zero-init the second Hadamard head so residuals start exactly at 0 and the
        # cumulative outputs all begin at the coarse prediction; training then grows
        # the residuals as needed (both heads still receive gradients).
        for head in self.tail_res_b:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _tail(self, i: int, h: Tensor) -> Tensor:
        if i == 0:
            return self.tail_first(h)
        return self.tail_res_a[i - 1](h) * self.tail_res_b[i - 1](h)

    def forward_all(self, x: Tensor) -> list[Tensor]:
        ys: list[Tensor] = []
        y = None
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            t = self._tail(i, h)
            y = t if y is None else y + t
            ys.append(y)
        return ys

    def forward_final(self, x: Tensor) -> Tensor:
        y = None
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            t = self._tail(i, h)
            y = t if y is None else y + t
        return y

    def forward_adaptive(self, x: Tensor, depths: Tensor) -> Tensor:
        depths = depths.to(device=x.device, dtype=torch.long).view(-1)
        out = x.new_zeros((x.shape[0], self.out_dim))
        idx = torch.arange(x.shape[0], device=x.device)  # rows still being propagated
        h = x
        y = None
        for i, layer in enumerate(self.layers):
            h = layer(h)
            t = self._tail(i, h)
            y = t if y is None else y + t
            # Points whose depth == i+1 exit here with their cumulative output.
            done = depths[idx] == (i + 1)
            if done.any():
                out[idx[done]] = y[done]
            # Keep propagating only points with depth > i+1.
            keep = depths[idx] > (i + 1)
            if not keep.any():
                break
            idx = idx[keep]
            h = h[keep]
            y = y[keep]
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_final(x)


class BaselineMLP(nn.Module):
    def __init__(self, hidden: int = 256, num_layers: int = 8, out_dim: int = 1,
                 w0: float = 30.0, in_dim: int = 3):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.w0 = w0
        self.in_dim = in_dim

        self.layers = _make_backbone(in_dim, hidden, num_layers, w0)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
