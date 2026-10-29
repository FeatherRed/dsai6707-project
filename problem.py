"""Unified objective interface.

The DE kernel searches in normalized [0,1] coordinates; :class:`Problem` maps
them to native bounds [lb, ub] and calls the user's vectorized objective
exactly once per evaluation batch. To plug in your own objective, implement a
function ``f(x)`` that maps a (N, D) torch Tensor to an (N,) Tensor of values
to minimize, then wrap it in :class:`Problem`.
"""
import torch


class Problem:
    def __init__(self, func, dim, lb=-5.0, ub=5.0, dtype=torch.float64, name="custom"):
        self.func, self.dim, self.dtype, self.name = func, dim, dtype, name
        self.lb = torch.full((dim,), float(lb), dtype=dtype)
        self.ub = torch.full((dim,), float(ub), dtype=dtype)

    def denormalize(self, x):
        """Map normalized [0,1] coordinates to native [lb, ub] coordinates."""
        x = torch.as_tensor(x, dtype=self.dtype)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        native = self.lb + (self.ub - self.lb) * torch.clamp(x, 0.0, 1.0)
        return native[0] if single else native

    def eval(self, x):
        """Evaluate a batch of normalized coordinates; returns (N,) costs."""
        native = self.denormalize(x)
        values = torch.as_tensor(self.func(native), dtype=self.dtype).reshape(-1)
        if values.numel() != len(native):
            raise ValueError("Objective must return one value per individual")
        return values


def sphere(x):
    """Demo objective: sum of squares; optimum 0 at the origin."""
    return (x * x).sum(dim=1)
