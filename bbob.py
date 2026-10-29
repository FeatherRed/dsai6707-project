"""Fixed 40D BBOB functions backed by the adjacent bbob40d.pt parameter file.

The 24 frozen instances and train/test split come from the original EvoX-meta
benchmark manifest. Instance generation is performed once during export;
runtime evaluation uses only saved float64 tensors and the original formulas.
COCO-derived numerical code: see LICENSE-COCO.md for upstream notices.
"""
import math
from functools import cache
from pathlib import Path

import torch

from problem import Problem

__all__ = ["BBOBProblem", "BBOB_NAMES"]

BBOB_NAMES = {
    1: "Sphere", 2: "Separable Ellipsoid", 3: "Rastrigin", 4: "Bueche-Rastrigin",
    5: "Linear Slope", 6: "Attractive Sector", 7: "Step Ellipsoid", 8: "Rosenbrock",
    9: "Rotated Rosenbrock", 10: "Rotated Ellipsoid", 11: "Discus", 12: "Bent Cigar",
    13: "Sharp Ridge", 14: "Different Powers", 15: "Rotated Rastrigin", 16: "Weierstrass",
    17: "Schaffers F7", 18: "Ill-conditioned Schaffers F7", 19: "Griewank-Rosenbrock F8F2",
    20: "Schwefel", 21: "Gallagher 101 Peaks", 22: "Gallagher 21 Peaks", 23: "Katsuura",
    24: "Lunacek bi-Rastrigin",
}


def _oscillate(value: torch.Tensor) -> torch.Tensor:
    """Apply the elementwise BBOB oscillation transformation."""

    magnitude = torch.where(value != 0.0, torch.abs(value), torch.ones_like(value))
    scaled_log = torch.log(magnitude) / 0.1
    # Multiply the exponent by 0.1 before ``exp``.  This is algebraically
    # identical to ``exp(t) ** 0.1`` but avoids an unnecessary float32
    # overflow for otherwise finite BBOB values (notably F6).
    positive = torch.exp(0.1 * (scaled_log + 0.49 * (torch.sin(scaled_log) + torch.sin(0.79 * scaled_log))))
    negative = -torch.exp(0.1 * (scaled_log + 0.49 * (torch.sin(0.55 * scaled_log) + torch.sin(0.31 * scaled_log))))
    return torch.where(value > 0.0, positive, torch.where(value < 0.0, negative, value))


def _asymmetric(value: torch.Tensor, beta: float) -> torch.Tensor:
    dimension = value.shape[-1]
    positive = value > 0.0
    safe = torch.where(positive, value, torch.ones_like(value))
    exponent = 1.0 + beta * torch.linspace(
        0.0,
        1.0,
        dimension,
        dtype=value.dtype,
        device=value.device,
    ) * torch.sqrt(safe)
    return torch.where(positive, safe**exponent, value)


def _boundary_penalty(value: torch.Tensor, factor: float) -> torch.Tensor:
    excess = torch.clamp(torch.abs(value) - 5.0, min=0.0)
    return factor * torch.sum(excess**2, dim=-1)


def _rosenbrock(value: torch.Tensor) -> torch.Tensor:
    return 100.0 * torch.sum(
        (value[:, :-1] ** 2 - value[:, 1:]) ** 2,
        dim=-1,
    ) + torch.sum((value[:, :-1] - 1.0) ** 2, dim=-1)


@cache
def _load_benchmark():
    path = Path(__file__).with_name("bbob40d.pt")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (data["format"] != "bbob40d-v1" or data["dimension"] != 40
            or data["dtype"] != "float64" or set(data["functions"]) != set(BBOB_NAMES)):
        raise ValueError(f"Invalid fixed BBOB40D parameter file: {path}")
    return data


class BBOBProblem(Problem):
    """One saved 40D instance; eval() accepts normalized [0,1] populations.

    evaluate() applies the native [-5,5] objective formula. The inherited
    Problem.eval() performs the coordinate mapping exactly once.
    """

    def __init__(self, function_id):
        if isinstance(function_id, bool) or not isinstance(function_id, int) or function_id not in BBOB_NAMES:
            raise ValueError("function_id must be an integer in 1..24")
        data = _load_benchmark()
        entry = data["functions"][function_id]
        self.function_id = function_id
        self.dimension = data["dimension"]
        self.instance_id = entry["instance_id"]
        self.split = entry["split"]
        for name, tensor in entry["state"].items():
            setattr(self, name, tensor.clone())
        super().__init__(self.evaluate, dim=self.dimension,
                         lb=data["bounds"][0], ub=data["bounds"][1],
                         dtype=torch.float64, name=f"bbob_f{function_id:02d}")

    @property
    def optimum(self):
        """Saved objective offset f_opt, for reporting only."""
        return float(self._f_opt.item())

    def evaluate(self, population: torch.Tensor) -> torch.Tensor:
        """Evaluate a batch of native coordinates using the saved instance."""
        if not isinstance(population, torch.Tensor):
            raise TypeError("population must be a torch.Tensor")
        if population.ndim != 2 or population.shape[1] != self.dimension:
            raise ValueError(f"population must have shape (N, {self.dimension})")
        if not population.is_floating_point():
            raise TypeError("population must have a floating dtype")
        if population.device != self.x_opt.device:
            raise ValueError("Frozen BBOB40D parameters and input must both be on CPU")
        value = population.to(self.x_opt.dtype)
        return (self._evaluate_true(value) + self._f_opt).to(population.dtype)

    def _evaluate_true(self, value: torch.Tensor) -> torch.Tensor:
        function_id = self.function_id
        dimension = self.dimension

        if function_id == 1:
            z = value - self.x_opt
            return torch.sum(z**2, dim=-1)

        if function_id == 2:
            z = _oscillate(value - self.x_opt)
            return torch.sum(self.scales * z**2, dim=-1)

        if function_id == 3:
            z = _oscillate(value - self.x_opt)
            z = _asymmetric(z, beta=0.2) * self.scales
            return 10.0 * (dimension - torch.sum(torch.cos(2.0 * math.pi * z), dim=-1)) + torch.sum(z**2, dim=-1)

        if function_id == 4:
            z = _oscillate(value - self.x_opt)
            even = torch.arange(dimension, device=value.device) % 2 == 0
            z = torch.where(even & (z > 0.0), 10.0 * z, z)
            z = z * self.scales
            core = 10.0 * (dimension - torch.sum(torch.cos(2.0 * math.pi * z), dim=-1)) + torch.sum(z**2, dim=-1)
            return core + _boundary_penalty(value, 100.0)

        if function_id == 5:
            clipped = torch.where(
                value * self.x_opt > 25.0,
                torch.sign(value) * 5.0,
                value,
            )
            return clipped @ self.scales + 5.0 * torch.sum(torch.abs(self.scales))

        if function_id == 6:
            z = (value - self.x_opt) @ self.linear_transform
            z = torch.where(z * self.x_opt > 0.0, 100.0 * z, z)
            return _oscillate(torch.sum(z**2, dim=-1)) ** 0.9

        if function_id == 7:
            z_hat = (value - self.x_opt) @ self.linear_transform
            first = z_hat[:, 0]
            rounded = torch.where(
                torch.abs(z_hat) > 0.5,
                torch.floor(z_hat + 0.5),
                torch.floor(10.0 * z_hat + 0.5) / 10.0,
            )
            z = rounded @ self.rotation
            core = 0.1 * torch.maximum(
                1e-4 * torch.abs(first),
                torch.sum(self.scales * z**2, dim=-1),
            )
            return core + _boundary_penalty(value, 1.0)

        if function_id == 8:
            scale = max(1.0, math.sqrt(dimension) / 8.0)
            z = scale * (value - self.x_opt) + 1.0
            return _rosenbrock(z)

        if function_id == 9:
            return _rosenbrock(value @ self.linear_transform + 0.5)

        if function_id == 10:
            z = _oscillate((value - self.x_opt) @ self.rotation)
            return torch.sum(self.scales * z**2, dim=-1)

        if function_id == 11:
            z = _oscillate((value - self.x_opt) @ self.rotation)
            return torch.sum(z**2, dim=-1) + (1e6 - 1.0) * z[:, 0] ** 2

        if function_id == 12:
            z = (value - self.x_opt) @ self.rotation
            z = _asymmetric(z, beta=0.5) @ self.rotation
            return 1e6 * torch.sum(z**2, dim=-1) + (1.0 - 1e6) * z[:, 0] ** 2

        if function_id == 13:
            z = (value - self.x_opt) @ self.linear_transform
            return z[:, 0] ** 2 + 100.0 * torch.sqrt(torch.sum(z[:, 1:] ** 2, dim=-1))

        if function_id == 14:
            z = (value - self.x_opt) @ self.rotation
            exponents = 2.0 + 4.0 * torch.linspace(
                0.0,
                1.0,
                dimension,
                dtype=value.dtype,
                device=value.device,
            )
            return torch.sqrt(torch.sum(torch.abs(z) ** exponents, dim=-1))

        if function_id == 15:
            z = (value - self.x_opt) @ self.rotation
            z = _asymmetric(_oscillate(z), beta=0.2) @ self.linear_transform
            return 10.0 * (dimension - torch.sum(torch.cos(2.0 * math.pi * z), dim=-1)) + torch.sum(z**2, dim=-1)

        if function_id == 16:
            z = (value - self.x_opt) @ self.rotation
            z = _oscillate(z) @ self.linear_transform
            angles = 2.0 * math.pi * (z[:, :, None] + 0.5) * self.weierstrass_b[None, None, :]
            series = torch.sum(
                self.weierstrass_a[None, None, :] * torch.cos(angles),
                dim=-1,
            )
            f_zero = torch.sum(self.weierstrass_a * torch.cos(math.pi * self.weierstrass_b))
            core = 10.0 * (torch.mean(series, dim=-1) - f_zero) ** 3
            return core + _boundary_penalty(value, 10.0 / dimension)

        if function_id in (17, 18):
            z = (value - self.x_opt) @ self.rotation
            z = _asymmetric(z, beta=0.5) @ self.linear_transform
            adjacent = z[:, :-1] ** 2 + z[:, 1:] ** 2
            core = (
                torch.mean(
                    adjacent**0.25 * (torch.sin(50.0 * adjacent**0.1) ** 2 + 1.0),
                    dim=-1,
                )
                ** 2
            )
            return core + _boundary_penalty(value, 10.0)

        if function_id == 19:
            z = value @ self.linear_transform + 0.5
            pair = 100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (1.0 - z[:, :-1]) ** 2
            return 10.0 + 10.0 * torch.sum(
                pair / 4000.0 - torch.cos(pair),
                dim=-1,
            ) / (dimension - 1.0)

        if function_id == 20:
            optimum_magnitude = 2.0 * torch.abs(self.x_opt)
            z = 2.0 * torch.sign(self.x_opt) * value
            z = torch.cat(
                [
                    z[:, :1],
                    z[:, 1:] + 0.25 * (z[:, :-1] - optimum_magnitude[:-1]),
                ],
                dim=-1,
            )
            z = 100.0 * (self.scales * (z - optimum_magnitude) + optimum_magnitude)
            penalty = 0.01 * torch.sum(
                torch.clamp(torch.abs(z) - 500.0, min=0.0) ** 2,
                dim=-1,
            )
            return 0.01 * (418.9828872724339 - torch.mean(z * torch.sin(torch.sqrt(torch.abs(z))), dim=-1)) + penalty

        if function_id in (21, 22):
            rotated = value @ self.rotation
            difference = rotated[:, None, :] - self.peak_centers[None, :, :]
            response = self.peak_values[None, :] * torch.exp(
                (-0.5 / dimension)
                * torch.sum(
                    self.peak_scales[None, :, :] * difference**2,
                    dim=-1,
                )
            )
            core = _oscillate(10.0 - torch.max(response, dim=-1).values) ** 2
            return core + _boundary_penalty(value, 1.0)

        if function_id == 23:
            z = (value - self.x_opt) @ self.linear_transform
            powers = 2.0 ** torch.arange(
                1,
                33,
                dtype=value.dtype,
                device=value.device,
            )
            expanded = z[:, :, None] * powers[None, None, :]
            fractional = torch.sum(
                torch.abs(expanded - torch.floor(expanded + 0.5)) / powers[None, None, :],
                dim=-1,
            )
            coordinate = (
                1.0
                + torch.arange(
                    1,
                    dimension + 1,
                    dtype=value.dtype,
                    device=value.device,
                )[None, :]
                * fractional
            )
            scale = 10.0 / dimension**2
            core = scale * torch.prod(coordinate, dim=-1) ** (10.0 / dimension**1.2) - scale
            return core + _boundary_penalty(value, 1.0)

        # F24: Lunacek bi-Rastrigin.
        x_hat = 2.0 * torch.sign(self.x_opt) * value
        mu_one = 2.5
        s = 1.0 - 0.5 / (math.sqrt(dimension + 20.0) - 4.1)
        mu_two = -math.sqrt((mu_one**2 - 1.0) / s)
        basin = torch.minimum(
            torch.sum((x_hat - mu_one) ** 2, dim=-1),
            dimension + s * torch.sum((x_hat - mu_two) ** 2, dim=-1),
        )
        z = (x_hat - mu_one) @ self.linear_transform
        rastrigin = 10.0 * (dimension - torch.sum(torch.cos(2.0 * math.pi * z), dim=-1))
        return basin + rastrigin + _boundary_penalty(value, 1e4)
