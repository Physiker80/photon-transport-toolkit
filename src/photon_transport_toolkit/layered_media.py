"""
Monte Carlo photon transport in a stack of homogeneous layers.

This module generalises :mod:`photon_transport_toolkit.monte_carlo` from a single
homogeneous slab to an arbitrary stack of layers, each with its own
absorption coefficient, scattering coefficient, anisotropy factor, and
refractive index. It follows the multi-layer formulation of Wang,
Jacques & Zheng (1995), which is why the single-layer module already
tracked the free path in optical-depth units (tau) rather than physical
distance when crossing a boundary: that choice generalises directly to
the case where mu_t differs across the boundary, which is exactly the
situation a single homogeneous layer never exercises.

The dimensionless free path tau is sampled once per flight (i.e. once
per scattering event, not once per layer). As the photon crosses each
internal layer boundary, the *remaining* tau is carried over and
consumed at the new layer's mu_t — physically correct because each
layer's step is drawn from the same underlying Poisson process, only
with a different rate.

Reuses the private helpers from photon_transport_toolkit.monte_carlo (Fresnel
reflectance, Henyey-Greenstein sampling, direction rotation) rather
than duplicating them, so any future correction to those functions
applies to both modules automatically.

Author: Noureddin Sedki
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from photon_transport_toolkit.monte_carlo import (
    _fresnel_reflectance,
    _refract_direction,
    _sample_henyey_greenstein,
    _scatter_direction,
)

__all__ = ["Layer", "LayeredMedium", "LayeredResult", "simulate_layered_medium"]


@dataclass(frozen=True)
class Layer:
    """A single homogeneous layer within a stack.

    Parameters
    ----------
    mu_a, mu_s : float
        Absorption and scattering coefficients [1/mm].
    g : float
        Henyey-Greenstein anisotropy factor.
    thickness : float
        Layer thickness [mm].
    n : float
        Refractive index of the layer.
    """

    mu_a: float
    mu_s: float
    g: float
    thickness: float
    n: float = 1.4

    def __post_init__(self) -> None:
        if self.mu_a < 0 or self.mu_s < 0:
            raise ValueError("mu_a and mu_s must be non-negative.")
        if self.mu_a == 0 and self.mu_s == 0:
            raise ValueError("At least one of mu_a, mu_s must be positive.")
        if not -1.0 < self.g < 1.0:
            raise ValueError("g must satisfy -1 < g < 1.")
        if self.thickness <= 0:
            raise ValueError("thickness must be positive.")
        if self.n <= 0:
            raise ValueError("n must be positive.")

    @property
    def mu_t(self) -> float:
        return self.mu_a + self.mu_s

    @property
    def albedo(self) -> float:
        return self.mu_s / self.mu_t


@dataclass(frozen=True)
class LayeredMedium:
    """An ordered stack of layers, illuminated from the top (layer 0 side).

    Parameters
    ----------
    layers : list[Layer]
        Layers in order from the illuminated (top) surface to the
        far (bottom) surface.
    n_outside_top, n_outside_bottom : float
        Refractive index of the surrounding medium above and below
        the stack (typically both 1.0 for air).
    """

    layers: list[Layer]
    n_outside_top: float = 1.0
    n_outside_bottom: float = 1.0

    def __post_init__(self) -> None:
        if len(self.layers) == 0:
            raise ValueError("A LayeredMedium needs at least one layer.")

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    @property
    def boundaries(self) -> np.ndarray:
        """Cumulative depth of each interface, boundaries[0] = 0."""
        z = np.zeros(len(self.layers) + 1)
        z[1:] = np.cumsum([layer.thickness for layer in self.layers])
        return z


class LayeredResult(NamedTuple):
    specular_reflectance: float
    diffuse_reflectance: float
    transmittance: float
    absorbed: float
    diffuse_reflectance_stderr: float
    transmittance_stderr: float
    absorbed_stderr: float
    n_photons: int

    @property
    def energy_balance(self) -> float:
        return (
            self.specular_reflectance
            + self.diffuse_reflectance
            + self.transmittance
            + self.absorbed
        )


def _trace_one_photon_layered(
    medium: LayeredMedium,
    boundaries: np.ndarray,
    rng: np.random.Generator,
    weight_threshold: float,
    roulette_survival: int,
) -> tuple[float, float, float]:
    """Trace a single photon through the layered stack.

    Returns (diffuse_reflected, transmitted, absorbed) weights, given a
    packet that has already lost its specular reflection at entry.
    """
    layers = medium.layers
    n_layers = len(layers)

    z = 0.0
    ux = uy = 0.0
    uz = 1.0
    i = 0  # current layer index

    r_specular = _fresnel_reflectance(1.0, medium.n_outside_top, layers[0].n)
    weight = 1.0 - r_specular

    reflected = transmitted = absorbed = 0.0

    while weight > 0.0:
        tau = -np.log(rng.random())

        while True:
            mu_t_i = layers[i].mu_t
            step = tau / mu_t_i

            if uz > 0.0:
                dist_boundary = (boundaries[i + 1] - z) / uz
            elif uz < 0.0:
                dist_boundary = (boundaries[i] - z) / uz
            else:
                dist_boundary = np.inf

            if step < dist_boundary:
                z += step * uz
                break

            # Advance exactly to the interface and consume that much tau.
            z = boundaries[i + 1] if uz > 0.0 else boundaries[i]
            tau -= dist_boundary * mu_t_i

            n_current = layers[i].n
            if uz > 0.0:
                if i + 1 == n_layers:
                    n_next, next_layer = medium.n_outside_bottom, None
                else:
                    n_next, next_layer = layers[i + 1].n, i + 1
            else:
                if i == 0:
                    n_next, next_layer = medium.n_outside_top, None
                else:
                    n_next, next_layer = layers[i - 1].n, i - 1

            r_boundary = _fresnel_reflectance(abs(uz), n_current, n_next)
            if rng.random() < r_boundary:
                uz = -uz  # internally reflected; stay in layer i
                continue

            if next_layer is None:
                if uz > 0.0:
                    transmitted += weight
                else:
                    reflected += weight
                return reflected, transmitted, absorbed

            ux, uy, uz = _refract_direction(ux, uy, uz, n_current, n_next)
            i = next_layer  # transmitted into the neighbouring layer

        # Absorption + scattering event, inside layer i.
        d_weight = weight * (1.0 - layers[i].albedo)
        absorbed += d_weight
        weight -= d_weight

        cos_theta = _sample_henyey_greenstein(layers[i].g, rng)
        phi = 2.0 * np.pi * rng.random()
        ux, uy, uz = _scatter_direction(ux, uy, uz, cos_theta, phi)

        if weight < weight_threshold:
            if rng.random() <= 1.0 / roulette_survival:
                weight *= roulette_survival
            else:
                absorbed += weight
                return reflected, transmitted, absorbed

    return reflected, transmitted, absorbed


def simulate_layered_medium(
    medium: LayeredMedium,
    n_photons: int = 100_000,
    seed: int | None = 0,
    n_batches: int = 10,
    weight_threshold: float = 1e-4,
    roulette_survival: int = 10,
) -> LayeredResult:
    """Run a Monte Carlo simulation of normally incident light on a layered medium.

    Same batching/uncertainty-estimation strategy as
    :func:`photon_transport_toolkit.monte_carlo.simulate_slab`, generalised to an
    arbitrary stack of layers.
    """
    if n_photons < n_batches:
        raise ValueError("n_photons must be at least n_batches.")
    if n_batches < 2:
        raise ValueError("At least two batches are required to estimate an uncertainty.")

    rng = np.random.default_rng(seed)
    boundaries = medium.boundaries
    per_batch = n_photons // n_batches
    total = per_batch * n_batches

    r_specular = _fresnel_reflectance(1.0, medium.n_outside_top, medium.layers[0].n)

    batch_r = np.empty(n_batches)
    batch_t = np.empty(n_batches)
    batch_a = np.empty(n_batches)

    for b in range(n_batches):
        acc_r = acc_t = acc_a = 0.0
        for _ in range(per_batch):
            r, t, a = _trace_one_photon_layered(
                medium, boundaries, rng, weight_threshold, roulette_survival
            )
            acc_r += r
            acc_t += t
            acc_a += a
        batch_r[b] = acc_r / per_batch
        batch_t[b] = acc_t / per_batch
        batch_a[b] = acc_a / per_batch

    def _mean_and_stderr(values: np.ndarray) -> tuple[float, float]:
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))

    r_mean, r_err = _mean_and_stderr(batch_r)
    t_mean, t_err = _mean_and_stderr(batch_t)
    a_mean, a_err = _mean_and_stderr(batch_a)

    return LayeredResult(
        specular_reflectance=r_specular,
        diffuse_reflectance=r_mean,
        transmittance=t_mean,
        absorbed=a_mean,
        diffuse_reflectance_stderr=r_err,
        transmittance_stderr=t_err,
        absorbed_stderr=a_err,
        n_photons=total,
    )
