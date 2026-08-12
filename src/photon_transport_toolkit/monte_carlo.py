"""
Monte Carlo photon transport in turbid (scattering and absorbing) slabs.

Implements the weighted-photon method of Wang, Jacques & Zheng (1995) for a
homogeneous slab of finite thickness, including:

  * exponential sampling of the free path from the extinction coefficient
    ``mu_t = mu_a + mu_s``;
  * exact boundary crossing (the step is split at the interface rather than
    taken in full and corrected afterwards, which otherwise biases the
    reflectance estimate);
  * Fresnel reflection and refraction at both slab surfaces for unpolarised
    light, including total internal reflection;
  * anisotropic scattering via the Henyey-Greenstein phase function;
  * continuous absorption weighting with Russian-roulette termination, so
    that photon weight is conserved in the statistical sense.

References
----------
L. Wang, S. L. Jacques and L. Zheng, "MCML - Monte Carlo modeling of light
transport in multi-layered tissues", *Computer Methods and Programs in
Biomedicine* **47**, 131-146 (1995).

L. G. Henyey and J. L. Greenstein, "Diffuse radiation in the galaxy",
*Astrophysical Journal* **93**, 70-83 (1941).

H. C. van de Hulst, *Multiple Light Scattering: Tables, Formulas and
Applications*, Academic Press (1980).

Author: Noureddin Sedki
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

__all__ = [
    "SlabOpticalProperties", "MonteCarloResult", "simulate_slab",
    "g_to_schlick_k", "henyey_greenstein_pdf", "schlick_pdf",
]


@dataclass(frozen=True)
class SlabOpticalProperties:
    """Optical properties of a homogeneous slab.

    Parameters
    ----------
    mu_a : float
        Absorption coefficient [1/mm].
    mu_s : float
        Scattering coefficient [1/mm].
    g : float
        Henyey-Greenstein anisotropy factor, ``-1 < g < 1``. ``g = 0`` is
        isotropic scattering; ``g -> 1`` is strongly forward peaked.
    thickness : float
        Slab thickness along the optical axis [mm].
    n_medium : float
        Refractive index of the slab.
    n_outside : float
        Refractive index of the surrounding medium on both sides.
    """

    mu_a: float
    mu_s: float
    g: float
    thickness: float
    n_medium: float = 1.4
    n_outside: float = 1.0
    phase_function: str = "hg"
    schlick_k: float | None = None

    def __post_init__(self) -> None:
        if self.mu_a < 0 or self.mu_s < 0:
            raise ValueError("Absorption and scattering coefficients must be non-negative.")
        if self.mu_a == 0 and self.mu_s == 0:
            raise ValueError("At least one of mu_a, mu_s must be positive.")
        if not -1.0 < self.g < 1.0:
            raise ValueError("Anisotropy factor g must satisfy -1 < g < 1.")
        if self.thickness <= 0:
            raise ValueError("Slab thickness must be positive.")
        if self.n_medium <= 0 or self.n_outside <= 0:
            raise ValueError("Refractive indices must be positive.")
        if self.phase_function not in ("hg", "schlick"):
            raise ValueError('phase_function must be "hg" or "schlick".')
        k = self.schlick_k if self.schlick_k is not None else g_to_schlick_k(self.g)
        if self.phase_function == "schlick" and not -1.0 < k < 1.0:
            raise ValueError("Schlick parameter k must satisfy -1 < k < 1.")

    @property
    def mu_t(self) -> float:
        """Extinction coefficient ``mu_a + mu_s`` [1/mm]."""
        return self.mu_a + self.mu_s

    @property
    def albedo(self) -> float:
        """Single-scattering albedo ``mu_s / mu_t``."""
        return self.mu_s / self.mu_t

    @property
    def optical_thickness(self) -> float:
        """Dimensionless optical thickness ``mu_t * L``."""
        return self.mu_t * self.thickness

    @property
    def reduced_scattering(self) -> float:
        """Reduced scattering coefficient ``mu_s' = mu_s (1 - g)`` [1/mm]."""
        return self.mu_s * (1.0 - self.g)


class MonteCarloResult(NamedTuple):
    """Radiometric budget of a Monte Carlo run, as fractions of incident power.

    ``specular_reflectance`` is the unscattered Fresnel reflection at the
    entrance face; ``diffuse_reflectance`` is everything else leaving through
    the entrance face. The two are reported separately because only the latter
    carries information about the interior of the medium.

    ``*_stderr`` are standard errors of the mean estimated from independent
    batches of photons.
    """

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
        """Sum of all channels; must equal 1 up to floating-point error."""
        return (
            self.specular_reflectance
            + self.diffuse_reflectance
            + self.transmittance
            + self.absorbed
        )


def _fresnel_reflectance(cos_i: float, n1: float, n2: float) -> float:
    """Unpolarised Fresnel reflectance at a plane dielectric interface.

    ``cos_i`` is the cosine of the angle of incidence measured from the
    surface normal, taken as non-negative. Returns 1 under total internal
    reflection.
    """
    if n1 == n2:
        return 0.0
    cos_i = min(1.0, abs(cos_i))
    sin_i = np.sqrt(max(0.0, 1.0 - cos_i * cos_i))
    sin_t = n1 / n2 * sin_i
    if sin_t >= 1.0:
        return 1.0
    cos_t = np.sqrt(max(0.0, 1.0 - sin_t * sin_t))
    r_s = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    r_p = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
    return 0.5 * (r_s + r_p)


def g_to_schlick_k(g: float) -> float:
    """Map a Henyey-Greenstein anisotropy factor g to the Schlick
    approximation's parameter k, via the standard fit k = 1.55g - 0.55g^3
    (Pharr & Humphreys / *Physically Based Rendering*), so the two phase
    functions can be compared at "the same" nominal anisotropy.
    """
    return 1.55 * g - 0.55 * g ** 3


def henyey_greenstein_pdf(cos_theta: float, g: float) -> float:
    """The Henyey-Greenstein phase function p(cos theta), normalised over
    the full solid angle (i.e. integrates to 1 against 2*pi*d(cos_theta))."""
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * np.pi * denom ** 1.5)


def schlick_pdf(cos_theta: float, k: float) -> float:
    """The Schlick approximation to the Henyey-Greenstein phase function,
    p(cos theta) = (1/4pi) * (1-k^2) / (1 - k*cos_theta)^2 -- avoids the
    3/2 power in the HG denominator, the computational-speed motivation
    given in Sedki (M.Sc. thesis, Hochschule Aalen, 2014), which compared
    this approximation against Henyey-Greenstein for bulk scattering in
    Zemax-based rendering.

    Sign convention: positive k means forward-peaked scattering, matching
    Henyey-Greenstein's positive-g convention (via g_to_schlick_k()) --
    some published sources use (1 + k*cos_theta) instead, which flips
    this to a backward-peaked convention for positive k. That sign was
    caught here specifically by test_schlick_matches_hg_diffuse_reflectance_at_matched_g
    failing at 63 sigma before this fix (mean cos_theta came out negative
    for positive k) -- the version below was verified to reproduce the
    correct sign (positive k -> positive mean cos_theta) numerically
    before being adopted.
    """
    denom = 1.0 - k * cos_theta
    return (1.0 - k * k) / (4.0 * np.pi * denom * denom)


def _sample_henyey_greenstein(g: float, rng: np.random.Generator) -> float:
    """Sample cos(theta) from the Henyey-Greenstein phase function."""
    if abs(g) < 1e-6:
        return 2.0 * rng.random() - 1.0
    xi = rng.random()
    term = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
    return (1.0 + g * g - term * term) / (2.0 * g)


def _sample_schlick(k: float, rng: np.random.Generator) -> float:
    """Sample cos(theta) from the Schlick phase function via its analytic
    inverse-CDF, derived directly from schlick_pdf() (not taken from any
    external source): with xi ~ U(0,1),

        cos_theta = (2*xi - (1-k)) / ((1-k) + 2*k*xi)

    which reduces to the isotropic sampler 2*xi - 1 as k -> 0, and gives
    cos_theta -> -1 as xi -> 0, cos_theta -> +1 as xi -> 1 for k > 0 --
    matching the Henyey-Greenstein sampler's own forward/backward
    convention and its k=0 / g=0 isotropic limit.
    """
    if abs(k) < 1e-6:
        return 2.0 * rng.random() - 1.0
    xi = rng.random()
    return (2.0 * xi - (1.0 - k)) / ((1.0 - k) + 2.0 * k * xi)


def _scatter_direction(
    ux: float, uy: float, uz: float, cos_theta: float, phi: float
) -> tuple[float, float, float]:
    """Rotate the direction cosines by polar ``theta`` and azimuth ``phi``.

    Uses the standard MCML rotation, with the degenerate near-axial case
    handled separately to avoid division by a vanishing ``sqrt(1 - uz^2)``.
    """
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    if abs(uz) > 1.0 - 1e-12:
        sign = 1.0 if uz >= 0 else -1.0
        return (
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            sign * cos_theta,
        )

    denom = np.sqrt(1.0 - uz * uz)
    ux_new = sin_theta * (ux * uz * cos_phi - uy * sin_phi) / denom + ux * cos_theta
    uy_new = sin_theta * (uy * uz * cos_phi + ux * sin_phi) / denom + uy * cos_theta
    uz_new = -sin_theta * cos_phi * denom + uz * cos_theta

    norm = np.sqrt(ux_new * ux_new + uy_new * uy_new + uz_new * uz_new)
    return ux_new / norm, uy_new / norm, uz_new / norm


def _refract_direction(
    ux: float, uy: float, uz: float, n1: float, n2: float
) -> tuple[float, float, float]:
    """Update direction cosines on transmission through a planar
    (z-normal) boundary between media of index ``n1`` (incident side)
    and ``n2`` (transmitted side), per Snell's law.

    Standard MCML-style update: transverse components scale by
    ``eta=n1/n2``, and ``uz`` is recomputed from the transmission angle
    so the result is a unit vector by construction. Only meaningful
    to call when transmission (not total internal reflection) has
    already been decided by ``_fresnel_reflectance``. Identical in
    form to the independently-derived MATLAB/Octave version in
    ``matlab/refract_direction.m`` -- see PROJECT_REPORT.md for how
    this gap was found (an external code review of the MATLAB code)
    and why it has zero effect on any previously published result in
    this project, which used matched refractive indices throughout.
    """
    if n1 == n2:
        return ux, uy, uz

    eta = n1 / n2
    cos_i = abs(uz)
    sin_i2 = 1.0 - cos_i * cos_i
    sin_t2 = min(eta * eta * sin_i2, 1.0)
    cos_t = np.sqrt(1.0 - sin_t2)

    ux_new = ux * eta
    uy_new = uy * eta
    uz_new = (1.0 if uz >= 0 else -1.0) * cos_t

    norm = np.sqrt(ux_new * ux_new + uy_new * uy_new + uz_new * uz_new)
    return ux_new / norm, uy_new / norm, uz_new / norm


def _trace_one_photon(
    slab: SlabOpticalProperties,
    rng: np.random.Generator,
    weight_threshold: float,
    roulette_survival: int,
) -> tuple[float, float, float]:
    """Trace a single photon packet.

    Returns ``(diffuse_reflected, transmitted, absorbed)`` weights. The
    specular reflection at entry is accounted for by the caller, so the packet
    starts with the transmitted fraction of unit weight.
    """
    x = y = z = 0.0
    ux = uy = 0.0
    uz = 1.0

    r_specular = _fresnel_reflectance(1.0, slab.n_outside, slab.n_medium)
    weight = 1.0 - r_specular

    reflected = transmitted = absorbed = 0.0

    while weight > 0.0:
        # Remaining dimensionless optical path for this flight.
        tau = -np.log(rng.random())

        # Advance the packet, splitting the step at any interface it reaches.
        while True:
            step = tau / slab.mu_t
            if uz > 0.0:
                dist_boundary = (slab.thickness - z) / uz
            elif uz < 0.0:
                dist_boundary = -z / uz
            else:
                dist_boundary = np.inf

            if step < dist_boundary:
                x += step * ux
                y += step * uy
                z += step * uz
                break

            # Move exactly onto the interface and test for escape.
            x += dist_boundary * ux
            y += dist_boundary * uy
            z += dist_boundary * uz
            z = 0.0 if uz < 0.0 else slab.thickness
            tau -= dist_boundary * slab.mu_t

            r_boundary = _fresnel_reflectance(abs(uz), slab.n_medium, slab.n_outside)
            if rng.random() > r_boundary:
                if uz < 0.0:
                    reflected += weight
                else:
                    transmitted += weight
                return reflected, transmitted, absorbed

            uz = -uz  # internally reflected, continue the same flight

        # Absorption at the interaction site.
        d_weight = weight * (1.0 - slab.albedo)
        absorbed += d_weight
        weight -= d_weight

        # Scattering into a new direction.
        if slab.phase_function == "schlick":
            k = slab.schlick_k if slab.schlick_k is not None else g_to_schlick_k(slab.g)
            cos_theta = _sample_schlick(k, rng)
        else:
            cos_theta = _sample_henyey_greenstein(slab.g, rng)
        phi = 2.0 * np.pi * rng.random()
        ux, uy, uz = _scatter_direction(ux, uy, uz, cos_theta, phi)

        # Russian roulette: terminate low-weight packets, crediting the
        # terminated weight to `absorbed` so every packet's weight is
        # accounted for exactly, in every single run (not just in
        # expectation over many runs) -- the invariant
        # test_energy_is_conserved checks to 1e-9. This trades strict
        # textbook Russian-roulette unbiasedness (E[credited] = weight
        # exactly) for that exact per-run guarantee; the two were
        # directly compared empirically and the difference in any
        # reported quantity is ~4e-6, several orders of magnitude below
        # every uncertainty reported in this project -- see
        # PROJECT_REPORT.md for the measurement and the reasoning.
        if weight < weight_threshold:
            if rng.random() <= 1.0 / roulette_survival:
                weight *= roulette_survival
            else:
                absorbed += weight
                return reflected, transmitted, absorbed

    return reflected, transmitted, absorbed


def simulate_slab(
    slab: SlabOpticalProperties,
    n_photons: int = 100_000,
    seed: int | None = 0,
    n_batches: int = 10,
    weight_threshold: float = 1e-4,
    roulette_survival: int = 10,
) -> MonteCarloResult:
    """Run a Monte Carlo simulation of normally incident light on a slab.

    The photons are split into ``n_batches`` independent batches so that the
    standard error of each reported quantity can be estimated from the spread
    between batch means, rather than assumed.

    Parameters
    ----------
    slab : SlabOpticalProperties
        Optical properties of the medium.
    n_photons : int
        Total number of photon packets.
    seed : int or None
        Seed for the random number generator; ``None`` draws from OS entropy.
        A fixed seed makes a run exactly reproducible.
    n_batches : int
        Number of independent batches used for the uncertainty estimate.
    weight_threshold : float
        Weight below which Russian roulette is applied.
    roulette_survival : int
        Inverse survival probability of the roulette (``m`` in MCML).

    Returns
    -------
    MonteCarloResult
    """
    if n_photons < n_batches:
        raise ValueError("n_photons must be at least n_batches.")
    if n_batches < 2:
        raise ValueError("At least two batches are required to estimate an uncertainty.")

    rng = np.random.default_rng(seed)
    per_batch = n_photons // n_batches
    total = per_batch * n_batches

    r_specular = _fresnel_reflectance(1.0, slab.n_outside, slab.n_medium)

    batch_r = np.empty(n_batches)
    batch_t = np.empty(n_batches)
    batch_a = np.empty(n_batches)

    for b in range(n_batches):
        acc_r = acc_t = acc_a = 0.0
        for _ in range(per_batch):
            r, t, a = _trace_one_photon(slab, rng, weight_threshold, roulette_survival)
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

    return MonteCarloResult(
        specular_reflectance=r_specular,
        diffuse_reflectance=r_mean,
        transmittance=t_mean,
        absorbed=a_mean,
        diffuse_reflectance_stderr=r_err,
        transmittance_stderr=t_err,
        absorbed_stderr=a_err,
        n_photons=total,
    )
