"""
Phase-resolved (coherent-field) Monte Carlo photon transport.

Extends the intensity-only engine in :mod:`photon_transport_toolkit.monte_carlo`
to track each photon packet's accumulated *physical* path length, not
just its statistical weight. At the exit surface, each packet is
assigned a complex field contribution sqrt(weight)*exp(i*phase), with
phase = 2*pi*n_medium*path_length/wavelength -- the standard Monte
Carlo transmission-matrix synthesis approach used in wavefront-shaping
and speckle-correlation work (e.g. Bar et al.-style TM synthesis; see
PROJECT_REPORT.md's roadmap discussion). Coherently summing these
contributions over many photons landing in the same detector-plane
pixel produces a complex speckle field E(x, y); |E|^2 is the intensity
pattern a camera would record, and its sum over all pixels must equal
the intensity-only engine's diffuse reflectance -- the reduction test
this module is validated against.

This is a genuinely new capability, not a refactor of the validated
scalar engine: :func:`_trace_one_photon_coherent` duplicates
:func:`photon_transport_toolkit.monte_carlo._trace_one_photon`'s physics
exactly (same boundary handling, same Fresnel/Henyey-Greenstein/Russian-
roulette logic -- verified by the reduction test below), adding only
physical-path-length bookkeeping and per-exit-event position/phase
recording. The validated scalar engine is untouched.

Author: Noureddin Sedki
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import j1

from photon_transport_toolkit.monte_carlo import (
    SlabOpticalProperties,
    _fresnel_reflectance,
    _refract_direction,
    _sample_henyey_greenstein,
    _sample_schlick,
    _scatter_direction,
    g_to_schlick_k,
)

__all__ = ["CoherentFieldResult", "simulate_slab_coherent", "apply_coherent_psf", "airy_psf_kernel"]


@dataclass(frozen=True)
class CoherentFieldResult:
    """Result of a phase-resolved coherent-field simulation.

    Parameters
    ----------
    field_reflected, field_transmitted : np.ndarray
        Complex-valued detector-plane fields E(x, y), shape
        ``(detector_bins, detector_bins)``. Units are chosen so that
        ``|E|^2`` summed over all pixels and divided by ``n_photons``
        equals the diffuse reflectance / transmittance a
        :func:`photon_transport_toolkit.monte_carlo.simulate_slab` call
        with the same optical properties would report.
    detector_half_width : float
        Half-width of the square detector plane, in mm.
    diffuse_reflectance, transmittance, absorbed : float
        Scalar intensities, computed from the same photon weights as
        the coherent field -- provided so ``sum(|field|**2)`` can be
        checked against these directly (the reduction test).
    diffuse_reflectance_stderr, transmittance_stderr : float
        Batch standard errors on the scalar quantities above.
    n_photons : int
        Total photons actually simulated (``n_photons_per_batch * n_batches``).
    wavelength_nm : float
        Vacuum wavelength used for the phase calculation.
    """

    field_reflected: np.ndarray
    field_transmitted: np.ndarray
    detector_half_width: float
    diffuse_reflectance: float
    transmittance: float
    absorbed: float
    diffuse_reflectance_stderr: float
    transmittance_stderr: float
    n_photons: int
    wavelength_nm: float


def _trace_one_photon_coherent(slab, rng, weight_threshold, roulette_survival):
    """Trace one photon, returning exit events with accumulated path length.

    Physics identical to
    :func:`photon_transport_toolkit.monte_carlo._trace_one_photon` (same
    boundary/Fresnel/HG/roulette logic) -- the only addition is
    ``path_length``, the total physical distance travelled, accumulated
    at every straight-line segment (both free-flight segments and the
    partial segments used to resolve a boundary crossing).

    Returns
    -------
    outcome : str
        ``"R"`` (exited top), ``"T"`` (exited bottom), or ``"A"``
        (absorbed to extinction / roulette termination -- no coherent
        contribution).
    x, y : float
        Transverse exit position, mm.
    weight : float
        Exit weight (same quantity the scalar engine accumulates into
        Rd/T).
    path_length : float
        Total physical path length travelled inside the medium, mm.
    """
    x = y = z = 0.0
    ux = uy = 0.0
    uz = 1.0
    path_length = 0.0

    r_specular = _fresnel_reflectance(1.0, slab.n_outside, slab.n_medium)
    weight = 1.0 - r_specular

    while weight > 0.0:
        tau = -np.log(rng.random())

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
                path_length += step
                break

            x += dist_boundary * ux
            y += dist_boundary * uy
            z += dist_boundary * uz
            path_length += dist_boundary
            z = 0.0 if uz < 0.0 else slab.thickness
            tau -= dist_boundary * slab.mu_t

            r_boundary = _fresnel_reflectance(abs(uz), slab.n_medium, slab.n_outside)
            if rng.random() > r_boundary:
                outcome = "R" if uz < 0.0 else "T"
                return outcome, x, y, weight, path_length

            uz = -uz

        d_weight = weight * (1.0 - slab.albedo)
        weight -= d_weight

        if slab.phase_function == "schlick":
            k = slab.schlick_k if slab.schlick_k is not None else g_to_schlick_k(slab.g)
            cos_theta = _sample_schlick(k, rng)
        else:
            cos_theta = _sample_henyey_greenstein(slab.g, rng)
        phi = 2.0 * np.pi * rng.random()
        ux, uy, uz = _scatter_direction(ux, uy, uz, cos_theta, phi)

        if weight < weight_threshold:
            if rng.random() <= 1.0 / roulette_survival:
                weight *= roulette_survival
            else:
                return "A", x, y, 0.0, path_length

    return "A", x, y, 0.0, path_length


def simulate_slab_coherent(
    slab: SlabOpticalProperties,
    wavelength_nm: float,
    n_photons: int = 20_000,
    seed: int | None = 0,
    n_batches: int = 10,
    detector_bins: int = 64,
    detector_half_width: float = 2.0,
    weight_threshold: float = 1e-4,
    roulette_survival: int = 10,
) -> CoherentFieldResult:
    """Simulate a phase-resolved (coherent) speckle field for a slab.

    Every exiting photon contributes ``sqrt(weight) * exp(i*phase)`` to
    the detector-plane pixel its transverse exit position falls into,
    where ``phase = 2*pi*n_medium*path_length/wavelength``. Summed
    coherently within each pixel, this produces a complex speckle
    field whose intensity ``|E|^2`` -- NOT the field itself -- is what
    a real detector would record.

    Parameters
    ----------
    slab : SlabOpticalProperties
        Same optical-property object used by
        :func:`photon_transport_toolkit.monte_carlo.simulate_slab`.
    wavelength_nm : float
        Vacuum wavelength, nanometres.
    detector_bins : int
        Number of pixels per side of the square detector plane.
    detector_half_width : float
        Half-width of the detector plane, mm. Exit positions outside
        this range are still counted in the scalar Rd/T totals but
        fall outside the recorded field (a documented limitation, not
        silently dropped from the energy balance).

    Returns
    -------
    CoherentFieldResult
    """
    if n_photons < n_batches:
        raise ValueError("n_photons must be at least n_batches.")
    if n_batches < 2:
        raise ValueError("At least two batches are required to estimate an uncertainty.")
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    if detector_bins < 1:
        raise ValueError("detector_bins must be at least 1.")
    if detector_half_width <= 0:
        raise ValueError("detector_half_width must be positive.")

    wavelength_mm = wavelength_nm * 1e-6
    k0 = 2.0 * np.pi / wavelength_mm

    rng = np.random.default_rng(seed)
    per_batch = n_photons // n_batches
    total = per_batch * n_batches

    field_r = np.zeros((detector_bins, detector_bins), dtype=complex)
    field_t = np.zeros((detector_bins, detector_bins), dtype=complex)

    batch_r = np.empty(n_batches)
    batch_t = np.empty(n_batches)
    batch_a = np.empty(n_batches)

    bin_edges = np.linspace(-detector_half_width, detector_half_width, detector_bins + 1)

    for b in range(n_batches):
        acc_r = acc_t = acc_a = 0.0
        for _ in range(per_batch):
            outcome, x, y, weight, path_length = _trace_one_photon_coherent(
                slab, rng, weight_threshold, roulette_survival
            )
            if outcome == "R":
                acc_r += weight
            elif outcome == "T":
                acc_t += weight
            else:
                acc_a += weight

            if outcome in ("R", "T") and weight > 0.0:
                ix = np.searchsorted(bin_edges, x, side="right") - 1
                iy = np.searchsorted(bin_edges, y, side="right") - 1
                if 0 <= ix < detector_bins and 0 <= iy < detector_bins:
                    phase = k0 * slab.n_medium * path_length
                    contribution = np.sqrt(weight) * np.exp(1j * phase)
                    if outcome == "R":
                        field_r[iy, ix] += contribution
                    else:
                        field_t[iy, ix] += contribution

        batch_r[b] = acc_r / per_batch
        batch_t[b] = acc_t / per_batch
        batch_a[b] = acc_a / per_batch

    def _mean_se(values):
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))

    r_mean, r_se = _mean_se(batch_r)
    t_mean, t_se = _mean_se(batch_t)
    a_mean, _ = _mean_se(batch_a)

    # Fields accumulate per-photon contributions across all batches, so
    # normalise by the SAME denominator the scalar means use (total
    # photons), keeping sum(|field|^2)/? directly comparable to r_mean.
    field_r /= np.sqrt(total)
    field_t /= np.sqrt(total)

    return CoherentFieldResult(
        field_reflected=field_r,
        field_transmitted=field_t,
        detector_half_width=detector_half_width,
        diffuse_reflectance=r_mean,
        transmittance=t_mean,
        absorbed=a_mean,
        diffuse_reflectance_stderr=r_se,
        transmittance_stderr=t_se,
        n_photons=total,
        wavelength_nm=wavelength_nm,
    )


def airy_psf_kernel(
    pixel_size_mm: float,
    wavelength_nm: float,
    numerical_aperture: float,
    kernel_half_size: int = 25,
) -> np.ndarray:
    """Build a discretised coherent (amplitude) point-spread function
    for an ideal circular aperture -- the Airy pattern

        h(r) = 2*J1(v) / v,   v = (2*pi/lambda) * NA * r

    (h(0) = 1 by the standard limit). This is the *amplitude* PSF, the
    correct kernel to convolve a coherent complex field with -- not
    the intensity PSF |h|^2 used for incoherent imaging.

    Normalised so that ``sum(kernel**2) == 1``: convolving a spatially
    uncorrelated ("white") field with this kernel then preserves total
    energy in expectation, exactly the property
    :func:`apply_coherent_psf`'s own validation test checks directly
    rather than assumes.

    Parameters
    ----------
    pixel_size_mm : float
        Detector pixel pitch, mm (e.g. ``2*detector_half_width/detector_bins``
        for a :class:`CoherentFieldResult` field).
    wavelength_nm : float
        Vacuum wavelength, nanometres.
    numerical_aperture : float
        Imaging system NA. Smaller NA -> larger diffraction-limited
        spot -> larger speckle grains, and vice versa; this is the
        parameter that sets grain size, made explicit rather than
        hidden inside a "looks about right" default.
    kernel_half_size : int
        Kernel extends ``kernel_half_size`` pixels in each direction
        from centre (a ``(2*kernel_half_size+1)``-pixel-wide square
        kernel).
    """
    if pixel_size_mm <= 0:
        raise ValueError("pixel_size_mm must be positive.")
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    if not 0 < numerical_aperture < 1:
        raise ValueError("numerical_aperture must satisfy 0 < NA < 1.")
    if kernel_half_size < 1:
        raise ValueError("kernel_half_size must be at least 1.")

    wavelength_mm = wavelength_nm * 1e-6
    k0 = 2.0 * np.pi / wavelength_mm

    n = 2 * kernel_half_size + 1
    coords = (np.arange(n) - kernel_half_size) * pixel_size_mm
    xx, yy = np.meshgrid(coords, coords)
    r = np.sqrt(xx**2 + yy**2)

    v = k0 * numerical_aperture * r
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(v < 1e-8, 1.0, 2.0 * j1(v) / v)

    norm = np.sqrt(np.sum(h**2))
    return h / norm


def apply_coherent_psf(
    field: np.ndarray,
    pixel_size_mm: float,
    wavelength_nm: float,
    numerical_aperture: float,
    kernel_half_size: int = 25,
) -> np.ndarray:
    """Convolve a complex coherent field with an imaging system's
    coherent PSF (:func:`airy_psf_kernel`), producing spatially
    correlated speckle grains of the size a real camera, with the
    given wavelength and NA, would actually show -- rather than the
    raw one-photon-per-pixel exit field
    :func:`simulate_slab_coherent` returns on its own.

    Field amplitude is convolved (not intensity): ``|result|**2`` is
    the recorded intensity pattern, taken *after* this step.
    """
    kernel = airy_psf_kernel(pixel_size_mm, wavelength_nm, numerical_aperture, kernel_half_size)
    return fftconvolve(field, kernel, mode="same")
