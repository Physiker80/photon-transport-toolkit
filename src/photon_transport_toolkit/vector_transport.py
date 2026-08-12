"""
Vector (polarization-resolved) *and* phase-resolved Monte Carlo transport.

This module unifies the two wave properties the scalar MCML engine in
:mod:`photon_transport_toolkit.monte_carlo` discards -- phase and
polarization -- inside a single photon tracer.

Why they cannot simply be bolted together
-----------------------------------------
A Stokes vector S = [I, Q, U, V] is a *quadratic* (intensity-like)
quantity: two Stokes vectors arriving at the same detector pixel add
incoherently, so a Stokes-Mueller engine can never produce
interference or speckle no matter how carefully its optical path
length is tracked. Conversely, the scalar coherent engine in
:mod:`photon_transport_toolkit.coherent_transport` carries a single
complex amplitude and therefore cannot represent a partially polarized
field at all.

The only state that carries both is the **Jones vector** -- two complex
amplitudes (E1, E2) in a transverse basis -- together with the
accumulated optical path length. Stokes-Mueller is then recovered as
the incoherent limit:

    I = |E1|^2 + |E2|^2      Q = |E1|^2 - |E2|^2
    U = 2 Re(E1 E2*)         V = -2 Im(E1 E2*)

This module implements *both* formulations independently
(:func:`_trace_one_photon_jones` from amplitude scattering matrices,
:func:`_trace_one_photon_stokes` from 4x4 Mueller matrices) and the
test suite requires them to agree statistically. That is the same
cross-validation strategy this project used for the MATLAB/Octave
re-derivation: two derivations that share no algebra, checked against
each other, rather than one derivation checked against itself.

Reference frame convention
--------------------------
Rather than the meridian-plane convention (which needs spherical
trigonometry for the second rotation angle and is a well-known source
of sign errors), each photon carries an explicit right-handed
orthonormal triad ``(e1, e2, u)`` with ``e1 x e2 = u``. The Jones /
Stokes components are always expressed in ``(e1, e2)``. Scattering
rotates the triad about ``u`` by the azimuth ``psi``, applies the
scattering matrix, then tilts the triad about ``e2`` by the polar
angle ``theta``. Orthonormality is an invariant that can be asserted
at any step -- and is, in the test suite.

Scattering model
----------------
Rayleigh (dipole) scattering, whose amplitude scattering matrix in the
scattering plane is ``diag(S2, S1) = diag(cos(theta), 1)``. Angles are
sampled by rejection from the *polarization-dependent* phase function
p(theta, psi) proportional to the post-scattering intensity
``|J R(psi) E|^2``; the state is then renormalized to its
pre-scattering intensity. Because the sampling density is proportional
to exactly the quantity that is divided out, the estimator is unbiased
*and* intensity is conserved to machine precision -- preserving this
project's exact energy-conservation invariant (see PROJECT_REPORT.md,
Russian-roulette discussion) rather than trading it away.

Note that this differs from a common shortcut seen in tutorial code:
sampling ``cos(theta)`` isotropically and then multiplying the Stokes
vector by the Mueller matrix *without* renormalizing. That silently
destroys a fraction ``1 - <M11> = 1/3`` of the packet's energy at every
Rayleigh scattering event. The energy-conservation test in
``tests/test_vector_transport.py`` fails loudly on that variant.

Boundaries
----------
Fresnel coefficients are applied as polarization-dependent complex
*amplitude* coefficients (r_s, r_p) in the local s/p basis, including
the differential phase retardance under total internal reflection --
the Fresnel-rhomb effect, which converts linear to elliptical
polarization on internal bounces and has no counterpart in the
unpolarized-average Fresnel reflectance the scalar engine uses.

Author: Noureddin Sedki
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from math import cos as _cos, exp as _exp, log as _log, sin as _sin, sqrt as _sqrt

from photon_transport_toolkit.monte_carlo import SlabOpticalProperties

__all__ = [
    "VectorFieldResult",
    "simulate_slab_vector",
    "rayleigh_mueller_matrix",
    "rayleigh_amplitude_matrix",
    "jones_to_stokes",
    "degree_of_polarization",
    "speckle_contrast",
    "depolarization_ladder",
]


# --------------------------------------------------------------------------
# Scattering matrices
# --------------------------------------------------------------------------


def rayleigh_amplitude_matrix(cos_theta: float) -> tuple[float, float]:
    """Rayleigh amplitude scattering elements ``(S2, S1)``.

    ``S2 = cos(theta)`` acts on the component parallel to the
    scattering plane, ``S1 = 1`` on the perpendicular component. The
    ``cos(theta)`` factor is the dipole radiation pattern: a dipole
    does not radiate along its own axis, which is precisely why
    Rayleigh scattering at 90 degrees is fully linearly polarized.
    """
    return float(cos_theta), 1.0


def rayleigh_mueller_matrix(cos_theta: float) -> np.ndarray:
    """4x4 Rayleigh Mueller matrix, normalized so ``M11(0) = 1``.

    Written from the standard Stokes-Mueller formulation, *not*
    derived from :func:`rayleigh_amplitude_matrix` -- the two are
    required to agree by
    ``test_mueller_matrix_matches_amplitude_matrix``, which is the
    point of implementing them separately.
    """
    c = float(cos_theta)
    c2 = c * c
    m11 = 0.5 * (1.0 + c2)
    m12 = 0.5 * (c2 - 1.0)
    return np.array(
        [
            [m11, m12, 0.0, 0.0],
            [m12, m11, 0.0, 0.0],
            [0.0, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, c],
        ]
    )


def jones_to_stokes(e1: complex, e2: complex) -> np.ndarray:
    """Convert a Jones vector to its Stokes vector.

    Sign convention for V follows ``V = -2 Im(E1 E2*)``; the Rayleigh
    Mueller matrix's ``m44 = cos(theta)`` is invariant under the
    opposite choice, so no result in this module depends on it, but it
    is fixed here so that reported V values are reproducible.
    """
    a = e1 * np.conj(e1)
    b = e2 * np.conj(e2)
    cross = e1 * np.conj(e2)
    return np.array(
        [
            float(a.real + b.real),
            float(a.real - b.real),
            float(2.0 * cross.real),
            float(-2.0 * cross.imag),
        ]
    )


def degree_of_polarization(stokes: np.ndarray) -> float:
    """``sqrt(Q^2 + U^2 + V^2) / I``, the degree of polarization."""
    stokes = np.asarray(stokes, dtype=float)
    intensity = stokes[0]
    if intensity <= 0.0:
        return 0.0
    return float(np.sqrt(stokes[1] ** 2 + stokes[2] ** 2 + stokes[3] ** 2) / intensity)


def speckle_contrast(intensity: np.ndarray) -> float:
    """Speckle contrast ``C = std(I) / mean(I)`` over the given pixels.

    For a fully developed speckle pattern in a *single* polarization
    channel the circular-Gaussian field statistics give C = 1. For
    detection with no analyzer, summing two uncorrelated orthogonal
    channels, Goodman's result is ``C = sqrt((1 + P^2)/2)`` with P the
    degree of polarization -- the relation
    ``examples/polarized_speckle_comparison.py`` tests this engine
    against.
    """
    intensity = np.asarray(intensity, dtype=float).ravel()
    mean = intensity.mean()
    if mean <= 0.0:
        return 0.0
    return float(intensity.std(ddof=1) / mean)


# --------------------------------------------------------------------------
# Local reference-frame algebra
# --------------------------------------------------------------------------


def _rotate_frame_about_u(e1, e2, cos_psi, sin_psi):
    """Rotate the transverse basis vectors by psi about the propagation axis."""
    n1 = (
        e1[0] * cos_psi + e2[0] * sin_psi,
        e1[1] * cos_psi + e2[1] * sin_psi,
        e1[2] * cos_psi + e2[2] * sin_psi,
    )
    n2 = (
        -e1[0] * sin_psi + e2[0] * cos_psi,
        -e1[1] * sin_psi + e2[1] * cos_psi,
        -e1[2] * sin_psi + e2[2] * cos_psi,
    )
    return n1, n2


def _tilt_frame_about_e2(e1, u, cos_theta, sin_theta):
    """Tilt ``u`` toward ``e1`` by theta, keeping ``e2`` fixed.

    New propagation direction ``u' = cos(theta) u + sin(theta) e1``;
    new parallel basis vector ``e1' = -sin(theta) u + cos(theta) e1``,
    which keeps the triad orthonormal and right-handed by construction.
    """
    u_new = (
        cos_theta * u[0] + sin_theta * e1[0],
        cos_theta * u[1] + sin_theta * e1[1],
        cos_theta * u[2] + sin_theta * e1[2],
    )
    e1_new = (
        -sin_theta * u[0] + cos_theta * e1[0],
        -sin_theta * u[1] + cos_theta * e1[1],
        -sin_theta * u[2] + cos_theta * e1[2],
    )
    return e1_new, u_new


def _renormalize_frame(e1, e2, u):
    """Re-orthonormalize the triad, suppressing accumulated round-off.

    Written with plain floats rather than NumPy arrays: this runs once
    per scattering event and a nearly conservative Rayleigh medium
    takes hundreds of those per photon, so array-allocation overhead
    here dominated the whole simulation before it was inlined.
    """
    ux, uy, uz = u
    inv = 1.0 / _sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux * inv, uy * inv, uz * inv
    ax, ay, az = e1
    dot = ax * ux + ay * uy + az * uz
    ax -= dot * ux
    ay -= dot * uy
    az -= dot * uz
    inv = 1.0 / _sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax * inv, ay * inv, az * inv
    return (ax, ay, az), (uy * az - uz * ay, uz * ax - ux * az, ux * ay - uy * ax), (ux, uy, uz)


def _sp_basis(u):
    """Return the s-direction for a plane boundary with normal ``z``.

    ``s`` is perpendicular to the plane of incidence, i.e. along
    ``u x z``. At normal incidence the plane of incidence is undefined
    and ``None`` is returned; the caller then leaves the basis alone,
    which is correct because r_s = r_p there and no rotation is needed.
    """
    sx, sy = u[1], -u[0]  # u x z  =  (uy, -ux, 0)
    norm = np.hypot(sx, sy)
    if norm < 1e-12:
        return None
    return (sx / norm, sy / norm, 0.0)


def _rotation_to(e1, e2, u, target_e2):
    """Angle (cos, sin) rotating the basis about ``u`` so ``e2 -> target_e2``."""
    cos_a = e2[0] * target_e2[0] + e2[1] * target_e2[1] + e2[2] * target_e2[2]
    # sin from the triple product (e2 x target) . u
    cx = e2[1] * target_e2[2] - e2[2] * target_e2[1]
    cy = e2[2] * target_e2[0] - e2[0] * target_e2[2]
    cz = e2[0] * target_e2[1] - e2[1] * target_e2[0]
    sin_a = cx * u[0] + cy * u[1] + cz * u[2]
    norm = np.hypot(cos_a, sin_a)
    if norm < 1e-12:
        return 1.0, 0.0
    return cos_a / norm, sin_a / norm


def _fresnel_amplitudes(cos_i: float, n1: float, n2: float):
    """Complex Fresnel amplitude reflection coefficients ``(r_s, r_p)``.

    Under total internal reflection ``cos_t`` becomes imaginary and the
    coefficients acquire unit modulus with *different* phases -- the
    retardance a Fresnel rhomb exploits. Returning them as complex
    numbers rather than as an unpolarized intensity average is the
    whole reason internal bounces can convert linear to elliptical
    polarization in this engine.
    """
    if n1 == n2:
        return 0.0 + 0.0j, 0.0 + 0.0j
    cos_i = min(1.0, abs(cos_i))
    sin_i2 = max(0.0, 1.0 - cos_i * cos_i)
    sin_t2 = (n1 / n2) ** 2 * sin_i2
    if sin_t2 >= 1.0:
        cos_t = 1j * np.sqrt(sin_t2 - 1.0)
    else:
        cos_t = complex(np.sqrt(1.0 - sin_t2), 0.0)
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return r_s, r_p


def _refract_direction_vec(u, n1, n2):
    """Snell refraction at a z-normal boundary, returned as a unit vector."""
    if n1 == n2:
        return u
    eta = n1 / n2
    cos_i = abs(u[2])
    sin_t2 = min(eta * eta * (1.0 - cos_i * cos_i), 1.0)
    cos_t = np.sqrt(max(0.0, 1.0 - sin_t2))
    ux, uy = u[0] * eta, u[1] * eta
    uz = cos_t if u[2] >= 0 else -cos_t
    norm = np.sqrt(ux * ux + uy * uy + uz * uz)
    return (ux / norm, uy / norm, uz / norm)


def _coherency_from_jones(j1, j2):
    """2x2 coherency matrix of a Jones vector, in its own (e1, e2) basis."""
    return np.array([[j1 * np.conj(j1), j1 * np.conj(j2)],
                     [j2 * np.conj(j1), j2 * np.conj(j2)]], dtype=complex)


def _coherency_from_stokes(stokes):
    """2x2 coherency matrix of a (possibly partially polarized) Stokes vector."""
    i, q, u_, v = stokes
    return 0.5 * np.array([[i + q, u_ - 1j * v],
                           [u_ + 1j * v, i - q]], dtype=complex)


def _project_to_lab(coherency, e1, e2):
    """Project a coherency matrix onto the laboratory x and y analyzer axes.

    An ideal linear analyzer along ``x`` transmits ``a_x^dagger J a_x``
    with ``a_x = (e1.x, e2.x)``. The projection is *not* a rotation: an
    obliquely exiting ray has a field component along ``z`` that no
    analyzer in the xy plane can detect, so ``I_x + I_y <= I``. That
    obliquity loss is physical for an analyzer-and-camera measurement
    and is therefore applied only to detector-plane quantities -- the
    radiometric R/T/A budget uses the packet's full intensity, so
    energy conservation is unaffected.
    """
    a = np.array([[e1[0], e2[0]], [e1[1], e2[1]]], dtype=complex)
    lab = a @ coherency @ a.conj().T
    ixx = float(lab[0, 0].real)
    iyy = float(lab[1, 1].real)
    ixy = lab[0, 1]
    stokes_lab = np.array([ixx + iyy, ixx - iyy,
                           2.0 * float(ixy.real), -2.0 * float(ixy.imag)])
    return ixx, iyy, stokes_lab


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorFieldResult:
    """Result of a combined polarization- and phase-resolved simulation.

    Attributes
    ----------
    field_co, field_cross : np.ndarray
        Complex detector-plane fields for the two orthogonal analyzer
        settings (lab x and y), shape ``(detector_bins, detector_bins)``.
        ``|field_co|^2`` is what a camera behind an x-analyzer records:
        a speckle pattern. Their *incoherent* counterparts are
        ``intensity_co`` / ``intensity_cross``.
    intensity_sq_co, intensity_sq_cross : np.ndarray
        Per-pixel sums of the *squared* single-photon intensities.
        Together with ``intensity_co``/``intensity_cross`` these give
        the effective number of interfering contributions per pixel,
        ``n_eff = (sum w)^2 / sum w^2`` -- the quantity that sets how
        far a finite-photon-budget speckle pattern still is from fully
        developed statistics. Recorded rather than assumed, so the
        finite-N bias in any measured speckle contrast can be computed
        instead of hand-waved.
    intensity_co, intensity_cross : np.ndarray
        Incoherent (Stokes-like) intensity images: per-photon
        intensities summed without phase. These are what a
        polarization-only engine predicts, and what a coherent engine
        predicts *on ensemble average*.
    stokes_images : np.ndarray
        Shape ``(4, detector_bins, detector_bins)`` -- I, Q, U, V
        accumulated incoherently in the lab frame.
    stokes_total : np.ndarray
        Shape ``(4,)`` -- Stokes vector of all reflected light, per
        incident photon.
    diffuse_reflectance, transmittance, absorbed : float
        Scalar radiometric budget, directly comparable to
        :func:`photon_transport_toolkit.monte_carlo.simulate_slab`.
    diffuse_reflectance_stderr, transmittance_stderr : float
        Batch standard errors.
    max_dop_violation : float
        Largest observed excess of ``sqrt(Q^2+U^2+V^2)/I`` over 1 across
        all traced photons. Physically must be 0; any positive value is
        an algebra bug, so it is measured rather than assumed.
    n_photons : int
    wavelength_nm : float
    detector_half_width : float
    formulation : str
        ``"jones"`` or ``"stokes"`` -- which of the two independent
        tracers produced this result.
    """

    field_co: np.ndarray
    field_cross: np.ndarray
    intensity_co: np.ndarray
    intensity_cross: np.ndarray
    intensity_sq_co: np.ndarray
    intensity_sq_cross: np.ndarray
    stokes_images: np.ndarray
    stokes_total: np.ndarray
    diffuse_reflectance: float
    transmittance: float
    absorbed: float
    diffuse_reflectance_stderr: float
    transmittance_stderr: float
    max_dop_violation: float
    n_photons: int
    wavelength_nm: float
    detector_half_width: float
    formulation: str


# --------------------------------------------------------------------------
# Incident-state helpers
# --------------------------------------------------------------------------


def _initial_jones(polarization: str, rng: np.random.Generator):
    """Jones vector of the incident beam, in the basis ``e1 = x, e2 = y``."""
    if polarization == "x":
        return 1.0 + 0.0j, 0.0 + 0.0j
    if polarization == "y":
        return 0.0 + 0.0j, 1.0 + 0.0j
    if polarization == "circular":
        inv = 1.0 / np.sqrt(2.0)
        return complex(inv, 0.0), complex(0.0, inv)
    if polarization == "unpolarized":
        # A random *fully* polarized state per photon, uniform on the
        # Poincare sphere. Averaged over photons this is an unpolarized
        # beam, while each individual packet stays a valid Jones vector
        # (DoP = 1) -- the standard way to represent natural light in a
        # coherent engine, since a Jones vector cannot itself be
        # partially polarized.
        # Uniform on the Poincare sphere means uniform in sin(2*chi),
        # NOT in cos(2*chi): the latter confines chi to [0, pi/2], so
        # V = sin(2*chi) never goes negative and the "unpolarized"
        # ensemble comes out 79% right-circularly polarized. Caught by
        # test_incident_states_are_physical, which measures the residual
        # polarization of the ensemble rather than trusting the sampler.
        sin_2chi = 2.0 * rng.random() - 1.0
        chi = 0.5 * np.arcsin(sin_2chi)
        psi = np.pi * rng.random()
        e1 = np.cos(psi) * np.cos(chi) - 1j * np.sin(psi) * np.sin(chi)
        e2 = np.sin(psi) * np.cos(chi) + 1j * np.cos(psi) * np.sin(chi)
        return complex(e1), complex(e2)
    raise ValueError('polarization must be "x", "y", "circular" or "unpolarized".')


# --------------------------------------------------------------------------
# Photon tracers
# --------------------------------------------------------------------------


def _trace_one_photon_jones(slab, rng, polarization, weight_threshold, roulette_survival,
                            track_dop=False, scatterer=None):
    """Trace one photon carrying a Jones vector and its optical path length.

    Returns ``(outcome, x, y, (j1, j2), (e1, e2), path_length,
    dop_violation)`` with the Jones components expressed in the
    photon's own exit frame ``(e1, e2)``; projection onto laboratory
    analyzer axes is left to the caller so that the radiometric budget
    can use the full intensity while the detector uses the projection.
    """
    x = y = z = 0.0
    u = (0.0, 0.0, 1.0)
    e1, e2 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    path_length = 0.0
    dop_violation = 0.0

    amp_albedo = _sqrt(slab.albedo)
    j1, j2 = _initial_jones(polarization, rng)

    # Entrance: normal incidence, so r_s = r_p and the polarization
    # state is untouched; only the intensity is reduced.
    r_s, r_p = _fresnel_amplitudes(1.0, slab.n_outside, slab.n_medium)
    amp = np.sqrt(max(0.0, 1.0 - abs(r_s) ** 2))
    j1 *= amp
    j2 *= amp

    steps = 0
    while (abs(j1) ** 2 + abs(j2) ** 2) > 0.0:
        steps += 1
        if steps > 100_000:  # pragma: no cover - runaway guard
            return "A", x, y, (0j, 0j), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), \
                path_length, dop_violation

        tau = -_log(rng.random())

        while True:
            step = tau / slab.mu_t
            if u[2] > 0.0:
                dist_boundary = (slab.thickness - z) / u[2]
            elif u[2] < 0.0:
                dist_boundary = -z / u[2]
            else:
                dist_boundary = np.inf

            if step < dist_boundary:
                x += step * u[0]
                y += step * u[1]
                z += step * u[2]
                path_length += step
                break

            x += dist_boundary * u[0]
            y += dist_boundary * u[1]
            z += dist_boundary * u[2]
            path_length += dist_boundary
            z = 0.0 if u[2] < 0.0 else slab.thickness
            tau -= dist_boundary * slab.mu_t

            # Align the basis so e2 = s, e1 = p (plane-of-incidence basis).
            s_dir = _sp_basis(u)
            if s_dir is not None:
                cos_a, sin_a = _rotation_to(e1, e2, u, s_dir)
                e1, e2 = _rotate_frame_about_u(e1, e2, cos_a, sin_a)
                j1, j2 = j1 * cos_a + j2 * sin_a, -j1 * sin_a + j2 * cos_a

            r_s, r_p = _fresnel_amplitudes(abs(u[2]), slab.n_medium, slab.n_outside)
            i_p, i_s = abs(j1) ** 2, abs(j2) ** 2
            intensity = i_p + i_s
            r_eff = (abs(r_p) ** 2 * i_p + abs(r_s) ** 2 * i_s) / intensity

            if rng.random() > r_eff:
                # Transmitted. Amplitude transmission coefficients follow
                # from t = 1 + r (s) and n1(1+r)/n2 (p); only their ratio
                # matters after renormalization to the packet intensity.
                t_s = 1.0 + r_s
                t_p = (slab.n_medium / slab.n_outside) * (1.0 + r_p)
                j1, j2 = j1 * t_p, j2 * t_s
                norm = np.sqrt(intensity / max(abs(j1) ** 2 + abs(j2) ** 2, 1e-300))
                j1 *= norm
                j2 *= norm
                u_out = _refract_direction_vec(u, slab.n_medium, slab.n_outside)
                e1, e2, u_out = _renormalize_frame(e1, e2, u_out)
                outcome = "R" if u[2] < 0.0 else "T"
                return outcome, x, y, (j1, j2), (e1, e2), path_length, dop_violation

            # Internally reflected.
            j1, j2 = j1 * r_p, j2 * r_s
            norm = np.sqrt(intensity / max(abs(j1) ** 2 + abs(j2) ** 2, 1e-300))
            j1 *= norm
            j2 *= norm
            u = (u[0], u[1], -u[2])
            e1 = (e2[1] * u[2] - e2[2] * u[1], e2[2] * u[0] - e2[0] * u[2],
                  e2[0] * u[1] - e2[1] * u[0])
            e1, e2, u = _renormalize_frame(e1, e2, u)

        # ---- absorption -------------------------------------------------
        amp = amp_albedo
        j1 *= amp
        j2 *= amp
        intensity = j1.real * j1.real + j1.imag * j1.imag + j2.real * j2.real + j2.imag * j2.imag

        # ---- scattering ---------------------------------------------------
        if scatterer is None:
            # Rayleigh: rejection sampling from |J R(psi) E|^2 directly.
            while True:
                cos_theta = 2.0 * rng.random() - 1.0
                psi = 6.283185307179586 * rng.random()
                cos_psi, sin_psi = _cos(psi), _sin(psi)
                p1 = j1 * cos_psi + j2 * sin_psi
                p2 = -j1 * sin_psi + j2 * cos_psi
                # Rayleigh amplitude matrix diag(S2, S1) = diag(cos_theta, 1)
                n1c, n2c = cos_theta * p1, p2
                i_new = (n1c.real * n1c.real + n1c.imag * n1c.imag
                         + n2c.real * n2c.real + n2c.imag * n2c.imag)
                if rng.random() * intensity <= i_new:
                    break
            renorm = _sqrt(intensity / i_new)
            j1, j2 = n1c * renorm, n2c * renorm
        else:
            # Mie: theta from the m11 inverse CDF, psi by bounded rejection
            # (see mie.MieScatterer.sample_jones). Uniform-cos(theta)
            # rejection would accept ~1 proposal in 10^4 at x = 20.
            cos_theta, cos_psi, sin_psi, j1, j2 = scatterer.sample_jones(j1, j2, rng)

        sin_theta = _sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        e1, e2 = _rotate_frame_about_u(e1, e2, cos_psi, sin_psi)
        e1, u = _tilt_frame_about_e2(e1, u, cos_theta, sin_theta)
        e1, e2, u = _renormalize_frame(e1, e2, u)

        if track_dop:
            dop = degree_of_polarization(jones_to_stokes(j1, j2))
            dop_violation = max(dop_violation, dop - 1.0)

        # ---- Russian roulette -------------------------------------------
        if intensity < weight_threshold:
            if rng.random() <= 1.0 / roulette_survival:
                boost = _sqrt(float(roulette_survival))
                j1 *= boost
                j2 *= boost
            else:
                return "A", x, y, (0j, 0j), (e1, e2), path_length, dop_violation

    return "A", x, y, (0j, 0j), (e1, e2), path_length, dop_violation


def _trace_one_photon_stokes(slab, rng, polarization, weight_threshold, roulette_survival,
                             scatterer=None):
    """Independent Stokes-Mueller tracer -- same geometry, different algebra.

    Deliberately written from the 4x4 Mueller formulation rather than
    by converting :func:`_trace_one_photon_jones`'s output, so that
    agreement between the two is evidence rather than tautology. It
    carries no phase and therefore returns no path length: that is the
    limitation being demonstrated, not an oversight.
    """
    x = y = z = 0.0
    u = (0.0, 0.0, 1.0)
    e1, e2 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    dop_violation = 0.0

    j1, j2 = _initial_jones(polarization, rng)
    stokes = jones_to_stokes(j1, j2)

    r_s, _ = _fresnel_amplitudes(1.0, slab.n_outside, slab.n_medium)
    stokes = stokes * (1.0 - abs(r_s) ** 2)

    steps = 0
    while stokes[0] > 0.0:
        steps += 1
        if steps > 100_000:  # pragma: no cover - runaway guard
            return "A", x, y, np.zeros(4), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), \
                dop_violation

        tau = -np.log(rng.random())

        while True:
            step = tau / slab.mu_t
            if u[2] > 0.0:
                dist_boundary = (slab.thickness - z) / u[2]
            elif u[2] < 0.0:
                dist_boundary = -z / u[2]
            else:
                dist_boundary = np.inf

            if step < dist_boundary:
                x += step * u[0]
                y += step * u[1]
                z += step * u[2]
                break

            x += dist_boundary * u[0]
            y += dist_boundary * u[1]
            z += dist_boundary * u[2]
            z = 0.0 if u[2] < 0.0 else slab.thickness
            tau -= dist_boundary * slab.mu_t

            s_dir = _sp_basis(u)
            if s_dir is not None:
                cos_a, sin_a = _rotation_to(e1, e2, u, s_dir)
                e1, e2 = _rotate_frame_about_u(e1, e2, cos_a, sin_a)
                stokes = _rotate_stokes(stokes, cos_a, sin_a)

            r_s, r_p = _fresnel_amplitudes(abs(u[2]), slab.n_medium, slab.n_outside)
            i_p = 0.5 * (stokes[0] + stokes[1])
            i_s = 0.5 * (stokes[0] - stokes[1])
            intensity = stokes[0]
            r_eff = (abs(r_p) ** 2 * i_p + abs(r_s) ** 2 * i_s) / intensity

            if rng.random() > r_eff:
                t_s = 1.0 + r_s
                t_p = (slab.n_medium / slab.n_outside) * (1.0 + r_p)
                stokes = _apply_diattenuator(stokes, t_p, t_s)
                stokes = stokes * (intensity / stokes[0])
                u_out = _refract_direction_vec(u, slab.n_medium, slab.n_outside)
                e1, e2, u_out = _renormalize_frame(e1, e2, u_out)
                outcome = "R" if u[2] < 0.0 else "T"
                return outcome, x, y, stokes, (e1, e2), dop_violation

            stokes = _apply_diattenuator(stokes, r_p, r_s)
            stokes = stokes * (intensity / stokes[0])
            u = (u[0], u[1], -u[2])
            e1 = (e2[1] * u[2] - e2[2] * u[1], e2[2] * u[0] - e2[0] * u[2],
                  e2[0] * u[1] - e2[1] * u[0])
            e1, e2, u = _renormalize_frame(e1, e2, u)

        stokes = stokes * slab.albedo
        intensity = stokes[0]

        if scatterer is None:
            while True:
                cos_theta = 2.0 * rng.random() - 1.0
                psi = 2.0 * np.pi * rng.random()
                cos_psi, sin_psi = np.cos(psi), np.sin(psi)
                rotated = _rotate_stokes(stokes, cos_psi, sin_psi)
                scattered = rayleigh_mueller_matrix(cos_theta) @ rotated
                if rng.random() * intensity <= scattered[0]:
                    break
            stokes = scattered * (intensity / scattered[0])
        else:
            cos_theta, cos_psi, sin_psi, stokes = scatterer.sample_stokes(stokes, rng)

        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        e1, e2 = _rotate_frame_about_u(e1, e2, cos_psi, sin_psi)
        e1, u = _tilt_frame_about_e2(e1, u, cos_theta, sin_theta)
        e1, e2, u = _renormalize_frame(e1, e2, u)

        dop_violation = max(dop_violation, degree_of_polarization(stokes) - 1.0)

        if intensity < weight_threshold:
            if rng.random() <= 1.0 / roulette_survival:
                stokes = stokes * roulette_survival
            else:
                return "A", x, y, np.zeros(4), (e1, e2), dop_violation

    return "A", x, y, np.zeros(4), (e1, e2), dop_violation


def _rotate_stokes(stokes, cos_a, sin_a):
    """Rotate a Stokes vector's reference frame by the angle with the given cos/sin."""
    cos2 = cos_a * cos_a - sin_a * sin_a
    sin2 = 2.0 * sin_a * cos_a
    return np.array(
        [
            stokes[0],
            stokes[1] * cos2 + stokes[2] * sin2,
            -stokes[1] * sin2 + stokes[2] * cos2,
            stokes[3],
        ]
    )


def _apply_diattenuator(stokes, t_p, t_s):
    """Apply a complex-amplitude diattenuator/retarder to a Stokes vector.

    ``t_p`` and ``t_s`` act on the parallel and perpendicular
    components; their modulus ratio produces diattenuation and their
    phase difference produces retardance (the total-internal-reflection
    case). This is the Mueller matrix of ``diag(t_p, t_s)``, written
    out directly.
    """
    a = abs(t_p) ** 2
    b = abs(t_s) ** 2
    cross = t_p * np.conj(t_s)
    m11 = 0.5 * (a + b)
    m12 = 0.5 * (a - b)
    m33 = float(cross.real)
    m34 = float(cross.imag)
    return np.array(
        [
            m11 * stokes[0] + m12 * stokes[1],
            m12 * stokes[0] + m11 * stokes[1],
            m33 * stokes[2] + m34 * stokes[3],
            -m34 * stokes[2] + m33 * stokes[3],
        ]
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def simulate_slab_vector(
    slab: SlabOpticalProperties,
    wavelength_nm: float,
    n_photons: int = 20_000,
    seed: int | None = 0,
    n_batches: int = 10,
    detector_bins: int = 64,
    detector_half_width: float = 2.0,
    polarization: str = "x",
    formulation: str = "jones",
    scatterer=None,
    track_dop: bool = False,
    weight_threshold: float = 1e-4,
    roulette_survival: int = 10,
) -> VectorFieldResult:
    """Simulate a slab with polarization *and* (for ``"jones"``) phase.

    Parameters
    ----------
    slab : SlabOpticalProperties
        Optical properties. ``slab.g`` must be 0: the scattering angles
        come either from Rayleigh scattering (``<cos theta> = 0`` by
        symmetry) or from ``scatterer``, which computes its own
        asymmetry parameter, so a nonzero ``slab.g`` would be silently
        ignored. Raising instead of ignoring is deliberate.
    scatterer : MieScatterer or None
        ``None`` (default) selects Rayleigh scattering. A
        :class:`photon_transport_toolkit.mie.MieScatterer` selects Mie
        scattering off spheres of the given size and index.
        ``slab.mu_s`` remains an *independent* input in that case: the
        particle size enters through the phase function and the
        polarization algebra only. That is deliberate, so that a
        Rayleigh and a Mie run can be compared at identical optical
        depth rather than at identical particle concentration -- the
        comparison then isolates the angular and polarization physics
        instead of confounding it with a different scattering
        coefficient.
    polarization : {"x", "y", "circular", "unpolarized"}
        Incident polarization state.
    track_dop : bool
        Record the largest violation of ``DoP <= 1`` seen at any
        scattering event. Off by default because it costs a Stokes
        conversion per event in a loop that runs hundreds of times per
        photon; the test suite switches it on.
    formulation : {"jones", "stokes"}
        Which tracer to use. ``"stokes"`` produces no coherent field
        (``field_co`` / ``field_cross`` are zero) -- that is the point
        of the comparison, not a missing feature.

    Returns
    -------
    VectorFieldResult
    """
    if slab.g != 0.0:
        raise ValueError(
            "scattering angles come from the phase function (Rayleigh, or the "
            "supplied scatterer), so slab.g must be 0 rather than a value that "
            "would be silently ignored."
        )
    if n_photons < n_batches:
        raise ValueError("n_photons must be at least n_batches.")
    if n_batches < 2:
        raise ValueError("At least two batches are required to estimate an uncertainty.")
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    if formulation not in ("jones", "stokes"):
        raise ValueError('formulation must be "jones" or "stokes".')

    wavelength_mm = wavelength_nm * 1e-6
    k0 = 2.0 * np.pi / wavelength_mm

    rng = np.random.default_rng(seed)
    per_batch = n_photons // n_batches
    total = per_batch * n_batches

    field_co = np.zeros((detector_bins, detector_bins), dtype=complex)
    field_cross = np.zeros((detector_bins, detector_bins), dtype=complex)
    inten_co = np.zeros((detector_bins, detector_bins))
    inten_cross = np.zeros((detector_bins, detector_bins))
    inten_sq_co = np.zeros((detector_bins, detector_bins))
    inten_sq_cross = np.zeros((detector_bins, detector_bins))
    stokes_images = np.zeros((4, detector_bins, detector_bins))
    stokes_total = np.zeros(4)

    batch_r = np.empty(n_batches)
    batch_t = np.empty(n_batches)
    batch_a = np.empty(n_batches)
    max_violation = 0.0

    edges = np.linspace(-detector_half_width, detector_half_width, detector_bins + 1)
    tracer = _trace_one_photon_jones if formulation == "jones" else _trace_one_photon_stokes

    r_sp, _ = _fresnel_amplitudes(1.0, slab.n_outside, slab.n_medium)
    launched = 1.0 - abs(r_sp) ** 2

    for b in range(n_batches):
        acc_r = acc_t = acc_a = 0.0
        for _ in range(per_batch):
            if formulation == "jones":
                outcome, x, y, (j1, j2), (e1, e2), path_length, viol = tracer(
                    slab, rng, polarization, weight_threshold, roulette_survival,
                    track_dop, scatterer
                )
                intensity = abs(j1) ** 2 + abs(j2) ** 2
                coherency = _coherency_from_jones(j1, j2)
            else:
                outcome, x, y, stokes_exit, (e1, e2), viol = tracer(
                    slab, rng, polarization, weight_threshold, roulette_survival,
                    scatterer
                )
                intensity = float(stokes_exit[0])
                coherency = _coherency_from_stokes(stokes_exit)
                path_length = 0.0
            max_violation = max(max_violation, viol)
            # Absorbed is defined as (launched - exited) per packet, so
            # R + T + A + specular = 1 holds *exactly*, per photon, with
            # roulette-terminated residual weight credited to absorption
            # -- the same exact-energy-conservation invariant the scalar
            # engine deliberately maintains (PROJECT_REPORT.md).
            acc_a += launched - intensity
            if outcome == "R":
                acc_r += intensity
            elif outcome == "T":
                acc_t += intensity

            if outcome != "R" or intensity <= 0.0:
                continue

            i_x, i_y, stokes_lab = _project_to_lab(coherency, e1, e2)
            stokes_total += stokes_lab
            ix = np.searchsorted(edges, x, side="right") - 1
            iy = np.searchsorted(edges, y, side="right") - 1
            if not (0 <= ix < detector_bins and 0 <= iy < detector_bins):
                continue

            stokes_images[:, iy, ix] += stokes_lab
            inten_co[iy, ix] += i_x
            inten_cross[iy, ix] += i_y
            inten_sq_co[iy, ix] += i_x * i_x
            inten_sq_cross[iy, ix] += i_y * i_y
            if formulation == "jones":
                phasor = np.exp(1j * k0 * slab.n_medium * path_length)
                field_co[iy, ix] += (j1 * e1[0] + j2 * e2[0]) * phasor
                field_cross[iy, ix] += (j1 * e1[1] + j2 * e2[1]) * phasor

        batch_r[b] = acc_r / per_batch
        batch_t[b] = acc_t / per_batch
        batch_a[b] = acc_a / per_batch

    def _mean_se(values):
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))

    r_mean, r_se = _mean_se(batch_r)
    t_mean, t_se = _mean_se(batch_t)
    a_mean, _ = _mean_se(batch_a)

    return VectorFieldResult(
        field_co=field_co / np.sqrt(total),
        field_cross=field_cross / np.sqrt(total),
        intensity_co=inten_co / total,
        intensity_cross=inten_cross / total,
        intensity_sq_co=inten_sq_co / (total * total),
        intensity_sq_cross=inten_sq_cross / (total * total),
        stokes_images=stokes_images / total,
        stokes_total=stokes_total / total,
        diffuse_reflectance=r_mean,
        transmittance=t_mean,
        absorbed=a_mean,
        diffuse_reflectance_stderr=r_se,
        transmittance_stderr=t_se,
        max_dop_violation=float(max_violation),
        n_photons=total,
        wavelength_nm=wavelength_nm,
        detector_half_width=detector_half_width,
        formulation=formulation,
    )


# --------------------------------------------------------------------------
# Depolarization vs scattering order
# --------------------------------------------------------------------------


def _canonical_stokes(j1, j2, e1, e2, u, ref):
    """Stokes vector in a frame canonically fixed by ``u`` and a reference axis.

    Measuring how much polarization survives multiple scattering needs a
    frame in which different photons' Stokes vectors are comparable, and
    the three obvious choices all fail in instructive ways:

    * The *per-photon* degree of polarization is useless: a Jones vector
      scattered by real amplitude matrices stays fully polarized, so it
      is identically 1 forever. Depolarization is an ensemble property
      here, not a single-packet one.
    * The photon's *own* post-scattering frame is worse than useless: it
      is rotated by the azimuth psi at every event, so Q and U average
      away even for a photon that was never deflected at all
      (theta = 0), reporting depolarization that did not happen.
    * The laboratory x/y analyzer projection (:func:`_project_to_lab`)
      is physically meaningful -- it is what a camera behind a polarizer
      records -- but it multiplies the circular component by an
      obliquity factor that the linear components do not carry, which
      systematically penalizes the circular channel and masks the
      very linear-vs-circular asymmetry being measured.

    The frame used here fixes ``e2`` along ``u x ref``, with ``ref`` the
    incident polarization direction. It is a function of the photon's
    current direction only, so it is common to all photons travelling
    the same way, it reduces to the incident frame for undeflected
    photons, and it contains no projection factor. Q then measures
    retention of the *original* linear direction and V retention of
    helicity, on the same footing.
    """
    sx = u[1] * ref[2] - u[2] * ref[1]
    sy = u[2] * ref[0] - u[0] * ref[2]
    sz = u[0] * ref[1] - u[1] * ref[0]
    norm = _sqrt(sx * sx + sy * sy + sz * sz)
    stokes = jones_to_stokes(j1, j2)
    if norm < 1e-9:  # u parallel to ref: the frame is undefined, and Q is
        return stokes  # meaningless for such a photon anyway (measure-zero set)
    cos_a, sin_a = _rotation_to(e1, e2, u, (sx / norm, sy / norm, sz / norm))
    return _rotate_stokes(stokes, cos_a, sin_a)


def depolarization_ladder(
    n_events: int = 40,
    n_photons: int = 2000,
    scatterer=None,
    polarization: str = "x",
    seed: int | None = 0,
    frame: str = "canonical",
):
    """Ensemble polarization retained after each successive scattering event.

    An unbounded-medium experiment: no slab, no boundaries, no
    absorption -- a photon's polarization state is pushed through
    ``n_events`` scattering events and the ensemble-mean Stokes vector
    is recorded after each one. With ``scatterer=None`` the scattering
    is Rayleigh; with a :class:`~photon_transport_toolkit.mie.MieScatterer`
    it is Mie off spheres of the chosen size.

    Parameters
    ----------
    frame : {"canonical", "lab"}
        Which reference frame the ensemble average is taken in. See
        :func:`_canonical_stokes` for why the default is not the
        photon's own frame. ``"lab"`` reproduces what an x/y analyzer
        and camera would measure, including the obliquity factor, and
        is provided so that the difference between the two can be
        shown rather than asserted.

    Returns
    -------
    (order, q_over_i, v_over_i) : tuple of np.ndarray
        Scattering order ``0..n_events``, the retained linear
        polarization ``|<Q>|/<I>`` and the retained circular
        polarization ``|<V>|/<I>``. Launch with ``polarization="x"``
        to follow the first and ``"circular"`` to follow the second.
    """
    if n_events < 1:
        raise ValueError("n_events must be at least 1.")
    if n_photons < 1:
        raise ValueError("n_photons must be at least 1.")
    if frame not in ("canonical", "lab"):
        raise ValueError('frame must be "canonical" or "lab".')

    rng = np.random.default_rng(seed)
    acc = np.zeros((n_events + 1, 4))
    ref = (1.0, 0.0, 0.0)

    for _ in range(n_photons):
        j1, j2 = _initial_jones(polarization, rng)
        e1, e2, u = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)

        for k in range(n_events + 1):
            if frame == "canonical":
                acc[k] += _canonical_stokes(j1, j2, e1, e2, u, ref)
            else:
                acc[k] += _project_to_lab(_coherency_from_jones(j1, j2), e1, e2)[2]
            if k == n_events:
                break

            intensity = abs(j1) ** 2 + abs(j2) ** 2
            if scatterer is None:
                while True:
                    cos_theta = 2.0 * rng.random() - 1.0
                    psi = 6.283185307179586 * rng.random()
                    cos_psi, sin_psi = _cos(psi), _sin(psi)
                    p1 = j1 * cos_psi + j2 * sin_psi
                    p2 = -j1 * sin_psi + j2 * cos_psi
                    n1c, n2c = cos_theta * p1, p2
                    i_new = abs(n1c) ** 2 + abs(n2c) ** 2
                    if rng.random() * intensity <= i_new:
                        break
                renorm = _sqrt(intensity / i_new)
                j1, j2 = n1c * renorm, n2c * renorm
            else:
                cos_theta, cos_psi, sin_psi, j1, j2 = scatterer.sample_jones(j1, j2, rng)

            sin_theta = _sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
            e1, e2 = _rotate_frame_about_u(e1, e2, cos_psi, sin_psi)
            e1, u = _tilt_frame_about_e2(e1, u, cos_theta, sin_theta)
            e1, e2, u = _renormalize_frame(e1, e2, u)

    order = np.arange(n_events + 1)
    return order, np.abs(acc[:, 1]) / acc[:, 0], np.abs(acc[:, 3]) / acc[:, 0]
