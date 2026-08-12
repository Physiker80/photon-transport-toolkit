"""Mie (spherical-particle) scattering for the vector transport engine.

Why this module exists
----------------------
:mod:`photon_transport_toolkit.vector_transport` was built on Rayleigh
(dipole) scattering, which is exact only for scatterers far smaller than
the wavelength. Real tissue scatterers are not: cell nuclei, mitochondria
and collagen fibre bundles are comparable to or larger than an optical
wavelength, and that single fact changes the polarization physics
qualitatively rather than quantitatively.

The specific consequence this module was written to model is the
*linear/circular depolarization asymmetry*. Under Rayleigh scattering,
linear polarization survives multiple scattering longer than circular
does. For large (Mie) scatterers the ordering reverses -- circular
polarization is retained over more scattering events than linear, the
effect known as **polarization memory**. A Rayleigh-only engine cannot
reproduce that reversal at all, because it has no size parameter to
vary. Since the reversal is precisely what makes circular-polarization
gating and Mueller-matrix imaging able to distinguish tissue types, it
is not a refinement: it is the effect.

Formulation
-----------
Standard Mie series (Bohren & Huffman 1983, ch. 4), written directly
from the recurrences rather than adapted from an existing code:

    a_n = [(D_n/m + n/x) psi_n  - psi_{n-1}] / [(D_n/m + n/x) xi_n  - xi_{n-1}]
    b_n = [(m D_n   + n/x) psi_n  - psi_{n-1}] / [(m D_n   + n/x) xi_n  - xi_{n-1}]

with the logarithmic derivative ``D_n(mx)`` computed by *downward*
recurrence (upward recurrence is numerically unstable for absorbing
particles -- the classic Mie-code failure mode), and the amplitude
scattering functions

    S1 = sum (2n+1)/(n(n+1)) (a_n pi_n + b_n tau_n)
    S2 = sum (2n+1)/(n(n+1)) (a_n tau_n + b_n pi_n)

For a homogeneous sphere the Mueller matrix has the four-element block
form ``[[m11, m12, 0, 0], [m12, m11, 0, 0], [0, 0, m33, m34],
[0, 0, -m34, m33]]`` with

    m11 = (|S1|^2 + |S2|^2)/2      m12 = (|S2|^2 - |S1|^2)/2
    m33 = Re(S1 S2*)               m34 = Im(S1 S2*)

Rayleigh is the ``x -> 0`` limit of the same expressions
(``S1 -> const``, ``S2 -> const * cos(theta)``), which is checked
directly in ``tests/test_mie.py`` rather than assumed.

Sampling strategy
-----------------
The Rayleigh tracer samples ``cos(theta)`` uniformly and rejects against
the polarization-dependent phase function. That is fine when the phase
function varies by a factor of two; a Mie phase function at
``x = 20`` is forward-peaked by four orders of magnitude, and uniform
rejection sampling would accept roughly one proposal in 10^4.

So sampling here is two-stage:

1. ``theta`` from the *unpolarized* phase function ``m11(theta)
   sin(theta)`` by inverse-CDF on a precomputed grid -- exact up to grid
   resolution, and O(1) per event.
2. ``psi`` uniform, then accept with probability
   ``p(theta, psi) / [m11 (1 + |m12|/m11)]``. Since ``|m12| <= m11``
   always, the acceptance rate is at worst 1/2 and typically much
   better, independently of how forward-peaked the particle is.

The joint density is then proportional to ``p(theta, psi) sin(theta)``,
which is the correct polarization-dependent scattering density. As in
the Rayleigh tracer, the packet is renormalized to its pre-scattering
intensity afterwards, so exact energy conservation is preserved.

Validation
----------
``tests/test_mie.py`` checks the series against (a) the analytic
Rayleigh limit, (b) the extinction paradox ``Q_ext -> 2`` for large
``x``, (c) an internal consistency identity -- the angular integral of
``(|S1|^2+|S2|^2)/2`` must equal ``pi x^2 Q_sca``, which uses the
angular functions and the efficiency series independently of each other,
(d) the asymmetry parameter from the series against the same quantity
integrated numerically from the phase function, and (e) where the
optional package ``miepython`` is installed, against that entirely
independent implementation.

Author: Noureddin Sedki
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MieScatterer",
    "mie_coefficients",
    "mie_efficiencies",
    "mie_amplitudes",
    "size_parameter",
]


# --------------------------------------------------------------------------
# Core Mie series
# --------------------------------------------------------------------------


def size_parameter(radius_um: float, wavelength_nm: float, n_medium: float) -> float:
    """Size parameter ``x = 2 pi a n_medium / lambda_vacuum``.

    Both lengths are converted to micrometres internally; the medium
    index enters because the relevant wavelength is the one *inside*
    the medium, a detail that is easy to drop and that shifts every
    subsequent number.
    """
    return 2.0 * np.pi * radius_um * n_medium / (wavelength_nm * 1e-3)


def _n_terms(x: float) -> int:
    """Wiscombe's series-truncation criterion, plus a safety margin."""
    return int(x + 4.0 * x ** (1.0 / 3.0) + 2.0) + 15


def mie_coefficients(x: float, m: complex) -> tuple[np.ndarray, np.ndarray]:
    """Mie coefficients ``(a_n, b_n)`` for size parameter ``x``, relative index ``m``.

    ``m`` is the particle index *relative to the surrounding medium*.
    """
    if x <= 0.0:
        raise ValueError("size parameter x must be positive.")
    m = complex(m)
    if m.real <= 0.0:
        raise ValueError("relative refractive index must have positive real part.")
    if m.imag < 0.0:
        raise ValueError(
            "absorbing particles take a *positive* imaginary index in the "
            "exp(-i omega t) convention used here."
        )

    nmax = _n_terms(x)
    mx = m * x

    # Logarithmic derivative D_n(mx) by downward recurrence, started well
    # above nmax where the recurrence has forgotten its seed.
    nmx = int(max(nmax, abs(mx)) + 16)
    d = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        d[n - 1] = n / mx - 1.0 / (d[n] + n / mx)
    d = d[1:nmax + 1]

    # Riccati-Bessel psi_n(x), chi_n(x) by upward recurrence (stable for
    # real argument).
    psi = np.zeros(nmax + 1)
    chi = np.zeros(nmax + 1)
    psi_m1, chi_m1 = np.sin(x), np.cos(x)          # n = 0
    psi_m2, chi_m2 = np.cos(x), -np.sin(x)         # n = -1
    for n in range(1, nmax + 1):
        psi[n] = (2.0 * n - 1.0) / x * psi_m1 - psi_m2
        chi[n] = (2.0 * n - 1.0) / x * chi_m1 - chi_m2
        psi_m2, psi_m1 = psi_m1, psi[n]
        chi_m2, chi_m1 = chi_m1, chi[n]
    psi[0], chi[0] = np.sin(x), np.cos(x)

    xi = psi - 1j * chi
    n_arr = np.arange(1, nmax + 1)
    psi_n, psi_nm1 = psi[1:], psi[:-1]
    xi_n, xi_nm1 = xi[1:], xi[:-1]

    da = d / m + n_arr / x
    db = d * m + n_arr / x
    a = (da * psi_n - psi_nm1) / (da * xi_n - xi_nm1)
    b = (db * psi_n - psi_nm1) / (db * xi_n - xi_nm1)
    return a, b


def mie_efficiencies(x: float, m: complex) -> tuple[float, float, float]:
    """Return ``(Q_ext, Q_sca, g)`` from the coefficient series."""
    a, b = mie_coefficients(x, m)
    n = np.arange(1, len(a) + 1)
    c = 2.0 * n + 1.0

    q_ext = float((2.0 / x**2) * np.sum(c * (a + b).real))
    q_sca = float((2.0 / x**2) * np.sum(c * (np.abs(a) ** 2 + np.abs(b) ** 2)))

    term1 = np.sum(
        (n[:-1] * (n[:-1] + 2.0) / (n[:-1] + 1.0))
        * (a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])).real
    )
    term2 = np.sum((c / (n * (n + 1.0))) * (a * np.conj(b)).real)
    g = float((4.0 / (x**2 * q_sca)) * (term1 + term2)) if q_sca > 0 else 0.0
    return q_ext, q_sca, g


def mie_amplitudes(cos_theta, a: np.ndarray, b: np.ndarray):
    """Amplitude scattering functions ``(S1, S2)`` at the given angles.

    Vectorized over ``cos_theta``. ``S1`` acts on the component
    perpendicular to the scattering plane and ``S2`` on the parallel
    one, matching the ``diag(S2, S1)`` Jones convention used by
    :mod:`photon_transport_toolkit.vector_transport`.
    """
    mu = np.atleast_1d(np.asarray(cos_theta, dtype=float))
    nmax = len(a)
    s1 = np.zeros_like(mu, dtype=complex)
    s2 = np.zeros_like(mu, dtype=complex)

    pi_nm1 = np.zeros_like(mu)      # pi_0
    pi_n = np.ones_like(mu)         # pi_1
    for n in range(1, nmax + 1):
        tau_n = n * mu * pi_n - (n + 1.0) * pi_nm1
        f = (2.0 * n + 1.0) / (n * (n + 1.0))
        s1 += f * (a[n - 1] * pi_n + b[n - 1] * tau_n)
        s2 += f * (a[n - 1] * tau_n + b[n - 1] * pi_n)
        pi_np1 = ((2.0 * n + 1.0) * mu * pi_n - (n + 1.0) * pi_nm1) / n
        pi_nm1, pi_n = pi_n, pi_np1
    return s1, s2


# --------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Tables:
    theta: np.ndarray
    s1_re: np.ndarray
    s1_im: np.ndarray
    s2_re: np.ndarray
    s2_im: np.ndarray
    cdf: np.ndarray
    dtheta: float


class MieScatterer:
    """A single-sphere Mie scatterer, ready to be sampled photon by photon.

    Parameters
    ----------
    radius_um : float
        Sphere radius in micrometres.
    wavelength_nm : float
        Vacuum wavelength.
    n_particle, n_medium : complex, float
        Refractive indices. ``n_particle`` may be complex (absorbing).
    n_angles : int
        Grid resolution for the sampling tables. The default resolves a
        Mie phase function's lobe structure up to ``x ~ 50``; the class
        raises rather than silently under-resolving beyond that.

    Notes
    -----
    ``self.g`` is the *computed* asymmetry parameter. It is not a free
    input, which is the substantive difference from the scalar engine:
    there ``g`` is chosen, here it follows from the particle size and
    index, and the phase function's higher moments follow with it.
    """

    def __init__(
        self,
        radius_um: float,
        wavelength_nm: float,
        n_particle: complex = 1.40,
        n_medium: float = 1.33,
        n_angles: int = 4001,
    ):
        if radius_um <= 0:
            raise ValueError("radius_um must be positive.")
        if wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive.")
        if n_angles < 201:
            raise ValueError("n_angles must be at least 201 to resolve the phase function.")

        self.radius_um = float(radius_um)
        self.wavelength_nm = float(wavelength_nm)
        self.n_particle = complex(n_particle)
        self.n_medium = float(n_medium)
        self.x = size_parameter(radius_um, wavelength_nm, n_medium)
        self.m = self.n_particle / self.n_medium

        if self.x > 50.0 and n_angles < 20001:
            raise ValueError(
                f"size parameter x = {self.x:.1f} needs a finer angular grid; "
                "pass n_angles >= 20001 or use a smaller particle."
            )

        self.a, self.b = mie_coefficients(self.x, self.m)
        self.q_ext, self.q_sca, self.g = mie_efficiencies(self.x, self.m)
        self._tables = self._build_tables(n_angles)

    # -- construction ------------------------------------------------------

    def _build_tables(self, n_angles: int) -> _Tables:
        theta = np.linspace(0.0, np.pi, n_angles)
        s1, s2 = mie_amplitudes(np.cos(theta), self.a, self.b)
        m11 = 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)

        weight = m11 * np.sin(theta)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]))])
        cdf /= cdf[-1]
        # Enforce strict monotonicity so the inverse lookup is well defined
        # even where the phase function has a deep minimum.
        cdf = np.maximum.accumulate(cdf)

        return _Tables(
            theta=theta,
            s1_re=s1.real.copy(), s1_im=s1.imag.copy(),
            s2_re=s2.real.copy(), s2_im=s2.imag.copy(),
            cdf=cdf,
            dtheta=float(theta[1] - theta[0]),
        )

    # -- public angular quantities ----------------------------------------

    def amplitudes(self, cos_theta):
        """Exact ``(S2, S1)`` from the series (no interpolation)."""
        s1, s2 = mie_amplitudes(cos_theta, self.a, self.b)
        if np.isscalar(cos_theta) or np.ndim(cos_theta) == 0:
            return complex(s2[0]), complex(s1[0])
        return s2, s1

    def phase_function(self, cos_theta):
        """Unpolarized phase function ``m11``, normalized so ``int p dOmega = 1``."""
        s1, s2 = mie_amplitudes(cos_theta, self.a, self.b)
        m11 = 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
        norm = np.pi * self.x**2 * self.q_sca
        return m11 / norm

    def mueller_matrix(self, cos_theta: float) -> np.ndarray:
        """4x4 single-sphere Mueller matrix, normalized to ``m11(0) = 1``... .

        Normalization is by ``m11`` at the requested angle's own scale:
        the matrix returned is the *unnormalized* one from ``S1, S2``,
        which is what the tracer needs (it renormalizes intensity
        itself). Divide by ``m11`` if a probability-like matrix is
        wanted.
        """
        s1, s2 = self.amplitudes(cos_theta)[::-1]  # (S1, S2)
        a2, b2 = abs(s1) ** 2, abs(s2) ** 2
        cross = s2 * np.conj(s1)
        m11 = 0.5 * (a2 + b2)
        m12 = 0.5 * (b2 - a2)
        m33 = float(cross.real)
        m34 = float(cross.imag)
        return np.array(
            [
                [m11, m12, 0.0, 0.0],
                [m12, m11, 0.0, 0.0],
                [0.0, 0.0, m33, m34],
                [0.0, 0.0, -m34, m33],
            ]
        )

    # -- sampling ----------------------------------------------------------

    def _interp(self, theta: float):
        """Linear interpolation of ``(S1, S2)`` on the precomputed grid."""
        t = self._tables
        pos = theta / t.dtheta
        i = int(pos)
        if i >= len(t.theta) - 1:
            i = len(t.theta) - 2
        f = pos - i
        g = 1.0 - f
        s1 = complex(g * t.s1_re[i] + f * t.s1_re[i + 1],
                     g * t.s1_im[i] + f * t.s1_im[i + 1])
        s2 = complex(g * t.s2_re[i] + f * t.s2_re[i + 1],
                     g * t.s2_im[i] + f * t.s2_im[i + 1])
        return s1, s2

    def sample_theta(self, u: float) -> float:
        """Inverse-CDF sample of the polar angle from ``m11 sin(theta)``."""
        t = self._tables
        return float(np.interp(u, t.cdf, t.theta))

    def sample_jones(self, j1: complex, j2: complex, rng) -> tuple[float, float, float,
                                                                  complex, complex]:
        """Sample one scattering event for a Jones vector.

        Returns ``(cos_theta, cos_psi, sin_psi, j1_new, j2_new)`` with
        the new Jones components expressed in the post-scattering
        frame and renormalized to the incoming intensity, so that
        intensity is conserved exactly.
        """
        intensity = (j1.real * j1.real + j1.imag * j1.imag
                     + j2.real * j2.real + j2.imag * j2.imag)
        while True:
            theta = self.sample_theta(rng.random())
            s1, s2 = self._interp(theta)
            a2, b2 = abs(s1) ** 2, abs(s2) ** 2
            m11 = 0.5 * (a2 + b2)
            m12 = 0.5 * (b2 - a2)
            bound = intensity * (m11 + abs(m12))

            psi = 6.283185307179586 * rng.random()
            cos_psi, sin_psi = np.cos(psi), np.sin(psi)
            p1 = j1 * cos_psi + j2 * sin_psi
            p2 = -j1 * sin_psi + j2 * cos_psi
            n1 = s2 * p1
            n2 = s1 * p2
            i_new = (n1.real * n1.real + n1.imag * n1.imag
                     + n2.real * n2.real + n2.imag * n2.imag)
            if rng.random() * bound <= i_new:
                renorm = np.sqrt(intensity / i_new)
                return (float(np.cos(theta)), float(cos_psi), float(sin_psi),
                        n1 * renorm, n2 * renorm)

    def sample_stokes(self, stokes: np.ndarray, rng):
        """Sample one scattering event for a Stokes vector.

        Independent of :meth:`sample_jones` in the same sense as the two
        tracers in :mod:`vector_transport`: it applies the 4x4 Mueller
        matrix rather than the amplitude matrix, and the test suite
        requires the two to agree statistically.
        """
        intensity = float(stokes[0])
        while True:
            theta = self.sample_theta(rng.random())
            s1, s2 = self._interp(theta)
            a2, b2 = abs(s1) ** 2, abs(s2) ** 2
            m11 = 0.5 * (a2 + b2)
            m12 = 0.5 * (b2 - a2)
            cross = s2 * np.conj(s1)
            m33, m34 = float(cross.real), float(cross.imag)
            bound = intensity * (m11 + abs(m12))

            psi = 2.0 * np.pi * rng.random()
            cos_psi, sin_psi = np.cos(psi), np.sin(psi)
            cos2, sin2 = cos_psi * cos_psi - sin_psi * sin_psi, 2.0 * sin_psi * cos_psi
            q = stokes[1] * cos2 + stokes[2] * sin2
            u_ = -stokes[1] * sin2 + stokes[2] * cos2
            i_new = m11 * stokes[0] + m12 * q
            if rng.random() * bound <= i_new:
                out = np.array([
                    i_new,
                    m12 * stokes[0] + m11 * q,
                    m33 * u_ + m34 * stokes[3],
                    -m34 * u_ + m33 * stokes[3],
                ])
                out *= intensity / i_new
                return float(np.cos(theta)), float(cos_psi), float(sin_psi), out

    def __repr__(self):  # pragma: no cover - cosmetic
        return (f"MieScatterer(radius_um={self.radius_um:g}, "
                f"wavelength_nm={self.wavelength_nm:g}, x={self.x:.3f}, "
                f"m={self.m:.4f}, g={self.g:.4f}, Qsca={self.q_sca:.4f})")
