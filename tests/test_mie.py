"""Tests for Mie scattering and its use in the vector transport engine.

The Mie series is easy to get subtly wrong and hard to eyeball, so the
checks here are of four independent kinds:

1. **Analytic limits** the series must reproduce (Rayleigh at small x,
   the extinction paradox at large x).
2. **Internal consistency identities** that use two different parts of
   the implementation against each other -- the angular functions
   against the efficiency series -- so that a shared error would have to
   be present in both to pass.
3. **Sampling correctness**: the sampler must reproduce the phase
   function it was built from, and the two samplers (Jones amplitude and
   Stokes-Mueller) must agree with each other.
4. **Cross-implementation**: where the optional package ``miepython`` is
   installed, against an entirely independent code. Skipped, not failed,
   when it is absent -- the toolkit must not acquire a runtime
   dependency for a test's convenience.

Author: Noureddin Sedki
License: MIT
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# np.trapezoid is NumPy >= 2.0; the project's stated floor is numpy >= 1.24,
# so the test must not silently require a newer NumPy than the library does.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

from photon_transport_toolkit import SlabOpticalProperties  # noqa: E402
from photon_transport_toolkit.mie import (  # noqa: E402
    MieScatterer,
    mie_amplitudes,
    mie_coefficients,
    mie_efficiencies,
    size_parameter,
)
from photon_transport_toolkit.vector_transport import (  # noqa: E402
    degree_of_polarization,
    depolarization_ladder,
    jones_to_stokes,
    simulate_slab_vector,
)


# --------------------------------------------------------------------------
# Analytic limits
# --------------------------------------------------------------------------


def test_small_particle_reproduces_the_rayleigh_cross_section():
    """Q_sca -> (8/3) x^4 |(m^2-1)/(m^2+2)|^2 as x -> 0."""
    x, m = 0.02, 1.5 + 0j
    _, q_sca, _ = mie_efficiencies(x, m)
    expected = (8.0 / 3.0) * x**4 * abs((m**2 - 1) / (m**2 + 2)) ** 2
    assert q_sca == pytest.approx(expected, rel=2e-3)


def test_small_particle_reproduces_the_rayleigh_phase_function():
    """The angular shape must become (1 + cos^2 theta)/2 at small x.

    Checked at x = 0.02 rather than 0.05: the leading finite-size
    correction breaks the exact forward/backward symmetry of the
    Rayleigh pattern at the 0.2% level already at x = 0.05, which is
    physics rather than error, so the limit is taken where it is
    actually a limit.
    """
    a, b = mie_coefficients(0.02, 1.5 + 0j)
    mu = np.linspace(-1.0, 1.0, 21)
    s1, s2 = mie_amplitudes(mu, a, b)
    m11 = 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
    rayleigh = 0.5 * (1.0 + mu**2)
    assert np.allclose(m11 / m11[-1], rayleigh / rayleigh[-1], rtol=1e-3)


def test_small_particle_amplitude_ratio_is_cos_theta():
    """S2/S1 -> cos(theta): the dipole pattern the Rayleigh tracer hard-codes."""
    a, b = mie_coefficients(0.05, 1.5 + 0j)
    mu = np.linspace(-0.95, 0.95, 15)
    s1, s2 = mie_amplitudes(mu, a, b)
    assert np.allclose((s2 / s1).real, mu, atol=2e-3)


def test_small_particle_asymmetry_parameter_vanishes():
    _, _, g = mie_efficiencies(0.05, 1.5 + 0j)
    assert abs(g) < 1e-3


def test_extinction_paradox():
    """Q_ext -> 2 for a large non-absorbing sphere, not 1."""
    q_ext, _, _ = mie_efficiencies(400.0, 1.33 + 0j)
    assert q_ext == pytest.approx(2.0, abs=0.05)


def test_non_absorbing_particle_scatters_all_it_extinguishes():
    q_ext, q_sca, _ = mie_efficiencies(7.5, 1.42 + 0j)
    assert q_sca == pytest.approx(q_ext, rel=1e-10)


def test_absorbing_particle_scatters_less_than_it_extinguishes():
    q_ext, q_sca, _ = mie_efficiencies(20.0, 1.4 + 0.01j)
    assert q_sca < q_ext
    # Downward recurrence for D_n is what keeps this case stable; an
    # upward recurrence would return nonsense here rather than fail.
    assert np.isfinite(q_sca) and q_sca > 0


# --------------------------------------------------------------------------
# Internal consistency
# --------------------------------------------------------------------------


def test_angular_integral_matches_the_scattering_efficiency():
    """int (|S1|^2+|S2|^2)/2 dOmega must equal pi x^2 Q_sca.

    The left side comes from the angular functions and the right from
    the coefficient series; they share only a and b, so agreement is
    evidence about the angular recurrences specifically.
    """
    for x, m in [(1.0, 1.33 + 0j), (5.213, 1.55 + 0j), (12.0, 1.2 + 0.005j)]:
        a, b = mie_coefficients(x, m)
        _, q_sca, _ = mie_efficiencies(x, m)
        theta = np.linspace(0.0, np.pi, 40001)
        s1, s2 = mie_amplitudes(np.cos(theta), a, b)
        m11 = 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
        integral = 2.0 * np.pi * _trapezoid(m11 * np.sin(theta), theta)
        assert integral == pytest.approx(np.pi * x**2 * q_sca, rel=1e-5)


def test_asymmetry_parameter_matches_numerical_integration():
    x, m = 5.213, 1.55 + 0j
    a, b = mie_coefficients(x, m)
    _, _, g_series = mie_efficiencies(x, m)
    theta = np.linspace(0.0, np.pi, 40001)
    s1, s2 = mie_amplitudes(np.cos(theta), a, b)
    m11 = 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
    weight = m11 * np.sin(theta)
    g_numeric = _trapezoid(weight * np.cos(theta), theta) / _trapezoid(weight, theta)
    assert g_series == pytest.approx(g_numeric, rel=1e-5)


def test_mueller_matrix_matches_the_amplitude_functions():
    s = MieScatterer(0.3, 633.0, 1.59, 1.33)
    for mu in (-0.7, -0.1, 0.35, 0.9):
        s2, s1 = s.amplitudes(mu)
        mm = s.mueller_matrix(mu)
        assert mm[0, 0] == pytest.approx(0.5 * (abs(s1) ** 2 + abs(s2) ** 2))
        assert mm[0, 1] == pytest.approx(0.5 * (abs(s2) ** 2 - abs(s1) ** 2))
        assert mm[2, 2] == pytest.approx((s2 * np.conj(s1)).real)
        assert mm[2, 3] == pytest.approx((s2 * np.conj(s1)).imag)


def test_size_parameter_uses_the_wavelength_in_the_medium():
    x_air = size_parameter(0.5, 500.0, 1.0)
    x_water = size_parameter(0.5, 500.0, 1.33)
    assert x_water == pytest.approx(1.33 * x_air)


# --------------------------------------------------------------------------
# Cross-implementation
# --------------------------------------------------------------------------


def test_against_an_independent_implementation():
    miepython = pytest.importorskip("miepython")
    for x, m in [(0.1, 1.5 + 0j), (5.213, 1.55 + 0j), (10.0, 1.5 + 0j), (20.0, 1.4 + 0.01j)]:
        q_ext, q_sca, g = mie_efficiencies(x, m)
        ref = miepython.efficiencies_mx(m, x)
        assert q_ext == pytest.approx(ref[0], rel=1e-9)
        assert q_sca == pytest.approx(ref[1], rel=1e-9)
        assert g == pytest.approx(ref[3], rel=1e-8)

    # Angular shape: miepython normalizes S1/S2 differently, so the
    # meaningful comparison is that the ratio is angle-independent.
    x, m = 5.213, 1.55 + 0j
    a, b = mie_coefficients(x, m)
    mu = np.cos(np.linspace(0.05, np.pi - 0.05, 25))
    s1, s2 = mie_amplitudes(mu, a, b)
    ref1, ref2 = miepython.S1_S2(m, x, mu, norm="qsca")
    r1 = np.abs(s1) / np.abs(ref1)
    r2 = np.abs(s2) / np.abs(ref2)
    assert np.allclose(r1, r1[0], rtol=1e-8)
    assert np.allclose(r2, r1[0], rtol=1e-8)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_sampler_reproduces_its_own_phase_function():
    """Sampled <cos theta> must match the asymmetry parameter from the series."""
    s = MieScatterer(0.3, 633.0, 1.59, 1.33)
    rng = np.random.default_rng(0)
    cos_theta = np.array([s.sample_theta(rng.random()) for _ in range(40_000)])
    mean_cos = np.cos(cos_theta).mean()
    stderr = np.cos(cos_theta).std(ddof=1) / np.sqrt(len(cos_theta))
    assert abs(mean_cos - s.g) < 4.0 * stderr


def test_sampler_conserves_intensity_exactly():
    s = MieScatterer(0.4, 633.0, 1.59, 1.33)
    rng = np.random.default_rng(1)
    j1, j2 = 0.6 + 0.2j, -0.3 + 0.5j
    intensity = abs(j1) ** 2 + abs(j2) ** 2
    for _ in range(200):
        _, _, _, j1, j2 = s.sample_jones(j1, j2, rng)
        assert abs(j1) ** 2 + abs(j2) ** 2 == pytest.approx(intensity, rel=1e-12)


def test_jones_and_stokes_samplers_agree_statistically():
    """Same particle, two samplers, one built from S1/S2 and one from the 4x4 matrix."""
    s = MieScatterer(0.25, 633.0, 1.59, 1.33)
    n = 6000

    rng = np.random.default_rng(3)
    j1, j2 = 1.0 + 0j, 0.0 + 0j
    stokes_j = np.zeros(4)
    for _ in range(n):
        _, _, _, a, b = s.sample_jones(j1, j2, rng)
        stokes_j += jones_to_stokes(a, b)

    rng = np.random.default_rng(4)
    start = jones_to_stokes(j1, j2)
    stokes_s = np.zeros(4)
    for _ in range(n):
        _, _, _, out = s.sample_stokes(start.copy(), rng)
        stokes_s += out

    stokes_j /= n
    stokes_s /= n
    # Q/I after one scattering event, the component that actually carries
    # the polarization physics.
    assert stokes_j[1] / stokes_j[0] == pytest.approx(stokes_s[1] / stokes_s[0], abs=0.03)


def test_degree_of_polarization_never_exceeds_unity_under_mie_scattering():
    s = MieScatterer(0.5, 633.0, 1.59, 1.33)
    rng = np.random.default_rng(5)
    j1, j2 = 1.0 / np.sqrt(2), 1j / np.sqrt(2)
    worst = 0.0
    for _ in range(500):
        _, _, _, j1, j2 = s.sample_jones(j1, j2, rng)
        worst = max(worst, degree_of_polarization(jones_to_stokes(j1, j2)) - 1.0)
    assert worst < 1e-12


# --------------------------------------------------------------------------
# Use inside the transport engine
# --------------------------------------------------------------------------


def test_mie_transport_conserves_energy_exactly():
    s = MieScatterer(0.3, 633.0, 1.59, 1.33)
    slab = SlabOpticalProperties(mu_a=0.1, mu_s=3.0, g=0.0, thickness=0.6)
    res = simulate_slab_vector(slab, 633.0, n_photons=1500, n_batches=3,
                               scatterer=s, detector_bins=8, seed=7)
    total = res.diffuse_reflectance + res.transmittance + res.absorbed
    r_sp = ((1.0 - slab.n_medium) / (1.0 + slab.n_medium)) ** 2
    assert total == pytest.approx(1.0 - r_sp, abs=1e-9)


def test_forward_peaked_mie_reflects_less_than_rayleigh():
    """A physical sanity check with an unambiguous direction.

    Same optical depth, same absorption: a g = 0.9 particle sends light
    forward, so diffuse reflectance must drop well below the isotropic
    Rayleigh case. This would catch a sampler that ignored the phase
    function's shape.
    """
    s = MieScatterer(0.4, 633.0, 1.59, 1.33)
    assert s.g > 0.85
    slab = SlabOpticalProperties(mu_a=0.1, mu_s=3.0, g=0.0, thickness=0.6)
    mie = simulate_slab_vector(slab, 633.0, n_photons=1500, n_batches=3,
                               scatterer=s, detector_bins=8, seed=7)
    ray = simulate_slab_vector(slab, 633.0, n_photons=1500, n_batches=3,
                               detector_bins=8, seed=7)
    assert mie.diffuse_reflectance < 0.5 * ray.diffuse_reflectance


def test_rayleigh_single_scattering_preserves_the_linear_direction_exactly():
    """An exact analytic anchor for the depolarization measure.

    A dipole driven along x radiates a field that is the component of x
    perpendicular to the observation direction -- so after *one*
    Rayleigh event the scattered light is still fully linearly
    polarized along the projection of the original axis, in every
    direction. The canonical-frame measure must therefore return
    exactly 1 at order 1, with no statistical scatter at all. Any
    frame-handling error shows up here immediately.
    """
    _, q, _ = depolarization_ladder(n_events=1, n_photons=300, polarization="x", seed=2)
    assert q[0] == pytest.approx(1.0, abs=1e-12)
    assert q[1] == pytest.approx(1.0, abs=1e-12)


def test_polarization_memory_reverses_the_linear_circular_ordering():
    """Rayleigh: linear survives longer. Large Mie: circular does.

    This is the headline claim the Mie extension exists to support, so
    it is asserted as a test rather than only plotted.
    """
    def survival(scatterer):
        _, q, _ = depolarization_ladder(n_events=6, n_photons=400, scatterer=scatterer,
                                        polarization="x", seed=11)
        _, _, v = depolarization_ladder(n_events=6, n_photons=400, scatterer=scatterer,
                                        polarization="circular", seed=12)
        return q[6], v[6]

    lin_r, circ_r = survival(None)
    assert lin_r > circ_r * 1.5          # Rayleigh: linear clearly ahead

    big = MieScatterer(0.4, 633.0, 1.59, 1.33)
    lin_m, circ_m = survival(big)
    assert circ_m > lin_m                 # Mie: the ordering has reversed


def test_invalid_particles_are_rejected():
    with pytest.raises(ValueError):
        MieScatterer(-0.1, 633.0)
    with pytest.raises(ValueError):
        MieScatterer(0.3, -633.0)
    with pytest.raises(ValueError):
        MieScatterer(0.3, 633.0, n_angles=10)
    with pytest.raises(ValueError):
        # Large particle with a grid too coarse to resolve its lobes.
        MieScatterer(5.0, 633.0, 1.59, 1.33, n_angles=4001)
    with pytest.raises(ValueError):
        mie_coefficients(-1.0, 1.5)
    with pytest.raises(ValueError):
        # Negative imaginary index would mean gain in this sign convention.
        mie_coefficients(1.0, 1.5 - 0.1j)


def test_ladder_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        depolarization_ladder(n_events=0)
    with pytest.raises(ValueError):
        depolarization_ladder(n_photons=0)
    with pytest.raises(ValueError):
        depolarization_ladder(frame="meridian")
