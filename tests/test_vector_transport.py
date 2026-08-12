"""
Validation for the combined polarization- and phase-resolved engine
(vector_transport.py) -- Phase 2 of the PhD research roadmap.

The organising idea of this file is that the module contains *two*
independent derivations of the same physics -- a 2x2 complex-amplitude
(Jones) tracer and a 4x4 real Mueller-matrix (Stokes) tracer -- written
from different algebra rather than one derived from the other. Most of
the tests below either check one against the other, or check both
against a closed-form result that neither was fitted to:

  * Rayleigh scattering at 90 degrees fully polarizes unpolarized
    light (the blue-sky result) -- exact, no tolerance needed;
  * degree of polarization can never exceed 1 -- an invariant that
    any wrong rotation angle or missing renormalization violates
    immediately;
  * energy is conserved exactly, to 1e-12, preserving the invariant
    the scalar engine deliberately maintains;
  * on ensemble average, the coherently summed intensity equals the
    incoherent sum -- the bridge back to the already-validated
    intensity-only engine.

The known failure mode this file is designed to catch is the tutorial
shortcut of sampling scattering angles isotropically and multiplying
by the Mueller matrix without renormalizing, which silently destroys
one third of the packet energy per Rayleigh event.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties  # noqa: E402
from photon_transport_toolkit.vector_transport import (  # noqa: E402
    _apply_diattenuator,
    _fresnel_amplitudes,
    _initial_jones,
    _renormalize_frame,
    _rotate_frame_about_u,
    _tilt_frame_about_e2,
    degree_of_polarization,
    jones_to_stokes,
    rayleigh_amplitude_matrix,
    rayleigh_mueller_matrix,
    simulate_slab_vector,
    speckle_contrast,
)

SLAB = SlabOpticalProperties(mu_a=0.1, mu_s=6.0, g=0.0, thickness=1.0)
LAUNCHED = 1.0 - abs(_fresnel_amplitudes(1.0, 1.0, SLAB.n_medium)[0]) ** 2


def _random_jones(rng):
    return complex(rng.normal(), rng.normal()), complex(rng.normal(), rng.normal())


# ---------------------------------------------------------------------------
# Algebra: the two formulations must describe the same optics
# ---------------------------------------------------------------------------


def test_mueller_matrix_matches_amplitude_matrix():
    """The 4x4 Rayleigh Mueller matrix and the 2x2 amplitude matrix are
    written independently in the module; applied to the same field they
    must give the same Stokes vector."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        cos_theta = 2.0 * rng.random() - 1.0
        e1, e2 = _random_jones(rng)
        s2, s1 = rayleigh_amplitude_matrix(cos_theta)
        via_jones = jones_to_stokes(s2 * e1, s1 * e2)
        via_mueller = rayleigh_mueller_matrix(cos_theta) @ jones_to_stokes(e1, e2)
        assert via_jones == pytest.approx(via_mueller, abs=1e-12)


def test_diattenuator_mueller_matches_jones():
    """The Fresnel boundary operator, applied in Stokes space, must
    match applying the complex amplitude coefficients to a Jones vector
    -- including the retardance term, which is the part a real-valued
    (intensity-only) Fresnel treatment throws away."""
    rng = np.random.default_rng(2)
    for _ in range(300):
        t_p = complex(rng.normal(), rng.normal())
        t_s = complex(rng.normal(), rng.normal())
        e1, e2 = _random_jones(rng)
        via_jones = jones_to_stokes(t_p * e1, t_s * e2)
        via_stokes = _apply_diattenuator(jones_to_stokes(e1, e2), t_p, t_s)
        assert via_jones == pytest.approx(via_stokes, abs=1e-12)


def test_frame_stays_orthonormal_under_repeated_scattering_rotations():
    """The (e1, e2, u) triad is the module's whole reference-frame
    bookkeeping; if it drifts from orthonormality the polarization
    components silently refer to the wrong axes."""
    rng = np.random.default_rng(3)
    e1, e2, u = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    for _ in range(5000):
        psi = 2.0 * np.pi * rng.random()
        cos_theta = 2.0 * rng.random() - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        e1, e2 = _rotate_frame_about_u(e1, e2, np.cos(psi), np.sin(psi))
        e1, u = _tilt_frame_about_e2(e1, u, cos_theta, sin_theta)
        e1, e2, u = _renormalize_frame(e1, e2, u)
    a1, a2, au = np.array(e1), np.array(e2), np.array(u)
    assert np.linalg.norm(a1) == pytest.approx(1.0, abs=1e-12)
    assert np.dot(a1, a2) == pytest.approx(0.0, abs=1e-12)
    assert np.cross(a1, a2) == pytest.approx(au, abs=1e-12)


# ---------------------------------------------------------------------------
# Closed-form physics neither formulation was fitted to
# ---------------------------------------------------------------------------


def test_rayleigh_ninety_degree_scattering_fully_polarizes_unpolarized_light():
    """The blue-sky result: a dipole cannot radiate along its own axis,
    so unpolarized light scattered through 90 degrees emerges 100%
    linearly polarized. Exact -- any error in the amplitude matrix
    shows up here immediately."""
    rng = np.random.default_rng(4)
    total = np.zeros(4)
    for _ in range(20000):
        e1, e2 = _initial_jones("unpolarized", rng)
        s2, s1 = rayleigh_amplitude_matrix(0.0)
        total += jones_to_stokes(s2 * e1, s1 * e2)
    assert degree_of_polarization(total) == pytest.approx(1.0, abs=1e-12)


def test_forward_scattering_leaves_polarization_untouched():
    """At theta = 0 the Rayleigh matrix is the identity: forward
    scattering is not a polarization event at all."""
    rng = np.random.default_rng(5)
    e1, e2 = _random_jones(rng)
    s2, s1 = rayleigh_amplitude_matrix(1.0)
    assert jones_to_stokes(s2 * e1, s1 * e2) == pytest.approx(jones_to_stokes(e1, e2), abs=1e-12)


def test_total_internal_reflection_is_a_pure_retarder():
    """Beyond the critical angle both Fresnel amplitude coefficients
    have unit modulus but *different* phase -- the Fresnel-rhomb
    effect. This is the physics an unpolarized-average Fresnel
    reflectance (as used by the scalar engine) cannot represent, so it
    is checked explicitly rather than assumed to come out right."""
    critical = np.arcsin(1.0 / 1.4)
    r_s, r_p = _fresnel_amplitudes(np.cos(critical + 0.25), 1.4, 1.0)
    assert abs(r_s) == pytest.approx(1.0, abs=1e-12)
    assert abs(r_p) == pytest.approx(1.0, abs=1e-12)
    retardance = np.angle(r_p) - np.angle(r_s)
    assert abs(retardance) > 0.1  # radians -- a real, not round-off, retardance


def test_below_critical_angle_reflection_agrees_with_the_scalar_engine():
    """The unpolarized-average of the complex amplitude coefficients
    must reproduce the scalar engine's Fresnel reflectance exactly --
    the vector treatment generalises the validated one, it does not
    replace it with something different."""
    from photon_transport_toolkit.monte_carlo import _fresnel_reflectance

    for cos_i in (1.0, 0.9, 0.5, 0.2):
        r_s, r_p = _fresnel_amplitudes(cos_i, 1.0, 1.4)
        vector_average = 0.5 * (abs(r_s) ** 2 + abs(r_p) ** 2)
        assert vector_average == pytest.approx(_fresnel_reflectance(cos_i, 1.0, 1.4), abs=1e-12)


def test_incident_states_are_physical():
    """Every launched Jones vector is fully polarized (a Jones vector
    cannot be anything else), while the 'unpolarized' ensemble averages
    to zero net polarization."""
    rng = np.random.default_rng(6)
    for state in ("x", "y", "circular"):
        e1, e2 = _initial_jones(state, rng)
        stokes = jones_to_stokes(e1, e2)
        assert stokes[0] == pytest.approx(1.0, abs=1e-12)
        assert degree_of_polarization(stokes) == pytest.approx(1.0, abs=1e-12)
    total = np.zeros(4)
    for _ in range(40000):
        total += jones_to_stokes(*_initial_jones("unpolarized", rng))
    assert degree_of_polarization(total) < 0.02


# ---------------------------------------------------------------------------
# Engine-level invariants
# ---------------------------------------------------------------------------


def test_energy_is_conserved_exactly():
    """R + T + A must equal the launched power (1 - specular) to
    machine precision, matching the exact invariant the scalar engine
    maintains. The isotropic-sampling-without-renormalization shortcut
    fails this by ~1/3 per scattering event."""
    res = simulate_slab_vector(SLAB, 633, n_photons=600, seed=8, n_batches=3,
                               detector_bins=8, detector_half_width=5.0)
    total = res.diffuse_reflectance + res.transmittance + res.absorbed
    assert total == pytest.approx(LAUNCHED, abs=1e-12)


def test_degree_of_polarization_never_exceeds_unity():
    """A hard physical bound, violated by any incorrect rotation or
    missing renormalization. Checked at *every* scattering event of
    every photon, not just at exit."""
    for formulation in ("jones", "stokes"):
        res = simulate_slab_vector(SLAB, 633, n_photons=400, seed=9, n_batches=2,
                                   detector_bins=8, detector_half_width=5.0,
                                   formulation=formulation, track_dop=True)
        assert res.max_dop_violation < 1e-9


def test_jones_and_stokes_formulations_agree():
    """Two independent derivations, same seed, same answer.

    They agree to machine precision rather than merely within
    statistics, because the rejection-sampling acceptance ratio is the
    same function of the state in both algebras and so both consume the
    random stream identically -- which makes this a far stronger check
    than a sigma-level comparison would be.
    """
    kwargs = dict(wavelength_nm=633, n_photons=600, seed=10, n_batches=3,
                  detector_bins=8, detector_half_width=5.0)
    a = simulate_slab_vector(SLAB, formulation="jones", **kwargs)
    b = simulate_slab_vector(SLAB, formulation="stokes", **kwargs)
    assert a.diffuse_reflectance == pytest.approx(b.diffuse_reflectance, abs=1e-12)
    assert a.transmittance == pytest.approx(b.transmittance, abs=1e-12)
    assert a.stokes_total == pytest.approx(b.stokes_total, abs=1e-12)
    assert a.intensity_co == pytest.approx(b.intensity_co, abs=1e-12)


def test_stokes_formulation_produces_no_coherent_field():
    """Documents the central limitation being demonstrated: a Stokes
    vector is a quadratic quantity and cannot be summed coherently, so
    the Mueller tracer yields an identically zero field however
    carefully it is run. This is the reason the Jones formulation
    exists, asserted rather than left as a claim in the docstring."""
    res = simulate_slab_vector(SLAB, 633, n_photons=400, seed=11, n_batches=2,
                               detector_bins=8, detector_half_width=5.0,
                               formulation="stokes")
    assert np.all(res.field_co == 0)
    assert np.all(res.field_cross == 0)
    assert res.intensity_co.sum() > 0  # but the incoherent image is real


def test_coherent_sum_reproduces_incoherent_sum_on_average():
    """The bridge back to the validated intensity-only engine: the
    ensemble mean of |sum E|^2 must equal sum |E|^2, since the cross
    terms average to zero for uncorrelated phases. Checked over many
    independent pixels and seeds, not on a single realization -- a
    single one fluctuates by tens of percent, which is real speckle
    statistics, not a bug (the same complication documented for
    coherent_transport.py)."""
    ratios = []
    for seed in range(6):
        res = simulate_slab_vector(SLAB, 633, n_photons=1500, seed=seed, n_batches=3,
                                   detector_bins=16, detector_half_width=1.5)
        coherent = np.abs(res.field_co) ** 2 + np.abs(res.field_cross) ** 2
        incoherent = res.intensity_co + res.intensity_cross
        mask = incoherent > 0
        ratios.append(coherent[mask].sum() / incoherent[mask].sum())
    ratios = np.array(ratios)
    stderr = ratios.std(ddof=1) / np.sqrt(len(ratios))
    assert abs(ratios.mean() - 1.0) < 4.0 * stderr


def test_cross_polarized_signal_grows_with_scattering():
    """Depolarization is the physical content of the polarization
    extension: a thicker, more strongly scattering slab must return a
    lower degree of linear polarization. If the Mueller/Jones algebra
    were wrong in a way that left polarization untouched, this ordering
    would not appear."""
    thin = SlabOpticalProperties(mu_a=0.1, mu_s=2.0, g=0.0, thickness=0.15)
    thick = SlabOpticalProperties(mu_a=0.1, mu_s=10.0, g=0.0, thickness=2.0)
    dolp = []
    for slab in (thin, thick):
        res = simulate_slab_vector(slab, 633, n_photons=2000, seed=12, n_batches=4,
                                   detector_bins=16, detector_half_width=2.0)
        dolp.append(res.stokes_total[1] / res.stokes_total[0])
    assert dolp[0] > dolp[1] > 0.0


def test_nonzero_g_is_rejected():
    """Scattering angles come from the phase function -- Rayleigh
    (<cos theta> = 0 by symmetry) or the supplied Mie scatterer, which
    computes its own g -- so a nonzero slab.g would be silently
    ignored. Refusing it is a deliberate choice: a silently-ignored
    parameter is worse than an error."""
    slab = SlabOpticalProperties(mu_a=0.1, mu_s=6.0, g=0.8, thickness=1.0)
    with pytest.raises(ValueError, match="slab.g must be 0"):
        simulate_slab_vector(slab, 633, n_photons=100, n_batches=2)


def test_speckle_contrast_of_a_known_distribution():
    """Fully developed speckle has negative-exponential intensity
    statistics, for which std equals mean and so C = 1."""
    rng = np.random.default_rng(13)
    field = rng.normal(size=200000) + 1j * rng.normal(size=200000)
    assert speckle_contrast(np.abs(field) ** 2) == pytest.approx(1.0, rel=0.02)
    assert speckle_contrast(np.ones(100)) == pytest.approx(0.0, abs=1e-12)
