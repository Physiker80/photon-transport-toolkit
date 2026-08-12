"""
Validation of the Monte Carlo transport model against limiting cases with
known analytical answers.

A Monte Carlo code cannot be verified by inspection: it has to be driven into
regimes where the correct answer is known independently, and shown to
reproduce it within its own quoted uncertainty. The tests below do that in
four ways.

  1. Energy conservation. R + T + A must equal 1 exactly, for every parameter
     set, since the weighting scheme is constructed to conserve weight.

  2. Beer-Lambert limit. With no scattering and index-matched boundaries, the
     transmittance must equal exp(-mu_a * L) analytically.

  3. Conservative-scattering limit. With no absorption and index-matched
     boundaries, absorption must be exactly zero and R + T must equal 1,
     regardless of how much scattering takes place.

  4. Forward-scattering limit. As g -> 1 the scattering becomes forward
     peaked and no longer redirects photons, so the transmittance must
     approach the Beer-Lambert result even when mu_s is large.

Run with:  pytest -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402


N_PHOTONS = 40_000


def test_energy_is_conserved():
    """R + T + A = 1 for a representative scattering, absorbing slab."""
    slab = SlabOpticalProperties(mu_a=0.05, mu_s=10.0, g=0.8, thickness=4.0)
    result = simulate_slab(slab, n_photons=N_PHOTONS, seed=1)
    assert result.energy_balance == pytest.approx(1.0, abs=1e-9)


def test_beer_lambert_without_scattering():
    """With mu_s = 0 and matched indices, T = exp(-mu_a * L) analytically."""
    mu_a, thickness = 0.4, 3.0
    slab = SlabOpticalProperties(
        mu_a=mu_a, mu_s=0.0, g=0.0, thickness=thickness, n_medium=1.0, n_outside=1.0
    )
    result = simulate_slab(slab, n_photons=N_PHOTONS, seed=2)

    expected = np.exp(-mu_a * thickness)
    tolerance = max(4.0 * result.transmittance_stderr, 1e-3)
    assert result.transmittance == pytest.approx(expected, abs=tolerance)
    assert result.diffuse_reflectance == pytest.approx(0.0, abs=1e-12)


def test_no_absorption_is_conservative():
    """With mu_a = 0 and matched indices, A = 0 and R + T = 1 exactly."""
    slab = SlabOpticalProperties(
        mu_a=0.0, mu_s=20.0, g=0.5, thickness=2.0, n_medium=1.0, n_outside=1.0
    )
    result = simulate_slab(slab, n_photons=N_PHOTONS, seed=3)

    assert result.absorbed == pytest.approx(0.0, abs=1e-12)
    assert result.diffuse_reflectance + result.transmittance == pytest.approx(1.0, abs=1e-9)


def test_forward_scattering_approaches_beer_lambert():
    """As g -> 1, strong scattering stops redirecting light."""
    mu_a, mu_s, thickness = 0.2, 15.0, 2.0
    slab = SlabOpticalProperties(
        mu_a=mu_a, mu_s=mu_s, g=0.999, thickness=thickness, n_medium=1.0, n_outside=1.0
    )
    result = simulate_slab(slab, n_photons=N_PHOTONS, seed=4)

    expected = np.exp(-mu_a * thickness)
    assert result.transmittance == pytest.approx(expected, rel=0.05)
    assert result.diffuse_reflectance < 0.02


def test_transmittance_decreases_with_scattering():
    """Diffuse transmittance must fall monotonically as mu_s increases."""
    transmittances = []
    for mu_s in (1.0, 4.0, 16.0, 64.0):
        slab = SlabOpticalProperties(mu_a=0.02, mu_s=mu_s, g=0.85, thickness=5.0)
        transmittances.append(simulate_slab(slab, n_photons=20_000, seed=5).transmittance)

    assert all(a > b for a, b in zip(transmittances, transmittances[1:]))


def test_specular_reflectance_matches_normal_incidence_fresnel():
    """Entrance-face specular reflection equals ((n1-n2)/(n1+n2))^2."""
    n1, n2 = 1.0, 1.5
    slab = SlabOpticalProperties(
        mu_a=0.1, mu_s=1.0, g=0.0, thickness=1.0, n_medium=n2, n_outside=n1
    )
    result = simulate_slab(slab, n_photons=2_000, seed=6)

    expected = ((n1 - n2) / (n1 + n2)) ** 2
    assert result.specular_reflectance == pytest.approx(expected, rel=1e-12)


def test_results_are_reproducible_under_a_fixed_seed():
    """The same seed must give bit-identical results."""
    slab = SlabOpticalProperties(mu_a=0.1, mu_s=5.0, g=0.7, thickness=3.0)
    a = simulate_slab(slab, n_photons=5_000, seed=42)
    b = simulate_slab(slab, n_photons=5_000, seed=42)
    assert a == b


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        SlabOpticalProperties(mu_a=-1.0, mu_s=1.0, g=0.0, thickness=1.0)
    with pytest.raises(ValueError):
        SlabOpticalProperties(mu_a=1.0, mu_s=1.0, g=1.0, thickness=1.0)
    with pytest.raises(ValueError):
        SlabOpticalProperties(mu_a=1.0, mu_s=1.0, g=0.0, thickness=0.0)
