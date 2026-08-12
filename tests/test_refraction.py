"""
Validation for _refract_direction() (photon_transport_toolkit.monte_carlo),
added in response to an external code-review finding: layered_media.py
changed which layer a photon was in on transmission across an internal
boundary without ever updating its direction (ux, uy, uz) -- correct
only when refractive indices are matched (the only case ever exercised
by a previously published result in this project, where the resulting
Fresnel reflectance is exactly 0 and no bending is physically needed).

Mirrors matlab/test_refraction.m so the same claim is checked the same
way in both independently-implemented engines.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit.monte_carlo import _refract_direction  # noqa: E402
from photon_transport_toolkit.layered_media import Layer, LayeredMedium, simulate_layered_medium  # noqa: E402


def test_refraction_satisfies_snells_law():
    n1, n2 = 1.0, 1.5
    theta_i = np.radians(30.0)
    ux, uy, uz = np.sin(theta_i), 0.0, np.cos(theta_i)

    ux2, uy2, uz2 = _refract_direction(ux, uy, uz, n1, n2)
    theta_t = np.arccos(abs(uz2))

    assert n1 * np.sin(theta_i) == pytest.approx(n2 * np.sin(theta_t), abs=1e-9)
    assert ux2**2 + uy2**2 + uz2**2 == pytest.approx(1.0, abs=1e-9)


def test_refraction_is_identity_when_indices_match():
    ux, uy, uz = 0.3, 0.1, np.sqrt(1 - 0.3**2 - 0.1**2)
    result = _refract_direction(ux, uy, uz, 1.4, 1.4)
    assert result == (ux, uy, uz)


def test_energy_conserved_with_mismatched_internal_indices():
    """The regime this bug only affects: genuinely different refractive
    indices between layers (every previously published result in this
    project used matched n throughout, where this reduces to a no-op)."""
    layers = [
        Layer(mu_a=0.0, mu_s=8.0, g=0.7, thickness=0.3, n=1.0),
        Layer(mu_a=0.0, mu_s=6.0, g=0.6, thickness=0.5, n=1.6),
        Layer(mu_a=0.0, mu_s=8.0, g=0.7, thickness=0.3, n=1.3),
    ]
    medium = LayeredMedium(layers=layers)
    result = simulate_layered_medium(medium, n_photons=1200, seed=5, n_batches=6)

    total = result.specular_reflectance + result.diffuse_reflectance + result.transmittance + result.absorbed
    se = np.hypot(result.diffuse_reflectance_stderr,
                  np.hypot(result.transmittance_stderr, result.absorbed_stderr))
    assert total == pytest.approx(1.0, abs=max(4 * se, 1e-6))
