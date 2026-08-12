"""
Validation of the layered-medium Monte Carlo model.

The single most important check here is the reduction test: a stack of
N layers that all share identical optical properties and refractive
index, with total thickness equal to a reference single-layer slab,
must reproduce the single-layer model's R/T/A within statistical
uncertainty. This is the layered code's equivalent of the analytical
limiting-case tests in test_monte_carlo.py — it does not prove the
layered algorithm is correct in general, but it proves that the
generalisation collapses correctly onto the already-validated
single-layer case, which is a necessary (though not sufficient)
condition for correctness.

Exact bit-for-bit agreement is neither expected nor required here: the
layered algorithm makes additional Fresnel-reflectance draws at every
internal (zero-contrast) interface that the single-layer code never
encounters, so the two implementations consume the random-number
stream differently even under a shared seed. Statistical agreement
within a generous multiple of the combined standard error is the
correct standard.

Run with: pytest -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402
from photon_transport_toolkit.layered_media import (  # noqa: E402
    Layer,
    LayeredMedium,
    simulate_layered_medium,
)

N_PHOTONS = 15_000


def test_identical_layers_reduce_to_homogeneous_slab():
    """Three identical layers must match the single-layer result."""
    mu_a, mu_s, g, n = 0.05, 8.0, 0.8, 1.4
    total_thickness = 3.0

    slab = SlabOpticalProperties(
        mu_a=mu_a, mu_s=mu_s, g=g, thickness=total_thickness,
        n_medium=n, n_outside=1.0,
    )
    reference = simulate_slab(slab, n_photons=N_PHOTONS, seed=1)

    medium = LayeredMedium(
        layers=[Layer(mu_a, mu_s, g, total_thickness / 3, n) for _ in range(3)],
        n_outside_top=1.0, n_outside_bottom=1.0,
    )
    layered = simulate_layered_medium(medium, n_photons=N_PHOTONS, seed=1)

    tol_r = 4 * np.hypot(reference.diffuse_reflectance_stderr, layered.diffuse_reflectance_stderr)
    tol_t = 4 * np.hypot(reference.transmittance_stderr, layered.transmittance_stderr)
    tol_a = 4 * np.hypot(reference.absorbed_stderr, layered.absorbed_stderr)

    assert layered.diffuse_reflectance == pytest.approx(reference.diffuse_reflectance, abs=max(tol_r, 5e-3))
    assert layered.transmittance == pytest.approx(reference.transmittance, abs=max(tol_t, 5e-3))
    assert layered.absorbed == pytest.approx(reference.absorbed, abs=max(tol_a, 5e-3))
    assert layered.specular_reflectance == pytest.approx(reference.specular_reflectance, rel=1e-12)


def test_single_layer_stack_matches_slab_exactly_in_expectation():
    """A LayeredMedium with exactly one layer is just the slab model."""
    mu_a, mu_s, g, n = 0.1, 5.0, 0.7, 1.33
    thickness = 2.0

    slab = SlabOpticalProperties(mu_a=mu_a, mu_s=mu_s, g=g, thickness=thickness, n_medium=n)
    reference = simulate_slab(slab, n_photons=N_PHOTONS, seed=2)

    medium = LayeredMedium(layers=[Layer(mu_a, mu_s, g, thickness, n)])
    layered = simulate_layered_medium(medium, n_photons=N_PHOTONS, seed=2)

    tol = 4 * max(reference.diffuse_reflectance_stderr, layered.diffuse_reflectance_stderr, 1e-3)
    assert layered.diffuse_reflectance == pytest.approx(reference.diffuse_reflectance, abs=tol)
    assert layered.transmittance == pytest.approx(reference.transmittance, abs=tol)


def test_energy_is_conserved_across_layers():
    medium = LayeredMedium(layers=[
        Layer(mu_a=0.4, mu_s=22.5, g=0.8, thickness=0.1, n=1.4),
        Layer(mu_a=0.05, mu_s=20.0, g=0.9, thickness=1.5, n=1.4),
    ])
    result = simulate_layered_medium(medium, n_photons=N_PHOTONS, seed=3)
    assert result.energy_balance == pytest.approx(1.0, abs=1e-9)


def test_index_matched_internal_boundary_is_transparent():
    """When n is equal across two layers, the internal boundary must not
    reflect: splitting one layer into two identical-index sub-layers with
    the same combined properties must reproduce the single-layer result,
    which is the n-matched special case of the reduction test above but
    checked for a different (asymmetric) parameter set for robustness."""
    mu_a, mu_s, g, n = 0.02, 12.0, 0.85, 1.0
    total_thickness = 4.0

    slab = SlabOpticalProperties(mu_a=mu_a, mu_s=mu_s, g=g, thickness=total_thickness, n_medium=n)
    reference = simulate_slab(slab, n_photons=N_PHOTONS, seed=4)

    medium = LayeredMedium(layers=[
        Layer(mu_a, mu_s, g, total_thickness * 0.3, n),
        Layer(mu_a, mu_s, g, total_thickness * 0.7, n),
    ])
    layered = simulate_layered_medium(medium, n_photons=N_PHOTONS, seed=4)

    tol = 4 * max(reference.diffuse_reflectance_stderr, layered.diffuse_reflectance_stderr, 1e-3)
    assert layered.diffuse_reflectance == pytest.approx(reference.diffuse_reflectance, abs=tol)


def test_boundaries_property_matches_cumulative_thickness():
    medium = LayeredMedium(layers=[
        Layer(0.1, 5.0, 0.8, 0.5, 1.4),
        Layer(0.1, 5.0, 0.8, 1.5, 1.4),
        Layer(0.1, 5.0, 0.8, 2.0, 1.4),
    ])
    b = medium.boundaries
    assert b[0] == pytest.approx(0.0)
    assert b[1] == pytest.approx(0.5)
    assert b[2] == pytest.approx(2.0)
    assert b[3] == pytest.approx(4.0)
    assert medium.total_thickness == pytest.approx(4.0)


def test_invalid_layer_parameters_are_rejected():
    with pytest.raises(ValueError):
        Layer(mu_a=-1.0, mu_s=1.0, g=0.0, thickness=1.0)
    with pytest.raises(ValueError):
        Layer(mu_a=0.1, mu_s=1.0, g=0.0, thickness=0.0)


def test_empty_medium_is_rejected():
    with pytest.raises(ValueError):
        LayeredMedium(layers=[])
