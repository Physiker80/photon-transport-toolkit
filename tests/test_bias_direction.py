"""
Confirms a specific physical claim about the layered-vs-homogeneous bias
explored in the skin examples: the direction of the diffuse-reflectance
bias depends on WHERE the absorption contrast sits (shallow vs deep),
not only on its magnitude.

This guards against a natural overgeneralisation -- "a homogeneous model
always overpredicts R" -- which the sign-flip below shows is false in
general: it depends on whether the stronger absorber is placed near the
entrance surface or deep in the medium.

Run with: pytest -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402
from photon_transport_toolkit.layered_media import Layer, LayeredMedium, simulate_layered_medium  # noqa: E402

N_PHOTONS = 4000


def _homogeneous_equivalent(surface: Layer, deep: Layer) -> SlabOpticalProperties:
    layers = [surface, deep]
    total = sum(l.thickness for l in layers)
    mu_a_avg = sum(l.mu_a * l.thickness for l in layers) / total
    musp_avg = sum(l.mu_s * (1 - l.g) * l.thickness for l in layers) / total
    g_avg = sum(l.g * l.thickness for l in layers) / total
    mu_s_avg = musp_avg / (1 - g_avg)
    return SlabOpticalProperties(mu_a=mu_a_avg, mu_s=mu_s_avg, g=g_avg,
                                  thickness=total, n_medium=1.4, n_outside=1.0)


def _delta_r(surface_mu_a: float, deep_mu_a: float) -> tuple[float, float]:
    surface = Layer(mu_a=surface_mu_a, mu_s=22.5, g=0.80, thickness=0.25, n=1.4)
    deep = Layer(mu_a=deep_mu_a, mu_s=20.0, g=0.90, thickness=1.5, n=1.4)
    layered = LayeredMedium(layers=[surface, deep])
    homog = _homogeneous_equivalent(surface, deep)

    res_l = simulate_layered_medium(layered, n_photons=N_PHOTONS, seed=1)
    res_h = simulate_slab(homog, n_photons=N_PHOTONS, seed=1)

    dR = res_l.diffuse_reflectance - res_h.diffuse_reflectance
    sR = np.hypot(res_l.diffuse_reflectance_stderr, res_h.diffuse_reflectance_stderr)
    return dR, sR


def test_bias_direction_flips_with_absorber_placement():
    """Strong absorber shallow vs. deep must give opposite-sign delta_R."""
    dR_shallow, sigma_shallow = _delta_r(surface_mu_a=0.50, deep_mu_a=0.05)
    dR_deep, sigma_deep = _delta_r(surface_mu_a=0.05, deep_mu_a=0.50)

    # Each deviation must itself be a robust (>5 sigma), non-trivial effect.
    assert abs(dR_shallow) > 5 * sigma_shallow
    assert abs(dR_deep) > 5 * sigma_deep

    # And the two must point in opposite directions.
    assert np.sign(dR_shallow) != np.sign(dR_deep)
