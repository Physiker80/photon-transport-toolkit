"""
Validation tests for photon_transport_toolkit.tissue_optics.

Checks the melanin absorption formula against a hand-computed reference
value, confirms the expected monotonic wavelength trends (melanin
absorbs more strongly at shorter wavelengths; reduced scattering with
b > 0 does too), and confirms the layer builders produce physically
valid Layer objects.

Run with: pytest -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit.tissue_optics import (  # noqa: E402
    melanin_absorption_mm, reduced_scattering_power_law,
    epidermis_layer, dermis_layer,
)


def test_melanin_absorption_matches_hand_computed_reference():
    """mua_mel(500 nm) = 6.6e11 * 500^-3.33 [1/cm], checked by direct
    computation, then converted to 1/mm."""
    expected_per_cm = 6.6e11 * 500.0 ** (-3.33)
    expected_per_mm = expected_per_cm / 10.0
    assert melanin_absorption_mm(500.0) == pytest.approx(expected_per_mm, rel=1e-9)


def test_melanin_absorption_decreases_with_wavelength():
    """Melanin absorbs much more strongly in the blue than the red --
    the physical basis for skin's characteristic reflectance spectrum."""
    a_blue = melanin_absorption_mm(450.0)
    a_red = melanin_absorption_mm(650.0)
    assert a_blue > a_red
    assert a_blue / a_red > 2.0  # should be a strong, not marginal, effect


def test_melanin_absorption_rejects_invalid_wavelength():
    with pytest.raises(ValueError):
        melanin_absorption_mm(0.0)
    with pytest.raises(ValueError):
        melanin_absorption_mm(-500.0)


def test_reduced_scattering_power_law_decreases_with_wavelength():
    musp_blue = reduced_scattering_power_law(450.0, musp_ref_mm=4.5, b=1.3)
    musp_red = reduced_scattering_power_law(650.0, musp_ref_mm=4.5, b=1.3)
    assert musp_blue > musp_red


def test_reduced_scattering_power_law_at_reference_wavelength():
    """At lambda = lambda_ref, mus' must equal musp_ref exactly."""
    val = reduced_scattering_power_law(550.0, musp_ref_mm=4.5, b=1.3, ref_wavelength_nm=550.0)
    assert val == pytest.approx(4.5)


def test_epidermis_layer_has_higher_absorption_than_dermis_in_blue():
    """Melanin is concentrated in the epidermis; at a strongly
    melanin-absorbed wavelength the epidermis should out-absorb the
    (melanin-free, in this module) dermis baseline."""
    epi = epidermis_layer(450.0, melanin_fraction=0.05)
    der = dermis_layer(450.0)
    assert epi.mu_a > der.mu_a


def test_layer_builders_return_valid_layers():
    for wl in (448, 505, 567, 590, 627, 655):
        epi = epidermis_layer(wl)
        der = dermis_layer(wl)
        for layer in (epi, der):
            assert layer.mu_a > 0
            assert layer.mu_s > 0
            assert -1.0 < layer.g < 1.0
            assert layer.thickness > 0
