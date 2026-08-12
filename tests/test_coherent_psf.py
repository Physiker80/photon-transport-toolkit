"""
Validation for the coherent-PSF imaging extension added to
coherent_transport.py: convolving the raw exit-surface field with an
imaging system's coherent (amplitude) point-spread function, so the
resulting |field|^2 shows spatially-correlated speckle grains of a
realistic size, rather than the pixel-to-pixel-uncorrelated raw field
simulate_slab_coherent() produces on its own.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit.coherent_transport import (  # noqa: E402
    airy_psf_kernel,
    apply_coherent_psf,
)


def test_airy_kernel_is_normalised():
    """sum(h**2) == 1 is the normalisation apply_coherent_psf's energy-
    conservation property depends on -- checked directly here, not
    assumed from the derivation alone."""
    k = airy_psf_kernel(pixel_size_mm=0.01, wavelength_nm=632.8,
                         numerical_aperture=0.05, kernel_half_size=30)
    assert np.sum(k**2) == pytest.approx(1.0, abs=1e-9)


def test_airy_kernel_peak_is_at_center():
    k = airy_psf_kernel(pixel_size_mm=0.01, wavelength_nm=632.8,
                         numerical_aperture=0.05, kernel_half_size=20)
    center = k.shape[0] // 2
    assert k[center, center] == k.max()


def test_smaller_na_gives_wider_kernel():
    """Physical sanity check: lower NA -> larger diffraction spot ->
    energy spread over more of the (normalised) kernel, i.e. a lower,
    broader peak."""
    k_low_na = airy_psf_kernel(pixel_size_mm=0.01, wavelength_nm=632.8,
                                numerical_aperture=0.01, kernel_half_size=40)
    k_high_na = airy_psf_kernel(pixel_size_mm=0.01, wavelength_nm=632.8,
                                 numerical_aperture=0.1, kernel_half_size=40)
    center = k_low_na.shape[0] // 2
    assert k_low_na[center, center] < k_high_na[center, center]


def test_psf_convolution_conserves_energy_for_a_random_phase_field():
    """The central physical property: convolving a spatially
    uncorrelated ('white', random-phase) field with the normalised
    coherent PSF must preserve total energy sum(|E|**2) in expectation
    -- checked directly on a synthetic random field far from any edge
    truncation, isolating the convolution's own correctness from the
    small-field edge effects a real simulated speckle field also has."""
    rng = np.random.default_rng(0)
    n = 200
    field = np.sqrt(rng.random((n, n))) * np.exp(1j * 2 * np.pi * rng.random((n, n)))
    energy_before = np.sum(np.abs(field) ** 2)

    imaged = apply_coherent_psf(field, pixel_size_mm=0.01, wavelength_nm=632.8,
                                 numerical_aperture=0.05, kernel_half_size=30)
    energy_after = np.sum(np.abs(imaged) ** 2)

    assert energy_after / energy_before == pytest.approx(1.0, abs=0.02)


def test_convolution_actually_correlates_neighbouring_pixels():
    """Direct evidence the PSF step does what it claims: a single
    bright pixel in an otherwise-empty field should spread into its
    neighbours after convolution (that spreading is the mechanism that
    turns uncorrelated single-photon pixels into resolvable grains)."""
    field = np.zeros((41, 41), dtype=complex)
    field[20, 20] = 1.0
    imaged = apply_coherent_psf(field, pixel_size_mm=0.05, wavelength_nm=632.8,
                                 numerical_aperture=0.02, kernel_half_size=15)
    intensity = np.abs(imaged) ** 2
    assert intensity[20, 21] > 0  # immediate neighbour now has nonzero intensity
    assert intensity[20, 21] < intensity[20, 20]  # but less than the center


def test_invalid_psf_parameters_are_rejected():
    with pytest.raises(ValueError):
        airy_psf_kernel(pixel_size_mm=0, wavelength_nm=633, numerical_aperture=0.05)
    with pytest.raises(ValueError):
        airy_psf_kernel(pixel_size_mm=0.01, wavelength_nm=633, numerical_aperture=1.5)
    with pytest.raises(ValueError):
        airy_psf_kernel(pixel_size_mm=0.01, wavelength_nm=633, numerical_aperture=0.05,
                         kernel_half_size=0)
