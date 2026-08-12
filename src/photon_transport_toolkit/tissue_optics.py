"""
Spectral tissue-optics parameterization for skin, built on top of
opticslab.layered_media.

This module deliberately includes ONLY formulas that have been checked
against an independent, citable source during development (see the
docstring of each function). Several quantities commonly quoted for
skin optics (e.g. a specific collagen g(lambda) fit, or a specific
wavelength-dependent refractive-index dispersion curve) were considered
for inclusion and explicitly LEFT OUT because their exact published
coefficients were not independently verified — adding an unverified
formula with a confident-looking docstring would be worse than leaving
the gap visible. Callers who need those quantities should supply their
own verified values directly to Layer(...).

What IS included and verified:

  * Melanin absorption, mua_mel(lambda) = 6.6e11 * lambda[nm]^-3.33 [1/cm],
    traced to S.L. Jacques' skin-optics summary (Oregon Medical Laser
    Center) and cross-confirmed against an independent secondary source
    (US patent literature citing the same Jacques formula) [1, 2].
  * A standard, textbook power-law wavelength dependence for reduced
    scattering, mus'(lambda) = mus'(lambda_ref) * (lambda/lambda_ref)^-b,
    the generic Mie/Rayleigh-regime scaling used throughout tissue
    optics (e.g. already used with this exact functional form in
    examples/skin_layered_vs_homogeneous.py) [3].

References
----------
[1] Jacques, S.L. "Skin Optics." Oregon Medical Laser Center News,
    Jan 1998. https://omlc.org/news/jan98/skinoptics.html
[2] Melanin absorption formula mu(lambda) = 6.6e11 * lambda^-3.33 [1/cm],
    independently cited in US patent literature (e.g. US11045661,
    US9636522) attributing the same Jacques-derived expression.
[3] Jacques, S.L. "Optical properties of biological tissues: a review."
    Phys. Med. Biol. 58, R37-R61 (2013).

Author: Noureddin Sedki
License: MIT
"""

from __future__ import annotations

from photon_transport_toolkit.layered_media import Layer

__all__ = ["melanin_absorption_mm", "reduced_scattering_power_law", "epidermis_layer", "dermis_layer"]


def melanin_absorption_mm(wavelength_nm: float) -> float:
    """Absorption coefficient of pure melanin at the given wavelength, in 1/mm.

    mua_mel(lambda) = 6.6e11 * lambda[nm]^-3.33  [1/cm]  (Jacques [1, 2])

    This is the absorption coefficient of melanin itself, not of the
    epidermis as a whole — multiply by a melanosome volume fraction
    (typically 0.01-0.15 for lightly to heavily pigmented skin) to get
    a contribution to epidermal mu_a. See epidermis_layer().
    """
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    mua_mel_per_cm = 6.6e11 * wavelength_nm ** (-3.33)
    return mua_mel_per_cm / 10.0  # cm^-1 -> mm^-1


def reduced_scattering_power_law(wavelength_nm: float, musp_ref_mm: float,
                                  b: float = 1.3, ref_wavelength_nm: float = 550.0) -> float:
    """Reduced scattering coefficient mus'(lambda) via the standard
    tissue-optics power law, mus'(lambda) = musp_ref * (lambda/lambda_ref)^-b.

    b is typically in the range ~1.0-2.0 for tissue (Rayleigh-to-Mie
    mixture); the default of 1.3 matches the illustrative value used
    in examples/skin_layered_vs_homogeneous.py. Shorter wavelengths
    scatter more strongly than longer ones for any b > 0, which is why
    blue light is scattered more than red in skin.
    """
    if wavelength_nm <= 0 or ref_wavelength_nm <= 0:
        raise ValueError("wavelengths must be positive.")
    return musp_ref_mm * (wavelength_nm / ref_wavelength_nm) ** (-b)


def epidermis_layer(wavelength_nm: float, thickness_mm: float = 0.10,
                     melanin_fraction: float = 0.02, baseline_mua_mm: float = 0.02,
                     musp_ref_mm: float = 4.5, b: float = 1.3,
                     g: float = 0.80, n: float = 1.4) -> Layer:
    """Build an epidermis Layer at a given wavelength.

    mu_a = melanin_fraction * melanin_absorption_mm(wavelength_nm) + baseline_mua_mm
    mu_s = mus'(wavelength_nm) / (1 - g), via reduced_scattering_power_law()

    baseline_mua_mm (melanin-free epidermal absorption, mostly from
    other chromophores and water) is a small illustrative placeholder,
    not a value traced to a specific verified source — override it if
    you have a better one.
    """
    mua = melanin_fraction * melanin_absorption_mm(wavelength_nm) + baseline_mua_mm
    musp = reduced_scattering_power_law(wavelength_nm, musp_ref_mm, b)
    mus = musp / (1 - g)
    return Layer(mu_a=mua, mu_s=mus, g=g, thickness=thickness_mm, n=n)


def dermis_layer(wavelength_nm: float, thickness_mm: float = 1.5,
                  baseline_mua_mm: float = 0.05, musp_ref_mm: float = 2.0,
                  b: float = 1.3, g: float = 0.90, n: float = 1.4) -> Layer:
    """Build a dermis Layer at a given wavelength.

    baseline_mua_mm bundles blood and other dermal absorption into a
    single illustrative constant (not wavelength-resolved) — this
    module deliberately does NOT include a blood-absorption spectral
    model, since the specific Hb/HbO2 extinction data needed for one
    was not independently verified during development. Supply your
    own wavelength-dependent value here if you have verified data.
    """
    musp = reduced_scattering_power_law(wavelength_nm, musp_ref_mm, b)
    mus = musp / (1 - g)
    return Layer(mu_a=baseline_mua_mm, mu_s=mus, g=g, thickness=thickness_mm, n=n)
