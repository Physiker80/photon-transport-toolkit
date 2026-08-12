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
  * Blood absorption, mu_a(lambda, SO2) from the tabulated molar
    extinction coefficients of oxy- and deoxyhemoglobin compiled by
    Prahl (Oregon Medical Laser Center) [4] — added specifically to
    build examples/hennessy_reproduction.py; earlier versions of this
    module deliberately left blood absorption out for the same reason
    melanin was included and collagen g(lambda) was not: this specific
    tabulated data had not yet been fetched and checked at the time.

References
----------
[1] Jacques, S.L. "Skin Optics." Oregon Medical Laser Center News,
    Jan 1998. https://omlc.org/news/jan98/skinoptics.html
[2] Melanin absorption formula mu(lambda) = 6.6e11 * lambda^-3.33 [1/cm],
    independently cited in US patent literature (e.g. US11045661,
    US9636522) attributing the same Jacques-derived expression.
[3] Jacques, S.L. "Optical properties of biological tissues: a review."
    Phys. Med. Biol. 58, R37-R61 (2013).
[4] Prahl, S.A. "Tabulated Molar Extinction Coefficient for Hemoglobin
    in Water." Oregon Medical Laser Center.
    https://omlc.org/spectra/hemoglobin/summary.html (fetched directly
    for this module; values below are transcribed exactly from that
    table, not estimated or interpolated from a plot).

Author: Noureddin Sedki
License: MIT
"""

from __future__ import annotations

from photon_transport_toolkit.layered_media import Layer

__all__ = ["melanin_absorption_mm", "reduced_scattering_power_law", "epidermis_layer",
          "dermis_layer", "blood_absorption_mm"]


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


# Molar extinction coefficients [cm^-1 / (moles/liter)], transcribed exactly
# from the Prahl/OMLC table (see module docstring, ref [4]) at 20 nm
# spacing, 500-740 nm -- the range relevant to visible-light DRS and to
# examples/hennessy_reproduction.py. Not an estimate: every value below
# is copied directly from the fetched table, not read off a plot or
# interpolated by eye.
_HB_WAVELENGTHS_NM = [500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740]
_EPS_HBO2_CM_PER_M = [20932.8, 24202.4, 53236.0, 32613.2, 50104.0, 3200.0,
                      942.0, 442.0, 319.6, 277.6, 290.0, 348.0, 446.0]
_EPS_HB_CM_PER_M = [20862.0, 31589.6, 46592.0, 53788.0, 37020.0, 14677.2,
                    6509.6, 4345.2, 3226.56, 2407.92, 1794.28, 1325.88, 1115.88]

_HB_MOLAR_MASS_G_PER_MOL = 64_500.0  # g Hb / mole, as used directly in the OMLC conversion


def _interp_extinction(wavelength_nm: float, table: list[float]) -> float:
    if not (_HB_WAVELENGTHS_NM[0] <= wavelength_nm <= _HB_WAVELENGTHS_NM[-1]):
        raise ValueError(
            f"blood_absorption_mm only has verified Prahl/OMLC data for "
            f"{_HB_WAVELENGTHS_NM[0]}-{_HB_WAVELENGTHS_NM[-1]} nm at present "
            f"(got {wavelength_nm} nm); extend _HB_WAVELENGTHS_NM/_EPS_* from "
            f"the full OMLC table before using other wavelengths, rather than "
            f"extrapolating."
        )
    for k in range(len(_HB_WAVELENGTHS_NM) - 1):
        w0, w1 = _HB_WAVELENGTHS_NM[k], _HB_WAVELENGTHS_NM[k + 1]
        if w0 <= wavelength_nm <= w1:
            f = (wavelength_nm - w0) / (w1 - w0)
            return table[k] + f * (table[k + 1] - table[k])
    return table[-1]  # exact match on the last grid point


def blood_absorption_mm(wavelength_nm: float, so2: float, c_hb_g_per_l: float = 2.3) -> float:
    """Absorption coefficient of whole blood at a given SO2, in 1/mm.

    mu_a = 2.303 * [SO2*eps_HbO2(lambda) + (1-SO2)*eps_Hb(lambda)]
                 * c_hb_g_per_l / 64500          (Beer-Lambert, per OMLC's
                                                   own stated conversion,
                                                   ref [4]; result in 1/cm,
                                                   divided by 10 for 1/mm)

    Parameters
    ----------
    wavelength_nm : float
        Currently restricted to 500-740 nm — the span this module has
        actually transcribed verified extinction data for (see the
        tables above). Raises rather than silently extrapolating outside
        that range.
    so2 : float
        Oxygen saturation, the fraction of hemoglobin bound to oxygen,
        0 (fully deoxygenated) to 1 (fully oxygenated).
    c_hb_g_per_l : float
        Hemoglobin concentration in the tissue volume actually being
        modeled (NOT whole-blood concentration ~150 g/L unless the
        "tissue" being modeled is blood itself). The default, 2.3 g/L,
        corresponds to a dermal blood-volume fraction of ~1.5% at whole
        blood Hb = 150 g/L -- a representative, illustrative choice
        within the physiological range typically cited for dermis
        (roughly 0.2-4% across the literature), not a fitted or
        subject-specific value. Override with your own value directly.
    """
    if not 0.0 <= so2 <= 1.0:
        raise ValueError("so2 must be in [0, 1].")
    if c_hb_g_per_l < 0.0:
        raise ValueError("c_hb_g_per_l must be non-negative.")
    eps_hbo2 = _interp_extinction(wavelength_nm, _EPS_HBO2_CM_PER_M)
    eps_hb = _interp_extinction(wavelength_nm, _EPS_HB_CM_PER_M)
    eps_mix = so2 * eps_hbo2 + (1.0 - so2) * eps_hb
    mu_a_per_cm = 2.303 * eps_mix * c_hb_g_per_l / _HB_MOLAR_MASS_G_PER_MOL
    return mu_a_per_cm / 10.0  # 1/cm -> 1/mm
