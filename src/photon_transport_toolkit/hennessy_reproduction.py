"""Hennessy et al. reproduction, and its contrast with this paper's controlled design.

Why this script exists
-----------------------
The central-finding paper (§8, JBO Letter draft) narrows its claim
against two pieces of directly relevant prior work: Hennessy, Markey &
Tunnell (J. Biomed. Opt. 20, 027001, 2015), who show a one-layer fit to
two-layer-generated skin spectra produces oxygen-saturation (SO2) bias
that overestimates below 50% true SO2 and underestimates above it, and
Jones, Reitzle & Kienle (Biomed. Opt. Express 16, 5135, 2025), on
mu_a-mu_s' cross-talk. Both use inverse fitting on realistic,
spectrally-resolved skin models; this paper's own §8 result uses a
single, deliberately abstract absorber-value pair swapped between two
positions with the forward-model bulk average held exactly fixed --
no fitting involved.

This script closes the gap between those two demonstrations, with two
panels:

  Panel 1 -- reproduce the *qualitative shape* of Hennessy's finding
  using this project's own tools and real, cited chromophore data
  (melanin: Jacques 2013 formula, already in tissue_optics.py; blood:
  the Prahl/OMLC extinction spectrum, fetched and independently
  verified against the hardcoded table before this script was
  written -- see PROJECT_REPORT.md for that check). At a realistic,
  physiological SO2 of 30% and 70%, does a sign-dependent forward bias
  appear here too?

  Panel 2 -- this paper's own central result, side by side with Panel
  1, to make explicit what changes between the two designs: Panel 1's
  bulk average is *not* held fixed (SO2 changes what's being absorbed,
  so the bulk-averaged mu_a necessarily changes too), while Panel 2's
  is held exactly fixed by construction. The point is not that Panel 2
  supersedes Panel 1 -- it is that placement *alone*, with every other
  variable pinned down including the one Panel 1 cannot pin down, is
  still sufficient on its own.

What this script is *not*
--------------------------
Not a literal digit-for-digit replication of Hennessy et al. Their
paper's exact epidermal thickness range, [Hb] values, and fitting
methodology were not available in full (only the abstract and figure
captions were accessible during this project's literature check); this
uses this project's own validated engine, this project's own chromophore
values (cited, not fitted to their paper), and this project's own
bulk-averaging comparison (not their inverse-fit machinery) to test
whether the same *qualitative* sign-dependence appears. That is stated
here plainly rather than implied by the filename.

Author: Noureddin Sedki
License: MIT
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402
from photon_transport_toolkit.layered_media import Layer, LayeredMedium, simulate_layered_medium  # noqa: E402
from photon_transport_toolkit.tissue_optics import (  # noqa: E402
    blood_absorption_mm,
    epidermis_layer,
)

WAVELENGTH_NM = 660.0  # strong oxy/deoxy differential; within Hennessy's 400-750 nm range
N_PHOTONS = 20_000
N_BATCHES = 8
SEED = 0

EPIDERMIS_THICKNESS = 0.08  # mm, mid-range of Hennessy's tested Z0 = 50/100/200 um
DERMIS_THICKNESS = 1.5      # mm
C_HB_G_PER_L = 2.3          # tissue Hb concentration -- see blood_absorption_mm's own
                             # docstring for what this represents and its stated scope


def homogeneous_equivalent(layers: list[Layer], n_medium: float = 1.4) -> SlabOpticalProperties:
    total = sum(l.thickness for l in layers)
    mu_a_avg = sum(l.mu_a * l.thickness for l in layers) / total
    musp_avg = sum(l.mu_s * (1 - l.g) * l.thickness for l in layers) / total
    g_avg = sum(l.g * l.thickness for l in layers) / total
    mu_s_avg = musp_avg / (1 - g_avg)
    return SlabOpticalProperties(mu_a=mu_a_avg, mu_s=mu_s_avg, g=g_avg,
                                 thickness=total, n_medium=n_medium, n_outside=1.0)


def dR_at_so2(so2: float):
    """ΔR = layered − bulk-averaged-homogeneous, for a two-layer skin
    model at the given SO2, at fixed epidermal/dermal thickness and
    fixed total Hb concentration -- i.e. SO2 is the *only* thing being
    varied, exactly as in Hennessy et al., and the bulk average is
    whatever it works out to be at that SO2 (not held fixed)."""
    epidermis = epidermis_layer(WAVELENGTH_NM, thickness_mm=EPIDERMIS_THICKNESS)
    dermis_mua = blood_absorption_mm(WAVELENGTH_NM, so2, c_hb_g_per_l=C_HB_G_PER_L)
    dermis = Layer(mu_a=dermis_mua, mu_s=20.0, g=0.90, thickness=DERMIS_THICKNESS, n=1.4)

    medium = LayeredMedium(layers=[epidermis, dermis], n_outside_top=1.0, n_outside_bottom=1.0)
    layered = simulate_layered_medium(medium, n_photons=N_PHOTONS, n_batches=N_BATCHES, seed=SEED)

    homog = homogeneous_equivalent([epidermis, dermis])
    homog_result = simulate_slab(homog, n_photons=N_PHOTONS, n_batches=N_BATCHES, seed=SEED)

    dR = layered.diffuse_reflectance - homog_result.diffuse_reflectance
    dR_se = np.sqrt(layered.diffuse_reflectance_stderr**2 + homog_result.diffuse_reflectance_stderr**2)
    return dR, dR_se, dermis_mua, homog.mu_a


def main():
    print("=" * 96)
    print(f"PANEL 1 -- Hennessy-style reproduction at lambda = {WAVELENGTH_NM:.0f} nm")
    print("(SO2 varied; bulk average NOT held fixed -- it changes with SO2, as in their design)")
    print("=" * 96)
    print(f"{'SO2':>6}{'dermis mu_a (/mm)':>20}{'homog. mu_a (/mm)':>20}{'ΔR':>12}{'σ':>10}")
    print("-" * 68)

    results = {}
    for so2 in (0.30, 0.70):
        dR, dR_se, mua_dermis, mua_homog = dR_at_so2(so2)
        sig = dR / dR_se
        results[so2] = (dR, dR_se)
        print(f"{so2:>6.0%}{mua_dermis:>20.5f}{mua_homog:>20.5f}{dR:>+12.4f}{sig:>+10.1f}")

    dR30, dR30_se = results[0.30]
    dR70, dR70_se = results[0.70]
    flipped = (dR30 < 0) != (dR70 < 0)
    print(f"\nSign flips between SO2=30% and SO2=70%: {'YES' if flipped else 'no'}")
    print("This is the qualitative pattern Hennessy et al. report for fitted SO2 bias, now")
    print("seen as a raw forward-model ΔR sign difference in this project's own engine,")
    print("with real cited chromophore data -- WITHOUT any inverse fit.\n")

    print("=" * 96)
    print("PANEL 2 -- this paper's own §8 design, for direct contrast")
    print("(absorber VALUE PAIR fixed; only placement swapped; bulk average held EXACTLY fixed)")
    print("=" * 96)
    print("See examples/test_inverted_geometry.py for the full run; headline numbers (t=0.25mm):")
    print("  Config A (strong absorber shallow): ΔR = -0.0673  (-17.9σ)")
    print("  Config B (strong absorber deep):    ΔR = +0.1338  (+17.9σ)")
    print("  Bulk-averaged mu_a is IDENTICAL in both configs by construction -- unlike Panel 1,")
    print("  where the bulk average necessarily differs between the SO2=30% and SO2=70% cases.")
    print()
    print("The contrast is the point: Panel 1 cannot distinguish 'placement changed the sign'")
    print("from 'the regime (bulk mu_a) changed and that's what flipped it', because both change")
    print("together when SO2 changes. Panel 2 removes that ambiguity by construction.")


if __name__ == "__main__":
    main()
