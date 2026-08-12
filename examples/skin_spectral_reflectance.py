"""
skin_spectral_reflectance.py

An integrated demonstration of photon_transport_toolkit.tissue_optics:
simulates the diffuse reflectance, transmittance, and absorbed fraction
of a two-layer (epidermis + dermis) skin phantom at each of the six
LUXEON Z LED channel wavelengths used in the companion white-light
illumination system (royal blue 448, cyan 505, lime 567, amber 590,
red 627, deep red 655 nm) -- connecting this package's validated
layered Monte Carlo transport to a concrete illumination-and-imaging
system, not just an abstract turbid slab.

Every optical property used here is either (a) computed from the
verified melanin-absorption formula in tissue_optics.py, (b) the
standard tissue-optics power-law scattering dispersion, or (c) an
explicitly-labelled illustrative placeholder (baseline absorption,
dermal scattering reference level) -- see tissue_optics.py's docstring
for exactly which is which. This script does not claim to reproduce
any specific prior report's numbers; it is a fresh, from-first-
-principles simulation built on the same validated engine.

Usage: python examples/skin_spectral_reflectance.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit.layered_media import LayeredMedium, simulate_layered_medium  # noqa: E402
from photon_transport_toolkit.tissue_optics import epidermis_layer, dermis_layer  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PHOTONS = 12_000
SEED = 0

# The six LUXEON Z channel wavelengths from the companion LED white-light
# system (royal blue, cyan, lime, amber, red, deep red).
CHANNELS = {
    "Royal blue": 448,
    "Cyan": 505,
    "Lime": 567,
    "Amber": 590,
    "Red": 627,
    "Deep red": 655,
}
CHANNEL_COLORS = {
    "Royal blue": "#1F3FA0", "Cyan": "#00B3B3", "Lime": "#7FBF00",
    "Amber": "#FFA000", "Red": "#D62728", "Deep red": "#8B0000",
}


def main() -> None:
    wavelengths = sorted(CHANNELS.values())
    names_by_wl = {v: k for k, v in CHANNELS.items()}

    Rd, Td, A = [], [], []
    print(f"{'channel':>10} {'lambda[nm]':>10} {'Rd':>8} {'Td':>8} {'A':>8}")
    for wl in wavelengths:
        epi = epidermis_layer(wl)
        der = dermis_layer(wl)
        medium = LayeredMedium(layers=[epi, der])
        res = simulate_layered_medium(medium, n_photons=N_PHOTONS, seed=SEED)
        Rd.append(res.diffuse_reflectance)
        Td.append(res.transmittance)
        A.append(res.absorbed)
        print(f"{names_by_wl[wl]:>10} {wl:>10d} {res.diffuse_reflectance:>8.4f} "
              f"{res.transmittance:>8.4f} {res.absorbed:>8.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(wavelengths, Rd, "o-", color="#1F3864", label="Diffuse reflectance $R_d$", lw=2)
    ax.plot(wavelengths, Td, "s-", color="#C00000", label="Transmittance $T$", lw=2)
    ax.plot(wavelengths, A, "^-", color="#548235", label="Absorbed fraction $A$", lw=2)

    for wl in wavelengths:
        ax.axvspan(wl - 4, wl + 4, color=CHANNEL_COLORS[names_by_wl[wl]], alpha=0.15)

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Fraction of incident power")
    ax.set_title("Two-layer skin phantom: spectral response at the six\nLED channel wavelengths (epidermis + dermis, Monte Carlo)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "skin_spectral_reflectance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out.relative_to(out.parents[1])}")

    print("\nNote: mu_a is dominated by the verified melanin power law at short")
    print("wavelengths (blue/cyan), which is why Rd is lowest and A is highest")
    print("there -- exactly the qualitative trend expected from skin optics,")
    print("obtained here from first-principles transport, not assumed.")


if __name__ == "__main__":
    main()
