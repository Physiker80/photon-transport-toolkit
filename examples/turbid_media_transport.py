"""
Light transport through a turbid slab as a function of scattering strength.

Generates figures/turbid_media_transport.png: the radiometric budget
(diffuse reflectance, transmittance, absorbed fraction) of a 5 mm slab as the
scattering coefficient is swept over two decades, with Monte Carlo error bars.

The sweep spans the transition from the single-scattering regime, where most
light passes through unscattered, to the diffusive regime, where the slab is
optically thick and reflectance dominates. The transport mean free path
1 / mu_s' is annotated to make that transition explicit.

Usage:  python examples/turbid_media_transport.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

MU_A = 0.02
G = 0.85
THICKNESS = 5.0
N_PHOTONS = 12_000


def main() -> None:
    mu_s_values = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])

    reflectance, transmittance, absorbed = [], [], []
    r_err, t_err, a_err = [], [], []

    header = f"{'mu_s':>7} {'mfp\' [mm]':>11} {'R_diff':>10} {'T':>10} {'A':>10}"
    print(header)
    print("-" * len(header))

    for mu_s in mu_s_values:
        slab = SlabOpticalProperties(
            mu_a=MU_A, mu_s=float(mu_s), g=G, thickness=THICKNESS
        )
        res = simulate_slab(slab, n_photons=N_PHOTONS, seed=7)

        reflectance.append(res.diffuse_reflectance)
        transmittance.append(res.transmittance)
        absorbed.append(res.absorbed)
        r_err.append(res.diffuse_reflectance_stderr)
        t_err.append(res.transmittance_stderr)
        a_err.append(res.absorbed_stderr)

        print(f"{mu_s:>7.1f} {1.0 / slab.reduced_scattering:>11.2f} "
              f"{res.diffuse_reflectance:>10.4f} {res.transmittance:>10.4f} "
              f"{res.absorbed:>10.4f}")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    ax.errorbar(mu_s_values, transmittance, yerr=t_err, fmt="o-", capsize=3,
                lw=1.8, color="#2E75B6", label="Diffuse transmittance $T$")
    ax.errorbar(mu_s_values, reflectance, yerr=r_err, fmt="s-", capsize=3,
                lw=1.8, color="#C00000", label="Diffuse reflectance $R$")
    ax.errorbar(mu_s_values, absorbed, yerr=a_err, fmt="^-", capsize=3,
                lw=1.8, color="#548235", label="Absorbed fraction $A$")

    # Mark where the slab becomes optically thick in the transport sense.
    mu_s_diffusive = 1.0 / (THICKNESS * (1.0 - G))
    ax.axvline(mu_s_diffusive, color="grey", ls="--", lw=1.0)
    ax.text(mu_s_diffusive * 1.1, 0.72,
            r"$\ell_{tr} = L$" + "\n(onset of\ndiffusive regime)",
            fontsize=8, color="grey")

    ax.set_xscale("log")
    ax.set_xlabel(r"Scattering coefficient $\mu_s$ [mm$^{-1}$]")
    ax.set_ylabel("Fraction of incident power")
    ax.set_title(
        f"Turbid slab, $L$ = {THICKNESS:.0f} mm, "
        rf"$\mu_a$ = {MU_A} mm$^{{-1}}$, $g$ = {G}"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="center right", fontsize=9)
    ax.set_ylim(0, 0.85)

    fig.tight_layout()
    out = FIG_DIR / "turbid_media_transport.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
