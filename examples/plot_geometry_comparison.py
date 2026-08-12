"""
plot_geometry_comparison.py

Produces a single figure that makes the sign-flip result from
test_inverted_geometry.py visually self-evident:

  Top row    -- schematic cross-sections of the two layer geometries
                (strong absorber shallow vs. strong absorber deep),
                with the incident beam and colour-coded absorption.
  Bottom row -- delta_R (layered - homogeneous), with Monte Carlo
                error bars, for both configurations across the same
                set of surface-layer thicknesses. Config A bars sit
                below zero, Config B bars sit above zero at every
                thickness tested -- the sign flip made visible.

Usage: python examples/plot_geometry_comparison.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402
from photon_transport_toolkit.layered_media import Layer, LayeredMedium, simulate_layered_medium  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PHOTONS = 9000
N_BATCHES = 6
SEED = 0
N_INDEX = 1.4

MU_A_LOW = 0.05
MU_A_HIGH = 0.50

SURFACE_MU_S, SURFACE_G = 22.5, 0.80
DEEP_MU_S, DEEP_G = 20.0, 0.90
DEEP_THICKNESS = 1.5

THICKNESSES = [0.05, 0.12, 0.25, 0.50]

RED = "#C00000"
BLUE = "#2E75B6"


def homogeneous_equivalent(surface: Layer, deep: Layer) -> SlabOpticalProperties:
    layers = [surface, deep]
    total = sum(l.thickness for l in layers)
    mu_a_avg = sum(l.mu_a * l.thickness for l in layers) / total
    musp_avg = sum(l.mu_s * (1 - l.g) * l.thickness for l in layers) / total
    g_avg = sum(l.g * l.thickness for l in layers) / total
    mu_s_avg = musp_avg / (1 - g_avg)
    return SlabOpticalProperties(mu_a=mu_a_avg, mu_s=mu_s_avg, g=g_avg,
                                  thickness=total, n_medium=N_INDEX, n_outside=1.0)


def run_config(surface_mu_a: float, deep_mu_a: float, thickness: float):
    surface = Layer(mu_a=surface_mu_a, mu_s=SURFACE_MU_S, g=SURFACE_G,
                     thickness=thickness, n=N_INDEX)
    deep = Layer(mu_a=deep_mu_a, mu_s=DEEP_MU_S, g=DEEP_G,
                 thickness=DEEP_THICKNESS, n=N_INDEX)
    layered = LayeredMedium(layers=[surface, deep])
    homog = homogeneous_equivalent(surface, deep)

    res_l = simulate_layered_medium(layered, n_photons=N_PHOTONS, seed=SEED, n_batches=N_BATCHES)
    res_h = simulate_slab(homog, n_photons=N_PHOTONS, seed=SEED, n_batches=N_BATCHES)

    dR = res_l.diffuse_reflectance - res_h.diffuse_reflectance
    sR = np.hypot(res_l.diffuse_reflectance_stderr, res_h.diffuse_reflectance_stderr)
    return dR, sR


def draw_stack_schematic(ax, strong_shallow: bool, title: str):
    """A simple cross-section: thin surface layer + thick deep layer,
    coloured by absorption strength, with an incident beam arrow."""
    surface_h = 0.9
    deep_h = 3.2
    width = 2.4

    surface_color = RED if strong_shallow else BLUE
    deep_color = BLUE if strong_shallow else RED

    ax.add_patch(mpatches.Rectangle((0, deep_h), width, surface_h,
                                     facecolor=surface_color, edgecolor="black", linewidth=1))
    ax.add_patch(mpatches.Rectangle((0, 0), width, deep_h,
                                     facecolor=deep_color, edgecolor="black", linewidth=1, alpha=0.85))

    # incident beam
    ax.annotate("", xy=(width / 2, deep_h + surface_h), xytext=(width / 2, deep_h + surface_h + 0.9),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    ax.text(width / 2, deep_h + surface_h + 1.05, "incident\nlight", ha="center", va="bottom", fontsize=8)

    label_a = r"$\mu_a$ high" if strong_shallow else r"$\mu_a$ low"
    label_b = r"$\mu_a$ low" if strong_shallow else r"$\mu_a$ high"
    ax.text(width + 0.15, deep_h + surface_h / 2, f"surface layer\n{label_a}\n(thin, varies)",
            fontsize=8, va="center")
    ax.text(width + 0.15, deep_h / 2, f"deep layer\n{label_b}\n(1.5 mm, fixed)",
            fontsize=8, va="center")

    ax.set_xlim(-0.3, width + 2.3)
    ax.set_ylim(-0.3, deep_h + surface_h + 1.4)
    ax.set_title(title, fontsize=10.5)
    ax.axis("off")


def main() -> None:
    dR_A, sR_A, dR_B, sR_B = [], [], [], []
    print(f"{'thickness':>10} {'config A (strong shallow)':>28} {'config B (strong deep)':>26}")
    for t in THICKNESSES:
        a_val, a_err = run_config(MU_A_HIGH, MU_A_LOW, t)
        b_val, b_err = run_config(MU_A_LOW, MU_A_HIGH, t)
        dR_A.append(a_val); sR_A.append(a_err)
        dR_B.append(b_val); sR_B.append(b_err)
        print(f"{t:>10.2f} {a_val:>+20.4f} +/-{a_err:.4f} {b_val:>+18.4f} +/-{b_err:.4f}")

    fig = plt.figure(figsize=(9.5, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.4], hspace=0.15, wspace=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    draw_stack_schematic(ax_a, strong_shallow=True, title="Config A: strong absorber SHALLOW")
    draw_stack_schematic(ax_b, strong_shallow=False, title="Config B: strong absorber DEEP")

    ax_data = fig.add_subplot(gs[1, :])
    x = np.arange(len(THICKNESSES))
    width = 0.35
    ax_data.bar(x - width / 2, dR_A, width, yerr=sR_A, capsize=4,
                color=RED, label="Config A: strong SHALLOW")
    ax_data.bar(x + width / 2, dR_B, width, yerr=sR_B, capsize=4,
                color=BLUE, label="Config B: strong DEEP")
    ax_data.axhline(0, color="black", lw=1)
    ax_data.set_xticks(x)
    ax_data.set_xticklabels([f"{t:.2f} mm" for t in THICKNESSES])
    ax_data.set_xlabel("Surface-layer thickness")
    ax_data.set_ylabel(r"$\Delta R$ = R$_{layered}$ $-$ R$_{homogeneous}$")
    ax_data.set_title("Same contrast, same thicknesses, opposite bias direction")
    ax_data.legend()
    ax_data.grid(axis="y", alpha=0.3)

    fig.suptitle("Where the absorber sits determines the sign of the bulk-averaging bias",
                 fontsize=12.5, y=0.995)
    fig.tight_layout()
    out = FIG_DIR / "geometry_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
