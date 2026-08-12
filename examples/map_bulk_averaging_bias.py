"""
map_bulk_averaging_bias.py

Systematically maps the size of the homogeneous-vs-layered bias explored
in skin_layered_vs_homogeneous.py, as a function of two physical
parameters:

  * epidermal thickness (mm) — how thin the superficial absorbing layer is
    relative to the dermis beneath it;
  * absorption contrast (dimensionless) — how much more strongly the
    epidermis absorbs than the dermis, mu_a_epidermis / mu_a_dermis.

For every (thickness, contrast) point on the grid, two Monte Carlo runs
are compared under identical incident conditions and a shared seed:

  1. the true two-layer medium (epidermis over dermis), and
  2. its single-layer "homogeneous equivalent", built from the exact
     thickness-weighted average of mu_a, the reduced scattering
     mu_s' = mu_s(1-g), and g — the natural quantities to average since
     they add linearly along the optical path.

The output is a bias map: delta_R and delta_A (layered minus
homogeneous) over the (thickness, contrast) plane, together with a
sigma-significance contour marking where the bias becomes larger than
the Monte Carlo noise floor at this photon budget — an explicit,
quantitative boundary for when "a homogeneous model is close enough"
stops being a safe assumption.

Dermis properties are held fixed at the literature-informed values used
in skin_layered_vs_homogeneous.py (mu_a=0.05 /mm, mu_s=20.0 /mm, g=0.9,
thickness=1.5 mm); only the epidermis's thickness and absorption are
swept, with its scattering (mu_s=22.5 /mm) and anisotropy (g=0.8) held
fixed, isolating the two parameters of interest.

Runtime note: this is a genuine parameter sweep (2 x grid_size^2 Monte
Carlo runs), not a quick example — expect several minutes depending on
grid resolution and N_PHOTONS.

Usage: python examples/map_bulk_averaging_bias.py
"""

import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402
from photon_transport_toolkit.layered_media import Layer, LayeredMedium, simulate_layered_medium  # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PHOTONS = 5000
N_BATCHES = 5
SEED = 0
N_INDEX = 1.4

# Fixed dermis (same values as skin_layered_vs_homogeneous.py).
DERMIS_MU_A = 0.05
DERMIS_MU_S = 20.0
DERMIS_G = 0.90
DERMIS_THICKNESS = 1.5

# Fixed epidermal optical texture; only thickness and mu_a are swept.
EPIDERMIS_MU_S = 22.5
EPIDERMIS_G = 0.80

# Sweep grid.
THICKNESSES = np.array([0.02, 0.05, 0.12, 0.25, 0.50])   # mm
CONTRASTS = np.array([1.0, 3.0, 6.0, 10.0, 16.0])         # mu_a_epi / mu_a_derm


def homogeneous_equivalent(epidermis: Layer, dermis: Layer) -> SlabOpticalProperties:
    """Thickness-weighted single-layer equivalent of a two-layer stack."""
    layers = [epidermis, dermis]
    total = sum(l.thickness for l in layers)
    mu_a_avg = sum(l.mu_a * l.thickness for l in layers) / total
    musp_avg = sum(l.mu_s * (1 - l.g) * l.thickness for l in layers) / total
    g_avg = sum(l.g * l.thickness for l in layers) / total
    mu_s_avg = musp_avg / (1 - g_avg)
    return SlabOpticalProperties(
        mu_a=mu_a_avg, mu_s=mu_s_avg, g=g_avg, thickness=total,
        n_medium=N_INDEX, n_outside=1.0,
    )


def main() -> None:
    nT, nC = len(THICKNESSES), len(CONTRASTS)
    delta_R = np.zeros((nC, nT))
    delta_A = np.zeros((nC, nT))
    sigma_R = np.zeros((nC, nT))
    sigma_A = np.zeros((nC, nT))

    t_start = time.time()
    total_points = nT * nC
    point = 0

    for i, contrast in enumerate(CONTRASTS):
        for j, thickness in enumerate(THICKNESSES):
            point += 1
            epidermis = Layer(
                mu_a=contrast * DERMIS_MU_A, mu_s=EPIDERMIS_MU_S,
                g=EPIDERMIS_G, thickness=thickness, n=N_INDEX,
            )
            dermis = Layer(
                mu_a=DERMIS_MU_A, mu_s=DERMIS_MU_S,
                g=DERMIS_G, thickness=DERMIS_THICKNESS, n=N_INDEX,
            )

            layered = LayeredMedium(layers=[epidermis, dermis])
            homog = homogeneous_equivalent(epidermis, dermis)

            res_l = simulate_layered_medium(
                layered, n_photons=N_PHOTONS, seed=SEED, n_batches=N_BATCHES
            )
            res_h = simulate_slab(
                homog, n_photons=N_PHOTONS, seed=SEED, n_batches=N_BATCHES
            )

            dR = res_l.diffuse_reflectance - res_h.diffuse_reflectance
            dA = res_l.absorbed - res_h.absorbed
            sR = np.hypot(res_l.diffuse_reflectance_stderr, res_h.diffuse_reflectance_stderr)
            sA = np.hypot(res_l.absorbed_stderr, res_h.absorbed_stderr)

            delta_R[i, j] = dR
            delta_A[i, j] = dA
            sigma_R[i, j] = dR / sR if sR > 0 else 0.0
            sigma_A[i, j] = dA / sA if sA > 0 else 0.0

            elapsed = time.time() - t_start
            eta = elapsed / point * (total_points - point)
            print(f"[{point:3d}/{total_points}] thickness={thickness:.2f} mm  "
                  f"contrast={contrast:5.1f}  dR={dR:+.4f} ({sigma_R[i,j]:+.1f} sigma)  "
                  f"dA={dA:+.4f} ({sigma_A[i,j]:+.1f} sigma)  "
                  f"[elapsed {elapsed:5.0f}s, eta {eta:5.0f}s]")

    np.savez(FIG_DIR / "bulk_averaging_bias_grid.npz",
             thicknesses=THICKNESSES, contrasts=CONTRASTS,
             delta_R=delta_R, delta_A=delta_A, sigma_R=sigma_R, sigma_A=sigma_A)

    # ---------------------------------------------------------------- plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, data, sig, label, cmap in [
        (axes[0], delta_R, sigma_R, r"$\Delta R$ (layered $-$ homogeneous)", "RdBu_r"),
        (axes[1], delta_A, sigma_A, r"$\Delta A$ (layered $-$ homogeneous)", "RdBu_r"),
    ]:
        vmax = np.abs(data).max()
        im = ax.pcolormesh(THICKNESSES, CONTRASTS, data, shading="nearest",
                            cmap=cmap, vmin=-vmax, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(label)

        # 3-sigma significance boundary: below this, the bias is not
        # distinguishable from Monte Carlo noise at this photon budget.
        cs = ax.contour(THICKNESSES, CONTRASTS, np.abs(sig), levels=[3.0],
                         colors="k", linewidths=1.8, linestyles="--")
        ax.clabel(cs, fmt={3.0: "3\u03c3 boundary"}, fontsize=8)

        ax.set_xscale("log")
        ax.set_xlabel("Epidermal thickness [mm]")
        ax.set_ylabel(r"Absorption contrast  $\mu_a^{epi} / \mu_a^{derm}$")

    axes[0].set_title("Bias in diffuse reflectance")
    axes[1].set_title("Bias in absorbed fraction")
    fig.suptitle(
        "Where does bulk-averaging break down? Layered vs. homogeneous-equivalent skin model\n"
        f"(N = {N_PHOTONS} photons/run, dermis fixed at "
        rf"$\mu_a$={DERMIS_MU_A}, $\mu_s$={DERMIS_MU_S}, $g$={DERMIS_G}, "
        f"L={DERMIS_THICKNESS} mm)",
        fontsize=10.5,
    )
    fig.tight_layout()
    out = FIG_DIR / "bulk_averaging_bias_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out.relative_to(out.parents[1])}")
    print(f"Saved grid data to {(FIG_DIR / 'bulk_averaging_bias_grid.npz').relative_to(FIG_DIR.parents[0])}")


if __name__ == "__main__":
    main()
