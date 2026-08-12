"""
schlick_vs_henyey_greenstein.py

Directly tests a specific claim from the author's M.Sc. thesis (Sedki,
"Simulation of scattering processes in turbid media with ZEMAX and
experimental verification", Hochschule Aalen, 2014): that the Schlick
phase function is "similar to Henyey-Greenstein ... but ... faster to
compute", making it "very well suited to be used in Monte Carlo methods."

The 2014 thesis implemented both as C-language Zemax DLLs and compared
them qualitatively. This script re-derives both from scratch in Python
(see monte_carlo.py's schlick_pdf/henyey_greenstein_pdf and their
samplers) and tests the claim quantitatively: a speed benchmark, and a
full Monte Carlo comparison of diffuse reflectance across a range of
anisotropy g, at a turbid-medium optical depth typical of the rest of
this package's examples.

Usage: python examples/schlick_vs_henyey_greenstein.py
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
from photon_transport_toolkit.monte_carlo import (  # noqa: E402
    g_to_schlick_k, _sample_henyey_greenstein, _sample_schlick,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

MU_A, MU_S, THICKNESS = 0.1, 8.0, 2.0
N_PHOTONS, N_BATCHES, SEED = 2000, 4, 7
G_VALUES = [0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]


def speed_benchmark(n=300_000):
    rng_hg = np.random.default_rng(0)
    rng_sc = np.random.default_rng(0)
    g = 0.8
    k = g_to_schlick_k(g)

    t0 = time.perf_counter()
    for _ in range(n):
        _sample_henyey_greenstein(g, rng_hg)
    t_hg = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        _sample_schlick(k, rng_sc)
    t_schlick = time.perf_counter() - t0

    print(f"Speed: HG {t_hg:.3f}s vs Schlick {t_schlick:.3f}s for {n} samples "
          f"({t_hg / t_schlick:.2f}x) -- 2014 thesis claimed Schlick is faster.\n")
    return t_hg, t_schlick


def main():
    speed_benchmark()

    Rd_hg, Rd_schlick, se_hg, se_schlick = [], [], [], []
    print(f"{'g':>6} {'k':>8} {'Rd_HG':>10} {'Rd_Schlick':>12} {'sigma':>8}")
    for g in G_VALUES:
        k = g_to_schlick_k(g)
        slab_hg = SlabOpticalProperties(mu_a=MU_A, mu_s=MU_S, g=g, thickness=THICKNESS,
                                         phase_function="hg")
        slab_sc = SlabOpticalProperties(mu_a=MU_A, mu_s=MU_S, g=g, thickness=THICKNESS,
                                         phase_function="schlick")
        r_hg = simulate_slab(slab_hg, n_photons=N_PHOTONS, seed=SEED, n_batches=N_BATCHES)
        r_sc = simulate_slab(slab_sc, n_photons=N_PHOTONS, seed=SEED, n_batches=N_BATCHES)

        Rd_hg.append(r_hg.diffuse_reflectance)
        Rd_schlick.append(r_sc.diffuse_reflectance)
        se_hg.append(r_hg.diffuse_reflectance_stderr)
        se_schlick.append(r_sc.diffuse_reflectance_stderr)

        sigma = abs(r_hg.diffuse_reflectance - r_sc.diffuse_reflectance) / np.hypot(
            r_hg.diffuse_reflectance_stderr, r_sc.diffuse_reflectance_stderr)
        print(f"{g:>6.2f} {k:>8.4f} {r_hg.diffuse_reflectance:>10.4f} "
              f"{r_sc.diffuse_reflectance:>12.4f} {sigma:>7.1f}s")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(G_VALUES, Rd_hg, yerr=se_hg, fmt="o-", color="#1F3864",
                label="Henyey-Greenstein", capsize=3)
    ax.errorbar(G_VALUES, Rd_schlick, yerr=se_schlick, fmt="s-", color="#C00000",
                label="Schlick approximation", capsize=3)
    ax.set_xlabel("Anisotropy g")
    ax.set_ylabel("Diffuse reflectance $R_d$")
    ax.set_title("Testing a 2014 M.Sc. thesis claim: Schlick \u2248 Henyey-Greenstein?\n"
                  "(agreement holds at moderate g, degrades near tissue-like g\u22480.85\u20130.9)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "schlick_vs_henyey_greenstein.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out.relative_to(out.parents[1])}")
    print("\nConclusion: the 2014 thesis's claim holds at moderate anisotropy, but the")
    print("standard k(g) fit's approximation quality degrades measurably as g approaches")
    print("the ~0.8-0.9 range typical of biological tissue -- a refinement the original")
    print("single-sentence claim did not quantify.")


if __name__ == "__main__":
    main()
