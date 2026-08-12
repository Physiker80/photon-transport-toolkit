"""
photon_count_comparison.py

Directly answers a practical question the interactive 3D demo (docs/index.html)
raises by offering both a "run 150" and a "run 150,000" button: does the
project's central finding (Section 8 / PROJECT_REPORT.md) actually depend on
which one you press?

Runs the exact homogeneous-vs-layered comparison from
examples/skin_layered_vs_homogeneous.py at two photon budgets -- N=150 and
N=150,000 -- and reports whether the bias (DeltaR) is statistically
resolvable at each.

N=150,000 in the layered engine takes several minutes in a single call
(~3ms/photon here vs ~0.4ms/photon for the homogeneous engine), so this
script chunks the large run across multiple independent batches -- the
same n_batches mechanism simulate_layered_medium() already uses
internally, just invoked repeatedly and combined by hand. This is
statistically identical to one big run: each chunk's mean is an
independent estimate of the same quantity, so the grand mean is their
average and the grand standard error is std(chunk means)/sqrt(n_chunks).

By default this script runs a SMALL version (suitable for CI / a quick
check) via --quick. The exact numbers reported in PROJECT_REPORT.md
Section 9 were produced with --full, which reproduces the real N=150 vs
N=150,000 comparison and takes several minutes.

Usage:
    python examples/photon_count_comparison.py --quick   (~10s, default)
    python examples/photon_count_comparison.py --full    (~8 min)
"""

import argparse
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

# Same two-layer skin-like medium as examples/skin_layered_vs_homogeneous.py
EPIDERMIS = Layer(mu_a=0.50, mu_s=22.5, g=0.80, thickness=0.20, n=1.4)
DERMIS = Layer(mu_a=0.05, mu_s=20.0, g=0.90, thickness=1.5, n=1.4)


def homogeneous_equivalent() -> SlabOpticalProperties:
    total_th = EPIDERMIS.thickness + DERMIS.thickness
    mua_avg = (EPIDERMIS.mu_a * EPIDERMIS.thickness + DERMIS.mu_a * DERMIS.thickness) / total_th
    musp_avg = (EPIDERMIS.mu_s * (1 - EPIDERMIS.g) * EPIDERMIS.thickness +
                DERMIS.mu_s * (1 - DERMIS.g) * DERMIS.thickness) / total_th
    g_avg = (EPIDERMIS.g * EPIDERMIS.thickness + DERMIS.g * DERMIS.thickness) / total_th
    mus_avg = musp_avg / (1 - g_avg)
    return SlabOpticalProperties(mu_a=mua_avg, mu_s=mus_avg, g=g_avg, thickness=total_th, n_medium=1.4)


def combine(means, ses):
    """Combine independent (mean, se) estimates of the same quantity into
    one grand mean and standard error, treating each as one more batch."""
    means = np.asarray(means)
    if len(means) == 1:
        return means[0], ses[0]
    return means.mean(), means.std(ddof=1) / np.sqrt(len(means))


def run_layered(n_photons_per_chunk, n_batches_per_chunk, n_chunks, seed0):
    means, ses = [], []
    medium = LayeredMedium(layers=[EPIDERMIS, DERMIS])
    for i in range(n_chunks):
        r = simulate_layered_medium(medium, n_photons=n_photons_per_chunk, seed=seed0 + i,
                                     n_batches=n_batches_per_chunk)
        means.append(r.diffuse_reflectance)
        ses.append(r.diffuse_reflectance_stderr)
        print(f"    chunk {i+1}/{n_chunks}: Rd={r.diffuse_reflectance:.5f} "
              f"+/- {r.diffuse_reflectance_stderr:.5f}")
    return combine(means, ses)


def run_homogeneous(n_photons, n_batches, seed):
    slab = homogeneous_equivalent()
    r = simulate_slab(slab, n_photons=n_photons, seed=seed, n_batches=n_batches)
    return r.diffuse_reflectance, r.diffuse_reflectance_stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="Reproduce the exact N=150 vs N=150,000 comparison (~8 min). "
                          "Default is a fast --quick version (~10s) with the same structure "
                          "but smaller N, for CI / a quick check.")
    args = ap.parse_args()

    if args.full:
        N_SMALL, SMALL_BATCHES = 50, 3          # 150 total
        LARGE_CHUNK_PHOTONS, LARGE_CHUNK_BATCHES, LARGE_N_CHUNKS = 6000, 3, 9  # ~162,000 total
        HOMOG_LARGE_PHOTONS, HOMOG_LARGE_BATCHES = 15000, 10  # 150,000 total, fits in one call
    else:
        N_SMALL, SMALL_BATCHES = 20, 3          # 60 total -- illustrative "small N"
        LARGE_CHUNK_PHOTONS, LARGE_CHUNK_BATCHES, LARGE_N_CHUNKS = 800, 3, 2  # ~4,800 total
        HOMOG_LARGE_PHOTONS, HOMOG_LARGE_BATCHES = 1600, 3

    print(f"Mode: {'--full (N=150 vs N=150,000, ~8 min)' if args.full else '--quick (small illustrative N, ~10s)'}\n")

    t0 = time.time()
    print("Homogeneous, small N...")
    homog_small = run_homogeneous(N_SMALL, SMALL_BATCHES, seed=1)
    print("Homogeneous, large N...")
    homog_large = run_homogeneous(HOMOG_LARGE_PHOTONS, HOMOG_LARGE_BATCHES, seed=201)

    print("\nLayered, small N...")
    r = simulate_layered_medium(LayeredMedium(layers=[EPIDERMIS, DERMIS]), n_photons=N_SMALL,
                                 seed=1, n_batches=SMALL_BATCHES)
    layered_small = (r.diffuse_reflectance, r.diffuse_reflectance_stderr)
    print(f"    Rd={layered_small[0]:.5f} +/- {layered_small[1]:.5f}")

    print("\nLayered, large N (chunked)...")
    layered_large = run_layered(LARGE_CHUNK_PHOTONS, LARGE_CHUNK_BATCHES, LARGE_N_CHUNKS, seed0=101)

    dR_small = layered_small[0] - homog_small[0]
    dR_small_se = np.hypot(layered_small[1], homog_small[1])
    dR_large = layered_large[0] - homog_large[0]
    dR_large_se = np.hypot(layered_large[1], homog_large[1])

    print(f"\n{'':20s} {'Rd (homog)':>16s} {'Rd (layered)':>16s} {'DeltaR':>10s} {'sigma':>8s}")
    print(f"{'small N':20s} {homog_small[0]:>9.5f}+/-{homog_small[1]:<6.5f} "
          f"{layered_small[0]:>9.5f}+/-{layered_small[1]:<6.5f} {dR_small:>+10.5f} "
          f"{dR_small/dR_small_se:>7.2f}")
    print(f"{'large N':20s} {homog_large[0]:>9.5f}+/-{homog_large[1]:<6.5f} "
          f"{layered_large[0]:>9.5f}+/-{layered_large[1]:<6.5f} {dR_large:>+10.5f} "
          f"{dR_large/dR_large_se:>7.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [homog_small[0], homog_large[0]], w,
           yerr=[homog_small[1], homog_large[1]], label="Homogeneous", color="#94a3c4", capsize=4)
    ax.bar(x + w/2, [layered_small[0], layered_large[0]], w,
           yerr=[layered_small[1], layered_large[1]], label="Layered", color="#C00000", capsize=4)
    n_small_label = f"N = {N_SMALL*SMALL_BATCHES}"
    n_large_label = f"N = {LARGE_CHUNK_PHOTONS*LARGE_CHUNK_BATCHES*LARGE_N_CHUNKS:,}"
    ax.set_xticks(x)
    ax.set_xticklabels([n_small_label, n_large_label])
    ax.set_ylabel("Diffuse reflectance $R_d$")
    ax.set_title("Same physics, two sample sizes")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    sig = [abs(dR_small/dR_small_se), abs(dR_large/dR_large_se)]
    bars = ax.bar([n_small_label, n_large_label], sig, color=["#94a3c4", "#C00000"])
    ax.axhline(3, color="gray", linestyle="--", linewidth=1, label="3\u03c3 threshold")
    ax.set_ylabel("|\u0394R| significance (\u03c3)")
    ax.set_title("Is the central finding even detectable?")
    ax.set_yscale("log")
    for b, s in zip(bars, sig):
        ax.text(b.get_x()+b.get_width()/2, s*1.15, f"{s:.2f}\u03c3", ha="center", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Why photon count matters: the same bias, at low vs high N", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "photon_count_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out.relative_to(out.parents[1])}  (took {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
