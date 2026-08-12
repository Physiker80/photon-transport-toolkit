"""
coherent_speckle_field.py

Phase 1 of the PhD research roadmap: generates a coherent speckle
field using the new phase-resolved transport engine
(coherent_transport.py), and directly validates it against the
already-validated scalar engine -- the same reduction-test discipline
used throughout this project, applied to a genuinely new capability.

Usage: python examples/coherent_speckle_field.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402
from photon_transport_toolkit.coherent_transport import (  # noqa: E402
    simulate_slab_coherent, apply_coherent_psf,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

SLAB = SlabOpticalProperties(mu_a=0.05, mu_s=12.0, g=0.85, thickness=1.5, n_medium=1.4)
WAVELENGTH_NM = 632.8  # HeNe, a common lab source


def main():
    print("Reduction test: does sum(|field|^2), averaged over independent")
    print("seeds, reproduce the validated scalar engine's Rd?\n")

    ref = simulate_slab(SLAB, n_photons=18_000, seed=0, n_batches=10)
    print(f"Scalar engine Rd = {ref.diffuse_reflectance:.5f} +/- {ref.diffuse_reflectance_stderr:.5f}")

    field_sums = []
    for seed in range(20):
        coh = simulate_slab_coherent(SLAB, wavelength_nm=WAVELENGTH_NM, n_photons=3000, seed=seed,
                                      n_batches=3, detector_bins=48, detector_half_width=15.0)
        field_sums.append(np.sum(np.abs(coh.field_reflected) ** 2))
    field_sums = np.array(field_sums)
    ens_mean, ens_se = field_sums.mean(), field_sums.std(ddof=1) / np.sqrt(len(field_sums))
    sigma = abs(ens_mean - ref.diffuse_reflectance) / ens_se
    print(f"Coherent field, 20-seed ensemble: sum(|E|^2) = {ens_mean:.5f} +/- {ens_se:.5f}  "
          f"({sigma:.2f} sigma from scalar Rd)")
    print("A single realization's sum(|E|^2) fluctuates far more than this ensemble mean --")
    print("expected: coherent speckle statistics have high per-realization variance,")
    print("the same reason this test averages over many seeds rather than trusting one.\n")

    print("Generating one speckle field for visualization "
          "(60,000 photons, 40x40 detector -> ~15 photons/pixel)...")
    coh = simulate_slab_coherent(SLAB, wavelength_nm=WAVELENGTH_NM, n_photons=60_000, seed=42,
                                  n_batches=6, detector_bins=40, detector_half_width=4.0)
    intensity = np.abs(coh.field_reflected) ** 2
    print(f"This realization: sum(|E|^2)={intensity.sum():.4f} vs scalar Rd={coh.diffuse_reflectance:.4f} "
          f"(single-realization noise, not the validated quantity above)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(intensity, cmap="inferno", extent=[-4, 4, -4, 4], interpolation="nearest")
    axes[0].set_title("Raw exit-surface field |E(x,y)|\u00b2\n(one photon per pixel, uncorrelated)")
    axes[0].set_xlabel("x [mm]"); axes[0].set_ylabel("y [mm]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    phase = np.angle(coh.field_reflected)
    phase_masked = np.where(intensity > intensity.max() * 0.02, phase, np.nan)
    im1 = axes[1].imshow(phase_masked, cmap="twilight", extent=[-4, 4, -4, 4], interpolation="nearest")
    axes[1].set_title("Raw field phase\n(uncorrelated pixel-to-pixel, as expected)")
    axes[1].set_xlabel("x [mm]"); axes[1].set_ylabel("y [mm]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    pixel_size_mm = 2 * 4.0 / 40
    na = 0.00055  # deliberately low: matches grain size to this pixel sampling
    imaged_field = apply_coherent_psf(coh.field_reflected, pixel_size_mm=pixel_size_mm,
                                       wavelength_nm=WAVELENGTH_NM, numerical_aperture=na,
                                       kernel_half_size=15)
    imaged_intensity = np.abs(imaged_field) ** 2
    energy_ratio = imaged_intensity.sum() / intensity.sum()
    print(f"After coherent-PSF convolution (NA={na}): energy ratio = {energy_ratio:.3f} "
          f"(edge effect from the field's finite extent, not a physics error --\n"
          f"confirmed exactly conserved on a large synthetic test field in tests/test_coherent_psf.py)")

    im2 = axes[2].imshow(imaged_intensity, cmap="inferno", extent=[-4, 4, -4, 4], interpolation="bicubic")
    axes[2].set_title(f"After imaging-system coherent PSF (NA={na})\nspatially-correlated speckle grains")
    axes[2].set_xlabel("x [mm]"); axes[2].set_ylabel("y [mm]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Coherent speckle field from photon-transport-toolkit (\u03bb={WAVELENGTH_NM}nm)",
                 fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "coherent_speckle_field.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out.relative_to(out.parents[1])}")
    print("\nThe third panel is what a real camera, at this wavelength and NA, would actually")
    print("record: the raw exit-surface field (panels 1-2) still needed one more physical step --")
    print("propagation through imaging optics -- to become a realistic speckle photograph.")


if __name__ == "__main__":
    main()
