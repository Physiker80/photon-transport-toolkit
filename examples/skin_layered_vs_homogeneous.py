"""
Homogeneous vs. layered turbid media: a realistic comparison.

This experiment asks a concrete physical question: if a two-layer
tissue (epidermis over dermis) is replaced by a single homogeneous
slab with the same *thickness-weighted average* optical properties —
the simplification implicitly made whenever a single-layer model is
fit to data from a genuinely layered medium — how much does the
predicted diffuse reflectance, transmittance, and absorbed fraction
actually change?

Optical properties are representative, literature-informed values for
skin at a visible wavelength (~630 nm), not a specific-patient
measurement. Reported ranges in the tissue-optics literature are wide
(absorption and reduced scattering can vary several-fold between
individuals, body sites, and studies), so the values below should be
read as *illustrative of typical order of magnitude and layer
contrast*, not as a validated skin model:

  Epidermis (~0.1 mm): melanin-dominated absorption, high scattering
    from keratin. mu_a ~ 0.4 /mm, reduced scattering mu_s' ~ 4.5 /mm,
    g ~ 0.8 -> mu_s = mu_s'/(1-g) = 22.5 /mm.
  Dermis (~1.5 mm): blood-dominated absorption (much lower than
    melanin at this wavelength), collagen-dominated forward scattering.
    mu_a ~ 0.05 /mm, mu_s' ~ 2.0 /mm, g ~ 0.9 -> mu_s = 20 /mm.

Representative sources for these orders of magnitude: Jacques, "Optical
properties of biological tissues: a review", Phys. Med. Biol. 58,
R37-R61 (2013); the melanin/dermis absorption and reduced-scattering
ranges tabulated by the Oregon Medical Laser Center (omlc.org); and
in-vivo layered skin measurements such as those of Karsten & Smit
et al. (spatially resolved diffuse reflectance spectroscopy cohort
studies) reporting dermal mu_s' in the 1.2-3.2 /mm range at visible
wavelengths.

To isolate the effect of *layering the absorption and scattering
alone* from the separate effect of a refractive-index mismatch
between layers, both layers here share the same refractive index
(n = 1.4, the commonly used bulk value for skin). A follow-up
experiment could reintroduce an index step to study that effect
separately.

Usage: python examples/skin_layered_vs_homogeneous.py
"""

import sys
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

N_PHOTONS = 60_000
SEED = 0
N_INDEX = 1.4  # shared refractive index, both layers and the homogeneous equivalent

EPIDERMIS = Layer(mu_a=0.40, mu_s=22.5, g=0.80, thickness=0.10, n=N_INDEX)
DERMIS = Layer(mu_a=0.05, mu_s=20.0, g=0.90, thickness=1.50, n=N_INDEX)


def thickness_weighted_average() -> SlabOpticalProperties:
    """Build the single-layer 'homogenised' equivalent of EPIDERMIS+DERMIS.

    mu_a and the reduced scattering mu_s' = mu_s(1-g) are averaged by
    thickness — the natural quantities to average, since they add
    linearly along the optical path in the diffusion/transport sense.
    g itself is also thickness-averaged for simplicity; mu_s is then
    recovered from the averaged mu_s' and g.
    """
    layers = [EPIDERMIS, DERMIS]
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
    homogeneous = thickness_weighted_average()
    layered = LayeredMedium(layers=[EPIDERMIS, DERMIS], n_outside_top=1.0, n_outside_bottom=1.0)

    print("Homogeneous equivalent (thickness-weighted average):")
    print(f"  mu_a = {homogeneous.mu_a:.4f} /mm, mu_s = {homogeneous.mu_s:.3f} /mm, "
          f"g = {homogeneous.g:.3f}, mu_s' = {homogeneous.reduced_scattering:.3f} /mm")
    print(f"  total thickness = {homogeneous.thickness:.2f} mm\n")

    res_homog = simulate_slab(homogeneous, n_photons=N_PHOTONS, seed=SEED)
    res_layered = simulate_layered_medium(layered, n_photons=N_PHOTONS, seed=SEED)

    def fmt(res, label):
        print(f"{label:22s}  R_diff = {res.diffuse_reflectance:.4f} +/- "
              f"{res.diffuse_reflectance_stderr:.4f}   "
              f"T = {res.transmittance:.4f} +/- {res.transmittance_stderr:.4f}   "
              f"A = {res.absorbed:.4f} +/- {res.absorbed_stderr:.4f}")

    print("Results (identical incident conditions, N =", N_PHOTONS, "photons):")
    fmt(res_homog, "Homogeneous slab")
    fmt(res_layered, "Layered (epi+dermis)")

    dR = res_layered.diffuse_reflectance - res_homog.diffuse_reflectance
    dT = res_layered.transmittance - res_homog.transmittance
    dA = res_layered.absorbed - res_homog.absorbed
    sigma_R = np.hypot(res_homog.diffuse_reflectance_stderr, res_layered.diffuse_reflectance_stderr)
    sigma_T = np.hypot(res_homog.transmittance_stderr, res_layered.transmittance_stderr)
    sigma_A = np.hypot(res_homog.absorbed_stderr, res_layered.absorbed_stderr)

    print(f"\nDifference (layered - homogeneous):")
    print(f"  delta R_diff = {dR:+.4f}  ({dR/sigma_R:+.1f} sigma)")
    print(f"  delta T      = {dT:+.4f}  ({dT/sigma_T:+.1f} sigma)")
    print(f"  delta A      = {dA:+.4f}  ({dA/sigma_A:+.1f} sigma)")

    # --- figure -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    labels = ["Diffuse\nreflectance", "Transmittance", "Absorbed\nfraction"]
    homog_vals = [res_homog.diffuse_reflectance, res_homog.transmittance, res_homog.absorbed]
    homog_err = [res_homog.diffuse_reflectance_stderr, res_homog.transmittance_stderr, res_homog.absorbed_stderr]
    layer_vals = [res_layered.diffuse_reflectance, res_layered.transmittance, res_layered.absorbed]
    layer_err = [res_layered.diffuse_reflectance_stderr, res_layered.transmittance_stderr, res_layered.absorbed_stderr]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, homog_vals, width, yerr=homog_err, capsize=4,
           label="Homogeneous (bulk-averaged)", color="#8FAADC")
    ax.bar(x + width / 2, layer_vals, width, yerr=layer_err, capsize=4,
           label="Layered (epidermis + dermis)", color="#C00000")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of incident power")
    ax.set_title("Homogeneous vs. layered skin-like medium\n(same bulk-averaged optical properties)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = FIG_DIR / "homogeneous_vs_layered_skin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out.relative_to(out.parents[1])}")


if __name__ == "__main__":
    main()
