"""
What each level of wave description can and cannot predict.

Four engines are run on the same media:

  1. scalar          -- monte_carlo.simulate_slab: weight only
  2. phase-only      -- coherent_transport.simulate_slab_coherent: weight + path length
  3. polarization-only -- vector_transport, "stokes" formulation: Stokes vector, no phase
  4. phase + polarization -- vector_transport, "jones" formulation: complex amplitudes + path length

The point of the comparison is a quantity that *only* the fourth can
predict. Speckle contrast is the natural candidate: it is a phase
effect, so engines 1 and 3 give nothing at all; but its value for
light detected without an analyzer depends on the degree of
polarization P, which engine 2 has no representation of. A phase-only
engine therefore predicts contrast C = 1 for every medium, whereas
Goodman's result for partially polarized speckle is

    C = sqrt((1 + P^2) / 2)

which falls to 1/sqrt(2) ~ 0.707 for fully depolarized light. This
script measures both P and C from the same photon histories and tests
that relation -- a prediction the combined engine can fail, made by
neither of its two halves.

Practical relevance: laser speckle contrast imaging (LSCI) reads flow
out of contrast. An LSCI instrument detecting without a polarizer has
a static contrast floor set by depolarization alone, with no flow
involved -- the size of that floor is exactly what the fourth engine
computes and the second one gets wrong by up to 41%.

Finite-photon-budget correction
-------------------------------
A pixel fed by a finite number of photon paths is not fully developed
speckle. For n_eff independent phasors the normalized intensity
rho = |sum E|^2 / sum |E|^2 has variance exactly 1 - 1/n_eff, and
n_eff = (sum w)^2 / (sum w^2) is measurable from the same run. Every
contrast below is therefore reported against its *exact* finite-n
expectation as well as the asymptotic value -- the same
"statistical vs practical significance" discipline used throughout
this project, applied to a speckle statistic.

Author: Noureddin Sedki
License: MIT
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402
from photon_transport_toolkit.coherent_transport import simulate_slab_coherent  # noqa: E402
from photon_transport_toolkit.vector_transport import simulate_slab_vector  # noqa: E402

WAVELENGTH_NM = 633.0
DETECTOR_BINS = 28
DETECTOR_HALF_WIDTH = 0.9
ANNULUS = (0.10, 0.80)
MIN_NEFF = 4.0

MEDIA = [
    ("thin,   weakly scattering", SlabOpticalProperties(mu_a=0.1, mu_s=1.5, g=0.0, thickness=0.15), 90_000),
    ("medium                   ", SlabOpticalProperties(mu_a=0.1, mu_s=3.0, g=0.0, thickness=0.60), 50_000),
    ("thick,  strongly scattering", SlabOpticalProperties(mu_a=0.1, mu_s=8.0, g=0.0, thickness=2.00), 25_000),
]


def analyse_speckle(res):
    """Measure P and speckle contrast in an annulus of the detector plane.

    The annulus avoids the central pixel (where the specular/ballistic
    contribution dominates) and the sparse outer pixels; pixels with
    fewer than MIN_NEFF effective contributions are dropped because
    their normalized intensity is degenerate, not because they are
    inconvenient.
    """
    n = res.intensity_co.shape[0]
    half = res.detector_half_width
    axis = np.linspace(-half + half / n, half - half / n, n)
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx, yy)

    s1x, s1y = res.intensity_co, res.intensity_cross
    s2x, s2y = res.intensity_sq_co, res.intensity_sq_cross
    with np.errstate(divide="ignore", invalid="ignore"):
        n_x = np.where(s2x > 0, s1x**2 / s2x, 0.0)
        n_y = np.where(s2y > 0, s1y**2 / s2y, 0.0)

    mask = (radius >= ANNULUS[0]) & (radius < ANNULUS[1]) & (n_x >= MIN_NEFF) & (n_y >= MIN_NEFF)

    i_x = np.abs(res.field_co) ** 2
    i_y = np.abs(res.field_cross) ** 2
    rho_co = i_x[mask] / s1x[mask]
    rho_un = (i_x[mask] + i_y[mask]) / (s1x[mask] + s1y[mask])

    pred_co = 1.0 - 1.0 / n_x[mask]
    pred_un = (s1x[mask] ** 2 * (1.0 - 1.0 / n_x[mask])
               + s1y[mask] ** 2 * (1.0 - 1.0 / n_y[mask])) / (s1x[mask] + s1y[mask]) ** 2

    total_x, total_y = s1x[mask].sum(), s1y[mask].sum()
    p_linear = (total_x - total_y) / (total_x + total_y)
    stokes = res.stokes_images[:, mask].sum(axis=1)
    p_full = np.sqrt(stokes[1] ** 2 + stokes[2] ** 2 + stokes[3] ** 2) / stokes[0]

    n_pix = mask.sum()
    # Variance-of-a-variance standard error, for a near-exponential
    # distribution: relative SE ~ sqrt(8/n_pix). Reported so the
    # comparison below is a sigma statement, not an eyeball one.
    rel_se = np.sqrt(8.0 / n_pix)

    return dict(
        n_pixels=int(n_pix),
        n_eff=float(n_x[mask].mean()),
        p_linear=float(p_linear),
        p_full=float(p_full),
        mean_rho=float(rho_co.mean()),
        var_co=float(rho_co.var(ddof=1)),
        var_co_pred=float(pred_co.mean()),
        var_un=float(rho_un.var(ddof=1)),
        var_un_pred=float(pred_un.mean()),
        goodman=float((1.0 + p_linear**2) / 2.0),
        rel_se=float(rel_se),
    )


def main():
    print("=" * 92)
    print("PHASE AND POLARIZATION IN MONTE CARLO: WHAT EACH ENGINE CAN PREDICT")
    print("=" * 92)

    rows = []
    for label, slab, n_photons in MEDIA:
        scalar = simulate_slab(slab, n_photons=n_photons // 4, seed=1, n_batches=10)
        coherent = simulate_slab_coherent(slab, WAVELENGTH_NM, n_photons=n_photons // 4,
                                          seed=1, n_batches=10, detector_bins=DETECTOR_BINS,
                                          detector_half_width=DETECTOR_HALF_WIDTH)
        stokes = simulate_slab_vector(slab, WAVELENGTH_NM, n_photons=n_photons // 6, seed=2,
                                      n_batches=6, detector_bins=DETECTOR_BINS,
                                      detector_half_width=DETECTOR_HALF_WIDTH,
                                      formulation="stokes")
        jones = simulate_slab_vector(slab, WAVELENGTH_NM, n_photons=n_photons, seed=3,
                                     n_batches=10, detector_bins=DETECTOR_BINS,
                                     detector_half_width=DETECTOR_HALF_WIDTH,
                                     formulation="jones")
        speckle = analyse_speckle(jones)

        print(f"\n--- {label.strip()}  (mu_s={slab.mu_s} /mm, L={slab.thickness} mm, "
              f"N={n_photons}) ---")
        print(f"  scalar engine (HG, g=0)      Rd = {scalar.diffuse_reflectance:.4f} "
              f"+- {scalar.diffuse_reflectance_stderr:.4f}   [no phase, no polarization]")
        print(f"  phase-only engine            Rd = {coherent.diffuse_reflectance:.4f} "
              f"+- {coherent.diffuse_reflectance_stderr:.4f}   speckle contrast: 1 by construction")
        print(f"  polarization-only (Mueller)  Rd = {stokes.diffuse_reflectance:.4f} "
              f"+- {stokes.diffuse_reflectance_stderr:.4f}   DoLP = "
              f"{stokes.stokes_total[1] / stokes.stokes_total[0]:.4f}   [no speckle at all]")
        print(f"  phase + polarization (Jones) Rd = {jones.diffuse_reflectance:.4f} "
              f"+- {jones.diffuse_reflectance_stderr:.4f}   energy check: "
              f"{jones.diffuse_reflectance + jones.transmittance + jones.absorbed:.12f}")

        d = speckle
        c_co = np.sqrt(max(d["var_co"], 0.0))
        c_un = np.sqrt(max(d["var_un"], 0.0))
        print(f"    detector annulus: {d['n_pixels']} pixels, n_eff = {d['n_eff']:.1f} "
              f"contributions/pixel, <rho> = {d['mean_rho']:.3f} (must be 1)")
        print(f"    P (linear, x/y) = {d['p_linear']:.3f}   P (full DoP) = {d['p_full']:.3f}")
        print(f"    co-polarized    : var(rho) = {d['var_co']:.3f}  vs finite-n prediction "
              f"{d['var_co_pred']:.3f}   -> C = {c_co:.3f}")
        print(f"    no analyzer     : var(rho) = {d['var_un']:.3f}  vs finite-n prediction "
              f"{d['var_un_pred']:.3f}   -> C = {c_un:.3f}")
        print(f"    Goodman (n->inf): (1+P^2)/2 = {d['goodman']:.3f}   "
              f"phase-only engine would say 1.000")
        dev = abs(d["var_un"] - d["var_un_pred"]) / (d["var_un_pred"] * d["rel_se"])
        print(f"    agreement with the combined-engine prediction: {dev:.1f} sigma")
        rows.append((label, slab, jones, d))

    print("\n" + "=" * 92)
    print("SUMMARY: contrast a phase-only engine predicts vs contrast the combined engine predicts")
    print("=" * 92)
    print(f"{'medium':<28}{'P':>8}{'C (phase-only)':>17}{'C (phase+pol)':>16}{'overestimate':>15}")
    for label, slab, jones, d in rows:
        c_true = np.sqrt(d["goodman"])
        print(f"{label.strip():<28}{d['p_linear']:>8.3f}{1.0:>17.3f}{c_true:>16.3f}"
              f"{100 * (1.0 / c_true - 1.0):>14.1f}%")

    _plot(rows)


def _plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("\n(matplotlib not available -- figure skipped)")
        return

    label, slab, jones, d = rows[-1]
    half = jones.detector_half_width
    extent = [-half, half, -half, half]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))

    speckle_co = np.abs(jones.field_co) ** 2
    axes[0].imshow(speckle_co, cmap="inferno", extent=extent, origin="lower",
                   vmax=np.percentile(speckle_co, 99))
    axes[0].set_title("Co-polarized speckle  $|E_x|^2$\n(needs phase AND polarization)")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")

    speckle_cross = np.abs(jones.field_cross) ** 2
    axes[1].imshow(speckle_cross, cmap="inferno", extent=extent, origin="lower",
                   vmax=np.percentile(speckle_cross, 99))
    axes[1].set_title("Cross-polarized speckle  $|E_y|^2$")
    axes[1].set_xlabel("x [mm]")

    with np.errstate(divide="ignore", invalid="ignore"):
        dolp = np.where(jones.stokes_images[0] > 0,
                        jones.stokes_images[1] / np.maximum(jones.stokes_images[0], 1e-30), 0.0)
    im = axes[2].imshow(dolp, cmap="coolwarm", extent=extent, origin="lower", vmin=-0.6, vmax=0.6)
    axes[2].set_title("Degree of linear polarization $Q/I$\n(polarization only -- no phase needed)")
    axes[2].set_xlabel("x [mm]")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    p_axis = np.linspace(0.0, 1.0, 200)
    axes[3].plot(p_axis, np.sqrt((1.0 + p_axis**2) / 2.0), "k-", lw=2,
                 label=r"Goodman  $C=\sqrt{(1+P^2)/2}$")
    axes[3].axhline(1.0, color="crimson", ls="--", lw=2,
                    label="phase-only engine (always 1)")
    for lbl, _s, _j, dd in rows:
        axes[3].plot(dd["p_linear"], np.sqrt(max(dd["var_un"], 0.0)) , "o", ms=9,
                     label=f"measured: {lbl.strip()}")
    axes[3].set_xlabel("degree of polarization $P$")
    axes[3].set_ylabel("speckle contrast $C$ (no analyzer)")
    axes[3].set_ylim(0.5, 1.15)
    axes[3].set_title("Contrast a phase-only engine\ncannot get right")
    axes[3].legend(fontsize=7, loc="lower right")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "polarized_speckle_comparison.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
