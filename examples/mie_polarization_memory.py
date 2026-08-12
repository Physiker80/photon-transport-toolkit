"""Polarization memory: the effect a Rayleigh-only engine cannot produce.

Under Rayleigh scattering, *linear* polarization survives multiple
scattering considerably longer than *circular* does. For scatterers
comparable to or larger than the wavelength the ordering reverses:
circular polarization is retained over more scattering events than
linear. That reversal -- polarization memory -- is the physical basis
for circular-polarization gating, and it cannot appear in an engine
without a particle size to vary.

This script sweeps particle radius at fixed optical depth and measures,
for each size, how many scattering events each polarization channel
survives. The crossover is the result; that it happens at all is a
prediction the combined phase+polarization engine of Section 13 could
have failed to reproduce.

Why the measurement frame is not incidental
-------------------------------------------
Three natural-looking ways to measure "how much polarization is left"
fail, each for a different reason, and finding that out empirically was
most of the work here:

  1. Per-photon degree of polarization: identically 1 forever. A Jones
     vector scattered by amplitude matrices stays fully polarized;
     depolarization is an ensemble property, not a per-packet one.
  2. Averaging in the photon's own post-scattering frame: reports
     depolarization that never happened, because that frame is rotated
     by the random azimuth psi at every event -- even for a photon
     scattered exactly forward, whose polarization did not change at
     all.
  3. Averaging what an x/y analyzer pair would see in the laboratory:
     physically meaningful, and reported below for exactly that reason,
     but it multiplies the circular component by a ray-obliquity factor
     the linear components do not carry. That systematically penalizes
     the circular channel and *hides the crossover entirely* -- the
     ordering never reverses under this measure.

The default measure fixes the transverse frame by the photon's own
direction and the incident polarization axis (``e2 || u x x_hat``). It
is common to all photons travelling the same way, reduces to the
incident frame for undeflected photons, and carries no projection
factor. Its correctness has an exact analytic anchor: after a single
Rayleigh event the retained linear polarization must be exactly 1.000,
with no statistical scatter, because a dipole driven along x radiates
the component of x perpendicular to the observation direction. That is
asserted in ``tests/test_mie.py``.

Author: Noureddin Sedki
License: MIT
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab_vector  # noqa: E402
from photon_transport_toolkit.mie import MieScatterer  # noqa: E402
from photon_transport_toolkit.vector_transport import depolarization_ladder  # noqa: E402

WAVELENGTH_NM = 633.0
N_MEDIUM = 1.33
N_PARTICLE = 1.59          # polystyrene in water: the standard calibration phantom
N_EVENTS = 40
N_PHOTONS = 700
RADII_UM = [0.05, 0.10, 0.20, 0.40, 0.80]


def survival_order(order, curve, level=1.0 / np.e):
    """Scattering order at which the retained polarization falls to 1/e."""
    y = np.abs(curve) / abs(curve[0])
    below = np.nonzero(y < level)[0]
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return 0.0
    return (i - 1) + (y[i - 1] - level) / (y[i - 1] - y[i])


def measure(scatterer, frame="canonical", seed_offset=0):
    _, lin, _ = depolarization_ladder(N_EVENTS, N_PHOTONS, scatterer, "x",
                                      seed=11 + seed_offset, frame=frame)
    order, _, circ = depolarization_ladder(N_EVENTS, N_PHOTONS, scatterer, "circular",
                                           seed=12 + seed_offset, frame=frame)
    return order, lin, circ


def main():
    print("=" * 88)
    print("POLARIZATION MEMORY: LINEAR VS CIRCULAR SURVIVAL VS PARTICLE SIZE")
    print(f"lambda = {WAVELENGTH_NM:.0f} nm, particle n = {N_PARTICLE}, "
          f"medium n = {N_MEDIUM}")
    print("=" * 88)
    print(f"{'scatterer':<22}{'x':>7}{'g':>8}{'n(lin)':>9}{'n(circ)':>10}"
          f"{'circ/lin':>10}   ordering")

    rows = []
    order, lin, circ = measure(None)
    n_lin, n_circ = survival_order(order, lin), survival_order(order, circ)
    print(f"{'Rayleigh (x -> 0)':<22}{0.0:>7.2f}{0.0:>8.3f}{n_lin:>9.2f}{n_circ:>10.2f}"
          f"{n_circ / n_lin:>10.2f}   linear survives")
    rows.append(("Rayleigh", 0.0, 0.0, order, lin, circ, n_lin, n_circ))

    for radius in RADII_UM:
        s = MieScatterer(radius, WAVELENGTH_NM, N_PARTICLE, N_MEDIUM)
        order, lin, circ = measure(s)
        n_lin, n_circ = survival_order(order, lin), survival_order(order, circ)
        ratio = n_circ / n_lin if np.isfinite(n_circ) else np.inf
        verdict = "linear survives" if ratio < 1.0 else "CIRCULAR survives"
        circ_txt = f"{n_circ:>10.2f}" if np.isfinite(n_circ) else f"{'>' + str(N_EVENTS):>10}"
        ratio_txt = f"{ratio:>10.2f}" if np.isfinite(ratio) else f"{'>' + f'{N_EVENTS / n_lin:.1f}':>10}"
        print(f"{'Mie a = ' + f'{radius:.2f} um':<22}{s.x:>7.2f}{s.g:>8.3f}"
              f"{n_lin:>9.2f}{circ_txt}{ratio_txt}   {verdict}")
        rows.append((f"a = {radius:.2f} um", s.x, s.g, order, lin, circ, n_lin, n_circ))

    # ---- the same sweep under the laboratory-analyzer measure --------------
    print("\n" + "-" * 88)
    print("Same sweep, measured as an x/y analyzer and camera would see it")
    print("(ray-obliquity factor included -- note the crossover disappears)")
    print("-" * 88)
    print(f"{'scatterer':<22}{'n(lin)':>9}{'n(circ)':>10}{'circ/lin':>10}")
    for label, scatterer in [("Rayleigh", None)] + [
        (f"a = {r:.2f} um", MieScatterer(r, WAVELENGTH_NM, N_PARTICLE, N_MEDIUM))
        for r in RADII_UM
    ]:
        o, lin, circ = measure(scatterer, frame="lab")
        n_lin, n_circ = survival_order(o, lin), survival_order(o, circ)
        circ_txt = f"{n_circ:>10.2f}" if np.isfinite(n_circ) else f"{'>' + str(N_EVENTS):>10}"
        ratio_txt = (f"{n_circ / n_lin:>10.2f}" if np.isfinite(n_circ) else f"{'--':>10}")
        print(f"{label:<22}{n_lin:>9.2f}{circ_txt}{ratio_txt}")

    # ---- transport-level consequence --------------------------------------
    print("\n" + "-" * 88)
    print("Consequence in an actual slab (mu_a=0.1, mu_s=3.0 /mm, L=0.6 mm, "
          "identical optical depth)")
    print("-" * 88)
    slab = SlabOpticalProperties(mu_a=0.1, mu_s=3.0, g=0.0, thickness=0.6)
    print(f"{'scatterer':<22}{'g':>8}{'Rd':>12}{'DoLP(refl)':>13}{'DoCP(refl)':>13}")
    for label, scatterer in [("Rayleigh", None)] + [
        (f"a = {r:.2f} um", MieScatterer(r, WAVELENGTH_NM, N_PARTICLE, N_MEDIUM))
        for r in (0.10, 0.40)
    ]:
        lin_run = simulate_slab_vector(slab, WAVELENGTH_NM, n_photons=9000, n_batches=6,
                                       scatterer=scatterer, polarization="x",
                                       detector_bins=16, seed=21)
        circ_run = simulate_slab_vector(slab, WAVELENGTH_NM, n_photons=9000, n_batches=6,
                                        scatterer=scatterer, polarization="circular",
                                        detector_bins=16, seed=22)
        g = 0.0 if scatterer is None else scatterer.g
        dolp = lin_run.stokes_total[1] / lin_run.stokes_total[0]
        docp = circ_run.stokes_total[3] / circ_run.stokes_total[0]
        print(f"{label:<22}{g:>8.3f}{lin_run.diffuse_reflectance:>12.4f}"
              f"{dolp:>13.4f}{abs(docp):>13.4f}")

    _plot(rows)


def _plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("\n(matplotlib not available -- figure skipped)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ray = rows[0]
    big = rows[-2]
    for ax, row, title in ((axes[0], ray, "Rayleigh (x → 0)"),
                           (axes[1], big, f"Mie, x = {big[1]:.1f}, g = {big[2]:.2f}")):
        order, lin, circ = row[3], row[4], row[5]
        ax.semilogy(order, np.maximum(lin, 1e-3), "o-", color="#1f4e79",
                    ms=3, label="linear  |⟨Q⟩|/⟨I⟩")
        ax.semilogy(order, np.maximum(circ, 1e-3), "s-", color="#b03060",
                    ms=3, label="circular  |⟨V⟩|/⟨I⟩")
        ax.axhline(1.0 / np.e, color="0.5", ls="--", lw=1)
        ax.set_xlabel("scattering events")
        ax.set_title(title)
        ax.set_ylim(1e-2, 1.4)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("polarization retained")

    ax = axes[2]
    xs = [r[1] for r in rows]
    n_lin = [r[6] for r in rows]
    n_circ = [r[7] if np.isfinite(r[7]) else np.nan for r in rows]
    ax.plot(xs, n_lin, "o-", color="#1f4e79", label="linear")
    ax.plot(xs, n_circ, "s-", color="#b03060", label="circular")
    capped = [r[1] for r in rows if not np.isfinite(r[7])]
    if capped:
        ax.plot(capped, [N_EVENTS] * len(capped), "^", color="#b03060", ms=9,
                label=f"circular > {N_EVENTS} (off scale)")
    ax.set_xlabel("size parameter x")
    ax.set_ylabel("scattering events to 1/e")
    ax.set_title("Polarization memory:\nthe ordering reverses with particle size")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "mie_polarization_memory.png"
    plt.savefig(out, dpi=130)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
