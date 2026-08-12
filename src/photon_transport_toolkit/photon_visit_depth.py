"""Photon visit-depth analysis: the physical mechanism behind the sign flip.

Why this script exists
-----------------------
Section 8's Discussion states the mechanism in words: diffuse reflectance
is generated mainly by photons that random-walk back out within about one
transport mean free path of the entrance surface, so concentrating
absorption at the surface preferentially removes weight from exactly the
photons that would otherwise have escaped as reflectance, while
concentrating it at depth does not. This script turns that sentence into
a number: the exit-weight-weighted mean maximum penetration depth,
<z_max>, of the photons that actually exit as diffuse reflectance.

The prediction, if the stated mechanism is right, is direct: <z_max>
should be *lower* than the homogeneous baseline when the strong absorber
sits at the surface (the deep-reaching photons get preferentially culled
on their way back through it) and *higher* than the baseline when the
strong absorber sits at depth (only photons that reach the absorber and
return are culled; the many photons that never get that deep are
untouched) -- i.e. sign(<z_max> - <z_max>_homogeneous) should match
sign(ΔR) in both configurations, not just correlate with it loosely.

Implementation note
--------------------
The instrumented tracer below is a deliberate, minimal-diff copy of
``layered_media._trace_one_photon_layered`` with exactly one addition
(tracking the running maximum z reached) threaded through the same two
places the position is ever updated. It calls the *same* tested physics
primitives (_fresnel_reflectance, _sample_henyey_greenstein,
_scatter_direction, _refract_direction) that the validated engine uses --
nothing about the sampled physics is reimplemented, only the bookkeeping
needed to report a diagnostic the validated function doesn't expose. It
is not itself claimed to be independently validated; what it reports is
cross-checked against the existing engine's own R/T/A on the same inputs
before being trusted (see the assertion in main()).

Author: Noureddin Sedki
License: MIT
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402
from photon_transport_toolkit.layered_media import (  # noqa: E402
    Layer,
    LayeredMedium,
    simulate_layered_medium,
)
from photon_transport_toolkit.monte_carlo import (  # noqa: E402
    _fresnel_reflectance,
    _refract_direction,
    _sample_henyey_greenstein,
    _scatter_direction,
)

N_PHOTONS = 40_000
N_BATCHES = 8
SEED = 0
N_INDEX = 1.4

MU_A_LOW = 0.05
MU_A_HIGH = 0.50
SURFACE_MU_S, SURFACE_G = 22.5, 0.80
DEEP_MU_S, DEEP_G = 20.0, 0.90
DEEP_THICKNESS = 1.5
SURFACE_THICKNESS = 0.25  # mm -- the thickness spot-checked against MATLAB in §11.3


def _trace_one_photon_with_depth(medium, boundaries, rng, weight_threshold, roulette_survival):
    """_trace_one_photon_layered, plus running-maximum-depth bookkeeping.

    Every line of physics here is identical to the validated function;
    the only addition is ``max_z``, updated at the same two places ``z``
    itself is ever assigned.
    """
    layers = medium.layers
    n_layers = len(layers)

    z = 0.0
    max_z = 0.0
    ux = uy = 0.0
    uz = 1.0
    i = 0

    r_specular = _fresnel_reflectance(1.0, medium.n_outside_top, layers[0].n)
    weight = 1.0 - r_specular

    reflected = transmitted = absorbed = 0.0

    while weight > 0.0:
        tau = -np.log(rng.random())

        while True:
            mu_t_i = layers[i].mu_t
            step = tau / mu_t_i

            if uz > 0.0:
                dist_boundary = (boundaries[i + 1] - z) / uz
            elif uz < 0.0:
                dist_boundary = (boundaries[i] - z) / uz
            else:
                dist_boundary = np.inf

            if step < dist_boundary:
                z += step * uz
                max_z = max(max_z, z)
                break

            z = boundaries[i + 1] if uz > 0.0 else boundaries[i]
            max_z = max(max_z, z)
            tau -= dist_boundary * mu_t_i

            n_current = layers[i].n
            if uz > 0.0:
                if i + 1 == n_layers:
                    n_next, next_layer = medium.n_outside_bottom, None
                else:
                    n_next, next_layer = layers[i + 1].n, i + 1
            else:
                if i == 0:
                    n_next, next_layer = medium.n_outside_top, None
                else:
                    n_next, next_layer = layers[i - 1].n, i - 1

            r_boundary = _fresnel_reflectance(abs(uz), n_current, n_next)
            if rng.random() < r_boundary:
                uz = -uz
                continue

            if next_layer is None:
                if uz > 0.0:
                    transmitted += weight
                else:
                    reflected += weight
                return reflected, transmitted, absorbed, max_z

            ux, uy, uz = _refract_direction(ux, uy, uz, n_current, n_next)
            i = next_layer

        d_weight = weight * (1.0 - layers[i].albedo)
        absorbed += d_weight
        weight -= d_weight

        cos_theta = _sample_henyey_greenstein(layers[i].g, rng)
        phi = 2.0 * np.pi * rng.random()
        ux, uy, uz = _scatter_direction(ux, uy, uz, cos_theta, phi)

        if weight < weight_threshold:
            if rng.random() <= 1.0 / roulette_survival:
                weight *= roulette_survival
            else:
                absorbed += weight
                return reflected, transmitted, absorbed, max_z

    return reflected, transmitted, absorbed, max_z


def run_with_depth(medium: LayeredMedium, n_photons: int, n_batches: int, seed: int,
                   split_depth: float | None = None):
    """Same batching convention as simulate_layered_medium, plus <z_max>_R.

    If ``split_depth`` is given, R-weight is additionally split into two
    sub-populations -- photons whose trajectory never crossed that depth
    ("shallow-only") and photons that did ("visited-deep") -- each with
    its own weight fraction and mean z_max. Used to test whether a single
    aggregate <z_max> is hiding two populations moving in different
    directions (see main()).
    """
    boundaries = medium.boundaries
    per_batch = n_photons // n_batches
    rd_fracs = []
    z_weighted_sum = 0.0
    r_weight_sum = 0.0
    shallow_w = deep_w = 0.0
    shallow_z_sum = deep_z_sum = 0.0

    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        acc_r = 0.0
        for _ in range(per_batch):
            r, t, a, zmax = _trace_one_photon_with_depth(medium, boundaries, rng, 1e-4, 10)
            acc_r += r
            if r > 0.0:
                z_weighted_sum += r * zmax
                r_weight_sum += r
                if split_depth is not None:
                    if zmax > split_depth:
                        deep_w += r
                        deep_z_sum += r * zmax
                    else:
                        shallow_w += r
                        shallow_z_sum += r * zmax
        rd_fracs.append(acc_r / per_batch)

    rd_fracs = np.array(rd_fracs)
    rd_mean = rd_fracs.mean()
    rd_stderr = rd_fracs.std(ddof=1) / np.sqrt(n_batches)
    mean_zmax = z_weighted_sum / r_weight_sum if r_weight_sum > 0 else float("nan")

    if split_depth is None:
        return rd_mean, rd_stderr, mean_zmax

    split = {
        "shallow_frac": shallow_w / r_weight_sum if r_weight_sum > 0 else float("nan"),
        "shallow_zmax": shallow_z_sum / shallow_w if shallow_w > 0 else float("nan"),
        "deep_frac": deep_w / r_weight_sum if r_weight_sum > 0 else float("nan"),
        "deep_zmax": deep_z_sum / deep_w if deep_w > 0 else float("nan"),
    }
    return rd_mean, rd_stderr, mean_zmax, split


def homogeneous_equivalent(surface: Layer, deep: Layer) -> SlabOpticalProperties:
    layers = [surface, deep]
    total = sum(l.thickness for l in layers)
    mu_a_avg = sum(l.mu_a * l.thickness for l in layers) / total
    musp_avg = sum(l.mu_s * (1 - l.g) * l.thickness for l in layers) / total
    g_avg = sum(l.g * l.thickness for l in layers) / total
    mu_s_avg = musp_avg / (1 - g_avg)
    return SlabOpticalProperties(mu_a=mu_a_avg, mu_s=mu_s_avg, g=g_avg,
                                 thickness=total, n_medium=N_INDEX, n_outside=1.0)


def main():
    print("=" * 92)
    print("PHOTON VISIT-DEPTH ANALYSIS -- mechanism behind the sign flip, at t = "
          f"{SURFACE_THICKNESS} mm")
    print("=" * 92)

    # --- sanity check: the instrumented tracer must reproduce the validated
    #     engine's own R to within statistical agreement before its <z_max>
    #     is trusted for anything.
    surface_a = Layer(mu_a=MU_A_HIGH, mu_s=SURFACE_MU_S, g=SURFACE_G,
                      thickness=SURFACE_THICKNESS, n=N_INDEX)
    deep_a = Layer(mu_a=MU_A_LOW, mu_s=DEEP_MU_S, g=DEEP_G,
                   thickness=DEEP_THICKNESS, n=N_INDEX)
    medium_a = LayeredMedium(layers=[surface_a, deep_a], n_outside_top=1.0, n_outside_bottom=1.0)

    ref_result = simulate_layered_medium(medium_a, n_photons=N_PHOTONS, n_batches=N_BATCHES, seed=SEED)
    rd_a, rd_a_se, z_a = run_with_depth(medium_a, N_PHOTONS, N_BATCHES, SEED)
    agree_sigma = abs(rd_a - ref_result.diffuse_reflectance) / \
        np.sqrt(rd_a_se**2 + ref_result.diffuse_reflectance_stderr**2)
    print(f"\nCross-check: instrumented tracer Rd = {rd_a:.4f} +/- {rd_a_se:.4f}")
    print(f"             validated engine    Rd = {ref_result.diffuse_reflectance:.4f} "
          f"+/- {ref_result.diffuse_reflectance_stderr:.4f}  ({agree_sigma:.2f} sigma)")
    assert agree_sigma < 3.0, "instrumented tracer disagrees with the validated engine -- stop"
    print("Agreement confirmed -- <z_max> from the instrumented tracer is trustworthy.\n")

    # --- Config A: strong absorber SHALLOW ---
    homog_a = homogeneous_equivalent(surface_a, deep_a)
    rd_a_h, rd_a_h_se, z_a_h = run_with_depth(
        LayeredMedium(layers=[Layer(mu_a=homog_a.mu_a, mu_s=homog_a.mu_s, g=homog_a.g,
                                    thickness=homog_a.thickness, n=N_INDEX)],
                     n_outside_top=1.0, n_outside_bottom=1.0),
        N_PHOTONS, N_BATCHES, SEED)

    # --- Config B: strong absorber DEEP ---
    surface_b = Layer(mu_a=MU_A_LOW, mu_s=SURFACE_MU_S, g=SURFACE_G,
                      thickness=SURFACE_THICKNESS, n=N_INDEX)
    deep_b = Layer(mu_a=MU_A_HIGH, mu_s=DEEP_MU_S, g=DEEP_G,
                   thickness=DEEP_THICKNESS, n=N_INDEX)
    medium_b = LayeredMedium(layers=[surface_b, deep_b], n_outside_top=1.0, n_outside_bottom=1.0)
    rd_b, rd_b_se, z_b = run_with_depth(medium_b, N_PHOTONS, N_BATCHES, SEED)

    homog_b = homogeneous_equivalent(surface_b, deep_b)
    rd_b_h, rd_b_h_se, z_b_h = run_with_depth(
        LayeredMedium(layers=[Layer(mu_a=homog_b.mu_a, mu_s=homog_b.mu_s, g=homog_b.g,
                                    thickness=homog_b.thickness, n=N_INDEX)],
                     n_outside_top=1.0, n_outside_bottom=1.0),
        N_PHOTONS, N_BATCHES, SEED)

    dR_a = rd_a - rd_a_h
    dR_a_se = np.sqrt(rd_a_se**2 + rd_a_h_se**2)
    dR_b = rd_b - rd_b_h
    dR_b_se = np.sqrt(rd_b_se**2 + rd_b_h_se**2)
    dz_a = z_a - z_a_h
    dz_b = z_b - z_b_h

    print(f"{'':22}{'Rd (layered)':>16}{'Rd (homog.)':>16}{'ΔR':>12}"
          f"{'<z_max> (mm)':>16}{'<z_max>_homog':>16}{'Δ<z_max>':>12}")
    print("-" * 110)
    print(f"{'A: strong SHALLOW':22}{rd_a:>16.4f}{rd_a_h:>16.4f}"
          f"{dR_a:>+12.4f}{z_a:>16.4f}{z_a_h:>16.4f}{dz_a:>+12.4f}")
    print(f"{'B: strong DEEP':22}{rd_b:>16.4f}{rd_b_h:>16.4f}"
          f"{dR_b:>+12.4f}{z_b:>16.4f}{z_b_h:>16.4f}{dz_b:>+12.4f}")
    print()
    print(f"sign(ΔR_A) = {'−' if dR_a < 0 else '+'}   sign(Δ<z_max>_A) = {'−' if dz_a < 0 else '+'}"
          f"   {'MATCH' if (dR_a < 0) == (dz_a < 0) else 'MISMATCH'}")
    print(f"sign(ΔR_B) = {'−' if dR_b < 0 else '+'}   sign(Δ<z_max>_B) = {'−' if dz_b < 0 else '+'}"
          f"   {'MATCH' if (dR_b < 0) == (dz_b < 0) else 'MISMATCH'}")

    # --- refined diagnostic: split R-weight at the surface/deep boundary ---
    # A single aggregate <z_max> can hide two sub-populations moving in
    # different directions. Split explicitly at the surface-layer
    # thickness: photons whose trajectory never crossed into the deep
    # layer ("shallow-only") vs those that did ("visited-deep").
    print("\n" + "=" * 92)
    print(f"REFINED: R-weight split at z = {SURFACE_THICKNESS} mm (the surface/deep interface)")
    print("=" * 92)

    _, _, _, split_a = run_with_depth(medium_a, N_PHOTONS, N_BATCHES, SEED, split_depth=SURFACE_THICKNESS)
    _, _, _, split_a_h = run_with_depth(
        LayeredMedium(layers=[Layer(mu_a=homog_a.mu_a, mu_s=homog_a.mu_s, g=homog_a.g,
                                    thickness=homog_a.thickness, n=N_INDEX)],
                     n_outside_top=1.0, n_outside_bottom=1.0),
        N_PHOTONS, N_BATCHES, SEED, split_depth=SURFACE_THICKNESS)
    _, _, _, split_b = run_with_depth(medium_b, N_PHOTONS, N_BATCHES, SEED, split_depth=SURFACE_THICKNESS)
    _, _, _, split_b_h = run_with_depth(
        LayeredMedium(layers=[Layer(mu_a=homog_b.mu_a, mu_s=homog_b.mu_s, g=homog_b.g,
                                    thickness=homog_b.thickness, n=N_INDEX)],
                     n_outside_top=1.0, n_outside_bottom=1.0),
        N_PHOTONS, N_BATCHES, SEED, split_depth=SURFACE_THICKNESS)

    for label, s, sh in (("A: strong SHALLOW", split_a, split_a_h), ("B: strong DEEP", split_b, split_b_h)):
        print(f"\n{label}:")
        print(f"  shallow-only R-weight fraction: layered={s['shallow_frac']:.3f}  "
              f"homog={sh['shallow_frac']:.3f}  Δ={s['shallow_frac']-sh['shallow_frac']:+.3f}")
        print(f"  visited-deep R-weight fraction: layered={s['deep_frac']:.3f}  "
              f"homog={sh['deep_frac']:.3f}  Δ={s['deep_frac']-sh['deep_frac']:+.3f}")


if __name__ == "__main__":
    main()
