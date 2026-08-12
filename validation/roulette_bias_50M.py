"""
roulette_bias_50M.py

Same measurement as roulette_bias_150k.py (PROJECT_REPORT.md Section
11.5), scaled to N=150; N=150,000; and N=50,000,000, to see whether the
Current-vs-Strict Russian-roulette bookkeeping difference stays exactly
where the N=150,000 run placed it (+3.7e-6, 17.7 sigma) or moves at
much higher statistical power.

At the N=150,000 timing already measured locally (177.8s, i.e. ~1.19
ms/photon in this deliberately roulette-heavy configuration), 50
million photons in the SAME pure-Python, one-photon-at-a-time form
would take on the order of 16 hours -- impractical. This version
JIT-compiles the hot per-photon loop with Numba, which typically gives
a 10-50x speedup for this kind of tight numerical loop with no
Python-object overhead, bringing 50M photons down to a realistic
range (expect roughly 20 minutes to a few hours depending on your
CPU -- the script prints progress after every batch so you can watch
the rate and extrapolate early).

Requirements: numpy, numba (``pip install numba``).

Numba's @njit does not support numpy's modern Generator API
(np.random.default_rng) called from inside jitted code, so this
version uses the legacy global numpy.random state (np.random.seed +
np.random.random) instead -- a different but equally valid
high-quality generator (Mersenne Twister rather than PCG64). Both
conventions are still computed from the exact same random draws per
photon in a single pass, so the Current-vs-Strict comparison itself
remains exact and free of extra sampling noise between the two
variants; only the choice of RNG *family* differs from the earlier
150k run, which is not expected to matter for a statistical
bias measurement like this one.

Usage:
    python roulette_bias_50M.py
"""

import time
import numpy as np
from numba import njit


# ---- physics helpers, JIT-compiled, same formulas as monte_carlo.py ----

@njit(cache=True)
def _fresnel_reflectance(cos_i, n1, n2):
    if n1 == n2:
        return 0.0
    sin_i2 = 1.0 - cos_i * cos_i
    sin_t2 = (n1 / n2) ** 2 * sin_i2
    if sin_t2 >= 1.0:
        return 1.0
    cos_t = np.sqrt(1.0 - sin_t2)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    return 0.5 * (rs * rs + rp * rp)


@njit(cache=True)
def _sample_henyey_greenstein(g):
    if abs(g) < 1e-6:
        return 2.0 * np.random.random() - 1.0
    xi = np.random.random()
    term = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
    return (1.0 + g * g - term * term) / (2.0 * g)


@njit(cache=True)
def _scatter_direction(ux, uy, uz, cos_theta, phi):
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    if abs(uz) > 1.0 - 1e-12:
        sign = 1.0 if uz >= 0 else -1.0
        return sin_theta * cos_phi, sin_theta * sin_phi, sign * cos_theta
    denom = np.sqrt(1.0 - uz * uz)
    ux_new = sin_theta * (ux * uz * cos_phi - uy * sin_phi) / denom + ux * cos_theta
    uy_new = sin_theta * (uy * uz * cos_phi + ux * sin_phi) / denom + uy * cos_theta
    uz_new = -sin_theta * cos_phi * denom + uz * cos_theta
    norm = np.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
    return ux_new / norm, uy_new / norm, uz_new / norm


@njit(cache=True)
def trace_photon(mu_a, mu_s, g, thickness, n_medium, n_outside,
                  weight_threshold, roulette_survival):
    mu_t = mu_a + mu_s
    albedo = mu_s / mu_t
    x = 0.0; y = 0.0; z = 0.0
    ux = 0.0; uy = 0.0; uz = 1.0

    r_specular = _fresnel_reflectance(1.0, n_outside, n_medium)
    weight = 1.0 - r_specular

    reflected = 0.0; transmitted = 0.0; absorbed = 0.0
    absorbed_unbiased = 0.0
    n_events = 0
    n_survive = 0

    while weight > 0.0:
        tau = -np.log(np.random.random())
        while True:
            step = tau / mu_t
            if uz > 0.0:
                dist_boundary = (thickness - z) / uz
            elif uz < 0.0:
                dist_boundary = -z / uz
            else:
                dist_boundary = 1.0e300

            if step < dist_boundary:
                x += step * ux; y += step * uy; z += step * uz
                break

            x += dist_boundary * ux; y += dist_boundary * uy; z += dist_boundary * uz
            z = 0.0 if uz < 0.0 else thickness
            tau -= dist_boundary * mu_t

            r_boundary = _fresnel_reflectance(abs(uz), n_medium, n_outside)
            if np.random.random() > r_boundary:
                if uz < 0.0:
                    reflected += weight
                else:
                    transmitted += weight
                return reflected, transmitted, absorbed, absorbed_unbiased, n_events, n_survive
            uz = -uz

        d_weight = weight * (1.0 - albedo)
        absorbed += d_weight
        absorbed_unbiased += d_weight
        weight -= d_weight

        cos_th = _sample_henyey_greenstein(g)
        phi = 2.0 * np.pi * np.random.random()
        ux, uy, uz = _scatter_direction(ux, uy, uz, cos_th, phi)

        if weight < weight_threshold:
            n_events += 1
            if np.random.random() <= 1.0 / roulette_survival:
                weight *= roulette_survival
                n_survive += 1
            else:
                absorbed += weight  # current shipped convention
                return reflected, transmitted, absorbed, absorbed_unbiased, n_events, n_survive

    return reflected, transmitted, absorbed, absorbed_unbiased, n_events, n_survive


@njit(cache=True)
def run_batch(n_photons, mu_a, mu_s, g, thickness, n_medium, n_outside,
              weight_threshold, roulette_survival):
    acc_r = 0.0; acc_t = 0.0; acc_a = 0.0; acc_a_ub = 0.0
    events = 0; survive = 0
    for _ in range(n_photons):
        r, t, a, a_ub, ev, sv = trace_photon(
            mu_a, mu_s, g, thickness, n_medium, n_outside,
            weight_threshold, roulette_survival
        )
        acc_r += r; acc_t += t; acc_a += a; acc_a_ub += a_ub
        events += ev; survive += sv
    return acc_r, acc_t, acc_a, acc_a_ub, events, survive


def run_tier(n_total, n_batches, mu_a, mu_s, g, thickness, n_medium, n_outside,
             weight_threshold, roulette_survival, seed, label):
    per_batch = n_total // n_batches
    r_specular = _fresnel_reflectance(1.0, n_outside, n_medium)

    batch_current = np.empty(n_batches)
    batch_strict = np.empty(n_batches)
    total_events = 0
    total_survive = 0

    np.random.seed(seed)
    t0 = time.time()
    for b in range(n_batches):
        acc_r, acc_t, acc_a, acc_a_ub, events, survive = run_batch(
            per_batch, mu_a, mu_s, g, thickness, n_medium, n_outside,
            weight_threshold, roulette_survival
        )
        total_events += events
        total_survive += survive
        batch_current[b] = r_specular + (acc_r + acc_t + acc_a) / per_batch
        batch_strict[b] = r_specular + (acc_r + acc_t + acc_a_ub) / per_batch

        elapsed = time.time() - t0
        rate = ((b + 1) * per_batch) / elapsed if elapsed > 0 else 0.0
        print(f"  [{label}] batch {b+1:3d}/{n_batches}  "
              f"elapsed={elapsed:7.1f}s  rate={rate:9.0f} photons/s  "
              f"events so far={total_events}")

    current_mean = batch_current.mean()
    current_se = batch_current.std(ddof=1) / np.sqrt(n_batches) if n_batches > 1 else 0.0
    strict_mean = batch_strict.mean()
    strict_se = batch_strict.std(ddof=1) / np.sqrt(n_batches) if n_batches > 1 else 0.0
    diff = current_mean - strict_mean
    diff_se = np.hypot(current_se, strict_se)
    total_time = time.time() - t0

    print(f"  [{label}] DONE in {total_time:.1f}s  |  "
          f"{total_events:,} roulette events across {n_total:,} photons "
          f"({total_events/n_total:.5f}/photon)  |  survivals={total_survive:,}")
    print(f"  [{label}] current={current_mean:.8f}+/-{current_se:.8f}  "
          f"strict={strict_mean:.8f}+/-{strict_se:.8f}  "
          f"diff={diff:.3e}+/-{diff_se:.3e}  "
          f"({abs(diff)/diff_se if diff_se>0 else float('nan'):.2f} sigma)")
    print()
    return dict(label=label, n=n_total, events=total_events, survive=total_survive,
                current=current_mean, current_se=current_se,
                strict=strict_mean, strict_se=strict_se,
                diff=diff, diff_se=diff_se, time=total_time)


def main():
    mu_a, mu_s, g, thickness = 0.05, 30.0, 0.0, 15.0
    n_medium, n_outside = 1.4, 1.0
    weight_threshold, roulette_survival = 1e-4, 10

    print("Warming up the JIT compiler (first call compiles, takes ~5-15s)...")
    _ = run_batch(50, mu_a, mu_s, g, thickness, n_medium, n_outside,
                  weight_threshold, roulette_survival)
    print("Compiled. Starting timed runs.\n")

    results = []
    results.append(run_tier(150, 5, mu_a, mu_s, g, thickness, n_medium, n_outside,
                             weight_threshold, roulette_survival, seed=1, label="N=150"))
    results.append(run_tier(150_000, 15, mu_a, mu_s, g, thickness, n_medium, n_outside,
                             weight_threshold, roulette_survival, seed=12345, label="N=150,000"))
    results.append(run_tier(50_000_000, 50, mu_a, mu_s, g, thickness, n_medium, n_outside,
                             weight_threshold, roulette_survival, seed=999, label="N=50,000,000"))

    print("=" * 90)
    print(f"{'Tier':>14} {'events/photon':>14} {'current':>14} {'strict':>14} {'diff':>14} {'sigma':>8}")
    for r in results:
        sig = abs(r['diff']) / r['diff_se'] if r['diff_se'] > 0 else float('nan')
        print(f"{r['label']:>14} {r['events']/r['n']:>14.5f} {r['current']:>14.8f} "
              f"{r['strict']:>14.8f} {r['diff']:>14.3e} {sig:>8.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
