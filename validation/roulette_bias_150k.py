"""
roulette_bias_150k.py

Definitive, statistically robust measurement of the Russian-roulette
bookkeeping question raised in PROJECT_REPORT.md Section 11.5: how big
is the difference between (a) this project's shipped convention
(a terminated roulette packet's residual weight is credited to
`absorbed`) and (b) the textbook-unbiased convention (nothing is
credited on termination), at a much larger, more decisive sample size
than the N=3000 spot-check reported there?

Self-contained: only needs numpy. Does not require the
photon-transport-toolkit package to be installed -- the physics
functions below are copied verbatim from
src/photon_transport_toolkit/monte_carlo.py so the result is exactly
representative of the shipped code.

Both conventions are computed from the SAME random photon
trajectories in a single pass (they only diverge at the moment a
roulette packet is terminated), so the comparison is exact and free
of additional Monte Carlo sampling noise between the two variants --
only the true effect of the bookkeeping choice is measured.

Configuration is deliberately chosen to maximise how often Russian
roulette actually triggers: near-isotropic scattering (g=0, so photons
take a long random walk per unit physical depth) in a thick,
high-albedo medium (mu_s >> mu_a, so weight decays very slowly per
scattering event, meaning many events -- and hence many roulette
checks -- occur before a photon exits).

Usage:
    python roulette_bias_150k.py

Expect this to take a while (it is an intentionally worst-case,
roulette-heavy configuration) -- progress is printed after each batch
so you can see it is working and gauge total time from the first
couple of batches.
"""

import time
import numpy as np

# ---- physics helpers, copied verbatim from monte_carlo.py ----

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


def _sample_henyey_greenstein(g, rng):
    if abs(g) < 1e-6:
        return 2.0 * rng.random() - 1.0
    xi = rng.random()
    term = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
    return (1.0 + g * g - term * term) / (2.0 * g)


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


# ---- instrumented single-photon trace: computes BOTH conventions at once ----

def trace_photon(mu_a, mu_s, g, thickness, n_medium, n_outside, rng,
                  weight_threshold, roulette_survival):
    mu_t = mu_a + mu_s
    albedo = mu_s / mu_t
    x = y = z = 0.0
    ux = uy = 0.0
    uz = 1.0

    r_specular = _fresnel_reflectance(1.0, n_outside, n_medium)
    weight = 1.0 - r_specular

    reflected = transmitted = absorbed = 0.0
    absorbed_unbiased = 0.0
    n_events = 0
    n_survive = 0

    while weight > 0.0:
        tau = -np.log(rng.random())
        while True:
            step = tau / mu_t
            if uz > 0.0:
                dist_boundary = (thickness - z) / uz
            elif uz < 0.0:
                dist_boundary = -z / uz
            else:
                dist_boundary = np.inf

            if step < dist_boundary:
                x += step * ux; y += step * uy; z += step * uz
                break

            x += dist_boundary * ux; y += dist_boundary * uy; z += dist_boundary * uz
            z = 0.0 if uz < 0.0 else thickness
            tau -= dist_boundary * mu_t

            r_boundary = _fresnel_reflectance(abs(uz), n_medium, n_outside)
            if rng.random() > r_boundary:
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

        cos_th = _sample_henyey_greenstein(g, rng)
        phi = 2.0 * np.pi * rng.random()
        ux, uy, uz = _scatter_direction(ux, uy, uz, cos_th, phi)

        if weight < weight_threshold:
            n_events += 1
            if rng.random() <= 1.0 / roulette_survival:
                weight *= roulette_survival
                n_survive += 1
            else:
                absorbed += weight  # current shipped convention
                # absorbed_unbiased: credits nothing here (strict convention)
                return reflected, transmitted, absorbed, absorbed_unbiased, n_events, n_survive

    return reflected, transmitted, absorbed, absorbed_unbiased, n_events, n_survive


def main():
    # Deliberately roulette-heavy configuration.
    mu_a, mu_s, g, thickness = 0.05, 30.0, 0.0, 15.0
    n_medium, n_outside = 1.4, 1.0
    weight_threshold, roulette_survival = 1e-4, 10

    N_TOTAL = 150_000
    N_BATCHES = 15
    PER_BATCH = N_TOTAL // N_BATCHES
    SEED = 12345

    r_specular = _fresnel_reflectance(1.0, n_outside, n_medium)

    batch_current_sum = np.empty(N_BATCHES)
    batch_strict_sum = np.empty(N_BATCHES)
    total_events = 0
    total_survive = 0

    rng = np.random.default_rng(SEED)
    t0 = time.time()

    for b in range(N_BATCHES):
        acc_r = acc_t = acc_a = acc_a_ub = 0.0
        events = survive = 0
        for _ in range(PER_BATCH):
            r, t, a, a_ub, ev, sv = trace_photon(
                mu_a, mu_s, g, thickness, n_medium, n_outside, rng,
                weight_threshold, roulette_survival
            )
            acc_r += r; acc_t += t; acc_a += a; acc_a_ub += a_ub
            events += ev; survive += sv

        total_events += events
        total_survive += survive
        batch_current_sum[b] = r_specular + (acc_r + acc_t + acc_a) / PER_BATCH
        batch_strict_sum[b] = r_specular + (acc_r + acc_t + acc_a_ub) / PER_BATCH

        elapsed = time.time() - t0
        print(f"batch {b+1:2d}/{N_BATCHES}  "
              f"(events so far: {total_events}, elapsed: {elapsed:.1f}s)  "
              f"current_sum={batch_current_sum[b]:.6f}  strict_sum={batch_strict_sum[b]:.6f}")

    current_mean = batch_current_sum.mean()
    current_se = batch_current_sum.std(ddof=1) / np.sqrt(N_BATCHES)
    strict_mean = batch_strict_sum.mean()
    strict_se = batch_strict_sum.std(ddof=1) / np.sqrt(N_BATCHES)
    diff = current_mean - strict_mean
    diff_se = np.hypot(current_se, strict_se)

    total_time = time.time() - t0

    print()
    print("=" * 70)
    print(f"Total photons: {N_TOTAL:,}  |  Total roulette events triggered: {total_events:,} "
          f"({total_events/N_TOTAL:.4f} per photon)  |  Survivals: {total_survive:,}")
    print(f"Total wall time: {total_time:.1f}s")
    print()
    print(f"CURRENT (shipped) convention  R+T+A(+Rsp) = {current_mean:.7f} +/- {current_se:.7f}")
    print(f"STRICT-unbiased convention    R+T+A(+Rsp) = {strict_mean:.7f} +/- {strict_se:.7f}")
    print(f"Difference (current - strict)             = {diff:.7f} +/- {diff_se:.7f}")
    print(f"Difference in sigma                        = {abs(diff)/diff_se:.2f} sigma")
    print("=" * 70)


if __name__ == "__main__":
    main()
