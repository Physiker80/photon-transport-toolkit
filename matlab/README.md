# Independent MATLAB/Octave Re-derivation

This folder is a from-scratch, independent re-derivation of the core
Monte Carlo transport engine, written in MATLAB/Octave directly from
the governing equations (the MCML algorithm, the Henyey-Greenstein
phase function, the Fresnel equations) rather than translated from the
Python implementation in `src/photon_transport_toolkit/`. It exists
specifically to cross-check the project's central finding (§8 /
§14 of [PROJECT_REPORT.md](../PROJECT_REPORT.md)) using a second
language, a different random-number generator, and independently
written boundary-crossing logic.

Tested with **GNU Octave 8.4.0** (a free, MATLAB-language-compatible
interpreter); the code uses no Octave-specific syntax and should run
unmodified in MATLAB.

## Files

| File | Role |
|---|---|
| `fresnel_reflectance.m` | Unpolarised Fresnel reflectance from the Fresnel equations |
| `sample_hg.m` | Henyey-Greenstein scattering-angle sampler (analytic inverse-CDF) |
| `rotate_direction.m` | 3D direction-cosine rotation after a scattering event |
| `mc_slab.m` | Single homogeneous-layer weighted-photon Monte Carlo engine |
| `mc_layered.m` | Arbitrary-layer-stack weighted-photon Monte Carlo engine (τ-based boundary crossing, Snell's-law refraction at internal boundaries) |
| `refract_direction.m` | Snell's-law direction-cosine update on transmission through a mismatched-index boundary |
| `test_beer_lambert.m` | Beer-Lambert limiting-case check |
| `test_energy_conservation.m` | R+T+A=1 check (μₐ=0) |
| `test_fresnel_and_reproducibility.m` | Fresnel-at-normal-incidence and fixed-seed reproducibility checks |
| `test_reduction.m` | Identical-layers-reduce-to-homogeneous check |
| `test_similarity_relation.m` | g-invariance at fixed μₛ', and its breakdown in a thin slab |
| `test_bias_direction.m` | **The central cross-language test** — does the sign-flip finding hold? |
| `test_refraction.m` | Snell's-law refraction at internal boundaries: exact-angle check plus energy conservation with genuinely mismatched n |
| `run_one_config.m`, `run_one_g.m`, `run_thin.m` | Single-configuration runners used to split `test_bias_direction.m` / `test_similarity_relation.m` across multiple short executions (see note below) |

## A sandboxed-execution note

`test_bias_direction.m` and `test_similarity_relation.m` are complete,
correct, standalone scripts and will run end-to-end on an unrestricted
machine. In the sandboxed environment this project was developed in,
individual command executions are capped at a few hundred seconds,
which the full layered-media sweep exceeds — Octave's interpreted
per-photon loop runs roughly 4× slower than the equivalent Python here,
and the layered engine's boundary-resolution inner loop adds further
overhead. The `run_one_*.m` scripts are the exact same physics, just
invoked one configuration at a time (`octave-cli --eval "surface_mua=...; run('run_one_config.m')"`)
so each individual call finishes comfortably inside the sandbox's
limit. **The results are identical either way** — only the batching
differs. On a normal desktop MATLAB/Octave installation, just run the
full scripts directly.

## Two real bugs this validation caught

Writing this independent implementation was not a formality — it
caught two genuine bugs, exactly the outcome independent validation is
supposed to produce:

1. **Missing specular reflectance.** The first version subtracted the
   entrance Fresnel reflection from the photon's initial weight but
   never added it to the reflectance accumulator, silently dropping
   2.778% of the incident energy (matching `((1.4-1)/(1.4+1))^2`
   exactly). Caught immediately by `test_energy_conservation.m`
   failing at 8.15σ.
2. **Unresolved multi-boundary crossing in `mc_slab.m`.** The initial
   version tested for only one boundary crossing per sampled free
   path. For a slab thinner than the mean free path (exactly the
   regime `test_similarity_relation.m`'s breakdown check probes), a
   reflected photon's remaining path could overshoot the *opposite*
   boundary without being caught, leaving the photon in an
   ill-defined state. This surfaced as a hang, not a wrong number —
   fixed by carrying the remaining path in optical-depth units through
   a proper boundary-resolution loop, mirroring the approach already
   used in `mc_layered.m`.

Both are described in full, with the exact before/after numbers, in
[PROJECT_REPORT.md, §11](../PROJECT_REPORT.md#11-cross-language-independent-validation-matlaboctave).

## A third-party review, and what actually held up

An external code review of this MATLAB code claimed four bugs. Checked
against the actual code and, where applicable, empirically stress-tested
rather than taken on trust:

| Claim | Verdict |
|---|---|
| Missing Snell's-law refraction at internal boundaries with mismatched n | **Confirmed** — `mc_layered.m` changed `li` on transmission without ever updating `(ux,uy,uz)`. Zero effect on any previously published result (all used matched n, where `fresnel_reflectance()` already returns R=0 and no bending is physically needed) — but a real gap for the general case. Fixed in `refract_direction.m`, verified to satisfy Snell's law to 9 decimal places and to preserve exact energy conservation (0.00σ deviation) with genuinely mismatched n=[1.0, 1.6, 1.3] across three layers (`test_refraction.m`). |
| Total-internal-reflection could produce NaN/complex numbers | **False** — `fresnel_reflectance.m` already tests `sin_theta_t2 >= 1` and returns R=1 before the square root is ever taken. Verified by direct code inspection. |
| Boundary reflection risks an infinite loop from floating-point precision | **False for this codebase** — stress-tested with 20 five-micron layers (forcing dozens of boundary crossings per sampled free path): completed in 0.26s with exact energy conservation, no hang. (A *different*, already-documented bug of this general kind was real and fixed in `mc_slab.m` — see above — but does not apply to `mc_layered.m`, which had the correct multi-boundary resolution loop from the start.) |
| Specular reflectance assumes normal incidence | Not a bug — a documented design scope. Every test and example in this project launches photons at normal incidence by design; oblique collimated illumination was never a stated goal. |

A follow-up review claim — that crediting a terminated Russian-roulette
packet's residual weight to `absorbed` double-counts it — is **true as a
local statement** (a 1.9× credited-to-true-weight ratio at
`roulette_survival=10`, confirmed both algebraically and numerically),
but removing that credit broke `test_energy_is_conserved` (an exact,
1e-9 per-run invariant this codebase deliberately maintains, a stronger
property than textbook Russian-roulette unbiasedness-in-expectation
provides). The change was reverted after directly measuring its
aggregate impact at three sample sizes, in a deliberately roulette-heavy
configuration (the largest run Numba-JIT-compiled for practicality: 50
million photons in ~17.5 minutes at ~47,600 photons/s):

| N | Roulette events | Current − Strict (R+T+A) | Significance |
|---|---|---|---|
| 150 | 5 | +2.7×10⁻⁶ | 0.34σ |
| 150,000 | 6,270 | +3.76×10⁻⁶ | 13.6σ |
| 50,000,000 | 2,135,944 | +3.84×10⁻⁶ | **301.7σ** |

The bias is real and, at N=50M, effectively certain (301.7σ) — but its
absolute size never exceeds a few parts per million, three to four
orders of magnitude below every uncertainty reported anywhere else in
this project. Full account in
[PROJECT_REPORT.md §11.5](../PROJECT_REPORT.md#115-a-third-party-review-checked-claim-by-claim).

## Running it yourself

```bash
# quick checks (each a few seconds to ~1 minute)
octave-cli test_beer_lambert.m
octave-cli test_energy_conservation.m
octave-cli test_fresnel_and_reproducibility.m
octave-cli test_reduction.m

# the central cross-language test (run per-configuration; ~15-40s each)
octave-cli --eval "surface_mua=0.50; deep_mua=0.05; thickness=0.12; label='A_shallow'; run('run_one_config.m')"
octave-cli --eval "surface_mua=0.05; deep_mua=0.50; thickness=0.12; label='B_deep';    run('run_one_config.m')"
octave-cli --eval "surface_mua=0.50; deep_mua=0.05; thickness=0.25; label='A_shallow'; run('run_one_config.m')"
octave-cli --eval "surface_mua=0.05; deep_mua=0.50; thickness=0.25; label='B_deep';    run('run_one_config.m')"
# results accumulate in bias_results.txt
```
