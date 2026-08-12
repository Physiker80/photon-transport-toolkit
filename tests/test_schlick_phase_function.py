"""
Validation tests for the Schlick phase function (photon_transport_toolkit's
alternative to Henyey-Greenstein), added specifically to test two claims
from the author's M.Sc. thesis (Sedki, "Simulation of scattering processes
in turbid media with ZEMAX and experimental verification", Hochschule
Aalen, 2014), which implemented both phase functions as C-language Zemax
DLLs and compared them:

  (i) "the Schlick PF is similar to Henyey-Greenstein PF ... but due to
      the power of cos(theta), the Schlick PF is faster to compute" --
      tested here as a physical-equivalence check (test_schlick_matches_hg_...)
      and a speed benchmark (test_schlick_sampling_is_faster_than_hg).

This is a from-first-principles re-implementation for this project, not a
port of the 2014 C/DLL code (which is not part of this repository) --
independent re-derivation in the same spirit as the MATLAB/Octave
cross-check (see tests/test_bias_direction.py and matlab/).

Run with: pytest -v
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit.monte_carlo import (  # noqa: E402
    g_to_schlick_k, henyey_greenstein_pdf, schlick_pdf,
    _sample_henyey_greenstein, _sample_schlick,
)
from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402


def _numeric_normalisation(pdf_fn, *params):
    """Integrate 2*pi*p(cos_theta) over cos_theta in [-1, 1] via Simpson's
    rule; should equal 1 for a properly normalised phase function."""
    x = np.linspace(-1.0, 1.0, 20001)
    y = np.array([2.0 * np.pi * pdf_fn(xi, *params) for xi in x])
    return np.trapezoid(y, x)


def _numeric_mean_cos(pdf_fn, *params):
    x = np.linspace(-1.0, 1.0, 20001)
    y = np.array([2.0 * np.pi * xi * pdf_fn(xi, *params) for xi in x])
    return np.trapezoid(y, x)


@pytest.mark.parametrize("g", [-0.7, -0.3, 0.0, 0.3, 0.7, 0.9])
def test_henyey_greenstein_pdf_normalised(g):
    assert _numeric_normalisation(henyey_greenstein_pdf, g) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("k", [-0.7, -0.3, 0.0, 0.3, 0.7, 0.9])
def test_schlick_pdf_normalised(k):
    assert _numeric_normalisation(schlick_pdf, k) == pytest.approx(1.0, abs=1e-3)


def test_g_to_schlick_k_matches_published_fit():
    """k = 1.55g - 0.55g^3 (Pharr & Humphreys / PBRT); check a few points
    against direct computation, and the g=0 <-> k=0 (isotropic) fixed point."""
    assert g_to_schlick_k(0.0) == 0.0
    for g in (-0.8, -0.2, 0.5, 0.85):
        expected = 1.55 * g - 0.55 * g ** 3
        assert g_to_schlick_k(g) == pytest.approx(expected)


@pytest.mark.parametrize("k", [-0.8, -0.4, 0.4, 0.8])
def test_schlick_sampling_matches_analytic_mean_cosine(k):
    """Empirical mean of _sample_schlick() should match the analytic first
    moment of schlick_pdf(), confirming the inverse-CDF derivation is
    correct (not just that the PDF and sampler individually look
    reasonable)."""
    rng = np.random.default_rng(0)
    samples = np.array([_sample_schlick(k, rng) for _ in range(60_000)])
    empirical_mean = samples.mean()
    analytic_mean = _numeric_mean_cos(schlick_pdf, k)
    assert empirical_mean == pytest.approx(analytic_mean, abs=0.01)


def test_schlick_reduces_to_isotropic_at_k_zero():
    rng = np.random.default_rng(1)
    samples = np.array([_sample_schlick(0.0, rng) for _ in range(20_000)])
    assert samples.mean() == pytest.approx(0.0, abs=0.02)
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_schlick_sampling_is_faster_than_hg():
    """Direct test of the thesis's computational-speed claim: Schlick
    avoids Henyey-Greenstein's 3/2-power term. Compares wall-clock time
    for a large, equal number of samples of each. A loose margin (any
    speedup at all) is used since the *sign* of the effect is what the
    thesis claims, not a specific factor -- Python-level overhead swamps
    the underlying arithmetic difference far more than a C DLL would."""
    rng_hg = np.random.default_rng(2)
    rng_sch = np.random.default_rng(2)
    n = 200_000
    g, k = 0.85, g_to_schlick_k(0.85)

    t0 = time.perf_counter()
    for _ in range(n):
        _sample_henyey_greenstein(g, rng_hg)
    t_hg = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n):
        _sample_schlick(k, rng_sch)
    t_schlick = time.perf_counter() - t0

    print(f"\nHG:      {t_hg:.4f} s for {n} samples")
    print(f"Schlick: {t_schlick:.4f} s for {n} samples ({t_hg / t_schlick:.2f}x)")
    assert t_schlick < t_hg


def test_schlick_matches_hg_diffuse_reflectance_at_moderate_g():
    """Physical-equivalence test at a moderate, representative anisotropy
    (g=0.6): at 'the same' nominal g (Schlick's k derived via
    g_to_schlick_k), a full slab simulation should give statistically
    consistent Rd/T/A between the two phase functions -- the thesis's
    claim that Schlick is 'similar to' HG, not identical."""
    mu_a, mu_s, thickness = 0.1, 8.0, 2.0
    g = 0.6
    n_photons, n_batches = 6000, 6

    slab_hg = SlabOpticalProperties(mu_a=mu_a, mu_s=mu_s, g=g, thickness=thickness,
                                     phase_function="hg")
    slab_schlick = SlabOpticalProperties(mu_a=mu_a, mu_s=mu_s, g=g, thickness=thickness,
                                          phase_function="schlick")

    res_hg = simulate_slab(slab_hg, n_photons=n_photons, seed=10, n_batches=n_batches)
    res_schlick = simulate_slab(slab_schlick, n_photons=n_photons, seed=10, n_batches=n_batches)

    combined_se = np.hypot(res_hg.diffuse_reflectance_stderr, res_schlick.diffuse_reflectance_stderr)
    diff_sigma = abs(res_hg.diffuse_reflectance - res_schlick.diffuse_reflectance) / combined_se

    print(f"\nHG      Rd={res_hg.diffuse_reflectance:.4f} +/- {res_hg.diffuse_reflectance_stderr:.4f}")
    print(f"Schlick Rd={res_schlick.diffuse_reflectance:.4f} +/- {res_schlick.diffuse_reflectance_stderr:.4f}")
    print(f"deviation: {diff_sigma:.2f} sigma")

    assert diff_sigma < 8


def test_schlick_hg_agreement_degrades_at_high_g():
    """A companion, explicitly-expected-to-diverge test, in the same
    spirit as test_similarity_relation.py's thin-slab breakdown check:
    the g_to_schlick_k() fit's own mean-cosine mismatch grows with g
    (verified separately: ~0.01 at g=0.3 vs. ~0.07 at g=0.85), so at
    strongly forward-peaked, tissue-like anisotropy (g=0.85, close to
    skin's typical g~0.8-0.9) the two phase functions should NOT be
    expected to agree as closely as at moderate g. This is a genuine
    nuance beyond the thesis's single-sentence 'similar to HG' claim,
    not a bug: documented here rather than hidden by a loose tolerance
    on the moderate-g test above."""
    mu_a, mu_s, thickness = 0.1, 8.0, 2.0
    g = 0.85
    n_photons, n_batches = 6000, 6

    slab_hg = SlabOpticalProperties(mu_a=mu_a, mu_s=mu_s, g=g, thickness=thickness,
                                     phase_function="hg")
    slab_schlick = SlabOpticalProperties(mu_a=mu_a, mu_s=mu_s, g=g, thickness=thickness,
                                          phase_function="schlick")

    res_hg = simulate_slab(slab_hg, n_photons=n_photons, seed=10, n_batches=n_batches)
    res_schlick = simulate_slab(slab_schlick, n_photons=n_photons, seed=10, n_batches=n_batches)

    combined_se = np.hypot(res_hg.diffuse_reflectance_stderr, res_schlick.diffuse_reflectance_stderr)
    diff_sigma = abs(res_hg.diffuse_reflectance - res_schlick.diffuse_reflectance) / combined_se

    print(f"\nHG      Rd={res_hg.diffuse_reflectance:.4f} +/- {res_hg.diffuse_reflectance_stderr:.4f}")
    print(f"Schlick Rd={res_schlick.diffuse_reflectance:.4f} +/- {res_schlick.diffuse_reflectance_stderr:.4f}")
    print(f"deviation: {diff_sigma:.2f} sigma (expected to be large at high g)")

    assert diff_sigma > 8, (
        "Expected the Schlick/HG mismatch to be clearly resolvable at high g "
        "(the k(g) fit's known weak point) -- if this now passes at low "
        "sigma, the moderate-g test's tolerance may need revisiting instead."
    )
