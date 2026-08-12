"""
Tests the similarity relation of radiative transport: a medium with
(mu_s, g) is statistically equivalent, after many scattering events, to
a medium with (mu_s' = mu_s*(1-g), g' = 0) -- same reduced scattering,
zero anisotropy. Concretely: sweeping g at FIXED reduced scattering
mu_s' = mu_s*(1-g) (by adjusting mu_s to compensate) should leave the
diffuse reflectance and transmittance unchanged, to good approximation,
for a diffusive (multiply-scattering) slab.

This is a standard tissue-optics sanity check (see e.g. Jacques 2013)
and was specifically flagged as a validation step worth having as a
permanent test after reviewing an external methodology document that
used it -- it complements the existing analytical-limit tests
(Beer-Lambert, energy conservation, Fresnel) with a genuine invariance
property of the transport equation itself, not just a limiting case.

Run with: pytest -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402

N_PHOTONS = 8_000
MUSP_FIXED = 8.0   # mm^-1, reduced scattering held fixed across the sweep
THICKNESS = 2.0    # mm, several transport mean free paths -> diffusive regime
MU_A = 0.05


def _run(g: float, seed: int):
    mu_s = MUSP_FIXED / (1 - g)
    slab = SlabOpticalProperties(mu_a=MU_A, mu_s=mu_s, g=g, thickness=THICKNESS,
                                  n_medium=1.4, n_outside=1.0)
    return simulate_slab(slab, n_photons=N_PHOTONS, seed=seed)


def test_similarity_relation_diffuse_reflectance_is_g_invariant():
    """R_diffuse at fixed mu_s' should agree across g in {0, 0.5, 0.8}
    to within a few combined standard errors -- the similarity relation
    is an approximation (exact only in the diffusion limit), so this
    checks for approximate agreement, not bit-for-bit equality."""
    results = {g: _run(g, seed=3) for g in (0.0, 0.5, 0.8)}

    r0 = results[0.0].diffuse_reflectance
    for g in (0.5, 0.8):
        rg = results[g].diffuse_reflectance
        tol = 6 * max(results[0.0].diffuse_reflectance_stderr,
                       results[g].diffuse_reflectance_stderr, 1e-3)
        assert rg == pytest.approx(r0, abs=tol), (
            f"R_diffuse at g={g} ({rg:.4f}) deviates from g=0 ({r0:.4f}) "
            f"by more than the similarity relation should allow at fixed mu_s'."
        )


def test_similarity_relation_breaks_down_for_thin_non_diffusive_slabs():
    """The similarity relation is a diffusion-limit approximation: for a
    slab only a fraction of a transport mean free path thick, forward
    peaking (g) directly controls how many photons punch straight
    through, and R_diffuse should NOT be g-invariant here. This guards
    against the first test passing for a trivial reason (e.g. a bug
    that makes R_diffuse independent of g everywhere)."""
    thin = 0.05  # mm, thin compared to 1/musp' = 1/8 mm

    def run_thin(g, seed):
        mu_s = MUSP_FIXED / (1 - g)
        slab = SlabOpticalProperties(mu_a=MU_A, mu_s=mu_s, g=g, thickness=thin,
                                      n_medium=1.4, n_outside=1.0)
        return simulate_slab(slab, n_photons=N_PHOTONS, seed=seed)

    r_iso = run_thin(0.0, seed=4).diffuse_reflectance
    r_fwd = run_thin(0.9, seed=4).diffuse_reflectance

    assert abs(r_iso - r_fwd) > 0.02, (
        "Expected the similarity relation to break down for a thin, "
        "non-diffusive slab, but R_diffuse barely changed with g -- "
        "check that g is actually affecting single-scattering behaviour."
    )
