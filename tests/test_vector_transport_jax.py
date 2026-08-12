"""Tests for the optional JAX-accelerated backend (vector_transport_jax.py).

Every test in this file needs the ``jax`` package specifically -- unlike
``tests/test_mie.py``'s single cross-check test, there is no meaningful
subset of this file that runs without it -- so the whole module is
skipped via ``pytest.importorskip`` when jax is absent, rather than
skipping test by test.

Correctness here is checked the same way the MATLAB/Octave re-derivation
was checked against the Python engine (PROJECT_REPORT.md, Cross-Language
Independent Validation): not bit-for-bit reproduction -- JAX's explicit
PRNG keys draw a different random stream than
``numpy.random.Generator`` from the same integer seed by construction --
but statistical agreement with the reference NumPy engine, within their
combined uncertainty, on a slab where both are computing the same
physics.

This file also pins down the two numbers this project's own benchmark
record depends on for its recommendation (see the module docstring of
``vector_transport_jax.py`` and PROJECT_REPORT.md): energy conservation
must be exact, and the round-based scattering sampler must resolve
every event within its budget for a physically ordinary slab -- if
either regresses, silently wrong results would follow, and both are
measured directly rather than assumed.

Author: Noureddin Sedki
License: MIT
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

jax = pytest.importorskip("jax", reason="the JAX backend is optional; see vector_transport_jax.py")

from photon_transport_toolkit import SlabOpticalProperties  # noqa: E402
from photon_transport_toolkit.vector_transport import simulate_slab_vector  # noqa: E402
from photon_transport_toolkit.vector_transport_jax import (  # noqa: E402
    RECOMMENDED_MIN_PHOTONS_FOR_JAX,
    recommend_backend,
    simulate_slab_vector_jax,
)

# Kept small deliberately: these tests check *correctness*, not the
# large-N regime the backend is actually for. Compile cost (a few
# seconds, see the module docstring's benchmark table) dominates at
# this size regardless of framework, which is exactly the point --
# this file is not a performance test.
_SLAB = SlabOpticalProperties(mu_a=0.1, mu_s=3.0, g=0.0, thickness=0.6)
_N_PHOTONS = 3000
_N_BATCHES = 6
_MAX_ROUNDS = 4000
_SEED = 7


def _combined_z(a, a_err, b, b_err):
    return abs(a - b) / np.hypot(a_err, b_err)


# --------------------------------------------------------------------------
# Physical invariants
# --------------------------------------------------------------------------


def test_energy_conservation_is_exact():
    """R + T + A + R_sp = 1, the same invariant vector_transport.py maintains.

    Checked to 1e-5 rather than the NumPy engine's 1e-9: this backend
    is float32/complex64 throughout (see the module docstring for why
    that precision is adequate at the batch sizes this backend targets),
    so the tolerance reflects that deliberate choice rather than a
    weaker guarantee about the physics.
    """
    res = simulate_slab_vector_jax(_SLAB, 633.0, n_photons=_N_PHOTONS, n_batches=_N_BATCHES,
                                   seed=_SEED, max_rounds=_MAX_ROUNDS)
    r_sp = ((_SLAB.n_outside - _SLAB.n_medium) / (_SLAB.n_outside + _SLAB.n_medium)) ** 2
    total = res.diffuse_reflectance + res.transmittance + res.absorbed + r_sp
    assert total == pytest.approx(1.0, abs=1e-5)


def test_degree_of_polarization_never_exceeds_unity():
    res = simulate_slab_vector_jax(_SLAB, 633.0, n_photons=_N_PHOTONS, n_batches=_N_BATCHES,
                                   seed=_SEED, max_rounds=_MAX_ROUNDS)
    # float32 rounding, not a physical violation -- see max_dop_violation's
    # own docstring; this bound is generous on purpose.
    assert res.max_dop_violation < 1e-4


def test_no_runaway_or_unresolved_scattering_for_an_ordinary_slab():
    """Both diagnostic counters should be exactly zero here.

    A nonzero ``n_unresolved`` would mean the round budget was too
    small for this (unremarkable) slab's photon lifetimes. A nonzero
    ``n_unresolved_scatter_events`` would mean ``scatter_rounds`` is
    too small for Rayleigh's rejection-sampling acceptance rate. Either
    would silently bias the result if it went unmeasured -- which is
    why both are counted rather than assumed away, and why this test
    exists rather than trusting the defaults.
    """
    res = simulate_slab_vector_jax(_SLAB, 633.0, n_photons=_N_PHOTONS, n_batches=_N_BATCHES,
                                   seed=_SEED, max_rounds=_MAX_ROUNDS)
    assert res.n_unresolved == 0
    assert res.n_unresolved_scatter_events == 0


# --------------------------------------------------------------------------
# Cross-validation against the reference NumPy engine
# --------------------------------------------------------------------------


@pytest.mark.parametrize("polarization", ["x", "circular", "unpolarized"])
def test_agrees_with_the_reference_engine_statistically(polarization):
    """Same slab, same physics, two independent RNG streams and frameworks.

    5-sigma rather than the usual 3: this backend's per-batch statistic
    is a Bernoulli-like fraction from only a few thousand photons per
    batch (kept small so the test suite stays fast), so batch-to-batch
    stderr estimated from just a handful of batches is itself noisy;
    5-sigma keeps the false-failure rate low without hiding a real
    disagreement, which would show up as a much larger z in practice
    (the checks during development, at this same size, landed at
    0.02-0.83 sigma across several seeds and polarizations).
    """
    ref = simulate_slab_vector(_SLAB, 633.0, n_photons=_N_PHOTONS, n_batches=_N_BATCHES,
                               seed=_SEED, polarization=polarization, detector_bins=4)
    jx = simulate_slab_vector_jax(_SLAB, 633.0, n_photons=_N_PHOTONS // _N_BATCHES,
                                  n_batches=_N_BATCHES, seed=_SEED, polarization=polarization,
                                  max_rounds=_MAX_ROUNDS)
    assert _combined_z(ref.diffuse_reflectance, ref.diffuse_reflectance_stderr,
                       jx.diffuse_reflectance, jx.diffuse_reflectance_stderr) < 5.0
    assert _combined_z(ref.transmittance, ref.transmittance_stderr,
                       jx.transmittance, jx.transmittance_stderr) < 5.0


def test_reflected_dolp_agrees_with_the_reference_engine():
    """The polarization result, not just the radiometric one.

    Deliberately at moderate scattering (this project's own §5.3 /
    §13.6 finding: Rayleigh and HG(g=0) agree well precisely in the
    diffusive regime), so a real disagreement in the frame algebra --
    the part unique to this backend's flattened boundary/scatter loop
    -- would not be masked by the Rayleigh-vs-HG gap discussed
    elsewhere in this project.
    """
    ref = simulate_slab_vector(_SLAB, 633.0, n_photons=_N_PHOTONS, n_batches=_N_BATCHES,
                               seed=11, polarization="x", detector_bins=4)
    jx = simulate_slab_vector_jax(_SLAB, 633.0, n_photons=_N_PHOTONS // _N_BATCHES,
                                  n_batches=_N_BATCHES, seed=11, polarization="x",
                                  max_rounds=_MAX_ROUNDS)

    def dolp(s):
        return np.hypot(s[1], s[2]) / s[0]

    # Loose tolerance: stokes_total's own uncertainty is not tracked
    # per-component (only Rd/T get a stderr in either engine), so this
    # is a sanity bound, not a precision claim.
    assert abs(dolp(ref.stokes_total) - dolp(jx.stokes_total)) < 0.08


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------


def test_nonzero_g_is_rejected():
    """Same deliberate refusal as vector_transport.py's own, for the same reason."""
    slab = SlabOpticalProperties(mu_a=0.1, mu_s=6.0, g=0.5, thickness=1.0)
    with pytest.raises(ValueError, match="slab.g must be 0"):
        simulate_slab_vector_jax(slab, 633.0, n_photons=500, n_batches=2)


def test_invalid_sizes_are_rejected():
    with pytest.raises(ValueError):
        simulate_slab_vector_jax(_SLAB, 633.0, n_photons=0, n_batches=2)
    with pytest.raises(ValueError):
        simulate_slab_vector_jax(_SLAB, 633.0, n_photons=500, n_batches=0)


def test_unknown_polarization_is_rejected():
    with pytest.raises(ValueError):
        simulate_slab_vector_jax(_SLAB, 633.0, n_photons=500, n_batches=2, polarization="bogus")


# --------------------------------------------------------------------------
# Backend recommendation
# --------------------------------------------------------------------------


def test_recommend_backend_matches_the_measured_crossover():
    """The threshold is a measured recommendation, not a round number.

    Values below come from timing both engines against each other on
    this project's own development hardware (single CPU core, no GPU
    available to test) at several batch sizes -- see
    PROJECT_REPORT.md's JAX-backend section for the table. The backend
    was *slower*, including steady state, below roughly this size;
    this test pins the constant to that finding so a future edit
    cannot silently drift the recommendation away from measured data
    back toward the optimistic (and empirically wrong, for this
    workload) assumption that batching alone implies a speedup.
    """
    assert recommend_backend(RECOMMENDED_MIN_PHOTONS_FOR_JAX - 1) == "numpy"
    assert recommend_backend(RECOMMENDED_MIN_PHOTONS_FOR_JAX) == "jax"
    assert recommend_backend(1000, n_batches=1) == "numpy"
    assert recommend_backend(RECOMMENDED_MIN_PHOTONS_FOR_JAX // 2, n_batches=2) == "jax"


def test_jax_backend_reports_itself_available_when_importable():
    from photon_transport_toolkit.vector_transport_jax import JAX_AVAILABLE
    assert JAX_AVAILABLE is True


def test_require_jax_raises_a_clear_error_when_unavailable(monkeypatch):
    """Simulates the no-jax-installed path without requiring an actual
    second environment -- the failure mode a person without jax
    installed will actually see.
    """
    import photon_transport_toolkit.vector_transport_jax as vtj
    monkeypatch.setattr(vtj, "JAX_AVAILABLE", False)
    with pytest.raises(ImportError, match="optional"):
        vtj._require_jax()
