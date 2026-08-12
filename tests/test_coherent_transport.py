"""
Validation for the coherent-field extension (coherent_transport.py) --
Phase 1 of the PhD research roadmap: extending the validated
intensity-only engine to track physical path length and phase.

The central claim tested here: coherently summing sqrt(weight)*exp(i*phase)
contributions over a detector plane and taking |field|^2 must reproduce
the already-validated scalar engine's diffuse reflectance, in
expectation. A genuine, expected complication (found during
development, not assumed away): a SINGLE realization's sum(|field|^2)
has much higher statistical variance than the scalar Rd estimator at
the same photon budget -- a real feature of coherent speckle statistics
(the effective number of independent samples is closer to the number
of speckle grains than the number of photons), not a bug. The
reduction test below accounts for this correctly by averaging over
many independent seeds, exactly like any other Monte Carlo quantity
in this project, rather than by loosening tolerance until a single
noisy realization happens to pass.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402
from photon_transport_toolkit.coherent_transport import simulate_slab_coherent  # noqa: E402


SLAB = SlabOpticalProperties(mu_a=0.1, mu_s=8.0, g=0.8, thickness=2.0)


def test_coherent_engine_scalar_part_matches_validated_engine_exactly():
    """With the same seed, the coherent tracer's R/T/A must match the
    validated scalar engine bit-for-bit -- the coherent extension adds
    path-length bookkeeping only, touching none of the physics the
    scalar engine is already validated against."""
    ref = simulate_slab(SLAB, n_photons=800, seed=3, n_batches=4)
    coh = simulate_slab_coherent(SLAB, wavelength_nm=550, n_photons=800, seed=3, n_batches=4,
                                  detector_bins=16, detector_half_width=15.0)
    assert coh.diffuse_reflectance == pytest.approx(ref.diffuse_reflectance, abs=1e-12)
    assert coh.transmittance == pytest.approx(ref.transmittance, abs=1e-12)


def test_out_of_bounds_photons_are_excluded_from_the_field_only():
    """A small detector should still report the correct scalar Rd (which
    doesn't care about transverse position) while visibly under-counting
    sum(|field|^2) (which does) -- confirms the documented truncation
    behaviour is real and one-directional (never over-counts), rather
    than silently broken."""
    coh_small = simulate_slab_coherent(SLAB, wavelength_nm=550, n_photons=1500, seed=5, n_batches=5,
                                        detector_bins=32, detector_half_width=1.0)
    coh_large = simulate_slab_coherent(SLAB, wavelength_nm=550, n_photons=1500, seed=5, n_batches=5,
                                        detector_bins=32, detector_half_width=20.0)
    assert coh_small.diffuse_reflectance == pytest.approx(coh_large.diffuse_reflectance, abs=1e-12)
    sum_small = np.sum(np.abs(coh_small.field_reflected) ** 2)
    sum_large = np.sum(np.abs(coh_large.field_reflected) ** 2)
    assert sum_small < sum_large  # truncation can only remove energy, never add it


def test_reduction_ensemble_mean_matches_scalar_diffuse_reflectance():
    """The central validation: E[sum(|field|^2)] over many independent
    seeds must equal the validated scalar engine's diffuse reflectance
    -- checked as an ensemble mean +/- standard error, the same way
    every other Monte Carlo claim in this project is checked, not as a
    single noisy realization."""
    ref = simulate_slab(SLAB, n_photons=3000 * 6, seed=0, n_batches=10)

    field_sums = []
    for seed in range(20):
        coh = simulate_slab_coherent(SLAB, wavelength_nm=550, n_photons=3000, seed=seed, n_batches=3,
                                      detector_bins=48, detector_half_width=15.0)
        field_sums.append(np.sum(np.abs(coh.field_reflected) ** 2))
    field_sums = np.array(field_sums)
    ensemble_mean = field_sums.mean()
    ensemble_se = field_sums.std(ddof=1) / np.sqrt(len(field_sums))

    sigma = abs(ensemble_mean - ref.diffuse_reflectance) / ensemble_se
    print(f"\nensemble mean(sum|E|^2)={ensemble_mean:.5f}+/-{ensemble_se:.5f} "
          f"vs scalar Rd={ref.diffuse_reflectance:.5f}  ({sigma:.2f} sigma)")
    assert sigma < 4


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        simulate_slab_coherent(SLAB, wavelength_nm=-1, n_photons=100, n_batches=2)
    with pytest.raises(ValueError):
        simulate_slab_coherent(SLAB, wavelength_nm=550, n_photons=100, n_batches=1)
    with pytest.raises(ValueError):
        simulate_slab_coherent(SLAB, wavelength_nm=550, n_photons=100, n_batches=2, detector_half_width=0)
