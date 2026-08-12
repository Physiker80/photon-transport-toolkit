"""
Quantitative validation report for the Monte Carlo transport model.

Prints a table comparing simulated transmittance against the analytical
Beer-Lambert result in the non-scattering limit, together with the Monte Carlo
standard error, so that agreement can be judged against the model's own
quoted uncertainty rather than by eye.

Usage:  python validation/validation_report.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from photon_transport_toolkit import SlabOpticalProperties, simulate_slab  # noqa: E402

N_PHOTONS = 20_000


def beer_lambert_validation() -> None:
    """Non-scattering slab: T must equal exp(-mu_a * L)."""
    print("\nBeer-Lambert limit (mu_s = 0, index matched)")
    print("-" * 78)
    print(f"{'mu_a [1/mm]':>12} {'L [mm]':>8} {'T (MC)':>12} {'T (exact)':>12} "
          f"{'deviation':>12} {'in 3 sigma':>11}")
    print("-" * 78)

    for mu_a, thickness in [(0.1, 2.0), (0.3, 3.0), (0.5, 4.0), (1.0, 2.0)]:
        slab = SlabOpticalProperties(
            mu_a=mu_a, mu_s=0.0, g=0.0, thickness=thickness,
            n_medium=1.0, n_outside=1.0,
        )
        res = simulate_slab(slab, n_photons=N_PHOTONS, seed=11)
        exact = np.exp(-mu_a * thickness)
        deviation = res.transmittance - exact
        within = abs(deviation) <= 3.0 * max(res.transmittance_stderr, 1e-12)
        print(f"{mu_a:>12.2f} {thickness:>8.1f} {res.transmittance:>12.5f} "
              f"{exact:>12.5f} {deviation:>+12.2e} {'yes' if within else 'NO':>11}")


def conservation_validation() -> None:
    """Absorbing and scattering slab: R + T + A must equal 1."""
    print("\nEnergy conservation (R + T + A = 1)")
    print("-" * 78)
    print(f"{'mu_s [1/mm]':>12} {'g':>6} {'R_diff':>10} {'T':>10} {'A':>10} "
          f"{'R_spec':>10} {'sum':>12}")
    print("-" * 78)

    for mu_s, g in [(1.0, 0.0), (10.0, 0.5), (50.0, 0.85)]:
        slab = SlabOpticalProperties(mu_a=0.05, mu_s=mu_s, g=g, thickness=3.0)
        res = simulate_slab(slab, n_photons=N_PHOTONS, seed=12)
        print(f"{mu_s:>12.1f} {g:>6.2f} {res.diffuse_reflectance:>10.5f} "
              f"{res.transmittance:>10.5f} {res.absorbed:>10.5f} "
              f"{res.specular_reflectance:>10.5f} {res.energy_balance:>12.9f}")


def convergence_validation() -> None:
    """The standard error must fall as 1/sqrt(N)."""
    print("\nStatistical convergence (stderr should scale as 1/sqrt(N))")
    print("-" * 78)
    print(f"{'N photons':>12} {'T':>12} {'stderr':>12} {'stderr*sqrt(N)':>18}")
    print("-" * 78)

    slab = SlabOpticalProperties(mu_a=0.05, mu_s=10.0, g=0.8, thickness=3.0)
    for n in (4_000, 16_000, 64_000):
        res = simulate_slab(slab, n_photons=n, seed=13)
        print(f"{n:>12d} {res.transmittance:>12.5f} {res.transmittance_stderr:>12.2e} "
              f"{res.transmittance_stderr * np.sqrt(n):>18.3f}")
    print("\nThe last column should be roughly constant if the error falls as")
    print("1/sqrt(N). It scatters by a few tens of percent here, which is")
    print("expected: the standard error is itself estimated from only 10")
    print("batches and therefore carries about 24% relative uncertainty.")
    print("Raising n_batches tightens this diagnostic at the cost of runtime.")


if __name__ == "__main__":
    print("=" * 78)
    print("Monte Carlo transport model - validation report")
    print("=" * 78)
    beer_lambert_validation()
    conservation_validation()
    convergence_validation()
    print()
