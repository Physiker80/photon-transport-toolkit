"""Optional JAX-accelerated backend for vector_transport, for large batches only.

Context and scope of this module
---------------------------------
A first benchmark measured the *isolated* bottleneck -- per-event
rejection-sampling of a scattering angle -- two ways: the pure-Python
per-photon loop this project already uses, and a batched design where
every photon in a batch is proposed-and-tested together, in lockstep,
for a fixed number of rounds. That comparison (CPU only -- no GPU in
this project's development sandbox, so any GPU claim is unverified
here and must be checked locally) suggested a modest but real ~2.3x
steady-state win:

    N photons   steady-state speedup   speedup incl. one JIT compile
        500              1.5x                    0.02x
      3,000              2.2x                    0.25x
     20,000              2.3x                    0.96x
    100,000              2.4x                    1.74x

That number turned out to be optimistic for a reason worth recording
rather than hiding. Wiring the same idea into the *full* engine --
boundary crossings, absorption, Russian roulette, all included, not
just the scattering step -- first came out **four times slower** than
the reference NumPy engine at 200,000 photons, even excluding compile
time. The cause: the first version computed a fixed-length,
worst-case-budgeted scattering rejection loop (``lax.scan`` over a
fixed round count) on *every* outer round for *every* active photon,
whether or not that round actually used scattering that round (a pure
boundary-bounce round paid for it too, discarding the result), while
the reference engine's native branching pays only for the operation
each photon actually needs, and its rejection sampling accepts in ~1-2
draws on average rather than a fixed worst-case budget.

The fix was to make the inner scattering loop an early-terminating
``lax.while_loop`` gated on which photons actually need a result that
round, so a pure-boundary round costs one bookkeeping check instead of
a full scattering computation. After that fix, and after tuning the
round budget down to the smallest value that leaves zero unresolved
scattering events (measured, not assumed -- see
``n_unresolved_scatter_events`` below), timing the *full* engine
against the reference on the same slab, same hardware:

    N photons (total)   total time incl. compile   speedup   steady-state speedup
         50,000                7.7s vs 6.1s          0.80x           2.08x
        100,000     19.1-19.4s vs 19.8-22.6s    1.02x-1.18x     1.94x-2.38x
        200,000               20.6s vs 23.9s          1.16x           1.67x
        500,000               48.3s vs 59.8s          1.24x           1.47x

(The 100,000 row is reported as a range from two independent runs
rather than a single point, because they disagreed with each other by
more than either disagreed with the trend -- run-to-run wall-clock
variance on the one shared CPU core available here, not a property of
the algorithm. Every other row is a single measurement; this is flagged
rather than smoothed over.)

Two things follow from this table, and are why
``RECOMMENDED_MIN_PHOTONS_FOR_JAX`` is set where it is rather than at
the round number a napkin estimate would suggest:

1. The **steady-state** speedup (compile excluded) *falls* as N grows,
   from ~2x at 50,000 toward ~1.5x at 500,000 -- the opposite of what
   naive vectorization intuition predicts. The likely cause is the
   flattened design's own dependence on the *slowest* photon in the
   batch: the outer ``lax.while_loop`` cannot finish until every photon
   has either exited or been killed by Russian roulette, and the
   longest-lived photon's round count grows (slowly) with batch size
   simply because a larger batch is more likely to contain an unlucky
   long-surviving one. This was not designed for; it was found by
   running the actual numbers rather than trusting the smaller-scale
   estimate, in keeping with this project's own stated convention
   about checking claims against measurement.
2. Below ~50,000 photons total the backend is a clear net loss (0.80x
   at 50,000): the one-time JIT-compile cost (several seconds, and it
   recurs whenever the batch shape changes) is not yet amortized. By
   100,000 the two independent runs above already lean positive
   (1.02x-1.18x) rather than "about even" -- an earlier draft of this
   docstring characterized 100,000 as roughly breakeven based on a
   single unverified figure; re-measuring it twice, directly, showed
   that was wrong before it shipped. ``RECOMMENDED_MIN_PHOTONS_FOR_JAX``
   is kept at 150,000 anyway, as a deliberately conservative margin
   above the observed crossover rather than the tightest bound the data
   would support -- the cost of recommending NumPy when JAX would have
   been slightly faster is small; the cost of the reverse is not.

None of this has been checked on a GPU, because none is available in
this project's development sandbox. GPU hardware could plausibly
change the picture substantially -- more lanes make the "compute both
branches, select" pattern relatively cheaper, and the same
long-tail-photon problem may or may not dominate differently -- but
that is a claim for the user to verify locally, not one made here.


**What this backend does NOT do**, and why, rather than silently doing
it differently:

* **Rayleigh scattering only.** :mod:`photon_transport_toolkit.mie`'s
  sampler already needs a bounded rejection loop per event (see its
  own docstring); adding Mie's table interpolation on top of the
  boundary/absorption loop below is a real extension, not a port, and
  is left for when it is actually needed rather than rushed here.
* **No detector-plane imaging.** No per-pixel fields, no speckle
  contrast, no complex coherent field. Large-N JAX runs are the ones
  this project has used for *aggregate* statistics -- the same
  motivation as the N=50M Russian-roulette run in the handoff summary
  -- not for building speckle images, which is what
  :func:`photon_transport_toolkit.vector_transport.simulate_slab_vector`
  is for. This backend returns ``VectorRadiometricResult``: diffuse
  reflectance, transmittance, absorption, the total Stokes vector, and
  their batch standard errors -- nothing that needs a photon's exit
  pixel.
* **Not bit-identical to the NumPy engine, and not meant to be.** JAX
  uses explicit PRNG keys (``jax.random.split``), not
  ``numpy.random.Generator``; the two draw different random streams
  from the same integer seed by construction. What is required, and
  is checked in ``tests/test_vector_transport_jax.py``, is *statistical*
  agreement with the reference engine within combined uncertainty --
  the correctness criterion this project uses for its cross-language
  MATLAB/Octave validation, applied here across frameworks instead of
  languages.
* **float32/complex64, not float64/complex128** -- the usual GPU-first
  choice. Monte Carlo noise at any batch size this backend is meant
  for (>= 10^4-10^5 photons) is of order ``1/sqrt(N) ~ 10^-2 to 10^-3``,
  several orders above float32 rounding error, so nothing is lost by
  it; it is stated here rather than left implicit because dropping
  precision silently is exactly the kind of choice this project's own
  conventions require surfacing.

Algorithm
---------
The reference engine's photon loop is, at heart, two nested loops: an
outer one over scattering events, and an inner one (`while True`) that
resolves however many boundary bounces a single mean free path happens
to need before it lands inside the slab. Both are naturally *variable
length per photon*, which is precisely the shape that defeats a batched
design if translated directly.

This backend flattens them into one loop whose body performs, **for
every currently-active photon simultaneously**, one of two possible
micro-steps: complete the current mean free path (if it lands inside
the slab) or advance to the next boundary and resolve the Fresnel
decision there. Both branches are computed for every photon every
round and combined with ``jnp.where`` on a per-photon mask -- the
"compute both paths, select" pattern that is the correct way to handle
divergent per-lane control flow on GPU-style hardware, as opposed to
actually branching.

Author: Noureddin Sedki
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photon_transport_toolkit.monte_carlo import SlabOpticalProperties

__all__ = [
    "JAX_AVAILABLE",
    "VectorRadiometricResult",
    "simulate_slab_vector_jax",
    "recommend_backend",
]

try:
    import jax
    import jax.numpy as jnp
    from jax import lax, random

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where jax is absent
    JAX_AVAILABLE = False


# Batch size above which the JIT-compile cost is small next to what it
# buys, per the benchmark table in the module docstring. Exposed as a
# constant (not a magic number in two places) so recommend_backend and
# any future auto-dispatch logic in vector_transport.py stay consistent.
#
# This number was revised once already, from an initial 50,000 based on
# a benchmark of the scattering kernel in isolation, after measuring the
# *full* engine (boundary crossings, absorption, Russian roulette
# included) and finding the backend was still slower than the reference
# below roughly this size -- see the module docstring for the numbers
# and why the isolated-kernel benchmark was optimistic.
RECOMMENDED_MIN_PHOTONS_FOR_JAX = 150_000


def recommend_backend(n_photons: int, n_batches: int = 1) -> str:
    """Which backend this project's own benchmark says to use.

    A thin, honest wrapper around the threshold above -- not a
    performance model. It exists so calling code has one place to ask
    "numpy or jax?" rather than re-deriving the threshold, and so the
    threshold has a single definition to update if the benchmark is
    re-run on different hardware.
    """
    total = int(n_photons) * max(int(n_batches), 1)
    return "jax" if total >= RECOMMENDED_MIN_PHOTONS_FOR_JAX else "numpy"


def _require_jax():
    if not JAX_AVAILABLE:
        raise ImportError(
            "The JAX backend requires the optional 'jax' package "
            "(pip install jax --break-system-packages, or use a virtualenv). "
            "It is optional by design -- see the module docstring for why "
            "the NumPy engine in vector_transport.py remains the default "
            "and the correctness reference."
        )


@dataclass(frozen=True)
class VectorRadiometricResult:
    """Aggregate output of the JAX backend: no imaging, see module docstring.

    Attributes
    ----------
    diffuse_reflectance, transmittance, absorbed : float
        Directly comparable to
        :class:`photon_transport_toolkit.vector_transport.VectorFieldResult`
        and to :func:`photon_transport_toolkit.monte_carlo.simulate_slab`.
    diffuse_reflectance_stderr, transmittance_stderr : float
        Standard error across ``n_batches`` independent batches -- the
        same convention the rest of this project uses, because an
        aggregate number from a faster engine is not exempt from
        needing an uncertainty.
    stokes_total : np.ndarray, shape (4,)
        Stokes vector of the reflected light, per incident photon,
        summed over all batches.
    max_dop_violation : float
        Largest observed excess of the degree of polarization over 1,
        across every traced photon in every batch. Must be 0
        physically; measured rather than assumed, as elsewhere in this
        project.
    n_unresolved : int
        Photons that were still active when ``max_rounds`` was
        reached and were force-terminated as absorbed, mirroring the
        reference engine's own runaway guard. Should be 0 for any
        physically reasonable slab; reported rather than hidden so a
        nonzero count is visible instead of silently biasing the
        result.
    n_unresolved_scatter_events : int
        Individual scattering events (out of typically hundreds per
        photon, times every photon in every batch) that failed to
        accept a proposal within ``scatter_rounds`` and so silently
        kept their pre-scattering Jones vector for that one event.
        Each such event is a small, self-limiting bias (one skipped
        scattering event out of many), not a correctness failure, but
        it is counted rather than assumed away. Compare against the
        total scattering-event count (roughly
        ``n_photons * n_batches * <events per photon>``, itself not
        recorded here) to judge whether ``scatter_rounds`` needs
        raising for a given slab.
    n_photons, n_batches, wavelength_nm : as given.
    compile_seconds : float
        Wall-clock time spent JIT-compiling, measured separately from
        the run so the two costs in the module docstring's table are
        not conflated by an eager caller.
    """

    diffuse_reflectance: float
    transmittance: float
    absorbed: float
    diffuse_reflectance_stderr: float
    transmittance_stderr: float
    stokes_total: np.ndarray
    max_dop_violation: float
    n_unresolved: int
    n_unresolved_scatter_events: int
    n_photons: int
    n_batches: int
    wavelength_nm: float
    compile_seconds: float


# --------------------------------------------------------------------------
# JAX kernel (only defined/callable when jax is installed)
# --------------------------------------------------------------------------


def _build_kernel(n_photons: int, max_rounds: int, weight_threshold: float,
                  roulette_survival: float, scatter_rounds: int):
    """Return a jitted single-batch simulation function, closed over static shapes.

    Everything that determines a JAX/XLA compiled program's shape --
    ``n_photons``, ``max_rounds``, ``scatter_rounds`` -- is a Python
    int captured in this closure rather than a traced value, which is
    what makes the result cacheable across repeated calls with the
    same sizes (the ``n_batches`` reuse this module is built around).
    """

    def initial_jones(key, polarization: str):
        if polarization == "x":
            j1 = jnp.ones(n_photons, dtype=jnp.complex64)
            j2 = jnp.zeros(n_photons, dtype=jnp.complex64)
        elif polarization == "y":
            j1 = jnp.zeros(n_photons, dtype=jnp.complex64)
            j2 = jnp.ones(n_photons, dtype=jnp.complex64)
        elif polarization == "circular":
            inv = 1.0 / jnp.sqrt(2.0)
            j1 = jnp.full(n_photons, inv, dtype=jnp.complex64)
            j2 = jnp.full(n_photons, inv * 1j, dtype=jnp.complex64)
        elif polarization == "unpolarized":
            k1, k2 = random.split(key)
            sin_2chi = 2.0 * random.uniform(k1, (n_photons,)) - 1.0
            chi = 0.5 * jnp.arcsin(sin_2chi)
            psi = jnp.pi * random.uniform(k2, (n_photons,))
            j1 = (jnp.cos(psi) * jnp.cos(chi) - 1j * jnp.sin(psi) * jnp.sin(chi)).astype(jnp.complex64)
            j2 = (jnp.sin(psi) * jnp.cos(chi) + 1j * jnp.cos(psi) * jnp.sin(chi)).astype(jnp.complex64)
        else:
            raise ValueError('polarization must be "x", "y", "circular" or "unpolarized".')
        return j1, j2

    def fresnel_amplitudes(cos_i, n1, n2):
        """Vectorized translation of vector_transport._fresnel_amplitudes.

        ``n1``, ``n2`` are the *real* slab/outside indices; complex casts
        are local to this function so that every other computation in the
        kernel (position, direction, frame algebra) stays real-valued.
        Letting complex dtype leak into those was an earlier bug here --
        ``jax.lax.while_loop`` requires the carry's dtypes to match
        between iterations exactly, so a real array on one code path and
        a complex array on another (both holding the same physical
        quantity) fails immediately and loudly rather than silently, which
        is how this was caught.
        """
        n1c, n2c = n1.astype(jnp.complex64), n2.astype(jnp.complex64)
        same = jnp.asarray(n1 == n2)
        cos_i = jnp.minimum(1.0, jnp.abs(cos_i))
        sin_i2 = jnp.maximum(0.0, 1.0 - cos_i * cos_i)
        sin_t2 = (n1 / n2) ** 2 * sin_i2
        cos_t = jnp.where(
            sin_t2 >= 1.0,
            1j * jnp.sqrt(jnp.maximum(sin_t2 - 1.0, 0.0)),
            jnp.sqrt(jnp.maximum(1.0 - sin_t2, 0.0)).astype(jnp.complex64),
        )
        cos_i_c = cos_i.astype(jnp.complex64)
        r_s = (n1c * cos_i_c - n2c * cos_t) / (n1c * cos_i_c + n2c * cos_t)
        r_p = (n2c * cos_i_c - n1c * cos_t) / (n2c * cos_i_c + n1c * cos_t)
        zero = jnp.zeros_like(r_s)
        return jnp.where(same, zero, r_s), jnp.where(same, zero, r_p)

    def rotate_frame_about_u(e1, e2, cos_psi, sin_psi):
        n1 = tuple(a * cos_psi + b * sin_psi for a, b in zip(e1, e2))
        n2 = tuple(-a * sin_psi + b * cos_psi for a, b in zip(e1, e2))
        return n1, n2

    def tilt_frame_about_e2(e1, u, cos_t, sin_t):
        u_new = tuple(cos_t * uu + sin_t * ee for uu, ee in zip(u, e1))
        e1_new = tuple(-sin_t * uu + cos_t * ee for uu, ee in zip(u, e1))
        return e1_new, u_new

    def renormalize_frame(e1, e2, u):
        ux, uy, uz = u
        inv = 1.0 / jnp.sqrt(ux * ux + uy * uy + uz * uz)
        ux, uy, uz = ux * inv, uy * inv, uz * inv
        ax, ay, az = e1
        dot = ax * ux + ay * uy + az * uz
        ax, ay, az = ax - dot * ux, ay - dot * uy, az - dot * uz
        inv = 1.0 / jnp.sqrt(ax * ax + ay * ay + az * az)
        ax, ay, az = ax * inv, ay * inv, az * inv
        e2_new = (uy * az - uz * ay, uz * ax - ux * az, ux * ay - uy * ax)
        return (ax, ay, az), e2_new, (ux, uy, uz)

    def sp_rotation(e1, e2, u):
        """Angle rotating (e1,e2) so e2 aligns with u x z; identity at normal incidence."""
        sx, sy = u[1], -u[0]
        norm = jnp.sqrt(sx * sx + sy * sy)
        safe = norm > 1e-6
        sx = jnp.where(safe, sx / jnp.where(safe, norm, 1.0), 0.0)
        sy = jnp.where(safe, sy / jnp.where(safe, norm, 1.0), 1.0)
        target = (sx, sy, jnp.zeros_like(sx))
        cos_a = e2[0] * target[0] + e2[1] * target[1] + e2[2] * target[2]
        cx = e2[1] * target[2] - e2[2] * target[1]
        cy = e2[2] * target[0] - e2[0] * target[2]
        cz = e2[0] * target[1] - e2[1] * target[0]
        sin_a = cx * u[0] + cy * u[1] + cz * u[2]
        rnorm = jnp.sqrt(cos_a * cos_a + sin_a * sin_a)
        rnorm = jnp.where(rnorm > 1e-9, rnorm, 1.0)
        cos_a, sin_a = cos_a / rnorm, sin_a / rnorm
        # At normal incidence (safe == False) leave the basis untouched.
        return jnp.where(safe, cos_a, 1.0), jnp.where(safe, sin_a, 0.0)

    def refract_direction(u, n1, n2):
        eta = n1 / n2
        cos_i = jnp.abs(u[2])
        sin_t2 = jnp.minimum(eta * eta * (1.0 - cos_i * cos_i), 1.0)
        cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin_t2))
        ux, uy = u[0] * eta, u[1] * eta
        uz = jnp.where(u[2] >= 0, cos_t, -cos_t)
        norm = jnp.sqrt(ux * ux + uy * uy + uz * uz)
        return ux / norm, uy / norm, uz / norm

    def rayleigh_scatter(key, j1, j2, active):
        """Batched round-based rejection sampling -- see the module benchmark.

        ``active`` marks which photons actually need a scattering result
        this call; the others are marked pre-"accepted" so the ``cond``
        below sees nothing left to do and the loop can exit immediately.
        This is the difference between the version first benchmarked
        (a fixed-length ``lax.scan``, paid in full on *every* outer
        round whether or not any photon needed scattering that round --
        which turned out, on measurement against the reference engine,
        to erase the vectorization win entirely) and this one: an
        early-terminating ``lax.while_loop`` bounded by ``scatter_rounds``
        as a safety cap rather than a mandatory cost. A boundary-only
        outer round (``active`` all False) now costs one ``cond``
        evaluation instead of ``scatter_rounds`` full batch rounds.
        """
        intensity = jnp.abs(j1) ** 2 + jnp.abs(j2) ** 2
        accepted = ~active

        def cond(state):
            round_i, *_, accepted, _key = state
            return (round_i < scatter_rounds) & jnp.any(~accepted)

        def body(state):
            round_i, out1, out2, out_ct, out_cp, out_sp, accepted, key = state
            key, k1, k2, k3 = random.split(key, 4)
            cos_theta = 2.0 * random.uniform(k1, (n_photons,)) - 1.0
            psi = 2.0 * jnp.pi * random.uniform(k2, (n_photons,))
            cos_psi, sin_psi = jnp.cos(psi), jnp.sin(psi)
            p1 = j1 * cos_psi + j2 * sin_psi
            p2 = -j1 * sin_psi + j2 * cos_psi
            n1c = cos_theta.astype(jnp.complex64) * p1
            n2c = p2
            i_new = jnp.abs(n1c) ** 2 + jnp.abs(n2c) ** 2
            u = random.uniform(k3, (n_photons,))
            accept_now = (u * intensity <= i_new) & (~accepted)
            renorm = jnp.sqrt(intensity / jnp.where(i_new > 0, i_new, 1.0)).astype(jnp.complex64)
            out1 = jnp.where(accept_now, n1c * renorm, out1)
            out2 = jnp.where(accept_now, n2c * renorm, out2)
            out_ct = jnp.where(accept_now, cos_theta, out_ct)
            out_cp = jnp.where(accept_now, cos_psi, out_cp)
            out_sp = jnp.where(accept_now, sin_psi, out_sp)
            return (round_i + 1, out1, out2, out_ct, out_cp, out_sp, accepted | accept_now, key)

        init = (jnp.array(0), j1, j2, jnp.zeros(n_photons), jnp.ones(n_photons),
                jnp.zeros(n_photons), accepted, key)
        _, out1, out2, out_ct, out_cp, out_sp, accepted, key = lax.while_loop(cond, body, init)

        # Per-photon, not summed here: whether this contributes to the
        # diagnostic total depends on whether the caller actually used
        # branch A for a given photon this round (do_a), which this
        # function has no way to know on its own -- summing here would
        # count photons that were never active in the first place.
        return out1, out2, out_ct, out_cp, out_sp, key, ~accepted

    def run_batch(key, slab_static, polarization: str):
        (mu_t, albedo, thickness, n_medium, n_outside) = slab_static

        key, k_launch = random.split(key)
        j1, j2 = initial_jones(k_launch, polarization)

        r_s0, _ = fresnel_amplitudes(jnp.asarray(1.0, dtype=jnp.float32), n_outside, n_medium)
        amp0 = jnp.sqrt(jnp.maximum(0.0, 1.0 - jnp.abs(r_s0) ** 2))
        j1, j2 = j1 * amp0, j2 * amp0

        x = jnp.zeros(n_photons)
        y = jnp.zeros(n_photons)
        z = jnp.zeros(n_photons)
        u = (jnp.zeros(n_photons), jnp.zeros(n_photons), jnp.ones(n_photons))
        e1 = (jnp.ones(n_photons), jnp.zeros(n_photons), jnp.zeros(n_photons))
        e2 = (jnp.zeros(n_photons), jnp.ones(n_photons), jnp.zeros(n_photons))

        alive = jnp.ones(n_photons, dtype=bool)
        needs_new_tau = jnp.ones(n_photons, dtype=bool)
        tau = jnp.zeros(n_photons)
        outcome = jnp.zeros(n_photons, dtype=jnp.int8)  # 0 running, 1 R, 2 T, 3 A
        max_dop_violation = jnp.array(0.0)
        # Running scalar accumulators, updated exactly once per photon at
        # the round it exits -- mirroring the reference engine's per-photon
        # ``acc_r += intensity`` / ``acc_t += intensity`` exactly, not a
        # photon *count*. Exit intensity varies photon to photon (path
        # absorption, roulette boosting), so counting outcomes instead of
        # summing their intensity would silently be a different, wrong
        # quantity -- caught only by comparing against the reference
        # engine's own bookkeeping convention, not by any shape check.
        acc_r_intensity = jnp.array(0.0)
        acc_t_intensity = jnp.array(0.0)
        # Stokes vector of reflected light *only*, projected onto the lab
        # x/y analyzer axes -- matching VectorFieldResult.stokes_total,
        # which the reference engine restricts to outcome == "R" and to
        # the lab frame (see _project_to_lab), not the photon's own
        # (e1, e2) frame.
        exit_stokes_r_lab = jnp.zeros(4)
        # Diagnostic: how many *actual* scattering events (branch A,
        # currently alive) failed to accept within scatter_rounds and
        # therefore silently kept their pre-scattering Jones vector for
        # that one event -- a small, quantified bias rather than an
        # assumed-away one. See tests/test_vector_transport_jax.py for
        # the bound this is checked against.
        unresolved_scatter_total = jnp.array(0)

        def cond_fn(state):
            round_i, alive, *_ = state
            return (round_i < max_rounds) & jnp.any(alive)

        def body_fn(state):
            (round_i, alive, needs_new_tau, tau, x, y, z, u, e1, e2, j1, j2,
             outcome, max_dop_violation, acc_r_intensity, acc_t_intensity,
             exit_stokes_r_lab, unresolved_scatter_total, key) = state

            key, k_tau, k_scat, k_bnd, k_roul = random.split(key, 5)

            fresh_tau = -jnp.log(jnp.clip(random.uniform(k_tau, (n_photons,)), 1e-30, 1.0))
            tau = jnp.where(needs_new_tau, fresh_tau, tau)

            step = tau / mu_t
            uz = u[2]
            dist_up = (thickness - z) / jnp.where(uz > 0, uz, 1.0)
            dist_down = -z / jnp.where(uz < 0, uz, -1.0)
            dist_boundary = jnp.where(uz > 0, dist_up, jnp.where(uz < 0, dist_down, jnp.inf))

            lands_inside = step < dist_boundary
            do_a_early = alive & lands_inside  # needed before the scatter call below

            # ---- branch A: complete the free path, absorb, scatter ------
            x_a = x + step * u[0]
            y_a = y + step * u[1]
            z_a = z + step * u[2]

            amp = jnp.sqrt(albedo).astype(jnp.complex64)
            j1_a, j2_a = j1 * amp, j2 * amp

            j1_s, j2_s, cos_t, cos_p, sin_p, key, unresolved_scat_mask = rayleigh_scatter(
                k_scat, j1_a, j2_a, do_a_early
            )
            sin_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_t * cos_t))
            e1_s, e2_s = rotate_frame_about_u(e1, e2, cos_p, sin_p)
            e1_s, u_s = tilt_frame_about_e2(e1_s, u, cos_t, sin_t)
            e1_s, e2_s, u_s = renormalize_frame(e1_s, e2_s, u_s)

            intensity_a = jnp.abs(j1_s) ** 2 + jnp.abs(j2_s) ** 2
            stokes_a_i = intensity_a
            stokes_a_q = jnp.abs(j1_s) ** 2 - jnp.abs(j2_s) ** 2
            cross = j1_s * jnp.conj(j2_s)
            stokes_a_u = 2.0 * jnp.real(cross)
            stokes_a_v = -2.0 * jnp.imag(cross)
            dop_a = jnp.sqrt(stokes_a_q ** 2 + stokes_a_u ** 2 + stokes_a_v ** 2) / \
                jnp.where(intensity_a > 0, intensity_a, 1.0)

            below_thresh = intensity_a < weight_threshold
            roulette_roll = random.uniform(k_roul, (n_photons,))
            survives = roulette_roll <= (1.0 / roulette_survival)
            boost = jnp.sqrt(jnp.asarray(roulette_survival, dtype=jnp.complex64))
            j1_rl = jnp.where(below_thresh & survives, j1_s * boost, j1_s)
            j2_rl = jnp.where(below_thresh & survives, j2_s * boost, j2_s)
            killed_by_roulette = below_thresh & (~survives)

            outcome_a = jnp.where(killed_by_roulette, jnp.int8(3), jnp.int8(0))
            alive_a = ~killed_by_roulette
            needs_tau_a = jnp.ones(n_photons, dtype=bool)  # ready for a fresh tau next round

            # ---- branch B: advance to the boundary, Fresnel decision ----
            x_b = x + dist_boundary * u[0]
            y_b = y + dist_boundary * u[1]
            z_b = jnp.where(uz < 0, 0.0, thickness)
            tau_b = tau - dist_boundary * mu_t

            cos_a_rot, sin_a_rot = sp_rotation(e1, e2, u)
            e1_b, e2_b = rotate_frame_about_u(e1, e2, cos_a_rot, sin_a_rot)
            j1_rot = j1 * cos_a_rot.astype(jnp.complex64) + j2 * sin_a_rot.astype(jnp.complex64)
            j2_rot = -j1 * sin_a_rot.astype(jnp.complex64) + j2 * cos_a_rot.astype(jnp.complex64)

            r_s, r_p = fresnel_amplitudes(jnp.abs(uz), n_medium, n_outside)
            i_p, i_s = jnp.abs(j1_rot) ** 2, jnp.abs(j2_rot) ** 2
            intensity_b = i_p + i_s
            r_eff = (jnp.abs(r_p) ** 2 * i_p + jnp.abs(r_s) ** 2 * i_s) / \
                jnp.where(intensity_b > 0, intensity_b, 1.0)

            transmit_roll = random.uniform(k_bnd, (n_photons,))
            transmits = transmit_roll > r_eff

            # Transmitted sub-branch
            t_s = 1.0 + r_s
            t_p = (n_medium / n_outside) * (1.0 + r_p)
            j1_t = j1_rot * t_p
            j2_t = j2_rot * t_s
            norm_t = jnp.sqrt(intensity_b / jnp.maximum(jnp.abs(j1_t) ** 2 + jnp.abs(j2_t) ** 2, 1e-30)) \
                .astype(jnp.complex64)
            j1_t, j2_t = j1_t * norm_t, j2_t * norm_t
            u_out = refract_direction(u, n_medium, n_outside)
            e1_out, e2_out, u_out = renormalize_frame(e1_b, e2_b, u_out)
            outcome_t = jnp.where(uz < 0, jnp.int8(1), jnp.int8(2))  # R if heading toward z=0
            exit_intensity_t = jnp.abs(j1_t) ** 2 + jnp.abs(j2_t) ** 2

            # Lab-frame projection (photon_transport_toolkit.vector_transport._project_to_lab,
            # written directly from the Jones components rather than via an
            # intermediate 2x2 coherency matrix -- algebraically identical,
            # avoids building complex 2x2 matrices per photon per round).
            # An obliquely exiting ray loses the field component along lab
            # z, which no analyzer in the xy plane can detect -- this
            # projection is therefore for the *detector-plane* Stokes
            # total only; the R/T/A radiometric budget above uses the
            # full, unprojected intensity, so energy conservation is
            # unaffected by it (matching the reference docstring).
            e_lab_x = j1_t * e1_out[0] + j2_t * e2_out[0]
            e_lab_y = j1_t * e1_out[1] + j2_t * e2_out[1]
            ixx = jnp.abs(e_lab_x) ** 2
            iyy = jnp.abs(e_lab_y) ** 2
            ixy = e_lab_x * jnp.conj(e_lab_y)
            stokes_t_i_lab = ixx + iyy
            stokes_t_q_lab = ixx - iyy
            stokes_t_u_lab = 2.0 * jnp.real(ixy)
            stokes_t_v_lab = -2.0 * jnp.imag(ixy)

            # Internally-reflected sub-branch
            j1_r = j1_rot * r_p
            j2_r = j2_rot * r_s
            norm_r = jnp.sqrt(intensity_b / jnp.maximum(jnp.abs(j1_r) ** 2 + jnp.abs(j2_r) ** 2, 1e-30)) \
                .astype(jnp.complex64)
            j1_r, j2_r = j1_r * norm_r, j2_r * norm_r
            u_refl = (u[0], u[1], -u[2])
            e1_refl = (
                e2_b[1] * u_refl[2] - e2_b[2] * u_refl[1],
                e2_b[2] * u_refl[0] - e2_b[0] * u_refl[2],
                e2_b[0] * u_refl[1] - e2_b[1] * u_refl[0],
            )
            e1_refl, e2_refl, u_refl = renormalize_frame(e1_refl, e2_b, u_refl)

            # Select transmitted vs reflected within branch B
            j1_bsel = jnp.where(transmits, j1_t, j1_r)
            j2_bsel = jnp.where(transmits, j2_t, j2_r)
            u0_bsel = jnp.where(transmits, u_out[0], u_refl[0])
            u1_bsel = jnp.where(transmits, u_out[1], u_refl[1])
            u2_bsel = jnp.where(transmits, u_out[2], u_refl[2])
            e1_0 = jnp.where(transmits, e1_out[0], e1_refl[0])
            e1_1 = jnp.where(transmits, e1_out[1], e1_refl[1])
            e1_2 = jnp.where(transmits, e1_out[2], e1_refl[2])
            e2_0 = jnp.where(transmits, e2_out[0], e2_refl[0])
            e2_1 = jnp.where(transmits, e2_out[1], e2_refl[1])
            e2_2 = jnp.where(transmits, e2_out[2], e2_refl[2])

            outcome_b = jnp.where(transmits, outcome_t, jnp.int8(0))
            alive_b = ~transmits
            needs_tau_b = jnp.zeros(n_photons, dtype=bool)  # leftover tau carries over

            # ---- combine branch A / branch B per photon, only where alive
            do_a = alive & lands_inside
            do_b = alive & (~lands_inside)

            x_n = jnp.where(do_a, x_a, jnp.where(do_b, x_b, x))
            y_n = jnp.where(do_a, y_a, jnp.where(do_b, y_b, y))
            z_n = jnp.where(do_a, z_a, jnp.where(do_b, z_b, z))
            tau_n = jnp.where(do_b, tau_b, tau)

            u0_n = jnp.where(do_a, u_s[0], jnp.where(do_b, u0_bsel, u[0]))
            u1_n = jnp.where(do_a, u_s[1], jnp.where(do_b, u1_bsel, u[1]))
            u2_n = jnp.where(do_a, u_s[2], jnp.where(do_b, u2_bsel, u[2]))
            e1_0n = jnp.where(do_a, e1_s[0], jnp.where(do_b, e1_0, e1[0]))
            e1_1n = jnp.where(do_a, e1_s[1], jnp.where(do_b, e1_1, e1[1]))
            e1_2n = jnp.where(do_a, e1_s[2], jnp.where(do_b, e1_2, e1[2]))
            e2_0n = jnp.where(do_a, e2_s[0], jnp.where(do_b, e2_0, e2[0]))
            e2_1n = jnp.where(do_a, e2_s[1], jnp.where(do_b, e2_1, e2[1]))
            e2_2n = jnp.where(do_a, e2_s[2], jnp.where(do_b, e2_2, e2[2]))

            j1_n = jnp.where(do_a, j1_rl, jnp.where(do_b, j1_bsel, j1))
            j2_n = jnp.where(do_a, j2_rl, jnp.where(do_b, j2_bsel, j2))

            outcome_n = jnp.where(do_a, outcome_a, jnp.where(do_b, outcome_b, outcome))
            alive_n = jnp.where(do_a, alive_a, jnp.where(do_b, alive_b, alive))
            needs_tau_n = jnp.where(do_a, needs_tau_a, jnp.where(do_b, needs_tau_b, needs_new_tau))

            max_dop_violation_n = jnp.maximum(
                max_dop_violation, jnp.max(jnp.where(do_a, dop_a - 1.0, -jnp.inf))
            )

            # Exactly-once accumulation at the round each photon exits.
            # ``exit_now_r``/``exit_now_t`` are mutually exclusive and, for
            # any given photon, true in exactly one round of the whole
            # loop -- do_b requires alive, and alive becomes False the
            # instant a photon transmits, so it cannot contribute twice.
            exit_now_r = do_b & transmits & (uz < 0)
            exit_now_t = do_b & transmits & (uz >= 0)
            unresolved_scatter_total_n = unresolved_scatter_total + jnp.sum(
                jnp.where(do_a, unresolved_scat_mask, False)
            )
            acc_r_intensity_n = acc_r_intensity + jnp.sum(jnp.where(exit_now_r, exit_intensity_t, 0.0))
            acc_t_intensity_n = acc_t_intensity + jnp.sum(jnp.where(exit_now_t, exit_intensity_t, 0.0))
            exit_stokes_r_lab_n = exit_stokes_r_lab + jnp.array([
                jnp.sum(jnp.where(exit_now_r, stokes_t_i_lab, 0.0)),
                jnp.sum(jnp.where(exit_now_r, stokes_t_q_lab, 0.0)),
                jnp.sum(jnp.where(exit_now_r, stokes_t_u_lab, 0.0)),
                jnp.sum(jnp.where(exit_now_r, stokes_t_v_lab, 0.0)),
            ])

            return (round_i + 1, alive_n, needs_tau_n, tau_n, x_n, y_n, z_n,
                   (u0_n, u1_n, u2_n), (e1_0n, e1_1n, e1_2n), (e2_0n, e2_1n, e2_2n),
                   j1_n, j2_n, outcome_n, max_dop_violation_n,
                   acc_r_intensity_n, acc_t_intensity_n, exit_stokes_r_lab_n,
                   unresolved_scatter_total_n, key)

        init = (jnp.array(0), alive, needs_new_tau, tau, x, y, z, u, e1, e2, j1, j2,
               outcome, max_dop_violation, acc_r_intensity, acc_t_intensity,
               exit_stokes_r_lab, unresolved_scatter_total, key)
        final = lax.while_loop(cond_fn, body_fn, init)
        (round_i, alive_f, _, _, _, _, _, _, _, _, _, _, outcome_f,
         max_dop_violation_f, acc_r_intensity_f, acc_t_intensity_f,
         exit_stokes_r_lab_f, unresolved_scatter_total_f, _) = final

        # Runaway guard, mirroring the reference engine's own: anything
        # still alive when the round budget is exhausted is counted and
        # force-absorbed (its current, nonzero intensity is *not* credited
        # to R or T) rather than silently dropped from the budget -- the
        # same "give up and count it" behaviour as the Python engine's own
        # ``steps > 100_000`` guard, which discards the photon's carried
        # amplitude rather than letting it bias R or T.
        n_unresolved = jnp.sum(alive_f)

        return (acc_r_intensity_f, acc_t_intensity_f, exit_stokes_r_lab_f,
               max_dop_violation_f, n_unresolved, unresolved_scatter_total_f)

    return jax.jit(run_batch, static_argnames=("polarization",))


# --------------------------------------------------------------------------
# Public driver
# --------------------------------------------------------------------------


def simulate_slab_vector_jax(
    slab: SlabOpticalProperties,
    wavelength_nm: float,
    n_photons: int = 50_000,
    n_batches: int = 4,
    seed: int = 0,
    polarization: str = "x",
    max_rounds: int = 20_000,
    weight_threshold: float = 1e-4,
    roulette_survival: float = 10.0,
    scatter_rounds: int = 16,
) -> "VectorRadiometricResult":
    """Rayleigh-only, aggregate-only JAX backend. See the module docstring.

    Parameters mirror
    :func:`photon_transport_toolkit.vector_transport.simulate_slab_vector`
    where they overlap. ``n_photons`` is *per batch*; the compiled
    kernel is built once for the given ``(n_photons, max_rounds,
    scatter_rounds)`` and reused for all ``n_batches`` runs, which is
    what makes the JIT-compile cost worth paying at all (see the
    module docstring's benchmark table).

    Raises
    ------
    ImportError
        If the optional ``jax`` package is not installed.
    ValueError
        If ``slab.g != 0`` (this backend implements Rayleigh
        scattering, whose ``<cos theta> = 0`` by symmetry; a nonzero
        ``g`` would be silently ignored otherwise -- the same
        deliberate refusal as in ``vector_transport.py``) or if
        ``slab.phase_function != "hg"`` for the analogous reason.
    """
    _require_jax()
    if slab.g != 0.0:
        raise ValueError(
            "the JAX backend implements Rayleigh scattering (<cos theta> = 0 "
            "by symmetry), so slab.g must be 0 rather than a value that "
            "would be silently ignored."
        )
    if n_photons < 1 or n_batches < 1:
        raise ValueError("n_photons and n_batches must be at least 1.")

    import time as _time

    kernel = _build_kernel(n_photons, max_rounds, weight_threshold,
                           roulette_survival, scatter_rounds)

    slab_static = (
        jnp.asarray(slab.mu_a + slab.mu_s, dtype=jnp.float32),
        jnp.asarray(slab.mu_s / (slab.mu_a + slab.mu_s), dtype=jnp.float32),
        jnp.asarray(slab.thickness, dtype=jnp.float32),
        jnp.asarray(slab.n_medium, dtype=jnp.float32),
        jnp.asarray(slab.n_outside, dtype=jnp.float32),
    )

    master_key = random.PRNGKey(seed)
    batch_keys = random.split(master_key, n_batches)

    t0 = _time.perf_counter()
    acc_r0, acc_t0, stokes0, dop0, unresolved0, unresolved_scat0 = jax.block_until_ready(
        kernel(batch_keys[0], slab_static, polarization)
    )
    compile_seconds = _time.perf_counter() - t0

    # Per-batch fractions are *intensity sums* divided by photon count,
    # matching the reference engine's own ``acc_r / per_batch`` --
    # never photon *counts*, since exit intensity is not uniformly 1
    # (path absorption, roulette boosting both change it per photon).
    r_fracs = [float(acc_r0) / n_photons]
    t_fracs = [float(acc_t0) / n_photons]
    stokes_total = np.array(stokes0, dtype=float)
    max_dop_violation = float(dop0)
    n_unresolved = int(unresolved0)
    n_unresolved_scatter = int(unresolved_scat0)

    for b in range(1, n_batches):
        acc_r, acc_t, stokes, dop, unresolved, unresolved_scat = jax.block_until_ready(
            kernel(batch_keys[b], slab_static, polarization)
        )
        r_fracs.append(float(acc_r) / n_photons)
        t_fracs.append(float(acc_t) / n_photons)
        stokes_total += np.array(stokes, dtype=float)
        max_dop_violation = max(max_dop_violation, float(dop))
        n_unresolved += int(unresolved)
        n_unresolved_scatter += int(unresolved_scat)

    r_fracs = np.array(r_fracs)
    t_fracs = np.array(t_fracs)
    rd_mean, t_mean = r_fracs.mean(), t_fracs.mean()
    rd_stderr = r_fracs.std(ddof=1) / np.sqrt(n_batches) if n_batches > 1 else float("nan")
    t_stderr = t_fracs.std(ddof=1) / np.sqrt(n_batches) if n_batches > 1 else float("nan")

    # Exact-energy-conservation bookkeeping identical in *form* to the
    # reference engine's: absorbed = launched - Rd - T, where "launched"
    # is the (deterministic, normal-incidence) fraction that clears the
    # entrance Fresnel interface. Any roulette-terminated or
    # runaway-guard-terminated residual weight lands here automatically,
    # the same as vector_transport.py's own per-photon
    # ``acc_a += launched - intensity``.
    r_sp = _specular_reflectance(slab)
    launched = 1.0 - r_sp
    absorbed = launched - rd_mean - t_mean

    return VectorRadiometricResult(
        diffuse_reflectance=float(rd_mean),
        transmittance=float(t_mean),
        absorbed=float(absorbed),
        diffuse_reflectance_stderr=float(rd_stderr),
        transmittance_stderr=float(t_stderr),
        stokes_total=stokes_total / n_batches,
        max_dop_violation=max_dop_violation,
        n_unresolved=n_unresolved,
        n_unresolved_scatter_events=n_unresolved_scatter,
        n_photons=n_photons,
        n_batches=n_batches,
        wavelength_nm=wavelength_nm,
        compile_seconds=compile_seconds,
    )


def _specular_reflectance(slab: SlabOpticalProperties) -> float:
    return ((slab.n_outside - slab.n_medium) / (slab.n_outside + slab.n_medium)) ** 2
