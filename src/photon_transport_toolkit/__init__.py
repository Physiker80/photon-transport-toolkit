"""photon_transport_toolkit - Monte Carlo photon transport in scattering and absorbing media.

Follows the MCML formulation (Wang, Jacques & Zheng, 1995), extended to
arbitrarily layered media, phase-resolved (coherent-field) transport,
combined polarization- and phase-resolved (Jones/Stokes) transport with
Rayleigh or Mie scattering, and tissue-specific spectral parameterization.

A geometrically unrelated sibling project,
`grating-spectrometer-model <https://github.com/Physiker80/grating-spectrometer-model>`_,
was originally developed in this same repository and has since been split
out to keep each project's scope focused as this one grew — see this
project's README, section "Related work", for the technical connection
between the two.

Author: Noureddin Sedki
License: MIT
"""

from photon_transport_toolkit.monte_carlo import (
    MonteCarloResult,
    SlabOpticalProperties,
    simulate_slab,
    g_to_schlick_k,
    henyey_greenstein_pdf,
    schlick_pdf,
)
from photon_transport_toolkit.layered_media import (
    Layer,
    LayeredMedium,
    LayeredResult,
    simulate_layered_medium,
)
from photon_transport_toolkit.tissue_optics import (
    melanin_absorption_mm,
    reduced_scattering_power_law,
    epidermis_layer,
    dermis_layer,
)
from photon_transport_toolkit.coherent_transport import (
    CoherentFieldResult,
    simulate_slab_coherent,
)
from photon_transport_toolkit.mie import (
    MieScatterer,
    mie_coefficients,
    mie_efficiencies,
    mie_amplitudes,
    size_parameter,
)
from photon_transport_toolkit.vector_transport_jax import (
    JAX_AVAILABLE,
    VectorRadiometricResult,
    simulate_slab_vector_jax,
    recommend_backend,
)
from photon_transport_toolkit.vector_transport import (
    VectorFieldResult,
    simulate_slab_vector,
    depolarization_ladder,
    rayleigh_mueller_matrix,
    rayleigh_amplitude_matrix,
    jones_to_stokes,
    degree_of_polarization,
    speckle_contrast,
)

__version__ = "1.7.0"

__all__ = [
    "SlabOpticalProperties",
    "MonteCarloResult",
    "simulate_slab",
    "g_to_schlick_k",
    "henyey_greenstein_pdf",
    "schlick_pdf",
    "Layer",
    "LayeredMedium",
    "LayeredResult",
    "simulate_layered_medium",
    "melanin_absorption_mm",
    "reduced_scattering_power_law",
    "epidermis_layer",
    "dermis_layer",
    "CoherentFieldResult",
    "simulate_slab_coherent",
    "VectorFieldResult",
    "simulate_slab_vector",
    "depolarization_ladder",
    "JAX_AVAILABLE",
    "VectorRadiometricResult",
    "simulate_slab_vector_jax",
    "recommend_backend",
    "MieScatterer",
    "mie_coefficients",
    "mie_efficiencies",
    "mie_amplitudes",
    "size_parameter",
    "rayleigh_mueller_matrix",
    "rayleigh_amplitude_matrix",
    "jones_to_stokes",
    "degree_of_polarization",
    "speckle_contrast",
]
