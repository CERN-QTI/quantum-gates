""" The Gates classes and instances provide a consistent interface for sampling noisy gates.

At the moment, we support the following gates: X, SX, CR, CNOT, CNOT_inv, SingleQubitGate. Note that Rz gates are
virtual on superconducting devices by IBM.

Attributes:
    standard_gates (Gates): Gates produced with constant pulses, the integrations are based on analytical solutions.
    numerical_gates (Gates): Gates produced with constant pulses, but the integrations are performed numerically.
    noise_free_gates (NoiseFreeGates): Gates in the noise free case, based on solving the equations analytically.
    almost_noise_free_gates (ScaledNoiseGates): Gates in the noise free case, but based on scaling the noise down.
"""

from ._gates.gates import Gates, NoiseFreeGates, ScaledNoiseGates, CustomNoiseGates, CustomNoiseChannelsGates
from ._gates.gates import standard_gates, noise_free_gates
from ._legacy.gates import LegacyGates as legacy_gates
