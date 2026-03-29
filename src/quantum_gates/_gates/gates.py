"""
This module contains all the noisy quantum gates functions, but parametrized according to different pulse shapes.

All the time duration are expressed in units of single-qubit gate time tg of IBM's devices.

Attributes:
    standard_gates (Gates): Gates produced with constant pulses, the integrations are based on analytical solutions.
    numerical_gates (Gates): Gates produced with constant pulses, but the integrations are performed numerically.
    noise_free_gates (NoiseFreeGates): Gates in the noise free case, based on solving the equations analytically.
    almost_noise_free_gates (ScaledNoiseGates): Gates in the noise free case, but based on scaling the noise down.
"""

import numpy as np
import warnings

from .pulse import Pulse, constant_pulse, constant_pulse_numerical
from .integrator import Integrator
from .factories import (
    BitflipFactory,
    DepolarizingFactory,
    RelaxationFactory,
    SingleQubitGateFactory,
    XFactory,
    SXFactory,
    CNOTFactory,
    CNOTInvFactory,
    CRFactory,
    ECRFactory,
    ECRInvFactory
)

class Gates:
    """Collection of the gates. Handles adding the pulse shape to the gates.

    This way, we do not have to pass the pulse shape each time we generate a gate.

    Example:
        .. code-block:: python

            from quantum_gates.pulses import GaussianPulse
            from quantum_gates.gates import Gates

            pulse = GaussianPulse(loc=1, scale=1)
            gateset = Gates(pulse)

            sampled_x = gateset.X(phi, p, T1, T2)
    """

    def __init__(self, pulse: Pulse=constant_pulse):
        self.integrator = Integrator(pulse)

        # Factories
        self.bitflip_c = BitflipFactory()
        self.depolarizing_c = DepolarizingFactory()
        self.relaxation_c = RelaxationFactory()
        self.single_qubit_gate_c = SingleQubitGateFactory(self.integrator)
        self.x_c = XFactory(self.integrator)
        self.sx_c = SXFactory(self.integrator)
        self.cr_c = CRFactory(self.integrator)
        self.cnot_c = CNOTFactory(self.integrator)
        self.cnot_inv_c = CNOTInvFactory(self.integrator)
        self.ecr_c = ECRFactory(self.integrator)
        self.ecr_inv_c = ECRInvFactory(self.integrator)

    def relaxation(self, Dt, T1, T2, *, qubit_index=None) -> np.array:
        return self.relaxation_c.construct(Dt, T1, T2)

    def bitflip(self, Dt, p, *, qubit_index=None) -> np.array:
        return self.bitflip_c.construct(Dt, p)

    def depolarizing(self, Dt, p, *, qubit_index=None) -> np.array:
        return self.depolarizing_c.construct(Dt, p)

    def single_qubit_gate(self, theta, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.single_qubit_gate_c.construct(theta, phi, p, T1, T2)

    def X(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.x_c.construct(phi, p, T1, T2)

    def SX(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.sx_c.construct(phi, p, T1, T2)

    def CR(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.cr_c.construct(theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)

    def CNOT(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.cnot_c.construct(phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg)

    def CNOT_inv(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.cnot_inv_c.construct(phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg)
    
    def ECR(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.ecr_c.construct(phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg)
    
    def ECR_inv(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.ecr_inv_c.construct(phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg)


class NoiseFreeGates(object):
    """ Version of Gates for the noiseless case.

    Has the same interface as the Gates class, but ignores the arguments specifying the noise.

    Example:
        .. code:: python

           from quantum_gates.gates import NoiseFreeGates

           gateset = NoiseFreeGates()
           sampled_x = gateset.X(phi, p, T1, T2)
    """

    def relaxation(self, Dt, T1, T2, *, qubit_index=None) -> np.array:
        """ Returns single qubit relaxation in noise free regime -> identity. """
        return np.eye(2)

    def bitflip(self, Dt, p, *, qubit_index=None) -> np.array:
        """ Returns single qubit bitflip in noise free regime -> identity. """
        return np.eye(2)

    def depolarizing(self, Dt, p, *, qubit_index=None) -> np.array:
        """ Returns single qubit depolarizing in noise free regime -> identity. """
        return np.eye(2)

    def single_qubit_gate(self, theta, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        """ Returns general single qubit gate in noise free regime. """
        U = np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
        return U

    def X(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        """ Returns X gate in noise free regime. """
        theta = np.pi
        U = np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
        return U

    def SX(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        """ Returns SX gate in noise free regime. """
        theta = np.pi / 2
        U = np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
        return U

    def CNOT(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        """ Returns CNOT gate in noise free regime. """
        # Constants
        tg = 35*10**(-9)
        t_cr = t_cnot/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)))))

        # Noise free gates
        first_cr = self.CR(-np.pi/4, -phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        second_cr = self.CR(np.pi/4, -phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        x_gate = self.X(-phi_ctr + np.pi / 2, p_single_ctr, T1_ctr, T2_ctr)
        sx_gate = self.SX(-phi_trg, p_single_trg, T1_trg, T2_trg)
        relaxation_gate = self.relaxation(tg, T1_trg, T2_trg)
        Y_Rz = self.single_qubit_gate(-np.pi, -phi_ctr + np.pi/2 + np.pi/2, p_single_ctr, T1_ctr, T2_ctr)

        return first_cr @ np.kron(x_gate, relaxation_gate) @ second_cr @ np.kron(Y_Rz, sx_gate)

    def CNOT_inv(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        """ Returns CNOT inverse gate in noise free regime. """
        # Constants
        tg = 35*10**(-9)
        t_cr = (t_cnot-3*tg)/2
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)**3))))

        # Noise free gates
        Ry = self.single_qubit_gate(-np.pi/2, -phi_trg-np.pi/2+np.pi/2, p_single_trg, T1_trg, T2_trg)
        Y_Z = self.single_qubit_gate(np.pi/2, -phi_ctr-np.pi+np.pi/2, p_single_ctr, T1_ctr, T2_ctr)
        first_sx_gate = self.SX(-phi_ctr - np.pi - np.pi / 2, p_single_ctr, T1_ctr, T2_ctr)
        second_sx_gate = self.SX(-phi_trg - np.pi / 2, p_single_ctr, T1_ctr, T2_ctr)
        first_cr = self.CR(-np.pi/4, -phi_ctr-np.pi, t_cr, p_cr, T1_trg, T2_trg, T1_ctr, T2_ctr)
        second_cr = self.CR(np.pi/4, -phi_ctr-np.pi, t_cr, p_cr, T1_trg, T2_trg, T1_ctr, T2_ctr)
        x_gate = self.X(-phi_trg - np.pi / 2, p_single_trg, T1_trg, T2_trg)
        relaxation_gate = self.relaxation(tg, T1_ctr, T2_ctr)

        result = np.kron(Ry, first_sx_gate) @ first_cr @ np.kron(x_gate, relaxation_gate) @ second_cr @ np.kron(second_sx_gate, Y_Z)
        return result

    def CR(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        """ Returns CR gate in noise free regime. """
        return np.array(
            [[np.cos(theta/2), -1J*np.sin(theta/2) * np.exp(-1J * phi), 0, 0],
             [-1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2), 0, 0],
             [0, 0, np.cos(theta/2), 1J*np.sin(theta/2) * np.exp(-1J * phi)],
             [0, 0, 1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
    
    def ECR(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        """ Returns ECR gate in noise free regime. """
        # Constants
        tg = 35*10**(-9)
        t_cr = t_ecr/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_ecr)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)))))

        # Noise free gates
        first_cr = self.CR(np.pi/4, np.pi-phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        second_cr = self.CR(-np.pi/4, np.pi-phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        x_gate = -1J* self.X(np.pi -phi_ctr , p_single_ctr, T1_ctr, T2_ctr)
        relaxation_gate = self.relaxation(tg, T1_trg, T2_trg)

        return first_cr @ np.kron(x_gate, relaxation_gate) @ second_cr 
    

    def ECR_inv(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        """ Returns ECR inverse gate in noise free regime. """
        # Constants
        tg = 35*10**(-9)
        t_cr = t_ecr/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_ecr)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)))))

        # Sample gates
        first_cr = self.CR(np.pi/4, np.pi-phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        second_cr = self.CR(-np.pi/4, np.pi-phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        x_gate = -1J* self.X(np.pi-phi_ctr, p_single_ctr, T1_ctr, T2_ctr)
        relaxation_gate = self.relaxation(tg, T1_trg, T2_trg)

        sx_gate_ctr_1 =  self.SX(-np.pi/2-phi_ctr, p_single_ctr, T1_ctr, T2_ctr)
        sx_gate_trg_1 =  self.SX(-np.pi/2-phi_trg, p_single_trg, T1_trg, T2_trg)

        sx_gate_ctr_2 =  self.SX(-np.pi/2-phi_ctr, p_single_ctr, T1_ctr, T2_ctr)
        sx_gate_trg_2 =  self.SX(-np.pi/2-phi_trg, p_single_trg, T1_trg, T2_trg)

        return 1j * np.kron(sx_gate_ctr_1, sx_gate_trg_1) @ (first_cr @ np.kron(x_gate , relaxation_gate) @ second_cr ) @ np.kron(sx_gate_ctr_2, sx_gate_trg_2)


class NoiseScalingMixin:
    """Shared noise scaling utilities (physically valid + numerically stable)."""

    @staticmethod
    def scale_p(p, scale):
        if p <= 0:
            return 0.0

        p_scaled = min(p * scale, 1.0 - 1e-12)

        if p_scaled >= 1.0:
            warnings.warn(
                f"scale_p: probability saturated (p={p}, scale={scale}) → {p_scaled:.3e}. "
                "Clamping to 1 - 1e-12.",
                RuntimeWarning
            )
            return 1.0 - 1e-12

        if p_scaled < 0 or not np.isfinite(p_scaled):
            warnings.warn(
                f"scale_p: invalid probability (p={p}, scale={scale}) → {p_scaled}. "
                "Clamping to 0.",
                RuntimeWarning
            )
            return 0.0

        return p_scaled

    @staticmethod
    def scale_T(T, scale, min_T=1e-12, max_T=1e17):
        if T <= 0:
            warnings.warn(
                f"scale_T: non-positive T={T}. Using min_T={min_T}.",
                RuntimeWarning
            )
            return min_T

        gamma = 1.0 / T
        gamma_scaled = gamma * scale

        if not np.isfinite(gamma_scaled) or gamma_scaled <= 0:
            warnings.warn(
                f"scale_T: invalid gamma_scaled={gamma_scaled}. Using min_T={min_T}.",
                RuntimeWarning
            )
            return min_T

        T_scaled = 1.0 / gamma_scaled

        # Clamp with warnings
        if T_scaled < min_T:
            warnings.warn(
                f"scale_T: T_scaled too small ({T_scaled:.3e}) → clamped to {min_T}.",
                RuntimeWarning
            )
            return min_T

        if T_scaled > max_T:
            warnings.warn(
                f"scale_T: T_scaled too large ({T_scaled:.3e}) → clamped to {max_T}.",
                RuntimeWarning
            )
            return max_T

        return T_scaled
    

    
class ScaledNoiseGates(NoiseScalingMixin):
    """ Version of Gates in which the noise is scaled by a certain factor noise_scale of at least 1e-15.

    The smaller the noise_scaling value, the less noisy the gates are.

    Examples:
        .. code:: python

            from quantum_gates.gates import ScaledNoiseGates

            gateset = ScaledNoiseGates(noise_scaling=0.1, pulse=pulse)  # 10x less noise
            sampled_x = gateset.X(phi, p, T1, T2)
    """

    def __init__(self, noise_scaling: float, pulse: Pulse=constant_pulse):
        assert noise_scaling >= 1e-20, f"Too small noise scaling {noise_scaling} < 1e-20."
        self.noise_scaling = noise_scaling
        self.gates = Gates(pulse)

    def relaxation(self, Dt, T1, T2, *, qubit_index=None) -> np.array:
            return self.gates.relaxation(
                Dt,
                self.scale_T(T1, self.noise_scaling),
                self.scale_T(T2, self.noise_scaling),
            )

    def bitflip(self, Dt, p, *, qubit_index=None) -> np.array:
        return self.gates.bitflip(Dt, self.scale_p(p, self.noise_scaling))

    def depolarizing(self, Dt, p, *, qubit_index=None) -> np.array:
        return self.gates.depolarizing(Dt, self.scale_p(p, self.noise_scaling))

    def single_qubit_gate(self, theta, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.single_qubit_gate(
            theta,
            phi,
            self.scale_p(p, self.noise_scaling),
            self.scale_T(T1, self.noise_scaling),
            self.scale_T(T2, self.noise_scaling)
        )

    def X(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.X(phi, self.scale_p(p, self.noise_scaling), self.scale_T(T1, self.noise_scaling), self.scale_T(T2, self.noise_scaling))

    def SX(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.SX(phi, self.scale_p(p, self.noise_scaling), self.scale_T(T1, self.noise_scaling), self.scale_T(T2, self.noise_scaling))

    def CR(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.CR(
            theta,
            phi,
            t_cr,
            self.scale_p(p_cr, self.noise_scaling),
            self.scale_T(T1_ctr, self.noise_scaling),
            self.scale_T(T2_ctr, self.noise_scaling),
            self.scale_T(T1_trg, self.noise_scaling),
            self.scale_T(T2_trg, self.noise_scaling)
        )

    def CNOT(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.CNOT(
            phi_ctr,
            phi_trg,
            t_cnot,
            self.scale_p(p_cnot, self.noise_scaling),
            self.scale_p(p_single_ctr, self.noise_scaling),
            self.scale_p(p_single_trg, self.noise_scaling),
            self.scale_T(T1_ctr, self.noise_scaling),
            self.scale_T(T2_ctr, self.noise_scaling),
            self.scale_T(T1_trg, self.noise_scaling),
            self.scale_T(T2_trg, self.noise_scaling)
        )

    def CNOT_inv(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.CNOT_inv(
            phi_ctr,
            phi_trg,
            t_cnot,
            self.scale_p(p_cnot, self.noise_scaling),
            self.scale_p(p_single_ctr, self.noise_scaling),
            self.scale_p(p_single_trg, self.noise_scaling),
            self.scale_T(T1_ctr, self.noise_scaling),
            self.scale_T(T2_ctr, self.noise_scaling),
            self.scale_T(T1_trg, self.noise_scaling),
            self.scale_T(T2_trg, self.noise_scaling)
        )
    
    def ECR(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg,*, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.ECR(
            phi_ctr,
            phi_trg,
            t_ecr,
            self.scale_p(p_ecr, self.noise_scaling),
            self.scale_p(p_single_ctr, self.noise_scaling),
            self.scale_p(p_single_trg, self.noise_scaling),
            self.scale_T(T1_ctr, self.noise_scaling),
            self.scale_T(T2_ctr, self.noise_scaling),
            self.scale_T(T1_trg, self.noise_scaling),
            self.scale_T(T2_trg, self.noise_scaling)
        )
    
    def ECR_inv(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.ECR_inv(
            phi_ctr,
            phi_trg,
            t_ecr,
            self.scale_p(p_ecr, self.noise_scaling),
            self.scale_p(p_single_ctr, self.noise_scaling),
            self.scale_p(p_single_trg, self.noise_scaling),
            self.scale_T(T1_ctr, self.noise_scaling),
            self.scale_T(T2_ctr, self.noise_scaling),
            self.scale_T(T1_trg, self.noise_scaling),
            self.scale_T(T2_trg, self.noise_scaling)
        )


class CustomNoiseGates(NoiseScalingMixin):
    """
    Gates with independently scalable device noise parameters.

    By default, all scalers are set to 1.0 (physical device noise).
    Smaller values reduce the effective noise strength.

    Parameters
    ----------
    p_scale : float
        Scaling factor for stochastic error probabilities (bitflip, depolarizing, etc.)
    T1_scale : float
        Scaling factor for T1 relaxation times
    T2_scale : float
        Scaling factor for T2 dephasing times
    """

    def __init__(self, p_scale: float = 1.0, T1_scale: float = 1.0, T2_scale: float = 1.0, pulse: Pulse=constant_pulse):
        # Validate p_scale
        if not np.isfinite(p_scale):
            raise ValueError(f"p_scale must be finite, got {p_scale}")
        if p_scale < 0:
            raise ValueError(f"p_scale must be >= 0, got {p_scale}")
        if p_scale > 1e3:
            raise ValueError(f"p_scale too large: {p_scale} > 1e3 (likely unphysical)")

        # Validate T1_scale / T2_scale
        for name, val in {"T1_scale": T1_scale, "T2_scale": T2_scale}.items():
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
            if val > 1e6:
                raise ValueError(f"{name} too large: {val} > 1e6 (unphysical regime)")

        # Store
        self.p_scale = float(p_scale)
        self.T1_scale = float(T1_scale)
        self.T2_scale = float(T2_scale)

        # Gate factory
        self.gates = Gates(pulse)

    def relaxation(self, Dt, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.relaxation(Dt, self.scale_T(T1, self.T1_scale), self.scale_T(T2, self.T2_scale))

    def bitflip(self, Dt, p, *, qubit_index=None) -> np.array:
        return self.gates.bitflip(Dt, self.scale_p(p, self.p_scale))

    def depolarizing(self, Dt, p, *, qubit_index=None) -> np.array:
        return self.gates.depolarizing(Dt, self.scale_p(p, self.p_scale))

    def single_qubit_gate(self, theta, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.single_qubit_gate(
            theta,
            phi,
            self.scale_p(p, self.p_scale),
            self.scale_T(T1, self.T1_scale),
            self.scale_T(T2, self.T2_scale)
        )

    def X(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.X(phi, self.scale_p(p, self.p_scale), self.scale_T(T1, self.T1_scale), self.scale_T(T2, self.T2_scale))

    def SX(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.SX(phi, self.scale_p(p, self.p_scale), self.scale_T(T1, self.T1_scale), self.scale_T(T2, self.T2_scale))

    def CR(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.CR(
            theta,
            phi,
            t_cr,
            self.scale_p(p_cr, self.p_scale),
            self.scale_T(T1_ctr, self.T1_scale),
            self.scale_T(T2_ctr, self.T2_scale),
            self.scale_T(T1_trg, self.T1_scale),
            self.scale_T(T2_trg, self.T2_scale),
        )

    def CNOT(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.CNOT(
            phi_ctr,
            phi_trg,
            t_cnot,
            self.scale_p(p_cnot, self.p_scale),
            self.scale_p(p_single_ctr, self.p_scale),
            self.scale_p(p_single_trg, self.p_scale),
            self.scale_T(T1_ctr, self.T1_scale),
            self.scale_T(T2_ctr, self.T2_scale),
            self.scale_T(T1_trg, self.T1_scale),
            self.scale_T(T2_trg, self.T2_scale),
        )

    def CNOT_inv(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.CNOT_inv(
            phi_ctr,
            phi_trg,
            t_cnot,
            self.scale_p(p_cnot, self.p_scale),
            self.scale_p(p_single_ctr, self.p_scale),
            self.scale_p(p_single_trg, self.p_scale),
            self.scale_T(T1_ctr, self.T1_scale),
            self.scale_T(T2_ctr, self.T2_scale),
            self.scale_T(T1_trg, self.T1_scale),
            self.scale_T(T2_trg, self.T2_scale)
        )
    
    def ECR(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.ECR(
            phi_ctr,
            phi_trg,
            t_ecr,
            self.scale_p(p_ecr, self.p_scale),
            self.scale_p(p_single_ctr, self.p_scale),
            self.scale_p(p_single_trg, self.p_scale),
            self.scale_T(T1_ctr, self.T1_scale),
            self.scale_T(T2_ctr, self.T2_scale),
            self.scale_T(T1_trg, self.T1_scale),
            self.scale_T(T2_trg, self.T2_scale)
        )
    
    def ECR_inv(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.ECR_inv(
            phi_ctr,
            phi_trg,
            t_ecr,
            self.scale_p(p_ecr, self.p_scale),
            self.scale_p(p_single_ctr, self.p_scale),
            self.scale_p(p_single_trg, self.p_scale),
            self.scale_T(T1_ctr, self.T1_scale),
            self.scale_T(T2_ctr, self.T2_scale),
            self.scale_T(T1_trg, self.T1_scale),
            self.scale_T(T2_trg, self.T2_scale),
        )
        
        
class SpecificNoiseGates:
    """
    Gates with explicitly specified noise parameters.

    If a parameter is provided (not None), it overrides the input value.
    Otherwise, the original value passed to the gate is used.

    Parameters
    ----------
    p_val : float or None
        Fixed stochastic error probability override
    T1_val : float or None
        Fixed T1 relaxation time override
    T2_val : float or None
        Fixed T2 dephasing time override
    """

    def __init__(
        self,
        p_val: float | None = None,
        T1_val: float | None = None,
        T2_val: float | None = None,
        pulse: Pulse = constant_pulse,
    ):
        # Validation helper
        def _validate_p(val):
            if val is None:
                return
            if not np.isfinite(val):
                raise ValueError(f"p_val must be finite, got {val}")
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"p_val must be in [0,1], got {val}")


        def _validate_T(name, val, min_val=1e-20, max_val=1e20):
            if val is None:
                return
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
            if val < min_val:
                raise ValueError(f"{name} too small: {val} < {min_val}")
            if val > max_val:
                raise ValueError(f"{name} too large: {val} > {max_val}")

        _validate_p(p_val)
        _validate_T("T1_val", T1_val)
        _validate_T("T2_val", T2_val)

        # Store overrides
        self.p_val = p_val
        self.T1_val = T1_val
        self.T2_val = T2_val

        self.gates = Gates(pulse)

    # Helpers
    def _p(self, p):
        return self.p_val if self.p_val is not None else p

    def _T1(self, T1):
        return self.T1_val if self.T1_val is not None else T1

    def _T2(self, T2):
        return self.T2_val if self.T2_val is not None else T2

    # Noise channels
    def relaxation(self, Dt, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.relaxation(Dt, self._T1(T1), self._T2(T2))

    def bitflip(self, Dt, p, *, qubit_index=None) -> np.array:
        return self.gates.bitflip(Dt, self._p(p))

    def depolarizing(self, Dt, p, *, qubit_index=None) -> np.array:
        return self.gates.depolarizing(Dt, self._p(p))

    # Single-qubit gates
    def single_qubit_gate(self, theta, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.single_qubit_gate(
            theta, phi,
            self._p(p),
            self._T1(T1),
            self._T2(T2),
        )

    def X(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.X(phi, self._p(p), self._T1(T1), self._T2(T2))
        

    def SX(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self.gates.SX(phi, self._p(p), self._T1(T1), self._T2(T2))

    # Two-qubit gates
    def CR(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg, *, ctr_index=None, trg_index=None) -> np.array:
        return self.gates.CR(
            theta, phi, t_cr,
            self._p(p_cr),
            self._T1(T1_ctr),
            self._T2(T2_ctr),
            self._T1(T1_trg),
            self._T2(T2_trg),
        )

    def CNOT(
        self, 
        phi_ctr, phi_trg, t_cnot, p_cnot, 
        p_single_ctr, p_single_trg, 
        T1_ctr, T2_ctr, T1_trg, T2_trg, 
        *, ctr_index=None, trg_index=None
        ) -> np.array:
        return self.gates.CNOT(
            phi_ctr,
            phi_trg,
            t_cnot,

            # Only override two-qubit noise
            self._p(p_cnot),

            # Keep single-qubit noise as-is
            p_single_ctr,
            p_single_trg,

            # Still override decoherence
            self._T1(T1_ctr),
            self._T2(T2_ctr),
            self._T1(T1_trg),
            self._T2(T2_trg),
        )

    def CNOT_inv(
        self, 
        phi_ctr, phi_trg, t_cnot, p_cnot,
        p_single_ctr, p_single_trg,
        T1_ctr, T2_ctr, T1_trg, T2_trg,
        *, ctr_index=None, trg_index=None
        )-> np.array:
        return self.gates.CNOT_inv(
            phi_ctr,
            phi_trg,
            t_cnot,
            # Only override two-qubit noise
            self._p(p_cnot),
            # Keep single-qubit noise as-is
            p_single_ctr,
            p_single_trg,
            # Still override decoherence
            self._T1(T1_ctr),
            self._T2(T2_ctr),
            self._T1(T1_trg),
            self._T2(T2_trg),
        )
    
    def ECR(
        self, 
        phi_ctr, phi_trg, t_ecr, 
        p_ecr, p_single_ctr, p_single_trg, 
        T1_ctr, T2_ctr, T1_trg, T2_trg, 
        *, ctr_index=None, trg_index=None
        ) -> np.array:
        return self.gates.ECR(
            phi_ctr,
            phi_trg,
            t_ecr,
            # Only override two-qubit noise
            self._p(p_ecr),
            # Keep single-qubit noise as-is
            p_single_ctr,
            p_single_trg,
            # Still override decoherence
            self._T1(T1_ctr),
            self._T2(T2_ctr),
            self._T1(T1_trg),
            self._T2(T2_trg),
        )
    
    def ECR_inv(
        self, 
        phi_ctr, phi_trg, t_ecr, 
        p_ecr, p_single_ctr, p_single_trg, 
        T1_ctr, T2_ctr, T1_trg, T2_trg,
        *, ctr_index=None, trg_index=None
        ) -> np.array:
        
        return self.gates.ECR_inv(
            phi_ctr,
            phi_trg,
            t_ecr,
            # Only override two-qubit noise
            self._p(p_ecr),
            # Keep single-qubit noise as-is
            p_single_ctr,
            p_single_trg,   
            self._T1(T1_ctr),
            self._T2(T2_ctr),
            self._T1(T1_trg),
            self._T2(T2_trg),
        )



class CustomNoiseChannelsGates(object):
    """
    Delegates gate construction to either:
      - NoiseFreeGates (for selected qubits)
      - CustomNoiseGates (for all others)

    Fully interface-compatible with existing gate classes.
    """

    def __init__(
        self,
        noiseless_qubits,
        p_scale=1.0,
        T1_scale=1.0,
        T2_scale=1.0,
        pulse=constant_pulse,
    ):
        # Validate noiseless_qubits
        if noiseless_qubits is None:
            noiseless_qubits = []

        if not all(isinstance(q, int) and q >= 0 for q in noiseless_qubits):
            raise ValueError(f"Invalid qubit indices: {noiseless_qubits}")

        self.noiseless_qubits = set(noiseless_qubits)

        # Noise models
        self.noise_free = NoiseFreeGates()
        self.noisy = CustomNoiseGates(
            p_scale=p_scale,
            T1_scale=T1_scale,
            T2_scale=T2_scale,
            pulse=pulse,
        )


    # Internal dispatchers

    def _select(self, qubit_index):
        if qubit_index is None:
            # Explicit fallback (debug-friendly)
            return self.noisy

        if not isinstance(qubit_index, int):
            raise ValueError(f"Invalid qubit index type: {qubit_index}")

        if qubit_index in self.noiseless_qubits:
            return self.noise_free

        return self.noisy

    def _select_two(self, ctr, trg):
        if ctr is None or trg is None:
            return self.noisy

        if not isinstance(ctr, int) or not isinstance(trg, int):
            raise ValueError(f"Invalid qubit indices: ctr={ctr}, trg={trg}")

        if ctr in self.noiseless_qubits and trg in self.noiseless_qubits:
            return self.noise_free

        return self.noisy
    

    # Single-qubit noise processes

    def relaxation(self, Dt, T1, T2, *, qubit_index=None) -> np.array:
        return self._select(qubit_index).relaxation(Dt, T1, T2)

    def bitflip(self, Dt, p, *, qubit_index=None) -> np.array:
        return self._select(qubit_index).bitflip(Dt, p)

    def depolarizing(self, Dt, p, *, qubit_index=None) -> np.array:
        return self._select(qubit_index).depolarizing(Dt, p)

    # Single-qubit gates

    def single_qubit_gate(self, theta, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self._select(qubit_index).single_qubit_gate(
            theta, phi, p, T1, T2
        )

    def X(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self._select(qubit_index).X(phi, p, T1, T2)

    def SX(self, phi, p, T1, T2, *, qubit_index=None) -> np.array:
        return self._select(qubit_index).SX(phi, p, T1, T2)

    # Two-qubit gates
    # Physical channels depend on both control and target qubits, so we check both indices to determine noise level.
    # If ctr in noiseless and trg in noiseless -> noise free CR, else -> noisy CR.

    def CR(
        self,
        theta, phi, t_cr, p_cr,
        T1_ctr, T2_ctr, T1_trg, T2_trg,
        *, ctr_index=None, trg_index=None
    ) -> np.array:
        return self._select_two(ctr=ctr_index, trg=trg_index).CR(
            theta, phi, t_cr, p_cr,
            T1_ctr, T2_ctr, T1_trg, T2_trg
        )

    def CNOT(
        self,
        phi_ctr, phi_trg, t_cnot,
        p_cnot, p_single_ctr, p_single_trg,
        T1_ctr, T2_ctr, T1_trg, T2_trg,
        *, ctr_index=None, trg_index=None
    ) -> np.array:
        return self._select_two(ctr=ctr_index, trg=trg_index).CNOT(
            phi_ctr, phi_trg, t_cnot,
            p_cnot, p_single_ctr, p_single_trg,
            T1_ctr, T2_ctr, T1_trg, T2_trg
        )

    def CNOT_inv(
        self,
        phi_ctr, phi_trg, t_cnot,
        p_cnot, p_single_ctr, p_single_trg,
        T1_ctr, T2_ctr, T1_trg, T2_trg,
        *, ctr_index=None, trg_index=None
    ) -> np.array:
        return self._select_two(ctr=ctr_index, trg=trg_index).CNOT_inv(
            phi_ctr, phi_trg, t_cnot,
            p_cnot, p_single_ctr, p_single_trg,
            T1_ctr, T2_ctr, T1_trg, T2_trg
        )

    def ECR(
        self,
        phi_ctr, phi_trg, t_ecr,
        p_ecr, p_single_ctr, p_single_trg,
        T1_ctr, T2_ctr, T1_trg, T2_trg,
        *, ctr_index=None, trg_index=None
    ) -> np.array:
        return self._select_two(ctr=ctr_index, trg=trg_index).ECR(
            phi_ctr, phi_trg, t_ecr,
            p_ecr, p_single_ctr, p_single_trg,
            T1_ctr, T2_ctr, T1_trg, T2_trg
        )

    def ECR_inv(
        self,
        phi_ctr, phi_trg, t_ecr,
        p_ecr, p_single_ctr, p_single_trg,
        T1_ctr, T2_ctr, T1_trg, T2_trg,
        *, ctr_index=None, trg_index=None
    ) -> np.array:
        return self._select_two(ctr=ctr_index, trg=trg_index).ECR_inv(
            phi_ctr, phi_trg, t_ecr,
            p_ecr, p_single_ctr, p_single_trg,
            T1_ctr, T2_ctr, T1_trg, T2_trg
        )



""" Instances of Gates with different noise levels and pulse shapes """

# Constant pulses
standard_gates = Gates(pulse=constant_pulse)
numerical_gates = Gates(pulse=constant_pulse_numerical)
noise_free_gates = NoiseFreeGates()
almost_noise_free_gates = ScaledNoiseGates(noise_scaling=1e-15)

# Reduced stochastic noise only
low_pauli_noise_gates = CustomNoiseGates(
    p_scale=0.1,
    T1_scale=1.0,
    T2_scale=1.0,
)

# Reduced decoherence only
long_coherence_gates = CustomNoiseGates(
    p_scale=1.0,
    T1_scale=0.1,
    T2_scale=0.1,
)