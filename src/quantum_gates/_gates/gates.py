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

    def relaxation(self, Dt, T1, T2) -> np.array:
        return self.relaxation_c.construct(Dt, T1, T2)

    def bitflip(self, Dt, p) -> np.array:
        return self.bitflip_c.construct(Dt, p)

    def depolarizing(self, Dt, p) -> np.array:
        return self.depolarizing_c.construct(Dt, p)

    def single_qubit_gate(self, theta, phi, p, T1, T2) -> np.array:
        return self.single_qubit_gate_c.construct(theta, phi, p, T1, T2)

    def X(self, phi, p, T1, T2) -> np.array:
        return self.x_c.construct(phi, p, T1, T2)

    def SX(self, phi, p, T1, T2) -> np.array:
        return self.sx_c.construct(phi, p, T1, T2)

    def CR(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.cr_c.construct(theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)

    def CNOT(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.cnot_c.construct(phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg)

    def CNOT_inv(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.cnot_inv_c.construct(phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg)
    
    def ECR(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.ecr_c.construct(phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg)
    
    def ECR_inv(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
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

    def relaxation(self, Dt, T1, T2) -> np.array:
        """ Returns single qubit relaxation in noise free regime -> identity. """
        return np.eye(2)

    def bitflip(self, Dt, p) -> np.array:
        """ Returns single qubit bitflip in noise free regime -> identity. """
        return np.eye(2)

    def depolarizing(self, Dt, p) -> np.array:
        """ Returns single qubit depolarizing in noise free regime -> identity. """
        return np.eye(2)

    def single_qubit_gate(self, theta, phi, p, T1, T2):
        """ Returns general single qubit gate in noise free regime. """
        U = np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
        return U

    def X(self, phi, p, T1, T2) -> np.array:
        """ Returns X gate in noise free regime. """
        theta = np.pi
        U = np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
        return U

    def SX(self, phi, p, T1, T2) -> np.array:
        """ Returns SX gate in noise free regime. """
        theta = np.pi / 2
        U = np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
        return U

    def CNOT(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
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

    def CNOT_inv(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
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

    def CR(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg):
        """ Returns CR gate in noise free regime. """
        return np.array(
            [[np.cos(theta/2), -1J*np.sin(theta/2) * np.exp(-1J * phi), 0, 0],
             [-1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2), 0, 0],
             [0, 0, np.cos(theta/2), 1J*np.sin(theta/2) * np.exp(-1J * phi)],
             [0, 0, 1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
    
    def ECR(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
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
    

    def ECR_inv(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
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

    

class ScaledNoiseGates(object):
    """ Version of Gates in which the noise is scaled by a certain factor noise_scale of at least 1e-15.

    The smaller the noise_scaling value, the less noisy the gates are.

    Examples:
        .. code:: python

            from quantum_gates.gates import ScaledNoiseGates

            gateset = ScaledNoiseGates(noise_scaling=0.1, pulse=pulse)  # 10x less noise
            sampled_x = gateset.X(phi, p, T1, T2)
    """

    def __init__(self, noise_scaling: float, pulse: Pulse=constant_pulse):
        assert noise_scaling >= 1e-15, f"Too small noise scaling {noise_scaling} < 1e-15."
        self.noise_scaling = noise_scaling
        self.gates = Gates(pulse)

    def relaxation(self, Dt, T1, T2) -> np.array:
        return self.gates.relaxation(Dt, T1 / self.noise_scaling, T2 / self.noise_scaling)

    def bitflip(self, Dt, p) -> np.array:
        return self.gates.bitflip(Dt, p * self.noise_scaling)

    def depolarizing(self, Dt, p) -> np.array:
        return self.gates.depolarizing(Dt, p * self.noise_scaling)

    def single_qubit_gate(self, theta, phi, p, T1, T2) -> np.array:
        return self.gates.single_qubit_gate(
            theta,
            phi,
            p * self.noise_scaling,
            T1 / self.noise_scaling,
            T2 / self.noise_scaling
        )

    def X(self, phi, p, T1, T2) -> np.array:
        return self.gates.X(phi, p * self.noise_scaling, T1 / self.noise_scaling, T2 / self.noise_scaling)

    def SX(self, phi, p, T1, T2) -> np.array:
        return self.gates.SX(phi, p * self.noise_scaling, T1 / self.noise_scaling, T2 / self.noise_scaling)

    def CR(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.CR(
            theta,
            phi,
            t_cr,
            p_cr * self.noise_scaling,
            T1_ctr / self.noise_scaling,
            T2_ctr / self.noise_scaling,
            T1_trg / self.noise_scaling,
            T2_trg / self.noise_scaling
        )

    def CNOT(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.CNOT(
            phi_ctr,
            phi_trg,
            t_cnot,
            p_cnot * self.noise_scaling,
            p_single_ctr * self.noise_scaling,
            p_single_trg * self.noise_scaling,
            T1_ctr / self.noise_scaling,
            T2_ctr / self.noise_scaling,
            T1_trg / self.noise_scaling,
            T2_trg / self.noise_scaling
        )

    def CNOT_inv(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.CNOT_inv(
            phi_ctr,
            phi_trg,
            t_cnot,
            p_cnot * self.noise_scaling,
            p_single_ctr * self.noise_scaling,
            p_single_trg * self.noise_scaling,
            T1_ctr / self.noise_scaling,
            T2_ctr / self.noise_scaling,
            T1_trg / self.noise_scaling,
            T2_trg / self.noise_scaling
        )
    
    def ECR(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.ECR(
            phi_ctr,
            phi_trg,
            t_ecr,
            p_ecr * self.noise_scaling,
            p_single_ctr * self.noise_scaling,
            p_single_trg * self.noise_scaling,
            T1_ctr / self.noise_scaling,
            T2_ctr / self.noise_scaling,
            T1_trg / self.noise_scaling,
            T2_trg / self.noise_scaling
        )
    
    def ECR_inv(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.ECR_inv(
            phi_ctr,
            phi_trg,
            t_ecr,
            p_ecr * self.noise_scaling,
            p_single_ctr * self.noise_scaling,
            p_single_trg * self.noise_scaling,
            T1_ctr / self.noise_scaling,
            T2_ctr / self.noise_scaling,
            T1_trg / self.noise_scaling,
            T2_trg / self.noise_scaling
        )


class CustomNoiseGates(object):
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
        for name, val in {
            "p_scale": p_scale,
            "T1_scale": T1_scale,
            "T2_scale": T2_scale,
        }.items():
            assert val >= 1e-15, f"{name} too small: {val} < 1e-15."
        
        self.p_scale = p_scale
        self.T1_scale = T1_scale
        self.T2_scale = T2_scale
        
        self.gates = Gates(pulse)

    def relaxation(self, Dt, T1, T2) -> np.array:
        return self.gates.relaxation(Dt, T1 / self.T1_scale, T2 / self.T2_scale)

    def bitflip(self, Dt, p) -> np.array:
        return self.gates.bitflip(Dt, p * self.p_scale)

    def depolarizing(self, Dt, p) -> np.array:
        return self.gates.depolarizing(Dt, p * self.p_scale)

    def single_qubit_gate(self, theta, phi, p, T1, T2) -> np.array:
        return self.gates.single_qubit_gate(
            theta,
            phi,
            p * self.p_scale,
            T1 / self.T1_scale,
            T2 / self.T2_scale
        )

    def X(self, phi, p, T1, T2) -> np.array:
        return self.gates.X(phi, p * self.p_scale, T1 / self.T1_scale, T2 / self.T2_scale)

    def SX(self, phi, p, T1, T2) -> np.array:
        return self.gates.SX(phi, p * self.p_scale, T1 / self.T1_scale, T2 / self.T2_scale)

    def CR(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.CR(
            theta,
            phi,
            t_cr,
            p_cr * self.p_scale,
            T1_ctr / self.T1_scale,
            T2_ctr / self.T2_scale,
            T1_trg / self.T1_scale,
            T2_trg / self.T2_scale,
        )

    def CNOT(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.CNOT(
            phi_ctr,
            phi_trg,
            t_cnot,
            p_cnot * self.p_scale,
            p_single_ctr * self.p_scale,
            p_single_trg * self.p_scale,
            T1_ctr / self.T1_scale,
            T2_ctr / self.T2_scale,
            T1_trg / self.T1_scale,
            T2_trg / self.T2_scale,
        )

    def CNOT_inv(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.CNOT_inv(
            phi_ctr,
            phi_trg,
            t_cnot,
            p_cnot * self.p_scale,
            p_single_ctr * self.p_scale,
            p_single_trg * self.p_scale,
            T1_ctr / self.T1_scale,
            T2_ctr / self.T2_scale,
            T1_trg / self.T1_scale,
            T2_trg / self.T2_scale
        )
    
    def ECR(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.ECR(
            phi_ctr,
            phi_trg,
            t_ecr,
            p_ecr * self.p_scale,
            p_single_ctr * self.p_scale,
            p_single_trg * self.p_scale,
            T1_ctr / self.T1_scale,
            T2_ctr / self.T2_scale,
            T1_trg / self.T1_scale,
            T2_trg / self.T2_scale
        )
    
    def ECR_inv(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        return self.gates.ECR_inv(
            phi_ctr,
            phi_trg,
            t_ecr,
            p_ecr * self.p_scale,
            p_single_ctr * self.p_scale,
            p_single_trg * self.p_scale,
            T1_ctr / self.T1_scale,
            T2_ctr / self.T2_scale,
            T1_trg / self.T1_scale,
            T2_trg / self.T2_scale,
        )

""" Instances """

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