import scipy.linalg
import numpy as np

from .integrator import Integrator


class BitflipFactory(object):

    def construct(self, tm, rout) -> np.array:
        """Generates the noisy gate for bitflips.

        This is the exact solution, a unitary matrix. It is used for bitflip error on measurements.

        Args:
            tm (float): Measurement time in ns.
            rout (float): Readout error probability.

        Returns:
            Array representing the bitflip noise gate.
        """
        tg = 35 * 10**(-9)
        Dtm = tm / tg
        e = np.sqrt(rout/Dtm)
        W = np.random.normal(0, np.sqrt(Dtm))
        result = np.array(
            [[np.cos(e * W), 1J * np.sin(e * W)],
             [1J * np.sin(e * W), np.cos(e * W)]]
        )
        return result


class DepolarizingFactory(object):

    def construct(self, Dt, p) -> np.array:
        """Generates the noisy gate for depolarization.

        This is the 2nd order approximated solution, a unitary matrix. It implements the single-qubit depolarizing error
        on idle qubits.

        Args:
            Dt (float): Idle time in ns.
            p (float): Single-qubit depolarizing error probability.

        Returns:
            Array representing the depolarizing noise gate.
        """
        tg = 35 * 10**(-9)
        Dt = Dt / tg
        ed = np.sqrt(p/4)
        W1 = np.random.normal(0, np.sqrt(Dt))
        W2 = np.random.normal(0, np.sqrt(Dt))
        W3 = np.random.normal(0, np.sqrt(Dt))
        X = np.array([[0,1],[1,0]])
        Y = np.array([[0,-1J],[1J,0]])
        Z = np.array([[1,0],[0,-1]])
        I1 = ed * X * W1
        I2 = ed * Y * W2
        I3 = ed * Z * W3
        result = scipy.linalg.expm(1J * I1 + 1J * I2 + 1J * I3)
        return result


class RelaxationFactory(object):

    def construct(self, Dt, T1, T2) -> np.array:
        """Generates the noisy gate for combined amplitude and phase damping.

        This is the exact solution, a non-unitary matrix. It implements the single-qubit relaxation error on idle
        qubits.

        Args:
            Dt (float): idle time in ns.
            T1 (float): qubit's amplitude damping time in ns.
            T2 (float): qubit's dephasing time in ns.

        Returns:
              Array representing the amplitude and phase damping noise gate.
        """
        # Constants
        tg = 35 * 10**(-9)
        Dt = Dt / tg

        # Helper function
        def V(Dt) -> float:
            return 1-np.exp(-e1**2 * Dt)

        # Calculations
        if T1 == 0:
            e1 = 0
        else:
            e1 = np.sqrt(tg/T1)

        if T2 == 0:
            ep = 0
        else:
            e2 = np.sqrt(tg/T2)
            ep_squared = (1/2) * (e2**2 - e1**2/2) #updated to handel T2 > 2T1
            
            # Handle case where T2 > 2*T1 (physical constraint violated)
            if ep_squared < 0:
                print(f"Warning: T2 ({T2}) > 2*T1 ({2*T1}), clamping ep to 0")
                ep = 0
            else:
                ep = np.sqrt(ep_squared)

        W = np.random.normal(0, np.sqrt(Dt))
        I = np.random.normal(0, np.sqrt(V(Dt)))
        result = np.array(
            [[np.exp(1J * ep * W), 1J * I * np.exp(-1J * ep * W)],
             [0, np.exp(-e1**2/2 * Dt) * np.exp(-1J * ep * W)]]
        )
        return result


class SingleQubitGateFactory(object):
    """Generates a general single qubit gate on devices from IBM with noise.

    This is the 2 order approximated solution, a non-unitary matrix.

    Args:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.

    Note:
        The pulse shape / parametrization is hidden in the integrator, such that we can use caching of integration
        result to speedup the code.

    Attributes:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.
    """

    def __init__(self, integrator: Integrator):
        self.integrator = integrator

    def construct(self, theta, phi, p, T1, T2) -> np.array:
        """Samples a general single qubit gate on devices from IBM with noise.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.

        Returns:
              Array representing a general single-qubit noisy quantum gate.
        """

        """ 0) CONSTANTS """

        tg = 35 * 10**(-9)
        ed = np.sqrt(p/4)

        # Amplitude damping time is zero
        if T1 == 0:
            e1 = 0
        else:
            e1 = np.sqrt(tg/T1)

        # Dephasing time is zero
        if T2 == 0:
            ep = 0
        else:
            e2 = np.sqrt(tg/T2)
            ep_squared = (1/2) * (e2**2 - e1**2/2) #updated to handel T2 > 2T1
            
            # Handle case where T2 > 2*T1 (physical constraint violated)
            if ep_squared < 0:
                print(f"Warning: T2 ({T2}) > 2*T1 ({2*T1}), clamping ep to 0")
                ep = 0
            else:
                ep = np.sqrt(ep_squared)

        """ 1) UNITARY CONTRIBUTION """

        # Unitary contribution due to drive Hamiltonian
        U = self._unitary_contribution(theta, phi)


        """ 2) DEPOLARIZATION CONTRIBUTION """

        # Variances and covariances for depolarization Itô processes depending on X(t)
        Idx1, Idx2, Wdx = self._ito_integrals_for_X_Y_sigma_min(theta)
        Idx = ed * np.array([[np.sin(phi)*Idx1,Wdx + (np.exp(-2*1J*phi)-1)*Idx2],[Wdx + (np.exp(+2*1J*phi)-1)*Idx2,-np.sin(phi)*Idx1]])

        #Variances and covariances for depolarization Itô processes depending on Y(t)
        Idy1, Idy2, Wdy = self._ito_integrals_for_X_Y_sigma_min(theta)
        Idy = ed * np.array([[-np.cos(phi)*Idy1, -1J*Wdy + 1J*(np.exp(-2*1J*phi)+1)*Idy2], [1J*Wdy - 1J*(np.exp(2*1J*phi)+1)*Idy2, np.cos(phi)*Idy1]])

        # Variances and covariances for depolarization Itô processes depending on Z(t)
        Idz1, Idz2 = self._ito_integrals_for_Z(theta)
        Idz = ed * np.array(
            [[Idz1, -1J * np.exp(-1J*phi) * Idz2],
             [1J * np.exp(1J*phi) * Idz2, -Idz1]]
        )


        """ 3) RELAXATION CONTRIBUTIONS """

        # Variances and covariances for relaxation Itô processes depending on sigma_min(t)
        Ir1, Ir2, Wr = self._ito_integrals_for_X_Y_sigma_min(theta)
        Ir = e1 * np.array([[-1J/2 * np.exp(1J*phi) * Ir1, Wr - Ir2], [np.exp(2*1J*phi)*Ir2,1J/2* np.exp(1J*phi) * Ir1]])

        # Deterministic contribution given by relaxation
        det1, det2, det3 = self._deterministic_relaxation(theta)
        deterministic = -e1**2/2 * np.array([[det1, 1J/2*np.exp(-1J*phi)*det2], [-1J/2*np.exp(1J*phi)*det2, det3]])

        # Variances and covariances for relaxation Itô processes depending on Z(t)
        Ip1, Ip2 = self._ito_integrals_for_Z(theta)
        Ip = ep * np.array([[Ip1, -1J * np.exp(-1J*phi) * Ip2], [1J * np.exp(1J*phi) * Ip2, -Ip1]])

        """ 4) COMBINE CONTRIBUTIONS """

        result = U @ scipy.linalg.expm(deterministic) @ scipy.linalg.expm(1J * Idx + 1J * Idy + 1J * Idz + 1J * Ir + 1J * Ip)
        return result

    def _unitary_contribution(self, theta, phi) -> np.array:
        """Unitary contribution due to drive Hamiltonian.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.

        Returns:
            Array representing the unitary contribution due to drive Hamiltonian.
        """
        U = np.array(
            [[np.cos(theta/2), - 1J * np.sin(theta/2) * np.exp(-1J * phi)],
             [- 1J * np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )
        return U

    def _ito_integrals_for_X_Y_sigma_min(self, theta) -> tuple[float]:
        """Ito integrals.

        Ito integrals for the following processes:
            * depolarization for X(t)
            * depolarization for Y(t)
            * relaxation for sigma_min(t).

        As illustration, we leave the variables names for X(t) in the calculation.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.

        Returns:
            Tuple of floats representing sampled results of the Ito integrals.
        """
        # Integral of sin(theta)**2
        Vdx_1 = self.integrator.integrate("sin(theta/a)**2", theta, 1)

        # Integral of sin**4(theta/2)
        Vdx_2 = self.integrator.integrate("sin(theta/(2*a))**4", theta, 1)

        # Integral of sin(theta) sin**2(theta/2)
        Covdx_12 = self.integrator.integrate("sin(theta/a)*sin(theta/(2*a))**2", theta, 1)

        # Integral of sin(theta)
        Covdx_1Wdx = self.integrator.integrate("sin(theta/a)", theta, 1)

        # Integral of sin**2(theta/2)
        Covdx_2Wdx = self.integrator.integrate("sin(theta/(2*a))**2", theta, 1)

        # Mean and covariance
        meand_x = np.array([0, 0, 0])
        covd_x = np.array([[Vdx_1, Covdx_12, Covdx_1Wdx], [Covdx_12, Vdx_2, Covdx_2Wdx], [Covdx_1Wdx, Covdx_2Wdx, 1]])

        # Sampling
        sample_dx = np.random.multivariate_normal(meand_x, covd_x, 1) # The variance of Wr is 1
        Idx1 = sample_dx[0,0]
        Idx2 = sample_dx[0,1]
        Wdx = sample_dx[0,2]

        return Idx1, Idx2, Wdx

    def _ito_integrals_for_Z(self, theta) -> tuple[float]:
        """Ito integrals.

        Ito integrals for the following processes:
            * depolarization for Z(t)
            * relaxation for Z(t).

        As illustration, we leave the variable names for the depolarization Itô processes depending on Z(t).

        Args:
            theta (float): angle of rotation on the Bloch sphere.

        Returns:
             Tuple of floats representing sampled results of the Ito integrals.
        """

        # Integral of cos(theta)**2
        Vdz_1 = self.integrator.integrate("cos(theta/a)**2", theta, 1)

        # Integral of sin(theta)**2
        Vdz_2 = self.integrator.integrate("sin(theta/a)**2", theta, 1)

        # Integral of sin(theta)*cos(theta)
        Covdz_12 = self.integrator.integrate("sin(theta/a)*cos(theta/a)", theta, 1)

        # Mean and covariance
        meand_z = np.array([0,0])
        covd_z = np.array(
            [[Vdz_1,Covdz_12],
             [Covdz_12, Vdz_2]]
        )

        # Sampling
        sample_dz = np.random.multivariate_normal(meand_z, covd_z, 1)
        Idz1 = sample_dz[0,0]
        Idz2 = sample_dz[0,1]

        return Idz1, Idz2

    def _deterministic_relaxation(self, theta):
        """Deterministic contribution given by relaxation

        Args:
            theta (float): angle of rotation on the Bloch sphere.

        Returns:
            Array representing the deterministic part of the relaxation process.
        """

        # Integral of sin(theta/(2*a))**2
        det1 = self.integrator.integrate("sin(theta/(2*a))**2", theta, 1)

        # Integral of sin(theta)
        det2 = self.integrator.integrate("sin(theta/a)", theta, 1)

        # Integral of cos(theta/2)**2
        det3 = self.integrator.integrate("cos(theta/(2*a))**2", theta, 1)

        return det1, det2, det3


class XFactory(object):
    """Factory for the X gate.

    Args:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.

    Note:
        The dependence on the pulse is hidden in the integrator.

    Attributes:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.
        constructor (SingleQubitGateFactory): Instance of the gate factory for general single qubit gates.
    """

    def __init__(self, integrator: Integrator):
        self.integrator = integrator
        self.constructor = SingleQubitGateFactory(self.integrator)

    def construct(self, phi, p, T1, T2) -> np.array:
        """Generates a noisy X gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the X single-qubit noisy quantum
        gate with depolarizing and relaxation errors during the unitary evolution.

        Args:
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.

        Returns:
              Array representing the X noisy quantum gate.
        """
        return self.constructor.construct(np.pi, phi, p, T1, T2)


class SXFactory(object):
    """Factory for the SX gate.

    Args:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.

    Note:
        The dependence on the pulse is hidden in the integrator.

    Attributes:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.
        constructor (SingleQubitGateFactory): Instance of the gate factory for general single qubit gates.
    """

    def __init__(self, integrator: Integrator):
        self.integrator = integrator
        self.constructor = SingleQubitGateFactory(self.integrator)

    def construct(self, phi, p, T1, T2) -> np.array:
        """Generates a noisy SX gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the SX single-qubit noisy quantum
        gate with depolarizing and relaxation errors during the unitary evolution.

        Args:
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.
            p (float): Single-qubit depolarizing error probability.
            T1 (float): Qubit's amplitude damping time in ns.
            T2 (float): Qubit's dephasing time in ns.

        Returns:
              Array representing the SX noisy quantum gate.
        """
        return self.constructor.construct(np.pi/2, phi, p, T1, T2)


class CRFactory(object):
    """Factory for constructing the noisy quantum gate for cross resonance (CR) two-qubit gate of IBM's devices.

    This is the 2 order approximated solution, non-unitary matrix.
    """

    def __init__(self, integrator):
        self.integrator = integrator

    def construct(self, theta, phi, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        """Generates a CR gate.

        This is the 2 order approximated solution, non-unitary matrix. It implements the CR two-qubit noisy quantum gate
        with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            theta (float): Angle of rotation on the Bloch sphere.
            phi (float): Phase of the drive defining axis of rotation on the Bloch sphere.
            t_cr (float): CR gate time in ns.
            p_cr (float): CR depolarizing error probability.
            T1_ctr (float): Control qubit's amplitude damping time in ns.
            T2_ctr (float): Control qubit's dephasing time in ns.
            T1_trg (float): Target qubit's amplitude damping time in ns.
            T2_trg (float): Target qubit's dephasing time in ns.

        Returns:
              CR two-qubit noisy quantum gate (numpy array)
        """


        """ 0) CONSTANTS """

        tg = 35 * 10**(-9)
        omega = theta
        a = t_cr / tg
        assert t_cr > 0, f"Expected t_cr to be > 0 but found {t_cr}"
        assert tg > 0, f"Expected tg to be > 0 but found {tg}"
        ed_cr = np.sqrt(p_cr/(4*a))

        if T1_ctr == 0:
            e1_ctr = 0
        else:
            e1_ctr = np.sqrt(tg/T1_ctr)

        if T2_ctr == 0:
            ep_ctr = 0
        else:
            e2_ctr = np.sqrt(tg/T2_ctr)
            ep_ctr_squared = (1/2) * (e2_ctr**2 - e1_ctr**2/2)
                
            if ep_ctr_squared < 0:
                # T2_ctr > 2*T1_ctr - clamp to zero
                ep_ctr = 0
            else:
                ep_ctr = np.sqrt(ep_ctr_squared)

        if T1_trg == 0:
            e1_trg = 0
        else:
            e1_trg = np.sqrt(tg/T1_trg)

        if T2_trg == 0:
            ep_trg = 0
        else:
            e2_trg = np.sqrt(tg/T2_trg)
            ep_trg_squared = (1/2) * (e2_trg**2 - e1_trg**2/2)
            
            if ep_trg_squared < 0:
                # T2_trg > 2*T1_trg - clamp to zero
                ep_trg = 0
            else:
                ep_trg = np.sqrt(ep_trg_squared)

        U = np.array(
            [[np.cos(theta/2), -1J*np.sin(theta/2) * np.exp(-1J * phi), 0, 0],
             [-1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2), 0, 0],
             [0, 0, np.cos(theta/2), 1J*np.sin(theta/2) * np.exp(-1J * phi)],
             [0, 0, 1J*np.sin(theta/2) * np.exp(1J * phi), np.cos(theta/2)]]
        )

        """ 1) RELAXATION CONTRIBUTIONS """

        # Variances and covariances for amplitude damping Itô processes depending on [tensor(sigma_min,ID)](t)
        Ir_ctr_1, Ir_ctr_2 = self._ito_integrals_for_depolarization_process(omega, phi, a)

        Ir_ctr = e1_ctr * np.array(
            [[0, 0, Ir_ctr_1, 1J*Ir_ctr_2 * np.exp(-1J * phi)],
             [0, 0, 1J*Ir_ctr_2 * np.exp(1J * phi), Ir_ctr_1],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        )

        # Variances and covariances for amplitude damping Itô processes depending on [tensor(ID,sigma_min)](t)
        Ir_trg_1, Ir_trg_2, Wr_trg = self._ito_integrals_for_depolarization_process_reversed_tensor(omega, a)
        Ir_trg = e1_trg * np.array(
            [[-1J*(1/2)*Ir_trg_1*np.exp(1J*phi), Wr_trg-Ir_trg_2, 0, 0],
             [Ir_trg_2*np.exp(2*1J*phi), 1J*(1/2)*Ir_trg_1*np.exp(1J*phi), 0, 0],
             [0, 0, 1J*(1/2)*Ir_trg_1*np.exp(1J*phi),Wr_trg-Ir_trg_2],
             [0, 0, Ir_trg_2*np.exp(2*1J*phi), -1J*(1/2)*Ir_trg_1*np.exp(1J*phi)]]
        )

        # Variances and covariances for phase damping Itô processes depending on [tensor(Z,ID)](t)
        Wp_ctr = np.random.normal(0, np.sqrt(a))
        Ip_ctr = ep_ctr * np.array(
            [[Wp_ctr, 0, 0, 0],
             [0, Wp_ctr, 0, 0],
             [0, 0, -Wp_ctr, 0],
             [0, 0, 0, -Wp_ctr]]
        )

        # Variances and covariances for phase damping Itô processes depending on [tensor(ID,Z)](t)
        Ip_trg_1, Ip_trg_2 = self._ito_integrals_for_depolarization_process(omega, phi, a)
        Ip_trg = ep_trg * np.array(
            [[Ip_trg_1, -1J*Ip_trg_2*np.exp(-1J*phi), 0, 0],
             [1J*Ip_trg_2*np.exp(1J*phi), -Ip_trg_1, 0, 0],
             [0, 0, Ip_trg_1, 1J*Ip_trg_2*np.exp(-1J*phi)],
             [0, 0, -1J*Ip_trg_2*np.exp(1J*phi), -Ip_trg_1]]
        )

        #Deterministic contribution given by relaxation
        det1 = (a*omega-a*np.sin(omega))/(2*omega)
        det2 = (a/omega)*(1-np.cos(omega))
        det3 = a/(2*omega)*(omega+np.sin(omega))

        deterministic_r_ctr = -e1_ctr**2/2 * np.array([[0,0,0,0],[0,0,0,0],[0,0,a,0],[0,0,0,a]])
        deterministic_r_trg = -e1_trg**2/2 * np.array(
            [[det1,1J*(1/2)*det2*np.exp(-1J*phi),0,0],
             [-1J*(1/2)*det2*np.exp(1J*phi),det3,0,0],
             [0,0,det1,-1J*(1/2)*det2*np.exp(-1J*phi)],[0,0,1J*(1/2)*det2*np.exp(1J*phi),det3]]
        )

        """ 2) DEPOLARIZATION CONTRIBUTIONS """

        # Variances and covariances for depolarization Itô processes depending on [tensor(X,ID)](t)
        Idx_ctr_1, Idx_ctr_2 = self._ito_integrals_for_depolarization_process(omega, phi, a)
        Idx_ctr = ed_cr * np.array(
            [[0, 0, Idx_ctr_1, 1J*Idx_ctr_2 * np.exp(-1J * phi)],
             [0, 0, 1J*Idx_ctr_2 * np.exp(1J * phi), Idx_ctr_1],
             [Idx_ctr_1, -1J*Idx_ctr_2 * np.exp(-1J * phi), 0, 0],
             [-1J*Idx_ctr_2 * np.exp(1J * phi), Idx_ctr_1, 0, 0]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(Y,ID)](t)
        Idy_ctr_1, Idy_ctr_2 = self._ito_integrals_for_depolarization_process(omega, phi, a)
        Idy_ctr = ed_cr * np.array(
            [[0, 0, -1J*Idy_ctr_1, Idy_ctr_2 * np.exp(-1J * phi)],
             [0, 0, Idy_ctr_2 * np.exp(1J * phi), -1J*Idy_ctr_1],
             [1J*Idy_ctr_1, Idy_ctr_2 * np.exp(-1J * phi), 0, 0],
             [Idy_ctr_2 * np.exp(1J * phi), 1J*Idy_ctr_1, 0, 0]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(Z,ID)](t)
        Wdz_ctr = np.random.normal(0, np.sqrt(a))
        Idz_ctr = ed_cr * np.array(
            [[Wdz_ctr, 0, 0, 0],
             [0, Wdz_ctr, 0, 0],
             [0, 0, -Wdz_ctr, 0],
             [0, 0, 0, -Wdz_ctr]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(ID,X)](t)
        Idx_trg_1, Idx_trg_2, Wdx_trg = self._ito_integrals_for_depolarization_process_reversed_tensor(omega, a)

        Idx_trg = ed_cr * np.array(
            [[Idx_trg_1 * np.sin(phi), Wdx_trg + (np.exp(-2*1J*phi)-1)*Idx_trg_2, 0, 0],
             [Wdx_trg + (np.exp(2*1J*phi)-1)*Idx_trg_2, -Idx_trg_1*np.sin(phi), 0, 0],
             [0,  0, -Idx_trg_1 * np.sin(phi), Wdx_trg + (np.exp(-2*1J*phi)-1)*Idx_trg_2],
             [0, 0, Wdx_trg + (np.exp(2*1J*phi)-1)*Idx_trg_2, Idx_trg_1 * np.sin(phi)]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(ID,Y)](t)
        Idy_trg_1, Idy_trg_2,  Wdy_trg = self._ito_integrals_for_depolarization_process_reversed_tensor(omega, a)
        Idy_trg = ed_cr * np.array(
            [[-Idy_trg_1*np.cos(phi), -1J*Wdy_trg + 1J * (np.exp(-2*1J*phi)+1)*Idy_trg_2, 0, 0],
             [1J*Wdy_trg - 1J * (np.exp(2*1J*phi)+1)*Idy_trg_2, Idy_trg_1*np.cos(phi), 0, 0],
             [0, 0, Idy_trg_1*np.cos(phi), -1J*Wdy_trg + 1J * (np.exp(-2*1J*phi)+1)*Idy_trg_2],
             [0, 0, 1J*Wdy_trg - 1J * (np.exp(2*1J*phi)+1)*Idy_trg_2, -Idy_trg_1*np.cos(phi)]]
        )

        # Variances and covariances for depolarization Itô processes depending on [tensor(ID,Z)](t)
        Idz_trg_1, Idz_trg_2 = self._ito_integrals_for_depolarization_process(omega, phi, a)
        Idz_trg = ed_cr * np.array(
            [[Idz_trg_1, -1J*Idz_trg_2*np.exp(-1J*phi), 0, 0],
             [1J*Idz_trg_2*np.exp(1J*phi), -Idz_trg_1, 0, 0],
             [0, 0, Idz_trg_1, 1J*Idz_trg_2*np.exp(-1J*phi)],
             [0, 0, -1J*Idz_trg_2*np.exp(1J*phi), -Idz_trg_1]]
        )

        """ 4) COMBINE CONTRIBUTIONS """
        result = U @ scipy.linalg.expm(deterministic_r_ctr + deterministic_r_trg) \
                 @ scipy.linalg.expm(
            1J * Ir_ctr + 1J * Ir_trg
            + 1J * Ip_ctr + 1J * Ip_trg
            + 1J * Idx_ctr + 1J * Idy_ctr + 1J * Idz_ctr
            + 1J * Idx_trg + 1J * Idy_trg + 1J * Idz_trg
        )
        return result

    def _ito_integrals_for_depolarization_process(self, omega, phi, a) -> tuple[float]:
        """ Ito integrals.

         Used for the depolarization Itô processes depending on one of
            * [tensor(ID,Z)](t)
            * [tensor(X,ID)](t)
            * [tensor(Y,ID)](t)
            * [tensor(sigma_min,ID)](t)

        As illustration, we leave the variable names from the version with [tensor(ID,Z)](t).

        Args:
            omega: integral of theta from t0 to t1.
            phi: phase of the drive defining axis of rotation on the Bloch sphere.
            a: fraction representing CR gate time / gate time.

        Returns:
            Tuple of floats representing sampled results of the Ito integrals.
        """

        # Integral of cos(omega/a)**2
        Vp_trg_1 = self.integrator.integrate("cos(theta/a)**2", omega, a)

        # Integral of sin(omega/a)**2
        Vp_trg_2 = self.integrator.integrate("sin(theta/a)**2", omega, a)

        # Integral of sin(omega/a)*cos(omega/a)
        Covp_trg_12 = self.integrator.integrate("sin(theta/a)*cos(theta/a)", omega, a)

        # Mean and covariance
        meanp_trg = [0, 0]
        covp_trg = [[Vp_trg_1, Covp_trg_12], [Covp_trg_12, Vp_trg_2]]

        # Sample
        sample_p_trg = np.random.multivariate_normal(meanp_trg, covp_trg, 1)
        Ip_trg_1 = sample_p_trg[0,0]
        Ip_trg_2 = sample_p_trg[0,1]

        return Ip_trg_1, Ip_trg_2

    def _ito_integrals_for_depolarization_process_reversed_tensor(self, omega, a) -> tuple[float]:
        """ Ito integrals.

        Used for the depolarization Itô processes depending on one of
            * [tensor(ID,X)](t)
            * [tensor(ID,Y)](t)

        As illustration, we leave the variable names from the version with [tensor(ID,Y)](t).

        Args:
            omega (float): Integral of theta from t0 to t1.
            a (float): Fraction representing CR gate time / gate time.

        Returns:
            Tuple of floats representing sampled results of the Ito integrals.
        """

        # Integral of sin**2(omega/a)
        Vdy_trg_1 = self.integrator.integrate("sin(theta/a)**2", omega, a)

        # Integral of sin(omega/(2*a))**4
        Vdy_trg_2 = self.integrator.integrate("sin(theta/(2*a))**4", omega, a)

        # Integral of sin(omega/a) sin**2(omega/(2*a))
        Covdy_trg_12 = self.integrator.integrate("sin(theta/a)*sin(theta/(2*a))**2", omega, a)

        # Integral of sin(omega/a)
        Covdy_trg_1Wdy = self.integrator.integrate("sin(theta/a)", omega, a)

        # Integral of sin(omega/(2*a))**2
        Covdy_trg_2Wdy = self.integrator.integrate("sin(theta/(2*a))**2", omega, a)

        meandy_trg = np.array([0, 0, 0])
        covdy_trg = np.array(
            [[Vdy_trg_1, Covdy_trg_12, Covdy_trg_1Wdy],
             [Covdy_trg_12, Vdy_trg_2, Covdy_trg_2Wdy],
             [Covdy_trg_1Wdy, Covdy_trg_2Wdy, a]]
        )

        # The variance of Wdy is a
        sample_dy_trg = np.random.multivariate_normal(meandy_trg, covdy_trg, 1)

        Idy_trg_1 = sample_dy_trg[0,0]
        Idy_trg_2 = sample_dy_trg[0,1]
        Wdy_trg = sample_dy_trg[0,2]

        return Idy_trg_1, Idy_trg_2,  Wdy_trg


class CNOTFactory(object):
    """Factory for constructing noisy CNOT gates.

    Args:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.

    Attributes:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.
        cr_c (CRFactory): Instance of the factory for creating CR gates.
        single_qubit_gate_c (SingelQubitGateFactory): Instance of the factory for creating general single qubit gates.
        x_c (XFactory): Instance of the factory for creating X gates.
        sx_c (XFactory): Instance of the factory for creating SX gates.
        relaxation_c (RelaxationFactory): Instance of the factory for creating gates for relaxation.
    """

    def __init__(self, integrator):
        self.integrator = integrator

        # Factories
        self.cr_c = CRFactory(self.integrator)
        self.single_qubit_gate_c = SingleQubitGateFactory(self.integrator)
        self.x_c = XFactory(self.integrator)
        self.sx_c = SXFactory(self.integrator)
        self.relaxation_c = RelaxationFactory()

    def construct(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg,
                  T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        """Generates a noisy CNOT gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the CNOT two-qubit noisy quantum
        gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            phi_ctr (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            phi_trg (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_cnot (float): CNOT gate time in ns.
            p_cnot (float): CNOT depolarizing error probability.
            p_single_ctr (float): Control qubit depolarizing error probability.
            p_single_trg (float): Target qubit depolarizing error probability.
            T1_ctr (float): Control qubit's amplitude damping time in ns.
            T2_ctr (float): Control qubit's dephasing time in ns.
            T1_trg (float): Target qubit's amplitude damping time in ns.
            T2_trg (float): Target qubit's dephasing time in ns.
            pulse_parametrization (callable): None or function that parametrized the pulse (None or callable)

        Returns:
              Array representing a CNOT two-qubit noisy quantum gate.
        """
        # Constants
        tg = 35*10**(-9)
        t_cr = t_cnot/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)))))

        # Sample gates
        first_cr = self.cr_c.construct(-np.pi/4, -phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        second_cr = self.cr_c.construct(np.pi/4, -phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        x_gate = self.x_c.construct(-phi_ctr+np.pi/2, p_single_ctr, T1_ctr, T2_ctr)
        sx_gate = self.sx_c.construct(-phi_trg, p_single_trg, T1_trg, T2_trg)
        relaxation_gate = self.relaxation_c.construct(tg, T1_trg, T2_trg)
        Y_Rz = self.single_qubit_gate_c.construct(-np.pi, -phi_ctr + np.pi/2 + np.pi/2, p_single_ctr, T1_ctr, T2_ctr)

        result = first_cr @ np.kron(x_gate, relaxation_gate) @ second_cr @ np.kron(Y_Rz, sx_gate)
        return result


class CNOTInvFactory(object):
    """Factory for constructing noisy inverse CNOT gates.

    Args:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.

    Attributes:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.
        cr_c (CRFactory): Instance of the factory for creating CR gates.
        single_qubit_gate_c (SingelQubitGateFactory): Instance of the factory for creating general single qubit gates.
        x_c (XFactory): Instance of the factory for creating X gates.
        sx_c (XFactory): Instance of the factory for creating SX gates.
        relaxation_c (RelaxationFactory): Instance of the factory for creating gates for relaxation.
    """

    def __init__(self, integrator):
        self.integrator = integrator

        # Factories
        self.cr_c = CRFactory(self.integrator)
        self.single_qubit_gate_c = SingleQubitGateFactory(self.integrator)
        self.x_c = XFactory(self.integrator)
        self.sx_c = SXFactory(self.integrator)
        self.relaxation_c = RelaxationFactory()

    def construct(self, phi_ctr, phi_trg, t_cnot, p_cnot, p_single_ctr, p_single_trg, T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        """Generates a reverse CNOT gate of IBM devices.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the reverse CNOT two-qubit noisy
        quantum gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            phi_ctr (float): control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            phi_trg (float): target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_cnot (float): reverse CNOT gate time in ns.
            p_cnot (float): reverse CNOT depolarizing error probability.
            p_single_ctr (float): control qubit depolarizing error probability.
            p_single_trg (float): target qubit depolarizing error probability.
            T1_ctr (float): control qubit's amplitude damping time in ns.
            T2_ctr (float): control qubit's dephasing time in ns.
            T1_trg (float): target qubit's amplitude damping time in ns.
            T2_trg (float): target qubit's dephasing time in ns.
            pulse_parametrization (callable): None or function that parametrized the pulse.

        Returns:
               Array representing the reverse CNOT two-qubit noisy quantum gate.
        """
        # Constants
        tg = 35*10**(-9)
        t_cr = (t_cnot-3*tg)/2
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_cnot)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)**3))))

        # Sample gates
        Ry = self.single_qubit_gate_c.construct(-np.pi/2, -phi_trg-np.pi/2+np.pi/2, p_single_trg, T1_trg, T2_trg)
        Y_Z = self.single_qubit_gate_c.construct(np.pi/2, -phi_ctr-np.pi+np.pi/2, p_single_ctr, T1_ctr, T2_ctr)
        first_sx_gate = self.sx_c.construct(-phi_ctr - np.pi - np.pi/2, p_single_ctr, T1_ctr, T2_ctr)
        second_sx_gate = self.sx_c.construct(-phi_trg - np.pi/2, p_single_ctr, T1_ctr, T2_ctr)
        first_cr = self.cr_c.construct(-np.pi/4, -phi_ctr-np.pi, t_cr, p_cr, T1_trg, T2_trg, T1_ctr, T2_ctr)
        second_cr = self.cr_c.construct(np.pi/4, -phi_ctr-np.pi, t_cr, p_cr, T1_trg, T2_trg, T1_ctr, T2_ctr)
        x_gate = self.x_c.construct(-phi_trg-np.pi/2, p_single_trg, T1_trg, T2_trg)
        relaxation_gate = self.relaxation_c.construct(tg, T1_ctr, T2_ctr)

        result = np.kron(Ry, first_sx_gate) @ first_cr @ np.kron(x_gate, relaxation_gate) @ second_cr @ np.kron(second_sx_gate, Y_Z)
        return result


class ECRFactory(object):
    """Factory for constructing noisy ECR gates.

    Args:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.

    Attributes:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.
        cr_c (CRFactory): Instance of the factory for creating CR gates.
        x_c (XFactory): Instance of the factory for creating X gates.
        relaxation_c (RelaxationFactory): Instance of the factory for creating gates for relaxation.
    """


    def __init__(self, integrator):
        self.integrator = integrator

        # Factories
        self.cr_c = CRFactory(self.integrator)
        self.x_c = XFactory(self.integrator)
        self.relaxation_c = RelaxationFactory()

    
    def construct(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg,
                  T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        """Generates a noisy ECR gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the ECR two-qubit noisy quantum
        gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            phi_ctr (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            phi_trg (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_ecr (float): ECR gate time in ns.
            p_ecr (float): ECR depolarizing error probability.
            p_single_ctr (float): Control qubit depolarizing error probability.
            p_single_trg (float): Target qubit depolarizing error probability.
            T1_ctr (float): Control qubit's amplitude damping time in ns.
            T2_ctr (float): Control qubit's dephasing time in ns.
            T1_trg (float): Target qubit's amplitude damping time in ns.
            T2_trg (float): Target qubit's dephasing time in ns.
            pulse_parametrization (callable): None or function that parametrized the pulse (None or callable)

        Returns:
              Array representing a ECR two-qubit noisy quantum gate.
        """
        # Constants
        tg = 35*10**(-9)
        t_cr = t_ecr/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_ecr)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)))))

        # Sample gates
        first_cr = self.cr_c.construct(np.pi/4, np.pi-phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        second_cr = self.cr_c.construct(-np.pi/4, np.pi-phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        x_gate = -1J* self.x_c.construct(np.pi-phi_ctr, p_single_ctr, T1_ctr, T2_ctr)
        relaxation_gate = self.relaxation_c.construct(tg, T1_trg, T2_trg)
        
        result = (first_cr @ np.kron(x_gate , relaxation_gate) @ second_cr )
        return result


class ECRInvFactory(object):
    """Factory for constructing noisy inverse ECR gates.

    Args:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.

    Attributes:
        integrator (Integrator): Used to perform the Ito integrals, hides the pulse waveform information.
        cr_c (CRFactory): Instance of the factory for creating CR gates.
        single_qubit_gate_c (SingelQubitGateFactory): Instance of the factory for creating general single qubit gates.
        x_c (XFactory): Instance of the factory for creating X gates.
        sx_c (XFactory): Instance of the factory for creating SX gates.
        relaxation_c (RelaxationFactory): Instance of the factory for creating gates for relaxation.

    """

    def __init__(self, integrator):
        self.integrator = integrator

        # Factories
        self.cr_c = CRFactory(self.integrator)
        self.x_c = XFactory(self.integrator)
        self.sx_c = SXFactory(self.integrator)
        self.single_qubit_gate_c = SingleQubitGateFactory(self.integrator)
        self.relaxation_c = RelaxationFactory()

    def construct(self, phi_ctr, phi_trg, t_ecr, p_ecr, p_single_ctr, p_single_trg,
                  T1_ctr, T2_ctr, T1_trg, T2_trg) -> np.array:
        """Generates a noisy inverse ECR gate.

        This is a 2nd order approximated solution, a non-unitary matrix. It implements the reverse ECR two-qubit noisy quantum
        gate with depolarizing and relaxation errors on both qubits during the unitary evolution.

        Args:
            phi_ctr (float): Control qubit phase of the drive defining axis of rotation on the Bloch sphere.
            phi_trg (float): Target qubit phase of the drive defining axis of rotation on the Bloch sphere.
            t_ecr (float): ECR gate time in ns.
            p_ecr (float): ECR depolarizing error probability.
            p_single_ctr (float): Control qubit depolarizing error probability.
            p_single_trg (float): Target qubit depolarizing error probability.
            T1_ctr (float): Control qubit's amplitude damping time in ns.
            T2_ctr (float): Control qubit's dephasing time in ns.
            T1_trg (float): Target qubit's amplitude damping time in ns.
            T2_trg (float): Target qubit's dephasing time in ns.
            pulse_parametrization (callable): None or function that parametrized the pulse (None or callable)

        Returns:
              Array representing a reverse ECR two-qubit noisy quantum gate.
        """
        # Constants
        tg = 35*10**(-9)
        t_cr = t_ecr/2-tg
        p_cr = (4/3) * (1 - np.sqrt(np.sqrt((1 - (3/4) * p_ecr)**2 / ((1-(3/4)*p_single_ctr)**2 * (1-(3/4)*p_single_trg)))))

        # Sample gates
        first_cr = self.cr_c.construct(np.pi/4, np.pi-phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        second_cr = self.cr_c.construct(-np.pi/4, np.pi-phi_trg, t_cr, p_cr, T1_ctr, T2_ctr, T1_trg, T2_trg)
        x_gate = -1J* self.x_c.construct(np.pi-phi_ctr, p_single_ctr, T1_ctr, T2_ctr)
        relaxation_gate = self.relaxation_c.construct(tg, T1_trg, T2_trg)

        sx_gate_ctr_1 =  self.sx_c.construct(-np.pi/2-phi_ctr, p_single_ctr, T1_ctr, T2_ctr)
        sx_gate_trg_1 =  self.sx_c.construct(-np.pi/2-phi_trg, p_single_trg, T1_trg, T2_trg)

        sx_gate_ctr_2 =  self.sx_c.construct(-np.pi/2-phi_ctr, p_single_ctr, T1_ctr, T2_ctr)
        sx_gate_trg_2 =  self.sx_c.construct(-np.pi/2-phi_trg, p_single_trg, T1_trg, T2_trg)

        result = 1j * np.kron(sx_gate_ctr_1, sx_gate_trg_1) @ (first_cr @ np.kron(x_gate , relaxation_gate) @ second_cr ) @ np.kron(sx_gate_ctr_2, sx_gate_trg_2)

        return result
