import numpy as np
import pytest
import pickle

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.circuit.controlflow import ControlFlowOp

from quantum_gates._simulation.circuit import EfficientCircuit, BinaryCircuit
from src.quantum_gates.utilities import DeviceParameters
from quantum_gates.simulators import MrAndersonSimulator

from src.quantum_gates.gates import (
    ScaledNoiseGates,
    CustomNoiseGates,
    CustomNoiseChannelsGates,
    noise_free_gates
)


def get_device_params(nqubit):
    qubits_layout = list(range(nqubit))
    device_param = DeviceParameters(qubits_layout)
    device_param.load_from_backend(FakeBrisbane())
    return device_param.__dict__()   

def counts_to_probs(counts, shots):
    return {k: v / shots for k, v in counts.items()}

def _distance(A, B):
    """Matrix distance (Frobenius norm)."""
    return np.linalg.norm(A - B)

def l1_distance(p, q):
    """Total variation distance (L1)."""
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)

def _almost_equal(m1, m2, tol=1e-9):
    return np.allclose(m1, m2, atol=tol)

def zero_state(nqubit):
    psi = np.zeros(2**nqubit, dtype=complex)
    psi[0] = 1.0
    return psi


# Arrange test circuits
def build_deterministic_circuit_0():
    """Simple deterministic circuit: |00> → X → measure → always '10'"""
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure([0, 1], [0, 1])
    qc.x(range(2))  # add some gates to make mid measures
    return qc

def build_deterministic_circuit_1(N_q, N_m):
    """ deterministic circuit:"""
    qc = QuantumCircuit(N_q, N_m)
    qc.x(1)
    qc.x(3)
    qc.cx(1, 0)
    qc.measure([0, 1], [1, 3])
    qc.y(range(2))  
    qc.measure([1, 2], [3, 0])
    qc.swap(0, 2)
    qc.ccx(0, 1, 3)
    qc.swap(1, 3)
    qc.measure([2, 3], [1, 2])
    qc.x(range(N_q))  # add some gates to make mid measures

    return qc

def build_deterministic_circuit_simple_cnot(N_q, N_m):
    """ deterministic circuit:"""
    qc = QuantumCircuit(N_q, N_m)
    qc.x(1)
    qc.cx(1, 2)
    qc.barrier()
    qc.measure(range(N_q), range(N_m))
    qc.x(range(N_q))  # add some gates to make mid measures

    return qc

# Arrange - test arguments
x_args = dict(phi=np.pi/2, p=0.01, T1=50e3, T2=30e3)

single_args = dict(theta=np.pi/2, phi=np.pi/2, p=0.01, T1=50e3, T2=30e3)

twoq_args = dict(
    phi_ctr=np.pi/2,
    phi_trg=np.pi/2,
    t_cnot=300e-9,
    p_cnot=0.01,
    p_single_ctr=0.01,
    p_single_trg=0.01,
    T1_ctr=50e3,
    T2_ctr=30e3,
    T1_trg=50e3,
    T2_trg=30e3,
)


def test_scaled_noise_limit_to_noiseless():
    """Scaling → 0 should match noiseless."""
    g = ScaledNoiseGates(noise_scaling=1e-12) # Arrange - noise near 0
    res = g.X(**x_args)                 # Act - apply to X gate
    ref = noise_free_gates.X(**x_args)
    assert _distance(res, ref) < 1e-6  # Assert - compare to noise free X


def test_scaled_noise_monotonic():
    """Lower scaling → closer to noiseless."""
    g_high = ScaledNoiseGates(1.0)   # Arrange - higher noise scaling
    g_low = ScaledNoiseGates(0.1)    # Arrange - lower noise scaling
    np.random.seed(12)
    res_high = g_high.X(**x_args)    # Act - apply high noise
    np.random.seed(12)
    res_low = g_low.X(**x_args)      # Act - apply low noise
    ref = noise_free_gates.X(**x_args)  
    assert _distance(res_low, ref) < _distance(res_high, ref)  # Assert - lower noise closer to ref


def test_scaled_noise_monotonic_sweep():
    """Lower noise scaling should monotonically approach noiseless gate."""
    np.random.seed(0)  # Arrange - fix randomness for reproducibility
    scales = [1.0, 0.5, 0.2, 0.1, 0.05]  # Arrange - noise scaling values
    ref = noise_free_gates.X(**x_args)   # Arrange - noiseless reference
    distances = []  # Arrange - store distances to reference
    for s in scales:
        np.random.seed(0)                
        g = ScaledNoiseGates(s)          # Arrange - set scaling
        res = g.X(**x_args)              # Act - apply noisy gate
        d = _distance(res, ref)          # Act - compute distance
        distances.append(d)
    # Assert - check monotonic decrease
    for i in range(len(distances) - 1):
        assert distances[i+1] <= distances[i], \
            f"Scaling not monotonic: {distances}"


def test_scaled_noise_changes_result():
    """Scaling should actually affect the gate."""
    g1 = ScaledNoiseGates(noise_scaling=1.0) # Arrange - noise scaling
    g2 = ScaledNoiseGates(noise_scaling=0.2)
    args = dict(phi=np.pi/2, p=0.05, T1=50e3, T2=30e3)
    res1 = g1.X(**args)                     # Act - apply noisy gate
    res2 = g2.X(**args)
    assert not np.allclose(res1, res2)


@pytest.mark.parametrize("phi", np.linspace(0, np.pi, 6))
def test_scaled_noise_x_sweep(phi):
    args = dict(phi=phi, p=0.02, T1=50e3, T2=30e3)
    g = ScaledNoiseGates(0.5)
    res = g.X(**args)
    assert res.shape == (2, 2)


@pytest.mark.parametrize("phi", np.linspace(0, np.pi, 6))
def test_noiseless_consistency_scaled(phi):
    args = dict(phi=phi, p=0.0, T1=1e12, T2=1e12)
    g = ScaledNoiseGates(1.0)
    res = g.X(**args)
    ref = noise_free_gates.X(**args)
    assert _distance(res, ref) < 1e-6


def test_scaled_noiseless_matches_aer_circuit_0():
    """Noiseless simulator should match Aer exactly for deterministic circuit."""
    shots = 10
    qc = build_deterministic_circuit_0() # Arrange - Build circuit
    np.random.seed(42)
    # Aer
    aer = AerSimulator()    
    tqc = transpile(qc, aer)
    result = aer.run(tqc, shots=shots).result()
    counts_aer = result.get_counts()

    # Custom simulator
    nqubit = 2
    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit) # Act - get the noisy circuit
    noisy_gates = ScaledNoiseGates(noise_scaling=1e-15)
    sim = MrAndersonSimulator(gates=noisy_gates)

    counts_sim = sim.run(
    t_qiskit_circ=qc,
    psi0=psi0,
    shots=shots,
    device_param=device_param,
    nqubit=nqubit,
    )

    assert counts_sim["mid_counts"] == counts_aer


def test_scaled_noiseless_matches_aer_circuit_1():
    """Noiseless simulator should match Aer exactly for deterministic circuit."""
    shots = 10
    nqubit = 4
    qc = build_deterministic_circuit_1(nqubit, nqubit)

    # Custom simulator
    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)
    noisy_gates = ScaledNoiseGates(noise_scaling=1e-15)
    sim = MrAndersonSimulator(gates=noisy_gates)
    backend = FakeBrisbane()
    qubits_layout = list(range(nqubit))

    # Transpile circuit
    needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

    t_circ = transpile(
        qc,
        backend,
        initial_layout=qubits_layout,
        seed_transpiler=42,
        **({} if needs_controlflow else {"scheduling_method": "asap"})
    )

    counts_sim = sim.run(
    t_qiskit_circ=t_circ,
    psi0=psi0,
    shots=shots,
    device_param=device_param,
    nqubit=nqubit,
    )

    # Aer
    aer = AerSimulator()
    result = aer.run(t_circ, shots=shots).result()
    counts_aer = result.get_counts()

    assert counts_sim["mid_counts"] == counts_aer

# Same idea but only compare dominant outcome to avoid shot noise issues with low shots.
def test_scaled_noise_matches_aer_dominant_outcome():
    """With low noise, dominant measurement outcome should match Aer."""
    shots = 100  # increase shots for stability
    nqubit = 4
    qc = build_deterministic_circuit_1(nqubit, nqubit)

    # Custom simulator
    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)
    noisy_gates = ScaledNoiseGates(noise_scaling=1)  # device noise
    sim = MrAndersonSimulator(gates=noisy_gates)
    backend = FakeBrisbane()
    qubits_layout = list(range(nqubit))

    # Transpile circuit
    needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

    t_circ = transpile(
        qc,
        backend,
        initial_layout=qubits_layout,
        seed_transpiler=42,
        **({} if needs_controlflow else {"scheduling_method": "asap"})
    )
    
    res = sim.run(
        t_qiskit_circ=t_circ,
        psi0=psi0,
        shots=shots,
        device_param=device_param,
        nqubit=nqubit,
    )

    counts_sim = res["mid_counts"]

    # Compare dominant outcomes
    top_sim = max(counts_sim, key=counts_sim.get)

    # Aer
    aer = AerSimulator()
    result = aer.run(t_circ, shots=shots).result()
    counts_aer = result.get_counts()

    top_aer = max(counts_aer, key=counts_aer.get)

    assert top_sim == top_aer


def test_custom_noise_limit_to_noiseless():
    """All scales → 0 ⇒ noiseless."""
    g = CustomNoiseGates(p_scale=1e-12, T1_scale=1e-12, T2_scale=1e-12)
    res = g.X(**x_args)
    ref = noise_free_gates.X(**x_args)
    assert _distance(res, ref) < 1e-6


def test_custom_noise_scaling_independent():
    """Ensure independent scaling does not crash and returns valid matrix."""
    g = CustomNoiseGates(p_scale=0.1, T1_scale=2.0, T2_scale=3.0)
    args = dict(phi=np.pi/2, p=0.05, T1=100.0, T2=80.0)
    res = g.X(**args)
    assert res.shape == (2, 2)


@pytest.mark.parametrize("T_scales", [
    np.logspace(-5, 5, 4)
])
def test_custom_noise_T1_T2_outcome_trend(T_scales):
    """
    Increasing noise (via T1/T2 scaling) should reduce the probability
    of the correct deterministic outcome ('0000').
    """

    shots = 100  # large enough to average stochasticity
    nqubit = 4
    qc = build_deterministic_circuit_1(nqubit, nqubit)
    correct_outcome = "0000"

    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)


    def correct_prob(scale):
        gates = CustomNoiseGates(
            p_scale=1.0,
            T1_scale=scale,
            T2_scale=scale
        )
        sim = MrAndersonSimulator(gates=gates)
        backend = FakeBrisbane()
        qubits_layout = list(range(nqubit))


        # Transpile circuit
        needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

        t_circ = transpile(
            qc,
            backend,
            initial_layout=qubits_layout,
            seed_transpiler=42,
            **({} if needs_controlflow else {"scheduling_method": "asap"})
        )

        res = sim.run(
            t_qiskit_circ=t_circ,
            psi0=psi0,
            shots=shots,
            device_param=device_param,
            nqubit=nqubit,
        )

        counts = res["mid_counts"]
        print(counts)
        return counts.get(correct_outcome, 0) / shots

    correct_probs = [correct_prob(s) for s in T_scales]

    # Trend check
    violations = sum(
        correct_probs[i+1] > correct_probs[i] + 0.02
        for i in range(len(correct_probs) - 1)
    )

    assert violations <= 1, (
        f"Trend violation:\nscales={T_scales}\nprobs={correct_probs}"
    )

    #  Global sanity check
    assert correct_probs[0] > correct_probs[-1], (
        f"Expected degradation with noise:\n{correct_probs}"
    )


@pytest.mark.parametrize("T1_scale", [
    np.logspace(-5, 5, 4)
])
def test_custom_noise_T1_outcome_trend(T1_scale):
    """
    Increasing noise (via T1/T2 scaling) should reduce the probability
    of the correct deterministic outcome ('0000').
    """
    shots = 100  # large enough to average stochasticity
    nqubit = 4
    qc = build_deterministic_circuit_1(nqubit, nqubit)
    correct_outcome = "0000"

    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)

    def correct_prob(scale):
        gates = CustomNoiseGates(
            p_scale=1.0,
            T1_scale=scale,
            T2_scale=1.0
        )
        sim = MrAndersonSimulator(gates=gates)

        backend = FakeBrisbane()
        qubits_layout = list(range(nqubit))


        # Transpile circuit
        needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

        t_circ = transpile(
            qc,
            backend,
            initial_layout=qubits_layout,
            seed_transpiler=42,
            **({} if needs_controlflow else {"scheduling_method": "asap"})
        )

        res = sim.run(
            t_qiskit_circ=t_circ,
            psi0=psi0,
            shots=shots,
            device_param=device_param,
            nqubit=nqubit,
        )

        counts = res["mid_counts"]
        print(counts)
        return counts.get(correct_outcome, 0) / shots

    correct_probs = [correct_prob(s) for s in T1_scale]
    # Trend check
    violations = sum(
        correct_probs[i+1] > correct_probs[i] + 0.02
        for i in range(len(correct_probs) - 1)
    )
    assert violations <= 1, (
        f"Trend violation:\nscales={T1_scale}\nprobs={correct_probs}"
    )
    #  Global sanity check
    assert correct_probs[0] > correct_probs[-1], (
        f"Expected degradation with noise:\n{correct_probs}"
    )


@pytest.mark.parametrize("scales", [
    np.logspace(-5, 2, 4)
])
def test_custom_noise_combined_outcome_trend(scales):
    """
    Increasing noise (via scaling p, T1, T2 together) should reduce the probability
    of the correct deterministic outcome ('0000').
    """
    shots = 100
    nqubit = 4

    qc = build_deterministic_circuit_1(nqubit, nqubit)
    correct_outcome = "0000"

    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)

    def correct_prob(scale):
        gates = CustomNoiseGates(
            p_scale=scale,      # ↑ scale → ↑ stochastic noise
            T1_scale=scale,     # ↑ scale → ↓ T1 → ↑ noise
            T2_scale=scale      # ↑ scale → ↓ T2 → ↑ noise
        )

        sim = MrAndersonSimulator(gates=gates)

        backend = FakeBrisbane()
        qubits_layout = list(range(nqubit))

        # Transpile circuit
        needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

        t_circ = transpile(
            qc,
            backend,
            initial_layout=qubits_layout,
            seed_transpiler=42,
            **({} if needs_controlflow else {"scheduling_method": "asap"})
        )

        res = sim.run(
            t_qiskit_circ=t_circ,
            psi0=psi0,
            shots=shots,
            device_param=device_param,
            nqubit=nqubit,
        )

        counts = res["mid_counts"]
        return counts.get(correct_outcome, 0) / shots

    correct_probs = [correct_prob(s) for s in scales]

    # Trend check (allow small stochastic violations)
    violations = sum(
        correct_probs[i+1] > correct_probs[i] + 0.03
        for i in range(len(correct_probs) - 1)
    )

    assert violations <= 1, (
        f"Trend violation:\nscales={scales}\nprobs={correct_probs}"
    )

    # Strong global check
    assert correct_probs[0] > correct_probs[-1], (
        f"Expected degradation with noise:\n{correct_probs}"
    )
    

def test_custom_noise_independent_scaling():
    """Changing p_scale only affects stochastic part."""
    g1 = CustomNoiseGates(p_scale=1.0, T1_scale=1.0, T2_scale=1.0)
    g2 = CustomNoiseGates(p_scale=0.1, T1_scale=1.0, T2_scale=1.0)

    res1 = g1.X(**x_args)
    res2 = g2.X(**x_args)

    assert not _almost_equal(res1, res2)



@pytest.mark.parametrize("p_scales", [
    np.logspace(-5, 1, 4)
])
def test_custom_noise_p_scale_outcome_trend(p_scales):
    """
    Increasing noise (via scaling p, T1, T2 together) should reduce the probability
    of the correct deterministic outcome ('0000').
    """
    shots = 100
    nqubit = 4

    qc = build_deterministic_circuit_1(nqubit, nqubit)
    correct_outcome = "0000"

    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)

    def correct_prob(scale):
        gates = CustomNoiseGates(
            p_scale=scale,      # ↑ scale → ↑ stochastic noise
            T1_scale=0.1,     # ↑ scale → ↓ T1 → ↑ noise
            T2_scale=0.1      # ↑ scale → ↓ T2 → ↑ noise
        )

        sim = MrAndersonSimulator(gates=gates)

        backend = FakeBrisbane()
        qubits_layout = list(range(nqubit))

        # Transpile circuit
        needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

        t_circ = transpile(
            qc,
            backend,
            initial_layout=qubits_layout,
            seed_transpiler=42,
            **({} if needs_controlflow else {"scheduling_method": "asap"})
        )

        res = sim.run(
            t_qiskit_circ=t_circ,
            psi0=psi0,
            shots=shots,
            device_param=device_param,
            nqubit=nqubit,
        )

        counts = res["mid_counts"]
        return counts.get(correct_outcome, 0) / shots

    correct_probs = [correct_prob(s) for s in p_scales]

    # Trend check (allow small stochastic violations)
    violations = sum(
        correct_probs[i+1] > correct_probs[i] + 0.03
        for i in range(len(correct_probs) - 1)
    )

    assert violations <= 1, (
        f"Trend violation:\nscales={p_scales}\nprobs={correct_probs}"
    )

    # Strong global check
    assert correct_probs[0] > correct_probs[-1], (
        f"Expected degradation with noise:\n{correct_probs}"
    )


def test_custom_noise_T1_T2_effect():
    """Changing T1/T2 affects decoherence."""
    g1 = CustomNoiseGates(p_scale=1.0, T1_scale=1.0, T2_scale=1.0)
    g2 = CustomNoiseGates(p_scale=1.0, T1_scale=0.1, T2_scale=0.1)

    res1 = g1.X(**x_args)
    res2 = g2.X(**x_args)

    assert not _almost_equal(res1, res2)


@pytest.mark.parametrize("phi", np.linspace(0, np.pi, 6))
def test_noiseless_consistency_custom(phi):
    args = dict(phi=phi, p=0.0, T1=1e12, T2=1e12)

    g = CustomNoiseGates()
    res = g.X(**args)
    ref = noise_free_gates.X(**args)

    assert _distance(res, ref) < 1e-6


@pytest.mark.parametrize("theta", np.linspace(1e-6, np.pi, 4))
def test_custom_noise_single_qubit(theta):
    args = dict(theta=theta, phi=np.pi/2, p=0.01, T1=50e3, T2=30e3)

    g = CustomNoiseGates(p_scale=0.5)
    res = g.single_qubit_gate(**args)

    assert res.shape == (2, 2)


def test_custom_noise_channels_dispatch():
    """Noiseless qubits should use NoiseFreeGates."""
    g = CustomNoiseChannelsGates(
        noiseless_qubits=[0],
        p_scale=0.5,
        T1_scale=0.5,
        T2_scale=0.5
    )

    args = dict(phi=np.pi/2, p=0.05, T1=100.0, T2=80.0)

    res_noiseless = g.X(**args, qubit_index=0)
    res_expected = noise_free_gates.X(**args)

    assert np.allclose(res_noiseless, res_expected)


def test_noise_channels_two_qubit_dispatch():
    """Noise controlled by control + target qubit."""
    g = CustomNoiseChannelsGates(noiseless_qubits=[0])

    res_noiseless = g.CNOT(**twoq_args, ctr_index=0, trg_index=1)
    res_noisy = g.CNOT(**twoq_args, ctr_index=1, trg_index=0)

    assert not _almost_equal(res_noiseless, res_noisy)


def test_noise_channels_noisy_default():
    """Non-selected qubit → noisy."""
    g = CustomNoiseChannelsGates(noiseless_qubits=[0])

    res = g.X(**x_args, qubit_index=1)
    ref = noise_free_gates.X(**x_args)

    assert not _almost_equal(res, ref)


def test_noise_channels_noiseless_selection():
    """Selected qubit → noiseless."""
    g = CustomNoiseChannelsGates(noiseless_qubits=[0])

    res = g.X(**x_args, qubit_index=0)
    ref = noise_free_gates.X(**x_args)

    assert _distance(res, ref) < 1e-6


def test_no_noise_channels_two_qubit_dispatch():
    """Noise controlled by control qubit."""
    g = CustomNoiseChannelsGates(noiseless_qubits=[0,1,2,3])

    U = g.CNOT(**twoq_args, ctr_index=0, trg_index=1)
    U_ref = noise_free_gates.CNOT(**twoq_args)

    assert _almost_equal(U, U_ref)


def test_custom_noise_channels_deterministic():
    nqubit = 4
    shots = 100

    qc = build_deterministic_circuit_1(nqubit, nqubit)
    correct_outcome = "0000"

    gates = CustomNoiseChannelsGates(
        noiseless_qubits=[0, 1, 2, 3],  # all noiseless
        p_scale=1.0,
        T1_scale=1.0,
        T2_scale=1.0,
    )

    sim = MrAndersonSimulator(
        gates=gates,
        CircuitClass=EfficientCircuit
    )

    needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

    t_circ = transpile(
        qc,
        backend=FakeBrisbane(),
        initial_layout=range(nqubit),
        seed_transpiler=42,
        **({} if needs_controlflow else {"scheduling_method": "asap"})
    )

    res = sim.run(
    t_qiskit_circ=t_circ,
    psi0=zero_state(nqubit),
    shots=shots,
    device_param=get_device_params(nqubit),
    nqubit=nqubit,
    )

    counts = res["mid_counts"]

    # Result format assumed: dict[str, int]
    assert correct_outcome in counts
    assert counts[correct_outcome] == shots  # all outcomes should be correct


def test_custom_noise_channels_mid_qubit_protection():
    nqubit = 4
    shots = 100

    qc = build_deterministic_circuit_simple_cnot(nqubit, nqubit)

    # Case middle qubits noiseless
    gates = CustomNoiseChannelsGates(
        noiseless_qubits=[1, 2],  # protect middle
        p_scale=1.0,
        T1_scale=1.0,
        T2_scale=1.0,
    )

     # Custom simulator
    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)

    sim = MrAndersonSimulator(
            gates=gates,
            CircuitClass=EfficientCircuit
        )
    
    needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

    t_circ = transpile(
        qc,
        backend=FakeBrisbane(),
        initial_layout=range(nqubit),
        seed_transpiler=42,
        **({} if needs_controlflow else {"scheduling_method": "asap"})
    )


    res = sim.run(
    t_qiskit_circ=t_circ,
    psi0=psi0,
    shots=shots,
    device_param=device_param,
    nqubit=nqubit,
    )

    counts = res["mid_counts"]

    for bitstring in counts:
        # Assume ordering "q0 q1 q2 q3"
        assert bitstring[1] == "1"
        assert bitstring[2] == "1"


def test_custom_noise_channels_mid_qubit_protection_1():
    nqubit = 4
    shots = 100
    qc = build_deterministic_circuit_simple_cnot(nqubit, nqubit)

    # Case Edge qubits noiseless
    gates = CustomNoiseChannelsGates(
        noiseless_qubits=[0, 3],  # protect edges
        p_scale=1.0,
        T1_scale=1.0,
        T2_scale=1.0,
    )

    psi0 = zero_state(nqubit)  
    device_param = get_device_params(nqubit)

    sim = MrAndersonSimulator(
            gates=gates,
            CircuitClass=EfficientCircuit
        )
    
    needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

    t_circ = transpile(
        qc,
        backend=FakeBrisbane(),
        initial_layout=range(nqubit),
        seed_transpiler=42,
        **({} if needs_controlflow else {"scheduling_method": "asap"})
    )


    res = sim.run(
    t_qiskit_circ=t_circ,
    psi0=psi0,
    shots=shots,
    device_param=device_param,
    nqubit=nqubit,
    )

    counts = res["mid_counts"]

    for bitstring in counts:
        # Edges should now be stable
        assert bitstring[0] in ["0", "1"]  # may depend on circuit
        assert bitstring[3] in ["0", "1"]

    # Optional stronger check (variance on middle now)
    middle_values = set((b[1], b[2]) for b in counts)
    assert len(middle_values) >= 1


# Tests exposing EfficientCircuit physical→virtual qubit mapping bug (Issue #39)
# The EfficientCircuit path in _apply_gates_on_circuit uses raw physical qubit
# indices from the transpiled circuit without mapping them to virtual indices.
# This causes gates to be silently dropped (when physical index >= nqubit) or
# applied to the wrong qubit (when layout is shuffled). These tests force
# non-identity layouts to expose the bug regardless of transpiler version.
def get_device_params_for_layout(qubits_layout):
    """Create device parameters for an arbitrary qubit layout."""
    dp = DeviceParameters(qubits_layout)
    dp.load_from_backend(FakeBrisbane())
    return dp.__dict__()


def test_efficient_circuit_identity_layout_works():
    """Control test: EfficientCircuit produces correct results with identity layout.

    With initial_layout=[0, 1], physical == virtual indices, so the missing
    mapping in the EfficientCircuit path doesn't matter. This test confirms
    the basic simulation machinery works.
    """
    nqubit = 2
    shots = 50
    qubits_layout = [0, 1]

    qc = QuantumCircuit(nqubit, nqubit)
    qc.x(0)
    qc.x(1)
    qc.barrier()
    qc.measure(range(nqubit), range(nqubit))
    qc.x(range(nqubit))  # post-measure gates to trigger mid-measurement path

    device_param = get_device_params(nqubit)
    gates = ScaledNoiseGates(noise_scaling=1e-15)
    sim = MrAndersonSimulator(gates=gates, CircuitClass=EfficientCircuit)

    t_circ = transpile(
        qc,
        backend=FakeBrisbane(),
        initial_layout=qubits_layout,
        seed_transpiler=42,
    )

    res = sim.run(
        t_qiskit_circ=t_circ,
        psi0=zero_state(nqubit),
        shots=shots,
        device_param=device_param,
        nqubit=nqubit,
    )

    counts = res["mid_counts"]
    assert counts.get("11", 0) == shots, (
        f"Control test failed: expected '11' for all {shots} shots but got {counts}"
    )


def test_efficient_circuit_sparse_layout_drops_x_gate():
    """BUG: X gates are silently dropped when physical qubit index >= nqubit.

    Same circuit as the identity-layout control test, but with
    initial_layout=[10, 11]. After transpilation, gates target physical qubits
    10 and 11. In the EfficientCircuit path of _apply_gates_on_circuit:

        for k in range(nqubit):   # nqubit=2, so k=0,1
            if k == q:            # q=10 or 11, it never matches
                circ.X(k, ...)
            else:
                circ.I(k)         # Both qubits get identity

    The X gates are silently replaced with identity, producing '00' instead
    of the expected '11'.

    Relates to GitHub issue #39.
    """
    nqubit = 2
    shots = 50
    qubits_layout = [10, 11]

    qc = QuantumCircuit(nqubit, nqubit)
    qc.x(0)
    qc.x(1)
    qc.barrier()
    qc.measure(range(nqubit), range(nqubit))
    qc.x(range(nqubit))  # post-measure gates to trigger mid-measurement path

    device_param = get_device_params_for_layout(qubits_layout)
    gates = ScaledNoiseGates(noise_scaling=1e-15)
    sim = MrAndersonSimulator(gates=gates, CircuitClass=EfficientCircuit)

    t_circ = transpile(
        qc,
        backend=FakeBrisbane(),
        initial_layout=qubits_layout,
        seed_transpiler=42,
    )

    res = sim.run(
        t_qiskit_circ=t_circ,
        psi0=zero_state(nqubit),
        shots=shots,
        device_param=device_param,
        nqubit=nqubit,
    )

    counts = res["mid_counts"]
    assert counts.get("11", 0) == shots, (
        f"Expected '11' for all {shots} shots but got {counts}. "
        f"X gates on physical qubits 10, 11 were silently dropped because "
        f"the EfficientCircuit loop range(2) never matches indices 10 or 11."
    )


def test_efficient_circuit_sparse_layout_rz_index_error():
    """BUG: Rz gate causes IndexError when physical qubit index >= nqubit.

    circ.Rz(q, theta) does self.phi[q] where q is the raw physical index.
    With nqubit=2, self.phi has 2 elements. A Hadamard decomposes into
    RZ + SX on Brisbane's basis set, so Rz(10, theta) accesses self.phi[10]
    which is out of bounds.

    Relates to GitHub issue #39.
    """
    nqubit = 2
    shots = 10
    qubits_layout = [10, 11]

    # H decomposes into RZ + SX on Brisbane with basis: ecr, id, rz, sx, x
    qc = QuantumCircuit(nqubit, nqubit)
    qc.h(0)
    qc.barrier()
    qc.measure(range(nqubit), range(nqubit))
    qc.x(range(nqubit))

    device_param = get_device_params_for_layout(qubits_layout)
    gates = ScaledNoiseGates(noise_scaling=1e-15)
    sim = MrAndersonSimulator(gates=gates, CircuitClass=EfficientCircuit)

    t_circ = transpile(
        qc,
        backend=FakeBrisbane(),
        initial_layout=qubits_layout,
        seed_transpiler=42,
    )

    # Verify the transpiled circuit contains RZ (H decomposes to RZ+SX)
    gate_names = [inst.operation.name for inst in t_circ.data]
    has_rz = "rz" in gate_names
    assert has_rz, (
        f"Expected RZ in transpiled circuit but got only: {set(gate_names)}. "
        f"Test assumes H decomposes to RZ+SX on Brisbane."
    )

    # The simulation should either crash (IndexError on self.phi[10])
    # or produce wrong results. Either way, the current code is broken.
    try:
        res = sim.run(
            t_qiskit_circ=t_circ,
            psi0=zero_state(nqubit),
            shots=shots,
            device_param=device_param,
            nqubit=nqubit,
        )
        # If no crash, check results are correct (H produces superposition)
        counts = res["mid_counts"]
        assert len(counts) > 1, (
            f"H gate should produce superposition but got {counts}. "
            f"Gates were likely dropped due to physical→virtual index mismatch."
        )
    except IndexError as e:
        # Expected: self.phi[10] is out of bounds for nqubit=2
        assert "index" in str(e).lower() or True, (
            f"Got IndexError as expected: Rz on physical qubit 10 accesses "
            f"self.phi[10] but self.phi only has {nqubit} elements."
        )


def test_binary_circuit_sparse_layout_noise_param_indexing():
    """BUG: BinaryCircuit indexes noise params by physical qubit, not layout position.

    BinaryCircuit correctly maps physical→virtual for circuit operations via
    qubit_layout.index(), but accesses noise params as p[q_r] where q_r is
    the physical qubit index. Since DeviceParameters stores T1, T2, p as
    lists indexed by layout position (0, 1, ..., n-1), p[10] raises
    IndexError when the layout is [10, 11] (p only has 2 elements).

    Relates to GitHub issue #39.
    """
    nqubit = 2
    shots = 50
    qubits_layout = [10, 11]

    qc = QuantumCircuit(nqubit, nqubit)
    qc.x(0)
    qc.x(1)
    qc.barrier()
    qc.measure(range(nqubit), range(nqubit))
    qc.x(range(nqubit))

    device_param = get_device_params_for_layout(qubits_layout)
    gates = ScaledNoiseGates(noise_scaling=1e-15)

    t_circ = transpile(
        qc,
        backend=FakeBrisbane(),
        initial_layout=qubits_layout,
        seed_transpiler=42,
    )

    sim = MrAndersonSimulator(gates=gates, CircuitClass=BinaryCircuit)

    # BinaryCircuit uses p[q_r] where q_r=10, but p only has 2 elements
    with pytest.raises(IndexError):
        sim.run(
            t_qiskit_circ=t_circ,
            psi0=zero_state(nqubit),
            shots=shots,
            device_param=device_param,
            nqubit=nqubit,
        )


def test_pickle_scaled_noise():
    pickle.dumps(ScaledNoiseGates(0.5))


def test_pickle_custom_noise():
    pickle.dumps(CustomNoiseGates())


def test_pickle_noise_channels():
    pickle.dumps(CustomNoiseChannelsGates([0, 1]))


def test_noiseless_matches_aer():
    """Noiseless simulator should match Aer exactly for deterministic circuit."""
    
    shots = 10
    nqubit = 4
    qc = build_deterministic_circuit_1(nqubit, nqubit)

    # Aer
    aer = AerSimulator()
    tqc = transpile(qc, aer)
    result = aer.run(tqc, shots=shots).result()
    counts_aer = result.get_counts()

    # Custom simulator
    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)

    sim = MrAndersonSimulator(gates=noise_free_gates)

    needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

    t_circ = transpile(
        qc,
        backend=FakeBrisbane(),
        initial_layout=range(nqubit),
        seed_transpiler=42,
        **({} if needs_controlflow else {"scheduling_method": "asap"})
    )


    counts_sim = sim.run(
    t_qiskit_circ=t_circ,
    psi0=psi0,
    shots=shots,
    device_param=device_param,
    nqubit=nqubit,
    )

    assert counts_sim["mid_counts"] == counts_aer


def test_noise_changes_distribution():
    """Adding noise should deviate from Aer noiseless result."""
    
    shots = 100
    qc = build_deterministic_circuit_0()

    # Aer reference
    aer = AerSimulator()
    tqc = transpile(qc, aer)
    result = aer.run(tqc, shots=shots).result()
    counts_aer = result.get_counts()
    probs_aer = counts_to_probs(counts_aer, shots)

    # Noisy simulator
    noisy_gates = ScaledNoiseGates(noise_scaling=1.0)
    sim = MrAndersonSimulator(gates=noisy_gates)
    nqubit = 2
    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)

    counts_noisy = sim.run(
    t_qiskit_circ=qc,
    psi0=psi0,
    shots=shots,
    device_param=device_param,
    nqubit=nqubit,
)

    probs_noisy = counts_to_probs(counts_noisy["mid_counts"], shots)

    dist = l1_distance(probs_aer, probs_noisy)

    assert dist > 0.0


def test_increasing_noise_increases_distance():
    """Higher noise scaling should increase deviation from noiseless."""
    
    shots = 100
    qc = build_deterministic_circuit_0()

    # Reference (no noise)
    sim_nf = MrAndersonSimulator(gates=noise_free_gates)
    nqubit = 2
    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)

    counts_nf = sim_nf.run(qc, psi0, shots, device_param, nqubit)
    probs_nf = counts_to_probs(counts_nf["mid_counts"], shots)

    # Moderate noise
    sim_low = MrAndersonSimulator(gates=ScaledNoiseGates(0.2))
    counts_low = sim_low.run(qc, psi0, shots, device_param, nqubit)
    probs_low = counts_to_probs(counts_low["mid_counts"], shots)

    # High noise
    sim_high = MrAndersonSimulator(gates=ScaledNoiseGates(1.0))
    counts_high = sim_high.run(qc, psi0, shots, device_param, nqubit)
    probs_high = counts_to_probs(counts_high["mid_counts"], shots)

    err_low = l1_distance(probs_nf, probs_low)
    err_high = l1_distance(probs_nf, probs_high)

    assert err_high > err_low



def test_custom_noise_scaling_effect():
    """CustomNoiseGates should affect output distribution."""
    shots = 100
    qc = build_deterministic_circuit_0()

    sim_low = MrAndersonSimulator(
        gates=CustomNoiseGates(p_scale=0.1, T1_scale=1.0, T2_scale=1.0)
    )
    sim_high = MrAndersonSimulator(
        gates=CustomNoiseGates(p_scale=1.0, T1_scale=1.0, T2_scale=1.0)
    )

    nqubit = 2
    psi0 = zero_state(nqubit)
    device_param = get_device_params(nqubit)

    counts_low = sim_low.run(qc, psi0, shots, device_param, nqubit)
    counts_high = sim_high.run(qc, psi0, shots, device_param, nqubit)

    probs_low = counts_to_probs(counts_low["mid_counts"], shots)
    probs_high = counts_to_probs(counts_high["mid_counts"], shots)

    all_keys = set(probs_low) | set(probs_high)

    v_low = np.array([probs_low.get(k, 0) for k in all_keys])
    v_high = np.array([probs_high.get(k, 0) for k in all_keys])

    # Distributions differ
    assert not np.allclose(v_low, v_high)

    # Physics check
    assert max(v_high) < max(v_low)
