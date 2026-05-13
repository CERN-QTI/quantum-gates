import pytest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from src.quantum_gates.gates import almost_noise_free_gates
from src.quantum_gates.utilities import DeviceParameters

_backend = FakeBrisbane()
_LAYOUT = list(range(5))

def _device_param(nqubits):
    dp = DeviceParameters(qubits_layout=_LAYOUT[:nqubits])
    dp.load_from_backend(_backend)
    return dp.__dict__()

def _transpile_mr_anderson(circ, nqubits):
    return transpile(
        circ,
        backend=_backend,
        initial_layout=_LAYOUT[:nqubits],
        seed_transpiler=42,
        optimization_level=0,
    )

def _run_mr_anderson(t_circ, nqubits, circuit_class, shots=500):
    psi0 = np.zeros(2**nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=almost_noise_free_gates, CircuitClass=circuit_class)
    result = sim.run(
        t_qiskit_circ=t_circ,
        psi0=psi0,
        shots=shots,
        device_param=_device_param(nqubits),
        nqubit=nqubits,
    )
    counts = result["mid_counts"]
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

def _run_aer(circ, shots=2000):
    sim = AerSimulator()
    t_circ = transpile(circ, sim)
    result = sim.run(t_circ, shots=shots).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}

def _dominant_outcomes(probs, threshold=0.05):
    return {k for k, v in probs.items() if v > threshold}

def _assert_matches_aer(qc, nqubits, circuit_class, shots_mr=500, shots_aer=2000):
    t_circ = _transpile_mr_anderson(qc, nqubits)
    mr_probs  = _run_mr_anderson(t_circ, nqubits, circuit_class, shots=shots_mr)
    aer_probs = _run_aer(qc, shots=shots_aer)
    mr_dominant  = _dominant_outcomes(mr_probs)
    aer_dominant = _dominant_outcomes(aer_probs)
    assert mr_dominant == aer_dominant, (
        f"MrAnderson dominant outcomes {mr_dominant} do not match "
        f"AER dominant outcomes {aer_dominant}"
    )

# ── Single qubit gates ────────────────────────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_x_gate(circuit_class):
    """X gate on qubit 0 should flip |0> to |1>."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(0)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end

    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_hadamard(circuit_class):
    """H gate should produce equal superposition."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_x_on_all_qubits(circuit_class):
    """X on all qubits should flip entire register."""
    # Arrange
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    for q in range(nqubits):
        qc.x(q)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_rz_gate(circuit_class):
    """RZ gate followed by measurement in Z basis should leave |0> unchanged."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.rz(np.pi / 4, 0)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


# ── Two qubit gates ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_cnot_flips_target(circuit_class):
    """CNOT with control |1> should flip target."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_cnot_control_zero(circuit_class):
    """CNOT with control |0> should leave state unchanged."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.cx(0, 1)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_bell_state(circuit_class):
    """H + CNOT should produce Bell state."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_ghz_state(circuit_class):
    """H + chain of CNOTs should produce GHZ state."""
    # Arrange
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


# ── Mid-circuit measurements ──────────────────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_mid_measurement_zero_state(circuit_class):
    """Mid-circuit measurement on |0> should always read 0."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits + 1)
    qc.measure(0, nqubits)
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_mid_measurement_after_x(circuit_class):
    """Mid-circuit measurement after X gate should always read 1."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits + 1)
    qc.x(0)
    qc.measure(0, nqubits)
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


# ── Reset ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_reset_after_x(circuit_class):
    """Reset after X gate should return qubit to |0>."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(0)
    qc.reset(0)
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_reset_bell_state(circuit_class):
    """Reset one qubit of a Bell state should leave the other in a mixed state."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.reset(0)
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


# ── Deeper circuits ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_alternating_x_cnot(circuit_class):
    """Alternating X and CNOT gates across multiple qubits."""
    # Arrange
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(0)
    qc.cx(0, 1)
    qc.x(2)
    qc.cx(1, 2)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_hadamard_on_all_qubits(circuit_class):
    """H on all qubits produces uniform superposition."""
    # Arrange
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    for q in range(nqubits):
        qc.h(q)
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_aer_match_multi_cycle_stabilizer(circuit_class):
    """Multi-cycle stabilizer-like circuit with resets and mid-measurements."""
    # Arrange
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits + 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(1, nqubits)
    qc.reset(1)
    qc.cx(0, 1)
    qc.measure(1, nqubits + 1)
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    #Just to make sure the measurement is actually a mid-circuit measurement and not just at the end
    
    # Act & Assert
    _assert_matches_aer(qc, nqubits, circuit_class)
