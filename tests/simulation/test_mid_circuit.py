import pytest
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import EfficientCircuit
from src.quantum_gates.circuits import BinaryCircuit
from src.quantum_gates.gates import standard_gates
from src.quantum_gates.gates import almost_noise_free_gates
from src.quantum_gates.utilities import DeviceParameters
from src.quantum_gates.quantum_algorithms import hadamard_reverse_qft_circ


_backend = FakeBrisbane()
_location = "tests/helpers/device_parameters/ibm_kyoto/"
_LAYOUT = [0, 1, 2, 3, 4]


def _device_param(nqubits):
    """Load device parameters for a linear layout of nqubits."""
    dp = DeviceParameters(qubits_layout=_LAYOUT[:nqubits])
    dp.load_from_texts(location=_location)
    return dp.__dict__()


def _transpile_standard(circ, nqubits):
    """Transpile a standard circuit (no mid-circuit ops) with ASAP scheduling."""
    return transpile(
        circ,
        backend=_backend,
        initial_layout=_LAYOUT[:nqubits],
        scheduling_method='asap',
        seed_transpiler=42,
    )


def _transpile_midcircuit(circ, nqubits):
    """Transpile a mid-circuit circuit without scheduling and at opt_level=0
    to preserve mid-circuit structure."""
    return transpile(
        circ,
        backend=_backend,
        initial_layout=_LAYOUT[:nqubits],
        seed_transpiler=42,
        optimization_level=0,
    )


def _run_sim(t_circ, nqubits, gates, circuit_class, shots=100):
    """Run the simulator and return the full result dict."""
    psi0 = np.zeros(2**nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=gates, CircuitClass=circuit_class)
    return sim.run(
        t_qiskit_circ=t_circ,
        psi0=psi0,
        shots=shots,
        device_param=_device_param(nqubits),
        nqubit=nqubits,
    )


@pytest.mark.parametrize("nqubits,circuit_class", [
    (2, EfficientCircuit),
    (3, EfficientCircuit),
    (2, BinaryCircuit),
    (3, BinaryCircuit),
])
def test_identity_zero_noise(nqubits, circuit_class):
    """Apply the non-trivial identity circuit (hadamard plus reverse qft circ) with noise-free gates and check that
    the result is |0...0> with probability ~1."""
    # Arrange
    circ = hadamard_reverse_qft_circ(nqubits)
    t_circ = _transpile_standard(circ, nqubits)

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=10)

    # Assert
    probs = result["probs"]
    zero_state = "0" * nqubits
    p_zero = probs.get(zero_state, 0.0)
    assert p_zero > 0.99, f"Expected P(|0...0>) ~ 1 but got {p_zero:.4f}"


def _ghz_with_mid_measure(nqubits):
    """Build a GHZ circuit with a mid-circuit measurement on qubit 0.

    Uses an extra classical bit for the mid-measurement. After the mid-measure,
    adds a small RZ rotation (virtual-Z which doesn't change probabilities) as a
    quantum operation so the transpiler classifies the measure as mid-circuit.
    """
    n_clbits = nqubits + 1  # extra clbit for mid-measure
    qc = QuantumCircuit(nqubits, n_clbits)
    qc.h(0)
    for j in range(1, nqubits):
        qc.cx(0, j)
    qc.measure(0, nqubits)
    qc.rz(0.001, 0)
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    return qc


@pytest.mark.parametrize("nqubits", [2, 3])
def test_mid_measure_ghz_produces_mid_counts(nqubits):
    """A mid-circuit measurement on a GHZ state should produce non-trivial
    mid_counts (both 0 and 1 outcomes)."""
    # Arrange
    circ = _ghz_with_mid_measure(nqubits)
    t_circ = _transpile_midcircuit(circ, nqubits)

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, BinaryCircuit, shots=500)

    # Assert
    # Mid counts should be non-empty, demonstrating that a measurement happened
    assert len(result["mid_counts"]) > 0, "Expected mid-circuit measurement counts"
    assert len(result["results"]) == 500

    # For a GHZ state, qubit 0 should measure 0 or 1 with roughly equal probability
    # Check per-shot mid-measurement outcomes
    outcomes = []
    for shot in result["results"]:
        for event in shot["mid"]:
            outcomes.extend([int(o) for o in event["outcome"]])

    n_zeros = outcomes.count(0)
    n_ones = outcomes.count(1)
    total = len(outcomes)
    assert total > 0, "No mid-circuit outcomes recorded"
    frac_zero = n_zeros / total
    assert 0.30 < frac_zero < 0.70, (
        f"Expected more or less 50/50 mid-measure outcomes for GHZ state but got "
        f"{n_zeros}/{n_ones} (frac_zero={frac_zero:.2f})"
    )


def _classical_state_with_mid_measure(nqubits):
    """Prepare |0...0>, mid-measure qubit 0, add RZ to keep it mid-circuit,
    then final-measure."""
    n_clbits = nqubits + 1
    qc = QuantumCircuit(nqubits, n_clbits)
    # State is |0...0>, mid-measure qubit 0 to extra clbit
    qc.measure(0, nqubits)
    # Virtual-Z to keep as mid-circuit
    qc.rz(0.001, 0)
    # Final measurement
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    return qc


@pytest.mark.parametrize("nqubits", [2, 3])
def test_classical_state_unaffected_by_mid_measure(nqubits):
    """Mid-circuit measurement on |0...0> in zero-noise should always give 0
    and leave the state unchanged."""
    # Arrange
    circ = _classical_state_with_mid_measure(nqubits)
    t_circ = _transpile_midcircuit(circ, nqubits)

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, BinaryCircuit, shots=100)
    probs = result["probs"]
    zero_state = "0" * nqubits
    p_zero = probs.get(zero_state, 0.0)

    # Assert
    assert p_zero > 0.99, (
        f"Expected |0...0> unaffected by mid-measure but got P={p_zero:.4f}, probs={probs}"
    )

    # All mid-circuit outcomes should be 0
    for shot in result["results"]:
        for event in shot["mid"]:
            for o in event["outcome"]:
                assert int(o) == 0, f"Expected mid-measure outcome 0 but got {o}"


def test_output_structure_standard_circuit():
    """For a standard circuit (no mid-circuit ops), the result dict should have
    the expected keys and valid probability distribution."""
    # Arrange
    nqubits = 2
    circ = hadamard_reverse_qft_circ(nqubits)
    t_circ = _transpile_standard(circ, nqubits)

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, EfficientCircuit, shots=10)

    # Assert
    # Check keys
    assert "probs" in result
    assert "results" in result
    assert "num_clbits" in result
    assert "mid_counts" in result
    assert "statevector_readout" in result

    # Probabilities should be a valid probability distribution
    probs = result["probs"]
    assert isinstance(probs, dict)
    values = np.array(list(probs.values()))
    assert np.all(values >= 0), "Probabilities must be non-negative"
    assert np.sum(values) == pytest.approx(1.0, abs=1e-6), "Probabilities must sum to 1"

    # results should have one entry per shot
    assert len(result["results"]) == 10

    # No mid-circuit events for a standard circuit
    for shot in result["results"]:
        assert len(shot["mid"]) == 0


def test_noise_scaling_identity_circuit():
    """Run the identity circuit with noisy vs almost-noise-free gates.
    The almost-noise-free case should have higher P(|0...0>)."""
    # Arrange
    nqubits = 2
    circ = hadamard_reverse_qft_circ(nqubits)
    t_circ = _transpile_standard(circ, nqubits)

    # Act
    result_noisy = _run_sim(t_circ, nqubits, standard_gates, BinaryCircuit, shots=200)
    result_clean = _run_sim(t_circ, nqubits, almost_noise_free_gates, BinaryCircuit, shots=200)

    # Assert
    zero_state = "0" * nqubits
    p_noisy = result_noisy["probs"].get(zero_state, 0.0)
    p_clean = result_clean["probs"].get(zero_state, 0.0)

    assert p_clean > p_noisy, (
        f"Expected less noise -> higher P(|0...0>) but got "
        f"p_clean={p_clean:.4f} <= p_noisy={p_noisy:.4f}"
    )


def _repetition_code_circuit(n_data=3, n_cycles=3):
    """Build a minimal repetition code circuit.

    n_data data qubits + 1 ancilla qubit. Each cycle:
      1. CNOT from each data qubit to ancilla (syndrome extraction)
      2. Mid-circuit measure ancilla
      3. Reset ancilla

    In the zero-noise regime starting from |0...0>, the syndrome should
    always be 0 and the data qubits should remain |0...0>.
    """
    n_total = n_data + 1
    ancilla = n_data
    n_clbits = n_cycles + n_data

    qc = QuantumCircuit(n_total, n_clbits)

    for cycle in range(n_cycles):
        # Syndrome extraction: parity check
        for d in range(n_data):
            qc.cx(d, ancilla)
        qc.measure(ancilla, cycle)
        qc.reset(ancilla)
        qc.barrier()

    # Final measurement of data qubits
    for d in range(n_data):
        qc.measure(d, n_cycles + d)

    return qc, n_total


@pytest.mark.parametrize("n_cycles", [1, 3, 5])
def test_repetition_code_zero_noise(n_cycles):
    """In zero noise, all syndrome measurements should be 0 and data qubits
    should remain |0...0> after repeated measure+reset cycles."""
    # Arrange
    n_data = 3
    circ, n_total = _repetition_code_circuit(n_data=n_data, n_cycles=n_cycles)
    t_circ = _transpile_midcircuit(circ, n_total)

    # Act
    result = _run_sim(t_circ, n_total, almost_noise_free_gates, BinaryCircuit, shots=50)

    # Assert
    # Check that mid-circuit measurement outcomes are all 0
    for shot in result["results"]:
        for event in shot["mid"]:
            for o in event["outcome"]:
                assert int(o) == 0, (
                    f"Expected syndrome=0 in zero noise but got {o} "
                    f"at step {event['step']}"
                )

    # Final probs: data qubits should all be |0>
    probs = result["probs"]
    total_prob_zero_data = 0.0
    for bitstr, p in probs.items():
        if len(bitstr) >= n_data:
            data_bits = bitstr[:n_data]
            if all(b == '0' for b in data_bits):
                total_prob_zero_data += p

    assert total_prob_zero_data > 0.90, (
        f"Expected data qubits to remain |0...0> but got P={total_prob_zero_data:.4f}"
    )
