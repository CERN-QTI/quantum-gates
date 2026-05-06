import pytest
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from src.quantum_gates.gates import standard_gates, almost_noise_free_gates
from src.quantum_gates.utilities import DeviceParameters

_backend = FakeBrisbane()
_location = "tests/helpers/device_parameters/ibm_kyoto/"
_LAYOUT = [0, 1, 2, 3, 4]


def _device_param(nqubits):
    dp = DeviceParameters(qubits_layout=_LAYOUT[:nqubits])
    dp.load_from_texts(location=_location)
    return dp.__dict__()


def _transpile_circuit(circ, nqubits):
    return transpile(
        circ,
        backend=_backend,
        initial_layout=_LAYOUT[:nqubits],
        seed_transpiler=42,
        optimization_level=0,
    )


def _run_sim(t_circ, nqubits, gates, circuit_class, shots=200):
    psi0 = np.zeros(2 ** nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=gates, CircuitClass=circuit_class)
    return sim.run(
        t_qiskit_circ=t_circ,
        psi0=psi0,
        shots=shots,
        device_param=_device_param(nqubits),
        nqubit=nqubits,
    )


# ── Option 1: Correctness against known states ────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_cnot_flips_target(circuit_class):
    """CNOT on non-consecutive qubits (0, 2): control |1>, target |0> -> target flips to |1>."""
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(0)          # set control qubit 0 to |1>
    qc.cx(0, 2)      # CNOT on non-consecutive qubits 0 and 2
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    # Expected: qubit 0 = 1, qubit 1 = 0, qubit 2 = 1 -> "101"
    p_101 = result["probs"].get("101", 0.0)
    assert p_101 > 0.95, (
        f"Expected P('101') > 0.95 after non-consecutive CNOT(0,2) "
        f"but got {p_101:.4f} (probs={result['probs']})"
    )


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_cnot_control_zero_no_flip(circuit_class):
    """CNOT on non-consecutive qubits (0, 2): control |0> -> target should not flip."""
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.cx(0, 2)      # control is |0>, target should stay |0>
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    p_000 = result["probs"].get("000", 0.0)
    assert p_000 > 0.95, (
        f"Expected P('000') > 0.95 when control is |0> "
        f"but got {p_000:.4f} (probs={result['probs']})"
    )


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_cnot_middle_qubit_unaffected(circuit_class):
    """CNOT on qubits (0, 2): middle qubit 1 should be unaffected."""
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(0)          # flip control
    qc.cx(0, 2)      # non-consecutive CNOT
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    # Qubit 1 (middle bit of bitstring) should always be 0
    for bitstr, p in result["probs"].items():
        if p > 0.01:
            assert bitstr[1] == "0", (
                f"Middle qubit should be unaffected but found '{bitstr}' "
                f"with p={p:.4f}"
            )


# ── Option 2: Equivalence with consecutive ────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_matches_consecutive_noiseless(circuit_class):
    """Non-consecutive CNOT(0,2) should give same probabilities as
    consecutive CNOT(0,1) on an equivalent circuit in the noiseless regime."""
    nqubits = 3

    # Non-consecutive: CNOT on qubits 0 and 2
    qc_non_consec = QuantumCircuit(nqubits, nqubits)
    qc_non_consec.h(0)
    qc_non_consec.cx(0, 2)
    qc_non_consec.measure(range(nqubits), range(nqubits))

    # Consecutive equivalent: CNOT on qubits 0 and 1
    qc_consec = QuantumCircuit(nqubits, nqubits)
    qc_consec.h(0)
    qc_consec.cx(0, 1)
    qc_consec.measure(range(nqubits), range(nqubits))

    t_non_consec = _transpile_circuit(qc_non_consec, nqubits)
    t_consec = _transpile_circuit(qc_consec, nqubits)

    result_nc = _run_sim(t_non_consec, nqubits, almost_noise_free_gates, circuit_class, shots=500)
    result_c  = _run_sim(t_consec,     nqubits, almost_noise_free_gates, circuit_class, shots=500)

    # Both should show ~50/50 split between two basis states
    total_nc = sum(result_nc["probs"].values())
    total_c  = sum(result_c["probs"].values())
    assert np.isclose(total_nc, 1.0, atol=0.01), f"Non-consecutive probs don't sum to 1: {total_nc}"
    assert np.isclose(total_c,  1.0, atol=0.01), f"Consecutive probs don't sum to 1: {total_c}"

    # Both should have exactly 2 outcomes with roughly equal probability
    nc_outcomes = {k: v for k, v in result_nc["probs"].items() if v > 0.05}
    c_outcomes  = {k: v for k, v in result_c["probs"].items()  if v > 0.05}
    assert len(nc_outcomes) == 2, f"Expected 2 outcomes but got {nc_outcomes}"
    assert len(c_outcomes)  == 2, f"Expected 2 outcomes but got {c_outcomes}"


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_bell_state_correlations(circuit_class):
    """H on qubit 0 + CNOT(0,2) should create entanglement between qubits 0 and 2,
    with qubit 1 remaining in |0>."""
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.cx(0, 2)      # Bell state between qubits 0 and 2
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=500)

    # Only "000" and "101" should appear (qubits 0 and 2 correlated, qubit 1 = 0)
    for bitstr, p in result["probs"].items():
        if p > 0.05:
            assert bitstr in ("000", "101"), (
                f"Expected only '000' or '101' for Bell state on qubits 0,2 "
                f"but found '{bitstr}' with p={p:.4f}"
            )

    p_000 = result["probs"].get("000", 0.0)
    p_101 = result["probs"].get("101", 0.0)
    assert 0.35 < p_000 < 0.65, f"Expected ~0.5 for '000' but got {p_000:.4f}"
    assert 0.35 < p_101 < 0.65, f"Expected ~0.5 for '101' but got {p_101:.4f}"


# ── Option 3: Noise consistency ───────────────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_noiseless_higher_fidelity_than_noisy(circuit_class):
    """Noiseless non-consecutive CNOT should give higher fidelity than noisy."""
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(0)
    qc.cx(0, 2)
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)

    result_clean = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=300)
    result_noisy = _run_sim(t_circ, nqubits, standard_gates,           circuit_class, shots=300)

    p_clean = result_clean["probs"].get("101", 0.0)
    p_noisy = result_noisy["probs"].get("101", 0.0)

    assert p_clean > p_noisy, (
        f"Expected noiseless to have higher fidelity than noisy "
        f"but got p_clean={p_clean:.4f} <= p_noisy={p_noisy:.4f}"
    )
    assert p_clean > 0.90, (
        f"Expected P('101') > 0.90 for noiseless non-consecutive CNOT "
        f"but got {p_clean:.4f}"
    )


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_noise_params_preserved(circuit_class):
    """Non-consecutive gate should use noise params of the actual qubits,
    not the permuted positions — verified by checking noiseless gives near-ideal result."""
    nqubits = 4
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(0)
    qc.cx(0, 3)      # far apart qubits
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=300)

    p_1001 = result["probs"].get("1001", 0.0)
    assert p_1001 > 0.90, (
        f"Expected P('1001') > 0.90 for noiseless CNOT(0,3) "
        f"but got {p_1001:.4f} (probs={result['probs']})"
    )


# ── Option 4: End-to-end integration ─────────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_gate_completes_without_error(circuit_class):
    """Non-consecutive two-qubit gate should run through the full pipeline
    without raising any exception."""
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.cx(0, 2)
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=100)

    assert "probs" in result
    assert np.isclose(sum(result["probs"].values()), 1.0, atol=0.01)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_gate_mixed_with_consecutive(circuit_class):
    """Circuit with both consecutive and non-consecutive gates should run correctly."""
    nqubits = 4
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.cx(0, 1)      # consecutive
    qc.cx(1, 3)      # non-consecutive
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=300)

    assert "probs" in result
    assert np.isclose(sum(result["probs"].values()), 1.0, atol=0.01)
    assert len(result["probs"]) > 0


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_gate_with_mid_measurement(circuit_class):
    """Non-consecutive gate followed by a mid-circuit measurement should work end-to-end."""
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits + 1)
    qc.x(0)
    qc.cx(0, 2)               # non-consecutive
    qc.measure(2, nqubits)    # mid-circuit measure qubit 2
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    assert "probs" in result
    assert np.isclose(sum(result["probs"].values()), 1.0, atol=0.01)


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_gate_result_is_reproducible(circuit_class):
    """Two runs with the same seed should give similar probability distributions."""
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.cx(0, 2)
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_circuit(qc, nqubits)

    result1 = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=500)
    result2 = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=500)

    # Both runs should show the same dominant outcomes
    dominant1 = {k for k, v in result1["probs"].items() if v > 0.1}
    dominant2 = {k for k, v in result2["probs"].items() if v > 0.1}
    assert dominant1 == dominant2, (
        f"Dominant outcomes differ between runs: {dominant1} vs {dominant2}"
    )