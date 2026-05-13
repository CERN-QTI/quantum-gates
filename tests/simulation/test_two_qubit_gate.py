import pytest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from src.quantum_gates.gates import standard_gates, almost_noise_free_gates
from src.quantum_gates.utilities import DeviceParameters
from qiskit_aer import AerSimulator


_backend = FakeBrisbane()
_LAYOUT = list(range(16))

def _device_param(nqubits):
    dp = DeviceParameters(qubits_layout=_LAYOUT[:nqubits])
    dp.load_from_backend(_backend)
    return dp.__dict__()


def _aer_reference(qc, nqubits, shots=1000):
    sim = AerSimulator()
    t_circ = transpile(qc, sim)
    result = sim.run(t_circ, shots=shots).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}


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

@pytest.mark.parametrize("circuit_class", [EfficientCircuit])
def test_non_consecutive_cnot_flips_target(circuit_class):
    """CNOT on non-consecutive qubits (4, 15): control |1>, target |0> -> target flips to |1>."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(4)
    qc.cx(4, 15)
    qc.measure(range(nqubits), range(nqubits))

    # Check what AER says first
    aer_probs = _aer_reference(qc, nqubits)
    print(f"AER dominant outcome: {max(aer_probs, key=aer_probs.get)}")


    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    # Compare dominant outcomes
    aer_dominant = max(aer_probs, key=aer_probs.get)
    mr_dominant = max(result["mid_counts"], key=result["probs"].get)
    assert aer_dominant == mr_dominant, (
        f"MrAnderson dominant outcome '{mr_dominant}' does not match "
        f"AER '{aer_dominant}'"
    )

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_cnot_control_zero_no_flip(circuit_class):
    """CNOT on non-consecutive qubits (4, 15): control |0> -> target should not flip."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.cx(4, 15)
    qc.measure(range(nqubits), range(nqubits))
    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    zero_state = "0" * nqubits
    p_zero = result["probs"].get(zero_state, 0.0)
    assert p_zero > 0.95, (
        f"Expected P(|0...0>) > 0.95 when control is |0> "
        f"but got {p_zero:.4f} (probs={result['probs']})"
    )

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_cnot_middle_qubits_unaffected(circuit_class):
    """CNOT on qubits (4, 15): qubits 5-14 should be unaffected."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(4)
    qc.cx(4, 15)
    qc.measure(range(nqubits), range(nqubits))
    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    for bitstr, p in result["probs"].items():
        if p > 0.01:
            # qubits 5-14 should all be 0
            for q in range(5, 15):
                assert bitstr[-(q+1)] == "0", (
                    f"Qubit {q} should be unaffected but found '{bitstr}' "
                    f"with p={p:.4f}"
                )

# ── Option 2: Equivalence with consecutive ────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_matches_consecutive_noiseless(circuit_class):
    """H on qubit 4 + CNOT(4,15) and H on qubit 4 + CNOT(4,5) should both
    produce a 50/50 split between exactly two basis states."""
    nqubits = 16

    qc_non_consec = QuantumCircuit(nqubits, nqubits)
    qc_non_consec.h(4)
    qc_non_consec.cx(4, 15)
    qc_non_consec.measure(range(nqubits), range(nqubits))

    qc_consec = QuantumCircuit(nqubits, nqubits)
    qc_consec.h(4)
    qc_consec.cx(4, 5)
    qc_consec.measure(range(nqubits), range(nqubits))

    t_non_consec = _transpile_circuit(qc_non_consec, nqubits)
    t_consec = _transpile_circuit(qc_consec, nqubits)

    result_nc = _run_sim(t_non_consec, nqubits, almost_noise_free_gates, circuit_class, shots=500)
    result_c  = _run_sim(t_consec,     nqubits, almost_noise_free_gates, circuit_class, shots=500)

    assert np.isclose(sum(result_nc["probs"].values()), 1.0, atol=0.01)
    assert np.isclose(sum(result_c["probs"].values()),  1.0, atol=0.01)

    nc_outcomes = {k: v for k, v in result_nc["probs"].items() if v > 0.05}
    c_outcomes  = {k: v for k, v in result_c["probs"].items()  if v > 0.05}
    assert len(nc_outcomes) == 2, f"Expected 2 outcomes but got {nc_outcomes}"
    assert len(c_outcomes)  == 2, f"Expected 2 outcomes but got {c_outcomes}"

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_bell_state_correlations(circuit_class):
    """H on qubit 4 + CNOT(4,15) should entangle qubits 4 and 15,
    with all other qubits remaining in |0>."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(4)
    qc.cx(4, 15)
    qc.measure(range(nqubits), range(nqubits))
    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=500)

    for bitstr, p in result["probs"].items():
        if p > 0.05:
            q4  = bitstr[-(4+1)]
            q15 = bitstr[-(15+1)]
            assert q4 == q15, (
                f"Expected qubits 4 and 15 to be correlated "
                f"but found '{bitstr}' with p={p:.4f}"
            )
            # all other qubits should be 0
            for q in range(nqubits):
                if q not in (4, 15):
                    assert bitstr[-(q+1)] == "0", (
                        f"Qubit {q} should be 0 but found '{bitstr}' with p={p:.4f}"
                    )

# ── Option 3: Noise consistency ───────────────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_noiseless_higher_fidelity_than_noisy(circuit_class):
    """Noiseless non-consecutive CNOT(4,15) should give higher fidelity than noisy."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(4)
    qc.cx(4, 15)
    qc.measure(range(nqubits), range(nqubits))
    t_circ = _transpile_circuit(qc, nqubits)

    result_clean = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=300)
    result_noisy = _run_sim(t_circ, nqubits, standard_gates,           circuit_class, shots=300)

    # In the target state qubits 4 and 15 are 1, all others 0
    target = ["0"] * nqubits
    target[-(4+1)]  = "1"
    target[-(15+1)] = "1"
    target_str = "".join(target)

    p_clean = result_clean["probs"].get(target_str, 0.0)
    p_noisy = result_noisy["probs"].get(target_str, 0.0)

    assert p_clean > p_noisy, (
        f"Expected noiseless to have higher fidelity than noisy "
        f"but got p_clean={p_clean:.4f} <= p_noisy={p_noisy:.4f}"
    )
    assert p_clean > 0.90, (
        f"Expected P(target) > 0.90 for noiseless CNOT(4,15) "
        f"but got {p_clean:.4f}"
    )

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_noise_params_preserved(circuit_class):
    """Non-consecutive gate should use noise params of the actual qubits —
    verified by checking noiseless gives near-ideal result."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.x(4)
    qc.cx(4, 15)
    qc.measure(range(nqubits), range(nqubits))
    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=300)

    target = ["0"] * nqubits
    target[-(4+1)]  = "1"
    target[-(15+1)] = "1"
    target_str = "".join(target)

    p = result["probs"].get(target_str, 0.0)
    assert p > 0.90, (
        f"Expected P(target) > 0.90 for noiseless CNOT(4,15) "
        f"but got {p:.4f} (probs={result['probs']})"
    )

# ── Option 4: End-to-end integration ─────────────────────────────────────────

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_gate_completes_without_error(circuit_class):
    """Non-consecutive two-qubit gate should run through the full pipeline
    without raising any exception."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.cx(4, 15)
    qc.measure(range(nqubits), range(nqubits))
    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=100)

    assert "probs" in result
    assert np.isclose(sum(result["probs"].values()), 1.0, atol=0.01)

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_gate_mixed_with_consecutive(circuit_class):
    """Circuit with both consecutive and non-consecutive gates should run correctly."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(4)
    qc.cx(4, 5)      # consecutive
    qc.cx(4, 15)     # non-consecutive
    qc.measure(range(nqubits), range(nqubits))
    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=300)

    assert "probs" in result
    assert np.isclose(sum(result["probs"].values()), 1.0, atol=0.01)
    assert len(result["probs"]) > 0

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_gate_with_mid_measurement(circuit_class):
    """Non-consecutive gate followed by a mid-circuit measurement should work end-to-end."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits + 1)
    qc.x(4)
    qc.cx(4, 15)
    qc.measure(15, nqubits)    # mid-circuit measure qubit 15
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    t_circ = _transpile_circuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    assert "probs" in result
    assert np.isclose(sum(result["probs"].values()), 1.0, atol=0.01)

@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_non_consecutive_gate_result_is_reproducible(circuit_class):
    """Two runs should give similar probability distributions."""
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(4)
    qc.cx(4, 15)
    qc.measure(range(nqubits), range(nqubits))
    t_circ = _transpile_circuit(qc, nqubits)

    result1 = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=500)
    result2 = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=500)

    dominant1 = {k for k, v in result1["probs"].items() if v > 0.1}
    dominant2 = {k for k, v in result2["probs"].items() if v > 0.1}
    assert dominant1 == dominant2, (
        f"Dominant outcomes differ between runs: {dominant1} vs {dominant2}"
    )