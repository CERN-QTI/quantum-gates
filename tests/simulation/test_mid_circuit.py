import pytest
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
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


def _meas_map(circ):
    """Return {physical_qubit: classical_bit} for every measure op in circ."""
    return {
        op.qubits[0]._index: circ.find_bit(op.clbits[0]).index
        for op in circ.data
        if op.operation.name == "measure"
    }


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
    should remain |0...0> after repeated measure and reset cycles."""
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


# Transpiler permutation tests
# See https://github.com/Qiskit/qiskit/issues/5839 and AER_testing_statevector.ipynb
# The Qiskit transpiler may insert SWAP gates when a circuit uses non-adjacent qubits, which permutes the qubit
# ordering in the statevector. These tests verify that our simulator still produces correct measurement probabilities
# despite such permutations.
def test_transpiler_routing_structure_and_measurement_map():
    """Directly inspect what the transpiler does to CX(0,2) on a linear
    topology and verify the simulator accounts for it.

    CX(0,2) is not natively supported, 0 and 2 are not adjacent on FakeBrisbane.
    The transpiler decomposes it into adjacent ECR gates, which:
      1. Adds many more gates than the original two (X and CX).
      2. Permutes the measurement map: instead of {0:0, 1:1, 2:2} the
         transpiler emits q0->c0, q2->c1, q1->c2  with qubits 1 and 2 swapped.

    The simulator reads this permuted map from the transpiled circuit and
    produces the correct logical output '101' (q0=1, q1=0, q2=1).
    """
    nqubits = 3
    circ = QuantumCircuit(3, 3)
    circ.x(0)
    circ.cx(0, 2)
    circ.barrier()
    circ.measure([0, 1, 2], [0, 1, 2])

    t_circ = _transpile_standard(circ, nqubits)

    # Routing inserted extra gates: original has 2 (X and CX), transpiled has many more
    routed = [op for op in t_circ.data if op.operation.name not in {"barrier", "delay", "measure"}]
    assert len(routed) > 2, f"Expected routing to insert extra gates but got {len(routed)}"

    # Measurement map is permuted to account for the SWAP: q1 and q2 exchanged
    assert _meas_map(t_circ) == {0: 0, 2: 1, 1: 2}, f"Unexpected measurement map: {_meas_map(t_circ)}"


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_transpiler_permutation_deterministic_state(circuit_class):
    """CX(0,2) forces the transpiler to route through qubit 1 on a linear
    topology.  The circuit prepares |101> deterministically. Verify the
    simulator produces one dominant outcome despite qubit permutations."""
    # Arrange: X(0), CX(0,2) -> |101>  with q0 and q2 two non-adjacent qubits
    nqubits = 3
    circ = QuantumCircuit(3, 3)
    circ.x(0)
    circ.cx(0, 2)
    circ.barrier()
    circ.measure([0, 1, 2], [0, 1, 2])

    t_circ = _transpile_standard(circ, nqubits)

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=100)
    probs = result["probs"]

    # Assert: |101> is prepared with high fidelity
    assert probs['101'] > 0.95, (
        f"Expected |101> to be prepared with p>0.95 after transpiler routing "
        f"but got probs={probs}"
    )


@pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
def test_transpiler_permutation_bell_state(circuit_class):
    """H(0) and CX(0,2) creates a Bell-like state between non-adjacent qubits,
    forcing the transpiler to route through qubit 1. The final measurement
    should show roughly 50/50 between two states."""
    # Arrange
    nqubits = 3
    circ = QuantumCircuit(3, 3)
    circ.h(0)
    circ.cx(0, 2)
    circ.barrier()
    circ.measure([0, 1, 2], [0, 1, 2])

    t_circ = _transpile_standard(circ, nqubits)

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=500)
    probs = result["probs"]

    # Assert: two dominant states should share ~1.0 probability
    sorted_probs = sorted(probs.values(), reverse=True)
    top_two_sum = sorted_probs[0] + sorted_probs[1]
    assert top_two_sum > 0.95, (
        f"Expected two dominant states summing to >0.95 but got {top_two_sum:.4f}"
    )
    # Each should be roughly 0.5
    assert 0.30 < sorted_probs[0] < 0.70, (
        f"Expected ~50% but dominant state has P={sorted_probs[0]:.4f}"
    )


def test_transpiler_permutation_mid_circuit_deterministic():
    """When a circuit forces routing (CX on non-adjacent qubits) and has mid-circuit measurements, verify that
    mid-circuit outcomes are correct despite qubit permutations.

    Circuit: X(0), CX(0,2) -> |101>, then mid-measure q0. Expected: mid-measure of q0 should always be 1.
    """
    # Arrange
    nqubits = 3
    n_clbits = nqubits + 1
    circ = QuantumCircuit(nqubits, n_clbits)
    circ.x(0)
    circ.cx(0, 2)
    circ.measure(0, nqubits)   # Mid-measure q0 to extra clbit
    circ.rz(0.001, 0)      # Keep it mid-circuit
    circ.barrier()
    circ.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_midcircuit(circ, nqubits)

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, BinaryCircuit, shots=100)

    # Assert: all mid-circuit outcomes should be 1, q0 was prepared with X
    for shot in result["results"]:
        for event in shot["mid"]:
            for o in event["outcome"]:
                assert int(o) == 1, (
                    f"Expected mid-measure outcome 1 for X-prepared qubit "
                    f"but got {o}"
                )


def test_mid_circuit_output_clbit_width():
    """Documents that mid_counts keys have num_clbits characters (total classical bits) while probs keys have nqubit
    characters (state vector width).

    Context: A contributor observed that simulating 3 qubits produced output strings like '0000' (4 chars). This happens
    because the circuit has 4 classical bits (3 for final measurement + 1 for mid-circuit measurement). The extra
    character is the mid-circuit classical bit, not an extra qubit.
    """
    # Arrange
    nqubits = 3
    n_clbits = nqubits + 1
    circ = QuantumCircuit(nqubits, n_clbits)
    circ.measure(0, nqubits)
    circ.rz(0.001, 0)
    circ.barrier()
    circ.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_midcircuit(circ, nqubits)

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, BinaryCircuit, shots=50)

    # Assert: probs keys should have nqubit characters
    for key in result["probs"]:
        assert len(key) == nqubits, (
            f"Expected probs key length {nqubits} but got '{key}' (len={len(key)})"
        )

    # Assert: mid_counts keys should have num_clbits characters
    assert result["num_clbits"] == n_clbits
    for key in result["mid_counts"]:
        assert len(key) == n_clbits, (
            f"Expected mid_counts key length {n_clbits} but got '{key}' "
            f"(len={len(key)}). The extra character comes from the additional "
            f"classical bit used for mid-circuit measurement, not an extra qubit."
        )


def test_final_measurements_are_included_in_qiskit_ordered_counts():
    """Final measurement writes should appear in the Aer-style count strings."""
    # Arrange
    nqubits = 2
    shots = 20
    circ = QuantumCircuit(nqubits, nqubits)
    circ.x(0)
    circ.measure([0, 1], [0, 1])

    t_circ = _transpile_standard(circ, nqubits)
    aer_counts = AerSimulator().run(t_circ, shots=shots).result().get_counts()

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, BinaryCircuit, shots=shots)

    # Assert
    assert result["mid_counts"] == aer_counts


def test_mid_and_final_measurements_share_classical_memory_like_aer():
    """Mid-circuit writes and final writes should be folded into one count key."""
    # Arrange
    nqubits = 3
    shots = 20
    circ = QuantumCircuit(nqubits, 5)
    circ.x(0)
    circ.x(2)
    circ.measure([0, 2], [3, 4])
    circ.rz(0.001, 0)
    circ.rz(0.001, 2)
    circ.barrier()
    circ.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_midcircuit(circ, nqubits)
    aer_counts = AerSimulator().run(t_circ, shots=shots).result().get_counts()

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, BinaryCircuit, shots=shots)

    # Assert
    assert result["mid_counts"] == aer_counts
    assert result["mid_only_counts"] == {"11000": shots}


def test_counts_match_aer_multi_register_formatting():
    """Multiple classical registers should be grouped like Qiskit get_counts()."""
    # Arrange
    nqubits = 3
    shots = 20
    q = QuantumRegister(nqubits, "q")
    a = ClassicalRegister(2, "a")
    b = ClassicalRegister(2, "b")
    circ = QuantumCircuit(q, a, b)
    circ.x(q[0])
    circ.x(q[2])
    circ.measure(q[0], a[0])
    circ.measure(q[1], a[1])
    circ.measure(q[2], b[1])

    t_circ = _transpile_standard(circ, nqubits)
    aer_counts = AerSimulator().run(t_circ, shots=shots).result().get_counts()

    # Act
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, BinaryCircuit, shots=shots)

    # Assert
    assert result["mid_counts"] == aer_counts

def _group_mid_events_by_round(mid_events, nqubits, n_mid, n_rounds):
    """Group scalar mid-measure events back into array-measure rounds.

    The simulator stores array-style qc.measure([...], [...]) as one scalar
    event per measured qubit. We reconstruct the original array-measure rounds
    using the classical-bit blocks assigned in the circuit.
    """
    grouped = []

    for r in range(n_rounds):
        c_start = nqubits + r * n_mid
        c_end = c_start + n_mid

        round_events = [
            event for event in mid_events
            if c_start <= event["clbits"][0] < c_end
        ]

        round_events = sorted(round_events, key=lambda e: e["clbits"][0])
        grouped.append([int(event["outcome"][0]) for event in round_events])

    return grouped


def _deterministic_array_mid_measure_circuit(nqubits, init_ones, measure_qubits, n_rounds=1):
    """Build deterministic circuits with array-style mid-circuit measurements.

    Each round measures `measure_qubits` using one array-style qc.measure(...) call.
    """
    n_mid = len(measure_qubits)
    n_clbits = nqubits + n_rounds * n_mid
    qc = QuantumCircuit(nqubits, n_clbits)

    for q in init_ones:
        qc.x(q)

    for r in range(n_rounds):
        c_start = nqubits + r * n_mid
        qc.measure(measure_qubits, range(c_start, c_start + n_mid))

        # Keep measurements mid-circuit.
        for q in measure_qubits:
            qc.rz(0.001, q)

        qc.barrier()

    qc.measure(range(nqubits), range(nqubits))

    expected_mid = [
        [1 if q in init_ones else 0 for q in measure_qubits]
        for _ in range(n_rounds)
    ]

    expected_final = "".join("1" if q in init_ones else "0" for q in range(nqubits))

    return qc, expected_mid, expected_final


@pytest.mark.parametrize(
    "nqubits,init_ones,measure_qubits,n_rounds",
    [
        pytest.param(3, [0, 1], [0, 1], 1, id="3q_measure_2_once"),
        pytest.param(5, [0, 2, 4], [0, 1, 2, 3, 4], 1, id="5q_measure_all_once"),
        pytest.param(5, [1, 3], [0, 1, 2, 3, 4], 3, id="5q_measure_all_3_rounds"),
        pytest.param(5, [0, 2, 4], [0, 2, 4], 5, id="5q_measure_subset_5_rounds"),
    ],
)
def test_array_mid_measure_deterministic_parametrized(
    nqubits,
    init_ones,
    measure_qubits,
    n_rounds,
):
    """Array-style mid-circuit measurements should record the correct outcomes
    for deterministic circuits and repeated measurement rounds."""
    circ, expected_mid, expected_final = _deterministic_array_mid_measure_circuit(
        nqubits=nqubits,
        init_ones=init_ones,
        measure_qubits=measure_qubits,
        n_rounds=n_rounds,
    )

    t_circ = _transpile_midcircuit(circ, nqubits)

    result = _run_sim(
        t_circ,
        nqubits,
        almost_noise_free_gates,
        BinaryCircuit,
        shots=100,
    )

    n_mid = len(measure_qubits)

    assert len(result["mid_counts"]) > 0
    assert len(result["results"]) == 100

    for shot in result["results"]:
        assert len(shot["mid"]) == n_rounds * n_mid, (
            f"Expected {n_rounds * n_mid} scalar mid-measure events, "
            f"got {shot['mid']}"
        )

        grouped = _group_mid_events_by_round(
            shot["mid"],
            nqubits=nqubits,
            n_mid=n_mid,
            n_rounds=n_rounds,
        )

        assert grouped == expected_mid, (
            f"Expected grouped mid outcomes {expected_mid}, got {grouped}"
        )

    probs = result["probs"]
    assert probs.get(expected_final, 0.0) > 0.95, (
        f"Expected final state |{expected_final}> but got probs={probs}"
    )


def _entangled_array_mid_measure_circuit(nqubits, entangle_qubits, n_rounds=1):
    """Prepare a GHZ/Bell-like state on entangle_qubits and repeatedly
    mid-measure them using array-style measurements."""
    n_mid = len(entangle_qubits)
    n_clbits = nqubits + n_rounds * n_mid
    qc = QuantumCircuit(nqubits, n_clbits)

    root = entangle_qubits[0]
    qc.h(root)

    for q in entangle_qubits[1:]:
        qc.cx(root, q)

    for r in range(n_rounds):
        c_start = nqubits + r * n_mid
        qc.measure(entangle_qubits, range(c_start, c_start + n_mid))

        for q in entangle_qubits:
            qc.rz(0.001, q)

        qc.barrier()

    qc.measure(range(nqubits), range(nqubits))

    return qc


@pytest.mark.parametrize(
    "nqubits,entangle_qubits,n_rounds",
    [
        pytest.param(2, [0, 1], 1, id="bell_2q_once"),
        pytest.param(3, [0, 1, 2], 1, id="ghz_3q_once"),
        pytest.param(5, [0, 1, 2, 3, 4], 3, id="ghz_5q_3_rounds"),
        pytest.param(5, [0, 2, 4], 4, id="subset_ghz_3q_4_rounds"),
    ],
)
def test_array_mid_measure_entangled_correlated_parametrized(
    nqubits,
    entangle_qubits,
    n_rounds,
):
    """Array-style mid-circuit measurements on Bell/GHZ states should remain
    correlated across measured qubits and repeated rounds."""
    circ = _entangled_array_mid_measure_circuit(
        nqubits=nqubits,
        entangle_qubits=entangle_qubits,
        n_rounds=n_rounds,
    )

    t_circ = _transpile_midcircuit(circ, nqubits)

    result = _run_sim(
        t_circ,
        nqubits,
        almost_noise_free_gates,
        BinaryCircuit,
        shots=500,
    )

    n_mid = len(entangle_qubits)
    first_round_values = []

    for shot in result["results"]:
        assert len(shot["mid"]) == n_rounds * n_mid, (
            f"Expected {n_rounds * n_mid} scalar mid-measure events, "
            f"got {shot['mid']}"
        )

        grouped = _group_mid_events_by_round(
            shot["mid"],
            nqubits=nqubits,
            n_mid=n_mid,
            n_rounds=n_rounds,
        )

        for observed in grouped:
            assert len(observed) == n_mid
            assert observed in ([0] * n_mid, [1] * n_mid), (
                f"Expected correlated outcome, got {observed}"
            )

        for observed in grouped[1:]:
            assert observed == grouped[0], (
                f"Expected repeated measurements to match first collapse, got {grouped}"
            )

        first_round_values.append(grouped[0][0])

    frac_zero = first_round_values.count(0) / len(first_round_values)

    assert 0.30 < frac_zero < 0.70, (
        f"Expected roughly 50/50 collapse but got frac_zero={frac_zero:.2f}"
    )

def _group_mid_events_by_round(mid_events, nqubits, n_mid, n_rounds):
    """Group scalar mid-measure events back into array-measure rounds.

    The simulator stores array-style qc.measure([...], [...]) as one scalar
    event per measured qubit. We reconstruct the original array-measure rounds
    using the classical-bit blocks assigned in the circuit.
    """
    grouped = []

    for r in range(n_rounds):
        c_start = nqubits + r * n_mid
        c_end = c_start + n_mid

        round_events = [
            event for event in mid_events
            if c_start <= event["clbits"][0] < c_end
        ]

        round_events = sorted(round_events, key=lambda e: e["clbits"][0])
        grouped.append([int(event["outcome"][0]) for event in round_events])

    return grouped


def _deterministic_array_mid_measure_circuit(nqubits, init_ones, measure_qubits, n_rounds=1):
    """Build deterministic circuits with array-style mid-circuit measurements.

    Each round measures `measure_qubits` using one array-style qc.measure(...) call.
    """
    n_mid = len(measure_qubits)
    n_clbits = nqubits + n_rounds * n_mid
    qc = QuantumCircuit(nqubits, n_clbits)

    for q in init_ones:
        qc.x(q)

    for r in range(n_rounds):
        c_start = nqubits + r * n_mid
        qc.measure(measure_qubits, range(c_start, c_start + n_mid))

        # Keep measurements mid-circuit.
        for q in measure_qubits:
            qc.rz(0.001, q)

        qc.barrier()

    qc.measure(range(nqubits), range(nqubits))

    expected_mid = [
        [1 if q in init_ones else 0 for q in measure_qubits]
        for _ in range(n_rounds)
    ]

    expected_final = "".join("1" if q in init_ones else "0" for q in reversed(range(nqubits)))

    return qc, expected_mid, expected_final


@pytest.mark.parametrize(
    "nqubits,init_ones,measure_qubits,n_rounds",
    [
        pytest.param(3, [0, 1], [0, 1], 1, id="3q_measure_2_once"),
        pytest.param(5, [0, 2, 4], [0, 1, 2, 3, 4], 1, id="5q_measure_all_once"),
        pytest.param(5, [1, 3], [0, 1, 2, 3, 4], 3, id="5q_measure_all_3_rounds"),
        pytest.param(5, [0, 2, 4], [0, 2, 4], 5, id="5q_measure_subset_5_rounds"),
    ],
)
def test_array_mid_measure_deterministic_parametrized(
    nqubits,
    init_ones,
    measure_qubits,
    n_rounds,
):
    """Array-style mid-circuit measurements should record the correct outcomes
    for deterministic circuits and repeated measurement rounds."""
    circ, expected_mid, expected_final = _deterministic_array_mid_measure_circuit(
        nqubits=nqubits,
        init_ones=init_ones,
        measure_qubits=measure_qubits,
        n_rounds=n_rounds,
    )

    t_circ = _transpile_midcircuit(circ, nqubits)

    result = _run_sim(
        t_circ,
        nqubits,
        almost_noise_free_gates,
        BinaryCircuit,
        shots=100,
    )

    n_mid = len(measure_qubits)

    assert len(result["mid_counts"]) > 0
    assert len(result["results"]) == 100

    for shot in result["results"]:
        assert len(shot["mid"]) == n_rounds * n_mid, (
            f"Expected {n_rounds * n_mid} scalar mid-measure events, "
            f"got {shot['mid']}"
        )

        grouped = _group_mid_events_by_round(
            shot["mid"],
            nqubits=nqubits,
            n_mid=n_mid,
            n_rounds=n_rounds,
        )

        assert grouped == expected_mid, (
            f"Expected grouped mid outcomes {expected_mid}, got {grouped}"
        )

    probs = result["probs"]
    assert probs.get(expected_final, 0.0) > 0.95, (
        f"Expected final state |{expected_final}> but got probs={probs}"
    )


def _entangled_array_mid_measure_circuit(nqubits, entangle_qubits, n_rounds=1):
    """Prepare a GHZ/Bell-like state on entangle_qubits and repeatedly
    mid-measure them using array-style measurements."""
    n_mid = len(entangle_qubits)
    n_clbits = nqubits + n_rounds * n_mid
    qc = QuantumCircuit(nqubits, n_clbits)

    root = entangle_qubits[0]
    qc.h(root)

    for q in entangle_qubits[1:]:
        qc.cx(root, q)

    for r in range(n_rounds):
        c_start = nqubits + r * n_mid
        qc.measure(entangle_qubits, range(c_start, c_start + n_mid))

        for q in entangle_qubits:
            qc.rz(0.001, q)

        qc.barrier()

    qc.measure(range(nqubits), range(nqubits))

    return qc


@pytest.mark.parametrize(
    "nqubits,entangle_qubits,n_rounds",
    [
        pytest.param(2, [0, 1], 1, id="bell_2q_once"),
        pytest.param(3, [0, 1, 2], 1, id="ghz_3q_once"),
        pytest.param(5, [0, 1, 2, 3, 4], 3, id="ghz_5q_3_rounds"),
        pytest.param(5, [0, 2, 4], 4, id="subset_ghz_3q_4_rounds"),
    ],
)
def test_array_mid_measure_entangled_correlated_parametrized(
    nqubits,
    entangle_qubits,
    n_rounds,
):
    """Array-style mid-circuit measurements on Bell/GHZ states should remain
    correlated across measured qubits and repeated rounds."""
    circ = _entangled_array_mid_measure_circuit(
        nqubits=nqubits,
        entangle_qubits=entangle_qubits,
        n_rounds=n_rounds,
    )

    t_circ = _transpile_midcircuit(circ, nqubits)

    result = _run_sim(
        t_circ,
        nqubits,
        almost_noise_free_gates,
        BinaryCircuit,
        shots=500,
    )

    n_mid = len(entangle_qubits)
    first_round_values = []

    for shot in result["results"]:
        assert len(shot["mid"]) == n_rounds * n_mid, (
            f"Expected {n_rounds * n_mid} scalar mid-measure events, "
            f"got {shot['mid']}"
        )

        grouped = _group_mid_events_by_round(
            shot["mid"],
            nqubits=nqubits,
            n_mid=n_mid,
            n_rounds=n_rounds,
        )

        for observed in grouped:
            assert len(observed) == n_mid
            assert observed in ([0] * n_mid, [1] * n_mid), (
                f"Expected correlated outcome, got {observed}"
            )

        for observed in grouped[1:]:
            assert observed == grouped[0], (
                f"Expected repeated measurements to match first collapse, got {grouped}"
            )

        first_round_values.append(grouped[0][0])

    frac_zero = first_round_values.count(0) / len(first_round_values)

    assert 0.30 < frac_zero < 0.70, (
        f"Expected roughly 50/50 collapse but got frac_zero={frac_zero:.2f}"
    )