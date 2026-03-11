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


def _transpile_midcircuit(circ, nqubits):
    """Transpile without scheduling and at opt_level=0 to preserve mid-circuit
    structure (resets, mid-measures, etc.)."""
    return transpile(
        circ,
        backend=_backend,
        initial_layout=_LAYOUT[:nqubits],
        seed_transpiler=42,
        optimization_level=0,
    )


def _run_sim(t_circ, nqubits, gates, circuit_class, shots=200):
    """Run the simulator and return the full result dict."""
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


def _build_reset_circuit(nqubits, prepare_ops=None, reset_qubits=None,
                         extra_clbits=0):
    """Generic helper that:
      1. Applies optional preparation gates (callable receiving the circuit).
      2. Resets the listed qubits.
      3. Measures all qubits into the first nqubits classical bits.

    Args:
        prepare_ops: callable(qc) that applies preparation gates.
        reset_qubits: list of qubit indices to reset (default: all).
        extra_clbits: extra classical bits appended after the nqubits measure bits.
    """
    if reset_qubits is None:
        reset_qubits = list(range(nqubits))
    n_clbits = nqubits + extra_clbits
    qc = QuantumCircuit(nqubits, n_clbits)
    if prepare_ops is not None:
        prepare_ops(qc)
    for q in reset_qubits:
        qc.reset(q)
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))
    return qc


@pytest.mark.parametrize("nqubits,circuit_class", [
    (2, BinaryCircuit),
    (3, BinaryCircuit),
    (2, EfficientCircuit),
    (3, EfficientCircuit),
])
def test_reset_already_zero_qubit(nqubits, circuit_class):
    """Resetting a qubit that is already |0⟩ should leave the state |0...0⟩."""
    circ = _build_reset_circuit(nqubits)          # No prep → qubits are |0⟩
    t_circ = _transpile_midcircuit(circ, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=100)

    zero_state = "0" * nqubits
    p_zero = result["probs"].get(zero_state, 0.0)
    assert p_zero > 0.99, (
        f"Expected P(|0...0⟩) > 0.99 after resetting already-zero qubits "
        f"but got {p_zero:.4f}  (probs={result['probs']})"
    )


@pytest.mark.parametrize("nqubits,circuit_class", [
    (2, BinaryCircuit),
    (3, BinaryCircuit),
    (2, EfficientCircuit),
    (3, EfficientCircuit),
])
def test_reset_after_x_gate(nqubits, circuit_class):
    """Resetting every qubit after flipping them all with X should return
    the system to |0...0⟩."""
    def flip_all(qc):
        for q in range(nqubits):
            qc.x(q)

    circ = _build_reset_circuit(nqubits, prepare_ops=flip_all)
    t_circ = _transpile_midcircuit(circ, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    zero_state = "0" * nqubits
    p_zero = result["probs"].get(zero_state, 0.0)
    assert p_zero > 0.95, (
        f"Expected P(|0...0⟩) > 0.95 after X+reset but got {p_zero:.4f} "
        f"(probs={result['probs']})"
    )


@pytest.mark.parametrize("circuit_class", [BinaryCircuit, EfficientCircuit])
def test_reset_single_qubit_leaves_others_intact(circuit_class):
    """Resetting only qubit 0 (after flipping it) should not disturb qubit 1."""
    nqubits = 2

    def flip_q0(qc):
        qc.x(0)   # q0 → |1⟩,  q1 stays |0⟩

    circ = _build_reset_circuit(nqubits, prepare_ops=flip_q0, reset_qubits=[0])
    t_circ = _transpile_midcircuit(circ, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    # Expected final state: q0=|0⟩, q1=|0⟩  →  "00"
    p_00 = result["probs"].get("00", 0.0)
    assert p_00 > 0.95, (
        f"Expected P('00') > 0.95 but got {p_00:.4f}  (probs={result['probs']})"
    )


@pytest.mark.parametrize("circuit_class", [BinaryCircuit, EfficientCircuit])
def test_reset_after_superposition(circuit_class):
    """Resetting a qubit in an equal superposition (|+⟩ = H|0⟩) should collapse
    it to |0⟩ regardless of which branch was sampled."""
    nqubits = 2

    def superpose_q0(qc):
        qc.h(0)   # q0 → |+⟩

    circ = _build_reset_circuit(nqubits, prepare_ops=superpose_q0, reset_qubits=[0])
    t_circ = _transpile_midcircuit(circ, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=300)

    # After reset, q0 must always be |0⟩ → only "00" should appear
    for bitstr, p in result["probs"].items():
        if p > 0.01:   # ignore floating-point dust
            assert bitstr[0] == "0", (
                f"After reset, qubit 0 must be |0⟩ but found outcome '{bitstr}' "
                f"with p={p:.4f}"
            )


@pytest.mark.parametrize("circuit_class", [BinaryCircuit, EfficientCircuit])
def test_reset_after_entanglement_breaks_correlation(circuit_class):
    """After a Bell state is prepared and one qubit is reset, the remaining
    qubit should no longer be correlated with it – the reset qubit is always
    |0⟩ and the other qubit is in a mixed state."""
    nqubits = 2

    def bell(qc):
        qc.h(0)
        qc.cx(0, 1)   # Bell state (|00⟩ + |11⟩) / √2

    circ = _build_reset_circuit(nqubits, prepare_ops=bell, reset_qubits=[0])
    t_circ = _transpile_midcircuit(circ, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=500)

    # q0 (leftmost bit in Qiskit's little-endian string) must always be 0
    for bitstr, p in result["probs"].items():
        if p > 0.01:
            assert bitstr[0] == "0", (
                f"Reset qubit should always be 0 but found '{bitstr}' "
                f"with p={p:.4f}"
            )

    # q1 should be roughly 50/50 between 0 and 1 (mixed after partial trace)
    p_q1_zero = sum(p for bs, p in result["probs"].items() if bs[1] == "0")
    assert 0.30 < p_q1_zero < 0.70, (
        f"Expected q1 to be ~50/50 after resetting entangled q0 "
        f"but got P(q1=0)={p_q1_zero:.4f}"
    )


@pytest.mark.parametrize("circuit_class", [BinaryCircuit, EfficientCircuit])
def test_reset_all_qubits_after_entanglement(circuit_class):
    """Resetting all qubits of a Bell state must return |00⟩ with certainty."""
    nqubits = 2

    def bell(qc):
        qc.h(0)
        qc.cx(0, 1)

    circ = _build_reset_circuit(nqubits, prepare_ops=bell)
    t_circ = _transpile_midcircuit(circ, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=300)

    p_00 = result["probs"].get("00", 0.0)
    assert p_00 > 0.95, (
        f"Expected P('00') > 0.95 after full reset of Bell state "
        f"but got {p_00:.4f}  (probs={result['probs']})"
    )


@pytest.mark.parametrize("circuit_class", [BinaryCircuit, EfficientCircuit])
def test_multiple_consecutive_resets(circuit_class):
    """Applying reset twice on the same qubit should be idempotent – the
    qubit ends up in |0⟩ after both resets."""
    nqubits = 2

    nq = nqubits
    qc = QuantumCircuit(nq, nq)
    qc.x(0)      # |1⟩
    qc.reset(0)  # → |0⟩
    qc.reset(0)  # → |0⟩  (idempotent)
    qc.barrier()
    qc.measure(range(nq), range(nq))

    t_circ = _transpile_midcircuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=200)

    p_00 = result["probs"].get("00", 0.0)
    assert p_00 > 0.95, (
        f"Expected P('00') > 0.95 after double reset but got {p_00:.4f}"
    )


@pytest.mark.parametrize("circuit_class", [BinaryCircuit, EfficientCircuit])
def test_reset_produces_no_mid_counts_entry(circuit_class):
    """A plain reset (no separate mid-circuit measure instruction) should not
    add an entry to `mid_counts` – the collapse is internal to the reset op."""
    nqubits = 2

    def flip_all(qc):
        for q in range(nqubits):
            qc.x(q)

    circ = _build_reset_circuit(nqubits, prepare_ops=flip_all)
    t_circ = _transpile_midcircuit(circ, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=100)

    # mid_counts should not contain an explicit classical write from reset
    # (reset uses cbit_list=None internally)
    for shot in result["results"]:
        for event in shot["mid"]:
            # Any recorded mid event must not have been written to a classical bit
            # by the reset path – cbit_list=None means no clbit write
            assert event.get("cbit") is None or event.get("cbit") == [], (
                f"Reset should not write to a classical bit but found {event}"
            )


@pytest.mark.parametrize("n_cycles,circuit_class", [
    (1, BinaryCircuit),
    (2, BinaryCircuit),
    (3, BinaryCircuit),
    (1, EfficientCircuit),
    (2, EfficientCircuit),
    (3, EfficientCircuit),
])
def test_reset_in_repeated_measure_reset_cycle(n_cycles, circuit_class):
    """Measure qubit 0 into an ancilla clbit, then reset it, repeat n_cycles
    times.  In the zero-noise case with the initial state |0...0⟩, every
    syndrome measurement should read 0."""
    nqubits = 2
    n_clbits = nqubits + n_cycles   # one extra clbit per cycle
    qc = QuantumCircuit(nqubits, n_clbits)

    for cycle in range(n_cycles):
        qc.measure(0, nqubits + cycle)   # measure q0 to ancilla clbit
        qc.reset(0)                            # reset q0 back to |0⟩
        qc.barrier()

    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_midcircuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates, circuit_class, shots=100)

    # All mid-circuit measurement outcomes should be 0
    for shot in result["results"]:
        for event in shot["mid"]:
            for o in event["outcome"]:
                assert o == 0, (
                    f"Expected syndrome 0 from |0⟩ but got {o} "
                    f"at step {event['step']}"
                )


@pytest.mark.parametrize("circuit_class", [BinaryCircuit, EfficientCircuit])
def test_reset_noise_free_vs_noisy(circuit_class):
    """In the zero-noise regime, reset after X should return |0...0⟩ with
    much higher fidelity than with standard (noisy) gates."""
    nqubits = 2

    def flip_all(qc):
        for q in range(nqubits):
            qc.x(q)

    circ = _build_reset_circuit(nqubits, prepare_ops=flip_all)
    t_circ = _transpile_midcircuit(circ, nqubits)

    result_clean = _run_sim(t_circ, nqubits, gates=almost_noise_free_gates,
                            circuit_class=circuit_class, shots=300)
    result_noisy = _run_sim(t_circ, nqubits, gates=standard_gates,
                            circuit_class=circuit_class, shots=300)

    zero_state = "0" * nqubits
    p_clean = result_clean["probs"].get(zero_state, 0.0)
    p_noisy = result_noisy["probs"].get(zero_state, 0.0)

    assert p_clean > p_noisy, (
        f"Expected noise-free reset to yield higher P(|0...0⟩) than noisy, "
        f"but got p_clean={p_clean:.4f} <= p_noisy={p_noisy:.4f}"
    )
    assert p_clean > 0.90, (
        f"Expected P(|0...0⟩) > 0.90 for noise-free reset but got {p_clean:.4f}"
    )


@pytest.mark.parametrize("n_resets,circuit_class", [
    (1, BinaryCircuit),
    (3, BinaryCircuit),
    (5, BinaryCircuit),
    (1, EfficientCircuit),
    (3, EfficientCircuit),
    (5, EfficientCircuit),
])
def test_repeated_reset_noise_accumulation(n_resets, circuit_class):
    """Applying more resets on a noisy system should not dramatically degrade
    fidelity compared with a single reset – each reset itself brings the qubit
    back toward |0⟩, so noise should not compound unboundedly."""
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    for _ in range(n_resets):
        qc.x(0)
        qc.reset(0)
    qc.barrier()
    qc.measure(range(nqubits), range(nqubits))

    t_circ = _transpile_midcircuit(qc, nqubits)
    result = _run_sim(t_circ, nqubits, gates=almost_noise_free_gates,
                      circuit_class=circuit_class, shots=300)

    p_00 = result["probs"].get("00", 0.0)
    # Each reset should keep the qubit near |0⟩; fidelity should stay reasonable
    assert p_00 > 0.80, (
        f"Expected P('00') > 0.80 after {n_resets} X+reset cycles but got "
        f"{p_00:.4f}"
    )
