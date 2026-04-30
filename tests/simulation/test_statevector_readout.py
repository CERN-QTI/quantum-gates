import pytest
import numpy as np

from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit

from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import BinaryCircuit, EfficientCircuit
from src.quantum_gates.gates import standard_gates, almost_noise_free_gates
from src.quantum_gates.utilities import DeviceParameters
from src.quantum_gates.utilities import (
    sv_normal_to_qiskit,
) 

_backend = FakeBrisbane()
_location = "tests/helpers/device_parameters/ibm_kyoto/"
_LAYOUT = [0, 1, 2, 3, 4]

ABS_TOL = 1e-6


def _device_param(nqubits):
    """Load device parameters for a linear layout of nqubits."""
    dp = DeviceParameters(qubits_layout=_LAYOUT[:nqubits])
    dp.load_from_texts(location=_location)
    return dp.__dict__()


def _transpile(circ, nqubits):
    """Transpile a standard circuit (no mid-circuit ops) with ASAP scheduling."""
    return transpile(
        circ,
        backend=_backend,
        initial_layout=_LAYOUT[:nqubits],
        scheduling_method='asap',
        seed_transpiler=42,
    )


def _run_sim(t_circ, nqubits, gates,
             circuit_class, shots=10):
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


def _sv_from_shot(shot_readout, label):
    """Extract the statevector for a given label from one shot's readout list."""
    for lbl, sv in shot_readout:
        if lbl == label:
            return sv
    return None


def _probs_from_sv(sv):
    """Convert a statevector to a probability array."""
    return np.abs(sv) ** 2


def _is_normalized(sv, tol=1e-6):
    return abs(np.sum(np.abs(sv) ** 2) - 1.0) < tol


def _almost_equal_sv(sv1, sv2, tol=ABS_TOL):
    """Check statevectors are equal up to a global phase and tolerance."""
    probs1 = _probs_from_sv(sv1)
    probs2 = _probs_from_sv(sv2)
    return np.allclose(probs1, probs2, atol=tol)


circuits = [BinaryCircuit, EfficientCircuit]


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_key_exists(circuit_class, gates=almost_noise_free_gates):
    """result['statevector_readout'] must always be present."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=5)

    # Assert
    assert "statevector_readout" in result, \
        "Expected 'statevector_readout' key in result dict"


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_is_list_of_tuples(circuit_class, gates=almost_noise_free_gates):
    """Each shot's readout should be a list of (label, array) tuples."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))
    
    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=5)
    readout = result["statevector_readout"]

    # Assert
    assert isinstance(readout, list), "statevector_readout should be a list"
    for shot_readout in readout:
        assert isinstance(shot_readout, list), \
            f"Each shot readout should be a list but got {type(shot_readout)}"
        for entry in shot_readout:
            assert isinstance(entry, tuple) and len(entry) == 2, \
                f"Each entry should be a (label, array) tuple but got {entry}"
            label, sv = entry
            assert isinstance(label, str), \
                f"Label should be a string but got {type(label)}"
            assert isinstance(sv, np.ndarray), \
                f"Statevector should be an ndarray but got {type(sv)}"


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_length_matches_shots(circuit_class, gates=almost_noise_free_gates):
    """statevector_readout should have one entry per shot."""
    # Arrange
    nqubits = 2
    shots = 7
    qc = QuantumCircuit(nqubits, nqubits)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots)

    # Assert
    assert len(result["statevector_readout"]) == shots, (
        f"Expected {shots} readout entries (one per shot) but got "
        f"{len(result['statevector_readout'])}"
    )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_correct_dimension(circuit_class, gates=almost_noise_free_gates):
    """Each saved statevector should have dimension 2^nqubits."""
    for nqubits in [2, 3]:
        # Arrange
        qc = QuantumCircuit(nqubits, nqubits)
        qc.barrier(label="save_sv_0")
        qc.measure(range(nqubits), range(nqubits))

        # Act
        result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=3)

        # Assert
        for shot_readout in result["statevector_readout"]:
            for label, sv in shot_readout:
                assert len(sv) == 2 ** nqubits, (
                    f"Expected statevector of length {2**nqubits} for "
                    f"{nqubits} qubits but got {len(sv)}"
                )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_zero_state(circuit_class, gates=almost_noise_free_gates):
    """Readout immediately on |0...0⟩ should recover the computational zero state."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.barrier(label="save_sv_0")   # no gates applied → |00⟩
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=5)

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv = _sv_from_shot(shot_readout, "save_sv_0")
        assert sv is not None, "Expected label 'save_sv_0' in readout"
        probs = _probs_from_sv(sv)
        assert probs[0] == pytest.approx(1.0, abs=ABS_TOL), (
            f"Expected P(|00⟩)=1 but got probs={probs}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_after_x_gate(circuit_class, gates=almost_noise_free_gates):
    """After X on all qubits, the saved statevector should be |1...1⟩."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    for q in range(nqubits):
        qc.x(q)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=5)

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv = _sv_from_shot(shot_readout, "save_sv_0")
        probs = _probs_from_sv(sv)
        # |11⟩ is the last basis state
        assert probs[-1] == pytest.approx(1.0, abs=ABS_TOL), (
            f"Expected P(|11⟩)=1 after X gates but got probs={probs}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_superposition(circuit_class, gates=almost_noise_free_gates):
    """After H on qubit 0, the saved statevector should show 50/50 on |0⟩ and |1⟩."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=5)

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv = _sv_from_shot(shot_readout, "save_sv_0")
        probs = _probs_from_sv(sv)
        # |00⟩ and |10⟩ (or |01⟩ depending on qubit ordering) should each be ~0.5
        p_q0_zero = probs[0] + probs[1]   # q0=0 states
        p_q0_one  = probs[2] + probs[3]   # q0=1 states
        assert p_q0_zero == pytest.approx(0.5, abs=0.05), (
            f"Expected P(q0=0)≈0.5 in |+⟩ state but got {p_q0_zero:.4f}"
        )
        assert p_q0_one == pytest.approx(0.5, abs=0.05), (
            f"Expected P(q0=1)≈0.5 in |+⟩ state but got {p_q0_one:.4f}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_is_normalized(circuit_class, gates=almost_noise_free_gates):
    """Every saved statevector must be a unit vector."""
    # Arrange
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=10)

    # Assert
    for shot_readout in result["statevector_readout"]:
        for label, sv in shot_readout:
            assert _is_normalized(sv), (
                f"Statevector at '{label}' is not normalized: "
                f"‖ψ‖²={np.sum(np.abs(sv)**2):.6f}"
            )


@pytest.mark.parametrize("n_barriers,circuit_class", [(n, c) for n in [2, 3, 4] for c in circuits])
def test_multiple_barriers_correct_count(n_barriers, circuit_class, gates=almost_noise_free_gates):
    """Each shot's readout should contain exactly n_barriers entries."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    for c in range(n_barriers):
        qc.h(0)
        qc.barrier(label=f"save_sv_{c}")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=5)

    # Assert
    for i, shot_readout in enumerate(result["statevector_readout"]):
        assert len(shot_readout) == n_barriers, (
            f"Shot {i}: expected {n_barriers} saved statevectors but got "
            f"{len(shot_readout)}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_multiple_barriers_labels_are_correct(circuit_class, gates=almost_noise_free_gates):
    """Labels in the readout should match the barrier labels in the circuit."""
    # Arrange
    nqubits = 2
    n_barriers = 3
    qc = QuantumCircuit(nqubits, nqubits)
    for c in range(n_barriers):
        qc.barrier(label=f"save_sv_{c}")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=3)
    expected_labels = {f"save_sv_{c}" for c in range(n_barriers)}

    # Assert
    for shot_readout in result["statevector_readout"]:
        found_labels = {lbl for lbl, _ in shot_readout}
        assert found_labels == expected_labels, (
            f"Expected labels {expected_labels} but found {found_labels}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_multiple_barriers_statevectors_evolve(circuit_class, gates=almost_noise_free_gates):
    """Successive save points should capture different states as the circuit evolves.

    Circuit: |0⟩ → save_sv_0 → H → save_sv_1 → X → save_sv_2
    sv_0 should be |00⟩, sv_1 should be a superposition, sv_2 should differ from sv_1.
    """
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)

    qc.barrier(label="save_sv_0")   # |00⟩
    qc.h(0)
    qc.barrier(label="save_sv_1")   # (|0⟩+|1⟩)/√2 ⊗ |0⟩
    qc.x(1)
    qc.barrier(label="save_sv_2")   # another state
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=5)

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv0 = _sv_from_shot(shot_readout, "save_sv_0")
        sv1 = _sv_from_shot(shot_readout, "save_sv_1")
        sv2 = _sv_from_shot(shot_readout, "save_sv_2")
        # sv0 should be |00⟩
        assert _probs_from_sv(sv0)[0] == pytest.approx(1.0, abs=ABS_TOL), \
            "save_sv_0 should be |00⟩"

        # sv1 should be a superposition (no single state dominates)
        probs1 = _probs_from_sv(sv1)
        assert np.max(probs1) < 0.99, \
            f"save_sv_1 should be a superposition but got probs={probs1}"

        # sv2 should differ from sv1 (X was applied between them)
        assert not _almost_equal_sv(sv1, sv2), \
            "save_sv_1 and save_sv_2 should differ after X gate"


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_after_mid_measurement(circuit_class, gates=almost_noise_free_gates):
    """A statevector saved after a mid-circuit measurement should reflect the
    post-measurement collapsed state, not the pre-measurement superposition."""
    # Arrange
    nqubits = 2
    n_clbits = nqubits + 1
    qc = QuantumCircuit(nqubits, n_clbits)

    qc.h(0)
    qc.cx(0, 1)                     # Bell state
    qc.measure(0, nqubits)          # mid-circuit measure q0
    qc.rz(0.001, 0)                 # keep as mid-circuit
    qc.barrier(label="save_sv_0")   # save after collapse
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=20)

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv = _sv_from_shot(shot_readout, "save_sv_0")
        assert sv is not None
        probs = _probs_from_sv(sv)
        # After collapsing a Bell state, the system is in a product state:
        # either |00⟩ or |11⟩ — one basis state should dominate
        assert np.max(probs) > 0.95, (
            f"After mid-circuit measurement, statevector should be near a "
            f"basis state but got probs={probs}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_before_and_after_mid_measurement(circuit_class, gates=almost_noise_free_gates):
    """Save before and after a mid-circuit measurement.
    Before: Bell state (no single state dominates).
    After: collapsed to a product state (one state dominates)."""
    # Arrange
    nqubits = 2
    n_clbits = nqubits + 1
    qc = QuantumCircuit(nqubits, n_clbits)

    qc.h(0)
    qc.cx(0, 1)
    qc.barrier(label="save_sv_0")   # before collapse — Bell state
    qc.measure(0, nqubits)
    qc.rz(0.001, 0)
    qc.barrier(label="save_sv_1")   # after collapse — product state
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=20)

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv_before = _sv_from_shot(shot_readout, "save_sv_0")
        sv_after  = _sv_from_shot(shot_readout, "save_sv_1")

        # Before: Bell state — no single basis state has probability > 0.6
        probs_before = _probs_from_sv(sv_before)
        assert np.max(probs_before) < 0.75, (
            f"Expected Bell state before mid-measure but got probs={probs_before}"
        )

        # After: collapsed — one basis state dominates
        probs_after = _probs_from_sv(sv_after)
        assert np.max(probs_after) > 0.90, (
            f"Expected collapsed state after mid-measure but got probs={probs_after}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_before_and_after_reset(circuit_class, gates=almost_noise_free_gates):
    """Save before and after a reset.
    Before: |1⟩ state. After: |0⟩ state."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)

    qc.x(0)
    qc.barrier(label="save_sv_0")   # should be |10⟩ (or |01⟩ depending on ordering)
    qc.reset(0)
    qc.barrier(label="save_sv_1")   # should be |00⟩
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=10)

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv_before = _sv_from_shot(shot_readout, "save_sv_0")
        sv_after  = _sv_from_shot(shot_readout, "save_sv_1")

        probs_before = _probs_from_sv(sv_before)
        probs_after  = _probs_from_sv(sv_after)

        # Before reset: |10⟩ — zero state should NOT dominate
        assert probs_before[0] < 0.1, (
            f"Before reset, |00⟩ should not dominate but got P(|00⟩)="
            f"{probs_before[0]:.4f}"
        )

        # After reset: |00⟩ — zero state should dominate
        assert probs_after[0] > 0.90, (
            f"After reset, expected P(|00⟩)>0.90 but got {probs_after[0]:.4f}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_reset_from_superposition(circuit_class, gates=almost_noise_free_gates):
    """Save after resetting a qubit that was in superposition.
    The saved statevector should show |0⟩ on the reset qubit regardless of
    which branch was sampled."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)

    qc.h(0)
    qc.reset(0)
    qc.barrier(label="save_sv_0")   # q0 must always be |0⟩ after reset
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result =_run_sim(_transpile(qc, nqubits), nqubits, gates, circuit_class, shots=30)

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv = _sv_from_shot(shot_readout, "save_sv_0")
        probs = _probs_from_sv(sv)
        # q0=0 states are indices 0 and 1 (q1 can be anything)
        p_q0_zero = probs[0] + probs[1]
        assert p_q0_zero > 0.95, (
            f"After reset from superposition, q0 should be |0⟩ but "
            f"P(q0=0)={p_q0_zero:.4f}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_noise_free_is_pure(circuit_class):
    """In the noise-free regime, the saved statevector should be very close to
    a pure state (one amplitude dominates after a deterministic preparation)."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    for q in range(nqubits):
        qc.x(q)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    result = _run_sim(_transpile(qc, nqubits), nqubits,
                      gates=almost_noise_free_gates,
                      circuit_class=circuit_class, shots=10)

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv = _sv_from_shot(shot_readout, "save_sv_0")
        probs = _probs_from_sv(sv)
        assert np.max(probs) > 0.99, (
            f"Noise-free |11⟩ preparation should give a near-pure statevector "
            f"but got probs={probs}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_noise_free_closer_to_ideal_than_noisy(circuit_class):
    """The noise-free statevector should be closer to the ideal |11⟩ state
    than the noisy one."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    for q in range(nqubits):
        qc.x(q)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    t_circ = _transpile(qc, nqubits)

    result_clean = _run_sim(t_circ, nqubits, almost_noise_free_gates,
                            circuit_class, shots=20)
    result_noisy = _run_sim(t_circ, nqubits, standard_gates,
                            circuit_class, shots=20)

    # Assert
    def avg_p_last(readout):
        """Average probability of the last basis state (|11⟩) across shots."""
        ps = []
        for shot_readout in readout:
            sv = _sv_from_shot(shot_readout, "save_sv_0")
            ps.append(_probs_from_sv(sv)[-1])
        return np.mean(ps)

    p_clean = avg_p_last(result_clean["statevector_readout"])
    p_noisy = avg_p_last(result_noisy["statevector_readout"])
    if p_noisy > 1:
        p_noisy = 2-p_noisy  # prob can be larger then 1 due to non-unitary noise
    assert p_clean > p_noisy, (
        f"Expected noise-free P(|11⟩) > noisy P(|11⟩) but got "
        f"p_clean={p_clean:.4f} <= p_noisy={p_noisy:.4f}"
    )


def _qiskit_statevectors(qc_with_saves, save_labels):
    """Run the circuit on Qiskit's AerSimulator and return a dict of
    {label: statevector} for each save_statevector instruction.

    We insert Qiskit's save_statevector instructions at the same points as
    our barrier(label="save_sv_*") markers, then read back the snapshots.
    """
    sim = AerSimulator(method="statevector")
    # AerSimulator needs save_statevector instructions injected at barrier points
    # Rebuild the circuit replacing each save barrier with save_statevector

    # Find the indices of qubits actually used in the transpiled circuit
    used_indices = sorted({
        qc_with_saves.find_bit(q).index
        for op in qc_with_saves.data
        for q in op.qubits
        if op.operation.name not in ("barrier", "delay")
    })
    n_used = len(used_indices)
    idx_map = {old: new for new, old in enumerate(used_indices)}
    
    new_qc = QuantumCircuit(n_used, qc_with_saves.num_clbits)

    for op in qc_with_saves.data:
        name = op.operation.name
        if name == "barrier" and op.operation.label in save_labels:
            new_qc.save_statevector(label=op.operation.label, pershot=False)
        elif name == "delay":
            qubit_idx = qc_with_saves.find_bit(op.qubits[0]).index
            if qubit_idx in idx_map:
                qargs = [new_qc.qubits[idx_map[qubit_idx]]]
                new_qc.append(op.operation, qargs, [])
        else:
            qargs = [qc_with_saves.find_bit(q).index for q in op.qubits]
            cargs = [qc_with_saves.find_bit(c).index for c in op.clbits]
            new_qc.append(op.operation, qargs, cargs)

    job = sim.run(new_qc, shots=1)
    result = job.result()
    data = result.data(0)
    return {label: np.array(data[label]) for label in save_labels}


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_matches_qiskit_zero_state(circuit_class):
    """On |0...0⟩ with no gates, our saved statevector should match Qiskit's
    AerSimulator output exactly (up to global phase)."""
    # Arrange
    nqubits = 2
    save_labels = ["save_sv_0"]

    qc = QuantumCircuit(nqubits, nqubits)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    t_circ = _transpile(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates,
                      circuit_class, shots=5)

    qiskit_svs = _qiskit_statevectors(t_circ, save_labels)
    qiskit_probs = _probs_from_sv(qiskit_svs["save_sv_0"])

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv = _sv_from_shot(shot_readout, "save_sv_0")
        sv_qiskit_perm = sv_normal_to_qiskit(sv)
        our_probs = _probs_from_sv(sv_qiskit_perm)
        assert np.allclose(our_probs, qiskit_probs, atol=1e-3), (
            f"Statevector probabilities do not match Qiskit AerSimulator.\n"
            f"Ours:   {our_probs}\n"
            f"Qiskit: {qiskit_probs}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_matches_qiskit_after_x(circuit_class):
    """After X on all qubits, our saved statevector probabilities should match
    Qiskit's AerSimulator (up to noise tolerance)."""
    # Arrange
    nqubits = 2
    save_labels = ["save_sv_0"]

    qc = QuantumCircuit(nqubits, nqubits)
    for q in range(nqubits):
        qc.x(q)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    t_circ = _transpile(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates,
                      circuit_class, shots=5)

    qiskit_svs = _qiskit_statevectors(t_circ, save_labels)
    qiskit_probs = _probs_from_sv(qiskit_svs["save_sv_0"])

    # Assert
    for shot_readout in result["statevector_readout"]:
        sv = _sv_from_shot(shot_readout, "save_sv_0")
        sv_qiskit_perm = sv_normal_to_qiskit(sv)
        our_probs = _probs_from_sv(sv_qiskit_perm)
        assert np.allclose(our_probs, qiskit_probs, atol=1e-2), (
            f"Statevector probabilities do not match Qiskit AerSimulator.\n"
            f"Ours:   {our_probs}\n"
            f"Qiskit: {qiskit_probs}"
        )


@pytest.mark.parametrize("circuit_class", circuits)
def test_statevector_readout_matches_qiskit_multiple_snapshots(circuit_class):
    """With multiple save points, each snapshot should match Qiskit's
    AerSimulator at the corresponding point in the circuit."""
    # Arrange
    nqubits = 2
    save_labels = ["save_sv_0", "save_sv_1", "save_sv_2"]

    qc = QuantumCircuit(nqubits, nqubits)
    qc.barrier(label="save_sv_0")   # |00⟩
    qc.h(0)
    qc.barrier(label="save_sv_1")   # superposition
    qc.h(0)
    qc.barrier(label="save_sv_2")   # back to |00⟩
    qc.measure(range(nqubits), range(nqubits))

    # Act
    t_circ = _transpile(qc, nqubits)
    result = _run_sim(t_circ, nqubits, almost_noise_free_gates,
                      circuit_class, shots=5)

    qiskit_svs = _qiskit_statevectors(t_circ, save_labels)

    # Assert
    for shot_readout in result["statevector_readout"]:
        for label in save_labels:
            sv = _sv_from_shot(shot_readout, label)
            sv_qiskit_perm = sv_normal_to_qiskit(sv)
            our_probs = _probs_from_sv(sv_qiskit_perm)
            qiskit_probs = _probs_from_sv(qiskit_svs[label])
            assert np.allclose(our_probs, qiskit_probs, atol=1e-2), (
                f"Snapshot '{label}' does not match Qiskit AerSimulator.\n"
                f"Ours:   {our_probs}\n"
                f"Qiskit: {qiskit_probs}"
            )