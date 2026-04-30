import pytest
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from src.quantum_gates.utilities import (
    sv_normal_to_qiskit,
    sv_qiskit_to_normal,
    extract_qubit_orders,
    permute_qiskit_sv_to_logical,
    permute_normal_sv_to_logical_normal,
)


_backend = FakeBrisbane()
_LAYOUT = [0, 1, 2, 3, 4]


def _basis_state_normal(bitstring: str) -> np.ndarray:
    """Return the normal-ordered basis state for a bitstring like '01'.
    Normal ordering: q0 is MSB, so '0100' → index 1."""
    n = len(bitstring)
    sv = np.zeros(2 ** n, dtype=complex)
    idx = int(bitstring, 2)
    sv[idx] = 1.0
    return sv


def _basis_state_qiskit(bitstring: str) -> np.ndarray:
    """Return the Qiskit-ordered basis state for a bitstring like '01'.
    Qiskit ordering: q0 is LSB, so '0010' → index 2
    but with reversed bit significance."""
    n = len(bitstring)
    sv = np.zeros(2 ** n, dtype=complex)
    # In Qiskit, index = sum(bit_i * 2^i) where bit_0 is rightmost in bitstring
    idx = int(bitstring[::-1], 2)
    sv[idx] = 1.0
    return sv


def test_sv_normal_to_qiskit_zero_state():
    """Normal |00⟩ should map to Qiskit |00⟩ — same index 0."""
    # Arrange
    sv = _basis_state_normal("00")

    # Act
    result = sv_normal_to_qiskit(sv)

    # Assert
    expected = np.array([1, 0, 0, 0], dtype=complex)
    assert np.allclose(result, expected), f"Expected {expected} but got {result}"


def test_sv_normal_to_qiskit_single_qubit_one():
    """Normal |1⟩ (index 1) should map to Qiskit |1⟩ (index 1) for 1 qubit."""
    # Arrange
    sv = np.array([0, 1], dtype=complex)

    # Act
    result = sv_normal_to_qiskit(sv)

    # Assert
    assert np.allclose(result, np.array([0, 1], dtype=complex))


@pytest.mark.parametrize("bitstring,expected_qiskit_idx", [
    ("00", 0),   # |00⟩ → index 0 in both
    ("01", 2),   # normal q0=0,q1=1 → qiskit q0=LSB → index 0b10 = 2
    ("10", 1),   # normal q0=1,q1=0 → qiskit → index 0b01 = 1
    ("11", 3),   # |11⟩ → index 3 in both
])
def test_sv_normal_to_qiskit_two_qubit_basis_states(bitstring, expected_qiskit_idx):
    """Each 2-qubit normal basis state should map to the correct Qiskit index."""
    # Arrange
    sv = _basis_state_normal(bitstring)

    # Act
    result = sv_normal_to_qiskit(sv)

    # Assert
    assert result[expected_qiskit_idx] == pytest.approx(1.0), (
        f"Normal |{bitstring}⟩ should map to Qiskit index {expected_qiskit_idx} "
        f"but got {result}"
    )


def test_sv_normal_to_qiskit_preserves_norm():
    """Conversion should preserve the norm of the statevector."""
    # Arrange
    sv = np.array([1, 1, 1, 1], dtype=complex) / 2.0

    # Act
    result = sv_normal_to_qiskit(sv)

    # Assert
    assert np.sum(np.abs(result) ** 2) == pytest.approx(1.0, abs=1e-9)


def test_sv_normal_to_qiskit_accepts_statevector_object():
    """Should accept a Qiskit Statevector object as input."""
    # Arrange
    sv = Statevector([1, 0, 0, 0])

    # Act
    result = sv_normal_to_qiskit(sv)

    # Assert
    assert isinstance(result, np.ndarray)
    assert result[0] == pytest.approx(1.0)


def test_sv_normal_to_qiskit_invalid_length():
    """Should raise ValueError for non-power-of-2 length."""
    with pytest.raises(ValueError):
        sv_normal_to_qiskit(np.array([1, 0, 0], dtype=complex))


def test_sv_qiskit_to_normal_zero_state():
    """Qiskit |00⟩ should map to normal |00⟩."""
    # Arrange
    sv = np.array([1, 0, 0, 0], dtype=complex)

    # Act
    result = sv_qiskit_to_normal(sv)

    # Assert
    assert np.allclose(result, np.array([1, 0, 0, 0], dtype=complex))


@pytest.mark.parametrize("bitstring,expected_normal_idx, expected_qiskit_idx", [
    ("00", 0, 0),
    ("01", 1, 2),
    ("10", 2, 1),
    ("11", 3, 3),
])
def test_sv_qiskit_to_normal_two_qubit_basis_states(bitstring, expected_normal_idx, expected_qiskit_idx):
    """Each 2-qubit Qiskit basis state should map to the correct normal index."""
    # Arrange
    sv = _basis_state_qiskit(bitstring)

    # Act
    result = sv_qiskit_to_normal(sv)

    # Assert
    assert result[expected_normal_idx] == pytest.approx(1.0), (
        f"Qiskit |{bitstring}⟩ should map to normal index {expected_normal_idx} "
        f"but got {result}"
    )
    assert sv[expected_qiskit_idx] == pytest.approx(1.0), (
        f"Qiskit |{bitstring}⟩ should map to qiskit index {expected_qiskit_idx} "
        f"but got {result}"
    )


def test_sv_qiskit_to_normal_preserves_norm():
    """Conversion should preserve the norm."""
    # Arrange
    sv = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)

    # Act
    result = sv_qiskit_to_normal(sv)

    # Assert
    assert np.sum(np.abs(result) ** 2) == pytest.approx(1.0, abs=1e-9)


def test_sv_qiskit_to_normal_invalid_length():
    """Should raise ValueError for non-power-of-2 length."""
    with pytest.raises(ValueError):
        sv_qiskit_to_normal(np.array([1, 0, 0], dtype=complex))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_roundtrip_normal_to_qiskit_to_normal(n):
    """normal → qiskit → normal should be the identity for any statevector."""
    # Arrange
    rng = np.random.default_rng(42)
    sv = rng.random(2 ** n) + 1j * rng.random(2 ** n)
    sv /= np.linalg.norm(sv)

    # Act
    result = sv_qiskit_to_normal(sv_normal_to_qiskit(sv))

    # Assert
    assert np.allclose(result, sv, atol=1e-12), (
        f"Roundtrip normal→qiskit→normal failed for n={n}"
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_roundtrip_qiskit_to_normal_to_qiskit(n):
    """qiskit → normal → qiskit should be the identity for any statevector."""
    # Arrange
    rng = np.random.default_rng(99)
    sv = rng.random(2 ** n) + 1j * rng.random(2 ** n)
    sv /= np.linalg.norm(sv)

    # Act
    result = sv_normal_to_qiskit(sv_qiskit_to_normal(sv))

    # Assert
    assert np.allclose(result, sv, atol=1e-12), (
        f"Roundtrip qiskit→normal→qiskit failed for n={n}"
    )


def test_permute_qiskit_sv_to_logical_identity():
    """Identity permutation [0, 1] should leave the statevector unchanged."""
    # Arrange
    sv = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)

    # Act
    result = permute_qiskit_sv_to_logical(sv, qubit_order=[0, 1])

    # Assert
    assert np.allclose(result, sv)


def test_permute_qiskit_sv_to_logical_swap():
    """Swapping qubits [1, 0] should permute the basis states correctly."""
    # Arrange
    # Qiskit |01⟩: q0=1, q1=0 → index 1
    sv = np.array([0, 1, 0, 0], dtype=complex)
    # After swapping logical↔physical: logical q0=physical q1, logical q1=physical q0
    # So Qiskit |01⟩ (q0=1,q1=0) becomes logical |10⟩ (q0=0,q1=1) → index 2

    # Act
    result = permute_qiskit_sv_to_logical(sv, qubit_order=[1, 0])

    # Assert
    assert result[2] == pytest.approx(1.0), (
        f"After qubit swap, expected amplitude at index 2 but got {result}"
    )


def test_permute_qiskit_sv_to_logical_preserves_norm():
    """Permutation should preserve the norm."""
    # Arrange
    rng = np.random.default_rng(7)
    sv = rng.random(8) + 1j * rng.random(8)
    sv /= np.linalg.norm(sv)

    # Act
    result = permute_qiskit_sv_to_logical(sv, qubit_order=[2, 0, 1])

    # Assert
    assert np.sum(np.abs(result) ** 2) == pytest.approx(1.0, abs=1e-9)


def test_permute_qiskit_sv_to_logical_three_qubits_cyclic():
    """Cyclic permutation [1, 2, 0] on a 3-qubit basis state."""
    # Qiskit basis state with only q1=1: index = 2^1 = 2
    sv = np.zeros(8, dtype=complex)
    sv[2] = 1.0
    result = permute_qiskit_sv_to_logical(sv, qubit_order=[1, 2, 0])
    # The result should be a valid basis state (one amplitude = 1)
    assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-9)
    assert np.sum(np.abs(result) ** 2) == pytest.approx(1.0, abs=1e-9)


def test_permute_normal_sv_to_logical_normal_identity():
    """Identity permutation [0, 1] should leave the statevector unchanged."""
    # Arrange
    sv = np.array([0.25, 0.25, 0.25, 0.25], dtype=complex)

    # Act
    result = permute_normal_sv_to_logical_normal(sv, qubit_order=[0, 1])

    # Assert
    assert np.allclose(result, sv)


def test_permute_normal_sv_to_logical_normal_swap():
    """Swapping qubits [1, 0] on normal |01⟩ (index 1) should give normal |10⟩ (index 2)."""
    # Arrange
    sv = _basis_state_normal("01")   # q0=0, q1=1 → index 1

    # Act
    result = permute_normal_sv_to_logical_normal(sv, qubit_order=[1, 0])

    # Assert
    assert result[2] == pytest.approx(1.0), (
        f"After qubit swap, expected amplitude at index 2 (|10⟩) but got {result}"
    )


def test_permute_normal_sv_to_logical_normal_preserves_norm():
    """Permutation should preserve the norm."""
    # Arrange
    rng = np.random.default_rng(13)
    sv = rng.random(8) + 1j * rng.random(8)
    sv /= np.linalg.norm(sv)

    # Act
    result = permute_normal_sv_to_logical_normal(sv, qubit_order=[2, 0, 1])

    # Assert
    assert np.sum(np.abs(result) ** 2) == pytest.approx(1.0, abs=1e-9)


def test_permute_normal_sv_to_logical_normal_three_qubits_basis():
    """Cyclic permutation on each 3-qubit basis state should produce a valid basis state."""
    qubit_order = [1, 2, 0]
    for i in range(8):
        # Arrange
        sv = np.zeros(8, dtype=complex)
        sv[i] = 1.0

        # Act
        result = permute_normal_sv_to_logical_normal(sv, qubit_order=qubit_order)

        # Assert
        assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-9), \
            f"Basis state {i} did not map to a valid basis state: {result}"
        assert np.sum(np.abs(result) ** 2) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("qubit_order", [[0, 1], [1, 0], [1, 2, 0], [2, 0, 1]])
def test_permute_consistency_via_conversion(qubit_order):
    """permute_normal then convert should equal convert then permute_qiskit.

    Both paths should produce the same result:
      Path A: permute_normal_sv_to_logical_normal → sv_normal_to_qiskit
      Path B: sv_normal_to_qiskit → permute_qiskit_sv_to_logical
    """
    # Arrange
    n = len(qubit_order)
    rng = np.random.default_rng(42)
    sv_normal = rng.random(2 ** n) + 1j * rng.random(2 ** n)
    sv_normal /= np.linalg.norm(sv_normal)

    # Act
    # Path A
    path_a = sv_normal_to_qiskit(
        permute_normal_sv_to_logical_normal(sv_normal, qubit_order)
    )

    # Path B
    path_b = permute_qiskit_sv_to_logical(
        sv_normal_to_qiskit(sv_normal), qubit_order
    )

    # Assert
    assert np.allclose(np.abs(path_a) ** 2, np.abs(path_b) ** 2, atol=1e-9), (
        f"Permutation paths A and B disagree for qubit_order={qubit_order}.\n"
        f"Path A probs: {np.abs(path_a)**2}\n"
        f"Path B probs: {np.abs(path_b)**2}"
    )
 
 
def _transpile(circ, nqubits):
    return transpile(
        circ,
        backend=_backend,
        initial_layout=_LAYOUT[:nqubits],
        seed_transpiler=42,
        optimization_level=0,
    )
 
 
def test_extract_qubit_orders_invalid_instruction_type():
    """Should raise ValueError for unknown instruction_type."""
    qc = QuantumCircuit(2, 2)
    with pytest.raises(ValueError):
        extract_qubit_orders(qc, instruction_type="invalid")
 
 
def test_extract_qubit_orders_barrier_empty_when_no_save():
    """Should return empty list when no save barriers are present."""
    # Arrange
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.barrier()   # regular barrier, no label
    qc.measure(range(2), range(2))

    # Act
    t_circ = _transpile(qc, 2)
    result = extract_qubit_orders(t_circ, instruction_type="barrier")

    # Assert
    assert result == [], f"Expected [] but got {result}"
 
 
def test_extract_qubit_orders_barrier_finds_save_labels():
    """Should return one entry per save barrier with the correct label."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.barrier(label="save_sv_0")
    qc.h(0)
    qc.barrier(label="save_sv_1")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    t_circ = _transpile(qc, nqubits)
 
    result = extract_qubit_orders(t_circ, instruction_type="barrier")
    labels = [r[0] for r in result]
 
    # Assert
    assert "save_sv_0" in labels, f"Expected 'save_sv_0' in {labels}"
    assert "save_sv_1" in labels, f"Expected 'save_sv_1' in {labels}"
 
 
def test_extract_qubit_orders_barrier_ignores_non_save_barriers():
    """Regular barriers without 'save' in the label should be ignored."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.barrier()                      # no label → ignored
    qc.barrier(label="sync")          # label without 'save' → ignored
    qc.barrier(label="save_sv_0")     # should be found
    qc.measure(range(nqubits), range(nqubits))

    # Act
    t_circ = _transpile(qc, nqubits)
 
    result = extract_qubit_orders(t_circ, instruction_type="barrier")

    # Assert
    assert len(result) == 1, f"Expected 1 entry but got {len(result)}: {result}"
    assert result[0][0] == "save_sv_0"
 
 
def test_extract_qubit_orders_barrier_qubit_indices_are_ints():
    """Qubit order entries should be lists of integers."""
    # Arrange
    nqubits = 2
    qc = QuantumCircuit(nqubits, nqubits)
    qc.barrier(label="save_sv_0")
    qc.measure(range(nqubits), range(nqubits))

    # Act
    t_circ = _transpile(qc, nqubits)
 
    result = extract_qubit_orders(t_circ, instruction_type="barrier")

    # Assert
    for label, qubit_order in result:
        assert isinstance(qubit_order, list), \
            f"qubit_order should be a list but got {type(qubit_order)}"
        for idx in qubit_order:
            assert isinstance(idx, int), \
                f"qubit index should be int but got {type(idx)}: {idx}"
 
 
@pytest.mark.parametrize("qubit_order", [[0, 1], [1, 0], [1, 2, 0], [2, 0, 1]])
def test_permute_consistency_via_conversion(qubit_order):
    """permute_normal then convert should equal convert then permute_qiskit.
 
    Both paths should produce the same result:
      Path A: permute_normal_sv_to_logical_normal → sv_normal_to_qiskit
      Path B: sv_normal_to_qiskit → permute_qiskit_sv_to_logical
    """
    # Arrange
    n = len(qubit_order)
    rng = np.random.default_rng(42)
    sv_normal = rng.random(2 ** n) + 1j * rng.random(2 ** n)
    sv_normal /= np.linalg.norm(sv_normal)
 
    # Act
    # Path A
    path_a = sv_normal_to_qiskit(
        permute_normal_sv_to_logical_normal(sv_normal, qubit_order)
    )
 
    # Path B
    path_b = permute_qiskit_sv_to_logical(
        sv_normal_to_qiskit(sv_normal), qubit_order
    )
 
    # Assert
    assert np.allclose(np.abs(path_a) ** 2, np.abs(path_b) ** 2, atol=1e-9), (
        f"Permutation paths A and B disagree for qubit_order={qubit_order}.\n"
        f"Path A probs: {np.abs(path_a)**2}\n"
        f"Path B probs: {np.abs(path_b)**2}"
    )
 