import numpy as np
import os
import pytest

from src.quantum_gates.utilities import post_process_split
from src.quantum_gates._utility.simulations_utility import apply_phase_to_qubit


location = 'tests/helpers/result_samples'


def test_apply_phase_to_qubit_raises_on_real_array():
    """Passing a real-valued array raises TypeError."""
    # Arrange
    psi = np.array([1.0, 0.0])

    # Act & Assert
    with pytest.raises(TypeError, match="Psi must be a complex array"):
        apply_phase_to_qubit(psi, qubit=0, dim=2, n=1, phase=np.pi / 4)


def test_apply_phase_to_qubit_zero_phase_is_identity():
    """Applying phase=0 leaves the statevector unchanged."""
    # Arrange
    psi = np.array([0.5, 0.5j, -0.5, -0.5j])
    expected = psi.copy()

    # Act
    result = apply_phase_to_qubit(psi.copy(), qubit=0, dim=4, n=2, phase=0.0)

    # Assert
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."


def test_apply_phase_to_qubit_zero_state_unaffected():
    """Applying any phase when qubit is always |0> leaves statevector unchanged."""
    # Arrange
    psi = np.array([1.0 + 0j, 0.0, 0.0, 0.0])  # |00>
    expected = psi.copy()

    # Act
    result = apply_phase_to_qubit(psi.copy(), qubit=0, dim=4, n=2, phase=np.pi / 3)

    # Assert
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."


def test_apply_phase_to_qubit_one_state():
    """Applying phase to |1> single-qubit state multiplies amplitude by exp(i*phase)."""
    # Arrange
    phase = np.pi / 4
    psi = np.array([0.0 + 0j, 1.0 + 0j])  # |1>
    expected = np.array([0.0, np.exp(1j * phase)])

    # Act
    result = apply_phase_to_qubit(psi.copy(), qubit=0, dim=2, n=1, phase=phase)

    # Assert
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."


def test_apply_phase_to_qubit_superposition():
    """Applying phase pi to qubit in |+> state gives |-> state."""
    # Arrange
    psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)  # |+>
    expected = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])  # |->

    # Act
    result = apply_phase_to_qubit(psi.copy(), qubit=0, dim=2, n=1, phase=np.pi)

    # Assert
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."


def test_apply_phase_to_qubit_two_qubits_targets_correct_qubit():
    """For 2 qubits, phase is applied only to amplitudes where the target qubit is |1>."""
    # Arrange
    # Equal superposition: |00>, |01>, |10>, |11> each with amplitude 0.5
    psi = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
    phase = np.pi / 2

    # Act, Assert
    result = apply_phase_to_qubit(psi.copy(), qubit=0, dim=4, n=2, phase=phase)
    expected = np.array([0.5, 0.5, 0.5 * np.exp(1j * phase), 0.5 * np.exp(1j * phase)])
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."

    # Act, Assert
    result = apply_phase_to_qubit(psi.copy(), qubit=1, dim=4, n=2, phase=phase)
    expected = np.array([0.5, 0.5 * np.exp(1j * phase), 0.5, 0.5 * np.exp(1j * phase)])
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."


def test_apply_phase_to_qubit_2pi_is_identity():
    """Applying phase 2*pi is equivalent to identity up to floating point."""
    # Arrange
    psi = np.array([0.6, 0.8j])

    # Act
    result = apply_phase_to_qubit(psi.copy(), qubit=0, dim=2, n=1, phase=2 * np.pi)

    # Assert
    assert np.allclose(result, psi), f"Expected {psi} but got {result}."


def test_post_process_split_mean():
    # Arrange
    source_filenames = [f"{location}/file{i}.txt" for i in range(1, 5)]
    target_filenames = [f"{location}/target.txt"]

    # Act
    post_process_split(source_filenames, target_filenames, 4)
    mean_array = np.loadtxt(target_filenames[0])

    # Assert
    mean_expected = np.array([0.8, 0.05, 0.05, 0.1])
    assert all((abs(mean_array[i] - mean_expected[i]) < 1e-9 for i in range(len(source_filenames)))), \
        f"Expected {mean_expected} but found {mean_array}."
    for f in target_filenames:
        os.remove(f)
