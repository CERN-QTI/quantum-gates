import numpy as np
import os
import pytest

from src.quantum_gates.utilities import post_process_split
from src.quantum_gates._utility.simulations_utility import apply_phase_to_qubit, apply_phase_corrections, compute_born_probability, collapse_statevector


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


def test_apply_phase_corrections_no_phases_is_identity():
    """All-zero phases leave the statevector unchanged."""
    # Arrange
    psi = np.array([0.5, 0.5j, -0.5, -0.5j])
    expected = psi.copy()

    # Act
    result = apply_phase_corrections(psi, phases=[0.0, 0.0])

    # Assert
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."


def test_apply_phase_corrections_preserves_norm():
    """Phase corrections preserve the L2 norm of the statevector."""
    # Arrange
    psi = np.array([0.5 + 0.1j, 0.3 - 0.2j, 0.4 + 0.3j, 0.1 - 0.5j])
    psi /= np.linalg.norm(psi)  # normalize
    phases = [np.pi / 3, 1.7]

    # Act
    result = apply_phase_corrections(psi, phases)

    # Assert
    assert abs(np.linalg.norm(result) - 1.0) < 1e-12, (
        f"Expected unit norm but got {np.linalg.norm(result)}"
    )


def test_apply_phase_corrections_single_qubit():
    """Single qubit in |1> with phase pi/4 gives exp(i*pi/4)."""
    # Arrange
    phase = np.pi / 4
    psi = np.array([0.0 + 0j, 1.0 + 0j])  # |1>
    expected = np.array([0.0, np.exp(1j * phase)])

    # Act
    result = apply_phase_corrections(psi, phases=[phase])

    # Assert
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."


def test_apply_phase_corrections_multi_qubit_selective():
    """Phase on qubit 0 only; qubit 1 amplitudes are untouched."""
    # Arrange
    psi = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
    phase = np.pi / 2

    # Act
    result = apply_phase_corrections(psi, phases=[phase, 0.0])

    # Assert — only indices 2 (|10>) and 3 (|11>) are rotated
    expected = np.array([0.5, 0.5, 0.5 * np.exp(1j * phase), 0.5 * np.exp(1j * phase)])
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."


def test_apply_phase_corrections_does_not_mutate_input():
    """The original statevector must not be modified."""
    # Arrange
    psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    original = psi.copy()

    # Act
    apply_phase_corrections(psi, phases=[np.pi])

    # Assert
    assert np.array_equal(psi, original), "Input statevector was mutated."


def test_apply_phase_corrections_matches_manual_loop():
    """Cross-check: result must equal a manual apply_phase_to_qubit loop."""
    # Arrange
    rng = np.random.default_rng(42)
    psi = rng.random(8) + 1j * rng.random(8)  # 3 qubits
    psi /= np.linalg.norm(psi)
    phases = [0.5, -1.2, np.pi]

    # Act
    result = apply_phase_corrections(psi, phases)

    # Manual loop (the original inline code)
    expected = psi.copy()
    dim = len(expected)
    n = 3
    for qubit in range(n):
        if phases[qubit] != 0:
            expected = apply_phase_to_qubit(expected, qubit, dim, n, phase=phases[qubit])

    # Assert
    assert np.allclose(result, expected), f"Expected {expected} but got {result}."


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


def test_born_prob_zero_state():
    # |0> state: probability of measuring 0 should be 1.0
    psi = np.array([1.0, 0.0], dtype=complex)
    assert np.isclose(compute_born_probability(psi, target_qubit=0, n=1), 1.0)

def test_born_prob_one_state():
    # |1> state: probability of measuring 0 should be 0.0
    psi = np.array([0.0, 1.0], dtype=complex)
    assert np.isclose(compute_born_probability(psi, target_qubit=0, n=1), 0.0)


def test_born_probability_superposition():
    # |+> state: p0 should be 0.5
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert np.isclose(compute_born_probability(psi, 0, 1), 0.5)

def test_born_prob_sums_to_one():
    # p0 + p1 should always equal 1 for a normalised statevector
    rng = np.random.default_rng(0)
    psi = rng.random(2**3) + 1j * rng.random(2**3)
    psi /= np.linalg.norm(psi)
    p0 = compute_born_probability(psi, target_qubit=1, n=3)
    assert np.isclose(p0 + (1.0 - p0), 1.0)

def test_old_vs_new_born_probability():
    # compare old loop implementation against new vectorised one
    psi = np.random.rand(2**4) + 1j * np.random.rand(2**4)
    psi /= np.linalg.norm(psi)
    n = 4
    target = 2

    # old
    p0_old = 0.0
    for idx, amp in enumerate(psi):
        if ((idx >> (n - 1 - target)) & 1) == 0:
            p0_old += amp.real**2 + amp.imag**2

    # new
    p0_new = compute_born_probability(psi, target, n)
    assert np.isclose(p0_old, p0_new)

def test_collapse_zeros_out_correct_amplitudes():
    # |+> collapsed to 0: |1> amplitude should be zeroed
    psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    result = collapse_statevector(psi.copy(), target_qubit=0, outcome=0, n=1)
    assert result[1] == 0.0
    assert result[0] != 0.0

def test_collapse_to_one_zeros_correct_amplitudes():
    # |+> collapsed to 1: |0> amplitude should be zeroed
    psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    result = collapse_statevector(psi.copy(), target_qubit=0, outcome=1, n=1)
    assert result[0] == 0.0
    assert result[1] != 0.0

def test_collapse_matches_old_loop():
    # regression test: new vectorised collapse matches old Python loop
    rng = np.random.default_rng(7)
    psi = rng.random(2**4) + 1j * rng.random(2**4)
    psi /= np.linalg.norm(psi)
    n, target, outcome = 4, 2, 0

    psi_old = psi.copy()
    mask_pos = n - 1 - target
    for idx in range(len(psi_old)):
        if ((idx >> mask_pos) & 1) != outcome:
            psi_old[idx] = 0.0

    psi_new = collapse_statevector(psi.copy(), target, outcome, n)
    np.testing.assert_array_almost_equal(psi_new, psi_old)

def test_collapse_preserves_correct_amplitudes():
    # amplitudes consistent with outcome should be untouched
    rng = np.random.default_rng(3)
    psi = rng.random(2**3) + 1j * rng.random(2**3)
    psi /= np.linalg.norm(psi)
    psi_original = psi.copy()

    result = collapse_statevector(psi.copy(), target_qubit=1, outcome=0, n=3)
    n, target = 3, 1
    for idx in range(len(result)):
        if ((idx >> (n - 1 - target)) & 1) == 0:
            assert result[idx] == psi_original[idx]