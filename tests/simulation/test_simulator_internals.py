"""Unit tests for internal helper functions of MrAndersonSimulator.
"""

import pytest
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import EfficientCircuit
from src.quantum_gates._simulation.circuit import BinaryCircuit
from src.quantum_gates.gates import noise_free_gates
from src.quantum_gates._gates.gates import almost_noise_free_gates
from src.quantum_gates.utilities import DeviceParameters


_backend = FakeBrisbane()
_location = "tests/helpers/device_parameters/ibm_kyoto/"


@pytest.fixture
def sim():
    """A bare simulator instance used to call internal methods."""
    return MrAndersonSimulator(gates=almost_noise_free_gates, CircuitClass=BinaryCircuit)


def _make_device_params(n_physical):
    """Build a minimal synthetic device-parameter dict for n_physical qubits.

    Each 1-D parameter is set to its physical index (T1[i] = i) so that
    after remapping we can verify which physical entries were picked.
    The 2-D interaction parameters (p_int, t_int) are set to i*100+j so
    that each (i,j) cell has a unique value.
    """
    indices = np.arange(n_physical, dtype=float)
    interaction = np.array(
        [[i * 100 + j for j in range(n_physical)] for i in range(n_physical)],
        dtype=float,
    )
    return {
        "T1": indices.copy(),
        "T2": indices.copy(),
        "p": indices.copy(),
        "rout": indices.copy(),
        "tm": indices.copy(),
        "p_int": interaction.copy(),
        "t_int": interaction.copy(),
    }


def _load_device_param(nqubits):
    dp = DeviceParameters(qubits_layout=list(range(nqubits)))
    dp.load_from_texts(location=_location)
    return dp.__dict__()


def _load_device_param_for_layout(layout):
    n_physical = max(layout) + 1
    dp = DeviceParameters(qubits_layout=list(range(n_physical)))
    dp.load_from_texts(location=_location)
    return dp.__dict__()


def _transpile_simple(circ, layout):
    return transpile(
        circ, backend=_backend, initial_layout=layout,
        seed_transpiler=42, optimization_level=0,
    )


class TestResolveSimulationLayout:
    """Physical → simulation index mapping resolution."""

    def test_contiguous_qubits_inferred_as_identity(self, sim):
        result = sim._resolve_simulation_layout(
            used_logicals=[0, 1, 2], nqubit=3, qubits_layout=None,
        )
        assert result == [0, 1, 2]

    def test_non_contiguous_inferred_when_count_matches(self, sim):
        result = sim._resolve_simulation_layout(
            used_logicals=[3, 7], nqubit=2, qubits_layout=None,
        )
        assert result == [3, 7]

    def test_idle_qubits_get_full_width(self, sim):
        result = sim._resolve_simulation_layout(
            used_logicals=[0, 2], nqubit=4, qubits_layout=None,
        )
        assert result == [0, 1, 2, 3]

    def test_explicit_layout_passed_through(self, sim):
        result = sim._resolve_simulation_layout(
            used_logicals=[5, 9], nqubit=2, qubits_layout=[5, 9],
        )
        assert result == [5, 9]

    def test_explicit_layout_with_idle_qubit(self, sim):
        result = sim._resolve_simulation_layout(
            used_logicals=[0, 9], nqubit=3, qubits_layout=[0, 5, 9],
        )
        assert result == [0, 5, 9]

    def test_error_duplicate_indices(self, sim):
        with pytest.raises(ValueError, match="duplicate"):
            sim._resolve_simulation_layout(
                used_logicals=[3, 3], nqubit=2, qubits_layout=[3, 3],
            )

    def test_error_length_mismatch(self, sim):
        with pytest.raises(ValueError, match="length"):
            sim._resolve_simulation_layout(
                used_logicals=[0, 1], nqubit=2, qubits_layout=[0, 1, 2],
            )

    def test_error_missing_used_qubit(self, sim):
        with pytest.raises(ValueError, match="missing"):
            sim._resolve_simulation_layout(
                used_logicals=[0, 7], nqubit=2, qubits_layout=[0, 5],
            )

    def test_error_ambiguous_inference(self, sim):
        with pytest.raises(ValueError, match="Cannot infer"):
            sim._resolve_simulation_layout(
                used_logicals=[3, 7], nqubit=4, qubits_layout=None,
            )

class TestRemapDeviceParameters:
    """Projecting device params from physical to simulation space."""

    def test_identity_mapping(self, sim):
        """[0,1,2] from a 5-qubit device → first 3 entries."""
        dp = _make_device_params(5)
        result = sim._remap_device_parameters(dp, sim_to_phys=[0, 1, 2])

        np.testing.assert_array_equal(result["T1"], [0, 1, 2])
        assert result["p_int"].shape == (3, 3)
        assert result["p_int"][0, 1] == 1.0  # phys0*100 + phys1

    def test_non_contiguous_mapping(self, sim):
        """[2,7] from 10 qubits: sim 0 ↔ phys 2, sim 1 ↔ phys 7."""
        dp = _make_device_params(10)
        result = sim._remap_device_parameters(dp, sim_to_phys=[2, 7])

        np.testing.assert_array_equal(result["T1"], [2, 7])
        np.testing.assert_array_equal(result["rout"], [2, 7])
        assert result["p_int"].shape == (2, 2)
        assert result["p_int"][0, 0] == 202.0  # (phys2, phys2)
        assert result["p_int"][0, 1] == 207.0  # (phys2, phys7)
        assert result["p_int"][1, 0] == 702.0  # (phys7, phys2)
        assert result["p_int"][1, 1] == 707.0  # (phys7, phys7)

    def test_reversed_ordering(self, sim):
        """[4,1] → simulation reverses the physical order."""
        dp = _make_device_params(5)
        result = sim._remap_device_parameters(dp, sim_to_phys=[4, 1])

        np.testing.assert_array_equal(result["T1"], [4, 1])
        assert result["p_int"][0, 1] == 401.0  # phys4*100 + phys1

    def test_single_qubit(self, sim):
        dp = _make_device_params(10)
        result = sim._remap_device_parameters(dp, sim_to_phys=[5])

        np.testing.assert_array_equal(result["T1"], [5])
        assert result["p_int"].shape == (1, 1)
        assert result["p_int"][0, 0] == 505.0

    def test_original_not_mutated(self, sim):
        dp = _make_device_params(10)
        t1_before = dp["T1"].copy()
        sim._remap_device_parameters(dp, sim_to_phys=[3, 8])
        np.testing.assert_array_equal(dp["T1"], t1_before)

    def test_error_device_params_too_small(self, sim):
        """Device has 5 qubits but layout references physical qubit 7."""
        dp = _make_device_params(5)
        with pytest.raises(ValueError, match="does not cover"):
            sim._remap_device_parameters(dp, sim_to_phys=[0, 7])

    def test_error_empty_layout(self, sim):
        dp = _make_device_params(5)
        with pytest.raises(ValueError, match="at least one"):
            sim._remap_device_parameters(dp, sim_to_phys=[])


class TestMeasurement:

    def test_two_qubits_measure_both(self, sim):
        """prob=[0.1, 0.2, 0.3, 0.4] for |00>,|01>,|10>,|11>."""
        prob = np.array([0.1, 0.2, 0.3, 0.4])
        result = sim._measurement(prob=prob, q_meas_list=[(0, 0), (1, 1)], n_qubit=2)

        assert result["00"] == pytest.approx(0.1)
        assert result["01"] == pytest.approx(0.2)
        assert result["10"] == pytest.approx(0.3)
        assert result["11"] == pytest.approx(0.4)

    def test_two_qubits_measure_first_only(self, sim):
        """Marginalize out qubit 1.
        Qubit 0 is leftmost: '0' in |00>,|01>, '1' in |10>,|11>.
        P('0')=0.1+0.2=0.3, P('1')=0.3+0.4=0.7."""
        prob = np.array([0.1, 0.2, 0.3, 0.4])
        result = sim._measurement(prob=prob, q_meas_list=[(0, 0)], n_qubit=2)

        assert result["0"] == pytest.approx(0.3)
        assert result["1"] == pytest.approx(0.7)

    def test_two_qubits_measure_second_only(self, sim):
        """Marginalize out qubit 0.
        Qubit 1 is rightmost: '0' in |00>,|10>, '1' in |01>,|11>.
        P('0')=0.1+0.3=0.4, P('1')=0.2+0.4=0.6."""
        prob = np.array([0.1, 0.2, 0.3, 0.4])
        result = sim._measurement(prob=prob, q_meas_list=[(1, 1)], n_qubit=2)

        assert result["0"] == pytest.approx(0.4)
        assert result["1"] == pytest.approx(0.6)

    def test_three_qubits_measure_outer_pair(self, sim):
        """3 qubits, measure 0 and 2 (skip middle). Uniform → 0.25 each."""
        prob = np.ones(8) / 8.0
        result = sim._measurement(prob=prob, q_meas_list=[(0, 0), (2, 2)], n_qubit=3)
        for key in ["00", "01", "10", "11"]:
            assert result[key] == pytest.approx(0.25)

    def test_single_qubit(self, sim):
        prob = np.array([0.7, 0.3])
        result = sim._measurement(prob=prob, q_meas_list=[(0, 0)], n_qubit=1)
        assert result["0"] == pytest.approx(0.7)
        assert result["1"] == pytest.approx(0.3)

    def test_deterministic_state(self, sim):
        """All probability in |10>."""
        prob = np.array([0.0, 0.0, 1.0, 0.0])
        result = sim._measurement(prob=prob, q_meas_list=[(0, 0), (1, 1)], n_qubit=2)
        assert result.get("10", 0.0) == pytest.approx(1.0)
        assert result.get("00", 0.0) == pytest.approx(0.0)

    def test_probabilities_sum_to_one(self, sim):
        prob = np.array([0.05, 0.10, 0.15, 0.20, 0.10, 0.15, 0.05, 0.20])
        result = sim._measurement(prob=prob, q_meas_list=[(0, 0), (2, 2)], n_qubit=3)
        assert sum(result.values()) == pytest.approx(1.0)


# ===================================================================
# 4. Input validation in run()
# ===================================================================

class TestRunInputValidation:
    """Verify that run() gives clear errors for common user mistakes."""

    def _simple_circuit(self):
        """X(0), measure both: the simplest non-trivial 2-qubit circuit."""
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure([0, 1], [0, 1])
        return qc

    def _good_run_args(self):
        """A set of valid arguments for sim.run()."""
        layout = [0, 1]
        circ = self._simple_circuit()
        t_circ = _transpile_simple(circ, layout)
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        return dict(
            t_qiskit_circ=t_circ,
            psi0=psi0,
            shots=5,
            device_param=_load_device_param(2),
            nqubit=2,
            bit_flip_bool=False,
        )

    def test_psi0_not_power_of_two_raises(self, sim):
        """psi0 with 3 elements is not a valid statevector."""
        args = self._good_run_args()
        args["psi0"] = np.array([1, 0, 0], dtype=complex)
        with pytest.raises(ValueError, match="power of two"):
            sim.run(**args)

    def test_shots_zero_raises(self, sim):
        args = self._good_run_args()
        args["shots"] = 0
        with pytest.raises(ValueError, match="positive"):
            sim.run(**args)

    def test_shots_negative_raises(self, sim):
        args = self._good_run_args()
        args["shots"] = -1
        with pytest.raises(ValueError, match="positive"):
            sim.run(**args)

    def test_device_param_not_dict_raises(self, sim):
        args = self._good_run_args()
        args["device_param"] = "not a dict"
        with pytest.raises(ValueError, match="dict"):
            sim.run(**args)

    def test_t_qiskit_circ_not_circuit_raises(self, sim):
        """Passing a non-QuantumCircuit should raise immediately, before
        any attribute access."""
        args = self._good_run_args()
        args["t_qiskit_circ"] = "not a circuit"
        with pytest.raises(ValueError, match="QuantumCircuit"):
            sim.run(**args)

    def test_device_param_missing_keys_raises(self, sim):
        """A dict with some required keys missing should give a clear error."""
        args = self._good_run_args()
        args["device_param"] = {"T1": [0.1, 0.2]}  # missing T2, p, etc.
        with pytest.raises(ValueError, match="missing required keys"):
            sim.run(**args)

    def test_device_params_too_small_for_layout_raises(self, sim):
        """Explicit layout [0,9] with device_param that only covers 3 qubits."""
        layout = [0, 9]
        circ = self._simple_circuit()
        t_circ = _transpile_simple(circ, layout)
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        # Synthetic params with only 3 entries — then claim layout [0,9]
        dp = _make_device_params(3)

        with pytest.raises(ValueError, match="does not cover"):
            sim.run(
                t_qiskit_circ=t_circ, psi0=psi0, shots=5,
                device_param=dp, nqubit=2,
                qubits_layout=[0, 9],  # phys qubit 9 not covered by dp
                bit_flip_bool=False,
            )

    def test_nqubit_mismatch_warns(self, sim):
        """If nqubit disagrees with psi0.size, a warning should be emitted."""
        args = self._good_run_args()
        args["nqubit"] = 5  # psi0 has 4 elements → 2 qubits
        with pytest.warns(UserWarning, match="does not match psi0"):
            sim.run(**args)

    def test_qubits_layout_missing_used_qubit_raises(self, sim):
        """Transpiled circuit uses [0,1] but user passes layout=[0,5]."""
        layout = [0, 1]
        circ = self._simple_circuit()
        t_circ = _transpile_simple(circ, layout)
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        dp = _load_device_param(10)

        with pytest.raises(ValueError, match="missing"):
            sim.run(
                t_qiskit_circ=t_circ, psi0=psi0, shots=5,
                device_param=dp, nqubit=2,
                qubits_layout=[0, 5],  # qubit 1 is missing
                bit_flip_bool=False,
            )

    def test_valid_run_returns_expected_keys(self, sim):
        """Smoke test: a valid call returns the documented dict structure."""
        args = self._good_run_args()
        result = sim.run(**args)

        assert isinstance(result, dict)
        for key in ("probs", "results", "num_clbits", "mid_counts", "statevector_readout"):
            assert key in result, f"Missing key '{key}' in run() output"

        probs = result["probs"]
        assert isinstance(probs, dict)
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


# ===================================================================
# 5. Integration: standard circuits without mid-circuit measurement
# ===================================================================

class TestStandardCircuitIntegration:
    """End-to-end tests for circuits *without* mid-circuit measurement,
    verifying that qubits_layout works correctly through the full pipeline."""

    @pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
    def test_x_gate_produces_correct_output(self, circuit_class):
        """X(0) on |00> should produce '10' with near-certainty."""
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure([0, 1], [0, 1])
        t_circ = _transpile_simple(qc, [0, 1])

        sim = MrAndersonSimulator(gates=noise_free_gates, CircuitClass=circuit_class)
        result = sim.run(
            t_qiskit_circ=t_circ,
            psi0=np.array([1, 0, 0, 0], dtype=complex),
            shots=10, device_param=_load_device_param(2), nqubit=2,
            bit_flip_bool=False,
        )
        assert result["probs"].get("10", 0.0) > 0.99

    @pytest.mark.parametrize("circuit_class", [EfficientCircuit, BinaryCircuit])
    def test_no_mid_circuit_events_for_standard_circuit(self, circuit_class):
        """Standard circuit should produce empty mid-circuit results."""
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure([0, 1], [0, 1])
        t_circ = _transpile_simple(qc, [0, 1])

        sim = MrAndersonSimulator(gates=noise_free_gates, CircuitClass=circuit_class)
        result = sim.run(
            t_qiskit_circ=t_circ,
            psi0=np.array([1, 0, 0, 0], dtype=complex),
            shots=5, device_param=_load_device_param(2), nqubit=2,
            bit_flip_bool=False,
        )
        for shot in result["results"]:
            assert len(shot["mid"]) == 0, "Standard circuit should not have mid-circuit events"


class TestMidCircuitConsistency:
    """Verify that mid-circuit measurement outcomes are physically consistent
    with what you'd expect from the statevector just before measurement."""

    def test_mid_measure_on_zero_state_always_gives_zero(self):
        """Mid-measure qubit 0 of |00> → outcome must always be 0."""
        nqubits = 2
        qc = QuantumCircuit(nqubits, nqubits + 1)
        # No gates → state is |00>
        qc.measure(0, nqubits)       # mid-measure to extra clbit
        qc.rz(0.001, 0)              # keep it classified as mid-circuit
        qc.barrier()
        qc.measure([0, 1], [0, 1])   # final measurement

        t_circ = _transpile_simple(qc, [0, 1])
        sim = MrAndersonSimulator(gates=noise_free_gates, CircuitClass=BinaryCircuit)
        result = sim.run(
            t_qiskit_circ=t_circ,
            psi0=np.array([1, 0, 0, 0], dtype=complex),
            shots=50, device_param=_load_device_param(nqubits), nqubit=nqubits,
            bit_flip_bool=False,
        )
        for shot in result["results"]:
            for event in shot["mid"]:
                for o in event["outcome"]:
                    assert int(o) == 0, f"Expected 0 but got {o}"

    def test_mid_measure_on_one_state_always_gives_one(self):
        """X(0) then mid-measure qubit 0 → outcome must always be 1."""
        nqubits = 2
        qc = QuantumCircuit(nqubits, nqubits + 1)
        qc.x(0)
        qc.measure(0, nqubits)
        qc.rz(0.001, 0)
        qc.barrier()
        qc.measure([0, 1], [0, 1])

        t_circ = _transpile_simple(qc, [0, 1])
        sim = MrAndersonSimulator(gates=noise_free_gates, CircuitClass=BinaryCircuit)
        result = sim.run(
            t_qiskit_circ=t_circ,
            psi0=np.array([1, 0, 0, 0], dtype=complex),
            shots=50, device_param=_load_device_param(nqubits), nqubit=nqubits,
            bit_flip_bool=False,
        )
        for shot in result["results"]:
            for event in shot["mid"]:
                for o in event["outcome"]:
                    assert int(o) == 1, f"Expected 1 but got {o}"

    def test_mid_measure_then_continue_preserves_collapsed_state(self):
        """Prepare |+0>, mid-measure qubit 0, then final measure.
        After mid-measure, qubit 0 collapses to the measured value.
        The final measurement of qubit 0 should match the mid-circuit outcome."""
        nqubits = 2
        qc = QuantumCircuit(nqubits, nqubits + 1)
        qc.h(0)                       # |+0> = (|00> + |10>)/√2
        qc.measure(0, nqubits)        # mid-measure qubit 0
        qc.rz(0.001, 0)              # keep mid-circuit
        qc.barrier()
        qc.measure([0, 1], [0, 1])

        t_circ = _transpile_simple(qc, [0, 1])
        sim = MrAndersonSimulator(gates=noise_free_gates, CircuitClass=BinaryCircuit)
        result = sim.run(
            t_qiskit_circ=t_circ,
            psi0=np.array([1, 0, 0, 0], dtype=complex),
            shots=200, device_param=_load_device_param(nqubits), nqubit=nqubits,
            bit_flip_bool=False,
        )

        # After collapsing, qubit 1 stays |0>. So final must be "00" or "10"
        # (never "01" or "11").
        probs = result["probs"]
        p_q1_is_one = probs.get("01", 0.0) + probs.get("11", 0.0)
        assert p_q1_is_one < 0.02, (
            f"Qubit 1 should remain 0 but got P(q1=1) = {p_q1_is_one:.3f}"
        )

    def test_reset_returns_qubit_to_zero(self):
        """X(0) → reset(0) → measure: qubit 0 should be |0> again."""
        nqubits = 2
        qc = QuantumCircuit(nqubits, nqubits)
        qc.x(0)
        qc.reset(0)
        qc.measure([0, 1], [0, 1])

        t_circ = _transpile_simple(qc, [0, 1])
        sim = MrAndersonSimulator(gates=noise_free_gates, CircuitClass=BinaryCircuit)
        result = sim.run(
            t_qiskit_circ=t_circ,
            psi0=np.array([1, 0, 0, 0], dtype=complex),
            shots=50, device_param=_load_device_param(nqubits), nqubit=nqubits,
            bit_flip_bool=False,
        )
        p_zero = result["probs"].get("00", 0.0)
        assert p_zero > 0.95, f"Expected |00> after reset but got P={p_zero:.3f}"


class TestSubsetVsFullSimulation:
    """When a circuit only touches a subset of qubits, verify that
    simulating just the active subset (via qubits_layout) gives
    the same answer as simulating the full register."""

    def test_full_width_vs_compact_simulation(self):
        """Transpile a 2-qubit circuit onto physical qubits [2,4].
        Run 1: full-width psi0 of size 2^5=32 (covers phys 0..4).
        Run 2: compact psi0 of size 2^2=4 with qubits_layout=[2,4].
        Both should give P('10') ≈ 1."""
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure([0, 1], [0, 1])

        layout = [2, 4]
        t_circ = _transpile_simple(qc, layout)
        dp = _load_device_param(5)  # covers qubits 0..4

        sim = MrAndersonSimulator(gates=noise_free_gates, CircuitClass=BinaryCircuit)

        # Full-width: 5-qubit statevector, qubits 0,1,3 idle
        result_full = sim.run(
            t_qiskit_circ=t_circ,
            psi0=np.array([1] + [0] * 31, dtype=complex),
            shots=10, device_param=dp, nqubit=5,
            bit_flip_bool=False,
        )

        # Compact: 2-qubit statevector, only the active qubits
        result_compact = sim.run(
            t_qiskit_circ=t_circ,
            psi0=np.array([1, 0, 0, 0], dtype=complex),
            shots=10, device_param=dp, nqubit=2,
            qubits_layout=layout,
            bit_flip_bool=False,
        )

        assert result_full["probs"].get("10", 0.0) > 0.99, (
            f"Full-width: expected '10' but got {result_full['probs']}"
        )
        assert result_compact["probs"].get("10", 0.0) > 0.99, (
            f"Compact: expected '10' but got {result_compact['probs']}"
        )
