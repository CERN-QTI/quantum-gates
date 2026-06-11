import numpy as np
import pytest
from qiskit import QuantumCircuit

from src.quantum_gates._utility.simulations_utility import compute_born_probabilities
from src.quantum_gates.circuits import BinaryCircuit, Circuit, EfficientCircuit
from src.quantum_gates.gates import NoiseFreeGates
from src.quantum_gates.simulators import MrAndersonSimulator


def _device_param(nqubits):
    return {
        "T1": np.ones(nqubits),
        "T2": np.ones(nqubits),
        "p": np.zeros(nqubits),
        "rout": np.zeros(nqubits),
        "p_int": np.zeros((nqubits, nqubits)),
        "t_int": np.zeros((nqubits, nqubits)),
        "tm": np.ones(nqubits),
        "dt": np.array([1.0]),
    }


def _make_circuit(circuit_class, nqubits, gates=None):
    return circuit_class(
        nqubit=nqubits,
        depth=1,
        gates=gates if gates is not None else NoiseFreeGates(),
    )


def _statevector_norm(psi):
    return float(np.vdot(psi, psi).real)


@pytest.mark.parametrize("circuit_class", [Circuit, EfficientCircuit, BinaryCircuit])
def test_mid_measurement_preserves_incoming_trace_not_unit_trace(circuit_class):
    """Unnormalized trajectories should not be forced back to norm one."""
    np.random.seed(1234)
    psi0 = np.zeros(4, dtype=complex)
    psi0[0] = 0.5
    psi0[3] = 0.5

    circ = _make_circuit(circuit_class, nqubits=2)
    psi, outcome = circ.mid_measurement(
        psi0=psi0,
        device_param=_device_param(2),
        add_bitflip=False,
        qubit_list=[1],
        cbit_list=None,
    )

    assert outcome in ([0], [1])
    assert _statevector_norm(psi0) == pytest.approx(0.5)
    assert _statevector_norm(psi) == pytest.approx(0.5)
    assert _statevector_norm(psi) != pytest.approx(1.0)

    measured_bit = outcome[0]
    allowed_indices = {0} if measured_bit == 0 else {3}
    nonzero_indices = set(np.flatnonzero(np.abs(psi) > 1e-12))
    assert nonzero_indices == allowed_indices
    assert np.max(np.abs(psi)) == pytest.approx(np.sqrt(0.5))


@pytest.mark.parametrize("circuit_class", [Circuit, EfficientCircuit, BinaryCircuit])
def test_sequential_mid_measurements_keep_unnormalized_trace(circuit_class):
    """Array-style sequential collapse should preserve the incoming trajectory weight."""
    np.random.seed(5678)
    psi0 = np.full(4, 0.25, dtype=complex)

    circ = _make_circuit(circuit_class, nqubits=2)
    psi, outcome = circ.mid_measurement(
        psi0=psi0,
        device_param=_device_param(2),
        add_bitflip=False,
        qubit_list=[0, 1],
        cbit_list=[2, 3],
    )

    assert len(outcome) == 2
    assert _statevector_norm(psi0) == pytest.approx(0.25)
    assert _statevector_norm(psi) == pytest.approx(0.25)
    assert np.count_nonzero(np.abs(psi) > 1e-12) == 1


@pytest.mark.parametrize("circuit_class", [Circuit, EfficientCircuit, BinaryCircuit])
def test_repeated_mid_measurements_keep_accumulated_trace_reduction(circuit_class):
    """Repeated mid-measures must not erase trace loss accumulated between them."""
    psi = np.array([1.0, 0.0], dtype=complex)
    attenuation = np.sqrt(0.8) * np.eye(2)
    traces = []

    for _ in range(4):
        circ = _make_circuit(circuit_class, nqubits=1)
        circ.apply(attenuation, i=0)
        psi = circ.statevector(psi)
        trace_before_measure = _statevector_norm(psi)
        psi, _ = circ.mid_measurement(
            psi0=psi,
            device_param=_device_param(1),
            add_bitflip=False,
            qubit_list=[0],
            cbit_list=None,
        )
        trace_after_measure = _statevector_norm(psi)
        traces.append(trace_after_measure)

        assert trace_after_measure == pytest.approx(trace_before_measure)

    assert traces == pytest.approx([0.8, 0.64, 0.512, 0.4096])
    assert all(after < before for before, after in zip(traces, traces[1:]))


def test_born_probabilities_are_raw_for_unnormalized_states():
    psi = np.array([np.sqrt(0.20), np.sqrt(0.05)], dtype=complex)

    p0, p1 = compute_born_probabilities(psi, target_qubit=0, n=1)

    assert p0 == pytest.approx(0.20)
    assert p1 == pytest.approx(0.05)
    assert p0 + p1 == pytest.approx(_statevector_norm(psi))
    assert p0 / (p0 + p1) == pytest.approx(0.80)
    assert p1 / (p0 + p1) == pytest.approx(0.20)


def test_simulator_mid_measurement_does_not_renormalize_saved_statevector():
    psi0 = np.zeros(4, dtype=complex)
    psi0[0] = 0.5
    psi0[3] = 0.5
    qc = QuantumCircuit(2, 3)
    qc.measure(1, 2)
    qc.barrier(label="save_sv_0")
    qc.rz(0.001, 1)
    qc.measure([0, 1], [0, 1])

    sim = MrAndersonSimulator(gates=NoiseFreeGates(), CircuitClass=BinaryCircuit)
    result = sim.run(
        t_qiskit_circ=qc,
        psi0=psi0,
        shots=20,
        device_param=_device_param(2),
        nqubit=2,
        bit_flip_bool=False,
    )

    for shot_readout in result["statevector_readout"]:
        saved = dict(shot_readout)["save_sv_0"]
        assert _statevector_norm(saved) == pytest.approx(0.5)
