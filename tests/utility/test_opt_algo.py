import pytest
import random
import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from src.quantum_gates.utilities import create_random_quantum_circuit, transpile_qiskit_circuit, setup_backend,fix_counts
from src.quantum_gates.utilities import Optimizer
from src.quantum_gates.utilities import DeviceParameters
from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import BinaryCircuit
from src.quantum_gates.gates import standard_gates
from src.quantum_gates.backends import BinaryBackend
from src.quantum_gates._simulation.simulator import _apply_gates_on_circuit


location = "tests/helpers/device_parameters/ibm_kyoto/"

backend = setup_backend(device_name=FakeBrisbane(), use_fake=True)


def level_optimization(level: int, result: list, q: list, qc: list, n: int, psi0: np.array, sim: MrAndersonSimulator, qubits_layout: list[int]):
    bb = BinaryBackend(n)
    opt = Optimizer(level_opt=level, circ_list=result, qubit_list=q)
    result = opt.optimize()

    for item in result:
        q_list = list(range(bb.nqubit))     # list of indexes of all qubit in the circuit
        q_used = item[1]                    # list of indexes of the qubit used in this moment

        if len(item[1]) == 1:               # check if the current is a 1 qubit
            q0 = item[1][0]
            q_list.remove(q0)               # remove index of the qubit from the list
        elif len(item[1]) == 2:             # check if the current is a 2 qubit gates
            q1 = item[1][0]
            q2 = item[1][1]
            q_list.remove(q1)               # remove index of the qubit from the list
            q_list.remove(q2)

        q_n_used = q_list

        k = len(q_n_used)

        if k == 0:                          # in case of 1 or 2 qubits circuit
            U = bb.create_dense(item=item, q_used=q_used, q_n_used=q_n_used)
            psi0 = U @ psi0

        elif k > 0:
            U = bb.create_sparse(item=item, q_n_used=q_n_used, q_used=q_used, N=bb.nqubit)
            psi0 = U.dot(psi0)

    probs = np.square(np.absolute(psi0))
    sums = sim._measurement(prob=probs, q_meas_list=qc, n_qubit=n, qubits_layout=qubits_layout)
    goal = dict(fix_counts(sums, len(qc)))
    goal_list = np.array([value for key, value in goal.items()])

    return goal_list


@pytest.mark.parametrize(
    "nqubits,depth,seed",
    [(n, d, seed) for n in [2, 3, 4, 5, 6, 7, 8] for d in [1, 2, 3, 4] for seed in [1,2,3,4]]
)
def test_optimization_algorithm(nqubits: int, depth: int, seed: int):
    qubit_layout = list(np.arange(nqubits))

    circ = create_random_quantum_circuit(n_qubit=nqubits, depth=depth, seed_circ=seed, measured_qubit=2)
    t_circ = transpile_qiskit_circuit(circ=circ, init_layout=qubit_layout,seed= 10, backend=backend)

    device_param = DeviceParameters(qubits_layout=qubit_layout)
    device_param.load_from_texts(location=location)
    device_param = device_param.__dict__()

    sim = MrAndersonSimulator(gates = standard_gates, CircuitClass=BinaryCircuit)
    q, qc, n = sim._process_layout(circ=t_circ)

    perform_test(
        sim=sim,
        t_circ=t_circ,
        qc=qc,
        qubits_layout=q,
        n=n,
        device_param=device_param,
    )


@pytest.mark.parametrize(
    "nqubits,depth,seed",
    [(n, d, seed) for n in [3, 4] for d in [1, 2, 3] for seed in [1,2,3]]
)
def test_optimization_algorithm_random_layout(nqubits: int, depth: int, seed: int):
    """Preparation"""

    random.seed(nqubits+depth+seed)
    qubit_layout = list(range(nqubits))
    print("qubit layout: ", qubit_layout)

    circ = create_random_quantum_circuit(n_qubit=nqubits, depth=depth, seed_circ=seed, measured_qubit=2)
    t_circ = transpile_qiskit_circuit(circ=circ, init_layout=qubit_layout,seed= 10, backend=backend)

    device_param = DeviceParameters(list(np.arange(max(qubit_layout)+1)))
    device_param.load_from_texts(location=location)
    device_param = device_param.__dict__()

    sim = MrAndersonSimulator(gates = standard_gates, CircuitClass= BinaryCircuit)
    q, qc, n = sim._process_layout(circ=t_circ)

    perform_test(
        sim=sim,
        t_circ=t_circ,
        qc=qc,
        qubits_layout=q,
        n=n,
        device_param=device_param,
    )


def perform_test(
        sim: MrAndersonSimulator,
        t_circ: QuantumCircuit,
        qc: list,
        qubits_layout: list, n: int,
        device_param: dict):

    print(f"qubits_layout: {qubits_layout}, qc: {qc}, n: {n}")

    n_rz, data, _data_measure = sim._preprocess_circuit(
        t_qiskit_circ=t_circ,
        n_qubit_used=n,
        used_logicals=qubits_layout,
    )

    circuit = BinaryCircuit(
        nqubit = n,
        depth=1, # BinaryCircuit ignores this value
        gates=standard_gates,
    )
    for chunk, flag in data:
        if flag == 0:
            _apply_gates_on_circuit(chunk, circuit, device_param, qubit_layout=qubits_layout)

    # Todo: Check if we should make this member public.
    result = circuit._info_gates_list

    psi0 = [1] + [0] * (2**n-1) # starting state
    psi0 = np.array(psi0)

    """Result 0 """

    goal0 = level_optimization(level=0, result=result, q=qubits_layout, qc=qc, n=n, psi0=psi0, sim=sim, qubits_layout=qubits_layout)
    print("goal 0: ", goal0)
    print("")

    """Result 1 """

    goal1 = level_optimization(level=1, result=result, q=qubits_layout, qc=qc, n=n, psi0=psi0, sim=sim, qubits_layout=qubits_layout)
    print("goal 1: ",goal1)
    check_01 = np.abs(goal0-goal1)
    print("check 01: ", check_01)
    print("")

    """Result 2 """

    goal2 = level_optimization(level=2, result=result, q=qubits_layout, qc=qc, n=n, psi0=psi0, sim=sim, qubits_layout=qubits_layout)
    print("goal 2: ", goal2)
    check_02 = np.abs(goal0-goal2)
    print("check 02: ", check_02)
    print("")

    """Result 3 """

    goal3 = level_optimization(level=3, result=result, q=qubits_layout, qc=qc, n=n, psi0=psi0, sim=sim, qubits_layout=qubits_layout)
    print("goal 3: ", goal3)
    check_03 = np.abs(goal0-goal3)
    print("check 03: ", check_03)
    print("")

    """Result 4 """

    goal4 = level_optimization(level= 4, result= result, q=qubits_layout, qc=qc, n=n, psi0=psi0, sim=sim, qubits_layout=qubits_layout)
    print("goal 4: ", goal4)
    check_04 = np.abs(goal0-goal4)
    print("check 04: ", check_04)
    print("")

    assert all((i < 10e-12 for i in check_01)), f"Results from level 0 and 1 mismatch"
    assert all((i < 10e-12 for i in check_02)), f"Results from level 0 and 2 mismatch"
    assert all((i < 10e-12 for i in check_03)), f"Results from level 0 and 3 mismatch"
    assert all((i < 10e-12 for i in check_04)), f"Results from level 0 and 4 mismatch"


def test_opt_level_1_merging_two_1q_gates_does_not_oob():
    U = np.eye(2, dtype=complex)
    gate_list = [[U, [0]], [U, [0]]]  # two successive 1q gates on same qubit
    opt = Optimizer(level_opt=1, qubit_list=[0], circ_list=gate_list)

    out = opt.opt_level_1(gate_list.copy())

    # After optimization, they should merge into a single gate on qubit 0.
    assert len(out) == 1
    assert out[0][1] == [0]
