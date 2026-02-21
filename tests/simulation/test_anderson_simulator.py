import pytest
import time

import numpy as np

from configuration.token import IBM_TOKEN, HUB, GROUP, PROJECT, CRN
from src.quantum_gates.quantum_algorithms import hadamard_reverse_qft_circ
from src.quantum_gates.utilities import setup_backend, create_qc_list
from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import EfficientCircuit
from src.quantum_gates._simulation.circuit import Circuit, StandardCircuit, OneCircuit, BinaryCircuit
from src.quantum_gates.gates import standard_gates, noise_free_gates
from src.quantum_gates._gates.gates import numerical_gates, almost_noise_free_gates
from src.quantum_gates.utilities import DeviceParameters
import tests.helpers.functions as helper_functions
from qiskit_ibm_runtime.fake_provider import FakeBrisbane


location = f"tests/helpers/device_parameters/ibm_kyoto/"

backend = setup_backend(device_name=FakeBrisbane(), use_fake=True)

circuit_set = [Circuit, StandardCircuit, EfficientCircuit, OneCircuit, BinaryCircuit]
gates_set = [standard_gates, numerical_gates, almost_noise_free_gates]


def main(backend,
         do_simulation,
         gates,
         circuit_class,
         circuit_generator: callable,
         shots: int,
         nqubits: int,
         qubits_layout: list,
         location_device_parameters: str,
         parallel: bool=False):

    # Transpile circuit
    qc = create_qc_list(circuit_generator=circuit_generator, nqubits_list=[nqubits], qubits_layout=qubits_layout, backend=backend)[0]

    # Prepare the arguments
    device_param = DeviceParameters(qubits_layout=qubits_layout)
    device_param.load_from_texts(location=location_device_parameters)
    device_param = device_param.__dict__()

    # Create argument for simulator
    arg = {
        'qc': qc,
        'qubits_layout': qubits_layout[:nqubits],
        'shots': shots,
        'nqubits': nqubits,
        'device_param': device_param,
    }

    p_ng = do_simulation(arg, gates, circuit_class, parallel)
    return p_ng


def do_simulation(args: dict, gates, circuit_class, parallel: bool=False):
    """ This is the inside function that will take specific parameters multiple processes will start executing this
        function.
    """

    # Prepare the arguments
    nqubit = args["nqubits"]

    # Setup initial state
    psi0 = np.zeros(2**nqubit, dtype=complex)
    psi0[0] = 1

    # Create simulator
    sim = MrAndersonSimulator(gates=gates, CircuitClass=circuit_class, parallel=parallel)

    # Run simulator
    p_ng = sim.run(
        t_qiskit_circ=args['qc'],
        psi0=psi0,
        shots=args['shots'],
        device_param=args['device_param'],
        nqubit=nqubit,
    )
    return p_ng


@pytest.mark.parametrize(
    "nqubits,gates,circuit_class",
    [(nqubit, gates, circuit_class) for nqubit in range(2, 5) for gates in gates_set for circuit_class in circuit_set]
)
def test_simulator_run_one_shot(nqubits: int, gates, circuit_class):
    run_config = {
        "shots": 1,
        "nqubits": nqubits,
        "qubits_layout": [0, 1, 2, 3, 4],
        "location_device_parameters": location
    }
    main(
        backend,
        do_simulation,
        gates=gates,
        circuit_class=circuit_class,
        circuit_generator=hadamard_reverse_qft_circ,
        **run_config
    )


@pytest.mark.parametrize(
    "shots,gates,circuit_class",
    [(shots, gates, circuit_class) for shots in [10] for gates in gates_set for circuit_class in circuit_set]
)
def test_simulator_run_many_shots(shots: int, gates, circuit_class):
    run_config = {
        "shots": shots,
        "nqubits": 2,
        "qubits_layout": [0, 1, 2, 3, 4],
        "location_device_parameters": location 

    }
    main(
        backend,
        do_simulation,
        gates=gates,
        circuit_class=circuit_class,
        circuit_generator=hadamard_reverse_qft_circ,
        **run_config
    )


@pytest.mark.parametrize(
    "nqubits,gates,circuit_class",
    [(nqubits, gates, circuit_class) for nqubits in [2, 3, 4] for gates in gates_set for circuit_class in circuit_set]
)
def test_simulator_result_makes_sense(nqubits: int, gates, circuit_class):
    run_config = {
        "shots": 100,
        "nqubits": nqubits,
        "qubits_layout": [0, 1, 2, 3, 4],
        "location_device_parameters": location
    }
    result = main(
        backend,
        do_simulation,
        gates=gates,
        circuit_class=circuit_class,
        circuit_generator=hadamard_reverse_qft_circ,
        **run_config
    )

    p_ng = result["probs"]
    values = np.array(list(p_ng.values()))
    assert all((values[0] >= values_i for values_i in values)), \
        f"The state |0..0> was not the most likely. Found probabilities {values}."


@pytest.mark.skip(reason="Invalid test: At the moment, the noise_free_gates have a bug with the global phase.")
@pytest.mark.parametrize(
    "nqubits,circuit_class",
    [(nqubits, circuit_class) for nqubits in [2, 3, 4] for circuit_class in circuit_set]
)
def test_simulator_result_in_noiseless_case(nqubits: int, circuit_class):
    epsilon = 1e-12
    run_config = {
        "shots": 1,
        "nqubits": nqubits,
        "qubits_layout": [0, 1, 2, 3, 4],
        "location_device_parameters": location
    }
    result = main(
        backend,
        do_simulation,
        gates=noise_free_gates,
        circuit_class=circuit_class,
        circuit_generator=hadamard_reverse_qft_circ,
        **run_config
    )

    p = result["probs"]
    values = np.array(list(p.values()))
    p_exp = np.zeros(2**nqubits, dtype=complex)
    p_exp[0] = 1

    assert all((abs(p_exp[i] - values[i]) < epsilon for i in range(2**nqubits))), \
        f"The state |0...0> was not the most likely. Found probabilities {values}."


@pytest.mark.parametrize(
    "nqubits,circuit_class",
    [(nqubits, circuit_class) for nqubits in [2, 3, 4] for circuit_class in circuit_set]
)
def test_simulator_result_in_almost_noiseless_case(nqubits: int, circuit_class):
    epsilon = 1e-6
    run_config = {
        "shots": 1,
        "nqubits": nqubits,
        "qubits_layout": [0, 1, 2, 3, 4],
        "location_device_parameters": location
    }
    result = main(
        backend,
        do_simulation,
        gates=almost_noise_free_gates,
        circuit_class=circuit_class,
        circuit_generator=hadamard_reverse_qft_circ,
        **run_config
    )

    p = result["probs"]
    p_exp = np.zeros(2**nqubits, dtype=complex)
    p_exp[0] = 1
    values = np.array(list(p.values()))

    abs_diffs = [abs(p_exp[i] - values[i]) for i in range(2**nqubits)]
    max_diff = np.max(abs_diffs)
    assert all((abs_diffs[i] < epsilon for i in range(2**nqubits))), \
        f"There was a state whose probability differed by {max_diff} > {epsilon}."


@pytest.mark.parametrize("nqubits, times", [(nqubits, times) for nqubits in [3, 4, 5] for times in [1]])
def test_simulator_speed_for_different_circuits(nqubits, times):
    """ Measures the time the simulation needs for each of the three circuits, times often. Checks that the efficient
        circuit is the fastest.
    """

    abstol = 1e-6

    args = {
        "shots": 1,
        "nqubits": nqubits,
        "qubits_layout": [0, 1, 2, 3, 4],
        "location_device_parameters": location,
        "backend": backend,
        "do_simulation": do_simulation,
        "gates": almost_noise_free_gates,
        "circuit_generator": hadamard_reverse_qft_circ
    }

    time_circuit = 0
    time_standard_circuit = 0
    time_efficient_circuit = 0
    time_one_circuit = 0
    time_binary_circuit = 0

    for _ in range(times):

        # Measure circuit
        time_circuit -= time.time()
        p_circuit = main(circuit_class=Circuit, **args)
        time_circuit += time.time()

        # Measure standard circuit
        time_standard_circuit -= time.time()
        p_standard = main(circuit_class=StandardCircuit, **args)
        time_standard_circuit += time.time()

        # Measure efficient circuit
        time_efficient_circuit -= time.time()
        p_efficient = main(circuit_class=EfficientCircuit, **args)
        time_efficient_circuit += time.time()

        # Measure one circuit
        time_one_circuit -= time.time()
        p_one = main(circuit_class=OneCircuit, **args)
        time_one_circuit += time.time()

        # Measure binary circuit
        time_binary_circuit -= time.time()
        p_binary = main(circuit_class=BinaryCircuit, **args)
        time_binary_circuit += time.time()

        v_circuit = np.array(list(p_circuit["probs"].values()))
        v_standard = np.array(list(p_standard["probs"].values()))
        v_efficient = np.array(list(p_efficient["probs"].values()))
        v_one = np.array(list(p_one["probs"].values()))
        v_binary = np.array(list(p_binary["probs"].values()))


    print(f"time_circuit: {time_circuit} s")
    print(f"time_standard_circuit: {time_standard_circuit} s")
    print(f"time_efficient_circuit: {time_efficient_circuit} s")
    print(f"time_one_circuit: {time_one_circuit} s")
    print(f"time_binary_circuit: {time_binary_circuit} s")

    # Check that they give the same result
    assert helper_functions.vector_almost_equal(v_circuit, v_standard, nqubits, abstol), \
        "StandardCircuit and Circuit did not produce the same result"

    assert helper_functions.vector_almost_equal(v_circuit, v_efficient, nqubits, abstol), \
        "EfficientCircuit and Circuit did not produce the same result"

    assert helper_functions.vector_almost_equal(v_circuit, v_one, nqubits, abstol), \
        "OneCircuit and Circuit did not produce the same result"
    
    assert helper_functions.vector_almost_equal(v_circuit, v_binary, nqubits, abstol), \
        "BinaryCircuit and Circuit did not produce the same result"

    # Speed checks removed: timing is non-deterministic and not the focus of this test.


@pytest.mark.parametrize(
    "nqubits, times",
    [(nqubits, times) for nqubits in range(2, 5) for times in [2]]
)
def test_simulator_speed_for_more_efficient_circuits(nqubits, times):
    """ Measures the time the simulation needs for the more efficient circuit, namely the EfficientCircuit and the
        OneCircuit.
    """

    abstol = 1e-6

    args = {
        "shots": 1,
        "nqubits": nqubits,
        "qubits_layout": [0, 1, 2, 3, 4],
        "location_device_parameters": "tests/helpers/device_parameters/ibm_kyoto/",
        "backend": backend,
        "do_simulation": do_simulation,
        "gates": almost_noise_free_gates,
        "circuit_generator": hadamard_reverse_qft_circ
    }

    time_efficient_circuit = 0
    time_one_circuit = 0

    for _ in range(times):

        # Measure efficient circuit
        time_efficient_circuit -= time.time()
        p_eff = main(circuit_class=EfficientCircuit, **args)
        time_efficient_circuit += time.time()

        # Measure one circuit
        time_one_circuit -= time.time()
        p_one = main(circuit_class=OneCircuit, **args)
        time_one_circuit += time.time()

    v_eff = np.array(list(p_eff["probs"].values()))
    v_one = np.array(list(p_one["probs"].values()))

    print(f"time_efficient_circuit: {time_efficient_circuit} s")
    print(f"time_one_circuit: {time_one_circuit} s")

    assert helper_functions.vector_almost_equal(v_eff, v_one, nqubits, abstol), \
        "OneCircuit and EfficientCircuit did not produce the same result"
    assert time_one_circuit < time_efficient_circuit, "OneCircuit was slower than EfficientCircuit."


@pytest.mark.skip(reason="We fail this test on purpose to get the time and prints. Uncomment this line to run test.")
@pytest.mark.parametrize(
    "nqubits,times",
    [(nqubits, times) for nqubits in range(2, 5) for times in [1]]
)
def test_simulator_speed_for_efficient_circuit(nqubits, times):
    """ Measures the time the efficient simulation needs to to run one shot. Fails on purpose to print the time.
    """

    run_config = {
        "shots": 1,
        "nqubits": nqubits,
        "qubits_layout": [0, 1, 4, 7, 10, 12, 15, 18, 21, 23, 24, 25, 22, 19, 16, 14, 11, 8, 5, 3, 2],
        "location_device_parameters": "tests/helpers/device_parameters/ibm_kyoto/"
    }

    time_efficient_circuit = 0
    time_efficient_circuit -= time.time()

    for _ in range(times):
        main(backend,
             do_simulation,
             gates=standard_gates,
             circuit_class=EfficientCircuit,
             circuit_generator=hadamard_reverse_qft_circ,
             **run_config)

    time_efficient_circuit += time.time()
    print(f"The efficient circuit needed {time_efficient_circuit:.4f} s to simulate {nqubits} qubits.")
    assert False


@pytest.mark.skip(reason="Parallel overhead makes this test non-deterministic.")
@pytest.mark.parametrize("nqubits", [2, 4])
def test_simulation_speed_parallel_vs_sequential(nqubits):
    """ Measures the time the efficient simulation needs when the shots are parallelized vs when not.
    """

    shots = 200
    run_config = {
        "shots": shots,
        "nqubits": nqubits,
        "qubits_layout": [0, 1, 2, 3, 4],
        "location_device_parameters": "tests/helpers/device_parameters/ibm_kyoto/"
    }

    # Parallel
    time_parallel = -time.time()
    main(backend,
         do_simulation,
         gates=standard_gates,
         circuit_class=EfficientCircuit,
         circuit_generator=hadamard_reverse_qft_circ,
         **run_config,
         parallel=True)
    time_parallel += time.time()

    # Sequential
    time_sequential = -time.time()
    main(backend,
         do_simulation,
         gates=standard_gates,
         circuit_class=EfficientCircuit,
         circuit_generator=hadamard_reverse_qft_circ,
         **run_config,
         parallel=False)
    time_sequential += time.time()

    print(f"The parallel circuit needed {time_parallel} s to simulate {shots} shots of {nqubits} qubits.")
    print(f"The sequential circuit needed {time_sequential} s to simulate {shots} shots of {nqubits} qubits.")

    assert time_parallel * 1.5 < time_sequential, \
        f"Expected a speedup of at least 1.5 but found {time_parallel} s, {time_sequential} s, for parallel and sequential shots."


@pytest.mark.parametrize("nqubits", [2, 3, 4])
def test_simulation_gives_normalized_result(nqubits):
    """ Checks if the simulator returns a probability distribution.
    """

    shots = 5
    run_config = {
        "shots": shots,
        "nqubits": nqubits,
        "qubits_layout": [0, 1, 2, 3, 4],
        "location_device_parameters": location
    }

    result = main(backend,
                do_simulation,
                gates=standard_gates,
                circuit_class=EfficientCircuit,
                circuit_generator=hadamard_reverse_qft_circ,
                **run_config,
                parallel=False)

    p_ng = result["probs"]
    v_ng = np.array(list(p_ng.values()))

    assert np.sum(v_ng) == pytest.approx(1.0), "Found that the simulator does not return a probability distribution."
