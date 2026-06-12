import cProfile
import pstats
import io
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from src.quantum_gates.gates import almost_noise_free_gates
from src.quantum_gates.utilities import DeviceParameters

from qiskit_ibm_runtime.fake_provider import FakeBrisbane
_backend = FakeBrisbane()
coupling_map = _backend.coupling_map
_LAYOUT = list(range(16))

def _device_param(nqubits):
    dp = DeviceParameters(qubits_layout=_LAYOUT[:nqubits])
    dp.load_from_backend(_backend)  # load from FakeBrisbane directly instead of text files
    return dp.__dict__()

def run_consecutive(shots=100):
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(4)
    qc.cx(4, 5)      # consecutive
    qc.measure(range(nqubits), range(nqubits))
    t_circ = transpile(qc, backend=_backend, initial_layout=_LAYOUT[:nqubits],
                       seed_transpiler=42, optimization_level=0)
    psi0 = np.zeros(2**nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=almost_noise_free_gates, CircuitClass=EfficientCircuit)
    return sim.run(t_qiskit_circ=t_circ, psi0=psi0, shots=shots,
                   device_param=_device_param(nqubits), nqubit=nqubits)

def run_non_consecutive_with_swaps(shots=100):
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(4)
    qc.cx(4, 14)      # not directly connected — transpiler inserts SWAPs
    qc.measure(range(nqubits), range(nqubits))

    t_circ = transpile(qc, backend=_backend,
                       initial_layout=_LAYOUT[:nqubits],
                       seed_transpiler=42,
                       optimization_level=0)
    print("SWAP route ops:", t_circ.count_ops())

    psi0 = np.zeros(2**nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=almost_noise_free_gates, CircuitClass=EfficientCircuit)
    return sim.run(t_qiskit_circ=t_circ, psi0=psi0, shots=shots,
                   device_param=_device_param(nqubits), nqubit=nqubits)

def run_non_consecutive(shots=100):
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(4)
    qc.cx(4, 15)      # non-consecutive
    qc.measure(range(nqubits), range(nqubits))
    t_circ = transpile(qc, backend=_backend, initial_layout=_LAYOUT[:nqubits],
                       seed_transpiler=42, optimization_level=0)
    psi0 = np.zeros(2**nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=almost_noise_free_gates, CircuitClass=EfficientCircuit)
    return sim.run(t_qiskit_circ=t_circ, psi0=psi0, shots=shots,
                   device_param=_device_param(nqubits), nqubit=nqubits)



def run_consecutive_binary(shots=100):
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(4)
    qc.cx(4, 5)
    qc.measure(range(nqubits), range(nqubits))
    t_circ = transpile(qc, backend=_backend, initial_layout=_LAYOUT[:nqubits],
                       seed_transpiler=42, optimization_level=0)
    psi0 = np.zeros(2**nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=almost_noise_free_gates, CircuitClass=BinaryCircuit)
    return sim.run(t_qiskit_circ=t_circ, psi0=psi0, shots=shots,
                   device_param=_device_param(nqubits), nqubit=nqubits)

def run_non_consecutive_binary(shots=100):
    nqubits = 16
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(4)
    qc.cx(4, 15)
    qc.measure(range(nqubits), range(nqubits))
    t_circ = transpile(qc, backend=_backend, initial_layout=_LAYOUT[:nqubits],
                       seed_transpiler=42, optimization_level=0)
    psi0 = np.zeros(2**nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=almost_noise_free_gates, CircuitClass=BinaryCircuit)
    return sim.run(t_qiskit_circ=t_circ, psi0=psi0, shots=shots,
                   device_param=_device_param(nqubits), nqubit=nqubits)


if __name__ == "__main__":
    shots = 100

    print("=" * 60)
    print(" CONSECUTIVE GATE - EFFICIENT CIRCUIT")
    print("=" * 60)
    pr = cProfile.Profile()
    pr.enable()
    run_consecutive(shots=shots)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    print("=" * 60)
    print(" NON-CONSECUTIVE GATE WITH SWAPPING - EFFICIENT CIRCUIT")
    print("=" * 60)
    pr = cProfile.Profile()
    pr.enable()
    run_non_consecutive_with_swaps(shots=shots)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    print("=" * 60)
    print(" NON-CONSECUTIVE GATE - EFFICIENT CIRCUIT")
    print("=" * 60)
    pr = cProfile.Profile()
    pr.enable()
    run_non_consecutive(shots=shots)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    print("=" * 60)
    print(" CONSECUTIVE GATE — BinaryCircuit")
    print("=" * 60)
    pr = cProfile.Profile()
    pr.enable()
    run_consecutive_binary(shots=shots)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    print("=" * 60)
    print(" NON-CONSECUTIVE GATE — BinaryCircuit")
    print("=" * 60)
    pr = cProfile.Profile()
    pr.enable()
    run_non_consecutive_binary(shots=shots)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())