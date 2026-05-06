import cProfile
import pstats
import io
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from src.quantum_gates.simulators import MrAndersonSimulator
from src.quantum_gates.circuits import EfficientCircuit
from src.quantum_gates.gates import almost_noise_free_gates
from src.quantum_gates.utilities import DeviceParameters

_backend = FakeBrisbane()
_location = "tests/helpers/device_parameters/ibm_kyoto/"
_LAYOUT = [0, 1, 2, 3, 4]

def _device_param(nqubits):
    dp = DeviceParameters(qubits_layout=_LAYOUT[:nqubits])
    dp.load_from_texts(location=_location)
    return dp.__dict__()

def run_consecutive(shots=100):
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.cx(0, 1)      # consecutive
    qc.measure(range(nqubits), range(nqubits))
    t_circ = transpile(qc, backend=_backend, initial_layout=_LAYOUT[:nqubits],
                       seed_transpiler=42, optimization_level=0)
    psi0 = np.zeros(2**nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=almost_noise_free_gates, CircuitClass=EfficientCircuit)
    return sim.run(t_qiskit_circ=t_circ, psi0=psi0, shots=shots,
                   device_param=_device_param(nqubits), nqubit=nqubits)

def run_non_consecutive(shots=100):
    nqubits = 3
    qc = QuantumCircuit(nqubits, nqubits)
    qc.h(0)
    qc.cx(0, 2)      # non-consecutive
    qc.measure(range(nqubits), range(nqubits))
    t_circ = transpile(qc, backend=_backend, initial_layout=_LAYOUT[:nqubits],
                       seed_transpiler=42, optimization_level=0)
    psi0 = np.zeros(2**nqubits, dtype=complex)
    psi0[0] = 1
    sim = MrAndersonSimulator(gates=almost_noise_free_gates, CircuitClass=EfficientCircuit)
    return sim.run(t_qiskit_circ=t_circ, psi0=psi0, shots=shots,
                   device_param=_device_param(nqubits), nqubit=nqubits)

if __name__ == "__main__":
    shots = 100

    print("=" * 60)
    print(" CONSECUTIVE GATE")
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
    print(" NON-CONSECUTIVE GATE")
    print("=" * 60)
    pr = cProfile.Profile()
    pr.enable()
    run_non_consecutive(shots=shots)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())