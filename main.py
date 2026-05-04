from quantum_gates._utility.RotatedSurfaceCodeLoom import RotatedSurfaceCodeLoom
from quantum_gates._simulation.circuit import EfficientCircuit
from quantum_gates._simulation.backend import EfficientBackend
import sys
import cProfile, pstats, io
from line_profiler import LineProfiler

sys.path.insert(0, "src")

code = RotatedSurfaceCodeLoom(distance=3, n_cycles=5, n_shots=100, noise=True, p=0.06)

#c Profiler
#pr = cProfile.Profile()
#pr.enable()

lp = LineProfiler()
lp.add_function(EfficientCircuit.mid_measurement)
lp.add_function(EfficientBackend._statevector_high_qubit_regime)
lp.enable()


logical_error_rate = code.run_circ("MrAnderson")

lp.disable()
lp.print_stats()

#c Profiler
#pr.disable()
#s = io.StringIO()
#ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
#ps.print_stats(30)

print("-----------------------------------")
#c Profiler
#print(f"Profiling results: {s.getvalue()}")


print("-----------------------------------")
print(f"Logical error rate: {logical_error_rate:.4f}")