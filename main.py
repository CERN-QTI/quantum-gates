from quantum_gates._utility.RotatedSurfaceCodeLoom import RotatedSurfaceCodeLoom
import sys
sys.path.insert(0, "src")

code = RotatedSurfaceCodeLoom(distance=5, n_cycles=5, n_shots=10000, noise=True, p=0.06)
logical_error_rate = code.run_circ("MrAnderson")
print(f"Logical error rate: {logical_error_rate:.4f}")