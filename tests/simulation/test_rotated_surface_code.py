# tests/test_rotated_surface_code.py
from quantum_gates._utility.RotatedSurfaceCodeLoom import RotatedSurfaceCodeLoom

def test_rotated_surface_code_runs():
    code = RotatedSurfaceCodeLoom(distance=3, n_cycles=5, n_shots=10, noise=True, p=0.06)
    logical_error_rate = code.run_circ("MrAnderson")
    assert isinstance(logical_error_rate, float)
    assert 0.0 <= logical_error_rate <= 1.0

def test_rotated_surface_code_noiseless():
    code = RotatedSurfaceCodeLoom(distance=3, n_cycles=5, n_shots=10, noise=False, p=0.0)
    logical_error_rate = code.run_circ("MrAnderson")
    assert isinstance(logical_error_rate, float)
    assert 0.0 <= logical_error_rate <= 1.0