import numpy as np
import pytest

from src.quantum_gates.gates import (
    SpecificNoiseGates,
    Gates,
    noise_free_gates,
)

# -----------------------------
# Fixtures / shared args
# -----------------------------

x_args = dict(
    phi=np.pi / 2,
    p=0.01,
    T1=50e3,
    T2=30e3,
)

twoq_args = dict(
    phi_ctr=np.pi/2,
    phi_trg=np.pi/2,
    t_cnot=300e-9,
    p_cnot=0.01,
    p_single_ctr=0.01,
    p_single_trg=0.01,
    T1_ctr=50e3,
    T2_ctr=30e3,
    T1_trg=50e3,
    T2_trg=30e3,
)

twoq_args_ecr = dict(
    phi_ctr=np.pi/2,
    phi_trg=np.pi/2,
    t_ecr=300e-9,
    p_ecr=0.01,
    p_single_ctr=0.01,
    p_single_trg=0.01,
    T1_ctr=50e3,
    T2_ctr=30e3,
    T1_trg=50e3,
    T2_trg=30e3,
)

def _almost_equal(m1, m2, tol=1e-9):
    return np.allclose(m1, m2, atol=tol)


# -----------------------------
# Validation tests
# -----------------------------

def test_specific_noise_invalid_p():
    with pytest.raises(ValueError):
        SpecificNoiseGates(p_val=-0.1)

    with pytest.raises(ValueError):
        SpecificNoiseGates(p_val=1.5)


def test_specific_noise_invalid_T():
    with pytest.raises(ValueError):
        SpecificNoiseGates(T1_val=-1.0)

    with pytest.raises(ValueError):
        SpecificNoiseGates(T2_val=0.0)


def test_specific_noise_none_allowed():
    # Should not raise
    g = SpecificNoiseGates(p_val=None, T1_val=None, T2_val=None)
    assert g is not None


# -----------------------------
# Override behavior
# -----------------------------

def test_specific_noise_independent_of_inputs():
    g = SpecificNoiseGates(p_val=0.02, T1_val=10e3, T2_val=5e3)
    np.random.seed(0)
    res1 = g.X(phi=np.pi/2, p=0.001, T1=1e3, T2=1e3)
    np.random.seed(0)
    res2 = g.X(phi=np.pi/2, p=0.9, T1=1e6, T2=1e6)

    assert _almost_equal(res1, res2)


def test_specific_noise_partial_override_p_only():
    """Override only p → T dependence should remain."""
    g = SpecificNoiseGates(p_val=0.02, T1_val=None, T2_val=None)

    np.random.seed(10)
    res1 = g.X(phi=np.pi/2, p=0.5, T1=50e3, T2=30e3)
    np.random.seed(10)
    res2 = g.X(phi=np.pi/2, p=1e-6, T1=50e3, T2=30e3)

    # same because p overridden
    assert _almost_equal(res1, res2)


def test_specific_noise_partial_override_T_only():
    """Override only T → p dependence should remain."""
    g = SpecificNoiseGates(p_val=None, T1_val=10e3, T2_val=5e3)

    np.random.seed(2)
    res1 = g.X(phi=np.pi/2, p=0.01, T1=1e6, T2=1e6)
    np.random.seed(2)
    res2 = g.X(phi=np.pi/2, p=0.02, T1=1e6, T2=1e6)

    # different because p not overridden
    assert not _almost_equal(res1, res2)


# -----------------------------
# Consistency tests
# -----------------------------

def test_specific_noise_matches_gates_when_none():
    """If no overrides → should match base Gates exactly."""
    g_specific = SpecificNoiseGates()
    g_base = Gates()

    np.random.seed(1)
    res1 = g_specific.X(**x_args)
    np.random.seed(1)
    res2 = g_base.X(**x_args)

    assert _almost_equal(res1, res2)


def test_specific_noise_noiseless_limit():
    """p=0 and large T → should approach noiseless gate."""
    g = SpecificNoiseGates(p_val=0.0, T1_val=1e12, T2_val=1e12)

    res = g.X(**x_args)
    ref = noise_free_gates.X(**x_args)

    assert _almost_equal(res, ref)


# -----------------------------
# Two-qubit consistency
# -----------------------------

def test_specific_noise_two_qubit_override():
    """Override applies consistently to both qubits."""
    g = SpecificNoiseGates(p_val=0.02, T1_val=10e3, T2_val=5e3)
    np.random.seed(5)
    res1 = g.CNOT(**twoq_args)
    np.random.seed(5)
    res2 = g.CNOT(**{**twoq_args, "p_cnot": 0.5, "T1_ctr": 1e6, "T2_ctr": 1e6})

    assert np.allclose(res1, res2)


def test_specific_noise_ECR_override():
    """Override applies consistently to both qubits."""
    g = SpecificNoiseGates(p_val=0.02, T1_val=10e3, T2_val=5e3)
    np.random.seed(5)
    res1 = g.ECR(**twoq_args_ecr)
    np.random.seed(5)
    res2 = g.ECR(**{**twoq_args_ecr, "p_ecr": 0.5, "T1_ctr": 1e6, "T2_ctr": 1e6})

    assert np.allclose(res1, res2)