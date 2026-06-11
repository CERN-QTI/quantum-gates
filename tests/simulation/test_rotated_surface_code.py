# tests/simulation/test_topology_comparison.py
from src.quantum_gates.utilities import RotatedSurfaceCodeLoom


def _run_and_report(topology, distance, n_cycles, n_shots, noise, p):
    code = RotatedSurfaceCodeLoom(
        distance=distance,
        n_cycles=n_cycles,
        n_shots=n_shots,
        noise=noise,
        p=p,
        topology=topology,      # "linear" or "grid"
    )
    error_rate = code.run_circ("MrAnderson")
    print(f"  [{topology:6s}] noise={noise}, d={distance}, "
          f"cycles={n_cycles}, shots={n_shots}, p={p} → error_rate={error_rate:.4f}")
    return error_rate


# ── noisy ────────────────────────────────────────────────────────────────────

def test_topology_noisy_error_rates_are_valid():
    print("\n=== Noisy: linear vs grid ===")
    linear = _run_and_report("linear", distance=3, n_cycles=5, n_shots=50, noise=True,  p=0.06)
    grid   = _run_and_report("grid",   distance=3, n_cycles=1, n_shots=1, noise=True,  p=0.06)

    assert 0.0 <= linear <= 1.0, f"Linear error rate out of range: {linear}"
    assert 0.0 <= grid   <= 1.0, f"Grid error rate out of range:   {grid}"
    print(f"  Δ(grid - linear) = {grid - linear:+.4f}")


# ── noiseless ────────────────────────────────────────────────────────────────

def test_topology_noiseless_error_rates_are_valid():
    print("\n=== Noiseless: linear vs grid ===")
    linear = _run_and_report("linear", distance=3, n_cycles=5, n_shots=50, noise=False, p=0.0)
    grid   = _run_and_report("grid",   distance=3, n_cycles=5, n_shots=50, noise=False, p=0.0)

    assert 0.0 <= linear <= 1.0, f"Linear error rate out of range: {linear}"
    assert 0.0 <= grid   <= 1.0, f"Grid error rate out of range:   {grid}"
    print(f"  Δ(grid - linear) = {grid - linear:+.4f}")


# ── noiseless should beat noisy for both topologies ──────────────────────────

def test_noiseless_lower_than_noisy_for_both_topologies():
    print("\n=== Sanity: noiseless < noisy for each topology ===")
    for topology in ("linear", "grid"):
        noisy     = _run_and_report(topology, distance=3, n_cycles=5, n_shots=50, noise=True,  p=0.06)
        noiseless = _run_and_report(topology, distance=3, n_cycles=5, n_shots=50, noise=False, p=0.0)
        print(f"  [{topology}] noisy={noisy:.4f}, noiseless={noiseless:.4f}")
        assert noiseless <= noisy, (
            f"[{topology}] expected noiseless ({noiseless:.4f}) "
            f"<= noisy ({noisy:.4f})"
        )