Qiskit Provider
===============

The library can be used as a Qiskit provider, so existing Qiskit code can
run noisy simulations without changing the call sites. ``NoisyGatesBackend``
plugs into Qiskit's provider interface and delegates the simulation to
``MrAndersonSimulator`` under the hood. See the
``tutorial_qiskit_NoisyGatesProvider.ipynb`` notebook in ``docs/tutorials/``
for a worked example.

.. automodule:: quantum_gates.qiskit_provider
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
