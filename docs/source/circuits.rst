Circuits
========

Quantum circuit can be constructed easily when we have a circuit object
to which we can apply gates by simply calling method like
circuit.H(i=0). This functionality is provided in the LegacyCircuit classes,
and packed together with the knowledge on how to sample the noisy gates
as stochastic matrices from a specific gate set.


.. _circuits_usage:

Usage
-----

The class is easy to use besides one tricky detail. For each gatetime
(timestep), one has to fill in all gates. For example, consider the two
qubit case. We apply an X gate on qubit 0 and then a CNOT gate on both
qubits with control on 0 and target on 1. Then we use the following
code:

.. code:: python

   from quantum_gates.circuits import EfficientCircuit
   from quantum_gates.gates import standard_gates

   # The depth can be set arbitrarily for this circuit class. 
   circuit = EfficientCircuit(nqubit=2, depth=0, gates=standard_gates)

   # First timestep -> Each qubit has to get a gate, this is why we apply even identities.
   circuit.H(i=0, ...)
   circuit.I(i=1) 

   # Second timestep -> All qubits received a gate, because CNOT is a two-qubit gate.
   circuit.CNOT(i=0, k=1, ...)

   # Evaluate the statevector
   psi1 = circuit.statevector(psi0=np.array([1, 0, 0, 0])) 

Not applying gates to each qubit will lead to errors.

``BinaryCircuit`` is a variant suitable for non-linear qubit topologies
where the qubit layout has gaps. All circuit classes also expose
``mid_measurement`` for projective measurement on a subset of qubits,
``reset`` for resetting a qubit, and ``statevector_readout`` for inspecting
the simulated state at a labelled barrier.


Classes and Instances
---------------------

.. automodule:: quantum_gates.circuits
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:


.. _circuits_possible_extensions:

Possible extensions
-------------------

In the future, we could add checks that warns the user in case not all
gates are applied.
