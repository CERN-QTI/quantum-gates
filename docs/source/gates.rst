Gates
=====

Here you learn how to use the different gate sets to sample noisy quantum
gates. The gate set combines multiple gate factories into a single
object, which is easy to import and pass in arguments.

Usage
-----

While the usage is fairly trivial, there is one dangerous subtlety that
the user should be aware of. As computer programs are deterministic, the
sampling from probability distributions often works by using a
pseudo-random number generator underneath. Thus, the user should not
accidentaly sample the same gates multiple times, which can happen when
the same simulation is run in parallel many times, and the seed of the
pseudo-random number generator is the same in each run. An easy fix would
be to set a random seed in the start of each simulation.

.. code:: python

   # Set random seed, otherwise each experiment gets the same result
   seed = (os.getpid() * int(time.time())) % 123456789
   np.random.seed(seed)

Another method would be to pass the seed to the simulation. When
reproducability of the simulation results is important, then this is the
better option. Note how the seed is set inside the simulation and not
outside.

.. code:: python

   import multiprocessing

   def simulation(seed) -> float: 
       """ Returns a random number in [0, 1] for a specific random seed. """ 
       np.random.seed(seed)
       return np.random.rand(1)

   args = range(100)
   p = multiprocessing.Pool(4)

   for result in p.imap_unordered(func=simulation, iterable=args, chunksize=100//4):
       print(f"Result: {result}")

Note that this possible issue is specific to multiprocessing on Linux
operating system, as it depends on how new processes are created.


Noise scaling
-------------

For finer control of the noise model, the module exposes
``ScaledNoiseGates`` (uniform scaling), ``CustomNoiseGates`` (independently
scalable ``p``, ``T1``, ``T2``), ``CustomNoiseChannelsGates`` (mix noisy and
noise-free qubits) and ``SpecificNoiseGates`` (override input noise with
fixed values). Pre-configured gate sets are available as ``standard_gates``,
``numerical_gates``, ``almost_noise_free_gates``, ``low_pauli_noise_gates``
and ``long_coherence_gates``.


Instances and Classes
---------------------

.. automodule:: quantum_gates.gates
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
