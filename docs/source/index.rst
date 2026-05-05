Welcome to the Quantum Gates documentation
==========================================

**Quantum Gates** is a Python library for simulating the noisy behaviour of 
real quantum devices. The noise is incorporated directly into the gates, 
which become stochastic matrices. The corresponding model was published in |prr_paper_link| 
in Physical Review Research.
 
.. |prr_paper_link| raw:: html

   <a href="https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043210" target="_blank">Di Bartolomeo, 2023</a>


.. note::

   This project is under active development.


How to install
--------------

Requirements
~~~~~~~~~~~~

The Python version should be 3.9 or later. We recommend using the library
together with a |ibm_quantum_lab_link| account, as it necessary for 
circuit compilation with Qiskit.

.. |ibm_quantum_lab_link| raw:: html

   <a href="https://quantum-computing.ibm.com/lab" target="_blank">IBM Quantum Lab</a>

Installation as a user
~~~~~~~~~~~~~~~~~~~~~~

The library is available on the |pip_link| with ``pip install quantum-gates``.

.. |pip_link| raw:: html

   <a href="https://pypi.org/project/quantum-gates/" target="_blank">Python Package Index (PyPI)</a>


Installation as a contributor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For users who want to contribute to the project or work with the latest source code, we recommend the following steps:

1. Clone the Repository from |github_link|:

   .. code-block:: bash

      git clone https://github.com/CERN-QTI/quantum-gates.git

2. Navigate to the project directory:

   .. code-block:: bash

      cd quantum-gates

3. Create and activate a virtual environment

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. Install the package in editable Mode:

   .. code-block:: bash

      pip install -e .

   Installing in editable mode allows you to work directly with the source code. Any changes you make will be reflected immediately without the need to reinstall the package.

.. |github_link| raw:: html

   <a href="https://github.com/CERN-QTI/quantum-gates" target="_blank">Github</a>

Functionality
--------------

Gates
~~~~~

To sample quantum gates with the noise incorporated in them, one can set
up a :doc:`gates` instance. The methods of this object return the matrices
as numpy array. The sampling is stochastic, and one can set the numpy seed
to always get the same sequence of noise. For finer control over noise, the
:doc:`gates <gates>` module also provides ``CustomNoiseGates``,
``CustomNoiseChannelsGates`` and ``SpecificNoiseGates`` along with gate sets
like ``numerical_gates`` and ``low_pauli_noise_gates``.


Factories
~~~~~~~~~

To produce :doc:`gates <gates>`, we use :doc:`factories <factories>`, such as the
CNOTFactory. One can combine factories into a custom :doc:`gates <gates>`
class. The factories have a construct() method, with a well documented 
signature. 

Pulses
~~~~~~

When constructing a set of quantum gates with the Gates class, one can
specify a Pulse instance. This pulse describes the shape of the RF pulses
used to implement the gates.

Integrators
~~~~~~~~~~~

Behind the scenes, we solve Ito integrals to deal with the different
pulse shapes. This is handled by the :doc:`integrator <integrators>`.

Simulators
~~~~~~~~~~

The :doc:`MrAndersonSimulator <simulators>` can be used to simulate
a quantum circuit transpiled with Qiskit with a specific 
:doc:`noisy gate set <gates>`.

Backends
~~~~~~~~

For the computation, we provide :doc:`backends <backends>` out of the box, 
such as the EfficientBackend that uses optimized tensor contractions
to simulate 20+ qubits with the statevector method.

Circuits
~~~~~~~~

The simulators can be configured with a :doc:`circuits` class, such as
EfficientCircuit. This class is responsible for sampling the
noisy gates. The class can be configured with a :doc:`gates` instance and one of
the :doc:`backends` that executes the statevector simulation. The circuit
classes also support mid-circuit measurement (``mid_measurement``), reset
gates (``reset``), and direct state-vector readout (``statevector_readout``).

Quantum Algorithms
~~~~~~~~~~~~~~~~~~

Four quantum algorithms are provided as functions which return the Qiskit circuit
for a specific number of qubits

* Hadamard reverse QFT circuit
* GHZ circuit
* QFT circuit
* Quantum Approximate Optimization Algorithm circuit


Legacy
~~~~~~

We also provide the :doc:`legacy <legacy>` implementations of the 
gates, simulator and circuit classes. They can be used for unit testing.

Utility
~~~~~~~

In performing quantum simulation, there are many steps that are performed repeatedly, such as setup the IBM backend with
setup_backend(), loading the noise information as DeviceParameters, transpiling the quantum circuits with
create_qc_list(), and executing the simulation in parallel  with multiprocessing_parallel_simulation() on a
powerful machine. For this reason, the most frequently used functions are part of the utilities.

Metrics
~~~~~~~

The :doc:`metrics <metrics>` module provides helpers like
``hellinger_distance`` for comparing simulated and reference distributions.

Qiskit Provider
~~~~~~~~~~~~~~~

The library exposes a :doc:`Qiskit provider <qiskit_provider>` so existing
Qiskit code can run noisy simulations without touching the call sites.


Structure
---------

.. toctree::
   :maxdepth: 3
   :caption: Gates

   gates

.. toctree::
   :maxdepth: 3
   :caption: Factories

   factories

.. toctree::
   :maxdepth: 3
   :caption: Pulses

   pulses

.. toctree::
   :maxdepth: 3
   :caption: Integrators

   integrators


.. toctree::
   :maxdepth: 2
   :caption: Simulators

   simulators

.. toctree::
   :maxdepth: 3
   :caption: Backends

   backends

.. toctree::
   :maxdepth: 3
   :caption: Circuits

   circuits

.. toctree::
   :maxdepth: 3
   :caption: Quantum Algorithm

   quantum_algorithms

.. toctree::
   :maxdepth: 3
   :caption: Legacy

   legacy

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   utilities

.. toctree::
   :maxdepth: 2
   :caption: Metrics

   metrics

.. toctree::
   :maxdepth: 2
   :caption: Qiskit Provider

   qiskit_provider


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

You may cite this library using the following BibTex entry:

.. code-block:: bibtex

   @article{PhysRevResearch.5.043210,
      title = {Noisy gates for simulating quantum computers},
      author = {Di Bartolomeo, Giovanni and Vischi, Michele and Cesa, Francesco and Wixinger, Roman and Grossi, Michele and Donadi, Sandro and Bassi, Angelo},
      journal = {Phys. Rev. Res.},
      volume = {5},
      issue = {4},
      pages = {043210},
      numpages = {19},
      year = {2023},
      month = {Dec},
      publisher = {American Physical Society},
      doi = {10.1103/PhysRevResearch.5.043210},
      url = {https://link.aps.org/doi/10.1103/PhysRevResearch.5.043210}
   }


Authors
-------

This project has been developed thanks to the effort of the following
people:

-  Giovanni Di Bartolomeo (dibartolomeo.giov@gmail.com)
-  Michele Vischi (vischimichele@gmail.com)
-  Francesco Cesa
-  Michele Grossi (michele.grossi@cern.ch)
-  Sandro Donadi
-  Angelo Bassi
-  Paolo Da Rold (paolo.darold@studenti.units.it)
-  Cherilyn Christen (cherilyn.christen@epfl.ch)
-  Nathan Pacey (npacey01@gmail.com)
-  Giulio Crognaletti (giulio.crognaletti@phd.units.it)
-  Roman Wixinger (roman.wixinger@gmail.com)
