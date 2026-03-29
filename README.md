# Noisy Quantum Gates [![Made at QMTS!](https://img.shields.io/badge/University%20of%20Trieste-Bassi%20Group-brightgreen)](http://www.qmts.it/) [![Made at CERN!](https://img.shields.io/badge/CERN-CERN%20openlab-brightgreen)](https://openlab.cern/) [![Made at CERN!](https://img.shields.io/badge/CERN-Open%20Source-%232980b9.svg)](https://home.cern) [![Made at CERN!](https://img.shields.io/badge/CERN-QTI-blue)](https://quantum.cern/our-governance)

Implementation of the Noisy Quantum Gates model, published in [Di Bartolomeo, 2023](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.043210). It is a novel method to simulate the noisy behaviour of quantum devices by incorporating the noise directly in the gates, which become stochastic matrices. 


## Documentations
The documentation for Noisy Quantum Gates can be accessed on the website 
<a href="https://quantum-gates.readthedocs.io/en/latest/index.html" target="_blank"> Read the Docs</a>.


## How to install
### Requirements
The Python version should be 3.9 or later. Find your Python version by typing `python` or `python3` in the CLI. 
We recommend using the repo together with an [IBM Quantum Lab](https://quantum-computing.ibm.com/lab) account, as it necessary for circuit compilation with Qiskit in many cases. 


### Installation as a user
The library is available on the Python Package Index (PyPI) with `pip install quantum-gates`. 


### Installation as a contributor
For users who want to have control over the source code, we recommend the following installation. 

1. Clone the repository:

```bash
git clone https://github.com/CERN-IT-INNOVATION/quantum-gates.git
```

2. Navigate to the project directory:

```bash
cd quantum_gates
```

3. Create virtual environment

You can either use your IDE to set this up automatically or do it manually in the CLI.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
This saves your environment from pollution.

4. Install the package in editable mode.

```bash
pip install -e .
```
This command installs the package in editable mode, allowing you to work directly with the source code. Any changes you 
make will be immediately available without the need to reinstall the package.


## Quickstart 
You can find this quickstart implemented in the tutorial notebook [here](docs/tutorials/notebooks/tutorial_quantum_gates.ipynb).

Execute the following code in a script or notebook. Add your IBM token to by defining it as the variable IBM_TOKEN = "your_token". Optimally, you save your token in a separate file that is not in your version control system, so you are not at risk of accidentally revealing your access token. 


```python
# Standard libraries
import numpy as np
import json

# Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram

# Own library
from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates
from quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from quantum_gates.utilities import DeviceParameters
from quantum_gates.utilities import setup_backend
IBM_TOKEN = "<your_token>"
```
We create a quantum circuit with Qiskit. 

```python
circ = QuantumCircuit(2,2)
circ.h(0)
circ.cx(0,1)
circ.barrier(range(2))
circ.measure(range(2),range(2))
circ.draw('mpl')
```

We load the configuration from a json file or from code with
```python
config = {
    "backend": {
        "hub": "ibm-q",
        "group": "open",
        "project": "main",
        "device_name": "ibm_kyiv"
    },
    "run": {
        "shots": 1000,
        "qubits_layout": [0, 1],
        "psi0": [1, 0, 0, 0]
    }
}
```
... and setup the Qiskit backend used for the circuit transpilation.

```python
backend_config = config["backend"]
backend = setup_backend(Token=IBM_TOKEN, **backend_config)
run_config = config["run"]
```

This allows us to load the device parameters, which represent the noise of the quantum hardware. 
```python
qubits_layout = run_config["qubits_layout"]
device_param = DeviceParameters(qubits_layout)
device_param.load_from_backend(backend)
device_param_lookup = device_param.__dict__()
```

Last, we perform the simulation ... 
```python
sim = MrAndersonSimulator(gates=standard_gates, CircuitClass=EfficientCircuit)

t_circ = transpile(
    circ,
    backend,
    scheduling_method='asap',
    initial_layout=qubits_layout,
    seed_transpiler=42
)

probs = sim.run(
    t_qiskit_circ=t_circ, 
    qubits_layout=qubits_layout, 
    psi0=np.array(run_config["psi0"]), 
    shots=run_config["shots"], 
    device_param=device_param_lookup,
    nqubit=2,
    level_opt = 0)

```
... and analyse the result. 

```python
plot_histogram(probs, bar_labels=False, legend=['Noisy Gates simulation'])
```

If you want to use a non-linear topology you must use the BinaryCircuit and slightly modify the code.
First of all we modify the qubit layout to match the topology of the device.

```python
config = {
    "run": {
        "shots": 1000,
        "qubits_layout": [0, 14],
        "psi0": [1, 0, 0, 0]
    }
}
```
Then also the command to import the parameter device has to change in order to import all the information of all the qubits up to the one with the max indices.

```python
device_param = DeviceParameters(list(np.arange(max(qubits_layout)+1)))
device_param.load_from_backend(backend)
device_param_lookup = device_param.__dict__()
```

Last, we perform the simulation ... 
```python
sim = MrAndersonSimulator(gates=standard_gates, CircuitClass=BinaryCircuit)

t_circ = transpile(
    circ,
    backend,
    scheduling_method='asap',
    initial_layout=qubits_layout,
    seed_transpiler=42
)

probs = sim.run(
    t_qiskit_circ=t_circ, 
    qubits_layout=qubits_layout, 
    psi0=np.array(run_config["psi0"]), 
    shots=run_config["shots"], 
    device_param=device_param_lookup,
    nqubit=2,
    level_opt = 4)

```
... and analyse the result. 

```python
plot_histogram(probs, bar_labels=False, legend=['Noisy Gates simulation'])
```

Remember that the ouput of the simulator follows the Big-Endian order, if you want to switch to Little-Endian order, which is the standard for Qiskit, you can use the command

```python
from quantum_gates.utilities import fix_counts

n_measured_qubit = 2
probs_little_endian = fix_counts(probs, n_measured_qubit)
plot_histogram(probs_little_endian, bar_labels=False, legend=['Noisy Gates simulation'])
```


# Usage
We recommend to read the [overview](https://quantum-gates.readthedocs.io/en/latest/index.html) of the documentation 
as a 2-minute preparation. You can import the package modules as shown in the Quickstart:

```python
from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates
from quantum_gates.circuits import EfficientCircuit
from quantum_gates.utilities import DeviceParameters, setup_backend
```


# Functionality
The main components are the [gates](https://quantum-gates.readthedocs.io/en/latest/gates.html), 
and the [simulator](https://quantum-gates.readthedocs.io/en/latest/simulators.html). 
One can configure the gates with different [pulse shapes](https://quantum-gates.readthedocs.io/en/latest/pulses.html>), 
and the simulator with different [circuit classes](https://quantum-gates.readthedocs.io/en/latest/circuits.html>) and 
[backends](https://quantum-gates.readthedocs.io/en/latest/backends.html). The circuit classes use a specific 
backend for the statevector simulation. 
The [EfficientBackend](https://quantum-gates.readthedocs.io/en/latest/backends.html) has the same functionality as 
the [StandardBackend](https://quantum-gates.readthedocs.io/en/latest/backends.html), but is much more performant 
thanks to optimized tensor contraction algorithms. We also provide various
[quantum algorithms](https://quantum-gates.readthedocs.io/en/latest/quantum_algorithms.html) as circuits, and 
scripts to run the circuits with the simulator, the IBM simulator, and a real IBM backend. Last, all functionality is 
unit tested and one can get sample code from the unit tests.


## Qiskit Support
While the quantum-gates library works as a provider for Qiskit and is part of the Qiskit Community ecosystem, there are
a few intentional differences to the Qiskit API described in this section.

### Statevector Readout via Labeled Barriers
There are a few design choices in this library that intentionally differ from standard Qiskit behaviour. These are important to understand when comparing results or debugging simulations with statevector readouts.

Statevector extraction is implemented using labeled barriers:

```python
circ.barrier(label="save_...")
```

This acts as a marker for when the simulator should record the statevector.

Why not use Qiskit’s `save_statevector`? Qiskit’s native `save_statevector` instruction does not survive the 
transpilation process, as it is not supported on real hardware. Using labeled barriers ensures compatibility with transpiled circuits and hardware-aware workflows.

---

### Statevector Ordering Convention

This library follows the theoretical ("by-hand") convention for tensor product ordering:

- quantum-gates (big-endian)
  - Qubit 0 is the leftmost (most significant index)
  - Basis ordering:
    $$|q_0 q_1\rangle = |00\rangle, |01\rangle, |10\rangle, |11\rangle$$

- Qiskit (little-endian):
  - Qubit 0 is the rightmost (least significant index) $|q_1 q_0\rangle $
  - Basis ordering is effectively reversed

Because of this, statevectors will **not directly match Qiskit outputs**.

To convert between conventions, use:
```python
from quantum_gates.utilities import sv_normal_to_qiskit
```

---

### Transpiler-Induced Statevector Permutations

Qiskit’s transpiler may permute qubit ordering, which affects statevectors but not density matrices (see https://github.com/Qiskit/qiskit/issues/5839).

This can lead to mismatches even if ordering conventions are handled correctly.

To fix this, we provide:

```python
from quantum_gates.utilities import permute_normal_sv_to_logical_normal
```

This utility restores the expected logical qubit ordering after transpilation.

---

When comparing with Qiskit results, discrepancies typically arise from:

- Different endianness conventions
- Transpiler qubit permutations
- Statevector extraction differences

Make sure to apply the provided utilities when performing comparisons.


# Unit Tests
We recommend running the unit tests once you are finished with the setup of your environment. As some tests need access
to IBM devices, you have to follow the steps outlined in token.py. Make sure that your token is active and you have 
accepted all license agreement with IBM in your IBM account. Before you run the tests, make sure that the device you are 
testing with (the on set in tests/simulation/test_anderson_simulator.py) is available in your account and the device 
parameters are prepared in the tests/utility/device_parameters folder. In the future we might upgrade the tests and mock 
these dependencies.

Afer activating your virtual environment, you can run the tests from root with 

```bash
python -m "pytest"
```

Using this command instead of just typing `pytest` will make sure that you are using
the right Python version to run pytest. You can also just run a subset of the test, for example:

```bash
python -m pytest -k test_gates_noiseless_cnot_inv
```

# How to contribute
Contributions are welcomed and should apply the usual git-flow: fork this repo and create a local branch. Commit often 
to ensure that each commit is easy to understand. Request a merge to the mainline often. Contribute to the test suite 
and verify the functionality with the unit tests when using a different Python version or dependency versions. 
Please remember to follow the [PEP 8 style guide](https://peps.python.org/pep-0008/), and add comments whenever it helps. 
The corresponding [authors](<#authors>) are happy to support you.

## Contribution guidelines

### Branching and commits
- Create feature branches from `main` named `feature/<description>` (e.g. `feature/mid-circuit-measurement`).
- Keep pull requests small and focused. A PR that touches one feature or fix is easier to review and less likely to
  introduce regressions. If your work spans multiple concerns, split it into separate PRs.
- Commit often with clear messages prefixed by the branch name, e.g. `[xyz] Added mid-measurement support`.
- Each commit should be self-contained and leave the codebase in a working state.

### Writing Tests
- Every new feature or bug fix should include corresponding unit tests.
- Tests serve a dual purpose: they verify correctness and document behaviour. A reviewer should be able to
  understand what a feature does by reading its tests.
- Place tests in the appropriate subdirectory under `tests/` (`simulation/`, `utility/`, `gates/`, `qiskit_provider/`). 
  The test location should mirror the structure of src/. For example, changes in src/gates/pulse.py should be 
  accompanied by tests in tests/gates/test_pulse.py.
- Use `pytest` parametrize to cover multiple configurations (qubit counts, circuit types, noise levels) without
  duplicating test logic.
- When testing against Qiskit's AER simulator, compare statevectors or measurement distributions to validate
  correctness (see `tests/simulation/test_mid_circuit.py` for examples).

### Pull Request Process

PRs are reviewed with three goals in mind:

1. Functional correctness: Does the code work as intended? Are edge cases handled? Do all tests pass?
2. Code quality: Does the change maintain or improve the codebase? Is there unnecessary duplication, dead code,
   or unclear naming? PRs should leave the code at least as clean as they found it to avoid accumulating technical debt.
3. Knowledge sharing: Reviews are didactical. Reviewers should explain why a change is requested. Authors should add 
   comments where the logic is non-obvious.

PR checklist for authors:
- [ ] All existing tests pass (`python -m pytest`).
- [ ] New tests are added for new functionality.
- [ ] Code follows PEP 8 convention whenever possible.
- [ ] No secrets, tokens, or large binary files are committed.
- [ ] The PR description explains what changed and why.
- [ ] If the change affects the public API, update the docstrings and README examples.

PR checklist for reviewers:
- [ ] Read the PR description and understand the motivation.
- [ ] Run the tests locally to confirm they pass.
- [ ] Check that new code has test coverage.
- [ ] Look for potential regressions, performance issues, and code clarity.
- [ ] Leave constructive comments and suggest concrete alternatives when requesting changes.


## Build 
You may also want to create your own distribution and test it. Navigate to the repository in your CLI of choice. 
Build the wheel with the command `python3 -m build --sdist --wheel .` and navigate to the distribution with `cd dist`. 
Use `ls` to display the name of the wheel, and run `pip install <filename>.whl` with the correct filename. 
Now you can use your version of the library. 

## Building the documentation

The documentation for Noisy Quantum Gates is built using Sphinx and is hosted on 
[ReadTheDocs](https://quantum-gates.readthedocs.io/en/latest/index.html). If you wish to build and view the 
documentation locally, follow these steps:

1. Navigate to the `docs` directory:

From the root of the project, navigate to the `docs` folder:

```bash
cd docs
```

2. Install the documentation requirements:

```bash
pip install -r requirements.txt
```
Note: It's recommended to perform this step in your virtual environment.

3. Build the HTML documentation:

Use the make command to build the HTML version of the documentation:

```bash
make html
```

4. View and check the documentation locally

Open the generated index.html file in your web browser to view the documentation:

```bash
open build/html/index.html      # On macOS
xdg-open build/html/index.html  # On Linux
start build\html\index.html     # On Windows (Command Prompt)
```

Or manually navigate to the build/html directory and open index.html with your preferred web browser.


# Credits
Please cite the work using the following BibTex entry:

```text
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
```


# Authors
This project has been developed thanks to the effort of the following people:

* Giovanni Di Bartolomeo (dibartolomeo.giov@gmail.com)
* Michele Vischi (vischimichele@gmail.com)
* Francesco Cesa
* Michele Grossi (michele.grossi@cern.ch) 
* Sandro Donadi
* Angelo Bassi 
* Paolo Da Rold 
* Cherilyn Christen
* Nathan Pacey (npacey01@gmail.com)
* Roman Wixinger (roman.wixinger@gmail.com)
