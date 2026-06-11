

import numpy as np
import pymatching

import qiskit.qasm3 as qasm3
from qiskit import transpile
from qiskit.circuit.controlflow import ControlFlowOp 
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from loom.eka import Eka, Lattice
from loom.eka.operations import (ResetAllDataQubits, Merge, MeasureBlockSyndromes, MeasureLogicalZ)
from loom.interpreter import interpret_eka
from loom.executor import EkaToStimConverter, EkaToQasmConverter
from loom_rotated_surface_code.code_factory import RotatedSurfaceCode
import stim

from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates, noise_free_gates
from quantum_gates.circuits import EfficientCircuit
from quantum_gates.utilities import DeviceParameters

class RotatedSurfaceCodeLoom:
    def __init__(self, distance: int = 3, n_cycles: int = 5, n_shots: int = 1000, noise = False, p = 0.01):
        self.distance = distance
        self.n_cycles = n_cycles
        self.n_shots = n_shots
        self.noise = noise
        self.p = p

        # ── Build experiment and interpret ───────────────────────────────────
        lattice = Lattice.square_2d((15, 15))
        rsc = RotatedSurfaceCode.create(
            distance, distance, lattice, unique_label="rsc"
        )
        eka_experiment = Eka(lattice, blocks=[rsc], operations=[
            ResetAllDataQubits("rsc"),
            MeasureBlockSyndromes("rsc", n_cycles=n_cycles),
            MeasureLogicalZ("rsc"),
        ])
        self.final_state = interpret_eka(eka_experiment)

        # ── Get QASM circuit and register mappings ───────────────────────────
        self.qasm_program, self.q_register, self.c_register = (
            EkaToQasmConverter().convert(self.final_state)
        )
        self.qiskit_circuit = qasm3.loads(self.qasm_program)
        self.n_qubits = self.qiskit_circuit.num_qubits
        # ── Get Stim circuit and index lookup ────────────────────────────────
        self.stim_circuit, _, stim_c_register = (
            EkaToStimConverter().convert(self.final_state)
        )
        self.label_to_stim_idx = {
            chan.label: idx for chan, idx in stim_c_register.items()
        }
        self.matcher, self.converter = self.build_matcher()

    def build_matcher(self):
        converter = self.stim_circuit.compile_m2d_converter()  # measurement-to-detector converter
        
        # ── 8. Decode and get logical error rate ─────────────────────────────────────
        noisy_lines = self.stim_noise_model()
        noisy_circuit = stim.Circuit('\n'.join(noisy_lines))

        dem = noisy_circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
        )

        matcher = pymatching.Matching(dem)
        return matcher, converter


    def _build_initial_layout(self):
        """
        Maps each qubit index to its (row, col) position on the combined
        rotated grid using:
        Data qubit      (r, c, 0) -> (r+c,   d+c-r-1)
        Stabilizer qubit (r, c, 1) -> (r+c-1, d+c-r-1)
        Maps each logical qubit index to a physical qubit index on the
        combined (2d-1) x (2d-1) rotated grid.
        """
        d = self.distance
        grid_width = 2 * d - 1
        initial_layout = [None] * self.n_qubits
        qubit_to_grid = {}  # qubit_index -> (grid_row, grid_col)
        grid_to_qubit = {}  # (grid_row, grid_col) -> qubit_index

        for channel, qubit_idx in self.q_register.items():
            r, c, qubit_type = map(int, channel.label.strip("()").split(", "))

            if qubit_type == 0:  # data qubit
                grid_row = r + c
                grid_col = d + c - r - 1
            else:  # stabilizer qubit
                grid_row = r + c - 1
                grid_col = d + c - r - 1

            qubit_to_grid[qubit_idx] = (grid_row, grid_col)
            grid_to_qubit[(grid_row, grid_col)] = qubit_idx

            physical_idx = grid_row * grid_width + grid_col

            initial_layout[qubit_idx] = physical_idx

        return initial_layout


    def _get_device_parameters(self, backend, topology):
        """
        This method loads single- and two-qubit noise parameters from a given
        backend into a DeviceParameters object and converts them into a lookup
        dictionary suitable for simulation.
        """

        device_param = DeviceParameters(list(range(self.n_qubits)))
        device_param.load_from_backend(backend)
        device_param_lookup = device_param.__dict__()
        # The standard t_int value for allowed ECR gates
        t_int_value = 6.6e-07
        #Add t_int values for all consecutive qubit pairs
        non_zero_p_int =  device_param_lookup['p_int'][ device_param_lookup['p_int'] != 0]
        # Get average p_int for fallback
        avg_p_int = np.mean(non_zero_p_int) if len(non_zero_p_int) > 0 else 0.01

        for i in range(self.n_qubits - 1):
            # Add both directions for the ECR gate
            device_param_lookup['t_int'][i, i+1] = t_int_value
            device_param_lookup['t_int'][i+1, i] = t_int_value

            if device_param_lookup['p_int'][i, i+1] == 0:
                # Look for nearest existing edge with p_int value
                found_p_int = None
                
                # Check adjacent qubits first
                for offset in [1, 2, 3]:
                    if i - offset >= 0 and device_param_lookup['p_int'][i-offset, i-offset+1] != 0:
                        found_p_int = device_param_lookup['p_int'][i-offset, i-offset+1]
                        break
                    if i + offset < self.n_qubits - 1 and device_param_lookup['p_int'][i+offset, i+offset+1] != 0:
                        found_p_int = device_param_lookup['p_int'][i+offset, i+offset+1]
                        break
                
                # Fall back to average if no nearby edge found
                if found_p_int is None:
                    found_p_int = avg_p_int
                
                device_param_lookup['p_int'][i, i+1] = found_p_int
                device_param_lookup['p_int'][i+1, i] = found_p_int
        return device_param_lookup  
    
    def _transpile_circ(self, topology):
        """
        This function transpiles the input circuit using a fixed basis gate set
        and a nearest-neighbor coupling map. It also detects whether the circuit
        contains control-flow operations and adapts the transpilation scheduling
        accordingly.
        """

        needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in self.qiskit_circuit.data)
        if topology == "linear":
            initial_layout = list(range(self.n_qubits))
            cm = CouplingMap([(i, i+1) for i in range(self.n_qubits - 1)])
        elif topology == "grid":
            d = self.distance
            grid_width = 2 * d - 1
            n_physical = grid_width * grid_width
            cm = CouplingMap(
                [(i, i+1) for i in range(n_physical - 1)] +
                [(i, i + grid_width) for i in range(n_physical - grid_width)]
            )
            initial_layout = self._build_initial_layout()

        t_circ = transpile(
            self.qiskit_circuit,
            None,
            basis_gates=['ecr', 'rz', 'sx','x'],
            coupling_map=cm,
            initial_layout=initial_layout,
            seed_transpiler=42,
            scheduling_method=needs_controlflow,
        )
        return t_circ

    def MrAnderson_run_circ(self, topology):
        print("n_qubits:", self.n_qubits)
        backend = FakeBrisbane() 
        
        if self.noise:
            set_gate = standard_gates
        else:
            set_gate = noise_free_gates
        bit_flip_bool = False
        sim = MrAndersonSimulator(gates=set_gate, CircuitClass=EfficientCircuit)

        initial_psi = np.zeros(2**self.n_qubits)
        initial_psi[0] = 1.0  # set |00...0⟩

        device_param_lookup = self._get_device_parameters(backend, topology)

        t_circ = self._transpile_circ(topology)

        print("Circuit transpiled")
        # Run simulation
        res  = sim.run( 
            t_qiskit_circ=t_circ, 
            psi0=initial_psi, 
            shots=self.n_shots, 
            device_param=device_param_lookup,
            nqubit=self.n_qubits,
            bit_flip_bool=bit_flip_bool,
            )
        print("Simulation complete")
        return res["mid_counts"]

    def AER_run_circ(self):
        if self.noise:
            # Build noise model
            noise_model = NoiseModel()

            # Single-qubit gate depolarizing noise
            #single_qubit_error = depolarizing_error(p, 1)
            #noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 's', 'sdg', 'x', 'y', 'z', 'id'])

            # Two-qubit gate depolarizing noise
            two_qubit_error = depolarizing_error(self.p, 2)
            noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx', 'cy', 'cz'])

            # Measurement noise
            measure_error = depolarizing_error(self.p, 1)
            noise_model.add_all_qubit_quantum_error(measure_error, ['measure'])

            # Run with noise
            simulator = AerSimulator(method='stabilizer', noise_model=noise_model)
        
        else:
            simulator = AerSimulator(method='stabilizer')

        transpiled = transpile(self.qiskit_circuit, simulator, optimization_level=0)
        result = simulator.run(transpiled, shots=self.n_shots).result()
    
        return result.get_counts() 

    def stim_noise_model(self):
        # Add tiny uniform noise just to generate correct graph topology with boundary edges
        lines = str(self.stim_circuit).split('\n')
        noisy_lines = []
        for line in lines:
            noisy_lines.append(line)
            if line.startswith('M ') or line.startswith('MX') or line.startswith('MY'):
                qubits = ' '.join(line.split()[1:])
                noisy_lines.append(f'X_ERROR({self.p}) {qubits}')
            elif line.startswith('CX') or line.startswith('CZ'):
                qubits = ' '.join(line.split()[1:])
                noisy_lines.append(f'DEPOLARIZE2({self.p}) {qubits}')
            elif line.startswith('H') or line.startswith('R '):
                qubits = ' '.join(line.split()[1:])
                noisy_lines.append(f'DEPOLARIZE1({self.p}) {qubits}')
        return noisy_lines

    def run_circ(self, simulator="MrAnderson", topology="linear"):
        if simulator == "MrAnderson":
            self.qiskit_circuit.barrier()
            self.qiskit_circuit.x(range(self.qiskit_circuit.num_qubits))
            qiskit_result = self.MrAnderson_run_circ(topology)
        elif simulator == "AER":
            qiskit_result = self.AER_run_circ()
            
        else: 
            raise ValueError(f"Unknown simulator: {simulator}")
        
        outcomes = EkaToQasmConverter.parse_target_run_outcome((qiskit_result, self.c_register))
        n_measurements = len(self.label_to_stim_idx)

        # Reconstruct the full (shots × measurements) matrix in Stim order
        raw_matrix = np.zeros((self.n_shots, n_measurements), dtype=np.bool_)
        for label, stim_idx in self.label_to_stim_idx.items():
            raw_matrix[:, stim_idx] = outcomes[label]

        detection_events, obs_flips = self.converter.convert(
            measurements=raw_matrix,
            separate_observables=True,
        )

        # Decode
        predicted_flips = self.matcher.decode_batch(detection_events)
        logical_error_rate = np.mean(predicted_flips != obs_flips)
        return logical_error_rate