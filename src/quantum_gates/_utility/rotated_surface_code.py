import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates, NoiseFreeGates
from quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from quantum_gates.utilities import DeviceParameters

import numpy as np
import matplotlib.pyplot as plt
import pymatching
from scipy.sparse import lil_matrix
from qiskit.circuit.controlflow import ControlFlowOp 

from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.fake_provider import FakeBrisbane


class RotatedSurfaceCode:
    def __init__(self, distance=3, cycles = 1):
        self.d = distance
        self.n_rows = 2 * distance + 1
        self.cycles = cycles
        self.n_data = distance**2 
        self.n_stabilizers = distance**2 - 1 
        self.n_qubits = self.n_data + self.n_stabilizers

        # Quantum register
        self.q_register = QuantumRegister(self.n_qubits, 'q')

        # One ancilla classical register per cycle
        self.cycle_cregs = []
        for cyc in range(self.cycles):
            creg = ClassicalRegister(self.n_stabilizers, f'c_ancilla_{cyc}')
            self.cycle_cregs.append(creg)

        # One classical register for data
        self.c_data = ClassicalRegister(self.n_data, 'c_data')

        # Build circuit
        self.qc = QuantumCircuit(self.q_register, *self.cycle_cregs, self.c_data)

        # define data and stabilizer indices based on even/odd parity

        data_qubits, self.x_stabilizers, self.z_stabilizers,  self.neighbors = self._get_surface_code_layout()
        self.stabilizers = self.x_stabilizers + self.z_stabilizers
        self.data_qubits = [global_idx for (row, global_idx) in data_qubits]


        # Sort stabilizers by global_index (second element of tuple)
        sorted_stabilizers = sorted(self.stabilizers, key=lambda x: x[1])

        # Create mapping with classical indices 0-7
        self.stabilizer_to_clbits = {
            stab: [i] for i, stab in enumerate(sorted_stabilizers)
        }

        print("Stabilizers:", self.stabilizers)
        print("Data qubits:", self.data_qubits)
        print("X stabilizers:", self.x_stabilizers)
        print("Z stabilizers:", self.z_stabilizers)
        print("n_qubits =", self.n_qubits)


        self._build_stabilizer_layer()
        # Compute stabilizer connections BEFORE setting up decoder
        self.stabilizer_to_dqubits = self._compute_stabilizer_connections()
    
        # Now setup the decoder (needs x_stabilizers, z_stabilizers, and stabilizer_connections)
        self.setup_decoder()
        
    def _get_surface_code_layout(self):
        """
        Construct the rotated surface code layout following user rules:

        Row structure:
        - Row 0: floor(d/2) Z stabilizers, then d Z stabilizers, then floor(d/2) Z stabilizers
        - Rows 1,3,... : d data qubits centered
        - Rows 2,4,... : d alternating X/Z stabilizers centered, starting with X
        - Last row (2d-2): same as row 0
        """

        d = self.d
        n_rows = self.n_rows
        half = d // 2

        data = []
        stab_x = []
        stab_z = []
        neighbors = {} 

        index = 0  # simple linear indexing

        for r in range(n_rows):

            # ---- Row 0 (top) and last row: Z boundary stabilizers ----
            if r == 0 or r == n_rows-1:
                if r == 0: factor = +1 * math.ceil(d/2)
                else: factor = -1 * d
                index_neighbor = 0
                for k in range(half):
                    stab_z.append((r, index)); 
                    neighbor = []
                    neighbor.append(index + factor  + index_neighbor )
                    neighbor.append(index + factor + 1  + index_neighbor )
                    neighbors[index] = neighbor
                    index_neighbor += 1
                    index += 1
                
                continue

            # ---- Data row (odd rows) ----
            if r % 2 == 1:
                for _ in range(d):
                    data.append((r, index))
                    index += 1
                continue

            # ---- Stabilizer row (even rows except boundary) ----
            if r % 2 == 0:
                # Alternating X/Z, starting with X
                for k in range(d):
                    if k % 2 == 0 :
                        stab_x.append((r, index))
                        if r %4 == 2:
                            if k == 0:
                                neighbor = []
                                neighbor.append(index - d)
                                neighbor.append(index + d)
                                neighbors[index] = neighbor
                            else:
                                neighbor = []
                                neighbor.append(index - d -1)
                                neighbor.append(index - d )
                                neighbor.append(index + d -1)
                                neighbor.append(index + d )
                                neighbors[index] = neighbor

                        else: 
                            if k == d -1:
                                neighbor = []
                                neighbor.append(index - d)
                                neighbor.append(index + d)
                                neighbors[index] = neighbor
                            else:
                                neighbor = []
                                neighbor.append(index - d )
                                neighbor.append(index - d +1)
                                neighbor.append(index + d )
                                neighbor.append(index + d +1)
                                neighbors[index] = neighbor
                    else:
                        stab_z.append((r, index))
                        if r % 4 == 2:
                            neighbor = []
                            neighbor.append(index - d -1)
                            neighbor.append(index - d )
                            neighbor.append(index + d -1)
                            neighbor.append(index + d )
                            neighbors[index] = neighbor
                        else: 
                            neighbor = []
                            neighbor.append(index - d)
                            neighbor.append(index - d +1)
                            neighbor.append(index + d)
                            neighbor.append(index + d +1)
                            neighbors[index] = neighbor
                    index += 1
                continue

        return data, stab_x, stab_z, neighbors


    
    

    def _build_stabilizer_layer(self):
        """Build one stabilizer measurement cycle for a distance-n surface code."""

        # repeat the stabilizer-measurement process for all cycles
        for cycle in range(self.cycles):
            # --- Reset all stabilizers at the beginning of the cycle ---
            for stabilizer in self.x_stabilizers + self.z_stabilizers:
                anc = stabilizer[1]
                self.qc.reset(anc)
                
            # --- X stabilizers ---
            for stabilizer in self.x_stabilizers:
                anc = stabilizer[1]
                self.qc.h(anc)  # prepare X stabilizer in |+>
                self.qc.barrier()

                # entangle with data qubits (ancilla is control)
                neighbor = self.neighbors[anc]
                for nb in neighbor:
                    self.qc.cx(anc, nb)

                self.qc.barrier()
                self.qc.h(anc)  # rotate back before measurement

            # --- Z stabilizers ---
            for stabilizer in self.z_stabilizers:
                anc = stabilizer[1]
                # entangle with data qubits (data is control, ancilla target)
                neighbor = self.neighbors[anc]
                for nb in neighbor:
                    self.qc.cx(nb, anc)

                self.qc.barrier()
            
            #conditional gate on the mid-circuit measurement results to run the decoder and determine the basis for the end measrement.

            # --- Measure all stabilizers for this cycle --- 
            
            stab_list = sorted(self.x_stabilizers + self.z_stabilizers)
            for i, stabilizer in enumerate(stab_list):
                anc = stabilizer[1]
                self.qc.measure(anc, self.cycle_cregs[cycle][i])

            self.qc.barrier(label="save_statevector")
        # --- Final measurement of data qubits in Z basis ---
        for i, data_qubit in enumerate(self.data_qubits):
            self.qc.measure(data_qubit, self.c_data[i])
        self.qc.barrier()

        self.qc.x(range(0,self.n_qubits))  # introduce bit-flip errors on all data qubits
        self.qc.barrier()


    def _compute_stabilizer_connections(self):
        """
        Compute which data qubits each stabilizer measures.
        Returns dict: stabilizer_index -> list of data qubit indices
        """
        connections = {}
        
        for stab in self.x_stabilizers + self.z_stabilizers:
            anc = stab[1]
                # entangle with data qubits (data is control, ancilla target)
            neighbors = self.neighbors[anc]
            connections[stab] = neighbors
        return connections
    

    def _extract_stabilizer_measurements(self, bitstring):
        """
        Extract measurement results for all stabilizers across all cycles.
        Returns separate arrays for X and Z stabilizers.
        """
        bits = bitstring[::-1]  # reverse Qiskit order
        print("Extracting stabilizer measurements from bitstring:", bits)
        x_syndromes = np.zeros((self.cycles, len(self.x_stabilizers)), dtype=int)
        z_syndromes = np.zeros((self.cycles, len(self.z_stabilizers)), dtype=int)

        for i, stab in enumerate(self.x_stabilizers):
            clbits = self.stabilizer_to_clbits[stab]
            print("clbits for X stabilizer", stab, ":", clbits)
            x_syndromes[:, i] = [int(bits[cb]) for cb in clbits]
        
        for i, stab in enumerate(self.z_stabilizers):
            clbits = self.stabilizer_to_clbits[stab]
            print("clbits for Z stabilizer", stab, ":", clbits)
            z_syndromes[:, i] = [int(bits[cb]) for cb in clbits]
        
        return x_syndromes, z_syndromes


    def run_surfacecode(self, noise, shots = 1):
        """Run the surface code circuit on the specified backend with noise model."""
        backend = FakeBrisbane()
        
        if noise:
            set_gate = standard_gates
            bit_flip_bool = True
        else:
            set_gate = NoiseFreeGates
            bit_flip_bool = False

        sim = MrAndersonSimulator(gates=set_gate, CircuitClass=EfficientCircuit)
        

        needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in self.qc.data)
        
        # Get backend's coupling map edges
        backend_coupling = backend.coupling_map

        cm = CouplingMap([(i, i+1) for i in range(self.n_qubits - 1)])

        t_circ = transpile(
            self.qc,
            backend = None,
            basis_gates=['ecr', 'rz', 'sx','x'],
            coupling_map=cm,
            initial_layout=list(range(self.n_qubits)),
            seed_transpiler=42,
            scheduling_method=needs_controlflow,      

        )


        # Check which qubits are actually used in transpiled circuit
        used_qubits: list[int] = []
        for instr in t_circ.data:
            op = instr.operation
            if op.name == 'delay' or op.name == 'barrier':
                continue
            # support any arity
            for qb in instr.qubits:
                q = qb._index
                if q not in used_qubits:
                    used_qubits.append(q)
                    
        print(f"Qubits used in transpiled circuit: {sorted(used_qubits)}")

        max_qubit = max(used_qubits)
        nqubit_actual = max_qubit + 1

        initial_psi = np.zeros(2**nqubit_actual)
        initial_psi[0] = 1.0  # set |00...0⟩

        device_param = DeviceParameters(list(range(nqubit_actual)))
        device_param.load_from_backend(backend)
        device_param_lookup = device_param.__dict__()
        # The standard t_int value for allowed ECR gates
        t_int_value = 6.6e-07
        #Add t_int values for all consecutive qubit pairs
        # Get average p_int
        non_zero_p_int =  device_param_lookup['p_int'][ device_param_lookup['p_int'] != 0]
        avg_p_int = np.mean(non_zero_p_int) if len(non_zero_p_int) > 0 else 0.01

        print(f"Using average p_int: {avg_p_int}")

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
                    print("Warning: No nearby p_int found")
                    found_p_int = avg_p_int
                
                device_param_lookup['p_int'][i, i+1] = found_p_int
                device_param_lookup['p_int'][i+1, i] = found_p_int
                print(f"Set p_int for edge ({i},{i+1}): {found_p_int}")

        print("Device parameters loaded for qubits:", device_param_lookup)
        res  = sim.run( 
            t_qiskit_circ=t_circ,  
            psi0=initial_psi, 
            shots=shots, 
            device_param=device_param_lookup,
            nqubit=nqubit_actual,
            bit_flip_bool=bit_flip_bool,
            )

        probs = res["probs"]
        results = res["results"]
        num_clbits = res["num_clbits"]
        mid_counts = res["mid_counts"]
        statevector_readout = res["statevector_readout"]

        # --- Process mid_counts to separate registers ---
        register_size= [self.n_data]+ [self.n_stabilizers]*self.cycles 
        
        processed_counts = self._split_counts(mid_counts, register_size)
        
        for bitstring, count in mid_counts.items():

            bitstring = bitstring.strip()
            syndrome_bits = bitstring[self.n_data:]
            predictions_x, predictions_z = self.decode_full(syndrome_bits)
            final_x_correction = np.bitwise_xor.reduce(predictions_x, axis=0)
            final_z_correction = np.bitwise_xor.reduce(predictions_z, axis=0)

            
            print("Syndrome bits:", syndrome_bits)
            print("Final X corrections:", final_x_correction)
            print("Final Z corrections:", final_z_correction)

            for i in range(len(predictions_x)): 
                print( 
                        f"X prediction {final_x_correction[i]}, "
                        f"Z prediction {final_z_correction[i]}"
                    )
                
                if final_x_correction[i] == 1:
                    new_bit = '1' if data_bits[i] == '0' else '0'
                    data_bits = data_bits[:i] + new_bit + data_bits[i+1:]

        
        return processed_counts, t_circ

    def _split_counts(self, raw_counts: dict, register_sizes: list) -> dict:
        """
        Splits the concatenated bitstrings in the dictionary keys based on a list of register sizes.

        Args:
            raw_counts (dict): Dictionary where keys are concatenated bitstrings.
            register_sizes (list): List of integers representing the size of each classical register.

        Returns:
            dict: A new dictionary with keys split by spaces according to register_sizes.
        """
        processed_counts = {}
        expected_length = sum(register_sizes)
        
        for raw_key, count in raw_counts.items():
            if len(raw_key) != expected_length:
                print(f"Warning: Skipping key {raw_key}. Expected total length {expected_length}, got {len(raw_key)}.")
                continue

            bit_chunks = []
            current_idx = 0
            
            # Dynamically slice the bitstring
            for size in register_sizes:
                chunk = raw_key[current_idx : current_idx + size]
                bit_chunks.append(chunk)
                current_idx += size

            # Join the chunks with spaces
            new_key = " ".join(bit_chunks)
            
            processed_counts[new_key] = count
            
        return processed_counts


# ========================================================================
#  PLOTTING METHODS
#  ========================================================================
   
    def analyze_results(self, processed_counts):
        """
        Pretty-print the already-processed measurement counts.
        Returns counts organized by cycle.
        """
        results_by_cycle = {}
        for bitstring, count in processed_counts.items():
            cycle_measurements = bitstring.split()
            for cycle_idx, cycle_bits in enumerate(cycle_measurements):
                if cycle_idx not in results_by_cycle:
                    results_by_cycle[cycle_idx] = {}
                results_by_cycle[cycle_idx][cycle_bits] = results_by_cycle[cycle_idx].get(cycle_bits, 0) + count
        
        return results_by_cycle

    def _plot_single_shot(self, bitstring, shot_idx=None):
        """Plot single-shot stabilizer measurements timeline."""
        plt.figure(figsize=(10, 3))
        
        # Use the helper function
        x_syndromes, z_syndromes = self._extract_stabilizer_measurements(bitstring)
        
        # Plot X stabilizers
        for i, stab in enumerate(self.x_stabilizers):
            bits = x_syndromes[:, i]  # Get column i (all cycles for this stabilizer)
            plt.step(
                range(1, len(bits) + 1),
                bits,
                where='mid',
                label=f'X stabilizer {stab}',
                marker='o',
                linewidth=2,
                markersize=6,
                alpha=0.7,
                linestyle='-'
            )

        # Plot Z stabilizers
        for i, stab in enumerate(self.z_stabilizers):
            bits = z_syndromes[:, i]  # Get column i (all cycles for this stabilizer)
            plt.step(
                range(1, len(bits) + 1),
                bits,
                where='mid',
                label=f'Z stabilizer {stab}',
                marker='s',
                linewidth=2,
                markersize=6,
                alpha=0.7,
                linestyle='--'
            )

        title = "Single-shot stabilizer measurements"
        if shot_idx is not None:
            title += f" — Shot {shot_idx + 1}"
        plt.title(title, fontsize=14)
        plt.xlabel("Cycle", fontsize=12)
        plt.ylabel("Measurement", fontsize=12)
        plt.yticks([0, 1])
        plt.xticks(range(1, self.cycles + 1))
        plt.grid(alpha=0.4)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    def plot_results(self, counts, plot_each_shot):
        """
        Plot measurement results for the surface code.
        - If 1 shot: timeline plot of stabilizer outcomes per cycle.
        - If >1 shots:
            - If plot_each_shot=True: plot each shot individually like single-shot mode.
            - If plot_each_shot=False: plot aggregated histogram (default behavior).
        """
        total_shots = self.shots
        # --- Case 1: only one shot ---
        if total_shots == 1:
            bitstring = list(counts.keys())[0][::-1]
            self._plot_single_shot(bitstring)
            return

        # --- Case 2: multiple shots ---
        if plot_each_shot:
            all_bitstrings = list(counts.keys())
            for i in range(total_shots):
                bitstring = all_bitstrings[i][::-1]   # get i-th bitstring, reversed
                self._plot_single_shot(bitstring, shot_idx=i)
        else:
            analyzed = self.analyze_results(counts)
            flat_results = {
                f"stab{stab}:{key}": val
                for stab, patterns in analyzed.items()
                for key, val in patterns.items()
            }

            labels = list(flat_results.keys())
            values = list(flat_results.values())

            plt.figure(figsize=(10, 3))
            plt.bar(range(len(labels)), values, color='steelblue', edgecolor='black')

            plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
            plt.xlabel("Stabilizer syndromes")
            plt.ylabel("Counts")
            plt.title(f"{total_shots} shots - Stabilizer syndromes")

            plt.tight_layout()
            plt.show()
    

# ========================================================================
#  DECODER METHODS
#  ========================================================================

    def _build_parity_check_matrix(self, stabilizer_type):
        """
        Build the parity-check matrix for the specified stabilizer type ('X' or 'Z').
        Returns a scipy sparse matrix in CSR format.
        """
        if stabilizer_type == 'X':
            stabs = self.x_stabilizers
        elif stabilizer_type == 'Z':
            stabs = self.z_stabilizers
        else:
            raise ValueError("stabilizer_type must be 'X' or 'Z'")

        n_stabs = len(stabs)
        n_data = len(self.data_qubits)

        # Create a sparse matrix in LIL format for easy construction
        H = lil_matrix((n_stabs, n_data), dtype=int)

        for i, stab in enumerate(stabs):
            data_qubits = self.stabilizer_to_dqubits[stab]
            for dq in data_qubits:
                data_idx = self.data_qubits.index(dq)
                H[i, data_idx] = 1
        
        print(H.toarray())
        return H.tocsr()
    
    def setup_decoder(self):
        """
        Setup the decoder by building parity-check matrices for X and Z stabilizers.
        """
        self.H_X = self._build_parity_check_matrix('X')
        self.H_Z = self._build_parity_check_matrix('Z')
        print("Decoder setup complete.")

 
    def _decode_per_cycle(self, syndrome1, syndrome2, which='X'):
        # Bitwise XOR between consecutive cycles
        syndrome = np.bitwise_xor(syndrome1, syndrome2).astype(np.uint8).flatten()

        # Choose which stabilizer matrix to use
        if which == 'X':
            H = self.H_X
        elif which == 'Z':
            H = self.H_Z
        else:
            raise ValueError("which must be 'X' or 'Z'")

        matching = pymatching.Matching(H)
    
        # Sanity check
        assert syndrome.shape[0] == H.shape[0], \
            f"Syndrome length {syndrome.shape[0]} != {H.shape[0]} stabilizers"

        prediction = matching.decode(syndrome)
        return prediction
    
    def decode_full(self, bitstring):
        """
        Decode the full syndrome history across all cycles.
        The bitstring contains concatenated stabilizer measurements
        """
        if isinstance(bitstring, dict):
            bitstring = list(bitstring.keys())[0]

        x_syndromes, z_syndromes = self._extract_stabilizer_measurements(bitstring)
        print("Extracted X syndromes:\n", x_syndromes)
        print("Extracted Z syndromes:\n", z_syndromes)
        n_cycles = self.cycles

        prediction_x = []
        prediction_z = []

        for c in range(n_cycles - 1):
            # decode between consecutive cycles
            x1, x2 = x_syndromes[c], x_syndromes[c + 1]
            z1, z2 = z_syndromes[c], z_syndromes[c + 1]

            prediction_x_cycle = self._decode_per_cycle(x1, x2, which="X")
            prediction_z_cycle = self._decode_per_cycle(z1, z2, which="Z")

            prediction_x.append(prediction_x_cycle)
            prediction_z.append(prediction_z_cycle)

        return np.array(prediction_x), np.array(prediction_z)
        