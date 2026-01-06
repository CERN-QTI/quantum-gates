import math
import numpy as np
import pymatching
from scipy.sparse import lil_matrix

from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates, NoiseFreeGates
from quantum_gates.gates import CustomNoiseChannelsGates
from quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from quantum_gates.utilities import DeviceParameters

from qiskit.circuit.controlflow import ControlFlowOp 
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile




class RotatedSurfaceCode:
    def __init__(self, distance=3, cycles = 1, aer = False, transpile = True, error = None):
        """
        Initialize a surface-code quantum circuit with a given code distance and
        number of stabilizer measurement cycles.

        This constructor sets up:
        - The surface-code lattice geometry (data and stabilizer qubits)
        - Quantum and classical registers
        - An initial logical state on the data qubits
        - Stabilizer measurement layers
        - Data structures required for decoding

        Parameters
        ----------
        distance : int, optional
            Code distance of the surface code. Determines the lattice size,
            number of data qubits (distance^2), and number of stabilizers
            (distance^2 - 1). Default is 3.

        cycles : int, optional
            Number of stabilizer measurement cycles to perform. Each cycle
            uses a separate classical register for ancilla measurements.
            Default is 1.

        aer : bool, optional
            If True, the circuit is intended to be run on the Qiskit Aer simulator.
            Default is False.

        transpile : bool, optional
            Whether to transpile the circuit before execution or return un-transpiled. Default is True.

        error : tuple or None, optional
            Optional injected error specified as (cycle, qubit_index, error_type),
            used for controlled fault injection during simulation. Default is None.

        """

        self.d = distance
        self.n_rows = 2 * distance + 1
        self.cycles = cycles
        self.n_data = distance**2 
        self.n_stabilizers = distance**2 - 1 
        self.n_qubits = self.n_data + self.n_stabilizers
        self.aer = aer
        self.transpile = transpile
        self.error = error  # tuple (cycle, qubit_index, error_type)

        self.initial_state = np.zeros(2**self.n_qubits, dtype=complex)
        indices = [np.int64(0), np.int64(9), np.int64(54), np.int64(63), np.int64(209), np.int64(216), np.int64(231), np.int64(238), np.int64(278), np.int64(287), np.int64(288), np.int64(297), np.int64(455), np.int64(462), np.int64(497), np.int64(504)]
        for indices in indices:
            self.initial_state[indices] = 1 / np.sqrt(16)
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
        self._build_stabilizer_layer()
        # Compute stabilizer connections BEFORE setting up decoder
        self.stabilizer_to_dqubits = self._compute_stabilizer_connections()
    
        # Now setup the decoder (needs x_stabilizers, z_stabilizers, and stabilizer_connections)
        self.setup_decoder()

# ========================================================================
#  LAYOUT METHODS
#  ========================================================================  

    def _get_surface_code_layout(self):
        """
        This method assigns indices to all qubits in the rotated surface
        code and classifies them as data qubits, X-type stabilizer qubits, or
        Z-type stabilizer qubits. It also determines the connectivity between
        each stabilizer qubit and its neighboring data qubits, accounting for
        edges.


        Returns
        -------
        data : list of tuple
            List of (row_index, global_qubit_index) for all data qubits.

        stab_x : list of tuple
            List of (row_index, global_qubit_index) for X-type stabilizer qubits.

        stab_z : list of tuple
            List of (row_index, global_qubit_index) for Z-type stabilizer qubits.

        neighbors : dict
            Dictionary mapping each stabilizer's global qubit index to a list
            of global indices of neighboring data qubits used in stabilizer
            measurements.
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
        """
        This method constructs the full sequence of stabilizer measurement cycles
        for the surface code. For each cycle, all stabilizer ancilla qubits are
        reset, entangled with their neighboring data qubits according to their
        X- or Z-type stabilizer definition, and then measured into a cycle-specific
        classical register.

        The method supports multiple stabilizer cycles, optional statevector
        saving when using the Aer simulator, and concludes with a final Z-basis
        measurement of all data qubits.

        """

        # repeat the stabilizer-measurement process for all cycles
        if self.aer and not self.transpile:
                self.qc.set_statevector(self.initial_state)
    
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
            
    
            # --- Measure all stabilizers for this cycle --- 
            
            stab_list = sorted(self.x_stabilizers + self.z_stabilizers)
            for i, stabilizer in enumerate(stab_list):
                anc = stabilizer[1]
                self.qc.measure(anc, self.cycle_cregs[cycle][i])

            
            if self.aer and not self.transpile:
                self.qc.save_statevector(label=f"save_sv_{cycle}")
            else:
                self.qc.barrier(label=f"save_sv_{cycle}")

            
        # --- Final measurement of data qubits in Z basis ---
        for i, data_qubit in enumerate(self.data_qubits):
            self.qc.measure(data_qubit, self.c_data[i])
        self.qc.barrier()

        self.qc.x(range(0,self.n_qubits))  # make sure it's detected as mid-measurement
        self.qc.barrier()

    def transpile_circ(qc, n_qubits):
        """
        This function transpiles the input circuit using a fixed basis gate set
        and a nearest-neighbor coupling map. It also detects whether the circuit
        contains control-flow operations and adapts the transpilation scheduling
        accordingly.
        """

        needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in qc.data)

        cm = CouplingMap([(i, i+1) for i in range(n_qubits - 1)])

        t_circ = transpile(
            qc,
            None,
            basis_gates=['ecr', 'rz', 'sx','x'],
            coupling_map=cm,
            initial_layout=list(range(n_qubits)),
            seed_transpiler=42,
            scheduling_method=needs_controlflow,
        )
        return t_circ
# ========================================================================
#  RUN METHOD
#  ========================================================================

    def run_surfacecode(self, correction, noise, p_scale=1, T1_scale=1, T2_scale=1, shots = 1):
        """
        Execute the surface-code circuit with optional noise and decoding.

        This method transpiles and runs the surface-code quantum circuit either
        noiselessly or with a custom noise model derived from device parameters.
        Depending on the configuration, the method can return the raw circuit,
        the transpiled circuit, intermediate measurement results, or decoded and
        corrected outcomes.

        Parameters
        ----------
        correction : bool
            If True, apply a decoder to the stabilizer measurement outcomes and
            return corrected logical results.

        noise : bool
            If True, run the circuit with custom noise channels scaled from
            device parameters. If False, use noiseless gates.

        p_scale : float
            Scaling factor applied to stochastic error probabilities in the
            noise model. Default is 1 (no scaling).

        T1_scale : float
            Scaling factor applied to T1 relaxation times in the noise model. Default is 1 (no scaling).

        T2_scale : float
            Scaling factor applied to T2 dephasing times in the noise model. Default is 1 (no scaling).

        shots : int, optional
            Number of simulation shots to perform. Default is 1.

        Returns
        -------
        If correction is False:
            mid_counts : dict
                Raw mid-circuit measurement counts from stabilizer registers.
            t_circ : QuantumCircuit
                The transpiled surface-code circuit.
            barrier_statevectors : dict
                Statevectors saved at stabilizer-cycle barriers.

        If correction is True:
            processed_counts : dict
                Corrected and register-separated measurement counts.
            data_counts : dict
                Final data-qubit measurement outcomes.
            t_circ : QuantumCircuit
                The transpiled surface-code circuit.
            barrier_statevectors : dict
                Statevectors saved at stabilizer-cycle barriers.
            predictions_x : array-like
                Decoder predictions for X-type errors.
            predictions_z : array-like
                Decoder predictions for Z-type errors.
        """

        backend = FakeBrisbane()
        stabilizer_indices = [coord[1] for coord in self.stabilizers]
        if noise:
            custom_channel_gates = CustomNoiseChannelsGates(
            noiseless_qubits=stabilizer_indices,  # qubits with no noise channels
            scale = False, # take device parameters as is
            p_scale=p_scale,
            T1_scale=T1_scale,
            T2_scale=T2_scale,
            )
            set_gate = custom_channel_gates
            bit_flip_bool = True
        else:
            set_gate = NoiseFreeGates
            bit_flip_bool = False
        

        sim = MrAndersonSimulator(gates=set_gate, CircuitClass=EfficientCircuit)
        

        
        if self.aer and not self.transpile:
            print("Running on AER without transpilation")
            return self.qc
        
        t_circ = RotatedSurfaceCode.transpile_circ(self.qc, self.n_qubits)

        if self.aer and self.transpile:
            return t_circ

            
        device_param_lookup = self._get_device_parameters(backend)
            
        res  = sim.run( 
                t_qiskit_circ=t_circ,  
                psi0=self.initial_state, 
                shots=shots, 
                device_param=device_param_lookup,
                nqubit=self.n_qubits,
                bit_flip_bool=bit_flip_bool,
                )

        probs = res["probs"]
        results = res["results"]
        num_clbits = res["num_clbits"]
        mid_counts = res["mid_counts"]
        barrier_statevectors = res["barrier_statevectors"]
            
        if not correction:
            return mid_counts, t_circ, barrier_statevectors
        corrected_counts, data_counts, predictions_x, predictions_z = self.decode_correct_counts(mid_counts)

        # --- Process mid_counts to separate registers ---
        register_size= [self.n_data]+ [self.n_stabilizers]*self.cycles 
        
        processed_counts = self._split_counts(corrected_counts, register_size)
        return processed_counts, data_counts, t_circ, barrier_statevectors, predictions_x, predictions_z

# ========================================================================
#  HELPER METHODS
#  ========================================================================
    
    def _compute_stabilizer_connections(self):
        """
        This method constructs a mapping from each stabilizer (X and Z type)
        to the list of data qubits it interacts with during stabilizer
        measurements, based on the surface-code layout.

        Returns
        -------
        dict
            Dictionary mapping each stabilizer (row_index, global_qubit_index)
            to a list of global indices of its neighboring data qubits.
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
        This method parses a full measurement bitstring produced by the circuit,
        accounts for Qiskit's little-endian classical bit ordering, and separates
        the stabilizer measurement outcomes by type (X or Z) and by stabilizer
        cycle.

        Parameters
        ----------
        bitstring : str
            Classical measurement bitstring returned by Qiskit, containing
            concatenated data and stabilizer register measurements.

        Returns
        -------
        x_syndromes : ndarray of int
            Array of shape (cycles, number_of_X_stabilizers) containing X-type
            stabilizer measurement outcomes for each cycle.

        z_syndromes : ndarray of int
            Array of shape (cycles, number_of_Z_stabilizers) containing Z-type
            stabilizer measurement outcomes for each cycle.
        """

        bits = bitstring[::-1]  # reverse Qiskit order
        x_syndromes = np.zeros((self.cycles, len(self.x_stabilizers)), dtype=int)
        z_syndromes = np.zeros((self.cycles, len(self.z_stabilizers)), dtype=int)
        for c in range(self.cycles):
            for i, stab in enumerate(self.x_stabilizers):
                clbits = self.stabilizer_to_clbits[stab][0]+ c * self.n_stabilizers
                x_syndromes[c, i] = int(bits[clbits]) 
            
            for i, stab in enumerate(self.z_stabilizers):
                clbits = self.stabilizer_to_clbits[stab][0]+ c * self.n_stabilizers
                z_syndromes[c, i] = int(bits[clbits])
        
        return x_syndromes, z_syndromes

    def _split_counts(self, raw_counts: dict, register_sizes: list) -> dict:
        """
        This method takes raw measurement counts with concatenated classical
        bitstrings and partitions each bitstring into substrings corresponding
        to individual classical registers (e.g. data register and stabilizer
        registers per cycle). The resulting keys are formatted with spaces
        separating the register bitstrings.
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

    def _get_device_parameters(self, backend):
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
        return H.tocsr()
    
    def setup_decoder(self):
        """
        Setup the decoder by building parity-check matrices for X and Z stabilizers.
        """
        self.H_X = self._build_parity_check_matrix('X')
        self.H_Z = self._build_parity_check_matrix('Z')

    def _decode_per_cycle(self, syndrome1, syndrome2, which='X'):
        """
        This method computes the detection-event syndrome by taking the bitwise
        XOR of stabilizer measurement outcomes from two consecutive cycles and
        decodes it using minimum-weight perfect matching. The decoding is
        performed separately for X- and Z-type stabilizers using the
        corresponding parity-check matrix.
        """

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
        This method extracts X- and Z-type stabilizer measurement outcomes from
        a concatenated measurement bitstring and performs cycle-by-cycle decoding
        using detection events between consecutive stabilizer rounds. 

        Parameters
        ----------
        bitstring : str or dict
            Concatenated stabilizer measurement bitstring, or a counts dictionary
            containing a single bitstring key.

        Returns
        -------
        prediction_x : ndarray
            Array of shape (cycles - 1, n_data_qubits) containing predicted
            X-type error locations for each pair of consecutive cycles.

        prediction_z : ndarray
            Array of shape (cycles - 1, n_data_qubits) containing predicted
            Z-type error locations for each pair of consecutive cycles.
        """

        if isinstance(bitstring, dict):
            bitstring = list(bitstring.keys())[0]
        
        x_syndromes, z_syndromes = self._extract_stabilizer_measurements(bitstring)
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
    
    def decode_correct_counts(self, mid_counts):
        """
        This method processes mid-circuit measurement counts by decoding the full
        stabilizer syndrome history, inferring data-qubit errors, and applying the
        resulting corrections to the measured data bits. Z-type corrections are
        applied directly to the data-bit outcomes.

        Parameters
        ----------
        mid_counts : dict
            Dictionary mapping concatenated data-and-syndrome bitstrings to
            their occurrence counts.

        Returns
        -------
        corrected_counts : dict
            Measurement counts with data-bit outcomes corrected using the
            decoded error predictions.

        data_counts : dict
            Counts of corrected data-bit outcomes only (syndrome bits removed).

        predictions_x : ndarray
            Predicted X-type error locations inferred from the stabilizer
            syndrome history.

        predictions_z : ndarray
            Predicted Z-type error locations inferred from the stabilizer
            syndrome history.
        """

        corrected_counts = {}
        data_counts = {}
        for bitstring, count in mid_counts.items():
            syndrome_bits = bitstring[self.n_data:]
            data_bits = bitstring[:self.n_data]
            predictions_x, predictions_z = self.decode_full(syndrome_bits)
            final_x_correction = np.bitwise_xor.reduce(predictions_x, axis=0)
            final_z_correction = np.bitwise_xor.reduce(predictions_z, axis=0)

            
            
            for i in range(len(final_x_correction)): 
                
                if final_z_correction[i] == 1:
                    new_bit = '1' if data_bits[i] == '0' else '0'
                    data_bits = data_bits[:i] + new_bit + data_bits[i+1:]
            # Create new key with corrected data
            new_key = data_bits +syndrome_bits
            corrected_counts[new_key] = corrected_counts.get(new_key, 0) + count
            data_counts[data_bits] = data_counts.get(data_bits, 0) + count
        return corrected_counts, data_counts, predictions_x, predictions_z