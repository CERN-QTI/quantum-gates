from qiskit import QuantumCircuit,QuantumRegister, ClassicalRegister, transpile

from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates, noise_free_gates
from quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from quantum_gates.utilities import DeviceParameters
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.controlflow import ControlFlowOp 


from qiskit_ibm_runtime.fake_provider import FakeBrisbane




class SurfaceCode:
    def __init__(self, distance=3, cycles = 1):
        self.d = distance
        self.cycles = cycles
        self.n_data = distance**2 + (distance - 1)**2
        self.n_stabilizers = 2*(distance-1)* distance
        self.n_clbits = self.n_stabilizers * cycles + self.n_data  # include data qubit measurements at end
        self.n_qubits = self.n_data + self.n_stabilizers
        self.width = 2*distance-1
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

        self.stabilizers = [i for i in range((self.n_qubits)) if i % 2 == 1]
        self.data_qubits, self.x_stabilizers, self.z_stabilizers = self._get_surface_code_layout()

        # maps stabilizer index -> list of classical bit indices (one per cycle)
        self.stabilizer_to_clbits = {
            stab: [cycle * self.n_stabilizers + i for cycle in range(self.cycles)]
            for i, stab in enumerate(self.stabilizers)
        }

        self._build_stabilizer_layer()
        self.stabilizer_connections = self._compute_stabilizer_connections()

        
    def _get_surface_code_layout(self):
        """
        Return indices of X stabilizers, Z stabilizers, and data qubits
        for a planar surface code of given distance.
        Layout follows the checkerboard pattern on a (2d-1) x (2d-1) grid.
        """
        # create coordinate grid
        rows, cols = np.meshgrid(np.arange(self.width), np.arange(self.width), indexing="ij")

        # checkerboard masks
        mask_data = (rows + cols) % 2 == 0
        mask_x    = (rows % 2 == 0) & (cols % 2 == 1)
        mask_z    = (rows % 2 == 1) & (cols % 2 == 0)

        # collect indices
        data_qubits   = [self._qubit_index(r, c) for r, c in zip(rows[mask_data], cols[mask_data])]
        x_stabilizers = [self._qubit_index(r, c) for r, c in zip(rows[mask_x], cols[mask_x])]
        z_stabilizers = [self._qubit_index(r, c) for r, c in zip(rows[mask_z], cols[mask_z])]

        return data_qubits, x_stabilizers, z_stabilizers

    
    def _qubit_index(self, r, c):
            return r * self.width + c
    
    
    def _get_neighbors(self, r, c):
            """Return indices of data qubits adjacent to stabilizer at (r,c)."""
            neighbors = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < self.width and 0 <= cc < self.width:
                    neighbors.append(self._qubit_index(rr, cc))
            return neighbors

    
    def _build_stabilizer_layer(self):
        """Build one stabilizer measurement cycle for a distance-n surface code."""

        # repeat the stabilizer-measurement process for all cycles
        for cycle in range(self.cycles):
            # --- Reset all stabilizers at the beginning of the cycle ---
            for anc in self.x_stabilizers + self.z_stabilizers:
                self.qc.reset(anc)
                
            # --- X stabilizers ---
            for anc in self.x_stabilizers:
                r, c = divmod(anc, self.width) # converts a 1D index into a (row, col) pair
                
                self.qc.h(anc)  # prepare X stabilizer in |+>
                self.qc.barrier()

                # entangle with data qubits (ancilla is control)
                for nb in self._get_neighbors(r, c):
                    self.qc.cx(anc, nb)

                self.qc.barrier()
                self.qc.h(anc)  # rotate back before measurement

            # --- Z stabilizers ---
            for anc in self.z_stabilizers:
                r, c = divmod(anc, self.width)
                # entangle with data qubits (data is control, ancilla target)
                for nb in self._get_neighbors(r, c):
                    self.qc.cx(nb, anc)

                self.qc.barrier()
            
            # --- Measure all stabilizers for this cycle --- 
            
            stab_list = sorted(self.x_stabilizers + self.z_stabilizers)
            for i, anc in enumerate(stab_list):
                self.qc.measure(anc, self.cycle_cregs[cycle][i])


            self.qc.barrier(label="save_statevector")
         # --- Final measurement of data qubits in Z basis ---
        for i, data_qubit in enumerate(self.data_qubits):
            self.qc.measure(data_qubit, self.c_data[i])
        self.qc.barrier()

        self.qc.x(range(0,self.n_qubits))  # introduce bit-flip errors on all data qubits
        

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

    def _extract_stabilizer_measurements(self, bitstring):
        """
        Extract measurement results for all stabilizers across all cycles.
        Returns separate arrays for X and Z stabilizers.
        """
        bits = bitstring[::-1]  # reverse Qiskit order
        
        x_syndromes = np.zeros((self.cycles, len(self.x_stabilizers)), dtype=int)
        z_syndromes = np.zeros((self.cycles, len(self.z_stabilizers)), dtype=int)
        
        for i, stab in enumerate(self.x_stabilizers):
            clbits = self.stabilizer_to_clbits[stab]
            x_syndromes[:, i] = [int(bits[cb]) for cb in clbits]
        
        for i, stab in enumerate(self.z_stabilizers):
            clbits = self.stabilizer_to_clbits[stab]
            z_syndromes[:, i] = [int(bits[cb]) for cb in clbits]
        
        return x_syndromes, z_syndromes
    

    def _compute_stabilizer_connections(self):
        """
        Compute which data qubits each stabilizer measures.
        Returns dict: stabilizer_index -> list of data qubit indices
        """
        connections = {}
        
        for stab in self.x_stabilizers + self.z_stabilizers:
            r, c = divmod(stab, self.width)
            neighbors = self._get_neighbors(r, c)
            # Filter to only include data qubits
            data_neighbors = [nb for nb in neighbors if nb in self.data_qubits]
            connections[stab] = data_neighbors
            
        
        return connections


    def run_surfacecode(self, noise, shots = 1):
        """Run the surface code circuit on the specified backend with noise model."""
        #backend = self._create_backend()
        backend = FakeBrisbane()
        if noise:
            set_gate = standard_gates
            bit_flip_bool = True
        else:
            set_gate = noise_free_gates
            bit_flip_bool = False

        sim = MrAndersonSimulator(gates=set_gate, CircuitClass=EfficientCircuit)
        
        needs_controlflow = any(isinstance(op.operation, ControlFlowOp) for op in self.qc.data)

        t_circ = transpile(
            self.qc,
            backend,
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

        #  Load via YOUR class and save JSON next to the script
        device_param = DeviceParameters(list(range(nqubit_actual)))
        device_param.load_from_backend(backend)
        device_param_lookup = device_param.__dict__()

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
        


        register_size= [self.n_data]+ [self.n_stabilizers]*self.cycles 
        print("Register sizes for splitting:", register_size)
        return self._split_counts(mid_counts, register_size), t_circ
    
    
   
    def analyze_results(self, counts):
        """
        Analyze measurement results using stabilizer-to-classical-bit mapping.
        Returns a dict: stabilizer → {bitstring_pattern: frequency}.
        """
        results = {stab: {} for stab in (self.x_stabilizers + self.z_stabilizers)}

        for bitstring, count in counts.items():
            # Use the helper function
            x_syndromes, z_syndromes = self._extract_stabilizer_measurements(bitstring)
            
            # Process X stabilizers
            for i, stab in enumerate(self.x_stabilizers):
                # Get measurements for this stabilizer across all cycles
                bits_for_stab = tuple(x_syndromes[:, i])
                results[stab][bits_for_stab] = results[stab].get(bits_for_stab, 0) + count
            
            # Process Z stabilizers
            for i, stab in enumerate(self.z_stabilizers):
                bits_for_stab = tuple(z_syndromes[:, i])
                results[stab][bits_for_stab] = results[stab].get(bits_for_stab, 0) + count

        # pretty string output
        pretty = {
            stab: {','.join(map(str, k)): v for k, v in res.items()} 
            for stab, res in results.items()
        }

        return pretty

    def plot_single_shot(self, bitstring, shot_idx=None):
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

    

    
