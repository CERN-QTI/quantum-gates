"""Performs simulations with the Noisy quantum gates approach.
"""
import numpy as np
import copy
import inspect
from typing import List, Tuple
from collections import Counter

from qiskit import QuantumCircuit

from .._gates.gates import Gates, standard_gates
from .._simulation.circuit import EfficientCircuit, BinaryCircuit
from .._simulation.circuit import Circuit, StandardCircuit


class MrAndersonSimulator(object):
    """
    Simulates a noisy quantum circuit by extracting gate instructions from a
    transpiled Qiskit ``QuantumCircuit`` and executing them with a custom backend.

    Shots may be parallelized, though this typically introduces overhead.
    For large workloads, consider parallelizing multiple simulator calls instead
    via ``src.utility.simulation_utility.perform_parallel_simulation``.

    Args:
        gates (Union[Gates, ScaledNoiseGates, NoiseFreeGates]):
            Gate set to be used, including pulse/noise definitions.
        CircuitClass (Union[Circuit, EfficientCircuit]):
            Backend computation class.
        parallel (bool):
            Whether to execute shots in parallel (default: False).

    Note:
        Use ``BinaryCircuit`` for non-linear qubit topologies. When doing so,
        ensure device parameters include all qubits up to the maximum index.

    Example:
        .. code:: python

            sim = MrAndersonSimulator(
                gates=standard_gates,
                CircuitClass=EfficientCircuit,
                parallel=False
            )

            out = sim.run(
                t_qiskit_circ=...,
                psi0=...,
                shots=1000,
                device_param=...,
                nqubit=2
            )

            print(out["probs"])

        Example output structure:

        .. code-block:: text

            {
                "probs": {...},               # final probability distribution
                "results": [...],             # per-shot mid/final measurement data
                "num_clbits": int,            # number of classical bits
                "mid_counts": {...},          # aggregated mid-circuit results
                "statevector_readout": [...]  # saved statevectors (if any)
            }

    Attributes:
        gates: Gate definitions including pulse/noise data.
        CircuitClass: Backend circuit evaluator.
        parallel: Whether shots run in parallel.
    """

    def __init__(self, gates: Gates=standard_gates, CircuitClass=BinaryCircuit, parallel: bool=False):
        self.gates = gates() if inspect.isclass(gates) else gates  # Contains the information about the pulses.
        self.CircuitClass = CircuitClass
        self.parallel = parallel
        
    
    def run(self,
        t_qiskit_circ,
        psi0: np.array,
        shots: int,
        device_param: dict,
        nqubit: int,
        qubit_layout=None,
        bit_flip_bool=True,) -> dict:
        """
        Execute a noisy simulation of a transpiled Qiskit circuit on the given
        device model, starting from state ``psi0`` and running ``shots`` realizations.

        Args:
            t_qiskit_circ (QuantumCircuit): Transpiled circuit to simulate.
            psi0 (np.ndarray): Initial state vector.
            shots (int): Number of Monte-Carlo realizations.
            device_param (dict): Noise/device configuration.
            nqubit (int): Number of qubits implied by ``psi0``.
            qubit_layout: Deprecated; must be ``None``.
            bit_flip_bool (bool): Apply bit-flip correction during measurement.

        Returns:
            dict containing:
                • **"probs"** – final measurement probabilities (big-endian).
                • **"results"** – per-shot mid/final measurement data.
                • **"num_clbits"** – number of classical bits in the circuit.
                • **"mid_counts"** – aggregated mid-circuit measurement bitstrings.
                • **"statevector_readout"** – measured statevectors, from mid measure.
        """
        if qubit_layout is not None:
            raise NotImplementedError("qubit_layout argument is deprecated; please use transpiled circuits instead.")
        
        # Process layout circuit
        used_logicals, q_meas_list, n_qubit_used = self._process_layout(t_qiskit_circ)
        
        # Get total classical bits (for Aer-style output)
        num_clbits = len(t_qiskit_circ.clbits)
        
        # Infer width from psi0 (must be power of two)
        nqubit = int(round(np.log2(psi0.size)))
        if 2**nqubit != psi0.size:
            raise ValueError(f"psi0 length {psi0.size} is not a power of two.")

        # strong validation against the FULL layout (not the used subset)
        self._validate_input_of_run(t_qiskit_circ, psi0, shots, device_param, nqubit)
        
        # optional: warn if there are idle qubits (kept in simulation; correct but may cost perf)
        if n_qubit_used < nqubit and hasattr(self, "_logger"):
            self._logger.warning(
                f"{nqubit - n_qubit_used} qubit(s) are idle (no operations). They will be simulated as idling qubits."
            )

        # Count rz gates, construct swap lookup, generate data (representation of circuit compatible with simulation)
        # preprocess circuit with the ACTIVE layout; nqubit is the full simulated width
        n_rz, data, data_measure = self._preprocess_circuit(
            t_qiskit_circ, n_qubit_used, used_logicals
        )

        # Read data and apply Noisy Quantum gates for many shots to get preliminary probabilities
        #  perform simulation (psi0 spans full width nqubit; active_layout routes gates)
        probs, all_results = self._perform_simulation(
            shots, data, n_rz, nqubit, device_param, psi0, data_measure, bit_flip_bool
        )

        # Normalize final probabilities
        reordered_arr = np.asarray(probs, dtype=float)
        total_prob = reordered_arr.sum()
        if total_prob <= 0.0:
            raise ValueError(f"Unphysical probability vector: sum={total_prob}.")
        final_arr = reordered_arr / total_prob
        
        # this keeps things correct whether 'probs' is over all qubits or only the used subset
        prob_width = int(round(np.log2(final_arr.size)))
        if 2**prob_width != final_arr.size:
            raise ValueError("Internal error: probability vector length is not a power of two.")

        # produce final counts-style readout
        counts_ng = self._measurament(
            prob=final_arr,
            q_meas_list=q_meas_list,
            n_qubit=prob_width,       # if your simulator always returns full width, this equals nqubit
        )
        
        # --- Build mid-circuit bitstrings with chronological processing ---
        combined_mid_strings = []
        statevector_readout = []

        for shot in all_results:
            # initialize all clbits to '0' (Aer default for unused)
            clbit_values = ['0'] * num_clbits

            # Sort events by step in ascending order (chronological)
            sorted_events = sorted(shot["mid"], key=lambda x: x["step"])
            
            # Process events in chronological order (later measurements overwrite earlier ones)
            for event in sorted_events:
                for c, val in zip(event["clbits"], event["outcome"]):
                    clbit_values[c] = str(val)
                
                statevector_readout.append(event["statevector"])

            # sort descending by clbit index (Aer display order)
            bitstring = ''.join(clbit_values[::-1])
            combined_mid_strings.append(bitstring)

        # --- Count occurrences ---
        mid_counts = Counter(combined_mid_strings)

        return {
            "probs": counts_ng,
            "results": all_results, # mid-circuit measurement results
            "num_clbits": num_clbits, # number of classical bits in circuit
            "mid_counts": dict(mid_counts), # mid-circuit measurement counts
            "statevector_readout": statevector_readout, # saved statevectors if any
        }
    
    
    def _process_layout(self, circ: QuantumCircuit) -> Tuple[List[int], List[Tuple[int, int]], int]:
        """
        Returns:
            used_q: list of physical qubit indices actually used (unsorted is fine)
            measure_qc: list of (qubit_index, flat_classical_bit_index)
            n_qubit: number of used qubits
        """
        used = set()
        measure_qc: List[Tuple[int, int]] = []

        for instr in circ.data:
            name = instr.operation.name
            if name in {"delay", "barrier"}:
                continue

            # record qubits touched by this instruction (any arity)
            for qb in instr.qubits:
                used.add(qb._index)

            # record measurements with flat classical index across all cregs
            if name == "measure":
                q = instr.qubits[0]._index
                c_flat = circ.find_bit(instr.clbits[0]).index
                measure_qc.append((q, c_flat))

        used_logicals = list(used)   # unsorted; sort if you want determinism
        return used_logicals, measure_qc, len(used_logicals)
 
    
    def _validate_input_of_run(self, t_qiskit_circ, psi0, shots, device_param, nqubit):
        """ Performs sanity checks on the input of the run() method. Raises an Exception if any mistakes are found. """

        # Check types
        if not isinstance(t_qiskit_circ, QuantumCircuit):
            raise ValueError(f"Expected argument t_qiskit_circ to be of type QuantumCircuit, but found {type(t_qiskit_circ)}.")
        
        if not isinstance(shots, int):
            raise ValueError(f"Expected argument shots to be of type int, but found {type(shots)}.")
        if not isinstance(device_param, dict):
            raise ValueError(f"Expected argument device_param to be of type dict, but found {type(device_param)}.")
        if not isinstance(nqubit, int):
            raise ValueError(f"Expected argument nqubit to be of type int, but found {type(nqubit)}.")

        # Check values
        if shots < 1:
            raise ValueError(f"Expected positive number of shots but found {shots}.")

        # Cross check number of qubits
        # psi0 must match the number of **used** qubits
        if psi0.shape != (2**nqubit,):
            raise ValueError(f"psi0 shape {psi0.shape} is incompatible with nqubit={nqubit} (expected {(2**nqubit,)}).")

        return

    # helper for pretty-printing preprocessed data (for debugging)
    def _pretty_print_data(self, data):
        """Print human-readable view of preprocessed circuit data."""
        for idx, (chunk, flag) in enumerate(data):
            if flag == 0:
                ops_str = " , ".join(
                    f"{op.operation.name}[{', '.join(str(q._index) for q in op.qubits)}]"
                    for op in chunk
                )
                print(f"Chunk {idx}: {ops_str}")
            else:
                op = chunk
                # handle mid_measurement tuple
                if isinstance(op, tuple) and op[0] == "mid_measurement":
                    meas_op = op[1]
                    q_idx = meas_op["q_idx"]
                    c_idx = meas_op["c_idx"]
                    print(f"Fancy {idx}: mid_measurement qubits={q_idx} clbits={c_idx}")
                    
                elif isinstance(op, tuple) and op[0] == "reset_qubits":
                    meas_op = op[1]
                    q_idx = meas_op["q_idx"]
                    print(f"Fancy {idx}: reset_qubits qubits={q_idx}")
                    
                else:
                    # normal fancy gate (Instruction)
                    print(
                        f"Fancy {idx}: {op.operation.name} "
                        f"qubits={[q._index for q in op.qubits]} "
                        f"clbits={[c._index for c in op.clbits]}"
                    )

    
    def _preprocess_circuit(self, t_qiskit_circ, n_qubit_used: int, used_logicals) -> tuple:
        """ Preprocess QuantumCircuit.data:
        - Count number of RZ gates (n_rz).
        - Track swaps (swap_detector). TODO: currently not used.
        - Structure data into chunks split around 'fancy-gates':
        - bring the circuit in a format (data) that is compatible with the rest of the simulation.
        data = [ [ops until first fancy], fancy_gate, [ops until next fancy], fancy_gate, ... ]
        """
        # Initialize
        n_rz = 0
        data = []
        data_measure = []
        current_chunk = []

        raw_data = t_qiskit_circ.data
    
        #  Build lookup: Clbit → register name
        used_set = set(used_logicals)

        # Optional: clbit -> (register_name, local_idx) (keep if you need it later) 
        clbit_to_reg = {}
        for reg in t_qiskit_circ.cregs:
            for local_i, bit in enumerate(reg):
                clbit_to_reg[bit] = (reg.name, local_i)

        # loop through the raw data, circuit representation from Qiskit
        # each op is a CircuitInstruction, each i is the compilation step
        for i, op in enumerate(raw_data):
            op_name = op.operation.name
            
            q_idx = [q._index for q in op.qubits]
            if any(q not in used_set for q in q_idx): continue
            c_idx = [t_qiskit_circ.find_bit(c).index for c in op.clbits]

            # ---- Fancy Gates Processing ----
            ## fancy_gates = {"reset_qubits","mid_measurement", "statevector_readout", "if_else"}
            # ---------------------- MEASUREMENT ----------------------
            if op_name == "measure":
                # Is this a mid measurement? (any quantum op after it)
                remaining_ops = raw_data[i+1:]
                has_future_quantum = any(
                    (future_op.operation.name not in {"measure", "barrier", "delay"}) 
                    for future_op in remaining_ops
                )
                
                if has_future_quantum:
                    # mid-circuit → treat as fancy gate
                    if current_chunk:
                        data.append((current_chunk, 0))
                        current_chunk = []
                    # instead of mutating, just store a tuple with a tag
                    data.append((
                        ("mid_measurement", {
                            "op": op,       # keep original CircuitInstruction for debugging if you like
                            "q_idx": q_idx, # flat qubit indices (aligned with op.qubits)
                            "c_idx": c_idx, # flat clbit indices (aligned with op.clbits)
                        }),
                        1
                    ))

                else:
                    # final measurement → store for later
                    q = op.qubits[0]._index
                    c = op.clbits[0]

                    c_reg = clbit_to_reg.get(op.clbits[0], "unknown")
                    c_idx = op.clbits[0]._index
                    data_measure.append((q, (c_reg, c_idx)))

            # ---------------------- RESET ----------------------
            elif op_name == "reset":
                
                if current_chunk:
                    data.append((current_chunk, 0))
                    current_chunk = []

                # Expand multi-qubit reset into separate entries.
                # Use the PHYSICAL/flat index from Qiskit (._index) for consistency.
                data.append((("reset_qubits", {"op": op, "q_idx": q_idx}), 1))
            
           # ---- Not yet implemented fancy gates ----
            elif op_name in ("if_else", "if_test", "control_flow", "switch_case"):
                raise NotImplementedError("if condition found in circuit, which is not implemented yet.")

            elif op_name in ("statevector_readout", "save_statevector", "save_state"):
                raise NotImplementedError("saving statevector(s) operation found in circuit, which is not implemented yet.")

            elif op_name in ("while_loop", "for_loop", "loop"):
                raise NotImplementedError("loop operation found in circuit, which is not implemented yet.")
                
            # ---------------------- BARRIER / DELAY ----------------------
            elif op_name == "barrier" or op_name == "delay":
                continue

            # non -fancy gates appending to current chunk
            elif op_name == "rz":
                n_rz += 1  # track rz count
                current_chunk.append(op)  # probably you also want to keep it
            else:
                # normal operation (only if in layout)
                current_chunk.append(op)

        # flush final chunk if non-empty
        if current_chunk:
            data.append((current_chunk,0))

        #self._pretty_print_data(data) # debug print
       
        return n_rz, data, data_measure

    
    def _perform_simulation(self,
                            shots: int,
                            data: list,
                            n_rz: int,
                            nqubit: int,
                            device_param: dict,
                            psi0: np.array,
                            data_measure: list,
                            bit_flip_bool: bool,) -> np.array:
        """ Performs the simulation shots many times and returns the resulting probability distribution.
        """
        # Setup results
        r_sum = np.zeros(2**nqubit)
        r_square_sum = np.zeros(2**nqubit)

        # Constants
        # depth is the number of gates / ops in the circuit (excluding rz's) +1
        depth = len(data) - n_rz + 1

        # Create list of args
        arg_list = [
            {
                "data": copy.deepcopy(data),
                "data_measure": copy.deepcopy(data_measure),   # <--- added
                "circ": self.CircuitClass(nqubit, depth, copy.deepcopy(self.gates)),
                "device_param": copy.deepcopy(device_param),
                "psi0": copy.deepcopy(psi0),
                "bit_flip_bool": bit_flip_bool,
                "num_qubit": nqubit,
            } for i in range(shots)
        ]
        
        all_results = []  # Store mid-circuit measurement results if needed

        # Perform computation parallel or sequentual
        if self.parallel:
            import multiprocessing

            # Configure pool
            cpu_count = multiprocessing.cpu_count()
            print(f"Our CPU count is {cpu_count}")

            n_processes = max(int(0.8 * cpu_count), 2)
            print(f"Use 80% of the cores, so {n_processes} processes.")

            chunksize = max(1, int(shots / n_processes) + (1 if shots % n_processes > 0 else 0))
            print(f"As we perform {shots} shots, we use a chunksize of {chunksize}.")

            # Compute
            p = multiprocessing.Pool(n_processes)
            for results_mid_measure, shot_result, final_outcomes in p.imap_unordered(func=_single_shot, iterable=arg_list, chunksize=chunksize):
                # Add shot
                r_sum += shot_result
                r_square_sum += np.square(shot_result)

                all_results.append({
                    "mid": results_mid_measure,
                    "final": final_outcomes
                })
            # Shut down pool
            p.close()
            p.join()

        else:
            for arg in arg_list:
                # Compute
                results_mid_measure, shot_result, final_outcomes = _single_shot(arg)

                r_sum += shot_result
                r_square_sum += np.square(shot_result)

                all_results.append({
                    "mid": results_mid_measure,
                    "final": final_outcomes
                })

        # Calculate result
        r_mean = r_sum / shots
        r_var = r_square_sum / shots - np.square(r_mean)
        
        return r_mean, all_results
    
    
    def _measurament(self, prob : np.array, q_meas_list : list, n_qubit: int) -> dict: 
        """This function take in input the measured qubits and the classical bits to store the information regarding also the swapping and give in ouput the probabilities of the possible outcomes.

        Args:
            prob (np.array): probabilities after the application of all the gates 
            q_meas_list (list): list of tuples, where each tuples indicate the qubit measured and the corresponding classic bit 
            n_qubit (int): total number of qubits 
            # qubit_layout(list): qubit layout after the transpilation rm

        Returns:
            dict: the keys are the possible states and the value the probabilities of measurement each state.
        """
        # create the vector with the bit strings
        binary_vector = np.array([format(i, f'0{n_qubit}b') for i in np.arange(2**n_qubit)], dtype=str)

        # Measured qubit indices directly correspond to transpiled circuit qubits
        q_meas = [q for q, _ in q_meas_list]

        # create a list of strings with the only string measured
        res = []

        for binary_str in binary_vector:
            temp = [binary_str[i] for i in q_meas]
            res.append(''.join(temp)) 
        
        # create a dictionary to store the probabilities of measure one of the possible state
        sums = {}
        for value, bit_string in zip(prob, res):
            if bit_string not in sums:
                sums[bit_string] = 0.0
            sums[bit_string] += value

        return sums


def _apply_gates_on_circuit(
        data: list,
        circ: Circuit or StandardCircuit or EfficientCircuit or BinaryCircuit, # type: ignore
        device_param: dict,
    ) -> None:
    """ Applies the operations specified in data on the circuit.

    The constants regarding the device and noise are passed in device_param.

    Args:
        data (list): List of circuit instructions as preprocessed by the simulator.
        circ (Union[Circuit, StandardCircuit, EfficientCircuit]): Performs the computations.
        device_param (dict): Lookup for the noise information.
        qubit_layout(list): Layout of the used qubit
    """

    # Unpack dict
    T1, T2, p, rout, p_int, t_int, tm, dt = (
        device_param["T1"],
        device_param["T2"],
        device_param["p"],
        device_param["rout"],
        device_param["p_int"],
        device_param["t_int"],
        device_param["tm"],
        device_param["dt"][0]
    )
    nqubit = circ.nqubit

    if isinstance(circ, BinaryCircuit): # if the class of circuit is Binary class the application is different
        # Apply gates
        for j in range(len(data)):
            if data[j].operation.name == 'rz':
                theta = float(data[j].operation.params[0])
                q_r = data[j].qubits[0]._index #real qubit
                circ.Rz(q_r, theta)

            if data[j].operation.name == 'sx':
                q_r = data[j].qubits[0]._index #real qubit
                circ.SX(q_r, p[q_r], T1[q_r], T2[q_r])
                        
            if data[j].operation.name == 'x':
                q_r = data[j].qubits[0]._index #real qubit
                circ.X(q_r, p[q_r], T1[q_r], T2[q_r])
                        
            if data[j].operation.name == 'ecr':
                q_ctr_r = data[j].qubits[0]._index # index control real qubit
                q_trg_r = data[j].qubits[1]._index # index target real qubit
                
                circ.ECR(q_ctr_r, q_trg_r, t_int[q_ctr_r][q_trg_r], p_int[q_ctr_r][q_trg_r], p[q_ctr_r], p[q_trg_r], T1[q_ctr_r], T2[q_ctr_r], T1[q_trg_r], T2[q_trg_r])

            if data[j].operation.name == 'cx':
                q_ctr_r = data[j].qubits[0]._index # index control real qubit
                q_trg_r = data[j].qubits[1]._index # index target real qubit
                
                circ.CNOT(q_ctr_r, q_trg_r, t_int[q_ctr_r][q_trg_r], p_int[q_ctr_r][q_trg_r], p[q_ctr_r], p[q_trg_r], T1[q_ctr_r], T2[q_ctr_r], T1[q_trg_r], T2[q_trg_r])
            
            if data[j].operation.name == 'delay':
                q_r = data[j].qubits[0]._index #real qubit
                time = data[j].operation.duration * dt
                circ.relaxation(q_r, time, T1[q_r], T2[q_r])              
        return
    
    else:
        # Apply gates
        for j in range(len(data)):

            if data[j].operation.name == 'rz':
                theta = float(data[j].operation.params[0])
                q = data[j].qubits[0]._index
                circ.Rz(q, theta)

            if data[j].operation.name == 'sx':
                q = data[j].qubits[0]._index
                for k in range(nqubit):
                    if k == q:
                        circ.SX(k, p[k], T1[k], T2[q])
                    else:
                        circ.I(k)

            if data[j].operation.name == 'x':
                q = data[j].qubits[0]._index
                for k in range(nqubit):
                    if k == q:
                        circ.X(k, p[k], T1[k], T2[q])
                    else:
                        circ.I(k)

            if data[j].operation.name == 'ecr':
                q_ctr = data[j].qubits[0]._index # index control qubit
                q_trg = data[j].qubits[1]._index # index target qubit
                for k in range(nqubit):
                    if k == q_ctr:
                        circ.ECR(k, q_trg, t_int[k][q_trg], p_int[k][q_trg], p[k], p[q_trg], T1[k], T2[k], T1[q_trg], T2[q_trg])
                    elif k == q_trg:
                        pass
                    else:
                        circ.I(k)

            if data[j].operation.name == 'cx':
                q_ctr = data[j].qubits[0]._index # index control qubit
                q_trg = data[j].qubits[1]._index # index target qubit
                for k in range(nqubit):
                    if k == q_ctr:
                        circ.CNOT(k, q_trg, t_int[k][q_trg], p_int[k][q_trg], p[k], p[q_trg], T1[k], T2[k], T1[q_trg], T2[q_trg])
                    elif k == q_trg:
                        pass
                    else:
                        circ.I(k)
            
            if data[j].operation.name == 'delay':
                q = data[j].qubits[0]._index
                time = data[j].operation.duration * dt
                for k in range(nqubit):
                    if k == q:
                        circ.relaxation(k, time, T1[k], T2[k])
                    else:
                        circ.I(k)

        return


def _single_shot(args: dict) -> np.array:
    circ = args["circ"]
    data = args["data"]
    data_measure = args.get("data_measure", [])  # <--- added
    device_param = args["device_param"]
    psi0 = args["psi0"]
    
    bit_flip_bool =  args["bit_flip_bool"]
    psi = psi0
    mid_results = []  # mid results
    saved_statevectors = []  # Store saved statevectors if needed
    
    circ.reset(phase_reset=True)  # reset internal state before starting
    
    for idx, (d, flag) in enumerate(data):
        if flag == 0:       
            _apply_gates_on_circuit(d, circ, device_param)
            psi = circ.statevector(psi)
            circ.reset(phase_reset=False)  # reset internal state for next chunk

        elif flag == 1:
            
            if isinstance(d, tuple) and d[0] == "mid_measurement":
                op = d[1]
                qubits  = op["q_idx"]
                clbits  = op["c_idx"]
                # Perform the mid-circuit measurement
                psi, outcome = circ.mid_measurement(psi, device_param, add_bitflip=bit_flip_bool, qubit_list=qubits, cbit_list = clbits)
                
                outcome1 = [int(x) for x in np.atleast_1d(outcome)]
                assert len(outcome1) == len(clbits), "Outcome/clbits length mismatch"
                
                # normalize just in case
                norm = np.linalg.norm(psi)
                if norm > 0:
                    psi /= norm

                # record debug info
                mid_results.append({
                    "step": idx,
                    "qubits": qubits,
                    "clbits": clbits,
                    "outcome": outcome,
                    "statevector": psi.copy(),
                })

            elif isinstance(d, tuple) and d[0] == "reset_qubits":
                op = d[1]
                qubits = op["q_idx"]   # already flat (transpiled indices)

                # --- Collapse without classical writes ---
                psi, outcomes = circ.mid_measurement(
                    psi0=psi,
                    device_param=device_param,
                    add_bitflip=bit_flip_bool,
                    qubit_list=qubits,
                    cbit_list=None,
                )

                # Extract noise params *once*
                T1, T2, p = device_param["T1"], device_param["T2"], device_param["p"]

                # --- Apply X/I layer ---
                touched = set()
                for q, res in zip(qubits, outcomes):
                    touched.add(q)
                    if res == 1:
                        circ.X(i=q, p=p[q], T1=T1[q], T2=T2[q])
                    else:
                        circ.I(i=q)

                for k in range(circ.nqubit):
                    if k not in touched:
                        circ.I(i=k)   # maintain full layer (no gaps)

                # --- Apply layer to state and reset builder ---
                psi_before = psi.copy()
                psi = circ.statevector(psi)
                circ.reset(phase_reset=False)  # reset internal state for next chunk

            else:
                op_name = d.operation.name
                if op_name == "statevector_readout":
                    raise NotImplementedError("statevector_readout not implemented yet")

                elif op_name == "if_else":
                    raise NotImplementedError("if_else not implemented yet")

                else:
                    raise ValueError(f"Unknown / not yet implemented gate: {op_name}")
    
    # mid_measurement results in Qiskit clbit order
    qiskit_order_mid_results = mid_results[::-1]  

    # --- Final Measurements ---
    # Born rule → probability distribution
    shot_result = np.square(np.abs(psi))

    # Final outcomes from last measurement
    final_outcomes = None
    if data_measure:
        probs = np.square(np.abs(psi))
        if probs.sum() == 0:
            raise ValueError("Statevector collapsed to zero norm after circuit execution.")
        probs = probs / probs.sum()   # normalize before sampling

        outcome_index = np.random.choice(len(probs), p=probs)

        bitstring = format(outcome_index, f"0{circ.nqubit}b")
        final_outcomes = {}
        for q, (c_reg, c_idx) in data_measure:
            final_outcomes[(c_reg, c_idx)] = int(bitstring[-(q+1)])  # big-endian bit order
    
    return qiskit_order_mid_results, shot_result, final_outcomes
