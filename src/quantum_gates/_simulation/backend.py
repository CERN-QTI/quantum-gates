"""Perform the classical computation of the quantum circuits.

The Backend classes evaluate the noisy quantum circuits as tensor contractions. By finding more efficient
contraction ordering, we can optimize the time and space complexity of the algorithm.

The scaling for the QFT circuit with a trivial implementation is:
- Time complexity: O((2**n)**3 * n**2)
- Space complexity: O((2**n)**2)

We improved this in EfficientBackend to
- Time complexity: O((2**n) * n**2)
- Space complexity: O(2**n)
This means we achieved a huge speedup in both time and space complexity, and we are able to simulate 21+ qubits.
"""

import numpy as np
import functools as ft
import opt_einsum as oe
import copy
import string
from scipy.sparse import coo_matrix

from .._utility.circ_optimizer import Optimizer


class StandardBackend(object):
    """Evaluates the circuits represented as tensor contractions in an trivial manner for the speed of the computations.

    Args:
        nqubit (int): Number of qubits.

    Note:
        The StandardBackend iteratively builds the matrices and directly applies them to the statevector. As the memory
        requirements for the matrix grow as O((2\ :sup:`n)`\ 2), this approach only scales up to 13 qubits on a normal
        machine.

    Attributes:
        nqubit (int): Number of qubits.
    """

    def __init__(self, nqubit):
        self.nqubit = nqubit

    def statevector(self, mp_list: list[list], psi0: np.array) -> np.array:
        """Propagates a statevector psi0 with a matrix product represented as list.

        Takes a list of matrix products, each represented as list. Matrix products with lower index are earlier in
        the circuit / are more on the left in the circuit diagram.

        Args:
            mp_list (list[list]): List of matrix products, each represented lists of matrices.
            psi0 (np.array): The statevector.

        Returns:
            The propagated statevector.
        """
        depth = len(mp_list)
        # Case: No gates have been applied
        if depth == 0:
            return np.eye(2**self.nqubit)
        # Case: Gates have been applied
        mp_array = np.array(copy.deepcopy(mp_list), dtype=object)
        propagator = ft.reduce(np.kron, mp_array[0,:])
        for i in range(1, depth):
            propagator = ft.reduce(np.kron, mp_array[i,:]) @ propagator
        return propagator @ psi0

class EfficientBackend(object):
    """Evaluates the quantum circuit represented as list of list of matrices with efficient tensor contractions.

    The EfficientBackend is optimized for general circuits and offers a significant speedup in the higher qubit regime,
    scaling to 20+ qubits.

    Args:
        nqubit (int): Number of qubits in the circuit.
        min_chunk_size (int): The matrices are grouped in chunks of at least this size, we recommend a value of 3.
        optimal_chunk_size (int): The backend aims at achieving an optimal chunk size of this value, normally 4.

    Note:
        Always use this version for the backend for simulating many qubits, it is way faster.

    Example:
        .. code:: python

            from quantum_gates.backend import EfficientBackend

            backend = EfficientBackend(nqubit=2)

            H, CNOT, identity = ...
            mp_list = [[H, np.eye(2)], [CNOT]]

            psi0 = np.array([1, 0, 0, 0])
            psi1 = backend.statevector(mp_list, psi0)  # Gives [1, 0, 0, 1] / sqrt(2)

    Attributes:
        nqubit (int): Number of qubits in the circuit.
        min_chunk_size (int): The matrices are grouped in chunks of at least this size, we recommend a value of 3.
        optimal_chunk_size (int): The backend aims at achieving an optimal chunk size of this value, normally 4.
    """

    def __init__(self, nqubit: int, min_chunk_size: int=3, optimal_chunk_size: int=4):
        self.nqubit = nqubit
        self.min_chunk_size = min_chunk_size
        self.optimal_chunk_size = optimal_chunk_size

    def statevector(self, mp_list: list, psi0: np.array) -> np.array:
        """Propagates a statevector based on a list of matrix products.

        Args:
             mp_list (list[list]): List of list that contain numpy arrays.
             psi0 (np.array): Statevector to be propagated.

        Returns:
            The propagated statevector.
        """
        assert len(mp_list) > 0, f"Expected non empty matrix product list, but found {mp_list}."
        psi1 = copy.deepcopy(psi0)

        # Few qubit regime -> No split
        if self.nqubit < 4:
            return self._statevector_low_qubit_regime(mp_list, psi1)

        # High qubit regime -> Many splits
        elif self.nqubit >= 2 * self.optimal_chunk_size:
            return self._statevector_high_qubit_regime(mp_list, psi1)

        # Middle regime -> Single split
        else:
            return self._statevector_medium_qubit_regime(mp_list, psi1)

    def _statevector_low_qubit_regime(self, mp_list: list, psi: np.array):
        """Propagator for the low qubit regime.

        In the low qubit regime (nqubit < 4), we just generate the expanded matrix product and apply it to psi.
        """
        for mp in mp_list:
            psi = ft.reduce(np.kron, mp) @ psi
        return psi

    def _statevector_medium_qubit_regime(self, mp_list: list, psi: np.array):
        """Propagator for the medium qubit regime.

        In the medium regime (4 <= nqubit < 8), we just generate the expanded matrix product and apply it to psi.
        """
        for mp in mp_list:
            split_index = self.nqubit // 2
            a1 = ft.reduce(np.kron, mp[:split_index])
            a2 = ft.reduce(np.kron, mp[split_index:])
            psi = self._opt_einsum_many_matrices(mp=[a1, a2], psi=psi)
        return psi

    def _statevector_high_qubit_regime(self, mp_list: list, psi: np.array):
        """Propagator for the high qubit regime.

        In the high regime (8 <= nqubit), we split the matrix product into chunks of more or less equal size.
        """

        for mp in mp_list:
            a_list_raw = self._chunk_list(mp, self.min_chunk_size, self.optimal_chunk_size)
            a_list = [ft.reduce(np.kron, chunk[:]) for chunk in a_list_raw]
            psi = self._opt_einsum_many_matrices(a_list, psi)
        return psi

    def _opt_einsum_many_matrices(self, mp: list, psi: np.array) -> np.array:
        """Performs the contractions of a matrix product with psi in an optimized way.

        For mp = [a1,..., an], calculates (a1⊗..⊗an)(psi) einsum. a1, ..., an are square matrices and psi is a vector
        with the same length as the dimension of a1 tensor ... tensor an from mp = [a1,..., an].

        Todo:
            Add example code from notebook.

        Returns:
            The contracted psi, representing the updated statevector.
        """

        nr_of_matrices = len(mp)
        assert nr_of_matrices * 2 <= 26, \
            "EfficientBackend._opt_einsum_direct() can only handle at most 13 matrices in the list mp."

        abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']

        contract_string = ",".join(f"{abc[2*i]}{abc[2*i+1]}" for i in range(nr_of_matrices))
        contract_string = contract_string + "," + "".join(f"{abc[2*i+1]}" for i in range(nr_of_matrices))
        contract_string = contract_string +"->" + "".join(f"{abc[2*i]}" for i in range(nr_of_matrices))

        psi_many_leggs = psi.view().reshape(tuple((a.shape[0] for a in mp)))
        return oe.contract(contract_string, *mp,  psi_many_leggs).reshape(psi.shape)

    def _chunk_list(self, l: list, min_chunk_size: int, optimal_chunk_size: int) -> list:
        """ Converts a list into a list of list, each representing a chunk.

        Assumes that we have at least enough items to create two chunks of optimal size. Joins the last chunk with the
        second last in case it does not reach the minimum chunk size.

        Example:
            l = [1, 2, 3, 4, 5, 6, 7], min_chunk_size = 2, optimal_chunK_size = 3 returns [[1,2,3], [1,2,3,4]].

        Returns:
            The produced list of list.
        """
        assert len(l) >= 2 * optimal_chunk_size, \
            f"Inserted l with length {len(l)} but expected at least length {2 * optimal_chunk_size}."

        # Create chunks of optimal size except last one
        chunks = [l[i:i + optimal_chunk_size] for i in range(0, len(l), optimal_chunk_size)]

        # Recombine last chunk if necessary
        if len(chunks[-1]) < min_chunk_size:
            chunks[-2] = chunks[-2] + chunks[-1]
            return chunks[:-1]
        return chunks


class BinaryBackend(object):
    """Evaluate the quantum circuit through the list of gate coming from the BinaryCircuit class.
    This backend use an algorithm usefull to go beyond the linear topology considering the indices of the gates in a
    smart way. If you are running a circuit with at most two qubit then it's made a dense matrix otherwise a sparse one.
    This to optimize the calculation.

    Args:
        nqubit (int): Total number of qubit used in the circuit

    Note:
        Always use this version for the backend for simulating circuit that use non linear topologies.

    Example:
        .. code:: python

            from quantum_gates.backend import BinaryBackend

            backend = BinaryBackend(nqubit=2)

            qubit_layout = [0,1]
            level_opt = 4
            H, CNOT, I = ...
            mp_list = [[H, [0]], [I, [1]], [CNOT, [0,1]]]

            psi0 = np.array([1, 0, 0, 0])
            psi1 = backend.statevector(mp_list, psi0, level_opt, qubit_layout)  # Gives [1, 0, 0, 1] / sqrt(2)

    """

    def __init__(self, nqubit: int):
        self.nqubit = nqubit

    def statevector(self, mp_list: list, psi0: np.array, qubit_layout: list=None) -> np.array:
        """Propagates a statevector based on a list of matrix products.

        Args:
             mp_list (list[list]): List of list that contain the gates as np.array and the qubit where they act.
             psi0 (np.array): Statevector to be propagated.
             qubit_layout(list): Layout of the qubit used in the optimizer

        Returns:
            The propagated statevector.
        """
        qubit_layout = qubit_layout if qubit_layout is not None else list(range(self.nqubit))
        assert len(mp_list) > 0, f"Expected non empty matrix product list, but found {mp_list}."
        psi1 = copy.deepcopy(psi0)
 
        mp_list_opt = Optimizer(level_opt= 4, circ_list= mp_list, qubit_list=qubit_layout).optimize()

        for item in mp_list_opt:

            q_list = list(range(self.nqubit)) # list of indexes of all qubit in the circuit
            q_used = item[1] # list of indexes of the qubit used in this moment

            if len(item[1]) == 1: # check if the current is a 1 qubit
                q = item[1][0]
                q_list.remove(q) # remove index of the qubit from the list
            elif len(item[1]) == 2: # check if the current is a 2 qubit gates
                q1 = item[1][0]
                q2 = item[1][1]
                q_list.remove(q1) # remove index of the qubit from the list
                q_list.remove(q2)

            q_n_used = q_list

            k = len(q_n_used)

            if k == 0: # in case of 1 or 2 qubits circuit
                U = self.create_dense(item=item, q_used=q_used, q_n_used=q_n_used)
                psi1 = U @ psi1

            elif k > 0:
                U = self.create_sparse(item=item, q_n_used=q_n_used, q_used=q_used, N=self.nqubit)
                psi1 = U.dot(psi1)

        return psi1
    
    def create_sparse(self, item: list, q_n_used: list, q_used: list, N: int):
        """Given a gate applied on some qubits the function return a sparse array to use in the statevector function

        Args:
            item (list): List representing the entries of the gate and in whithc qubit is applied
            q_n_used (list): list of not used qubit
            q_used (list): list of used qubit
            N (int): Number of total qubit

        Raises:
            ValueError: If the number of total qubit doesn't match the sum of used and not used qubit there is a problem

        Returns:
            Sparse matrix: Sparse matrix representing the gate
        """

        k = len(q_n_used) # number of not used qubit
        m = len(q_used) # m = n-k number of used qubit

        if k+m != N:
            raise ValueError(
                f"You lose some qubit along the way, total number of qubit is {N} "
                f"and the sum of used and not used qubit is {k+m}"
            )

        # Quantities for the sparse matrix
        data = []
        row_indices = []
        col_indices = []

        # Generate only the non zero element in the matrix according to the number of not used qubit
        for i in range(2**k):
                k_str = f"{i*(2**k+1):0{2*k}b}"
                for j in range(2**(2*(m))):
                    m_str = f"{j:0{2*(m)}b}"
                    n_str = self.join_str(k_str, m_str, q_n_used, q_used, k, m) # merge string
                    row_indices.append(int(n_str[:N],2)) # row index
                    col_indices.append(int(n_str[N:],2)) # column index
                    d = 1
                    if len(item[1]) == 1: # 1 qubit gate
                        gate = item[0]
                        qubit = item[1][0]
                        d *= gate[int(n_str[qubit]),int(n_str[qubit+N])]
                    else:                 # 2 qubit gate
                        gate = item[0]
                        qubit1 = item[1][0] 
                        qubit2 = item[1][1]
                        index1 = int(n_str[qubit1] + n_str[qubit2],2)
                        index2 = int(n_str[qubit1+N] + n_str[qubit2+N],2)
                        d *= gate[index1, index2]
                    data.append(d)

        coo = coo_matrix((data, (row_indices, col_indices)), shape=(2**N, 2**N), dtype= complex)
        csr = coo.tocsr()

        return csr
    
    def join_str(self, k_str: str, m_str: str, q_n_used: list[int], q_used: list[int], k: int, m: int) -> str:
        """Join the list of the identities coming from the not used qubit and the list of the used qubit
        Args:
            k_str (str): indexes of not used qubit in binary form different from 0
            m_str (str): indexes of used qubit from the gates
            q_n_used (list[int]): list of not used qubit
            q_used (list[int]): list of used qubit
            k (int): number of not used qubit 
            m (int): number of used qubit

        Returns:
            tot_str (str): string that represent the element of the matrix
        """
        n = k+m
        tot_str = [0] * 2*n
        if len(q_n_used) != k or len(q_used) != m:
            raise ValueError("Mismatch number of qubit provided and number of qubit in the lists")

        for i, q in enumerate(q_n_used):
            tot_str[q] = k_str[i]
            tot_str[q+n] = k_str[i+k]

        for i, q in enumerate(q_used):
            tot_str[q] = m_str[i]
            tot_str[q+n] = m_str[i+m]

        return ''.join(map(str, tot_str))
    
    def create_dense(self, item : list, q_used: tuple, q_n_used: tuple) -> np.array:
        """Convert the moment in the matrix that acts in the statevector. This function use an algorithm that calculate only the non zero value
        If I have N qubit and I don't use k of it, than the complexity is 2**(2*n-k)

        Args:
            q_used (tuple): tuple of qubit used in this moment
            q_n_used (tuple): tuple of qubit not used in this moment
            item (list): list of gates as matrices and the qubits that are used

        Returns:
            U (np.array): Matrix 2^N x 2^N that is applied to the statevector
        """

        N = self.nqubit

        k = len(q_n_used) # number of not used qubit
        m = len(q_used) # m = n-k number of used qubit

        if k+m != N:
            raise ValueError("You lose some qubit along the way")

        D = np.zeros((2**N, 2**N), dtype=complex)
 
        for i in range(2**N): 
            for j in range(2**N):
                binary = f"{i:0{N}b}{j:0{N}b}" # binary representation of the indices of the matrix
                D[i,j] = 1
                if len(item[1]) == 1: # 1 qubit gate
                    gate = item[0]
                    qubit = item[1][0]
                    D[i,j] *= gate[int(binary[qubit]),int(binary[qubit+N])]
                else:                 # 2 qubit gate
                    gate = item[0]
                    qubit1 = item[1][0] 
                    qubit2 = item[1][1]
                    index1 = int(binary[qubit1] + binary[qubit2],2)
                    index2 = int(binary[qubit1+N] + binary[qubit2+N],2)
                    D[i,j] *= gate[index1,index2]
            
        return D


class BackendForOnes(object):
    """Version of the backend which is optimized for circuits that contain many identities.

    This backend was build for the H inv QFT circuit, which contains between 25 - 90% identities. It performs
    the contraction in chunks and ignores the legs that are contracted with ones.

    Note:
        This backend is experimental and should be used with caution.

    Todo:
        Finish development and perform optimization.
    """

    def __init__(self, nqubit: int):
        self.nqubit = nqubit
        self.identity = np.eye(2)
        self.low_qubit_regime = 6  # Up to this qubit number we are in this regime

    def statevector(self, mp_list: list, psi0: np.array):
        assert len(mp_list) > 0, f"Expected non empty matrix product list, but found {mp_list}."
        psi1 = copy.deepcopy(psi0)

        # Few qubit regime -> No split
        if self.nqubit <= self.low_qubit_regime:
            return self._statevector_low_qubit_regime(mp_list, psi1)

        # High qubit regime -> Many splits
        else:
            return self._statevector_high_qubit_regime(mp_list, psi1)

    def _is_identity(self, matrix) -> bool:
        """ Checks if a matrix is the 2x2 identity matrix.
        """
        return isinstance(matrix, np.ndarray) and np.array_equal(matrix, self.identity)

    def _kronecker(self, a_list) -> np.array:
        """ Takes a list of matrices and computes their combined Kronecker product.
            For len(a_list) >= 4 we use divide-and-conquer, which speeds up the computation.
        """
        n = len(a_list)

        if n == 0:
            raise Exception("Function kronecker_with_einsum received empty list.")
        elif n == 1:
            return a_list[0]
        elif n == 2:
            return np.kron(a_list[0], a_list[1])
        elif n == 3:
            return ft.reduce(np.kron, a_list)
        else:
            # Divide-and-conquer
            return np.kron(self._kronecker(a_list[:n//2]), self._kronecker(a_list[n//2:]))

    def _statevector_low_qubit_regime(self, mp_list: list, psi: np.array):
        """ In the low qubit regime (nqubit < 4), we just generate the expanded matrix product and apply it to psi.
        """
        for mp in mp_list:
            psi = self._kronecker(mp) @ psi
        return psi

    def _statevector_high_qubit_regime(self, mp_list: list, psi: np.array):
        """ In the high regime, we iteratively contract psi with new matrix products and leave out the identity
            matrices.
        """
        for mp in mp_list:
            psi = self._opt_einsum_ignoring_ones(mp, psi)
        return psi

    def _opt_einsum_ignoring_ones(self, mp: list, psi: np.array):
        """ For mp = [a1,..., an], calculates (a1⊗..⊗an)(psi) einsum. a1, ..., an are square matrices and psi is a vector
            with the same length as the dimension of a1 tensor ... tensor an from mp = [a1,..., an].

            Note: In this version, we keep track of indices i: ai = np.eye(2) and omit this contraction.
        """

        # Input validation
        matrices = [m for m in mp if isinstance(m, np.ndarray)]
        nr_of_matrices = len(matrices)
        assert nr_of_matrices <= 26, \
            "EfficientBackend._opt_einsum_direct() can only handle at most 26 matrices in the list mp."

        abc = list(string.ascii_lowercase)  # Left indices
        ABC = list(string.ascii_uppercase)  # Right indices

        # Build groups of 1s and non-1s. Contract the non-1s and leave the 1s.
        non_one_chunks = []
        last_one_was_identity = self._is_identity(matrices[0])
        shape = [matrices[0].shape[0]]
        column_is_identity = [last_one_was_identity]
        prototype = [] if last_one_was_identity else [matrices[0]]

        # Handle the rest
        for m in matrices[1:]:
            cur_is_identity = self._is_identity(m)
            if last_one_was_identity:
                if cur_is_identity:
                    # We just add another identity
                    shape[-1] *= 2
                else:
                    # We start a new chunk of non-1s
                    prototype = [m]
                    shape.append(m.shape[0])
                    column_is_identity.append(False)
            else:
                if cur_is_identity:
                    # Contract the previous chunk of non-1s
                    # Split if it is too big
                    n_terms = len(prototype)

                    if n_terms >= 19:
                        # Create four chunks
                        chunk1 = self._kronecker(prototype[:n_terms//4])
                        chunk2 = self._kronecker(prototype[n_terms//4:2*n_terms//4])
                        chunk3 = self._kronecker(prototype[2*n_terms//4:3*n_terms//4])
                        chunk4 = self._kronecker(prototype[3*n_terms//4:])
                        non_one_chunks.append(chunk1)
                        non_one_chunks.append(chunk2)
                        non_one_chunks.append(chunk3)
                        non_one_chunks.append(chunk4)
                        shape[-1] = chunk1.shape[0]
                        shape.append(chunk2.shape[0])
                        shape.append(chunk3.shape[0])
                        shape.append(chunk4.shape[0])
                        column_is_identity.append(False)
                        column_is_identity.append(False)
                        column_is_identity.append(False)

                    elif n_terms >= 11:
                        # Create three chunks
                        chunk1 = self._kronecker(prototype[:n_terms//3])
                        chunk2 = self._kronecker(prototype[n_terms//3:2*n_terms//3])
                        chunk3 = self._kronecker(prototype[2*n_terms//3:])
                        non_one_chunks.append(chunk1)
                        non_one_chunks.append(chunk2)
                        non_one_chunks.append(chunk3)
                        shape[-1] = chunk1.shape[0]
                        shape.append(chunk2.shape[0])
                        shape.append(chunk3.shape[0])
                        column_is_identity.append(False)
                        column_is_identity.append(False)

                    elif n_terms >= 8:
                        # Create two terms
                        chunk1 = self._kronecker(prototype[:n_terms//2])
                        chunk2 = self._kronecker(prototype[n_terms//2:])
                        non_one_chunks.append(chunk1)
                        non_one_chunks.append(chunk2)
                        shape[-1] = chunk1.shape[0]
                        shape.append(chunk2.shape[0])
                        column_is_identity.append(False)

                    else:
                        non_one_chunks.append(self._kronecker(prototype))

                    prototype = []
                    # Start with new identites
                    shape.append(2)
                    column_is_identity.append(True)
                else:
                    # We continue our chunk of non-1s
                    prototype.append(m)
                    shape[-1] *= m.shape[0]
            last_one_was_identity = cur_is_identity

        if not last_one_was_identity:
            assert len(prototype) > 0, "Expected list with non-zero length for prototype."
            # Contract the previous chunk of non-1s
            # Split if it is too big
            n_terms = len(prototype)
            if n_terms >= 19:
                # Create four chunks
                chunk1 = self._kronecker(prototype[:n_terms//4])
                chunk2 = self._kronecker(prototype[n_terms//4:2*n_terms//4])
                chunk3 = self._kronecker(prototype[2*n_terms//4:3*n_terms//4])
                chunk4 = self._kronecker(prototype[3*n_terms//4:])
                non_one_chunks.append(chunk1)
                non_one_chunks.append(chunk2)
                non_one_chunks.append(chunk3)
                non_one_chunks.append(chunk4)
                shape[-1] = chunk1.shape[0]
                shape.append(chunk2.shape[0])
                shape.append(chunk3.shape[0])
                shape.append(chunk4.shape[0])
                column_is_identity.append(False)
                column_is_identity.append(False)
                column_is_identity.append(False)

            elif n_terms >= 14:
                # Create three chunks
                chunk1 = self._kronecker(prototype[:n_terms//3])
                chunk2 = self._kronecker(prototype[n_terms//3:2*n_terms//3])
                chunk3 = self._kronecker(prototype[2*n_terms//3:])
                non_one_chunks.append(chunk1)
                non_one_chunks.append(chunk2)
                non_one_chunks.append(chunk3)
                shape[-1] = chunk1.shape[0]
                shape.append(chunk2.shape[0])
                shape.append(chunk3.shape[0])
                column_is_identity.append(False)
                column_is_identity.append(False)

            elif n_terms >= 8:
                # Create two terms
                chunk1 = self._kronecker(prototype[:n_terms//2])
                chunk2 = self._kronecker(prototype[n_terms//2:])
                non_one_chunks.append(chunk1)
                non_one_chunks.append(chunk2)
                shape[-1] = chunk1.shape[0]
                shape.append(chunk2.shape[0])
                column_is_identity.append(False)

            else:
                non_one_chunks.append(self._kronecker(prototype))

        # If all matrices were identities, we can just return psi as is
        if all(column_is_identity):
            return psi

        # Build contract string
        cs_matrix_list = []
        cs_tensor = ""
        cs_result = ""

        for i, are_identities in enumerate(column_is_identity):
            cs_tensor = cs_tensor + ABC[i]
            if are_identities:
                # This leg stays the same, uppercase -> uppercase
                cs_result = cs_result + ABC[i]
            else:
                # This legs is contracted, uppercase -> lowercase
                cs_result = cs_result + abc[i]
                cs_matrix_list.append(f"{abc[i]}{ABC[i]}")

        cs_matrices = ",".join(cs_matrix_list)
        cs = cs_matrices + "," + cs_tensor + "->" + cs_result

        psi_many_leggs = psi.view().reshape(shape)
        return oe.contract(cs, *non_one_chunks,  psi_many_leggs).reshape(psi.shape)
