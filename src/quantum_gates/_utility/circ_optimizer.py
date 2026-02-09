"""
This file implement a class that represents an optimization algorithm for the output of the circuit class. 
It takes the list of gate and qubit on which the gate is applied and compacts the list so there are lesser matrix to calculate in the backend
Pay attention that it is designend mainly to work with BinaryCircuit and BinaryBackend

"""

import numpy as np

class Optimizer(object):
    """This class implements an algorithm used before calculate the statevector in BinaryBackend to optimize the output coming from the BinaryCircuit class. 
    The output is the list of gate and the qubit(s) on which it's applied. The idea is to reduce the lentgh of the list watching at the pattern of near 
    gates and compress them in less gate.
    The algorithm works on 4 different levels, higher the level is and more it's expected that the length will be reduced.

    Example:
        .. code:: python

            from quantum_gates.utilities import Optimizator
            from quantum_gates.circuits import BinaryCircuit
            from quantum_gates.gates import standard_gates

            N_QUBIT = 10
            depth = 0

            circuit = BinaryCircuit(nqubit = N_QUBIT, depth=depth, gates= standard_gates)

            # fill the info gates list 

            result = circuit._info_gates_list
            opt = Optimizator(level_opt= 4, circ_list= result, qubit_list=qubits_layout_t)

            result_opt_4 = opt.optimize()


    """

    def __init__(self, level_opt:int, circ_list: list, qubit_list: list):
        """

        Args:
            level_opt (int): At which level you want optimize
            circ_list (list): List of matrices representing the gates and the qubits on which the gates act coming from BinaryCircuit
            qubit_list (list): Layout of the qubit used in stage 4 to calculate how many qubits there are

        Raises:
            ValueError: The possible values for the optimization go from 0 (no opt) to 4 (max opt)
        """
        if level_opt > 4 or level_opt < 0:
            raise ValueError(f"Not exist the level {level_opt} of optimizaion. They are from 0 to 4")
        self.circ_list = circ_list
        self.level_opt = level_opt
        self.qubit_list = qubit_list

    def optimize(self) -> list:
        """This function runs the optimization algorithm at the optimazion level chosen. 
        There are some exception for circuit with only one qubit or too short

        Returns:
            list: Return the optimized list of gates ready to go in the statevector function
        """

        if len(self.circ_list) <= 2:
            level = 0
        elif len(self.qubit_list) == 1:
            level = 0
        else:
            level = self.level_opt

        for i in self.circ_list:
            if len(i[1]) == 2:
                if i[1][1] == -1:
                    i[1] = [i[1][0]]
                i[1] = list(i[1])    

        if level == 0:
            return self.circ_list

        if level == 1:
            result_1 = self.opt_level_1(gate_list=self.circ_list)
            return result_1
        
        elif level == 2:
            result_1 = self.opt_level_1(gate_list=self.circ_list)
            result_2 = self.opt_level_2(gate_list=result_1)
            return result_2
        
        elif level == 3:
            result_1 = self.opt_level_1(gate_list=self.circ_list)
            result_2 = self.opt_level_2(gate_list=result_1)
            result_3 = self.opt_level_3(gate_list=result_2)
            return result_3

        elif level == 4:
            result_1 = self.opt_level_1(gate_list=self.circ_list)
            result_2 = self.opt_level_2(gate_list=result_1)
            result_3 = self.opt_level_3(gate_list=result_2)
            result_4 = self.opt_level_4(gate_list=result_3)
            return result_4

    def opt_level_1(self, gate_list: list) -> list:
        """This is the first level of optimiazion. It watch at the near single qubit gate: 
        If there are two or more closed gates that act on the same qubit then they are multiplicated together to obtain a single gate.

        Args:
            gate_list (list): List of gate coming from BinaryCircuit

        Returns:
            list: Return the first level optimize list of gate
        """
        result_1 = []
        if len(gate_list) < 2:
            return gate_list

        while gate_list:
            if len(gate_list[0][1]) == 1: # check if the current element is a 1q gate
                qubit = gate_list[0][1]
                c = 1
                # count how many successive gate have the same qubit
                while c < len(gate_list) and len(gate_list[c][1]) == 1 and qubit == gate_list[c][1]:
                    c += 1
                if c > 1:
                    gates = []
                    for i in range(c): #store in a list all the matrices of the gates involved
                        gates.append(gate_list[i][0])
                    # calculate the gate
                    gate = np.identity(2)
                    for j in range(c):
                        gate = gates[j] @ gate
                    result_1.append([gate,qubit]) # append the gate with the qubit
                    gate_list = gate_list[c:]
                else: # if there is only 1 consequent 2q gate with the same qubit just happend the 
                    result_1.append(gate_list[0])
                    gate_list = gate_list[c:]
            else:
                result_1.append(gate_list[0])
                gate_list = gate_list[1:] 
            if len(gate_list) == 1:
                result_1.append(gate_list[0])
                gate_list.remove(gate_list[0])
        
        return result_1
    
    def opt_level_2(self, gate_list: list) -> list:
        """This is the second level of optimization. It watch at the two single qubit gate before and after a two qubit gate:
        If there are one or two single qubit gate before a two qubit gate that act on the same qubit then there are multiplicate together.
        The same is done if there are one or two single qubit gate after the two qubit gate.
        In this way most of the single qubit gate are deleted and the information is stored inside the two qubit gates

        Args:
            gate_list (list): List of gate coming from the first level of optimization

        Returns:
            list: Return the second level optimize list of gate
        """

        q2_gate = 0
        result_2 = []

        for i in gate_list: #count the number of two qubit gate in the result
            if len(i[1]) == 2:
                q2_gate += 1

        if q2_gate > 0: # check that exists almost one two qubit gate 
            for i in range(q2_gate): # iterate over the number of  two qubit gate
                
                # create the snippet
                _snippet = []
                counter = 0
                while len(gate_list[counter][1]) != 2:

                    _snippet.append(gate_list[counter])
                    counter += 1
                _snippet.append(gate_list[counter]) # append the 2q gate
                if len(gate_list) > counter+1: # assert that there are gate after the one considered
                    if len(gate_list[counter+1][1]) == 1: # append the right after gate if it is a 1 q gate
                        _snippet.append(gate_list[counter+1])
                        if len(gate_list) > counter+2:
                            if len(gate_list[counter+2][1]) == 1: # if it's appended the first closer watch if append the second after gate if it is a 1 q gate
                                _snippet.append(gate_list[counter+2])
                # the snippet now is done

                # remove from result the snippet
                gate_list = gate_list[len(_snippet):]            

                # process the snippet
                processed_snippet = self.process_snippet(snippet = _snippet)
                
                # append to result_2 the processed snippet
                result_2 += processed_snippet

            # process the last part of the gate_list which are only single qubit gates.
            result_2 += gate_list
        else:
            result_2 = gate_list
        return result_2
    
    def opt_level_3(self, gate_list: list) -> list:
        """This is the third level of optimization. It watch at the near two qubit gates.
        If there are two or more closed gates that act on the same qubits then they are multiplicated together to obtain a single two qubit gate.

        Args:
            gate_list (list): List of gate coming from the second level of optimization

        Returns:
            list: Return the third level optimize list of gate
        """

        result_3 = []

        while gate_list:
            if len(gate_list[0][1]) == 2 and len(gate_list) > 1: # check if the current element is a 2q gate
                qubit = gate_list[0][1]
                c = 1
                while len(gate_list[c][1]) == 2 and qubit == gate_list[c][1]: # count how many successive gate have the same qubit
                    c += 1
                    if c >= len(gate_list): # if the list of gate_list is finished break the loop
                        break
                        
                if c > 1:
                    gates = []
                    for i in range(c): #store in a list all the matrices of the gates involved
                        gates.append(gate_list[i][0])
                    # calculate the gate
                    gate = np.identity(4)
                    for j in range(c):
                        gate = gates[j] @ gate
                    result_3.append([gate,qubit]) # append the gate with the qubit
                    gate_list = gate_list[c:]
                else: # if there is only 1 consequent 2q gate with the same qubit just happend the 
                    result_3.append(gate_list[0])
                    gate_list = gate_list[c:]
            else:
                result_3.append(gate_list[0])
                gate_list = gate_list[1:]
            if len(gate_list) == 1:
                result_3.append(gate_list[0])
                gate_list.remove(gate_list[0])
        
        return result_3
    
    def opt_level_4(self, gate_list: list) -> list:
        """This is the fourth level of optimization. It watch at the last part after the last two qubit gate of the list where there are only single qubit gate:
        Considering the used qubit it scan all the last single qubit gates and if there are two or more gates that act on the same qubit they are multiplicated together

        Args:
            gate_list (list): List of gate coming from the third level of optimization

        Raises:
            ValueError: The calculated number of gates in the last part doesn't match the input list

        Returns:
            list: Return the fourth level optimize list of gate
        """

        last_part = []
        l = 0
        gate_list = gate_list[::-1] # reverse the list of the gate
        reorder_qubit_list = list(range(len(self.qubit_list)))

        # check if there is almost one two qubit gate
        for item in gate_list:
            if len(item[1]) == 2:
                q2_gate_check = True
                break
            else: 
                q2_gate_check = False

        if q2_gate_check: # there is almost one two qubit gate
            while len(gate_list[l][1]) == 1: # append in a list the last gate after the 2q gate
                last_part.append(gate_list[l])
                l+=1

            gate_list = gate_list[::-1] # re-reverse the list of the gate 
            length_last_part = len(last_part) # calculate the length of the last part

            if l != length_last_part:
                raise ValueError("Mismatch between length of last part and gates in it")

            if l > 1: # check if there almost two element in the last part
                last_part = last_part[::-1] # reverse the list to obtain the right order
                result_4 = gate_list[:-l] # add all the gate before the last part a result_4
                for q_i in reorder_qubit_list: # scan all the used qubit 
                    if len(last_part) > 1:
                        indices = [index for index, element in enumerate(last_part) if element[1][0] == q_i] # find the indices of the element for the qubit q_i
                        if len(indices) > 1:
                            gate = np.identity(2)
                            for ind in indices:
                                gate = last_part[ind][0] @ gate
                            result_4.append([gate,[q_i]])
                        elif len(indices) == 1:
                            ind = indices[0]
                            result_4.append([last_part[ind][0],[q_i]])
                        last_part = [element for i, element in enumerate(last_part) if i not in indices] # remove from the last part the used items in this iteration
                    elif len(last_part) == 1:
                        result_4.append(last_part[0])
                        break
                    elif len(last_part) == 0:
                        break
            else:
                result_4 = gate_list
        else: # there is only one qubit gate
            last_part = gate_list
            length_last_part = len(last_part) # calculate the length of the last part
            if length_last_part > 1: # check if there almost two element in the last part
                last_part = last_part[::-1] # reverse the list to obtain the right order
                result_4 = []
                for q_i in reorder_qubit_list: # scan all the used qubit 
                    if len(last_part) > 1:
                        indices = [index for index, element in enumerate(last_part) if element[1][0] == q_i] # find the indices of the element for the qubit q_i
                        if len(indices) > 1:
                            gate = np.identity(2)
                            for ind in indices:
                                gate = last_part[ind][0] @ gate
                            result_4.append([gate,[q_i]])
                        else: 
                            ind = indices[0]
                            result_4.append([last_part[ind][0],[q_i]])
                        last_part = [element for i, element in enumerate(last_part) if i not in indices] # remove from the last part the used items in this iteration
                    elif len(last_part) == 1:
                        result_4.append(last_part[0])
                        break
                    elif len(last_part) == 0:
                        break
            else:
                result_4 = last_part[::-1]
        return result_4
    
    def process_snippet(self, snippet: list) -> list:
        """This is an auxiliary function used at the second level to isolate and analyze each two-qubit gate along with any neighboring single-qubit gates that act on the same qubits.

        Args:
            snippet (list): Little list of gate that contain the two qubit gate and its neighbour

        Raises:
            ValueError: If the gate is not a two qubit gate is raised an error

        Returns:
            list: Processed list 
        """
        loc = 0 # Index of the 2q gates in the snippet
        n_elem = len(snippet) -1 # Number of elements in the snippet starting from 0
        proces_snippet = [] 

        # Find the index of the 2q gates in the snippet
        for i,item in enumerate(snippet):
            if len(item[1]) ==  2:
                loc = i

        # Before the two qubit gate

        # Check there are almost two gate before the 2q gate
        if loc - 2 >= 0: 

            # Check if the two gates before are compatible with the 2q gate
            if snippet[loc-2][1][0] == snippet[loc][1][0] and snippet[loc-1][1][0] == snippet[loc][1][1]: 
                gate = snippet[loc][0] @ np.kron(snippet[loc-2][0],snippet[loc-1][0])
                qubits = snippet[loc][1]

                # Check if there are gate before the ones used 
                if loc - 2 > 0: 
                    for i in range(loc-2):
                        proces_snippet.append(snippet[i])
                    proces_snippet.append([gate,qubits])
                else:
                    proces_snippet.append([gate,qubits])

            # Check if the two gates before are compatible with the 2q gate with inverted qubit
            elif snippet[loc-2][1][0] == snippet[loc][1][1] and snippet[loc-1][1][0] == snippet[loc][1][0]: 
                gate = snippet[loc][0] @ np.kron(snippet[loc-1][0],snippet[loc-2][0])
                qubits = snippet[loc][1]

                # Check if there are gate before the ones used 
                if loc - 2 > 0: 
                    for i in range(loc-2):
                        proces_snippet.append(snippet[i])
                    proces_snippet.append([gate,qubits])
                else:
                    proces_snippet.append([gate,qubits])

            # The closest gate is compatible with control qubit, but not the second closer
            elif snippet[loc-1][1][0] == snippet[loc][1][0]: 
                gate = snippet[loc][0] @ np.kron(snippet[loc-1][0],np.identity(2))
                qubits = snippet[loc][1]
                for i in range(loc-1):
                    proces_snippet.append(snippet[i]) 
                proces_snippet.append([gate,qubits])

            # The closest gate is compatible with target qubit, but not the second closer
            elif snippet[loc-1][1][0] == snippet[loc][1][1]: 
                gate = snippet[loc][0] @ np.kron(np.identity(2),snippet[loc-1][0])
                qubits = snippet[loc][1]
                for i in range(loc-1):
                    proces_snippet.append(snippet[i]) 
                proces_snippet.append([gate,qubits])
            
            # No one of the item before match with the 2q gate
            else: 
                for i in range(loc+1):
                    proces_snippet.append(snippet[i])
        
        # Check if there is one gate before the 2q gate
        elif loc -1 >= 0: 

            # The closer gate is compatible with control qubit
            if snippet[loc-1][1][0] == snippet[loc][1][0]: 
                gate = snippet[loc][0] @ np.kron(snippet[loc-1][0],np.identity(2))
                qubits = snippet[loc][1]
                if loc - 1 > 0:
                    for i in range(loc-1):
                        proces_snippet.append(snippet[i])
                    proces_snippet.append([gate,qubits])
                else:
                    proces_snippet.append([gate,qubits])

            # The closer gate is compatible with target qubit
            elif snippet[loc-1][1][0] == snippet[loc][1][1]: 
                gate = snippet[loc][0] @ np.kron(np.identity(2),snippet[loc-1][0])
                qubits = snippet[loc][1]
                if loc - 1 > 0:
                    for i in range(loc-1):
                        proces_snippet.append(snippet[i])
                    proces_snippet.append([gate,qubits])
                else:
                    proces_snippet.append([gate,qubits])

            # The closer gate is neither compatible with the targer nor the control qubit
            else: 
                for i in range(loc+1):
                    proces_snippet.append(snippet[i])

        # If there aren't items before the 2q qubit just append the item
        else: 
            proces_snippet.append(snippet[0]) 
            if len(snippet[0][1]) != 2:
                raise ValueError(f"Expected an item that act on 2 qubit, gate with {snippet[0][1]} qubit found")

        # After the two qubit gate

        # Check if there are two items before the 2q gate
        if loc + 2 <= n_elem: 

            # Check if the two gates before are compatible with the 2q gate
            if snippet[loc+2][1][0] == snippet[loc][1][0] and snippet[loc+1][1][0] == snippet[loc][1][1]: 
                gate = np.kron(snippet[loc+2][0],snippet[loc+1][0]) @ proces_snippet[-1][0]
                qubits = proces_snippet[-1][1]
                proces_snippet = proces_snippet[:-1]
                proces_snippet.append([gate,qubits])

            # Check if the two gates before are compatible with the 2q gate with inverted qubit
            elif snippet[loc+2][1][0] == snippet[loc][1][1] and snippet[loc+1][1][0] == snippet[loc][1][0]: 
                gate = np.kron(snippet[loc+1][0],snippet[loc+2][0]) @ proces_snippet[-1][0]
                qubits = proces_snippet[-1][1]
                proces_snippet = proces_snippet[:-1]
                proces_snippet.append([gate,qubits])

            # The second closest gate is compatible with the control qubti not the first gate
            elif snippet[loc+2][1][0] == snippet[loc][1][0] and snippet[loc+1][1][0] != snippet[loc][1][1]: 
                gate = np.kron(snippet[loc+2][0],np.identity(2)) @ proces_snippet[-1][0]
                qubits = proces_snippet[-1][1]
                proces_snippet = proces_snippet[:-1]
                proces_snippet.append([gate,qubits])
                proces_snippet.append(snippet[loc+1])

            # The second closest gate is compatible with the control qubti not the first gate
            elif snippet[loc+2][1][0] == snippet[loc][1][1] and snippet[loc+1][1][0] != snippet[loc][1][0]: 
                gate = np.kron(np.identity(2),snippet[loc+2][0]) @ proces_snippet[-1][0]
                qubits = proces_snippet[-1][1]
                proces_snippet = proces_snippet[:-1]
                proces_snippet.append([gate,qubits])
                proces_snippet.append(snippet[loc+1])

            # The closest gate is compatible with control qubit, not the second
            elif snippet[loc+1][1][0] == snippet[loc][1][0]: 
                gate = np.kron(snippet[loc+1][0],np.identity(2)) @ proces_snippet[-1][0]
                qubits = proces_snippet[-1][1]
                proces_snippet = proces_snippet[:-1]
                proces_snippet.append([gate,qubits])
                proces_snippet.append(snippet[loc+2])

            # The closest gate is compatible with target qubit, not the second
            elif snippet[loc+1][1][0] == snippet[loc][1][1]: 
                gate = np.kron(np.identity(2),snippet[loc+1][0]) @ proces_snippet[-1][0]
                qubits = proces_snippet[-1][1]
                proces_snippet = proces_snippet[:-1]
                proces_snippet.append([gate,qubits])
                proces_snippet.append(snippet[loc+2]) 

             # Nor the first or the second after is compatible with the 2q gate  
            else:
                proces_snippet.append(snippet[loc+1])
                proces_snippet.append(snippet[loc+2])

        # Check if there is one item before the 2q gate        
        elif loc+1 <= n_elem: 

            # The closest gate is compatible with control qubit
            if snippet[loc+1][1][0] == snippet[loc][1][0]: 
                gate = np.kron(snippet[loc+1][0],np.identity(2)) @ proces_snippet[-1][0]
                qubits = proces_snippet[-1][1]
                proces_snippet = proces_snippet[:-1]
                proces_snippet.append([gate,qubits]) 

            # The closest gate is compatible with target qubit
            elif snippet[loc-1][1][0] == snippet[loc][1][1]: 
                gate = np.kron(np.identity(2),snippet[loc+1][0]) @ proces_snippet[-1][0]
                qubits = proces_snippet[-1][1]
                proces_snippet = proces_snippet[:-1]
                proces_snippet.append([gate,qubits])
            else: 
                proces_snippet.append(snippet[loc+1])
        
        return proces_snippet
