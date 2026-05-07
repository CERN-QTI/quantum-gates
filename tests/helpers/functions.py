"""
Helper functions for the unit tests.
"""

import random

from tests.helpers.gates import X, SX, Z, H, CNOT, identity, single_qubit_gate_list


def generate_random_matrix_products(nqubits: int, steps: int, prob_cnot: float=0.5, many_identites=False):
    """ Generates list of matrix products for nqubits. In this version, we just use X, Z, H, 1 (single qubit) and
        CNOT (two qubit) matrices in the matrix products. CNOT gates are selected with a probability (prop_cnot) in
        cases there is still budget for two qubit gates.
        Generate list both for first class of backends (Efficient, Standard, Ones) and also for second class (Binary) that use different type of mp_list
    """

    if many_identites:
        gate_list = [X, SX, Z, H] + [identity for _ in range(40)]
    else:
        gate_list = list(single_qubit_gate_list) + [identity]

    # Result
    mp_list = []
    mp_list_b = []

    # Generate
    for step in range(steps):
        mp = []
        budget = nqubits
        while budget > 0:
            # Case: Only one qubit left
            if budget == 1:
                gate = random.choice(gate_list)
                mp.append(gate)
                the_info = [gate,[nqubits-budget, -1]]
                budget -= 1

            # Case: More than one qubit left -> We can apply two qubit gates.
            else:
                # Select CNOT with probability prop_cnot.
                if random.random() < prob_cnot:
                    mp.append(CNOT)
                    mp.append(1)
                    the_info = [CNOT, [nqubits-budget, nqubits-budget+1]]
                    budget -= 2
                else:
                    gate = random.choice(gate_list)
                    mp.append(gate)
                    the_info = [gate, [nqubits-budget, -1]]
                    budget -= 1
            mp_list_b.append(the_info)
        assert budget == 0
        mp_list.append(mp)
    return mp_list, mp_list_b


def vector_almost_equal(m1, m2, nqubit: int, abstol: float=1e-9):
    """ Check if m1 and m2 are close.
    """
    diff = m1 - m2
    return all((abs(diff[i]) < abstol for i in range(2**nqubit)))


def matrix_almost_equal(m1, m2, nqubit: int, abstol: float=1e-9):
    """ Check if m1 and m2 are close. """
    diff = m1 - m2
    return all((abs(diff[i,j]) < abstol) for i in range(2**nqubit) for j in range(2**nqubit))
