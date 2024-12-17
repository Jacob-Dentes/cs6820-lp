"""
Authors: Jacob Dentes

An implementation of the Simplex method
for solving Ax <= b lps
"""
import numpy as np
import scipy as sc
from lp import InfeasibleException, UnboundedException

# also known as "minimum index"
def blands_rule(pivot_inp):
    _, b, _, A_b, _, _, _, system_sol, tolerance, non_basis, basis, reduced_costs = pivot_inp
    # get first with a reduced cost
    entering_idx = np.argmax(reduced_costs >= tolerance)
    entering = non_basis[entering_idx]
    
    # min ratio test
    denominators = system_sol[:, [entering_idx]].todense().flatten()
    pos_denominators = denominators > 0
    if pos_denominators.sum() <= 0:
        raise UnboundedException("LP unbounded")
    leaving_idx = np.argmin(sc.sparse.linalg.spsolve(A_b, b)[pos_denominators] / denominators[pos_denominators])
    leaving = basis[pos_denominators][leaving_idx]

    return (entering, leaving)

# also known as "least used"
def zadehs_rule(pivot_inp):
    if len(pivot_inp) < 13:
        pivot_inp.append(np.zeros(pivot_inp[0].shape[1]))
    _, b, _, A_b, _, _, _, system_sol, tolerance, non_basis, basis, reduced_costs, used = pivot_inp
    # get least used variable
    entering_options = (reduced_costs >= tolerance).nonzero()
    least_used = np.argmin(used[non_basis[entering_options]]) # index of least used entering option
    entering = non_basis[entering_options][least_used]    
    entering_idx = np.argmax(non_basis == entering)

    pivot_inp[12][entering] += 1

    # min ratio test
    denominators = system_sol[:, [entering_idx]].todense().flatten()
    pos_denominators = denominators > 0
    if pos_denominators.sum() <= 0:
        raise UnboundedException("LP unbounded")
    leaving_idx = np.argmin(sc.sparse.linalg.spsolve(A_b, b)[pos_denominators] / denominators[pos_denominators])
    leaving = basis[pos_denominators][leaving_idx]

    return (entering, leaving)

# also known as "least recently used"
def cunninghams_rule(pivot_inp):
    if len(pivot_inp) < 13:
        pivot_inp.append(np.zeros(pivot_inp[0].shape[1]))
        pivot_inp.append(0)

    _, b, _, A_b, _, _, _, system_sol, tolerance, non_basis, basis, reduced_costs, used, iter = pivot_inp
    # get least recently used variable
    entering_options = (reduced_costs >= tolerance).nonzero()
    least_rec_used = np.argmin(used[non_basis[entering_options]]) # index of least recently used entering option
    entering = non_basis[entering_options][least_rec_used]    
    entering_idx = np.argmax(non_basis == entering)

    pivot_inp[12][entering] = iter
    pivot_inp[13] += 1

    # min ratio test
    denominators = system_sol[:, [entering_idx]].todense().flatten()
    pos_denominators = denominators > 0
    if pos_denominators.sum() <= 0:
        raise UnboundedException("LP unbounded")
    leaving_idx = np.argmin(sc.sparse.linalg.spsolve(A_b, b)[pos_denominators] / denominators[pos_denominators])
    leaving = basis[pos_denominators][leaving_idx]

    return (entering, leaving)
    

def _simplex_aux(A, b, c, basis, tolerance, pivot_rule=blands_rule):
    # Solves simplex starting at an initial basis
    # basis should be a numpy array of variable indices
    basis = np.array(basis)
    non_basis = np.array([i for i in range(len(c)) if i not in basis])

    pivot_inp = [None] * 12

    pivots = 0
    while True:
        A_b = A[:, basis]
        A_n = A[:, non_basis]
        c_b = c[basis]
        c_n = c[non_basis]

        system_sol = sc.sparse.linalg.spsolve(A_b, A_n)

        reduced_costs = c_n.T - c_b.T @ system_sol
        if reduced_costs.max() < tolerance:
            break

        pivot_inp[:12] = [A, 
                b, 
                c,
                A_b,
                A_n, 
                c_b,
                c_n,
                system_sol,
                tolerance, 
                non_basis, 
                basis, 
                reduced_costs]

        entering, leaving = pivot_rule(pivot_inp)
        
        basis[basis == leaving] = entering
        non_basis[non_basis == entering] = leaving

        basis.sort()
        non_basis.sort()
        pivots += 1

    x_b = sc.sparse.linalg.spsolve(A_b, b)
    sol = np.zeros(len(c))
    sol[basis] = x_b
    return (sol, pivots, basis)

def simplex(A, b, c, pivot_rule=blands_rule, tolerance=1e-10, count_pivots=False):
    """
    Use simplex method to solve Max (c^T x) subject to Ax <= b
    """
    c = np.append(c, np.zeros(A.shape[0]))
    A = np.append(A, np.eye(A.shape[0]), axis=1)

    # first phase simplex
    if (b < 0).sum() > 0:
        artificial_vars = (-1.0 * np.eye(A.shape[0]))[:, b < 0]

        A_aux = np.append(A, artificial_vars, axis=1)
        c_aux = np.append(np.zeros(len(c)), -1.0 * np.ones(artificial_vars.shape[1]))

        aux_basis = []
        for i in range(A.shape[0]):
            if b[i] >= 0:
                aux_basis.append(A.shape[1] - A.shape[0] + i)
        aux_basis += list(range(A.shape[1], A_aux.shape[1]))
        aux_basis = np.array(aux_basis)

        phase_1 = _simplex_aux(sc.sparse.csc_array(A_aux), b, c_aux, aux_basis, tolerance, pivot_rule)

        if np.sum(phase_1[0][A.shape[1]+1:]) > tolerance:
            raise InfeasibleException("LP infeasible")

        init_basis = phase_1[2]
        bs = (b < 0).nonzero()[0]
        for i in range(len(init_basis)):
            if init_basis[i] >= A.shape[1]:
                # artificial variable w/ value 0 in basis
                # want to replace with corresponding slack var
                og_vars = A.shape[1] - A.shape[0]
                replace_with = og_vars + bs[init_basis[i] - A.shape[1]]
                init_basis[i] = replace_with
        init_basis = np.sort(init_basis)

    else:
        phase_1 = (None, 0)
        init_basis = np.array(list(range(A.shape[1] - A.shape[0], A.shape[1])))
    
    # second phase simplex
    phase_2 = _simplex_aux(sc.sparse.csc_array(A), b, c, init_basis, tolerance, pivot_rule)[:A.shape[1]-A.shape[0]]
    if count_pivots:
        return (phase_2[0], phase_2[1] + phase_1[1])
    return phase_2[0]
