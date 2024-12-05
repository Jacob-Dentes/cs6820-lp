"""
Uses Scipy's implementation as a correctness comparison
"""
from scipy.optimize import linprog
from lp import InfeasibleException, UnboundedException

def solve(A, b, c, tolerance=1e-10):
    result = linprog(-1*c, A, b, method='revised simplex', options={'tol': tolerance})
    
    if result.status == 2:
        raise InfeasibleException("Problem infeasible")
    if result.status == 3:
        raise UnboundedException("Problem unbounded")
    if result.status == 4:
        raise Exception("Scipy encountered numerical issues")

    return result.x

