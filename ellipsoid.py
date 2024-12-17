import numpy as np
from lp import InfeasibleException
from numpy.linalg import norm, inv

def ellipsoid_method(A, b, c, tolerance=1e-10, max_iter=1000):
    """
    Solves LP: maximize c^T x, subject to Ax <= b using the ellipsoid method.
    
    Arguments:
    - A: constraint matrix
    - b: constraint vector
    - c: objective coefficients
    - tolerance: precision for termination
    - max_iter: maximum number of iterations
    
    Returns:
    - x: solution vector
    """
    original_length = A.shape[1]
    # Reduce to LP search
    A2 = np.zeros((A.shape[0] + A.shape[1], A.shape[0] + A.shape[1]))
    # Add the dual constraint matrix
    A2[:A.shape[0], :A.shape[1]] = A
    A2[A.shape[0]:, A.shape[1]:] = -A.T

    b2 = np.append(np.append(b, -c), np.zeros(1))
    c2 = np.zeros(A.shape[0] + A.shape[1])
    # want the primal obj >= dual obj
    # <==> dual_obj - primal_obj <= 0
    A2 = np.append(A2, [np.append(-c, b)], axis=0)

    b2 = np.append(b2, np.zeros(A2.shape[1]))
    A2 = np.append(A2, -np.eye(A2.shape[1]), axis=0)
    A, b, c = A2, b2, c2
    
    m, n = A.shape
    x = np.zeros(n)
    P = np.eye(n) * 2**(n)  # Initial ellipsoid
    
    for iteration in range(max_iter):
        # Check feasibility
        violations = A @ x - b
        if all(violations <= tolerance):  # All constraints satisfied
            break

        # Find the most violated constraint
        idx = np.argmax(violations)
        a = A[idx]
        b_i = b[idx]

        # Check if solution satisfies the objective with tolerance
        # grad = c / norm(c)
        # if np.abs(c @ x - grad @ x) < tolerance:
        #     break

        # Update ellipsoid (center and shape matrix)
        a_P_a = a @ P @ a
        if a_P_a <= tolerance:
            raise InfeasibleException(f"Ellipsoid is too small at iteration {iteration}")
        
        P = (n**2 / (n**2 - 1)) * (P - (2 / (n + 1)) * np.outer(P @ a, P @ a) / a_P_a)
        x = x - (1 / (n + 1)) * P @ a / np.sqrt(a_P_a)

    return x[:original_length]
