import numpy as np
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
    m, n = A.shape
    x = np.zeros(n)
    P = np.eye(n) * n**2  # Initial ellipsoid
    
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
        grad = c / norm(c)
        if np.abs(c @ x - grad @ x) < tolerance:
            break

        # Update ellipsoid (center and shape matrix)
        a_P_a = a @ P @ a
        if a_P_a <= tolerance:
            raise Exception("Ellipsoid is too small")
        
        P = (n**2 / (n**2 - 1)) * (P - (2 / (n + 1)) * np.outer(P @ a, P @ a) / a_P_a)
        x = x - (1 / (n + 1)) * P @ a / np.sqrt(a_P_a)

    return x
