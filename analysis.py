import timeit

import numpy as np
import random
import matplotlib.pyplot as plt

from lp import InfeasibleException
import test_lp
from simplex import *
from correct_solver import solve
from ellipsoid import ellipsoid_method

# seed the generator for consistent results
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# number of times to run each algorithm
TRIALS = 10

def bland_simplex(A, b, c):
    try:
        simplex(A, b, c, blands_rule)
    except InfeasibleException:
        pass
def zadeh_simplex(A, b, c):
    try:
        simplex(A, b, c, zadehs_rule)
    except InfeasibleException:
        pass
def cunningham_simplex(A, b, c):
    try:
        simplex(A, b, c, cunninghams_rule)
    except InfeasibleException:
        pass
def ellipsoid_func(A, b, c):
    try:
        ellipsoid_method(A, b, c, max_iter=100_000, tolerance=1e-10)
    except InfeasibleException:
        pass

# time functions in list fs on TRIALS knapsack instances of size n
def test_knapsack(fs, n):    
    instances = [test_lp.formulate_gaussian_knapsack(n) for _ in range(TRIALS)]
    instances = [(x.get_A(), x.get_b(), x.get_c()) for _, x in instances]

    times = []
    for f in fs:
        func = lambda: [f(A, b, c) for A, b, c in instances]
        times.append(timeit.timeit(func, number=1))
    
    return tuple(times)

fs = [bland_simplex, zadeh_simplex, cunningham_simplex, ellipsoid_func]
x = list(range(5, 20))
results = [test_knapsack(fs, n) for n in x]

b, z, c, e = list(map(list, zip(*results)))

plt.plot(x, b, label="Bland's rule")
plt.plot(x, z, label="Zadeh's rule")
plt.plot(x, c, label="Cunningham's rule")
plt.plot(x, e, label="Ellipsoid Method")

plt.legend()
plt.title("Performance Comparison on Knapsack Instances")
plt.xlabel("Dimension")
plt.ylabel("Time (s)")
plt.show()


# time functions in list fs on TRIALS knapsack instances of size n
def test_kleeminty(fs, n):    
    instances = [test_lp.formulate_kleeminty_cube(n) for _ in range(TRIALS)]
    instances = [(x.get_A(), x.get_b(), x.get_c()) for _, x in instances]

    times = []
    for f in fs:
        func = lambda: [f(A, b, c) for A, b, c in instances]
        times.append(timeit.timeit(func, number=1))
    
    return tuple(times)

fs = [bland_simplex, zadeh_simplex, cunningham_simplex, ellipsoid_func]
x = list(range(5, 12))
results = [test_kleeminty(fs, n) for n in x]

b, z, c, e = list(map(list, zip(*results)))

plt.plot(x, b, label="Bland's rule")
plt.plot(x, z, label="Zadeh's rule")
plt.plot(x, c, label="Cunningham's rule")
plt.plot(x, e, label="Ellipsoid Method")

plt.legend()
plt.title("Performance Comparison on Klee-Minty Instances")
plt.xlabel("Dimension")
plt.ylabel("Time (s)")
plt.show()


TRIALS = 20
# time functions in list fs on TRIALS knapsack instances of size n
def is_feasible_mcfp(instance):
    try:
        solve(*instance)
        return True
    except InfeasibleException:
        return False

def test_mcfp(fs, n):    
    instances = [test_lp.formulate_random_mcfp(n, 2*n, seed=np.random.randint(0, 2**12))[1] for _ in range(TRIALS)]
    instances = [(x.get_A(), x.get_b(), x.get_c()) for x in instances]
    for i in range(len(instances)):
        while not is_feasible_mcfp(instances[i]):
            new_instance = test_lp.formulate_random_mcfp(n, 2*n, seed=np.random.randint(0, 2**12))[1]
            instances[i] = (new_instance.get_A(), new_instance.get_b(), new_instance.get_c())

    times = []
    for f in fs:
        func = lambda: [f(A, b, c) for A, b, c in instances]
        times.append(timeit.timeit(func, number=1))
    
    return tuple(times)

fs = [bland_simplex, zadeh_simplex, cunningham_simplex, ellipsoid_func]
x = list(range(2, 8))
results = [test_mcfp(fs, n) for n in x]

b, z, c, e = list(map(list, zip(*results)))

plt.plot(x, b, label="Bland's rule")
plt.plot(x, z, label="Zadeh's rule")
plt.plot(x, c, label="Cunningham's rule")
plt.plot(x, e, label="Ellipsoid Method")

plt.legend()
plt.title("Performance Comparison on Min-cost Flow Instances")
plt.xlabel("Dimension")
plt.ylabel("Time (s)")
plt.show()

TRIALS = 10
# time functions in list fs on TRIALS knapsack instances of size n
def test_alvisfriedman(fs, n):    
    instances = [test_lp.formulate_alvisfriedman(n)[-1] for _ in range(TRIALS)]
    instances = [(x.get_A(), x.get_b(), x.get_c()) for x in instances]

    times = []
    for f in fs:
        func = lambda: [f(A, b, c) for A, b, c in instances]
        times.append(timeit.timeit(func, number=1))
    
    return tuple(times)

fs = [bland_simplex, zadeh_simplex, cunningham_simplex]
x = list(range(3, 8))
results = [test_alvisfriedman(fs, n) for n in x]

b, z, c = list(map(list, zip(*results)))

plt.plot((np.array(x) - 1) * 10, b, label="Bland's rule")
plt.plot((np.array(x) - 1) * 10, z, label="Zadeh's rule")
plt.plot((np.array(x) - 1) * 10, c, label="Cunningham's rule")

plt.legend()
plt.title("Performance Comparison on Alvis-Friedmann Instances")
plt.xlabel("Dimension")
plt.ylabel("Time (s)")
plt.show()
