import timeit

import numpy as np
import random
import matplotlib.pyplot as plt

import test_lp
from simplex import *
from ellipsoid import ellipsoid_method

# seed the generator for consistent results
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# number of times to run each algorithm
TRIALS = 10

simplex_closure = lambda x: lambda a, b, c: simplex(a, b, c, x)
bland_simplex = simplex_closure(blands_rule)
zadeh_simplex = simplex_closure(zadehs_rule)
cunningham_simplex = simplex_closure(cunninghams_rule)

# time functions in list fs on TRIALS knapsack instances of size n
def test_knapsack(fs, n):    
    instances = [test_lp.formulate_gaussian_knapsack(n) for _ in range(TRIALS)]
    instances = [(x.get_A(), x.get_b(), x.get_c()) for _, x in instances]

    times = []
    for f in fs:
        func = lambda: [f(A, b, c) for A, b, c in instances]
        times.append(timeit.timeit(func, number=1))
    
    return tuple(times)

fs = [bland_simplex, zadeh_simplex, cunningham_simplex, ellipsoid_method]
x = list(range(5, 10))
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

fs = [bland_simplex, zadeh_simplex, cunningham_simplex, ellipsoid_method]
x = list(range(5, 10))
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

