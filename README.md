# CS 6820 LP
A set of Python implementations for linear programming. This package was made for a Cornell CS 6820 project to compare methods for solving linear programs.
It is not meant to be used for production purposes, see [PySCIPOpt](https://github.com/scipopt/PySCIPOpt) for a production-ready alternative.

## Authors:
- Daniel Cao
- Jacob Dentes
- Santiago Lai
- Hangyu Zhou

## Installation
The package does not need to be installed, just place the `.py` files in the working directory of your project.
Ensure that your environment has a recent release of Numpy and Scipy. 
Your environment also needs Networkx if you want to run the Min-cost flow examples.

## Features
### lp.py
The file `lp.py` contains a Python class LP for creating linear programs using a builder pattern inspired by SCIP.
Specifically, you can declare variables, constraints, and objectives with an intuitive syntax.
After creation, the module will convert the LP to standard form:

$$\max \quad c^T x$$

$$\text{s.t.} \quad Ax \leq b$$

$$\quad x \geq 0$$

The following is an example of usage for `lp.py`. More examples can be found in `test_lp.py`.
```
from lp import LP, MAXIMIZE

program = LP()

# Declare the LP variables.
# NOTE: Variables are non-negative by default, pass in 'nonnegative=False' for a free variable
x1 = program.add_var("x1")
x2 = program.add_var("x2")

# Add a constraint between the variables
program.add_constr(x1 + 5 <= x2)

x3 = program.add_var("x3")
program.add_constr(1*x2 + x2 - 3 == 2 + x3 * 0.5)

# Set a maximization objective
program.set_objective(9 * x1 - 2*x2 + x3, MAXIMIZE)

print(program.get_A())
print(program.get_b())
print(program.get_c())
```

### simplex.py
The file `simplex.py` contains an implementation of the revised simplex method with multiple options for the pivoting algorithm.
Currently, the pivoting algorithms are:
1. Bland's rule
2. Zadeh's rule (also known as "least used")
3. Cunningham's rule (also known as "least recently used")

The available methods require the LP to be in the same standard form as output by `lp.py`. An example of usage is below, it assumes you have 
an LP instance `program` (for example, the one created above). More examples can be found in `test_lp.py`.
```
from simplex import simplex
from lp import LP, MAXIMIZE

program = LP()
### CREATE THE LP USING THE BUILDER PATTERN ###

# Get the standard form
A = program.get_A()
b = program.get_b()
c = program.get_c()

# Get the solution to the problem
# NOTE: This method can raise an InfeasibleException or an UnboundedException
sol = simplex(A, b, c)

# Extract the values of the variables
# Note: Always evaluate to extract values, the 'sol' variable is meaningless on its own
x1_value = x1.evaluate(sol)
x2_value = x2.evaluate(sol)
x3_value = x3.evaluate(sol)
```

### ellipsoid.py
The file `ellipsoid.py` contains an implementation of ellipsoid method. It contains a function `ellipsoid_method` that has an identical API to the `simplex` method from `simplex.py`, except it also supports a `max_iter` argument to bound the number of iterations. The ellipsoid implementation does not use fixed point arithmetic, so may suffer strong numerical issues for large problems.

### correct_solver.py
The file `correct_solver.py` contains a function for using Scipy to evaluate LP's created using `lp.py`. This is used for testing purposes.

### test_lp.py
The file `test_lp.py` contains several example functions for testing our implementations. It has functions for constructing fractional knapsack, Klee-Minty cube, min-cost flow, and Alvis-Friedmann instances. You need Networkx to import this module.

### analysis.py
The file `analysis.py` contains our experiments. It formulates several LPs, runs each method on them, and reports the times. You need Networkx to run this script.
