from lp import LP, MAXIMIZE, MINIMIZE, InfeasibleException
from simplex import simplex
from ellipsoid import ellipsoid_method
from correct_solver import solve
import random
import numpy as np
import networkx as nx

def simple_example():
    program = LP()

    x1 = program.add_var("x1")
    x2 = program.add_var("x2")

    program.add_constr(x1 + 5 <= x2)

    x3 = program.add_var("x3")
    program.add_constr(1*x2 + x2 - 3 == 2 + x3 * 0.5)

    program.set_objective(9 * x1 - 2*x2 + x3, MAXIMIZE)

    print(program.get_A())
    print(program.get_b())
    print(program.get_c())
    print("")

def example_knapsack():
    knapsack_lp = LP()
    knapsack_values = [1, 3, 2]
    knapsack_weights = [1, 2, 3]
    knapsack_W = 4

    x = {i: knapsack_lp.add_var(f"x{i}") for i in range(3)}
    for i in x:
        knapsack_lp.add_constr(x[i] <= 1)

    knapsack_lp.add_constr(sum(knapsack_weights[i] * x[i] for i in x) <= knapsack_W)
    knapsack_lp.set_objective(sum(knapsack_values[i] * x[i] for i in x), MAXIMIZE)

    A = knapsack_lp.get_A()
    b = knapsack_lp.get_b()
    c = knapsack_lp.get_c()

    print(A)
    print(b)
    print(c)
    print("")

    sol = simplex(A, b, c)
    print(f"Simplex Knapsack Solution: {[x[i].evaluate(sol) for i in x]}")
    print(f"Value: {knapsack_lp.objective.evaluate(sol)}")

    try:
        e_sol = ellipsoid_method(A, b, c)
        print(f"Ellipsoid Knapsack Solution: {[x[i].evaluate(e_sol) for i in x]}")
        print(f"Value: {knapsack_lp.objective.evaluate(e_sol)}")
    except InfeasibleException:
        print("Ellipsoid failed on this knapsack")
        

    c_sol = solve(A, b, c)
    print(f"Scipy Knapsack Solution: {[x[i].evaluate(c_sol) for i in x]}")
    print(f"Value: {knapsack_lp.objective.evaluate(c_sol)}")

    for i in x:
        knapsack_lp.add_constr(x[i] >= 0.4)

    sol2 = simplex(knapsack_lp.get_A(), knapsack_lp.get_b(), knapsack_lp.get_c())
    c_sol2 = solve(knapsack_lp.get_A(), knapsack_lp.get_b(), knapsack_lp.get_c())
    print(f"Modified Example Knapsack Solution: {[x[i].evaluate(sol2) for i in x]}")
    print(f"Value: {knapsack_lp.objective.evaluate(sol2)}")
    print(f"Scipy Modified Example Knapsack Solution: {[x[i].evaluate(c_sol2) for i in x]}")
    print(f"Value: {knapsack_lp.objective.evaluate(c_sol2)}")

def formulate_knapsack(knapsack_values, knapsack_weights, knapsack_W):
    knapsack_lp = LP()

    x = {i: knapsack_lp.add_var(f"x{i}") for i in range(len(knapsack_values))}
    for i in x:
        knapsack_lp.add_constr(x[i] <= 1)

    knapsack_lp.add_constr(sum(knapsack_weights[i] * x[i] for i in x) <= knapsack_W)
    knapsack_lp.set_objective(sum(knapsack_values[i] * x[i] for i in x), MAXIMIZE)

    return x, knapsack_lp

def formulate_uniformly_random_knapsack(n=5):
    knapsack_values = [random.randint(0, 100) for i in range(n)]
    knapsack_weights = [random.randint(0, 100) for i in range(n)]
    knapsack_W = random.randint(0, 100)

    return formulate_knapsack(knapsack_values, knapsack_weights, knapsack_W)

def formulate_gaussian_knapsack(n=5):
    # resample a normal distribution until values positive
    def positive_normal(loc, scale, n):
        ret_arr = np.random.normal(loc, scale, n)
        mask = ret_arr <= 0
        while mask.sum() > 0:
            ret_arr[mask] = np.random.normal(loc, scale, mask.sum())
            mask = ret_arr <= 0
        return np.round(ret_arr, 2)

    knapsack_values = positive_normal(50, 10, n)
    knapsack_weights = positive_normal(50, 10, n)
    knapsack_W = positive_normal(50 * n / 2, 10 * n / 2, 1)[0]

    return formulate_knapsack(knapsack_values, knapsack_weights, knapsack_W)


def example_random_knapsack(method=formulate_uniformly_random_knapsack):
    x, knapsack_lp = method(5)

    A = knapsack_lp.get_A()
    b = knapsack_lp.get_b()
    c = knapsack_lp.get_c()

    print(A)
    print(b)
    print(c)
    print("")

    sol = simplex(A, b, c)
    print(f"Simplex Knapsack Solution: {[x[i].evaluate(sol) for i in x]}")
    print(f"Value: {knapsack_lp.objective.evaluate(sol)}")

    try:
        e_sol = ellipsoid_method(A, b, c)
        print(f"Ellipsoid Knapsack Solution: {[x[i].evaluate(e_sol) for i in x]}")
        print(f"Value: {knapsack_lp.objective.evaluate(e_sol)}")
    except InfeasibleException:
        print("Ellipsoid failed on this knapsack")

    c_sol = solve(A, b, c)
    print(f"Scipy Knapsack Solution: {[x[i].evaluate(c_sol) for i in x]}")
    print(f"Value: {knapsack_lp.objective.evaluate(c_sol)}")

def formulate_random_mcfp(n = 5, m = 10, capacity_range = (1, 11), demand_range = (1, 6), cost_range = (1, 6)):
    mcfp_lp = LP()
    mcfp_values = np.random.randint(cost_range[0], cost_range[1], m)
    x = {i: mcfp_lp.add_var(f"x{i}") for i in range(m)}
    mcfp_lp.set_objective(sum(mcfp_values[i] * x[i] for i in x), MINIMIZE)
    d = np.random.randint(demand_range[0], demand_range[1])

    G = nx.gnm_random_graph(n, m, directed=True)
    for i, (u, v) in enumerate(G.edges()):
        mcfp_lp.add_constr(x[i] <= np.random.randint(capacity_range[0], capacity_range[1]))
        mcfp_lp.add_constr(x[i] >= 0)
    for i,  node in enumerate(G.nodes()):
        if i == 0: # source node
            node_out = []
            for j, (u, v) in enumerate(G.edges()):
                if u == node:
                    node_out.append(j)
            mcfp_lp.add_constr(sum(x[i] for i in node_out) == d)
        elif i == n - 1: # sink node
            node_in = []
            for j, (u, v) in enumerate(G.edges()):
                if v == node:
                    node_in.append(j)
            mcfp_lp.add_constr(sum(x[i] for i in node_in) == d)
        else:
            node_in = []
            node_out = []
            for j, (u, v) in enumerate(G.edges()):
                if v == node:
                    node_in.append(j)
                elif u == node:
                    node_out.append(j)
            mcfp_lp.add_constr(sum(x[i] for i in node_in) == sum(x[i] for i in node_out))

    return x, mcfp_lp

def example_random_mcfp():
    x, mcfp_lp = formulate_random_mcfp()
    print(mcfp_lp.get_A())
    print(mcfp_lp.get_b())
    print(mcfp_lp.get_c())
    print("")

    sol = simplex(mcfp_lp.get_A(), mcfp_lp.get_b(), mcfp_lp.get_c())
    print(f"Example MCFP Solution: {[x[i].evaluate(sol) for i in x]}")
    print(f"Value: {mcfp_lp.objective.evaluate(sol)}")
    c_sol = solve(mcfp_lp.get_A(), mcfp_lp.get_b(), mcfp_lp.get_c())
    print(f"Scipy MCFP Solution: {[x[i].evaluate(c_sol) for i in x]}")
    print(f"Value: {mcfp_lp.objective.evaluate(c_sol)}")


def formulate_kleeminty_cube(n=5):
    kleemintycube_lp = LP()
    kleemintycube_values = [2 ** (n - 1 - i) for i in range(n)]
    x = {i: kleemintycube_lp.add_var(f"x{i}") for i in range(n)}
    for i in x:
        kleemintycube_weights = [2 ** j for j in range(i + 1, 1, -1)] + [1] + ([0] * (n - i - 1))
        kleemintycube_lp.add_constr(sum(kleemintycube_weights[i] * x[i] for i in x) <= 5 ** (i+1))
        kleemintycube_lp.add_constr(x[i] >= 0)
    kleemintycube_lp.set_objective(sum(kleemintycube_values[i] * x[i] for i in x), MAXIMIZE)

    return x, kleemintycube_lp
    
def example_kleemintycube():
    x, kleemintycube_lp = formulate_kleeminty_cube(5)

    A = kleemintycube_lp.get_A()
    b = kleemintycube_lp.get_b()
    c = kleemintycube_lp.get_c()
    print(A)
    print(b)
    print(c)
    print("")

    sol = simplex(A, b, c)
    print(f"Simplex Klee Minty Cube Solution: {[x[i].evaluate(sol) for i in x]}")
    print(f"Value: {kleemintycube_lp.objective.evaluate(sol)}")

    try:
        e_sol = ellipsoid_method(A, b, c, tolerance=1e-30, max_iter=10_000)
        print(f"Ellipsoid Knapsack Solution: {[x[i].evaluate(e_sol) for i in x]}")
        print(f"Value: {kleemintycube_lp.objective.evaluate(e_sol)}")
    except InfeasibleException:
        print("Ellipsoid failed on kleeminty cube")

    c_sol = solve(A, b, c)
    print(f"Scipy Klee Minty Cube Solution: {[x[i].evaluate(c_sol) for i in x]}")
    print(f"Value: {kleemintycube_lp.objective.evaluate(c_sol)}")


if __name__ == '__main__':
    example_knapsack()
    example_random_knapsack()
    example_random_knapsack(formulate_gaussian_knapsack)
    example_kleemintycube()
