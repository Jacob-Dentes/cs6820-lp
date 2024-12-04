from lp import LP, MAXIMIZE
from simplex import simplex

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

knapsack_lp = LP()
knapsack_values = [1, 3, 2]
knapsack_weights = [1, 2, 3]
knapsack_W = 4

x = {i: knapsack_lp.add_var(f"x{i}") for i in range(3)}
for i in x:
    knapsack_lp.add_constr(x[i] <= 1)

knapsack_lp.add_constr(sum(knapsack_weights[i] * x[i] for i in x) <= knapsack_W)
knapsack_lp.set_objective(sum(knapsack_values[i] * x[i] for i in x), MAXIMIZE)

print(knapsack_lp.get_A())
print(knapsack_lp.get_b())
print(knapsack_lp.get_c())
print("")

sol = simplex(knapsack_lp.get_A(), knapsack_lp.get_b(), knapsack_lp.get_c())
print(f"Example Knapsack Solution: {[x[i].evaluate(sol) for i in x]}")
print(f"Value: {knapsack_lp.objective.evaluate(sol)}")

for i in x:
    knapsack_lp.add_constr(x[i] >= 0.4)

sol2 = simplex(knapsack_lp.get_A(), knapsack_lp.get_b(), knapsack_lp.get_c())
print(f"Modified Example Knapsack Solution: {[x[i].evaluate(sol2) for i in x]}")
print(f"Value: {knapsack_lp.objective.evaluate(sol2)}")
