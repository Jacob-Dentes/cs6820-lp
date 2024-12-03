from lp import LP, MAXIMIZE

program = LP()

x1 = program.add_var("x1")
x2 = program.add_var("x2")

program.add_constr(x1 + 5 <= x2)

x3 = program.add_var("x3")
program.add_constr(2*x2 - 3 == 2 + x3 * 0.5)

program.set_objective(9 * x1 - 2*x2 + x3, MAXIMIZE)

print(program.get_A())
print(program.get_b())
print(program.get_c())
