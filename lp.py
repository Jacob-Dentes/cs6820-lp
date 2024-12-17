"""
Authors: Jacob Dentes

Holds a class for an LP input
"""
import numpy as np

MAXIMIZE = 1.0
MINIMIZE = -1.0

class InfeasibleException(Exception):
    def __init__(self, msg):
        self.message = msg

class UnboundedException(Exception):
    def __init__(self, msg):
        self.message = msg

class Constraint():
    """
    Represents a constraint in (coeffs * vars <= b) form
    """
    def __init__(self, coeffs, vars, b):
        self.coeffs = coeffs
        self.vars = vars
        self.b = b

def as_expr(value):
    if isinstance(value, float) or isinstance(value, int):
        return Expression([], [], float(value))
    if isinstance(value, Variable):
        return Expression([1.0], [value], 0.0)
    assert isinstance(value, Expression), "Invalid type used as expression"

    return value

class Expression():
    """
    Represents a linear combination of variables
    """
    def __init__(self, coeffs, variables, constant):
        """
        Create an expression that is variables added together where
        variable[i]'s coefficient is coeffs[i]. Constant is an added
        scalar (not multiplied by a variable)

        coeffs: a list of floats
        variables: a list of Variable objects
        constant: a float
        """
        assert len(coeffs) == len(variables)
        self.coeffs = coeffs
        self.variables = variables
        self.constant = constant

    def evaluate(self, assignment):
        """
        Give the value of this expression with variable assignments
        given by assignment

        assignment: an arraylike of all variable values ordered by index
        """
        acc = self.constant
        for i, var in enumerate(self.variables):
            acc += self.coeffs[i] * assignment[var.index]

        return acc

    def __add__(self, other):
        """
        Add an expression to a variable or expression
        """
        other = as_expr(other)
        return Expression(self.coeffs + other.coeffs, self.variables + other.variables, self.constant + other.constant)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        """
        Multiply an expression by a scalar coefficient
        """
        assert (isinstance(other, float) or isinstance(other, int)), "Non-scalar multiplication of expression"

        return Expression([float(other)*c for c in self.coeffs], self.variables, float(other)*self.constant)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __le__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Constraint(self.coeffs, self.variables, other - self.constant)
        return self - as_expr(other) <= 0

    def __ge__(self, other):
        return as_expr(other) <= self 

    def __eq__(self, other):
        return [self <= other, self >= other]
        
class Variable():
    """
    Represents an LP variable
    """
    def __init__(self, index, name=None):
        """
        Creates a new variable
        """
        if name is None:
            name = f"variable_{index}"
        self.index = index
        self.name = name

    def evaluate(self, assignment):
        return as_expr(self).evaluate(assignment)

    def __add__(self, other):
        return as_expr(self) + other

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        return as_expr(self) * other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __le__(self, other):
        return as_expr(self) <= other

    def __ge__(self, other):
        return as_expr(self) >= other

    def __eq__(self, other):
        return as_expr(self) == other

class LP():
    def __init__(self):
        self.variables = []
        self.constraints = []
        self.objective = Expression([], [], 0.0)
        self.sense = MAXIMIZE

    def add_var(self, name=None, nonnegative=True):
        """
        Create a fresh variable to use in constraints

        If nonnegative is True (default) then the variable will be >= 0
        """
        if nonnegative:
            var = Variable(len(self.variables), name)
            self.variables.append(var)
            return var

        var1 = self.add_var()
        var2 = self.add_var()
        return var1 - var2

    def add_constr(self, constr):
        """
        Add a constraint to the LP
        """
        if isinstance(constr, list):
            self.constraints += constr
        else:
            self.constraints.append(constr)

    def set_objective(self, objective, sense):
        """
        Set the objective with sense (MINIMIZE or MAXIMIZE)
        """
        if isinstance(objective, float) or isinstance(objective, int):
            objective = Expression([], [], float(objective))
        if isinstance(objective, Variable):
            objective = Expression([1.0], [objective], 0.0)
        assert isinstance(objective, Expression)
        self.objective = objective
        sense * objective # ensure that sense is valid
        self.sense = sense

    def get_A(self):
        """
        Returns the constraint matrix of the LP as a numpy array.
        Given in Ax <= b, x >= 0 form.
        """
        a = np.zeros((len(self.constraints), len(self.variables)))
        for i, constr in enumerate(self.constraints):
            for j, variable in enumerate(constr.vars):
                a[i, variable.index] += constr.coeffs[j]

        return a

    def get_b(self):
        """
        Returns the RHS scalars of the LP as a numpy array.
        Given in Ax <= b, x >= 0 form
        """
        b = np.zeros(len(self.constraints))
        for i, constr in enumerate(self.constraints):
                b[i] = constr.b

        return b

    def get_c(self):
        """
        Returns the vector c as a numpy array.
        Given in maximize c^T x form
        """
        objective = self.sense * self.objective
        c = np.zeros(len(self.variables))
        for i, var in enumerate(objective.variables):
            c[var.index] += objective.coeffs[i]

        return c

