"""
Classes for storing objective functions and their associated
gradients.

TODO: possibly look into some sort of symbolic implementation? Sympy?
"""

from inspect import getargspec


class ObjectiveFunction(object):
    """
    Holds an objective function and its gradient
    in a nice interface for the engine to use.

    Objective and gradient will be accessed by the
    get_objective and get_gradient methods,
    such that they can overriden on a subclass if we want
    to do symbolic computations and the like.
    """
    def __init__(self, objective, objective_gradient, variables, starting_values=None):
        """
        Args:
            objective: an objective function f that takes a vector of
                solution values.
            objective_gradient: the gradient of the above objective function
                i.e. grad(f), that gives us information about the objective
                value of changing the solution values.
            variables: the gurobi variables that are in this objective
                function.
            starting_value: a list of lists of values of variables to
                add starting cuts to the model at.
        """

        # Make sure that we fail on objectives with bad arg lists
        # TODO: python 3 compatibility
        obj_spec = getargspec(objective)
        grad_spec = getargspec(objective_gradient)
        if (len(obj_spec.args) + len(obj_spec.varargs or []) !=
                len(grad_spec.args) + len(grad_spec.varargs or [])):
            raise ValueError('Objective and gradient must take the same number of arguments')

        self._objective = objective
        self._objective_gradient = objective_gradient
        self.variables = variables
        self.starting_values = starting_values or []

    def get_objective(self, solution_values):
        return self._objective(*solution_values)

    def get_gradient(self, solution_values):
        return self._objective_gradient(*solution_values)
