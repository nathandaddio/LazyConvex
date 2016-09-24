"""
The engine to run lazy convex outer approximation.

TODO: disaggregation, warm start, heuristic posting
"""

import traceback

import gurobipy

TOLERANCE = 10**(-5)


class LazyConvexEngine(object):
    def __init__(self, model, objective_function, objective_variables):
        self._model = model
        self._model.setParam('LazyConstraints', 1)
        self._objective_function = objective_function
        self._objective_variables = objective_variables

        self._approximation_variable = self._add_approximation_variable()

    def _add_approximation_variable(self):
        return self._model.addVar(
            name="convex_approximation_variable",
            obj=1.0
        )

    def optimize(self):
        # Hackily make a partial function here because
        # gurobi is very picky about its callback interface
        def callback(model, where):
            try:
                self._approximation_callback(model, where)
            except:
                traceback.print_exc()

        self._model.optimize(callback)

    def _approximation_callback(self, model, where):
        if where == gurobipy.GRB.Callback.MIPSOL:
            approximation_value = model.cbGetSolution(self._approximation_variable)

            objective_variable_values = model.cbGetSolution(self._objective_variables)
            actual_value = self._objective_function.get_objective(objective_variable_values)

            if actual_value - approximation_value > TOLERANCE:
                self._add_approximation(model, actual_value, objective_variable_values)

    def _add_approximation(self, model, actual_value, objective_variable_values):
        """
        Adds convex outer approximation of the form
        f(x) >= f(a) + grad(f)(a) * (x-a)
        """
        gradient = self._objective_function.get_gradient(objective_variable_values)
        model.cbLazy(
            self._approximation_variable >=
            actual_value +
            gurobipy.quicksum(
                grad * (var - value)
                for grad, var, value in zip(
                    gradient,
                    self._objective_variables,
                    objective_variable_values
                )
            )
        )

    @property
    def objVal(self):
        return self._model.objVal
