"""
The engine to run lazy convex outer approximation.

TODO: disaggregation, warm start, heuristic posting, starting cuts
"""

import traceback

import gurobipy

TOLERANCE = 10**(-5)

# Store away some of these nasty callback codes
MIPSOL = gurobipy.GRB.Callback.MIPSOL
MIPNODE = gurobipy.GRB.Callback.MIPNODE
MIPNODE_STATUS = gurobipy.GRB.Callback.MIPNODE_STATUS
NODE_COUNT = gurobipy.GRB.Callback.MIPNODE_NODCNT

MIPNODE_BEST_SOLUTION = gurobipy.GRB.Callback.MIPNODE_OBJBST
MIPSOL_OBJ = gurobipy.GRB.Callback.MIPSOL_OBJ


# Lets us not care about gurobi's inconsistent variable value
# getting interface.
def get_value_lookup(model, where):
    if where == MIPSOL:
        return model.cbGetSolution
    elif where == MIPNODE:
        return model.cbGetNodeRel
    else:
        raise ValueError('{} is not a valid callback code'.format(where))


class LazyConvexEngine(object):
    def __init__(self, model, objective_functions,
                 run_warm_start=True, use_heuristic=True):
        """
        Args:
            model: the gurobipy model, missing the convex objective terms
            objective_functions: a list of ObjectiveFunction objects that
                are the missing convex objective terms
            warm_start: runs a warm start at the root node to generate
                further cuts on the LP relaxation of the model
            use_heuristic: whether or not to post heuristic solutions
                as a hack to deal with some of the issues gurobi has with
                lazy constraints.

        """
        self._model = model
        self._model.setParam('LazyConstraints', 1)
        self._objective_functions = objective_functions
        self._run_warm_start = run_warm_start
        self._use_heuristic = use_heuristic

        self._model_variables = self._model.getVars()

        self._best_solution = gurobipy.GRB.INFINITY
        self._best_solution_values = None

        self._approximation_variables = self._add_approximation_variables()
        self._model.update()

        self._starting_cuts = self._add_starting_cuts()

    def _add_approximation_variables(self):
        return {
            objective_function:
                self._model.addVar(name="convex_approximation_variable_{}".format(num), obj=1.0)
            for num, objective_function in enumerate(self._objective_functions)
        }

    def _add_starting_cuts(self):
        return {
            (objective_function, num):
                self._add_approximation(
                    model=self._model,
                    approximation_variable=self._approximation_variables[objective_function],
                    objective_function=objective_function,
                    actual_value=objective_function.get_objective(starting_values),
                    objective_variable_values=starting_values,
                    constraint_adder=self._model.addConstr
                )
            for objective_function in self._objective_functions
            for num, starting_values in enumerate(objective_function.starting_values)
        }

    def optimize(self):
        # Hackily make a partial function here because
        # gurobi is very picky about its callback interface
        def callback(model, where):
            try:
                self._approximation_callback(model, where)
            except:
                traceback.print_exc()
                raise

        self._model.optimize(callback)

    def _approximation_callback(self, model, where):
        at_mip_sol = where == MIPSOL
        at_mip_node = where == MIPNODE and model.cbGet(MIPNODE_STATUS) == 2
        at_root_node = (self._run_warm_start and
                        at_mip_node and
                        model.cbGet(NODE_COUNT) == 0)

        if at_mip_node and self._use_heuristic:
            if model.cbGet(MIPNODE_BEST_SOLUTION) > self._best_solution:
                model.cbSet(self._model_variables, self._best_solution_values)
                return

        if at_mip_sol or at_root_node:
            for objective_function in self._objective_functions:
                approximation_variable = self._approximation_variables[objective_function]
                values = self._get_values(approximation_variable, objective_function, model, where)
                approximation_value, objective_variable_values, actual_value = values

                # Check for a new best solution to save it away
                if at_mip_sol:
                    # Need to calculate the actual objective by removing the approximation
                    # values
                    actual_objective = (model.cbGet(MIPSOL_OBJ) -
                                        approximation_value +
                                        actual_value)
                    if actual_objective < self._best_solution:
                        self._best_solution = actual_objective
                        self._best_solution_values = model.cbGetSolution(self._model_variables)

                if actual_value - approximation_value > TOLERANCE:
                    self._add_approximation(
                        model, approximation_variable, objective_function, actual_value,
                        objective_variable_values
                    )

    def _get_values(self, approximation_variable, objective_function, model, where):
        get_value = get_value_lookup(model, where)
        approximation_value = get_value(approximation_variable)
        objective_variable_values = get_value(objective_function.variables)
        actual_value = objective_function.get_objective(objective_variable_values)

        return approximation_value, objective_variable_values, actual_value

    def _add_approximation(self, model, approximation_variable, objective_function,
                           actual_value, objective_variable_values,
                           constraint_adder=None):
        """
        Adds convex outer approximation of the form
        f(x) >= f(a) + grad(f)(a) * (x-a)
        """
        gradient = objective_function.get_gradient(objective_variable_values)

        if constraint_adder is None:
            constraint_adder = model.cbLazy

        return constraint_adder(
            approximation_variable >=
            actual_value +
            gurobipy.quicksum(
                grad * (var - value)
                for grad, var, value in zip(
                    gradient,
                    objective_function.variables,
                    objective_variable_values
                )
            )
        )

    @property
    def objVal(self):
        return self._model.objVal
