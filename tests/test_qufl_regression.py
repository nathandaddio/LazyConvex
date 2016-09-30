import random

import pytest

import gurobipy

from lazyConvex import LazyConvexEngine, ObjectiveFunction

TOLERANCE = 10**(-5)


def gap_within_tolerance(actual, other):
    return abs(actual-other) / actual < TOLERANCE


@pytest.mark.regression
@pytest.mark.parametrize(
    "num_facilities", [5, 10, 20]
)
def test_qufl_regression(num_facilities):
    """
    Generates an instance of the quadratic uncapacitated facility
    location problem (qUFL), and a gurobi model to solve it.

    Uses the Kochetov and Ivanenko cost structure.
    """
    facilities = range(num_facilities)

    facility_costs = [3000 for facility in facilities]
    assignment_costs = [
        [random.randint(0, 10000) for facility in facilities]
        for facility in facilities
    ]

    qufl_model = gurobipy.Model('qUFL Test')

    is_facility = [
        qufl_model.addVar(vtype=gurobipy.GRB.BINARY)
        for facility in facilities
    ]

    is_assigned = [
        [qufl_model.addVar() for facility in facilities]
        for facility in facilities
    ]

    qufl_model.update()

    total_facility_cost = gurobipy.quicksum(
        facility_costs[facility] * is_facility[facility]
        for facility in facilities
    )

    assignment_cost = gurobipy.quicksum(
        assignment_costs[facility][other_facility] *
        is_assigned[facility][other_facility] *
        is_assigned[facility][other_facility]
        for facility in facilities
        for other_facility in facilities
    )

    qufl_model.setObjective(total_facility_cost + assignment_cost)

    must_be_assigned_constraints = {
        facility:
            qufl_model.addConstr(
                gurobipy.quicksum(
                    is_assigned[facility][other_facility]
                    for other_facility in facilities
                ) == 1
            )
        for facility in facilities
    }

    assign_only_if_facility_constraints = {
        (facility, other_facility):
            qufl_model.addConstr(
                is_assigned[facility][other_facility] <= is_facility[other_facility]
            )
        for facility in facilities
        for other_facility in facilities
    }

    qufl_model.optimize()

    naive_objective = qufl_model.objVal

    # Need to set the model back to an unoptimised state
    # so that we get rid of the old solve information
    qufl_model.reset()

    # Since we're doing convex outer approximation
    # on the quadratic terms, we reset the objective to
    # just be the linear facility cost terms
    qufl_model.setObjective(total_facility_cost)

    from itertools import chain

    def get_obj_fn(assignment_cost):
        def f(x):
            return assignment_cost * x ** 2
        return f

    def get_grad_fn(assignment_cost):
        def g(x):
            return [2 * assignment_cost * x]
        return g

    # Get an objective function for each assignment variable to completely
    # disaggregate the problem
    objective_fns = [
        ObjectiveFunction(
            get_obj_fn(assign_cost), get_grad_fn(assign_cost), [objective_var], [[0.5]]
        )
        for objective_var, assign_cost in
        zip(list(chain(*is_assigned)), list(chain(*assignment_costs)))
    ]

    qufl_lazy_model = LazyConvexEngine(qufl_model, objective_fns)

    qufl_lazy_model.optimize()

    assert gap_within_tolerance(naive_objective, qufl_lazy_model.objVal)

    assert gap_within_tolerance(qufl_lazy_model._best_solution, qufl_lazy_model._model.objVal)
