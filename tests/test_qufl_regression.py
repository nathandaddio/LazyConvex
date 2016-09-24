import random

import pytest

import gurobipy

from lazyConvex import LazyConvexEngine, ObjectiveFunction

TOLERANCE = 10**(-5)

@pytest.mark.regression
def test_qufl_regression():
    """
    Generates an instance of the quadratic uncapacitated facility
    location problem (qUFL), and a gurobi model to solve it.

    Uses the Kochetov and Ivanenko cost structure.
    """
    facilities = range(7)

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
        [qufl_model.addVar(vtype=gurobipy.GRB.BINARY) for facility in facilities]
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

    # TODO: generate lazy convex model here, solve

    # Need to set the model back to an unoptimised state
    # so that we get rid of the old solve information
    qufl_model.reset()

    # Since we're doing convex outer approximation
    # on the quadratic terms, we reset the objective to
    # just be the linear facility cost terms
    qufl_model.setObjective(total_facility_cost)

    from itertools import chain

    objective_vars = list(chain(*is_assigned))

    def obj_fn(*args):
        return sum(cost * arg**2 for cost, arg in zip(list(chain(*assignment_costs)), args))

    def grad_fn(*args):
        return [2 * cost * arg for cost, arg in zip(list(chain(*assignment_costs)), args)]

    objective_fn = ObjectiveFunction(obj_fn, grad_fn)

    qufl_lazy_model = LazyConvexEngine(qufl_model, objective_fn, objective_vars)

    qufl_lazy_model.optimize()

    assert abs(naive_objective - qufl_lazy_model.objVal) < TOLERANCE
