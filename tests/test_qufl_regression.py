import pytest

import gurobipy


@pytest.fixture
def qufl_model_and_data():
    """
    Generates an instance of the quadratic uncapacitated facility
    location problem (qUFL), and a gurobi model to solve it
    """
    facility_costs = []
