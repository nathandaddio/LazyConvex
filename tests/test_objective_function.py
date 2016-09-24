import pytest

from lazyConvex import ObjectiveFunction


@pytest.mark.objective_function
def test_objective_function_validates_bad_arg_lengths():
    def fn(x, y):
        return x + y

    def actual_gradient(x, y):
        return [1, 1]

    def different_argument_gradient(x):
        return [1, 1]

    # Make sure it works with the actual gradient
    ObjectiveFunction(fn, actual_gradient)

    # Make sure it fails on a gradient function with bad args
    with pytest.raises(ValueError):
        ObjectiveFunction(fn, different_argument_gradient)
