# lazyConvex
Uses Convex outer approximation and dynamic constraint generation within branch and bound to solve convex mixed-integer nonlinear programming problems with Gurobi.
Specifically solves problems of the form
```
min c^T * y + f(x)
s.t. Ax + By <= b
x >= 0, y in Y
```
Where `f(x)` is convex and non-negative (or a sum of such functions), the variables `x` are continuous, and the variables `y` are mixed-integer.
## Usage
Formulate your ``model`` as usual in Gurobipy, but leave out the `f(x)` term. Create `ObjectiveFunction` objects for them, of the form
```python
objective = ObjectiveFunction(fn, grad_fn, gurobi_vars)
```
where ``fn`` is a python function representing objective term, ``grad_fn`` is the gradient (or subgradient) of the function ``fn``, and ``gurobi_vars`` is a list of the specific gurobi variables that this function should be evaluated at.
Then, create an instance of the ``LazyConvexEngine`` and call ``optimize()`` to solve the problem.
```python
engine = LazyConvexEngine(model, [objective])
engine.optimize()
```
This will then utilise Gurobi's lazy constraints to dynamically approximate the value of the objective function.
