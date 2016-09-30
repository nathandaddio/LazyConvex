# LazyConvex
Uses convex outer approximation and dynamic constraint generation within
branch and bound to solve convex mixed-integer nonlinear programming problems
with Gurobi and Python.

Specifically solves problems of the form
```
min c^T * y + f(x)
s.t. Ax + By <= b
x >= 0, y in Y
```
where `f(x)` is convex and non-negative (or a sum of such functions), the
variables `x` are continuous, and the variables `y` are mixed-integer.

The basic principle is to approximate `f(x)` with supporting hyperplanes of the form
```
f(x) >= f(a) + grad(f)(a) * (x-a)
```
where `a` is a solution to the problem. Note that these constraints
are linear.

We solve a mixed-integer linear
programming problem (MILP) and dynamically add these constraints as solutions
are found, using Gurobi's lazy constraint functionality.

The key performance benefit is that we solve MILPs, instead of mixed-integer
convex programming problems.
The extra time spent dynamically adding constraints should hopefully
be offset by the increased efficiency of solving MILPs.
Likewise, modern solvers typically have a plethora
of clever techniques they use when solving MILPs, that they may not
have for convex problems.

## Usage
Formulate your ``model`` as usual in Gurobipy, but leave out the `f(x)` term.
Create `ObjectiveFunction` objects for them, of the form
```python
objective = ObjectiveFunction(fn, grad_fn, gurobi_vars)
```
where ``fn`` is a python function representing the objective term, ``grad_fn`` is the gradient (or subgradient) of the function ``fn``, and ``gurobi_vars`` is a list of the specific gurobi variables that this function should be evaluated at. ``fn`` and ``grad_fn`` should take the values of the ``gurobi_vars`` as its argument.
Then, create an instance of the ``LazyConvexEngine`` and call ``optimize()`` to solve the problem.
```python
engine = LazyConvexEngine(model, [objective])
engine.optimize()
```
This will then utilise Gurobi's lazy constraints to dynamically approximate the value of the objective function.

Has further functionality around adding starting cuts and warm starting at the root node.
Also handles some of Gurobi's issues around posting heuristics to
ensure that the best objective is updated properly.
