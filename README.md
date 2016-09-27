# lazyConvex
Uses Convex outer approximation and lazy constraints to solve convex MINLPs with Gurobi.

Useful for solving problems of the form
```
min c^T * y + f(x)
s.t. Ax + By <= b
x, y >= 0, y in Z
```
Where f(x) is convex and non-negative.
