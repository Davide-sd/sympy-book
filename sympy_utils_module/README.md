# Sympy Utils

This module contains a few functions and classes built for the book "Symbolic Computation with SymPy", which can also be useful on our every-day problems. Here is a list:

* `get_lambda`: create a lambda function to numerically evaluate a symbolic expressions by sorting alphabetically the function arguments.
* `render_tree`: shows the expression tree.
* `linearize`: returns a linear approximation of a non-linear function.
* `laplace_transform_ode`: apply the Laplace Transform to an expression and rewrite it in an algebraic form.
* `my_latex`: converts the given symbolic expression to LaTeX string representation using the `MyLatexPrinter` class.
* `SawtoothWave`: this class represent a symbolic Saw Tooth Wave.
* `Constant`: represent a generic integer or float constant: it will be treaded as a symbol during symbolic computations, whereas it will be converted to a number during numerical evaluation.
* `Equation`: represents an equation, relating together two expressions, the left-hand side and the right-hand side.

## Requirements

* `sympy`, at least version 1.7.0
* `numpy`
* `matplotlib`
* `mpmath`
* `graphviz`

## Installation

1. Open a terminal and move into the module folder, `sympy_utils_module`.
2. `pip3 install .`
