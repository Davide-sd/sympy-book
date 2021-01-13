# Symbolic Computation with SymPy

This repository contains the necessary Jupyter Notebooks and code to follow the book **"Symbolic Computation with Python and [SymPy](https://github.com/sympy/sympy/)"**, which can be purchased at:

* Printed version from [Amazon](https://www.amazon.com/dp/B08QWBY5WV/ref=sr_1_1?dchild=1&keywords=Symbolic+Computation+with+Python+and+SymPy&qid=1608389036&sr=8-1): it should be available in all amazon markets (COM, DE, IT, UK, ...).
* Ebook coming soon.

<a href="assets/cover.jpg"><img src="assets/cover.jpg" width=400/></a>

Please, read the [preview](assets/book-preview.pdf) to understand what this book offers and who should read it.

## Notebooks

The repository contains a list of notebooks, each one named after a chapter. Notebooks only contain the code: they are meant to be followed by Readers. The explanations for the code can be found in the book.

Some chapters offers an *Advanced Topics* section, which is meant to be optional but highly recommended.

**The chapters containing exercises are not included in this repository**: this is done intentionally to force the Readers to actively follow the book, code their attempts and understand the different steps, hence gaining experience with SymPy.

## Code

The following code files are included in the repository, which are going to be used by some notebooks:

* `sympy_utils.py`: it contains functions and classes that can be useful to any SymPy user. Take a look at the `README.md` file contained in the `sympy_utils_module` folder to see a list of available functions/classes.
* `test_equation.py`: it contains test cases for the `Equation` class, included in `sympy_utils.py`.

## Module

The function and classes contained in `sympy_utils.py` can also be useful in our every-day problems. Hence, a package has been created (the `sympy_utils_module` folder), which can be easily installed on our systems:

* Open a terminal and move into the `sympy_utils_module` folder.
* `pip3 install .`