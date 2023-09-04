from sympy import (
    AtomicExpr, sympify, Integer, Rational, NumberSymbol, dotprint,
    Basic, Equality, Expr, Derivative, Integral, simplify, collect,
    collect_const, expand, factor, symbols, GreaterThan, sin, lambdify
)
from sympy.printing import latex
from sympy.printing.pretty.stringpict import prettyForm
import mpmath.libmp as mlib
import math
from graphviz import Source
import numpy as np

################################################################################
################################# PLOTTING #####################################
################################################################################

def plot_arrows_direction_from_line(line, N=6, hw=.025, skipfirst=False):
    """ Add uniformly spaced arrows along the direction of a line.
    This is a wrapper function for plot_arrows_direction().
    
    Parameters
    ----------
        line : matplotlib.lines.Line2D
            line object containing the x, y data and the color
        N : int
            Number of arrows to be added
        hw : float
            Head width of the arrows
        skipfirst : boolean
            If True, don't plot the first arrow. Default to False.
    """
    from matplotlib.lines import Line2D
    if isinstance(line, list):
        line = line[0]
    if not isinstance(line, Line2D):
        raise TypeError("`line` must be an instance of matplotlib.lines.Line2D")
    x = line.get_xdata()
    y = line.get_ydata()
    c = line.get_color()
    plot_arrows_direction(x, y, c, N, hw, skipfirst)
    
def plot_arrows_direction(x, y, color, N=6, hw=.025, skipfirst=False):
    """ Add uniformly spaced arrows along the direction of a line.
    
    Parameters
    ----------
        x : np.ndarray
            x-coordinates of the line
        y : np.ndarray
            y-coordinates of the line
        color : int, string, tuple
            Color of the arrows. If:
                + type(color) == int: color is going to be "C" + str(color)
        N : int
            Number of arrows to be added
        hw : float
            Head width of the arrows
        skipfirst : boolean
            If True, don't plot the first arrow. Default to False.
    """
    import matplotlib.pyplot as plt
    if isinstance(color, int):
        color = "C" + str(color)
    dx = x - np.roll(x, -1)
    dy = y - np.roll(y, -1)
    lengths = np.cumsum(np.sqrt(dx**2 + dy**2))
    delta = lengths[-1] / N
    
    if skipfirst:
        _range = range(1, N)
    else:
        _range = range(N)
    for j in _range:
        start_idx = np.where(lengths >= j * delta)[0][0]
        end_idx = start_idx + 5
        adx = x[end_idx] - x[start_idx]
        ady = y[end_idx] - y[start_idx]
        plt.arrow(x[start_idx], y[start_idx], adx, ady, color=color,
              shape='full', lw=0, length_includes_head=True, head_width=hw)


################################################################################
################################# LAMBDIFY #####################################
################################################################################

def get_lambda(expr, modules=["numpy", "scipy"], **kwargs):
    """ Create a lambda function to numerically evaluate expr by sorting 
    alphabetically the function arguments.

    Parameters
    ----------
        expr : Expr
            The Sympy expression to convert.
        modules : str
            The numerical module to use for evaluation. Default to 
            ["numpy", "scipy"]. See help(lambdify) for other choices.
        **kwargs
            Other keyword arguments to the function lambdify.

    Returns
    -------
        s : list
            The function signature: a list of the ordered function arguments.
        f : lambda
            The generated lambda function.ò 
    """
    from sympy.utilities.iterables import ordered
    signature = list(ordered(expr.free_symbols))
    return signature, lambdify(signature, expr, modules=modules, **kwargs)

class SymbolRegistry(dict):
    """ This is an extended dictionary meant to be used in substitution
    operations when our symbols contain Latex syntax. The keys are
    symbols containing Latex syntax; the values are the associated
    symbols without Latex syntax.
    
    The dictionary expose the instance method symbols(*args, **kwargs),
    in which:
    * args can be:
        1. a dictionary of string keys/values used to create symbols.
        2. a list of string keys and a list of string values used
            to create symbols
    * kwargs represent the keyword arguments to specify assumptions to
        symbols.
    This method returns the symbols containing the Latex syntax, so
    that they can be used to create and manipulate expressions.
        
    Examples
    ========
    
    sr = SymbolRegistry()
    d = {
        r"\dot{m}_{H}^{prop}": "mdot_H_prop",
        "T_{H_{2}}^{tank}": "T_H2_tank",
        "c_{p}^{H_{2}}": "cp_H2",
        "Q_{Ne}": "Q_Ne"
    }
    mdot, T, cp, Q = sr.symbols(d)
    display(sr)
    expr = Q / (mdot * cp) + T
    display(expr.subs(sr))
    """
    def symbols(self, *args, **kwargs):
        if len(args) == 1:
            ret_symbols = []
            for k, v in args[0].items():
                s1 = symbols(k, **kwargs)
                s2 = symbols(v, **kwargs)
                self[s1] = s2
                ret_symbols.append(s1)
            if len(ret_symbols) == 1:
                return ret_symbols[0]
            return ret_symbols
        elif len(args) == 2:
            s1 = symbols(args[0], **kwargs)
            s2 = symbols(args[1], **kwargs)
            if len(s1) != len(s2):
                raise ValueError("`s1` and `s2` must produce the same number of symbols.")
            for k, v in zip(s1, s2):
                self[k] = v
            if len(s1) == 1:
                return s1[0]
            return s1
        else:
            raise ValueError("Please, read this function's docstring.")

SR = SymbolRegistry

################################################################################
############################### LATEX PRINTER ##################################
################################################################################

from sympy.printing.latex import LatexPrinter
from sympy.core.function import AppliedUndef
from sympy.printing.conventions import requires_partial

class MyLatexPrinter(LatexPrinter):
    """ Extended Latex printer with two new options:

    1. applied_no_args=False: wheter applied undefined function should be
        rendered with their arguments. Default to False.
    2. der_as_subscript=False: wheter derivatives of undefined function should
        be rendered as a subscript, for example df/dx=(f)_x.
        Default to False.
    """
    def __init__(self, settings=None):
        self._default_settings.update({
                "applied_no_args": False,
                "der_as_subscript": False,
            }
        )
        super().__init__(settings)
        
    def _print_Function(self, expr, exp=None):
        if isinstance(expr, AppliedUndef) and self._settings["applied_no_args"]:
            if exp is None:
                return expr.func.__name__
            else:
                return r'%s^{%s}' % (expr.func.__name__, exp)
        return super()._print_Function(expr, exp)
    
    def _print_Derivative(self, expr):
        if requires_partial(expr.expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = r'd'

        tex = []
        dim = 0
        for x, num in expr.variable_count:
            dim += num
            if num == 1:
                tex.append(r"%s" % self._print(x))
            else:
                tex.append(r"%s^{%s}" % (self.parenthesize_super(self._print(x)),
                                        self._print(num)))
        
        if (isinstance(expr.expr, AppliedUndef) 
            and self._settings["applied_no_args"]
                and self._settings["der_as_subscript"]):
            tex = ",".join(tex)
            return self._print_Function(expr.expr) + r"_{%s}" % tex
        
        
        tex = diff_symbol + " " + (diff_symbol + " ").join(reversed(tex))
        if (isinstance(expr.expr, AppliedUndef) 
            and self._settings["applied_no_args"]):
            if dim == 1:
                return r"\frac{%s %s}{%s}" % (diff_symbol, self._print_Function(expr.expr), tex)
            else:
                return r"\frac{%s^{%s} %s}{%s}" % (diff_symbol, self._print(dim), self._print_Function(expr.expr), tex)
        
        return super()._print_Derivative(expr)

def my_latex(expr, **settings):
    """ Convert the given expression to LaTeX string representation using
    the MyLatexPrinter class, which exposes two further options:

    1. applied_no_args=False: wheter applied undefined function should be
        rendered with their arguments. Default to False.
    2. der_as_subscript=False: wheter derivatives of undefined function should
        be rendered as a subscript, for example df/dx=(f)_x.
        Default to False.
    """
    return MyLatexPrinter(settings).doprint(expr)


################################################################################
############################### FUNCTIONS ######################################
################################################################################

from sympy import Function, Number, frac, pi, sstr, S

class SawtoothWave(Function):
    """ Create a symbolic sawtooth function based on the following
    definition:
    https://mathworld.wolfram.com/SawtoothWave.html
    """
    def __new__(cls, x, A=1, T=1, phi=0):
        x, A, T, phi = [S(t) for t in [x, A, T, phi]]
        r = cls.eval(x, A, T, phi)
        if r is None:
            if not isinstance(x, Symbol):
                raise TypeError("x must be an instance of Symbol")
            obj = Basic.__new__(cls, x, A, T, phi)
            return obj
        return r
    
    @staticmethod
    def _func(x, A, T, phi):
        return A * frac(x / T + (phi / (2 * pi)))
    
    @classmethod
    def eval(cls, x, A, T, phi):
        if isinstance(x, Number):
            return cls._func(x, A, T, phi)
    
    @property
    def period(self):
        return self.args[2]
    
    @property
    def phase(self):
        return self.args[3]
    
    @property
    def amplitude(self):
        return self.args[1]
    
    def _eval_subs(self, old, new):
        args = list(self.args)
        for i, a in enumerate(args):
            args[i] = a._subs(old, new)
        return self.func(*args)
    
    def _latex(self, printer, *args):
        return r"\left({}\right)".format(printer._print(self._func(*self.args)))
    
    def __repr__(self):
        x, A, T, phi = self.args
        return self.func.__name__ + "(x={}, A={}, T={}, phi={})".format(x, A, T, phi)
    
    def __str__(self):
        return sstr(self, order=None)
    
    def _getcode(self, module, printer):
        x, A, T, phi = self.args
        return '({} * {}({}))[0]'.format(
            printer._print(A),
            printer._module_format(module),
            printer._print(x / T + (phi / (2 * pi))))
    
    def _numpycode(self, printer, *args):
        return self._getcode("numpy.modf", printer)
    
    def _pythoncode(self, printer, *args):
        return self._getcode("math.modf", printer)
    
    def _mpmathcode(self, printer, *args):
        x, A, T, phi = self.args
        return '{} * {}({}, 1)'.format(
            printer._print(A),
            printer._module_format("mpmath.fmod"),
            printer._print(x / T + (phi / (2 * pi))))



################################################################################
############################## CONSTANT NUMBERS ################################
################################################################################

class Constant(NumberSymbol):
    """ Represent a generic integer or float constant: it will be treaded as a 
    symbol during symbolic computations, whereas it will be converted to a 
    number during numerical evaluation.

    Examples
    ========

    t = Constant(2.5, r"\tau")
    display(t, t.evalf(), t + 2)
    """
    is_real = True

    def __new__(cls, value, name, latex="", pretty=""):
        """
        Parameters
        ----------
            value : float
                The numerical value of the constant
            name : string
                Used to render the symbol when calling print()
            latex (optional) : string
                Latex code representing representing this constant. If not
                provided, `name` will be used instead.
            pretty (optional) : string
                Used to render the symbol when calling pprint(). Unicode strings
                are admissible. If not provided, `name` will be used instead.
        """
        if isinstance(value, Integer):
            value = value.p
        if not isinstance(value, (int, float)):
            raise TypeError("'value' must be a Python's int or float. \n" +
                "Instead, got {}".format(type(value)))
        if not all([isinstance(a, str) for a in [name, latex, pretty]]):
            raise TypeError("Parameters name, latex, pretty must be of type string")
        
        obj = AtomicExpr.__new__(cls)
        obj._value = value
        obj._name = name
        obj._latex_str = latex
        obj._pretty_str = pretty
        return obj

    def _as_mpf_val(self, prec):
        return mlib.from_float(self._value, prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(math.floor(self._value)), Integer(math.ceil(self._value)))
        elif issubclass(number_cls, Rational):
            pass
    
    def _latex(self, printer):
        if self._latex_str:
            return self._latex_str
        return self._name
    
    def _sympyrepr(self, printer, *args):
        return (self.func.__name__ + 
            "(value={}, name='{}', latex='{}', pretty='{}')".format(
                self._value, self._name, self._latex_str, self._pretty_str
            ))

    def _sympystr(self, printer, *args):
        return self._name

    def _pretty(self, printer, *args):
        if printer._use_unicode and self._pretty_str:
            return prettyForm(self._pretty_str)
        return prettyForm(self._name)


################################################################################
################################## UTILITIES ###################################
################################################################################

def render_tree(expr, filename, format="png"):
    """ Show the expression tree. This function saves in the current folder two
    files associated to `expr`:
        1. filename.gv: contains the DOT representation.
        2. filename.png: the DOT representation has been rendered and saved into 
            this image.

    Parameters
    ----------
        expr : the symbolic expression
        filename : string
            The name of the generated files.
        format : string
            The format of the output image. Default to "png".
    """
    if not isinstance(filename, str):
        raise TypeError("'filename' must be of type str. " + 
            "Instead, got: {}".format(type(filename)))
    s = Source(dotprint(expr), filename=filename + ".gv", format=format)
    s.view()


################################################################################
###################### LINEARIZATION - SERIES EXPANSION ########################
################################################################################

from sympy import Dummy, degree
def linearize(expr, order=1, tup=None, n=3, apply_lin=True):
    """ Get a linear approximation of a non-linear function.

    Parameters
    ==========

    expr : Expr
        The Sympy expression to linearize.
    
    order : int
        An optional value up to which the series is to be expanded 
        for each undefined function. The default value is 1.
    
    tup : list
        an optional list of tuples in the form(f(x), f0) where the first element
        is the undefined applied function and the second element is the value
        around which f(x) is calculated. If tup is not provided, the function
        will extract from expr the undefined functions and perform the
        expansions around the value 0.
    
    n : int
        Represents the actual order used by the series expansion. Default to 3
        to speed up the process.
    
    apply_lin : bool
        This keyword controls the linearization; if set to False, the expression
        is only expanded and not linearized. In such a case, `order` will not be
        representative of the expression. Default to True.
    """
    if n < order:
        n = order
    if not tup:
        from sympy.core.function import AppliedUndef
        tup = []
        funcs = list(expr.find(AppliedUndef))
        for f in funcs:
            tup.append((f, 0))

    subs_dict = dict()
    for t in tup:
        f, f0 = t
        s = Dummy(f.func.name)
        expr = expr.subs(f, s)
        subs_dict[s] = f
        expr = expr.series(s, f0, n).removeO()

    if apply_lin:
        expr = expr.expand()
        get_degree = lambda expr, symbols: sum([degree(expr, gen=s) for s in symbols])
        args = [a for a in expr.args if get_degree(a, list(subs_dict.keys())) <= order]
        expr = expr.func(*args)
    expr = expr.subs(subs_dict)
    return expr
