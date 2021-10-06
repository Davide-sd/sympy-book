from sympy import (
    AtomicExpr, sympify, Integer, Rational, NumberSymbol, dotprint,
    Basic, Equality, Expr, Derivative, Integral, simplify, collect,
    collect_const, expand, factor, symbols, GreaterThan, sin, lambdify
)
from sympy.core.add import _unevaluated_Add, Add
from sympy.core.decorators import _sympifyit
from sympy.core.evalf import EvalfMixin
from sympy.core.relational import Relational
from sympy.printing import latex
from sympy.printing.pretty.stringpict import prettyForm
import mpmath.libmp as mlib
import math
from graphviz import Source
import unittest as ut
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

################################################################################
############################## LAPLACE TRANSFORM ###############################
################################################################################

from sympy.core.function import AppliedUndef
from sympy import Symbol, Derivative, Eq, laplace_transform, LaplaceTransform
def laplace_transform_ode(expr, subs=True):
    """ Apply the Laplace Transform to an expression and rewrite it in an 
    algebraic form.

    Parameters
    ----------
        expr : Expr
            The Sympy expression to transform.
        subs : boolean
            Default value to True. If True, substitute L[x(t)](s) with X.
    
    Returns
    -------
        t : Symbols
            Symbol representing the original function domain.
        t : Symbols
            Symbol representing the s-domain.
        transf_expr : Expr
            The Laplace-transformed expression.
    """
    s = Symbol("s")
    
    # perform the laplace transform of a derivative an unknown function
    def laplace_transform_derivative(lap):
        der = lap.args[0]
        f, (t, n) = der.args
        return s**n * laplace_transform(f, t, s) - sum([s**(n - i) * f.diff(t, i - 1).subs(t, 0) for i in range(1, n + 1)])
    
    # extract the original function domain
    derivs = expr.atoms(Derivative)
    funcs = set().union(*[d.atoms(AppliedUndef) for d in derivs])
    func = funcs.pop()
    t = func.args[0]
    
    # apply the Laplace transform to the expression
    if isinstance(expr, Eq):
        expr = expr.func(*[laplace_transform(a, t, s, noconds=True) for a in expr.args])
    else:
        expr = laplace_transform(expr, t, s)
    
    # select the unevaluated LaplaceTransform objects
    laps = expr.atoms(LaplaceTransform)
    for lap in laps:
        # if the unevaluated LaplaceTransform contains a derivative
        if lap.atoms(Derivative):
            # perform the transformation according to the rule
            transform = laplace_transform_derivative(lap)
            expr = expr.subs(lap, transform)
    
    # substitute unevaluated LaplaceTransform with a symbol
    if subs:
        laps = expr.atoms(LaplaceTransform)
        for lap in laps:
            name = lap.args[0].func.name.capitalize()
            expr = expr.subs(lap, Symbol(name))
        
    return t, s, expr

################################################################################
################################# EQUATION #####################################
################################################################################

class Equation(Basic, EvalfMixin):
    """ Represent an equation, relating together two expressions, the
    left-hand side and the right-hand side. Objects of type Equation
    support mathematical operations.

    Examples
    ========

    x, y, t = symbols("x, y, t")
    eq1 = Equation(x**2 + x, y**2 + y)
    eq2 = Equation(t**2, t + 1)
    display(eq1 + eq2)
    # x**2 + x + t**2 = y**2 + y + t + 1

    display(eq1 / eq2)
    # (x**2 + x) / t**2 = (y**2 + y) / (t + 1)

    eq1.applyfunc(sin)
    # sin(x**2 + x) = sin(y**2 + y)

    eq1.dolhs(lambda t: t.collect(x))
    # x * (x + 1) = y**2 + y

    eq2.dorhs(lambda t: t.collect(y))
    # x**2 + x = y * (y + 1)

    eq1.diff(x)
    # 2 * x = 0

    eq1.integrate(x)
    # x**3 / 3 + x**2 / 2 = (y**2 + y) + x
    """

    def __new__(cls, lhs, rhs=None):
        lhs = sympify(lhs)
        rhs = sympify(rhs)
        
        if isinstance(rhs, Relational):
            raise TypeError("Right hand side cannot be a Relational.")
        if isinstance(lhs, Relational) and rhs:
            raise TypeError("When left hand side is a Relational, right hand side must be None.")
        if not rhs:
            if isinstance(lhs, Relational):
                lhs, rhs = lhs.args
            else:
                rhs = sympify(0)
        if not (isinstance(lhs, Expr) and isinstance(rhs, Expr)):
            raise TypeError("`lhs` and `rhs` must be instances of the class Expr, so that mathematical operations can be applied.")
        return Basic.__new__(cls, lhs, rhs)
    
    @classmethod
    def _binary_op(cls, a, b, func):
        if isinstance(a, cls) and not isinstance(b, cls):
            return cls(func(a.lhs, b), func(a.rhs, b))
        elif isinstance(b, cls) and not isinstance(a, cls):
            return cls(func(a, b.lhs), func(a, b.rhs))
        elif isinstance(a, cls) and isinstance(b, cls):
            return cls(func(a.lhs, b.lhs), func(a.rhs, b.rhs))
        else:
            raise TypeError('One of a or b should be an equation')
    
    def __add__(self, other):
        return self._binary_op(self, other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary_op(other, self, lambda a, b: a + b)

    def __mul__(self, other):
        return self._binary_op(self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary_op(other, self, lambda a, b: a * b)

    def __sub__(self, other):
        return self._binary_op(self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary_op(other, self, lambda a, b: a - b)

    def __truediv__(self, other):
        return self._binary_op(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary_op(other, self, lambda a, b: a / b)

    def __mod__(self, other):
        return self._binary_op(self, other, lambda a, b: a % b)

    def __rmod__(self,other):
        return self._binary_op(other, self, lambda a, b: a % b)

    def __pow__(self, other):
        return self._binary_op(self, other, lambda a, b: a ** b)

    def __rpow__(self, other):
        return self._binary_op(other, self, lambda a, b: a ** b)
    
    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]

    @property
    def reversed(self):
        return self.func(self.rhs, self.lhs)
    
    def as_expr(self):
        return self.args[0] - self.args[1]
    
    def as_relational(self, cls_relational=Equality):
        if not issubclass(cls_relational, Relational):
            raise TypeError("`cls_relational` must be a sub-class of Relational")
        return cls_relational(self.lhs, self.rhs)

    def _eval_applyfunc(self, func, *args, **kwargs):
        lhs, rhs = self.args
        side = kwargs.pop("side", "both")
        if side == "both":
            if isinstance(func, str) and hasattr(lhs, func) and hasattr(rhs, func): # methods like doit(), subs(), xreplace, ...
                return self.func(getattr(lhs, func)(*args, **kwargs), getattr(rhs, func)(*args, **kwargs))
            elif hasattr(lhs, "applyfunc") and hasattr(rhs, "applyfunc"):
                return self.func(lhs.applyfunc(func, *args, **kwargs), rhs.applyfunc(func, *args, **kwargs))
            return self.func(func(lhs, *args, **kwargs), func(rhs, *args, **kwargs))
        elif side == "left":
            if isinstance(func, str) and hasattr(lhs, func):
                return self.func(getattr(lhs, func)(*args, **kwargs), rhs)
            elif hasattr(lhs, "applyfunc"):
                return self.func(lhs.applyfunc(func, *args, **kwargs), rhs)
            return self.func(func(lhs, *args, **kwargs), rhs)
        elif side == "right":
            if isinstance(func, str) and hasattr(rhs, func):
                return self.func(lhs, getattr(rhs, func)(*args, **kwargs))
            elif hasattr(lhs, "applyfunc"):
                return self.func(lhs, rhs.applyfunc(func, *args, **kwargs))
            return self.func(lhs, func(rhs, *args, **kwargs))
        return self

    def applyfunc(self, func, *args, **kwargs):
        """
        If either side of the equation has a defined subfunction (attribute) of name ``func``, that will be applied
        instead of the global function. The operation is applied to both sides.
        """
        if not callable(func):
            raise TypeError("`func` must be callable.")
        return self._eval_applyfunc(func, *args, **kwargs)
    
    def expand(self, *args, **kwargs):
        return self.applyfunc(expand, *args, **kwargs)

    def simplify(self, *args, **kwargs):
        return self.applyfunc(simplify, *args, **kwargs)

    def factor(self, *args, **kwargs):
        return self.applyfunc(factor, *args, **kwargs)

    def collect(self, *args, **kwargs):
        return self.applyfunc(collect, *args, **kwargs)
   
    def collect_const(self, *args, **kwargs):
        return self.applyfunc(collect_const, *args, **kwargs)
    
    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False, side="both"):
        options = {'subs':subs, 'maxn':maxn, 'chop':chop, 'strict':strict,
                'quad':quad, 'verbose':verbose}
        return self._applyfunc(lambda i: i.evalf(n, **options), side=side)
    
    def diff(self, *args, **kwargs):
        return self._eval_diff_integrate(Derivative, *args, **kwargs)
    
    def _eval_diff_integrate(self, Opcls, *args, **kwargs):
        evaluate = kwargs.pop("evaluate", True)
        lhs, rhs = self.args
        lhs = Opcls(lhs, *args)
        rhs = Opcls(rhs, *args)
        return self._eval_side(lhs, rhs, **kwargs)

    def _eval_derivative(self, s):
        return self.func(self.lhs.diff(s), self.rhs.diff(s))
    
    def _eval_Integral(self, *symbols, **assumptions):
        return self.func(self.lhs.integrate(*symbols, **assumptions), self.rhs.integrate(*symbols, **assumptions))
    
    def _eval_side(self, lhs, rhs, **kwargs):
        return self.func(lhs, rhs)._eval_applyfunc("doit", **kwargs)
        
    def doit(self, **kwargs):
        lhs, rhs = self.args
        return self._eval_side(lhs, rhs, **kwargs)
    
    def do(self, func):
        return self.applyfunc(func)

    def dolhs(self, func):
        return self.applyfunc(func, side="left")

    def dorhs(self, func):
        return self.applyfunc(func, side="right")
    
    def integrate(self, *args, **kwargs):
        return self._eval_diff_integrate(Integral, *args, **kwargs)

    def subs(self, *args, **kwargs):
        new_args = args
        if len(args) == 1 and isinstance(args[0], self.func):
            new_args = args[0].args
        return self._eval_applyfunc("subs", *new_args, **kwargs)

    def xreplace(self, rule, **kwargs):
        if isinstance(rule, self.func):
            rule = {rule.lhs: rule.rhs}
        return self._eval_applyfunc("xreplace", rule, **kwargs)
    
    def _eval_rewrite_as_Add(self, *args, **kwargs):
        """return Equation(L, R) as L - R. To control the evaluation of
        the result set pass `evaluate=True` to give L - R;
        if `evaluate=None` then terms in L and R will not cancel
        but they will be listed in canonical order; otherwise
        non-canonical args will be returned.
        Examples
        ========
        >>> from sympy import Eq, Add
        >>> from sympy.abc import b, x
        >>> eq = Eq(x + b, x - b)
        >>> eq.rewrite(Add)
        2*b
        >>> eq.rewrite(Add, evaluate=None).args
        (b, b, x, -x)
        >>> eq.rewrite(Add, evaluate=False).args
        (b, x, b, -x)
        """
        L, R = args
        evaluate = kwargs.get('evaluate', True)
        if evaluate:
            # allow cancellation of args
            return L - R
        args = Add.make_args(L) + Add.make_args(-R)
        if evaluate is None:
            # no cancellation, but canonical
            return _unevaluated_Add(*args)
        # no cancellation, not canonical
        return Add._from_args(args)

    def __repr__(self):
        return str(self.lhs) + " = " + str(self.rhs)
        
    def _latex(self, printer):
        return (latex(self.lhs) + "=" + latex(self.rhs))
