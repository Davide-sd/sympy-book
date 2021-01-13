import unittest as ut
from sympy import (
    var, Mul, Equality, GreaterThan, pi, sin, cos, S, Interval, 
    Derivative, Integral, root, symbols, Add
)
from sympy_utils import Equation

class test_Equation(ut.TestCase):
    def setUp(self):
        var("x:z")
    
    def _do_test(self, eq, lhs, rhs):
        self.assertIsInstance(eq, Equation)
        self.assertEqual(eq.lhs, lhs)
        self.assertEqual(eq.rhs, rhs)

    def test_creation(self):
        eq1 = Equation(x * y, y + z)
        self._do_test(eq1, x * y, y + z) 

        equality = Equality(x * y, y + z)
        eq2 = Equation(equality)
        self._do_test(eq2, equality.lhs, equality.rhs) 

        eq3 = Equation(x * y)
        self._do_test(eq3, x * y, 0)
        
        def func(lhs, rhs):
            with self.assertRaises(TypeError):
                Equation(lhs, rhs)
        
        interval = Interval(5, 10)
        for i in [(equality, x * y), (equality, equality), (interval, 2)]:
            func(*i)
    
    def test_operations(self):
        eq = Equation(x * y, y + z)
        self._do_test(eq + 2, x * y + 2, y + z + 2)
        self._do_test(2 + eq, x * y + 2, y + z + 2)
        self._do_test(eq - 2, x * y - 2, y + z - 2)
        self._do_test(2 - eq, 2 - x * y, 2 - (y + z))
        self._do_test(eq * 2, x * y * 2, (y + z) * 2)
        self._do_test(2 * eq, x * y * 2, (y + z) * 2)
        self._do_test(eq / 2, x * y / 2, (y + z) / 2)
        self._do_test(2 / eq, 2 / (x * y), 2 / (y + z))
        self._do_test(eq ** 2, (x * y) ** 2, (y + z) ** 2)
        self._do_test(2 ** eq, 2 ** (x * y), 2 ** (y + z))
        self._do_test(eq % 2, (x * y) % 2, (y + z) % 2)
        self._do_test(2 % eq, 2 % (x * y), 2 % (y + z))

        self._do_test(eq + eq, 2 * x * y, 2 * y + 2 * z)
        self._do_test(eq - eq, 0, 0)
        self._do_test(eq * eq, (x * y) ** 2, (y + z) ** 2)
        self._do_test(eq / eq, 1, 1)
        self._do_test(eq ** eq, (x * y) ** (x * y), (y + z) ** (y + z))
        self._do_test(eq % eq, 0, 0)
    
    def test_reversed(self):
        eq = Equation(x * y, y + z)
        self.assertEqual(eq.reversed, Equation(y + z, x * y))
    
    def test_convert(self):
        eq = Equation(x * y, y + z)
        self.assertEqual(eq.as_expr(), x * y - (y + z))
        r1 = eq.as_relational()
        self.assertTrue(isinstance(r1, Equality) and (r1 == Equality(x * y, y + z)))
        r2 = eq.as_relational(GreaterThan)
        self.assertTrue(isinstance(r2, GreaterThan) and (r2 == GreaterThan(x * y, y + z)))

    def test_applyfunc(self):
        eq1 = Equation(x * y, y + z)
        eq2 = eq1.applyfunc(sin, side="both")
        eq3 = eq1.applyfunc(sin, side="left")
        eq4 = eq1.applyfunc(sin, side="right")
        with self.assertRaises(TypeError):
            eq1.applyfunc("sin", side="none")
        self.assertTrue((eq2.lhs == sin(x * y)) and (eq2.rhs == sin(y + z)))
        self.assertTrue((eq3.lhs == sin(x * y)) and (eq3.rhs == y + z))
        self.assertTrue((eq4.lhs == x * y) and (eq4.rhs == sin(y + z)))

    def test_diff(self):
        eq1 = Equation(x + x**2, x**3 + x**4)
        eq2 = eq1.diff(x)
        eq3 = eq1.diff(x, side="left")
        eq4 = eq1.diff(x, side="right")
        eq5 = eq1.diff(x, side="")
        self.assertEqual(eq2, Equation(1 + 2 * x, 3 * x**2 + 4 * x**3))
        self.assertEqual(eq3, Equation(1 + 2 * x, Derivative(x**3 + x**4)))
        self.assertEqual(eq4, Equation(Derivative(x + x**2), 3 * x**2 + 4 * x**3))
        self.assertEqual(eq5, Equation(Derivative(x + x**2), Derivative(x**3 + x**4)))

    def test_integrate(self):
        eq1 = Equation(x + x**2, x**3 + x**4)
        eq2 = eq1.integrate(x)
        eq3 = eq1.integrate(x, side="left")
        eq4 = eq1.integrate(x, side="right")
        eq5 = eq1.integrate(x, side="")
        self.assertEqual(eq2, Equation(x**2 / 2 + x**3 / 3, x**4 / 4 + x**5 / 5))
        self.assertEqual(eq3, Equation(x**2 / 2 + x**3 / 3, Integral(x**3 + x**4)))
        self.assertEqual(eq4, Equation(Integral(x + x**2), x**4 / 4 + x**5 / 5))
        self.assertEqual(eq5, Equation(Integral(x + x**2), Integral(x**3 + x**4)))

    def test_doit(self):
        eq1 = Equation(Derivative(x + x**2), Derivative(x**3 + x**4))
        eq2 = eq1.doit()
        eq3 = eq1.doit(side="left")
        eq4 = eq1.doit(side="right")
        eq5 = eq1.doit(side="")
        self.assertEqual(eq2, Equation(1 + 2 * x, 3 * x**2 + 4 * x**3))
        self.assertEqual(eq3, Equation(1 + 2 * x, Derivative(x**3 + x**4)))
        self.assertEqual(eq4, Equation(Derivative(x + x**2), 3 * x**2 + 4 * x**3))
        self.assertEqual(eq5, eq1)
    
    def test_subs(self):
        x, y, z, t = symbols("x:z, t", real=True, positive=True)
        expr1 = -pi * x**3 * (S(1) / 6 + y / 4 / x) + z

        eq1 = Equation(expr1, expr1)
        eq2 = eq1.subs(x**3, t)
        eq3 = eq1.subs(x**3, t, side="left")
        eq4 = eq1.subs(x**3, t, side="right")
        eq5 = eq1.subs(x**3, t, side="")
        expr2 = -pi * t * (S(1) / 6 + y / 4 / root(t, 3)) + z
        self.assertEqual(eq2, Equation(expr2, expr2))
        self.assertEqual(eq3, Equation(expr2, expr1))
        self.assertEqual(eq4, Equation(expr1, expr2))
        self.assertEqual(eq5, eq1)

        eq6 = Equation(x + y, x * y)
        eq7 = Equation(x, y**2)
        eq8 = eq6.subs(eq7)
        self.assertEqual(eq8, Equation(y + y**2, y**3))
    
    def test_xreplace(self):
        x, y, z, t = symbols("x:z, t", real=True, positive=True)
        expr1 = -pi * x**3 * (S(1) / 6 + y / 4 / x) + z

        eq1 = Equation(expr1, expr1)
        eq2 = eq1.xreplace({x**3: t})
        eq3 = eq1.xreplace({x**3: t}, side="left")
        eq4 = eq1.xreplace({x**3: t}, side="right")
        eq5 = eq1.xreplace({x**3: t}, side="")
        expr2 = -pi * t * (S(1) / 6 + y / 4 / x) + z
        self.assertEqual(eq2, Equation(expr2, expr2))
        self.assertEqual(eq3, Equation(expr2, expr1))
        self.assertEqual(eq4, Equation(expr1, expr2))
        self.assertEqual(eq5, eq1)
    
    def test_as_expr(self):
        left = x + x**2
        right = cos(x)
        eq = Equation(left, right)
        self.assertEqual(eq.as_expr(), left - right)
    
    def test_as_relational(self):
        left = x + x**2
        right = cos(x)
        eq = Equation(left, right)
        self.assertEqual(eq.as_relational(), Equality(left, right))
        self.assertEqual(eq.as_relational(GreaterThan), GreaterThan(left, right))

    def test_do(self):
        expr1 = pi * x**3 / 6 + pi * x**2 * y / 4
        expr2 = pi * x**3 * (S(1) / 6 + y / 4 / x)
        eq1 = Equation(expr1, expr1)
        eq2 = eq1.do(lambda t: (expr1.factor() / x**3).expand().collect(pi) * x**3)
        eq3 = eq1.dolhs(lambda t: (expr1.factor() / x**3).expand().collect(pi) * x**3)
        eq4 = eq1.dorhs(lambda t: (expr1.factor() / x**3).expand().collect(pi) * x**3)
        self.assertEqual(eq2, Equation(expr2, expr2))
        self.assertEqual(eq3, Equation(expr2, expr1))
        self.assertEqual(eq4, Equation(expr1, expr2))

    def test_rewrite(self):
        expr1 = pi * x**3 / 6 + pi * x**2 * y / 4
        expr2 = y
        eq1 = Equation(expr1, expr1)
        eq2 = Equation(expr1, expr2)
        eq3 = eq1.rewrite(Add)
        eq4 = eq2.rewrite(Add)
        self.assertEqual(eq3, 0)
        self.assertEqual(eq4, expr1 - expr2)

if __name__ == "__main__":
    ut.main()