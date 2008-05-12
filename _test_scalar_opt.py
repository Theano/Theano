## PENDING REWRITE OF scalar_opt.py

#  unittest

# from gof import Result, Op, Env, modes
# import gof

# from scalar import *
# from scalar_opt import *


# def inputs():
#     return floats('xyz')

# def more_inputs():
#     return floats('abcd')


# class _test_opts(unittest.TestCase):

#     def test_pow_to_sqr(self):
#         x, y, z = floats('xyz')
#         e = x ** 2.0
#         g = Env([x], [e])
#         assert str(g) == "[pow(x, 2.0)]"
#         pow2sqr_float.optimize(g)
#         assert str(g) == "[sqr(x)]"


# class _test_canonize(unittest.TestCase):

# #     def test_muldiv(self):
# #         x, y, z = inputs()
# #         a, b, c, d = more_inputs()
# # #        e = (2.0 * x) / (2.0 * y)
# # #        e = (2.0 * x) / (4.0 * y)
# # #        e = x / (y / z)
# # #        e = (x * y) / x
# # #        e = (x / y) * (y / z) * (z / x)
# # #        e = (a / b) * (b / c) * (c / d)
# # #        e = (a * b) / (b * c) / (c * d)
# # #        e = 2 * x / 2
# #         e = x / y / x
# #         g = Env([x, y, z, a, b, c, d], [e])
# #         print g
# #         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
# #         divfn = lambda x, y: x / y
# #         invfn = lambda x: 1 / x
# #         Canonizer(mul, div, inv, mulfn, divfn, invfn).optimize(g)
# #         print g
        
# #     def test_plusmin(self):
# #         x, y, z = inputs()
# #         a, b, c, d = more_inputs()
# # #        e = x - x
# # #        e = (2.0 + x) - (2.0 + y)
# # #        e = (2.0 + x) - (4.0 + y)
# # #        e = x - (y - z)
# # #        e = (x + y) - x
# # #        e = (x - y) + (y - z) + (z - x)
# # #        e = (a - b) + (b - c) + (c - d)
# # #        e = x + -y
# # #        e = a - b - b + a + b + c + b - c
# # #        e = x + log(y) - x + y
# #         e = 2.0 + x + 4.0
# #         g = Env([x, y, z, a, b, c, d], [e])
# #         print g
# #         gof.ConstantFinder().optimize(g)
# #         addfn = lambda *inputs: sum(inputs)
# #         subfn = lambda x, y: x - y
# #         negfn = lambda x: -x
# #         Canonizer(Add, Sub, Neg, addfn, subfn, negfn).optimize(g)
# #         print g

# #     def test_both(self):
# #         x, y, z = inputs()
# #         a, b, c, d = more_inputs()
# #         e0 = (x * y / x)
# #         e = e0 + e0 - e0
# #         g = Env([x, y, z, a, b, c, d], [e])
# #         print g
# #         gof.ConstantFinder().optimize(g)
# #         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
# #         divfn = lambda x, y: x / y
# #         invfn = lambda x: 1 / x
# #         Canonizer(Mul, Div, Inv, mulfn, divfn, invfn).optimize(g)
# #         addfn = lambda *inputs: reduce(lambda x, y: x + y, (0,) + inputs)
# #         subfn = lambda x, y: x - y
# #         negfn = lambda x: -x
# #         Canonizer(Add, Sub, Neg, addfn, subfn, negfn).optimize(g)
# #         print g

# #     def test_group_powers(self):
# #         x, y, z, a, b, c, d = floats('xyzabcd')

# ###################
# #         c1, c2 = constant(1.), constant(2.)
# #         #e = pow(x, c1) * pow(x, y) / pow(x, 7.0) # <-- fucked
# #         #f = -- moving from div(mul.out, pow.out) to pow(x, sub.out)
# #         e = div(mul(pow(x, 2.0), pow(x, y)), pow(x, 7.0))

# #         g = Env([x, y, z, a, b, c, d], [e])
# #         print g
# #         print g.inputs, g.outputs, g.orphans
# #         f = sub(add(2.0, y), add(7.0))
# #         g.replace(e, pow(x, f))
# #         print g
# #         print g.inputs, g.outputs, g.orphans
# #         g.replace(f, sub(add(2.0, y), add(7.0))) # -- moving from sub(add.out, add.out) to sub(add.out, add.out)
# #         print g
# #         print g.inputs, g.outputs, g.orphans
# ###################

# # #        e = x * exp(y) * exp(z)
# # #        e = x * pow(x, y) * pow(x, z)
# # #        e = pow(x, y) / pow(x, z)
# #         e = pow(x, 2.0) * pow(x, y) / pow(x, 7.0) # <-- fucked
# # #        e = pow(x - x, y)
# # #        e = pow(x, 2.0 + y - 7.0)
# # #        e = pow(x, 2.0) * pow(x, y) / pow(x, 7.0) / pow(x, z)
# # #        e = pow(x, 2.0 + y - 7.0 - z)
# # #        e = x ** y / x ** y
# # #        e = x ** y / x ** (y - 1.0)
# # #        e = exp(x) * a * exp(y) / exp(z)
# #         g = Env([x, y, z, a, b, c, d], [e])
# #         g.extend(gof.PrintListener(g))
# #         print g, g.orphans
# #         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
# #         divfn = lambda x, y: x / y
# #         invfn = lambda x: 1 / x
# #         Canonizer(mul, div, inv, mulfn, divfn, invfn, group_powers).optimize(g)
# #         print g, g.orphans
# #         addfn = lambda *inputs: reduce(lambda x, y: x + y, (0,) + inputs)
# #         subfn = lambda x, y: x - y
# #         negfn = lambda x: -x
# #         Canonizer(add, sub, neg, addfn, subfn, negfn).optimize(g)
# #         print g, g.orphans
# #         pow2one_float.optimize(g)
# #         pow2x_float.optimize(g)
# #         print g, g.orphans
        


# if __name__ == '__main__':
#     unittest.main()

