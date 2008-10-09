## PENDING REWRITE OF tensor_opt.py


import unittest

from theano import gof
from theano.tensor.opt import *
from theano import tensor
from theano.tensor import Tensor
from theano.gof import Env
from theano.tensor.elemwise import DimShuffle
import numpy
#import scalar_opt



def inputs(xbc = (0, 0), ybc = (0, 0), zbc = (0, 0)):
    x = Tensor(broadcastable = xbc, dtype = 'float64')('x')
    y = Tensor(broadcastable = ybc, dtype = 'float64')('y')
    z = Tensor(broadcastable = zbc, dtype = 'float64')('z')
    return x, y, z

# class _test_inplace_opt(unittest.TestCase):

#     def test_straightforward(self):
#         x, y, z = inputs()
#         e = x + y + z
#         g = Env([x, y], [e])
#         self.failUnless(str(g) == "[Broadcast{Add}(Broadcast{Add}(x, y), z)]")
#         inplace_optimizer.optimize(g)
#         self.failUnless(str(g) == "[Broadcast{Add}{0: 0}(Broadcast{Add}{0: 0}(x, y), z)]")

#     def test_multiple_uses(self):
#         x, y, z = inputs()
#         e0 = x + y
#         e1 = x * y
#         g = Env([x, y], [e0, e1])
#         self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}(x, y)]")
#         inplace_optimizer.optimize(g)
#         self.failUnless(str(g) == "[Broadcast{Add}{0: 0}(x, y), Broadcast{Mul}(x, y)]" \
#             or str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, y)]")

#     def test_user_inplace(self):
#         x, y, z = inputs()
#         e0 = x + y
#         e1 = tensor._mul_inplace(x, y)
#         g = Env([x, y], [e0, e1])
#         self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, y)]")
#         inplace_optimizer.optimize(g)
#         self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, y)]")

#     def test_inplace_on_second_argument(self):
#         x, y, z = inputs()
#         e0 = x + y
#         e1 = tensor._mul_inplace(x, z)
#         g = Env([x, y], [e0, e1])
#         self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, z)]")
#         inplace_optimizer.optimize(g)
#         self.failUnless(str(g) == "[Broadcast{Add}{0: 1}(x, y), Broadcast{Mul}{0: 0}(x, z)]")


ds = lambda x, y: DimShuffle(x.type.broadcastable, y)(x)

class test_dimshuffle_lift(unittest.TestCase):

    def test_double_transpose(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 0)), (1, 0))
        g = Env([x], [e])
        self.failUnless(str(g) == "[DimShuffle{1,0}(DimShuffle{1,0}(x))]")
        dimshuffle_lift.optimize(g)
        self.failUnless(str(g) == "[x]")

    def test_merge2(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 'x', 0)), (2, 0, 'x', 1))
        g = Env([x], [e])
        self.failUnless(str(g) == "[DimShuffle{2,0,x,1}(DimShuffle{1,x,0}(x))]", str(g))
        dimshuffle_lift.optimize(g)
        self.failUnless(str(g) == "[DimShuffle{0,1,x,x}(x)]", str(g))

    def test_elim3(self):
        x, y, z = inputs()
        e = ds(ds(ds(x, (0, 'x', 1)), (2, 0, 'x', 1)), (1, 0))
        g = Env([x], [e])
        self.failUnless(str(g) == "[DimShuffle{1,0}(DimShuffle{2,0,x,1}(DimShuffle{0,x,1}(x)))]", str(g))
        dimshuffle_lift.optimize(g)
        self.failUnless(str(g) == "[x]", str(g))

    def test_lift(self):
        x, y, z = inputs([False]*1, [False]*2, [False]*3)
        e = x + y + z
        g = Env([x, y, z], [e])
        self.failUnless(str(g) == "[add(InplaceDimShuffle{x,0,1}(add(InplaceDimShuffle{x,0}(x), y)), z)]", str(g))
        dimshuffle_lift.optimize(g)
        self.failUnless(str(g) == "[add(add(InplaceDimShuffle{x,x,0}(x), InplaceDimShuffle{x,0,1}(y)), z)]", str(g))




from theano.tensor import *

#from sandbox import pprint

class test_greedy_distribute(unittest.TestCase):
    def test_main(self):
        a, b, c, d, x, y, z = matrices('abcdxyz')
        e = (a/z + b/x) * x * z
        g = Env([a,b,c,d,x,y,z], [e])
        ##print pprint.pp.process(g.outputs[0])
        mul_canonizer.optimize(g)
        gof.TopoOptimizer(gof.LocalOptGroup(local_fill_cut, local_fill_lift), order = 'out_to_in').optimize(g)
        gof.TopoOptimizer(gof.LocalOptGroup(local_greedy_distributor), order = 'out_to_in').optimize(g)
        ##print pprint.pp.process(g.outputs[0])
        


class test_canonize(unittest.TestCase):

    def test_muldiv(self):
        x, y, z = matrices('xyz')
        a, b, c, d = matrices('abcd')
#        e = (2.0 * x) / (2.0 * y)
#        e = (2.0 * x) / (4.0 * y)
#        e = x / (y / z)
#        e = (x * y) / x
#        e = (x / y) * (y / z) * (z / x)
#        e = (a / b) * (b / c) * (c / d)
#        e = (a * b) / (b * c) / (c * d)
#        e = 2 * x / 2
#        e = x / y / x
        e = (x / x) * (y / y)
        g = Env([x, y, z, a, b, c, d], [e])
        ##print pprint.pp.process(g.outputs[0])
        mul_canonizer.optimize(g)
        gof.TopoOptimizer(gof.LocalOptGroup(local_fill_cut, local_fill_lift), order = 'out_to_in').optimize(g)
        ##print pprint.pp.process(g.outputs[0])
        
#     def test_plusmin(self):
#         x, y, z = inputs()
#         a, b, c, d = more_inputs()
# #        e = x - x
# #        e = (2.0 + x) - (2.0 + y)
# #        e = (2.0 + x) - (4.0 + y)
# #        e = x - (y - z)
# #        e = (x + y) - x
# #        e = (x - y) + (y - z) + (z - x)
# #        e = (a - b) + (b - c) + (c - d)
# #        e = x + -y
# #        e = a - b - b + a + b + c + b - c
# #        e = x + log(y) - x + y
#         e = 2.0 + x + 4.0
#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         gof.ConstantFinder().optimize(g)
#         addfn = lambda *inputs: sum(inputs)
#         subfn = lambda x, y: x - y
#         negfn = lambda x: -x
#         Canonizer(Add, Sub, Neg, addfn, subfn, negfn).optimize(g)
#         print g

#     def test_both(self):
#         x, y, z = inputs()
#         a, b, c, d = more_inputs()
#         e0 = (x * y / x)
#         e = e0 + e0 - e0
#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         gof.ConstantFinder().optimize(g)
#         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
#         divfn = lambda x, y: x / y
#         invfn = lambda x: 1 / x
#         Canonizer(Mul, Div, Inv, mulfn, divfn, invfn).optimize(g)
#         addfn = lambda *inputs: reduce(lambda x, y: x + y, (0,) + inputs)
#         subfn = lambda x, y: x - y
#         negfn = lambda x: -x
#         Canonizer(Add, Sub, Neg, addfn, subfn, negfn).optimize(g)
#         print g

#     def test_group_powers(self):
#         x, y, z, a, b, c, d = floats('xyzabcd')

###################
#         c1, c2 = constant(1.), constant(2.)
#         #e = pow(x, c1) * pow(x, y) / pow(x, 7.0) # <-- fucked
#         #f = -- moving from div(mul.out, pow.out) to pow(x, sub.out)
#         e = div(mul(pow(x, 2.0), pow(x, y)), pow(x, 7.0))

#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         print g.inputs, g.outputs, g.orphans
#         f = sub(add(2.0, y), add(7.0))
#         g.replace(e, pow(x, f))
#         print g
#         print g.inputs, g.outputs, g.orphans
#         g.replace(f, sub(add(2.0, y), add(7.0))) # -- moving from sub(add.out, add.out) to sub(add.out, add.out)
#         print g
#         print g.inputs, g.outputs, g.orphans
###################

# #        e = x * exp(y) * exp(z)
# #        e = x * pow(x, y) * pow(x, z)
# #        e = pow(x, y) / pow(x, z)
#         e = pow(x, 2.0) * pow(x, y) / pow(x, 7.0) # <-- fucked
# #        e = pow(x - x, y)
# #        e = pow(x, 2.0 + y - 7.0)
# #        e = pow(x, 2.0) * pow(x, y) / pow(x, 7.0) / pow(x, z)
# #        e = pow(x, 2.0 + y - 7.0 - z)
# #        e = x ** y / x ** y
# #        e = x ** y / x ** (y - 1.0)
# #        e = exp(x) * a * exp(y) / exp(z)
#         g = Env([x, y, z, a, b, c, d], [e])
#         g.extend(gof.PrintListener(g))
#         print g, g.orphans
#         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
#         divfn = lambda x, y: x / y
#         invfn = lambda x: 1 / x
#         Canonizer(mul, div, inv, mulfn, divfn, invfn, group_powers).optimize(g)
#         print g, g.orphans
#         addfn = lambda *inputs: reduce(lambda x, y: x + y, (0,) + inputs)
#         subfn = lambda x, y: x - y
#         negfn = lambda x: -x
#         Canonizer(add, sub, neg, addfn, subfn, negfn).optimize(g)
#         print g, g.orphans
#         pow2one_float.optimize(g)
#         pow2x_float.optimize(g)
#         print g, g.orphans
        







# class _test_cliques(unittest.TestCase):

#     def test_straightforward(self):
#         x, y, z = inputs()
#         m = y * z
#         d = tensor.dot(x, m)
#         d.name = 'd'
#         e = x + y + d
#         g = Env([x, y, z], [e])
#         cliques = find_cliques(g)
#         self.failUnless(len(cliques) == 2)
#         (i1, o1), (i2, o2) = cliques
#         self.failUnless(str(Env(i1, o1)) == "[Broadcast{Add}(Broadcast{Add}(x, y), d)]")
#         self.failUnless(str(Env(i2, o2)) == "[Broadcast{Mul}(y, z)]")
# #         print g
# #         for i, o in find_cliques(g):
# #             print "-->", Env(i, [o])

#     def test_broadcasting(self):
#         x, y, z = inputs([0]*1, [0]*2, [0]*3)
#         e = x + y + z
#         g = Env([x, y, z], [e])
#         lift_dimshuffle.optimize(g)
#         self.failUnless(len(find_cliques(g, through_broadcast = True)) == 1)
#         self.failUnless(len(find_cliques(g, through_broadcast = False)) == 2)
# #         print g
# #         for i, o in find_cliques(g, True):
# #             print "-->", Env(i, [o])


# # class _test_clique_opt(unittest.TestCase):

# #     def test_straightforward(self):
# #         x, y, z = inputs()
# #         e = x ** 2.0 #x * x
# #         g = Env([x], [e])
# #         gof.ConstantFinder().optimize(g)
# #         opt = CliqueOptimizer(through_broadcast = False,
# #                               scalar_optimizer = scalar_opt.opt2,
# #                               make_composite = False)
# #         print g
# #         opt.optimize(g)
# #         print g

# #     def test_inplace(self):
# #         x, y, z = inputs()
# #         #e = tensor._add_inplace(x, y + z)
# #         e = x + tensor._add_inplace(y, z)
# #         g = Env([x, y, z], [e])
# #         opt = CliqueOptimizer(through_broadcast = False,
# #                               scalar_optimizer = None,
# #                               make_composite = True)
# #         print g
# #         opt.optimize(g)
# #         print g
# # #        print g.outputs[0].owner.c_code(['x', 'y', 'z'], ['e'], dict(fail = "FAIL;", id = 0))
# #         print gof.OpWiseCLinker(g).make_function()(numpy.ones((5, 5)), numpy.ones((5, 5)), numpy.ones((5, 5)))

# #     def test_straightforward(self):
# #         x, y, z = inputs()
# #         e = x + y + z
# #         g = Env([x, y, z], [e])
# #         opt = CliqueOptimizer(through_broadcast = False,
# #                               scalar_optimizer = None,
# #                               make_composite = True)
# #         print g
# #         opt.optimize(g)
# #         print g
# # #        print g.outputs[0].owner.c_code(['x', 'y', 'z'], ['e'], dict(fail = "FAIL;", id = 0))
# #         print gof.OpWiseCLinker(g).make_function()(numpy.ones((5, 5)), numpy.ones((5, 5)), numpy.ones((5, 5)))

# #     def test_straightforward2(self):
# #         x, y, z = inputs()
# #         m = y * z
# #         d = tensor.dot(x, m)
# #         d.name = 'd'
# #         e = x + y + d
# #         g = Env([x, y, z], [e])
# #         opt = CliqueOptimizer(through_broadcast = False,
# #                               scalar_optimizer = None,
# #                               make_composite = True)
# #         print g
# #         opt.optimize(g)
# #         print g
# # #        print g.outputs[0].owner.c_code(['x', 'y', 'z'], ['e'], dict(fail = "FAIL;", id = 0))
# #         print gof.OpWiseCLinker(g).make_function()(numpy.ones((5, 5)), numpy.ones((5, 5)), numpy.ones((5, 5)))
        
    


if __name__ == '__main__':
    unittest.main()




