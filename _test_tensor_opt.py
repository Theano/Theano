
import unittest

import gof
from tensor_opt import *
import tensor
from tensor import Tensor
from gof import Env
from elemwise import DimShuffle
import numpy
import scalar_opt


def inputs(xbc = (0, 0), ybc = (0, 0), zbc = (0, 0)):
    x = Tensor(broadcastable = xbc, dtype = 'float64')('x')
    y = Tensor(broadcastable = ybc, dtype = 'float64')('y')
    z = Tensor(broadcastable = zbc, dtype = 'float64')('z')
    return x, y, z

ds = DimShuffle

class _test_inplace_opt(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = x + y + z
        g = Env([x, y], [e])
        self.failUnless(str(g) == "[Broadcast{Add}(Broadcast{Add}(x, y), z)]")
        inplace_optimizer.optimize(g)
        self.failUnless(str(g) == "[Broadcast{Add}{0: 0}(Broadcast{Add}{0: 0}(x, y), z)]")

    def test_multiple_uses(self):
        x, y, z = inputs()
        e0 = x + y
        e1 = x * y
        g = Env([x, y], [e0, e1])
        self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}(x, y)]")
        inplace_optimizer.optimize(g)
        self.failUnless(str(g) == "[Broadcast{Add}{0: 0}(x, y), Broadcast{Mul}(x, y)]" \
            or str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, y)]")

    def test_user_inplace(self):
        x, y, z = inputs()
        e0 = x + y
        e1 = tensor.mul_inplace(x, y)
        g = Env([x, y], [e0, e1])
        self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, y)]")
        inplace_optimizer.optimize(g)
        self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, y)]")

    def test_inplace_on_second_argument(self):
        x, y, z = inputs()
        e0 = x + y
        e1 = tensor.mul_inplace(x, z)
        g = Env([x, y], [e0, e1])
        self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, z)]")
        inplace_optimizer.optimize(g)
        self.failUnless(str(g) == "[Broadcast{Add}{0: 1}(x, y), Broadcast{Mul}{0: 0}(x, z)]")


class _test_dimshuffle_lift(unittest.TestCase):

    def test_double_transpose(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 0)), (1, 0))
        g = Env([x], [e])
        self.failUnless(str(g) == "[InplaceDimShuffle{1,0}(InplaceDimShuffle{1,0}(x))]")
        lift_dimshuffle.optimize(g)
        self.failUnless(str(g) == "[x]")

    def test_merge2(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 'x', 0)), (2, 0, 'x', 1))
        g = Env([x], [e])
        self.failUnless(str(g) == "[InplaceDimShuffle{2,0,x,1}(InplaceDimShuffle{1,x,0}(x))]", str(g))
        lift_dimshuffle.optimize(g)
        self.failUnless(str(g) == "[InplaceDimShuffle{0,1,x,x}(x)]", str(g))

    def test_elim3(self):
        x, y, z = inputs()
        e = ds(ds(ds(x, (0, 'x', 1)), (2, 0, 'x', 1)), (1, 0))
        g = Env([x], [e])
        self.failUnless(str(g) == "[InplaceDimShuffle{1,0}(InplaceDimShuffle{2,0,x,1}(InplaceDimShuffle{0,x,1}(x)))]", str(g))
        lift_dimshuffle.optimize(g)
        self.failUnless(str(g) == "[x]", str(g))

    def test_lift(self):
        x, y, z = inputs([0]*1, [0]*2, [0]*3)
        e = x + y + z
        g = Env([x, y, z], [e])
        self.failUnless(str(g) == "[Broadcast{Add}(InplaceDimShuffle{x,0,1}(Broadcast{Add}(InplaceDimShuffle{x,0}(x), y)), z)]", str(g))
        lift_dimshuffle.optimize(g)
        self.failUnless(str(g) == "[Broadcast{Add}(Broadcast{Add}(InplaceDimShuffle{x,x,0}(x), InplaceDimShuffle{x,0,1}(y)), z)]", str(g))


class _test_cliques(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        m = y * z
        d = tensor.dot(x, m)
        d.name = 'd'
        e = x + y + d
        g = Env([x, y, z], [e])
        cliques = find_cliques(g)
        self.failUnless(len(cliques) == 2)
        (i1, o1), (i2, o2) = cliques
        self.failUnless(str(Env(i1, o1)) == "[Broadcast{Add}(Broadcast{Add}(x, y), d)]")
        self.failUnless(str(Env(i2, o2)) == "[Broadcast{Mul}(y, z)]")
#         print g
#         for i, o in find_cliques(g):
#             print "-->", Env(i, [o])

    def test_broadcasting(self):
        x, y, z = inputs([0]*1, [0]*2, [0]*3)
        e = x + y + z
        g = Env([x, y, z], [e])
        lift_dimshuffle.optimize(g)
        self.failUnless(len(find_cliques(g, through_broadcast = True)) == 1)
        self.failUnless(len(find_cliques(g, through_broadcast = False)) == 2)
#         print g
#         for i, o in find_cliques(g, True):
#             print "-->", Env(i, [o])


# class _test_clique_opt(unittest.TestCase):

#     def test_straightforward(self):
#         x, y, z = inputs()
#         e = x ** 2.0 #x * x
#         g = Env([x], [e])
#         gof.ConstantFinder().optimize(g)
#         opt = CliqueOptimizer(through_broadcast = False,
#                               scalar_optimizer = scalar_opt.opt2,
#                               make_composite = False)
#         print g
#         opt.optimize(g)
#         print g

#     def test_inplace(self):
#         x, y, z = inputs()
#         #e = tensor.add_inplace(x, y + z)
#         e = x + tensor.add_inplace(y, z)
#         g = Env([x, y, z], [e])
#         opt = CliqueOptimizer(through_broadcast = False,
#                               scalar_optimizer = None,
#                               make_composite = True)
#         print g
#         opt.optimize(g)
#         print g
# #        print g.outputs[0].owner.c_code(['x', 'y', 'z'], ['e'], dict(fail = "FAIL;", id = 0))
#         print gof.OpWiseCLinker(g).make_function()(numpy.ones((5, 5)), numpy.ones((5, 5)), numpy.ones((5, 5)))

#     def test_straightforward(self):
#         x, y, z = inputs()
#         e = x + y + z
#         g = Env([x, y, z], [e])
#         opt = CliqueOptimizer(through_broadcast = False,
#                               scalar_optimizer = None,
#                               make_composite = True)
#         print g
#         opt.optimize(g)
#         print g
# #        print g.outputs[0].owner.c_code(['x', 'y', 'z'], ['e'], dict(fail = "FAIL;", id = 0))
#         print gof.OpWiseCLinker(g).make_function()(numpy.ones((5, 5)), numpy.ones((5, 5)), numpy.ones((5, 5)))

#     def test_straightforward2(self):
#         x, y, z = inputs()
#         m = y * z
#         d = tensor.dot(x, m)
#         d.name = 'd'
#         e = x + y + d
#         g = Env([x, y, z], [e])
#         opt = CliqueOptimizer(through_broadcast = False,
#                               scalar_optimizer = None,
#                               make_composite = True)
#         print g
#         opt.optimize(g)
#         print g
# #        print g.outputs[0].owner.c_code(['x', 'y', 'z'], ['e'], dict(fail = "FAIL;", id = 0))
#         print gof.OpWiseCLinker(g).make_function()(numpy.ones((5, 5)), numpy.ones((5, 5)), numpy.ones((5, 5)))
        
    


if __name__ == '__main__':
    unittest.main()




