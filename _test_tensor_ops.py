

import unittest

from gof import ResultBase, Op, Env, modes
import gof

from tensor import *
from tensor_ops import *

import numpy

import sys

from scipy import weave


def inputs():
    l1 = [[1.0, 2.0], [3.0, 4.0]]
    l2 = [[3.0, 4.0], [1.0, 2.0]]
    l3 = numpy.ones((2, 3))
    x = modes.build(tensor(l1, 'x'))
    y = modes.build(tensor(l2, 'y'))
    z = modes.build(tensor(l3, 'z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
    return Env(inputs, outputs, features = features, consistency_check = validate)


class _test_TensorOps(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
#        e = mul(add(x, y), 2)
        e = (x + y) * 2
        fn, i, o = gof.PerformLinker(env([x, y], [e])).make_thunk(True)
        fn()
        assert (e.data == numpy.array([[8, 12], [8, 12]])).all()

    def test_1(self):
        x, y, z = inputs()
        e = dot(x, z).T
        fn, i, o = gof.PerformLinker(env([x, z], [e])).make_thunk(True)
        fn()
        assert (e.data == numpy.array([[3, 3, 3], [7, 7, 7]]).T).all()

    def test_2(self):
        x, y, z = inputs()
        x = x.data
        y = weave.inline("""
        PyObject* p = PyArray_Transpose(x_array, NULL);
        return_val = p;
        """, ['x'])
        print y

#     def test_0(self):
#         x, y, z = inputs()
#         e = transpose(x)
#         g = env([x], [e])
#         fn, (i, ), (o, ) = gof.cc.CLinker(g).make_thunk()
# #        print sys.getrefcount(i.data)
#         for blah in xrange(10000):
#             i.data = numpy.ones((1000, 1000)) # [[1.0, 2.0], [3.0, 4.0]]
#             fn()
# #        print sys.getrefcount(i.data)
# #        print sys.getrefcount(o.data)
#         print o.data
# #        assert res == numpy.asarray(arr)

# #     def test_1(self):
# #         x, y, z = inputs()
# #         e = mul(add(x, y), div(x, y))
# #         g = env([x, y], [e])
# #         fn = gof.cc.CLinker(g).make_function()
# #         assert fn(1.0, 2.0) == 1.5
# #         assert e.data == 1.5




if __name__ == '__main__':
    unittest.main()






