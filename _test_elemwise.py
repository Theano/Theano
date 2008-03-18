
import unittest
import numpy

from tensor import tinit, Tensor
import gof
from gof import modes, Env

from elemwise import *


class ElemwiseAdd(Elemwise):

    def __init__(self, x, y):
        self.inputs = (x, y)
        self.outputs = [Tensor(dtype = x.dtype, broadcastable = x.broadcastable)]
    
    def var_desc(self):
        return [('x', 1), ('y', 1)], [('z', 1)]

#     def destroy_map(self):
#         return {self.out: [self.inputs[0]]}

    def c_code_foreach(self):
        return "%(z)s_i = %(x)s_i + %(y)s_i;"


def inputs():
    l1 = [[1.0, 2.0], [3.0, 4.0]]
    l2 = [[3.0, 4.0], [1.0, 2.0]]
    l3 = numpy.ones((2, 3))
    x = modes.build(tinit(l1, name = 'x'))
    y = modes.build(tinit(l2, name = 'y'))
    z = modes.build(tinit(l3, name = 'z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
    return Env(inputs, outputs, features = features, consistency_check = validate)


class _test_Elemwise(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
        e = ElemwiseAdd(x, y).out
        fn, i, o = gof.CLinker(env([x, y], [e])).make_thunk(True)
        fn()
        assert (e.data == numpy.array([[4, 6], [4, 6]])).all()
        x.data.resize((1, 4))
        y.data.resize((1, 4))
        fn()
        assert (e.data == numpy.array([[4, 6, 4, 6]])).all()

#     def test_1(self):
#         x, y, z = inputs()
#         e = ElemwiseAdd(x, y).out
#         fn, i, o = gof.CLinker(env([x, y], [e])).make_thunk(True)
#         fn()
#         assert (e.data == numpy.array([[4, 6], [4, 6]])).all()
#         x.data.resize((1, 4))
#         y.data.resize((1, 4))
#         fn()
#         assert (e.data == numpy.array([[4, 6, 4, 6]])).all()



if __name__ == '__main__':
    unittest.main()
