
import unittest

from result import ResultBase
from op import Op
from opt import PatternOptimizer, OpSubOptimizer

from env import Env, InconsistencyError
from toolbox import *



class MyResult(ResultBase):

    def __init__(self, name):
        ResultBase.__init__(self, role = None, data = [1000], name = name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class MyOp(Op):
    nin = -1

    def __init__(self, *inputs):
        assert len(inputs) == self.nin
        for input in inputs:
            if not isinstance(input, MyResult):
                raise Exception("Error 1")
        self.inputs = inputs
        self.outputs = [MyResult(self.__class__.__name__ + "_R")]

class Sigmoid(MyOp):
    nin = 1

class Add(MyOp):
    nin = 2

class Dot(MyOp):
    nin = 2


import modes
modes.make_constructors(globals())

def inputs():
    x = modes.build(MyResult('x'))
    y = modes.build(MyResult('y'))
    z = modes.build(MyResult('z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
#     inputs = [input.r for input in inputs]
#     outputs = [output.r for output in outputs]
    return Env(inputs, outputs, features = features, consistency_check = validate)


class _test_EquivTool(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
        sx = sigmoid(x)
        e = add(sx, sigmoid(y))
        g = env([x, y, z], [e], features = [EquivTool])
        assert g.equiv(sx) is sx
        g.replace(sx, dot(x, z))
        assert g.equiv(sx) is not sx
        assert isinstance(g.equiv(sx).owner, Dot)



if __name__ == '__main__':
    unittest.main()

    
