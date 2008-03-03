
import unittest

from modes import *


from result import ResultBase
from op import Op
from env import Env


class Double(ResultBase):

    def __init__(self, data, name = "oignon"):
        assert isinstance(data, float)
        ResultBase.__init__(self, role = None, data = data, name = name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __add__(self, other):
        return Add(self, other)


def convert(x):
    if isinstance(x, float):
        return Double(x)
    elif isinstance(x, Double):
        return x
    raise Exception("Error 1")

class MyOp(Op):

    nin = -1

    def __init__(self, *inputs):
        assert len(inputs) == self.nin
        inputs = [convert(input) for input in inputs]
                    
        self.inputs = inputs
        self.outputs = [Double(0.0, self.__class__.__name__ + "_R")]
    
    def perform(self):
        self.outputs[0].data = self.impl(*[input.data for input in self.inputs])


class Unary(MyOp):
    nin = 1

class Binary(MyOp):
    nin = 2


class Add(Binary):
    def impl(self, x, y):
        return x + y
        
class Sub(Binary):
    def impl(self, x, y):
        return x - y
        
class Mul(Binary):
    def impl(self, x, y):
        return x * y
        
class Div(Binary):
    def impl(self, x, y):
        return x / y

make_constructors(globals())


def inputs(mode):
    x = mode(Double(1.0, 'x'))
    y = mode(Double(2.0, 'y'))
    z = mode(Double(3.0, 'z'))
    return x, y, z

def env(inputs, outputs, validate = True):
    inputs = [input.r for input in inputs]
    outputs = [output.r for output in outputs]
    return Env(inputs, outputs, features = [], consistency_check = validate)


class _test_Modes(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs(BuildMode)
        e = add(add(x, y), z)
        g = env([x, y, z], [e])
        assert str(g) == "[Add(Add(x, y), z)]"
        assert e.r.data == 0.0
        
    def test_1(self):
        x, y, z = inputs(BuildEvalMode)
        e = add(add(x, y), z)
        g = env([x, y, z], [e])
        assert str(g) == "[Add(Add(x, y), z)]"
        assert e.r.data == 6.0
        
    def test_2(self):
        x, y, z = inputs(EvalMode)
        e = add(add(x, y), z)
        g = env([x, y, z], [e])
        assert str(g) == "[Add_R]"
        assert e.r.data == 6.0

    def test_3(self):
        x, y, z = inputs(BuildMode)
        e = x + y + z
        g = env([x, y, z], [e])
        assert str(g) == "[Add(Add(x, y), z)]"
        assert e.r.data == 0.0
        
    def test_4(self):
        x, y, z = inputs(BuildEvalMode)
        e = x + 34.0
        g = env([x, y, z], [e])
        assert str(g) == "[Add(x, oignon)]"
        assert e.r.data == 35.0
        
    def test_5(self):
        xb, yb, zb = inputs(BuildMode)
        xe, ye, ze = inputs(EvalMode)
        try:
            e = xb + ye
        except TypeError:
            # Trying to add inputs from different modes is forbidden
            pass
        else:
            raise Exception("Expected an error.")
        

if __name__ == '__main__':
    unittest.main()





