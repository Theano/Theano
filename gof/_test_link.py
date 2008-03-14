

import unittest

from result import ResultBase
from op import Op
from env import Env

from link import *

class Double(ResultBase):

    def __init__(self, data, name = "oignon"):
        ResultBase.__init__(self, role = None, name = name)
        assert isinstance(data, float)
        self.data = data

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class MyOp(Op):

    nin = -1

    def __init__(self, *inputs):
        assert len(inputs) == self.nin
        for input in inputs:
            if not isinstance(input, Double):
                raise Exception("Error 1")
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

class RaiseErr(Unary):
    def impl(self, x):
        raise NotImplementedError()


import modes
modes.make_constructors(globals())

def inputs():
    x = modes.build(Double(1.0, 'x'))
    y = modes.build(Double(2.0, 'y'))
    z = modes.build(Double(3.0, 'z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
#     inputs = [input.r for input in inputs]
#     outputs = [output.r for output in outputs]
    return Env(inputs, outputs, features = features, consistency_check = validate)

def perform_linker(env):
    lnk = PerformLinker(env)
    return lnk


class _test_PerformLinker(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn, i, o = perform_linker(env([x, y, z], [e])).make_thunk(True)
        fn()
        assert e.data == 1.5

    def test_1(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn, i, o = perform_linker(env([x, y, z], [e])).make_thunk(False)
        fn()
        assert e.data != 1.5

    def test_2(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn = perform_linker(env([x, y, z], [e])).make_function()
        assert fn(1.0, 2.0, 3.0) == 1.5
        assert e.data != 1.5

    def test_input_output_same(self):
        x, y, z = inputs()
        a,d = add(x,y), div(x,y)
        e = mul(a,d)
        fn = perform_linker(env([e], [e])).make_function()
        self.failUnless(1 is fn(1))

    def test_input_dependency0(self):
        x, y, z = inputs()
        a,d = add(x,y), div(x,y)
        e = mul(a,d)
        fn = perform_linker(env([x, a], [e])).make_function()
        self.failUnless(fn(1.0,9.0) == 4.5)

    def test_skiphole(self):
        x,y,z = inputs()
        a = add(x,y)
        r = RaiseErr(a).out
        e = add(r,a)
        fn = perform_linker(env([x, y,r], [e])).make_function()
        self.failUnless(fn(1.0,2.0,4.5) == 7.5)

    def test_disconnected_input_output(self):
        x,y,z = inputs()
        a = add(x,y)
        fn = perform_linker(env([z], [a])).make_function()
        self.failUnless(fn(1.0) == 3.0)
        self.failUnless(fn(2.0) == 3.0)



if __name__ == '__main__':
    unittest.main()




        
