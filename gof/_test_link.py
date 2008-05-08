

import unittest

import graph
from graph import Result, Apply, Constant
from type import Type
from op import Op
import env
import toolbox

from link import *

#from _test_result import Double


def as_result(x):
    assert isinstance(x, Result)
    return x

class TDouble(Type):
    def filter(self, data):
        return float(data)

tdouble = TDouble()

def double(name):
    return Result(tdouble, None, None, name = name)


class MyOp(Op):

    def __init__(self, nin, name, impl = None):
        self.nin = nin
        self.name = name
        if impl:
            self.impl = impl
    
    def make_node(self, *inputs):
        assert len(inputs) == self.nin
        inputs = map(as_result, inputs)
        for input in inputs:
            if input.type is not tdouble:
                raise Exception("Error 1")
        outputs = [double(self.name + "_R")]
        return Apply(self, inputs, outputs)

    def __str__(self):
        return self.name
    
    def perform(self, node, inputs, (out, )):
        out[0] = self.impl(*inputs)

add = MyOp(2, 'Add', lambda x, y: x + y)
sub = MyOp(2, 'Sub', lambda x, y: x - y)
mul = MyOp(2, 'Mul', lambda x, y: x * y)
div = MyOp(2, 'Div', lambda x, y: x / y)

def notimpl(self, x):
    raise NotImplementedError()

raise_err = MyOp(1, 'RaiseErr', notimpl)



def inputs():
    x = double('x')
    y = double('y')
    z = double('z')
    return x, y, z

def perform_linker(env):
    lnk = PerformLinker(env)
    return lnk

def Env(inputs, outputs):
    e = env.Env(inputs, outputs)
    return e


class _test_PerformLinker(unittest.TestCase):

    def test_thunk(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn, i, o = perform_linker(Env([x, y, z], [e])).make_thunk()
        i[0].data = 1
        i[1].data = 2
        fn()
        assert o[0].data == 1.5
        
    def test_function(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        fn = perform_linker(Env([x, y, z], [e])).make_function()
        assert fn(1.0, 2.0, 3.0) == 1.5

    def test_constant(self):
        x, y, z = inputs()
        y = Constant(tdouble, 2.0)
        e = mul(add(x, y), div(x, y))
        fn = perform_linker(Env([x], [e])).make_function()
        assert fn(1.0) == 1.5

    def test_input_output_same(self):
        x, y, z = inputs()
        fn = perform_linker(Env([x], [x])).make_function()
        self.failUnless(1.0 is fn(1.0))

    def test_input_dependency0(self):
        x, y, z = inputs()
        a,d = add(x,y), div(x,y)
        e = mul(a,d)
        fn = perform_linker(Env(*graph.clone([x, y, a], [e]))).make_function()
        self.failUnless(fn(1.0,2.0,9.0) == 4.5)

    def test_skiphole(self):
        x,y,z = inputs()
        a = add(x,y)
        r = raise_err(a)
        e = add(r,a)
        fn = perform_linker(Env(*graph.clone([x, y,r], [e]))).make_function()
        self.failUnless(fn(1.0,2.0,4.5) == 7.5)



#     def test_disconnected_input_output(self):
#         x,y,z = inputs()
#         a = add(x,y)
#         a.data = 3.0 # simulate orphan calculation
#         fn = perform_linker(env([z], [a])).make_function(inplace=True)
#         self.failUnless(fn(1.0) == 3.0)
#         self.failUnless(fn(2.0) == 3.0)

#     def test_thunk_inplace(self):
#         x, y, z = inputs()
#         e = mul(add(x, y), div(x, y))
#         fn, i, o = perform_linker(Env([x, y, z], [e])).make_thunk(True)
#         fn()
#         assert e.data == 1.5

#     def test_thunk_not_inplace(self):
#         x, y, z = inputs()
#         e = mul(add(x, y), div(x, y))
#         fn, i, o = perform_linker(env([x, y, z], [e])).make_thunk(False)
#         fn()
#         assert o[0].data == 1.5
#         assert e.data != 1.5

#     def test_function(self):
#         x, y, z = inputs()
#         e = mul(add(x, y), div(x, y))
#         fn = perform_linker(env([x, y, z], [e])).make_function()
#         assert fn(1.0, 2.0, 3.0) == 1.5
#         assert e.data != 1.5 # not inplace


if __name__ == '__main__':
    unittest.main()
        



        
