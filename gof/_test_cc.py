
import unittest

from cc import *
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

    def c_type(self):
        return "double"

    def c_data_extract(self):
        return """
        %(type)s %(name)s = PyFloat_AsDouble(py_%(name)s);
        %(fail)s
        """
    
    def c_data_sync(self):
        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = PyFloat_FromDouble(%(name)s);
        if (!py_%(name)s)
            py_%(name)s = Py_None;
        """


class MyOp(Op):

    nin = -1

    def __init__(self, *inputs):
        assert len(inputs) == self.nin
        for input in inputs:
            if not isinstance(input, Double):
                raise Exception("Error 1")
        self.inputs = inputs
        self.outputs = [Double(0.0, self.__class__.__name__ + "_R")]

class Unary(MyOp):
    nin = 1
    def c_var_names(self):
        return [['x'], ['z']]

class Binary(MyOp):
    nin = 2
    def c_var_names(self):
        return [['x', 'y'], ['z']]

        
class Add(Binary):
    def c_code(self):
        return "%(z)s = %(x)s + %(y)s;"
        
class Sub(Binary):
    def c_code(self):
        return "%(z)s = %(x)s - %(y)s;"
        
class Mul(Binary):
    def c_code(self):
        return "%(z)s = %(x)s * %(y)s;"
        
class Div(Binary):
    def c_code(self):
        return "%(z)s = %(x)s / %(y)s;"


import modes
modes.make_constructors(globals())

def inputs():
    x = modes.BuildMode(Double(1.0, 'x'))
    y = modes.BuildMode(Double(2.0, 'y'))
    z = modes.BuildMode(Double(3.0, 'z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
    inputs = [input.r for input in inputs]
    outputs = [output.r for output in outputs]
    return Env(inputs, outputs, features = features, consistency_check = validate)


class _test_CLinker(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
        e = mul(add(x, y), div(x, y))
        lnk = CLinker(env([x, y, z], [e]))
        print lnk.code_gen()


if __name__ == '__main__':
    unittest.main()
