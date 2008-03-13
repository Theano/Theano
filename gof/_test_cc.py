
import unittest

from cc import *
from result import ResultBase
from op import Op
from env import Env

class Double(ResultBase):

    def __init__(self, data, name = "oignon"):
        ResultBase.__init__(self, role = None, name = name)
        assert isinstance(data, float)
        self.data = data

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

#    def c_is_simple(self): return True
    
    def c_declare(self):
        return "double %(name)s; void* %(name)s_bad_thing;"

    def c_init(self):
        return """
        %(name)s = 0;
        %(name)s_bad_thing = malloc(100000);
        printf("Initializing %(name)s\\n"); 
        """

    def c_literal(self):
        return str(self.data)

    def c_extract(self):
        return """
        if (!PyFloat_Check(py_%(name)s)) {
            PyErr_SetString(PyExc_TypeError, "not a double!");
            %(fail)s
        }
        %(name)s = PyFloat_AsDouble(py_%(name)s);
        %(name)s_bad_thing = NULL;
        printf("Extracting %(name)s\\n");
        """
    
    def c_sync(self):
        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = PyFloat_FromDouble(%(name)s);
        if (!py_%(name)s)
            py_%(name)s = Py_None;
        printf("Syncing %(name)s\\n");
        """

    def c_cleanup(self):
        return """
        printf("Cleaning up %(name)s\\n");
        if (%(name)s_bad_thing)
            free(%(name)s_bad_thing);
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
    x = modes.build(Double(1.0, 'x'))
    y = modes.build(Double(2.0, 'y'))
    z = modes.build(Double(3.0, 'z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
#     inputs = [input.r for input in inputs]
#     outputs = [output.r for output in outputs]
    return Env(inputs, outputs, features = features, consistency_check = validate)


class _test_CLinker(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
        e = add(mul(add(x, y), div(x, y)), sub(sub(x, y), z))
        lnk = CLinker(env([x, y, z], [e])) #, [x.r, y.r, z.r], [e.r])
        cgen = lnk.code_gen()
        fn = lnk.make_function() #[x.r, y.r, z.r], [e.r])
        print fn(2.0, 2.0, 2.0)
#        fn = 0

    def test_1(self):
        x, y, z = inputs()
        z.constant = True
        e = add(mul(add(x, y), div(x, y)), sub(sub(x, y), z))
        lnk = CLinker(env([x, y], [e])) #, [x.r, y.r], [e.r])
        cgen = lnk.code_gen()
        fn = lnk.make_function() #[x.r, y.r], [e.r])
        print fn(2.0, 2.0)
#        fn = 0

    def test_2(self):
        x, y, z = inputs()
        op = Add(x, y)
        lnk = CLinker(op)
        cgen = lnk.code_gen()
        fn = lnk.make_function() #[x.r, y.r], [op.out])
        print fn(2.0, 7.0)
#        fn = 0

    def test_3(self):
        x, y, z = inputs()
        e = add(mul(add(x, y), div(x, y)), sub(sub(x, y), z))
        lnk = OpWiseCLinker(env([x, y, z], [e]))
        fn = lnk.make_function()
        print fn(2.0, 2.0, 2.0)
#        fn = 0

if __name__ == '__main__':
    unittest.main()
