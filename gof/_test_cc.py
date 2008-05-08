
import unittest

from link import PerformLinker, Profiler
from cc import *
from type import Type
from graph import Result, Apply, Constant
from op import Op
import env
import toolbox

def as_result(x):
    assert isinstance(x, Result)
    return x

class TDouble(Type):
    def filter(self, data):
        return float(data)

    def c_declare(self, name, sub):
        return "double %(name)s; void* %(name)s_bad_thing;" % locals()

    def c_init(self, name, sub):
        return """
        %(name)s = 0;
        %(name)s_bad_thing = malloc(100000);
        //printf("Initializing %(name)s\\n"); 
        """ % locals()

    def c_literal(self, data):
        return str(data)

    def c_extract(self, name, sub):
        return """
        if (!PyFloat_Check(py_%(name)s)) {
            PyErr_SetString(PyExc_TypeError, "not a double!");
            %(fail)s
        }
        %(name)s = PyFloat_AsDouble(py_%(name)s);
        %(name)s_bad_thing = NULL;
        //printf("Extracting %(name)s\\n");
        """ % dict(locals(), **sub)
    
    def c_sync(self, name, sub):
        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = PyFloat_FromDouble(%(name)s);
        if (!py_%(name)s)
            py_%(name)s = Py_None;
        //printf("Syncing %(name)s\\n");
        """ % locals()

    def c_cleanup(self, name, sub):
        return """
        //printf("Cleaning up %(name)s\\n");
        if (%(name)s_bad_thing)
            free(%(name)s_bad_thing);
        """ % locals()

tdouble = TDouble()

def double(name):
    return Result(tdouble, None, None, name = name)


class MyOp(Op):

    def __init__(self, nin, name):
        self.nin = nin
        self.name = name
    
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


class Unary(MyOp):
    def __init__(self):
        MyOp.__init__(self, 1, self.__class__.__name__)

class Binary(MyOp):
    def __init__(self):
        MyOp.__init__(self, 2, self.__class__.__name__)

        
class Add(Binary):
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s + %(y)s;" % locals()
    def impl(self, x, y):
        return x + y
add = Add()
        
class Sub(Binary):
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s - %(y)s;" % locals()
    def impl(self, x, y):
        return -10 # erroneous (most of the time)
sub = Sub()
        
class Mul(Binary):
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s * %(y)s;" % locals()
    def impl(self, x, y):
        return x * y
mul = Mul()
        
class Div(Binary):
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s / %(y)s;" % locals()
    def impl(self, x, y):
        return x / y
div = Div()


def inputs():
    x = double('x')
    y = double('y')
    z = double('z')
    return x, y, z


def Env(inputs, outputs):
    e = env.Env(inputs, outputs)
    return e


class _test_CLinker(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = add(mul(add(x, y), div(x, y)), sub(sub(x, y), z))
        lnk = CLinker(Env([x, y, z], [e]))
        fn = lnk.make_function()
        self.failUnless(fn(2.0, 2.0, 2.0) == 2.0)

#     def test_orphan(self):
#         x, y, z = inputs()
#         z = Constant(tdouble, 4.12345678)
#         e = add(mul(add(x, y), div(x, y)), sub(sub(x, y), z))
#         lnk = CLinker(Env([x, y], [e]))
#         fn = lnk.make_function()
#         self.failUnless(abs(fn(2.0, 2.0) + 0.12345678) < 1e-9)
#         print lnk.code_gen()
#         self.failUnless("4.12345678" not in lnk.code_gen()) # we do not expect the number to be inlined

    def test_literal_inlining(self):
        x, y, z = inputs()
        z = Constant(tdouble, 4.12345678)
        e = add(mul(add(x, y), div(x, y)), sub(sub(x, y), z))
        lnk = CLinker(Env([x, y], [e]))
        fn = lnk.make_function()
        self.failUnless(abs(fn(2.0, 2.0) + 0.12345678) < 1e-9)
        self.failUnless("4.12345678" in lnk.code_gen()) # we expect the number to be inlined

    def test_single_node(self):
        x, y, z = inputs()
        node = add.make_node(x, y)
        lnk = CLinker(Env(node.inputs, node.outputs))
        fn = lnk.make_function()
        self.failUnless(fn(2.0, 7.0) == 9)
    
    def test_dups(self):
        # Testing that duplicate inputs are allowed.
        x, y, z = inputs()
        e = add(x, x)
        lnk = CLinker(Env([x, x], [e]))
        fn = lnk.make_function()
        self.failUnless(fn(2.0, 2.0) == 4)
        # note: for now the behavior of fn(2.0, 7.0) is undefined

    def test_dups_inner(self):
        # Testing that duplicates are allowed inside the graph
        x, y, z = inputs()
        e = add(mul(y, y), add(x, z))
        lnk = CLinker(Env([x, y, z], [e]))
        fn = lnk.make_function()
        self.failUnless(fn(1.0, 2.0, 3.0) == 8.0)
    


class _test_OpWiseCLinker(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = add(mul(add(x, y), div(x, y)), sub(sub(x, y), z))
        lnk = OpWiseCLinker(Env([x, y, z], [e]))
        fn = lnk.make_function()
        self.failUnless(fn(2.0, 2.0, 2.0) == 2.0)

    def test_constant(self):
        x, y, z = inputs()
        x = Constant(tdouble, 7.2, name = 'x')
        e = add(mul(x, y), mul(y, z))
        lnk = OpWiseCLinker(Env([y, z], [e]))
        fn = lnk.make_function()
        res = fn(1.5, 3.0)
        self.failUnless(res == 15.3, res)


class MyExc(Exception):
    pass
def _my_checker(x, y):
    if x[0] != y[0]:
        raise MyExc("Output mismatch.", {'performlinker': x[0], 'clinker': y[0]})


class _test_DualLinker(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = inputs()
        e = add(mul(x, y), mul(y, z)) # add and mul are correct in C and in Python
        lnk = DualLinker(Env([x, y, z], [e]), checker = _my_checker)
        fn = lnk.make_function()
        res = fn(7.2, 1.5, 3.0)
        self.failUnless(res == 15.3, res)

    def test_mismatch(self):
        x, y, z = inputs()
        e = sub(mul(x, y), mul(y, z)) # sub is correct in C but erroneous in Python
        g = Env([x, y, z], [e])
        lnk = DualLinker(g, checker = _my_checker)
        fn = lnk.make_function()

        self.failUnless(CLinker(g).make_function()(1.0, 2.0, 3.0) == -4.0) # good
        self.failUnless(OpWiseCLinker(g).make_function()(1.0, 2.0, 3.0) == -4.0) # good
        self.failUnless(PerformLinker(g).make_function()(1.0, 2.0, 3.0) == -10.0) # (purposely) wrong
        
        try:
            # this runs OpWiseCLinker and PerformLinker in parallel and feeds
            # results of matching operations to _my_checker to verify that they
            # are the same.
            res = fn(1.0, 2.0, 3.0)
            self.fail()
        except MyExc, e:
            pass
        else:
            self.fail()

#     def test_orphan(self):
#         x, y, z = inputs()
#         x = Constant(tdouble, 7.2, name = 'x')
#         e = add(mul(x, y), mul(y, z)) # add and mul are correct in C and in Python
#         lnk = DualLinker(Env([y, z], [e]), checker = _my_checker)
#         fn = lnk.make_function()
#         res = fn(1.5, 3.0)
#         self.failUnless(res == 15.3, res)



        
if __name__ == '__main__':
    unittest.main()
