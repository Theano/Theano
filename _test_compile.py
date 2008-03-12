import unittest
import gof, gof.modes, gof.opt

from compile import *

class Double(gof.result.ResultBase):

    def __init__(self, data, name = "oignon"):
        assert isinstance(data, float)
        gof.result.ResultBase.__init__(self, role = None, data = data, name = name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class MyOp(gof.op.Op):

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


def env(inputs, outputs, validate = True, features = []):
    return gof.env.Env(inputs, outputs, features = features, consistency_check = validate)
def perform_linker(env):
    lnk = gof.link.PerformLinker(env)
    return lnk

def graph1():
    x = gof.modes.build(Double(1.0, 'x'))
    y = gof.modes.build(Double(2.0, 'y'))
    z = gof.modes.build(Double(3.0, 'z'))

    o = Mul(Add(x, y).out, Div(x, y).out).out
    return [x,y,z], [o]

def graph2():
    x = gof.modes.build(Double(1.0, 'x'))
    y = gof.modes.build(Double(2.0, 'y'))
    z = gof.modes.build(Double(3.0, 'z'))

    o = Mul(Add(x, y).out, Div(x, y).out).out
    return [x,y,z], [o, o, o.owner.inputs[1]]



class _test_compile(unittest.TestCase):

    def test_link_noopt(self):
        gi, go = graph1()
        fn, i, o = perform_linker(env(gi, go)).make_thunk(True)
        fn()
        self.failUnless(go[0].data == 1.5)

    def test_link_opt(self):
        opt = gof.opt.PatternOptimizer((Div, '1', '2'), (Div, '2', '1'))
        gi, go = graph1()
        e = env(gi, go)
        opt.optimize(e)
        fn, i, o = perform_linker(e).make_thunk(True)
        fn()
        self.failUnless(go[0].data == 6.0)

    def test_noopt(self):
        gi, go = graph1()
        p = Function(gi,go)
        self.failUnless(p() == 1.5)

    def test_opt(self):
        opt = gof.opt.PatternOptimizer((Div, '1', '2'), (Div, '2', '1'))
        gi, go = graph1()
        p = Function(gi,go, optimizer=opt)
        self.failUnless(p() == 6.0)

    def test_multiout(self):
        opt = gof.opt.PatternOptimizer((Div, '1', '2'), (Div, '2', '1'))
        gi, go = graph2()
        p = Function(gi,go, optimizer=opt)
        a,b,c = p()
        self.failUnless(a == 6.0)
        self.failUnless(b == 6.0)
        self.failUnless(a is b)
        self.failUnless(c == 2.0)


if __name__ == '__main__':

    unittest.main()

