import unittest
import gof, gof.modes, gof.opt

from compile import *

class Double(gof.result.ResultBase):

    def __init__(self, data, name = "oignon"):
        assert isinstance(data, float)
        gof.result.ResultBase.__init__(self, role = None, name = name)
        self.data = data

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __copy__(self):
        return self.__class__(self.data, self.name)

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

def graph1(): # (x+y) * (x/z)
    x = gof.modes.build(Double(1.0, 'x'))
    y = gof.modes.build(Double(3.0, 'y'))
    z = gof.modes.build(Double(4.0, 'z'))

    o = Mul(Add(x, y).out, Div(x, z).out).out
    return [x,y,z], [o]



class T_what:
    def test_nothing(self):
        pass

class T_Function(unittest.TestCase):
    def test_noopt(self):
        gi, go = graph1()
        p = Function(gi,go)
        self.failUnless(p(1.0,3.0,4.0) == 1.0)


    def test_link_noopt(self):
        gi, go = graph1()
        fn, i, o = perform_linker(env(gi, go)).make_thunk(True)
        fn()
        self.failUnless(go[0].data == 1.0)

    def test_link_opt(self):
        opt = gof.opt.PatternOptimizer((Div, '1', '2'), (Div, '2', '1'))
        gi, go = graph1()
        e = env(gi, go)
        opt.optimize(e)
        fn, i, o = perform_linker(e).make_thunk(True)
        fn()
        self.failUnless(go[0].data == 16.0)
    def test_opt(self):
        opt = gof.opt.PatternOptimizer((Div, '1', '2'), (Div, '2', '1'))
        gi, go = graph1()
        p = Function(gi,go, optimizer=opt)
        self.failUnless(p(1.,3.,4.) == 16.0)

    def test_multiout(self):
        def graph2():
            x = gof.modes.build(Double(1.0, 'x'))
            y = gof.modes.build(Double(3.0, 'y'))
            z = gof.modes.build(Double(4.0, 'z'))

            o = Mul(Add(x, y).out, Div(x, z).out).out
            return [x,y,z], [o, o.owner.inputs[1]]
        opt = gof.opt.PatternOptimizer((Div, '1', '2'), (Div, '2', '1'))
        gi, go = graph2()
        p = Function(gi,go, optimizer=opt)
        a,b = p(1.,3.,4.)
        self.failUnless(a == 16.0)
        self.failUnless(b == 4.0)

    def test_orphans(self):
        gi, go = graph1()
        opt = None
        p0 = Function(gi[0:0], go, optimizer=opt)

        self.failUnless(p0() == 1.0)

        p3 = Function(gi,go, optimizer=opt)
        p2 = Function(gi[0:2], go, optimizer=opt)
        p1 = Function(gi[0:1], go, optimizer=opt)
        try:
            self.failUnless(p3() == 6.0)
            self.fail()
        except TypeError, e:
            self.failUnless(e[0].split()[0:3] == ['Function','call', 'takes'])

        self.failUnless(p3(1.,3.,4.) == 1.0)
        self.failUnless(p2(1.,3.) == 1.0)
        self.failUnless(p1(1.,) == 1.0)


    def test_some_constant_outputs(self):
        x = gof.modes.build(Double(1.0, 'x'))
        y = gof.modes.build(Double(3.0, 'y'))
        z = gof.modes.build(Double(4.0, 'z'))
        xy = Mul(x,y).out
        zz = Mul(z,z).out

        p0 = Function([x,y], [xy, zz])
        self.failUnless(p0(1.,3.) == [3.0,16.0])
        self.failUnless(p0(1.5,4.) == [6.0,16.0])
        self.failUnless(x.data == 1.0)
        self.failUnless(y.data == 3.0)
        self.failUnless(z.data == 4.0)

        p1 = Function([z], [xy, zz],unpack_single=False)
        self.failUnless(p1(4.) == [3.0,16.0]) #returns 6.0, 16.10
        self.failUnless(p1(5.) == [3.0,25.0])


if __name__ == '__main__':

    unittest.main()

