import unittest
import gof, gof.opt

import compile
from compile import *
from scalar import *
import tensor

# class Double(gof.result.Result):

#     def __init__(self, data, name = "oignon"):
#         assert isinstance(data, float)
#         gof.result.Result.__init__(self, role = None, name = name)
#         self.data = data

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return self.name

#     def __copy__(self):
#         return self.__class__(self.data, self.name)

# class MyOp(gof.op.Op):

#     nin = -1

#     def __init__(self, *inputs):
#         assert len(inputs) == self.nin
#         for input in inputs:
#             if not isinstance(input, Double):
#                 raise Exception("Error 1")
#         self.inputs = inputs
#         self.outputs = [Double(0.0, self.__class__.__name__ + "_R")]
    
#     def perform(self):
#         self.outputs[0].data = self.impl(*[input.data for input in self.inputs])


# class Unary(MyOp):
#     nin = 1

# class Binary(MyOp):
#     nin = 2

        
# class Add(Binary):
#     def impl(self, x, y):
#         return x + y
        
# class Sub(Binary):
#     def impl(self, x, y):
#         return x - y
        
# class Mul(Binary):
#     def impl(self, x, y):
#         return x * y
        
# class Div(Binary):
#     def impl(self, x, y):
#         return x / y


# def env(inputs, outputs, validate = True, features = []):
#     return gof.env.Env(inputs, outputs, features = features, consistency_check = validate)
# def perform_linker(env):
#     lnk = gof.link.PerformLinker(env)
#     return lnk

# def graph1(): # (x+y) * (x/z)
#     x = gof.modes.build(Double(1.0, 'x'))
#     y = gof.modes.build(Double(3.0, 'y'))
#     z = gof.modes.build(Double(4.0, 'z'))

#     o = Mul(Add(x, y).out, Div(x, z).out).out
#     return [x,y,z], [o]


def graph1(): # (x+y) * (x/z)
    x, y, z = floats('xyz')
    o = mul(add(x, y), div(x, z))
    return [x,y,z], [o]


class T_Function(unittest.TestCase):
    
    def test_noopt(self):
        gi, go = graph1()
        p = function(gi, go, optimizer = None, linker = 'py')
        self.failUnless(p(1.0,3.0,4.0) == 1.0)

#     def test_link_noopt(self):
#         gi, go = graph1()
#         fn, i, o = perform_linker(env(gi, go)).make_thunk(True)
#         fn()
#         self.failUnless(go[0].data == 1.0)

#     def test_link_opt(self):
#         opt = gof.opt.PatternOptimizer((Div, '1', '2'), (Div, '2', '1'))
#         gi, go = graph1()
#         e = env(gi, go)
#         opt.optimize(e)
#         fn, i, o = perform_linker(e).make_thunk(True)
#         fn()
#         self.failUnless(go[0].data == 16.0)

    def test_opt(self):
        opt = gof.opt.PatternOptimizer((div, '1', '2'), (div, '2', '1'))
        gi, go = graph1()
        p = function(gi,go, optimizer=opt.optimize, linker = 'py')
        self.failUnless(p(1.,3.,4.) == 16.0)

    def test_multiout(self):
        def graph2():
            x, y, z = floats('xyz')
            o = mul(add(x, y), div(x, z))
            return [x,y,z], [o, o.owner.inputs[1]]
        opt = gof.opt.PatternOptimizer((div, '1', '2'), (div, '2', '1'))
        gi, go = graph2()
        p = function(gi,go, optimizer=opt.optimize)
        a,b = p(1.,3.,4.)
        self.failUnless(a == 16.0)
        self.failUnless(b == 4.0)

    def test_make_many_functions(self):
        x, y, z = tensor.scalars('xyz')
        e0, e1, e2 = x+y+z, x*y-z, z*z+x*x+y*y
        f1 = function([x, y, z], [e0])
        f2 = function([x, y, z], [e0])
        f3 = function([x, y, z], [e1])
        f4 = function([x, y, z], [e2])
        f5 = function([e0], [e0 * e0])
        ff = FunctionFactory([x, y, z], [e0])
        f6 = ff.create()
        f7 = ff.create()
        f8 = ff.create()
        f9 = ff.partial(1.0, 2.0)
        assert f1(1.0, 2.0, 3.0) == 6.0
        assert f2(1.0, 2.0, 3.0) == 6.0
        assert f3(1.0, 2.0, 3.0) == -1.0
        assert f4(1.0, 2.0, 3.0) == 14.0
        assert f5(7.0) == 49.0
        assert f6(1.0, 2.0, 3.0) == 6.0
        assert f7(1.0, 2.0, 3.0) == 6.0
        assert f8(1.0, 2.0, 3.0) == 6.0
        assert f9(3.0) == 6.0

    def test_no_inputs(self):
        x, y, z = tensor.value(1.0), tensor.value(2.0), tensor.value(3.0)
        e = x*x + y*y + z*z
        assert function([], [e], linker = 'py')() == 14.0
        assert function([], [e], linker = 'c')() == 14.0
        assert function([], [e], linker = 'c|py')() == 14.0
        assert function([], [e], linker = 'c&py')() == 14.0
        assert eval_outputs([e]) == 14.0
        assert fast_compute(e) == 14.0

    def test_closure(self):
        x, y, z = tensor.scalars('xyz')
        v = tensor.value(numpy.zeros(()))
        e = x + tensor.add_inplace(v, 1)
        f = function([x], [e])
        assert f(1.) == 2.
        assert f(1.) == 3.
        assert f(1.) == 4.

    def test_borrow_true(self):
        x, y, z = tensor.scalars('xyz')
        e = x + y + z
        f = function([x, y, z], [e], borrow_outputs = True)
        res1 = f(1.0, 2.0, 3.0)
        assert res1 == 6.0
        res2 = f(1.0, 3.0, 5.0)
        assert res1 is res2
        assert res1 == 9.0
        assert res2 == 9.0

    def test_borrow_false(self):
        x, y, z = tensor.scalars('xyz')
        e = x + y + z
        for linker in 'py c c|py c&py'.split():
            f = function([x, y, z], [e], borrow_outputs = False, linker = linker)
            res1 = f(1.0, 2.0, 3.0)
            self.failUnless(res1 == 6.0, (res1, linker))
            res2 = f(1.0, 3.0, 5.0)
            self.failUnless(res1 is not res2, (res1, res2, linker))
            self.failUnless(res1 == 6.0, (res1, linker))
            self.failUnless(res2 == 9.0, (res2, linker))

    def test_borrow_false_through_inplace(self):
        x, y, z = tensor.scalars('xyz')
        # if borrow_outputs is False, we must not reuse the temporary created for x+y
        e = tensor.add_inplace(x + y, z)
        for linker in 'py c c|py c&py'.split():
            f = function([x, y, z], [e], borrow_outputs = False, linker = linker)
            res1 = f(1.0, 2.0, 3.0)
            self.failUnless(res1 == 6.0, (res1, linker))
            res2 = f(1.0, 3.0, 5.0)
            self.failUnless(res1 is not res2, (res1, res2, linker))
            self.failUnless(res1 == 6.0, (res1, linker))
            self.failUnless(res2 == 9.0, (res2, linker))


class T_fast_compute(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = tensor.value(1.0), tensor.value(2.0), tensor.value(3.0)
        e = x*x + y*y + z*z
        assert fast_compute(e) == 14.0
        assert compile._fcache[(e, )]() == 14.0
        
        
        
#     def test_orphans(self):
#         gi, go = graph1()
#         opt = None
#         p0 = function(gi[0:0], go, optimizer = None, linker = 'py')

#         self.failUnless(p0() == 1.0)

#         p3 = Function(gi,go)
#         p2 = Function(gi[0:2], go)
#         p1 = Function(gi[0:1], go)
#         try:
#             self.failUnless(p3() == 6.0)
#             self.fail()
#         except TypeError, e:
#             self.failUnless(e[0].split()[0:3] == ['Function','call', 'takes'])

#         self.failUnless(p3(1.,3.,4.) == 1.0)
#         self.failUnless(p2(1.,3.) == 1.0)
#         self.failUnless(p1(1.,) == 1.0)


#     def test_some_constant_outputs(self):
#         x = gof.modes.build(Double(1.0, 'x'))
#         y = gof.modes.build(Double(3.0, 'y'))
#         z = gof.modes.build(Double(4.0, 'z'))
#         xy = Mul(x,y).out
#         zz = Mul(z,z).out

#         p0 = Function([x,y], [xy, zz])
#         self.failUnless(p0(1.,3.) == [3.0,16.0])
#         self.failUnless(p0(1.5,4.) == [6.0,16.0])
#         self.failUnless(x.data == 1.0)
#         self.failUnless(y.data == 3.0)
#         self.failUnless(z.data == 4.0)

#         p1 = Function([z], [xy, zz],unpack_single=False)
#         self.failUnless(p1(4.) == [3.0,16.0]) #returns 6.0, 16.10
#         self.failUnless(p1(5.) == [3.0,25.0])


if __name__ == '__main__':

    unittest.main()

