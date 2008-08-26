import unittest
import gof, gof.opt

import compile
from compile import *
from scalar import *
import tensor


PatternOptimizer = lambda p1, p2, ign=True: gof.OpKeyOptimizer(gof.PatternSub(p1, p2), ignore_newtrees=ign)


def graph1(): # (x+y) * (x/z)
    x, y, z = floats('xyz')
    o = mul(add(x, y), div(x, z))
    return [x,y,z], [o]


class T_Function(unittest.TestCase):
    
    def test_noopt(self):
        gi, go = graph1()
        p = function(gi, go, optimizer = None, linker = 'py')
        self.failUnless(p(1.0,3.0,4.0) == 1.0)

    def test_opt(self):
        opt = PatternOptimizer((div, '1', '2'), (div, '2', '1'))
        gi, go = graph1()
        p = function(gi,go, optimizer=opt.optimize, linker = 'py')
        self.failUnless(p(1.,3.,4.) == 16.0)

    def test_multiout(self):
        def graph2():
            x, y, z = floats('xyz')
            o = mul(add(x, y), div(x, z))
            return [x,y,z], [o, o.owner.inputs[1]]
        opt = PatternOptimizer((div, '1', '2'), (div, '2', '1'))
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


import tensor as T
import numpy as N
class T_OpFromGraph(unittest.TestCase):

    def test_straightforward(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e], linker='c|py')
        f = op(x, y, z) - op(y, z, x)
        fn = function([x, y, z], [f])
        xv, yv, zv = N.ones((2, 2)), N.ones((2, 2))*3, N.ones((2, 2))*5
        assert numpy.all(8.0 == fn(xv, yv, zv))
        assert numpy.all(8.0 == fn(xv, yv, zv))
    
    def test_size_changes(self):
        x, y, z = T.matrices('xyz')
        e = T.dot(x, y)
        op = OpFromGraph([x, y], [e], linker='c|py')
        f = op(x, op(y, z))
        fn = function([x, y, z], [f])
        xv, yv, zv = N.ones((2, 3)), N.ones((3, 4))*3, N.ones((4, 5))*5
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)
        res = fn(xv, yv, zv)
        assert res.shape == (2, 5)
        assert numpy.all(180.0 == res)
    
    def test_grad(self):
        x, y, z = T.matrices('xyz')
        e = x + y * z
        op = OpFromGraph([x, y, z], [e], linker='c|py', grad_depth = 2)
        f = op(x, y, z)
        f = f - T.grad(f, y)
        fn = function([x, y, z], [f])
        xv, yv, zv = N.ones((2, 2)), N.ones((2, 2))*3, N.ones((2, 2))*5
        assert numpy.all(11.0 == fn(xv, yv, zv))


class T_state(unittest.TestCase):

    def test_accumulator():
        """Test low-level interface with state."""
        x = T.scalar('x')
        s = T.scalar('s')

        fn, states = theano.function_states(inputs = [x], outputs = [], states = [(s, 0, s+x)])

        sum = 0
        for inc in [1, 4, 5,23, -324]:
            sum += inc
            fn(inc)
            assert sum == states[0].value

    def test_perceptron():
        """Test high-level state interface."""

        mu0 = numpy.array([1.0,0.0])
        mu1 = numpy.array([0.0,0.1])
        si0 = numpy.ones_like(mu0) #unit variance
        si1 = numpy.ones_like(mu1) #unit variance

        #implicit internal state
        label = T.random.bernoulli(0.5) 

        #implicit internal state for each DiagGaussian
        x = label * T.random.DiagGaussian(mu0, si0) \
                + (1 - label) * T.random.DiagGaussian(mu1,si1)

        w = T.tensor.dvector()
        b = T.tensor.dscalar()
        lr = 0.01

        decision = dot(x,w) + b > 0
        new_w = w + neq(label, decision) * lr * x
        new_b = b + neq(label, decision) * (label * (-lr) + (1-label)*lr)

        init_w = numpy.array([0.0, 0.0])
        init_b = 0.0

        io_stream = T.function([], [label, x])

        perceptron_learn = T.function([x, label], [decision], 
                state={
                    'w':(w, init_w, update_w),
                    'b':(b, init_b, update_b),
                    'lr':(lr, 0.01)})

        perceptron_use = T.function([x], [decision],
                state={
                    'w':(w, perceptron_learn.shared['w']),
                    'b':(b, perceptron_learn.shared['b'])})

        errs = 0
        for i in xrange(100):
            il, ix = io_stream()

            d0 = perceptron_use(ix)
            d1 = perceptron_learn(ix, il)

            assert d0 == d1

            errs += (d0 != d1)

            print d0
        print 'errs =', errs 


if __name__ == '__main__':

    unittest.main()

