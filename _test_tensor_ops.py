

import unittest

from gof import ResultBase, Op, Env, modes
import gof

from tensor import *
from tensor_ops import *

import numpy

import sys

from scipy import weave


def inputs():
    l1 = [[1.0, 2.0], [3.0, 4.0]]
    l2 = [[3.0, 4.0], [1.0, 2.0]]
    l3 = numpy.ones((2, 3))
    x = modes.build(tensor(l1, 'x'))
    y = modes.build(tensor(l2, 'y'))
    z = modes.build(tensor(l3, 'z'))
    return x, y, z

def env(inputs, outputs, validate = True, features = []):
    return Env(inputs, outputs, features = features, consistency_check = validate)


class _test_TensorOps(unittest.TestCase):

    def test_0(self):
        x, y, z = inputs()
#        e = mul(add(x, y), 2)
        e = (x + y) * 2
        fn, i, o = gof.PerformLinker(env([x, y], [e])).make_thunk(True)
        fn()
        assert (e.data == numpy.array([[8, 12], [8, 12]])).all()

    def test_1(self):
        x, y, z = inputs()
        e = dot(x, z).T
        fn, i, o = gof.PerformLinker(env([x, z], [e])).make_thunk(True)
        fn()
        assert (e.data == numpy.array([[3, 3, 3], [7, 7, 7]]).T).all()

    def test_2(self):
        x, y, z = inputs()
        x = x.data
        y = weave.inline("""
        PyObject* p = PyArray_Transpose(x_array, NULL);
        return_val = p;
        """, ['x'])
        print y

#     def test_0(self):
#         x, y, z = inputs()
#         e = transpose(x)
#         g = env([x], [e])
#         fn, (i, ), (o, ) = gof.cc.CLinker(g).make_thunk()
# #        print sys.getrefcount(i.data)
#         for blah in xrange(10000):
#             i.data = numpy.ones((1000, 1000)) # [[1.0, 2.0], [3.0, 4.0]]
#             fn()
# #        print sys.getrefcount(i.data)
# #        print sys.getrefcount(o.data)
#         print o.data
# #        assert res == numpy.asarray(arr)

# #     def test_1(self):
# #         x, y, z = inputs()
# #         e = mul(add(x, y), div(x, y))
# #         g = env([x, y], [e])
# #         fn = gof.cc.CLinker(g).make_function()
# #         assert fn(1.0, 2.0) == 1.5
# #         assert e.data == 1.5


from core import *

import unittest
import gradient

#useful mostly for unit tests
def _approx_eq(a,b,eps=1.0e-9):
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    if a.shape != b.shape:
        return False
    return numpy.max(numpy.abs(a-b)) < eps

if 1: # run gradient tests
    def _scalar(x):
        rval = numpy.zeros(())
        rval.itemset(x)
        return rval

    def _test_grad(self, op_cls, args, n_tests=1,eps=0.0000001, tol=0.0001):
        """unittest.TestCase.failUnless( analytic gradient matches finite-diff gradient )
        
        The criterion is that every input gradient must match every
        finite-difference gradient (using stepsize of eps) to relative precision
        tol.
        """
        def _finite_diff1(f, x, eps, f_of_x = None):
            if f_of_x is None: f_of_x = f(x)
            y_eps = f(x+eps)
            return (y_eps - f_of_x) / eps
        def _scalar_f(op_cls, args, R, arg_idx, coord=None):
            m = args[arg_idx].data
            if () == m.shape:
                def rval(x):
                    old_x = float(m)
                    m.itemset(x)
                    y = float(sum(mul_elemwise(R, op_cls(*args))).data)
                    m.itemset(old_x)
                    return y
                return rval
            else:
                def rval(x):
                    old_x = m.__getitem__(coord)
                    #print old_x.shape
                    #print x.shape
                    m.__setitem__(coord, x)
                    y = float(sum(mul_elemwise(R, op_cls(*args))).data)
                    m.__setitem__(coord, old_x)
                    return y
                return rval

        self.failUnless(hasattr(op_cls, 'update_gradient'), op_cls)
        op_out = op_cls(*args)
        if len(op_out.owner.outputs) > 1:
            raise NotImplementedError('cant autotest gradient of op with multiple outputs')
            # we could make loop over outputs making random projections R for each,
            # but this doesn't handle the case where not all the outputs are
            # differentiable... so I leave this as TODO for now -jsb.
        R = numpy.random.rand(*op_out.shape)
        y = sum(mul_elemwise(R, op_out))
        g = gradient.grad(y)

        def abs_rel_err(a,b):
            return abs( (a-b) / (a+b+eps))

        for idx in range(len(args)):
            #print 'aaaaaaa', op_cls, [i.shape for i in args]
            g_i = g(args[idx])
            if g_i is gradient.Undefined:
                continue
            if args[idx].shape == ():
                fd_grad = _finite_diff1(_scalar_f(op_cls, args, R, idx),
                        args[idx].data, eps, y.data)
                err = abs_rel_err(fd_grad,g_i.data)
                self.failUnless( err < tol, (err, op_cls, idx))
            elif len(args[idx].shape) == 1:
                for i in xrange(args[idx].shape[0]):
                    fd_grad = _finite_diff1(_scalar_f(op_cls, args, R, idx, (i,)),
                            args[idx].data[i], eps, y.data)
                    err = abs_rel_err(fd_grad,g_i.data[i])
                    self.failUnless( abs(err) < tol, (err, op_cls, idx, i))
            elif len(args[idx].shape) == 2:
                for i in xrange(args[idx].shape[0]):
                    for j in xrange(args[idx].shape[1]):
                        fd_grad = _finite_diff1(_scalar_f(op_cls, args, R, idx, (i,j)),
                                args[idx].data[i,j], eps, y.data)
                        err = abs_rel_err(fd_grad,g_i.data[i,j])
                        self.failUnless( abs(err) < tol, (err, op_cls, idx, i, j))
            else:
                raise NotImplementedError()

    def _testgrad_unary_elemwise_randnearzero(op_cls, n_tests=1,eps=0.000001, tol=0.0001):
        class test_some_op_gradient(unittest.TestCase):
            def setUp(self):
                gof.lib.build_eval_mode()
                numpy.random.seed([234,234,23333])
            def tearDown(self):
                gof.lib.pop_mode()

            def test0(self):
                """Gradient Test with a small scalar"""
                _test_grad(self, op_cls,
                        (Numpy2(data=(numpy.ones(()))*0.03),),
                        n_tests, eps, tol)
            def test1(self):
                """Gradient Test with a medium scalar"""
                _test_grad(self, op_cls,
                        (Numpy2(data=(numpy.ones(()))*1.03),),
                        n_tests, eps, tol)
            def test2(self):
                """Gradient Test with a big scalar"""
                _test_grad(self, op_cls, 
                        (Numpy2(data=(numpy.ones(()))*90.03),),
                        n_tests, eps, tol)
            def test3(self):
                """Gradient Test with a vector"""
                _test_grad(self, op_cls,
                        (Numpy2(data=numpy.random.rand(3)+0.01),),
                        n_tests, eps, tol)
            def test4(self):
                """Gradient Test with a matrix"""
                _test_grad(self, op_cls,
                        (Numpy2(data=numpy.random.rand(2,3)*4),),
                        n_tests, eps, tol)
        return test_some_op_gradient
    neg_test = _testgrad_unary_elemwise_randnearzero(neg)
    twice_test = _testgrad_unary_elemwise_randnearzero(twice)
    exp_test = _testgrad_unary_elemwise_randnearzero(exp)
    sqr_test = _testgrad_unary_elemwise_randnearzero(sqr)
    sqrt_test = _testgrad_unary_elemwise_randnearzero(sqrt)
    inv_test = _testgrad_unary_elemwise_randnearzero(inv_elemwise)
    transpose_test = _testgrad_unary_elemwise_randnearzero(transpose)

    def _testgrad_unary_elemwise_randpositive(op_cls, n_tests=1,eps=0.000001, tol=0.0001):
        class test_some_op_gradient(unittest.TestCase):
            def setUp(self):
                gof.lib.build_eval_mode()
                numpy.random.seed([234,234,23333])
            def tearDown(self):
                gof.lib.pop_mode()

            def test0(self):
                """Gradient Test with a small scalar"""
                _test_grad(self, op_cls,
                        (Numpy2(data=numpy.ones(())*0.03),),
                        n_tests, eps, tol)
            def test1(self):
                """Gradient Test with a medium scalar"""
                _test_grad(self, op_cls,
                        (Numpy2(data=numpy.ones(())*1.03),),
                        n_tests, eps, tol)
            def test2(self):
                """Gradient Test with a big scalar"""
                _test_grad(self, op_cls, 
                        (Numpy2(data=numpy.ones(())*90.03),),
                        n_tests, eps, tol)
            def test3(self):
                """Gradient Test with a vector"""
                _test_grad(self, op_cls,
                        (Numpy2(data=numpy.random.rand(3)+0.01),),
                        n_tests, eps, tol)
            def test4(self):
                """Gradient Test with a matrix"""
                _test_grad(self, op_cls,
                        (Numpy2(data=numpy.random.rand(2,3)*4),),
                        n_tests, eps, tol)
        return test_some_op_gradient
    log_test = _testgrad_unary_elemwise_randpositive(log)
    log2_test = _testgrad_unary_elemwise_randpositive(log2)
    sqrt_test = _testgrad_unary_elemwise_randpositive(sqrt)

    def _testgrad_binary_elemwise(op_cls, domain, n_tests=1,eps=0.000001, tol=0.0001):
        class test_some_op_gradient(unittest.TestCase):
            def setUp(self):
                gof.lib.build_eval_mode()
                numpy.random.seed([234,234,23333])
            def tearDown(self):
                gof.lib.pop_mode()
            def mytest(self, *raw_args):
                args = [Numpy2(data=d(a)) for a,d in zip(raw_args,domain)]
                _test_grad(self, op_cls, args, n_tests, eps, tol)
            def test0(self):
                """Gradient test low"""
                self.mytest(numpy.zeros(()), numpy.zeros(()))
            def test1(self):
                """Gradient test middle"""
                self.mytest(numpy.ones(())*.5, numpy.ones(())*0.5)
            def test2(self):
                """Gradient test high"""
                self.mytest(numpy.ones(()), numpy.ones(()))
            def test3(self):
                """Gradient test with a vector"""
                self.mytest(numpy.random.rand(4),numpy.random.rand(4))
            def test4(self):
                """Gradient test with a matrix"""
                self.mytest(numpy.random.rand(3,2),numpy.random.rand(3,2))
        return test_some_op_gradient
    add_test = _testgrad_binary_elemwise(add_elemwise, [lambda x:(x-0.5)*50]*2)
    sub_test = _testgrad_binary_elemwise(sub_elemwise, [lambda x:(x-0.5)*50]*2)
    mul_test = _testgrad_binary_elemwise(mul_elemwise, [lambda x:(x-0.5)*50]*2)
    div_test = _testgrad_binary_elemwise(div_elemwise, [lambda x:(x-0.4)*50]*2)
    pow_test = _testgrad_binary_elemwise(pow_elemwise, [lambda x:x*10+0.01, lambda x:(x-0.5)*4])

    def _testgrad_binary_scalar(op_cls, domain, n_tests=1,eps=0.000001, tol=0.0001):
        class test_some_op_gradient(unittest.TestCase):
            def setUp(self):
                gof.lib.build_eval_mode()
                numpy.random.seed([234,234,23333])
            def tearDown(self):
                gof.lib.pop_mode()
            def mytest(self, *raw_args):
                args = [Numpy2(data=domain[0](raw_args[0])),
                        Numpy2(data=_scalar(domain[1](raw_args[1])))]
                #print repr(args[0].data), repr(args[1].data)
                _test_grad(self, op_cls, args, n_tests, eps, tol)
            def test0_low(self):
                self.mytest(numpy.zeros(()), _scalar(0))
            def test1_middle(self):
                self.mytest(numpy.ones(())*.5, _scalar(0.5))
            def test2_high(self):
                self.mytest(numpy.ones(()), _scalar(1.0))
            def test3_vector(self):
                self.mytest(numpy.random.rand(4),_scalar(numpy.random.rand()))
            def test4_matrix(self):
                self.mytest(numpy.random.rand(3,2),_scalar(numpy.random.rand()))
        test_some_op_gradient.__name__ = str(op_cls.__name__) + '_test'
        return test_some_op_gradient
    add_scalar_test = _testgrad_binary_scalar(add_scalar, [lambda x:(x-0.5)*50]*2)
    mul_scalar_test = _testgrad_binary_scalar(mul_scalar, [lambda x:(x-0.5)*50]*2)
    pow_scalar_l_test = _testgrad_binary_scalar(pow_scalar_l,
            [lambda x:(x-0.5)*10, lambda x:(x+0.01)*10.0])
    pow_scalar_r_test = _testgrad_binary_scalar(pow_scalar_r, 
            [lambda x:(x+0.01)*10, lambda x:(x-0.5)*10.0])
    fill_test = _testgrad_binary_scalar(fill, [lambda x:(x-0.5)*50]*2)

    class test_some_op_gradient(unittest.TestCase):
        def setUp(self):
            gof.lib.build_eval_mode()
            numpy.random.seed([234,234,23333])
        def tearDown(self):
            gof.lib.pop_mode()
        def mytest(self, *raw_args):
            n_tests = 1
            eps = 0.000001
            tol=0.00001
            args = [Numpy2(data=raw_args[0]),
                    Numpy2(data=raw_args[1])]
            #print repr(args[0].data), repr(args[1].data)
            _test_grad(self, dot, args, n_tests, eps, tol)
        def test0(self):
            """Gradient test low"""
            self.mytest(numpy.zeros(()), _scalar(0))
        def test1(self):
            """Gradient test middle"""
            self.mytest(_scalar(0.5), _scalar(0.5))
        def test2(self):
            """Gradient test high"""
            self.mytest(numpy.ones(()), _scalar(1.0))
        def test3(self):
            """Gradient test dot with vectors"""
            self.mytest(numpy.random.rand(4),numpy.random.rand(4))
        def test4(self):
            """Gradient test dot with matrices"""
            self.mytest(numpy.random.rand(3,2),numpy.random.rand(2,4))
        def _notyet_test5(self):
            """Gradient test dot with 3d-tensor on left"""
            self.mytest(numpy.random.rand(3,4,2),numpy.random.rand(2,5))
        def _notyet_test6(self):
            """Gradient test dot with 3d-tensor on right"""
            self.mytest(numpy.random.rand(4,2),numpy.random.rand(3,2,5))

class testCase_slicing(unittest.TestCase):
    def setUp(self):
        build_eval_mode()
    def tearDown(self):
        pop_mode()

    def test_getitem0(self):
        a = numpy.ones((4,4))
        wa1 = wrap(a)[:,1]
        try:
            err = wa1 + a
        except ValueError, e:
            self.failUnless(str(e) == \
                    'The dimensions of the inputs do not match.',
                    'Wrong ValueError')
            return
        self.fail('add should not have succeeded')

    def test_getitem1(self):
        a = numpy.ones((4,4))
        wa1 = wrap(a)[1]
        self.failUnless(wa1.data.shape == (4,))

    def test_getslice_0d_all(self):
        """Test getslice does not work on 0d array """
        a = numpy.ones(())
        try:
            wa1 = wrap(a)[:]
        except IndexError, e:
            self.failUnless(str(e) == "0-d arrays can't be indexed.")
            return
        self.fail()
    def test_getslice_1d_all(self):
        """Test getslice on 1d array"""
        a = numpy.ones(4)
        wa1 = wrap(a)[:]
        self.failUnless(wa1.data.shape == (4,), 'wrong shape')
        self.failUnless(numpy.all(wa1.data == a), 'unequal value')

        a[1] = 3.4
        self.failUnless(wa1.data[1] == 3.4, 'not a view')

        try:
            wa1[2] = 2.5
        except TypeError, e:
            self.failUnless("object does not support item assignment" in str(e))
            return
        self.fail()
    def test_getslice_3d_all(self):
        """Test getslice on 3d array"""
        a = numpy.ones((4,5,6))
        wa1 = wrap(a)[:]
        self.failUnless(wa1.data.shape == (4,5,6), 'wrong shape')
        self.failUnless(numpy.all(wa1.data == a), 'unequal value')

        a[1,1,1] = 3.4
        self.failUnless(wa1.data[1,1,1] == 3.4, 'not a view')
    def test_getslice_1d_some(self):
        """Test getslice on 1d array"""
        a = numpy.ones(5)
        wa1 = wrap(a)[1:3]
        a[2] = 5.0
        a[3] = 2.5
        self.failUnless(wa1.data.shape == (2,))
        self.failUnless(a[1] == wa1.data[0])
        self.failUnless(a[2] == wa1.data[1])
    def test_getslice_1d_step(self):
        """Test getslice on 1d array"""
        a = numpy.ones(8)
        wa1 = wrap(a)[0:8:2]
        for i in xrange(8): a[i] = i

        self.failUnless(wa1.shape == (4,))
        for i in xrange(4):
            self.failUnless(a[i*2] == wa1.data[i])
    def test_getslice_3d_float(self):
        """Test getslice on 3d array"""
        a = numpy.asarray(range(4*5*6))
        a.resize((4,5,6))
        wa1 = wrap(a)[1:3]
        self.failUnless(wa1.shape == (2,5,6))
        self.failUnless(numpy.all(a[1:3] == wa1.data))
        a[1] *= -1.0
        self.failUnless(numpy.all(a[1:3] == wa1.data))
    def test_getslice_3d_one(self):
        """Test getslice on 3d array"""
        a = numpy.asarray(range(4*5*6))
        a.resize((4,5,6))
        wa = wrap(a)
        wa_123 = wa[1,2,3]
        self.failUnless(wa_123.shape == (), wa_123.shape)

class test_Numpy2(unittest.TestCase):
    def setUp(self):
        build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        pop_mode()
    def test_0(self):
        r = Numpy2()
    def test_1(self):
        o = numpy.ones((3,3))
        r = Numpy2(data=o)
        self.failUnless(r.data is o)
        self.failUnless(r.shape == (3,3))
        self.failUnless(str(r.dtype) == 'float64')

    def test_2(self):
        r = Numpy2(data=[(3,3),'int32'])
        self.failUnless(r.data is None)
        self.failUnless(r.shape == (3,3))
        self.failUnless(str(r.dtype) == 'int32')
        r.alloc()
        self.failUnless(isinstance(r.data, numpy.ndarray))
        self.failUnless(r.shape == (3,3))
        self.failUnless(str(r.dtype) == 'int32')

    def test_3(self):
        a = Numpy2(data=numpy.ones((2,2)))
        b = Numpy2(data=numpy.ones((2,2)))
        c = add(a,b)
        self.failUnless(_approx_eq(c, numpy.ones((2,2))*2))

    def test_4(self):
        ones = numpy.ones((2,2))
        a = Numpy2(data=ones)
        o = numpy.asarray(a)
        self.failUnless((ones == o).all())

    def test_5(self):
        ones = numpy.ones((2,2))
        self.failUnless(_approx_eq(Numpy2(data=ones), Numpy2(data=ones)))

class testCase_producer_build_mode(unittest.TestCase):
    def test_0(self):
        """producer in build mode"""
        build_mode()
        a = ones(4)
        self.failUnless(a.data is None, a.data)
        self.failUnless(a.state is gof.result.Empty, a.state)
        self.failUnless(a.shape == 4, a.shape)
        self.failUnless(str(a.dtype) == 'float64', a.dtype)
        pop_mode()
    def test_1(self):
        """producer in build_eval mode"""
        build_eval_mode()
        a = ones(4)
        self.failUnless((a.data == numpy.ones(4)).all(), a.data)
        self.failUnless(a.state is gof.result.Computed, a.state)
        self.failUnless(a.shape == (4,), a.shape)
        self.failUnless(str(a.dtype) == 'float64', a.dtype)
        pop_mode()

class testCase_add_build_mode(unittest.TestCase):
    def setUp(self):
        build_mode()
        numpy.random.seed(44)
    def tearDown(self):
        pop_mode()

class testCase_dot(unittest.TestCase):
    def setUp(self):
        build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        pop_mode()

    @staticmethod
    def rand(*args):
        return numpy.random.rand(*args)

    def cmp_dot(self,x,y):
        def spec(x):
            x = numpy.asarray(x)
            return type(x), x.dtype, x.shape
        zspec = dot.specs(spec(x), spec(y))
        nz = numpy.dot(x,y)
        self.failUnless(zspec == spec(nz))
        self.failUnless(_approx_eq(dot(x,y), numpy.dot(x,y)))

    def cmp_dot_comp(self, x,y):
        x = numpy.asarray(x)
        y = numpy.asarray(y)
        z = dot(x,y)
        p = compile.single(z)
        if len(x.shape):
            x[:] = numpy.random.rand(*x.shape)
        else:
            x.fill(numpy.random.rand(*x.shape))
        if len(y.shape):
            y[:] = numpy.random.rand(*y.shape)
        else:
            y.fill(numpy.random.rand(*y.shape))
        p() # recalculate z
        self.failUnless(_approx_eq(z, numpy.dot(x,y)))

    def test_dot_0d_0d(self): self.cmp_dot(1.1, 2.2)
    def test_dot_0d_1d(self): self.cmp_dot(1.1, self.rand(5))
    def test_dot_0d_2d(self): self.cmp_dot(3.0, self.rand(6,7))
    def test_dot_0d_3d(self): self.cmp_dot(3.0, self.rand(8,6,7))
    def test_dot_1d_0d(self): self.cmp_dot(self.rand(5), 1.1 )
    def test_dot_1d_1d(self): self.cmp_dot(self.rand(5), self.rand(5))
    def test_dot_1d_2d(self): self.cmp_dot(self.rand(6), self.rand(6,7))
    def test_dot_1d_3d(self): self.cmp_dot(self.rand(6), self.rand(8,6,7))
    def test_dot_2d_0d(self): self.cmp_dot(self.rand(5,6), 1.0)
    def test_dot_2d_1d(self): self.cmp_dot(self.rand(5,6), self.rand(6))
    def test_dot_2d_2d(self): self.cmp_dot(self.rand(5,6), self.rand(6,7))
    def test_dot_2d_3d(self): self.cmp_dot(self.rand(5,6), self.rand(8,6,7))
    def test_dot_3d_0d(self): self.cmp_dot(self.rand(4,5,6), 1.0)
    def test_dot_3d_1d(self): self.cmp_dot(self.rand(4,5,6), self.rand(6))
    def test_dot_3d_2d(self): self.cmp_dot(self.rand(4,5,6), self.rand(6,7))
    def test_dot_3d_3d(self): self.cmp_dot(self.rand(4,5,6), self.rand(8,6,7))
    def test_dot_0d_0d_(self): self.cmp_dot_comp(1.1, 2.2)
    def test_dot_0d_1d_(self): self.cmp_dot_comp(1.1, self.rand(5))
    def test_dot_0d_2d_(self): self.cmp_dot_comp(3.0, self.rand(6,7))
    def test_dot_0d_3d_(self): self.cmp_dot_comp(3.0, self.rand(8,6,7))
    def test_dot_1d_0d_(self): self.cmp_dot_comp(self.rand(5), 1.1 )
    def test_dot_1d_1d_(self): self.cmp_dot_comp(self.rand(5), self.rand(5))
    def test_dot_1d_2d_(self): self.cmp_dot_comp(self.rand(6), self.rand(6,7))
    def test_dot_1d_3d_(self): self.cmp_dot_comp(self.rand(6), self.rand(8,6,7))
    def test_dot_2d_0d_(self): self.cmp_dot_comp(self.rand(5,6), 1.0)
    def test_dot_2d_1d_(self): self.cmp_dot_comp(self.rand(5,6), self.rand(6))
    def test_dot_2d_2d_(self): self.cmp_dot_comp(self.rand(5,6), self.rand(6,7))
    def test_dot_2d_3d_(self): self.cmp_dot_comp(self.rand(5,6), self.rand(8,6,7))
    def test_dot_3d_0d_(self): self.cmp_dot_comp(self.rand(4,5,6), 1.0)
    def test_dot_3d_1d_(self): self.cmp_dot_comp(self.rand(4,5,6), self.rand(6))
    def test_dot_3d_2d_(self): self.cmp_dot_comp(self.rand(4,5,6), self.rand(6,7))
    def test_dot_3d_3d_(self): self.cmp_dot_comp(self.rand(4,5,6), self.rand(8,6,7))

    def test_dot_fail_1_1(self):
        x = numpy.random.rand(5)
        y = numpy.random.rand(6)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()

    def test_dot_fail_1_2(self):
        x = numpy.random.rand(5)
        y = numpy.random.rand(6,4)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_1_3(self):
        x = numpy.random.rand(5)
        y = numpy.random.rand(6,4,7)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_2_1(self):
        x = numpy.random.rand(5,4)
        y = numpy.random.rand(6)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_2_2(self):
        x = numpy.random.rand(5,4)
        y = numpy.random.rand(6,7)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_2_3(self):
        x = numpy.random.rand(5,4)
        y = numpy.random.rand(6,7,8)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_3_1(self):
        x = numpy.random.rand(5,4,3)
        y = numpy.random.rand(6)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_3_2(self):
        x = numpy.random.rand(5,4,3)
        y = numpy.random.rand(6,7)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()
    def test_dot_fail_3_3(self):
        x = numpy.random.rand(5,4,3)
        y = numpy.random.rand(6,7,8)
        try:
            z = dot(x,y)
        except ValueError, e:
            self.failUnless(str(e) == 'objects are not aligned', e)
            return
        self.fail()

class testCase_transpose(unittest.TestCase):

    def setUp(self):
        build_eval_mode()

    def tearDown(self):
        pop_mode()
    
    def test_1d_alias(self):
        a = numpy.ones(10)
        ta = transpose(a)
        self.failUnless(ta.data.shape == a.shape)
        self.failUnless(numpy.all(ta.data == a))
        a[3] *= -1.0
        self.failUnless(numpy.all(ta.data == a))

    def test_1d_copy(self):
        a = numpy.ones(10)
        ta = transpose_copy(a)
        self.failUnless(ta.data.shape == a.shape)
        self.failUnless(numpy.all(ta.data == a))
        a[3] *= -1.0
        self.failIf(numpy.all(ta.data == a))

    def test_2d_alias(self):
        a = numpy.ones((10,3))
        ta = transpose(a)
        self.failUnless(ta.data.shape == (3,10))

    def test_3d_alias(self):
        a = numpy.ones((10,3,5))
        ta = transpose(a)
        self.failUnless(ta.data.shape == (5,3,10))
        a[9,0,0] = 5.0
        self.failUnless(ta.data[0,0,9] == 5.0)

    def test_3d_copy(self):
        a = numpy.ones((10,3,5))
        ta = transpose_copy(a)
        self.failUnless(ta.data.shape == (5,3,10))
        a[9,0,0] = 5.0
        self.failUnless(ta.data[0,0,9] == 1.0)

class testCase_power(unittest.TestCase):
    def setUp(self):
        build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        pop_mode()
    def test1(self):
        r = numpy.random.rand(50)
        exp_r = exp(r)
        self.failUnless(exp_r.__array__().__class__ is numpy.ndarray)

    def test_0(self):
        r = numpy.random.rand(50)

        exp_r = exp(r)
        n_exp_r = numpy.exp(r)
        self.failUnless( _approx_eq(exp_r, n_exp_r), 
                (exp_r, exp_r.data, n_exp_r,
                    numpy.max(numpy.abs(n_exp_r.__sub__(exp_r.__array__())))))

        log_exp_r = log(exp_r)
        self.failUnless( _approx_eq(log_exp_r, r), log_exp_r)

    def test_1(self):
        r = numpy.random.rand(50)
        r2 = pow(r,2)
        self.failUnless( _approx_eq(r2, r*r))

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()






