from tensor import *
import tensor as T

import unittest
from copy import copy
from compile import Function
import gradient

#TODO: consider moving this function / functionality to gradient.py
#      rationale: it's tricky, and necessary everytime you want to verify
#      gradient numerically

def verify_grad(testcase, op_cls, pt_list, n_tests=1, rng=numpy.random, eps=0.0000001, tol=0.0001):
    """testcase.failUnless( analytic gradient matches finite-diff gradient) """

    for test_num in xrange(n_tests):
        for pt in pt_list:
            tensor_pt = [tensor(p) for p in pt]
            o = op_cls(*tensor_pt)
            if len(o.outputs) > 1:
                raise NotImplementedError('cant (yet) autotest gradient of op with multiple outputs')
                # we could make loop over outputs making random projections R for each,
                # but this doesn't handle the case where not all the outputs are
                # differentiable... so I leave this as TODO for now -jsb.
            o_fn = Function(tensor_pt, o.outputs)
            o_fn_out = o_fn(*pt)
            random_projection = rng.rand(*o_fn_out.shape)
            t_r = tensor(random_projection)

            #random projection of o onto t_r
            cost = sum(t_r * o.outputs[0])
            cost_fn = Function(tensor_pt, [cost])

            num_grad = gradient.numeric_grad(cost_fn, pt)

            grad_fn = Function(tensor_pt, gradient.grad(cost, tensor_pt))
            
            analytic_grad = grad_fn()
            if not isinstance(analytic_grad, (list, tuple)):
                analytic_grad = [analytic_grad]

            if num_grad.max_err(analytic_grad) > 1.0e-4:
                raise Exception(verify_grad.E_grad)
verify_grad.E_grad = 'gradient error exceeded tolerance'


class T_tensor(unittest.TestCase):
    def test0(self): # allocate from a scalar float
        t = tensor(1.0)
        self.failUnless(isinstance(t, Tensor))
        self.failUnless(t.dtype == 'float64')
        self.failUnless(t.broadcastable == ())
        self.failUnless(t.role == None)
        self.failUnless(isinstance(t.data, numpy.ndarray))
        self.failUnless(str(t.data.dtype) == 'float64')
        self.failUnless(t.data == 1.0)
    def test0_int(self): # allocate from a scalar float
        t = tensor(1)
        self.failUnless(isinstance(t, Tensor))
        self.failUnless(t.dtype == 'int64')
    def test1(self): # allocate from a vector of ints, not broadcastable
        t = tensor(numpy.ones(5,dtype='int32'))
        self.failUnless(isinstance(t, Tensor))
        self.failUnless(t.dtype == 'int32')
        self.failUnless(t.broadcastable == (0,))
        self.failUnless(isinstance(t.data, numpy.ndarray))
        self.failUnless(str(t.data.dtype) == 'int32')
    def test2(self): # allocate from a column matrix of complex with name
        t = tensor(numpy.ones((5,1),dtype='complex64'),name='bart')
        self.failUnless(isinstance(t, Tensor))
        self.failUnless(t.dtype == 'complex64')
        self.failUnless(t.broadcastable == (0,1))
        self.failUnless(isinstance(t.data, numpy.ndarray))
        self.failUnless(t.name == 'bart')
    def test2b(self): # allocate from a column matrix, not broadcastable
        t = tensor(numpy.ones((5,1),dtype='complex64'),broadcastable=0)
        self.failUnless(isinstance(t, Tensor))
        self.failUnless(t.dtype == 'complex64')
        self.failUnless(t.broadcastable == (0,0))
        self.failUnless(isinstance(t.data, numpy.ndarray))
    def test_data_normal(self): #test that assigning to .data works when it should
        t = tensor(numpy.ones((5,1),dtype='complex64'), broadcastable=0)
        o27 = numpy.ones((2,7))
        t.data = o27
        lst = t._data
        self.failUnless(t.data.shape == (2,7))
        self.failUnless(t.data is o27)
        self.failUnless(t._data is lst)
    def test_data_badrank0(self):
        t = tensor(numpy.ones((5,1),dtype='complex64'), broadcastable=0)
        try:
            t.data = numpy.ones((2,7,1))
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is Tensor.filter.E_rank)
        try:
            t.data = numpy.ones(1)
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is Tensor.filter.E_rank)
    def test_data_badrank1(self):
        t = tensor(numpy.ones((1,1),dtype='complex64'), broadcastable=1)
        try:
            t.data = numpy.ones((1,1,1))
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is Tensor.filter.E_rank)
        try:
            t.data = numpy.ones(1)
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is Tensor.filter.E_rank)
    def test_data_badshape0(self):
        t = tensor(numpy.ones((1,1),dtype='complex64'), broadcastable=1)
        try:
            t.data = numpy.ones((1,2))
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is Tensor.filter.E_shape)
        try:
            t.data = numpy.ones((0,1))
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is Tensor.filter.E_shape)

class T_stdlib(unittest.TestCase):
    def test0(self):
        t = tensor(1.0)
        tt = copy(t)
        self.failUnless(t.dtype == tt.dtype)
        self.failUnless(t.broadcastable == tt.broadcastable)
        self.failUnless(t.broadcastable is tt.broadcastable)
        self.failIf(t.data is tt.data)

def check_eq(self, node_in, node_out, arg_in, arg_out):
    fn = Function([node_in], [node_out])
    self.failUnless( numpy.all(fn(arg_in) == arg_out), (arg_in, arg_out))

def check_eq2(self, inputs, output, args_in, arg_out):
    fn = Function(inputs, [output])
    val = fn(*args_in)
    self.failUnless( numpy.all(val == arg_out), (val, arg_out))


class T_abs(unittest.TestCase):
    def test_impl(self):
        t = tensor(1.0)
        check_eq(self, t, abs(t), 1.0, 1.0)
        check_eq(self, t, abs(t), -1.0, 1.0)

        for shape in (2,), (3,4):
            t = tensor(numpy.ones(shape))
            d = numpy.random.rand(*shape)*2-1.0
            check_eq(self, t, abs(t), d, abs(d))
            check_eq(self, t, abs(t), -d, abs(-d))

    def test_grad(self):
        verify_grad(self, Abs, [[numpy.ones(())], [numpy.ones(3)]])

    class AbsBadGrad(T._Elemwise):
        def impl(self, x):
            return numpy.abs(x)
        def grad(self, x, gz):
            return scale(gz * sgn(x),0.9)
        def c_foreach(self, (x_i, ), (z_i, )):
            return "z_i = abs(x_i);"

    def test_badgrad(self):
        try:
            verify_grad(self, T_abs.AbsBadGrad, [[numpy.ones(())], [numpy.ones(3)]])
            self.fail()
        except Exception, e:
            self.failUnless(str(e) == verify_grad.E_grad, str(e))



class T_sum(unittest.TestCase):
    def test_impl(self):
        t = tensor(0.0)
        check_eq(self, t, Sum(t).out, 1.0, 1.0)
        check_eq(self, t, Sum(t).out, -1.0, -1.0)

        t = tensor([0.0, 0.0])
        d = numpy.asarray([-0.4, 1.2])
        check_eq(self, t, Sum(t).out, d, numpy.sum(d))
        check_eq(self, t, Sum(t).out, -d, -numpy.sum(d))

class T_mul(unittest.TestCase):
    def test_elemwise(self):
        a = tensor(0.0)
        b = tensor(0.0)
        check_eq2(self, [a,b], mul_elemwise(a,b), [3.0, 4.0], 12.0)
        check_eq2(self, [a,b], mul_elemwise(a,a), [-1.0,2.0], 1.0)
        check_eq2(self, [a,b], mul(a,b), [3.0, 4.0], 12.0)
        check_eq2(self, [a,b], mul(a,a), [-1.0,2.0], 1.0)

        a = tensor(numpy.ones(2))
        b = tensor(numpy.ones(2))
        aa = numpy.asarray([-0.5, 4.0])
        bb = numpy.asarray([-0.5, 2.0])
        check_eq2(self, [a,b], mul_elemwise(a,b), [aa,bb], numpy.asarray([0.25, 8.0]))
        check_eq2(self, [a,b], mul_elemwise(a,b), [aa,aa], numpy.asarray([0.25, 16.0]))
        check_eq2(self, [a,b], mul(a,b), [aa,bb], numpy.asarray([0.25, 8.0]))
        check_eq2(self, [a,b], mul(a,b), [aa,aa], numpy.asarray([0.25, 16.0]))

    def test_wrong_shapes(self):
        a = tensor(numpy.ones(3))
        b = tensor(numpy.ones(4))
        try:
            check_eq2(self, [a,b], MulElemwise(a,b).out,
                    [numpy.ones(3), numpy.ones(4)], 1.0)
            self.fail()
        except ValueError, e:
            self.failUnless(e is T._assert_same_shapes.E_shape)


if __name__ == '__main__':
    unittest.main()
