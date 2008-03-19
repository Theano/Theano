from tensor import *
import tensor # for hidden symbols

import unittest
from copy import copy
from compile import Function, eval_outputs
import gradient
import gof, gof.graph


#TODO: consider moving this function / functionality to gradient.py
#      rationale: it's tricky, and necessary everytime you want to verify
#      gradient numerically

def verify_grad(testcase, op_cls, pt, n_tests=1, rng=numpy.random, eps=0.0000001, tol=0.0001):
    """testcase.failUnless( analytic gradient matches finite-diff gradient) """
    pt = [numpy.asarray(p) for p in pt]

    for test_num in xrange(n_tests):
        tensor_pt = [tinit(p,name='input %i'%i) for i,p in enumerate(pt)]
        o = op_cls(*tensor_pt)
        if len(o.outputs) > 1:
            raise NotImplementedError('cant (yet) autotest gradient of op with multiple outputs')
            # we could make loop over outputs making random projections R for each,
            # but this doesn't handle the case where not all the outputs are
            # differentiable... so I leave this as TODO for now -JB.
        o_fn = Function(tensor_pt, o.outputs)
        o_fn_out = o_fn(*pt)
        random_projection = rng.rand(*o_fn_out.shape)
        t_r = tinit(random_projection)

        #random projection of o onto t_r
        cost = sum(t_r * o.outputs[0])
        cost_fn = Function(tensor_pt, [cost])

        num_grad = gradient.numeric_grad(cost_fn, pt)

        symbolic_grad = gradient.grad(cost, tensor_pt,tinit(1.0,name='g_cost'))
        if 0:
            print '-------'
            print '----------'
            for op in gof.graph.io_toposort(tensor_pt, symbolic_grad):
                print op
        grad_fn = Function(tensor_pt, symbolic_grad)
        
        analytic_grad = grad_fn(*pt)
        if not isinstance(analytic_grad, (list, tuple)):
            analytic_grad = [analytic_grad]

        if num_grad.max_err(analytic_grad) > 1.0e-4:
            raise Exception(verify_grad.E_grad)
verify_grad.E_grad = 'gradient error exceeded tolerance'



def check_eq(self, node_in, node_out, arg_in, arg_out):
    fn = Function([node_in], [node_out])
    self.failUnless( numpy.all(fn(arg_in) == arg_out), (arg_in, arg_out))

def check_eq2(self, inputs, output, args_in, arg_out):
    fn = Function(inputs, [output])
    val = fn(*args_in)
    self.failUnless( numpy.all(val == arg_out), (val, arg_out))

def check_eq2_c(self, inputs, output, args_in, arg_out):
    fn = Function(inputs, [output], linker_cls = gof.CLinker)
    val = fn(*args_in)
    self.failUnless( numpy.all(val == arg_out), (val, arg_out))


class T_transpose(unittest.TestCase):
    def test0(self):
        n = tinit(numpy.ones(()))
        t = transpose(n)
        self.failUnless(t.owner.__class__ is Transpose)
        f = Function([n], [t])
        tval = f(n.data)
        self.failUnless(tval.shape == n.data.shape)

        #test aliasing
        tval += 55.0
        self.failUnless(n.data == 56.0)
        
    def test1(self):
        n = tinit(numpy.ones(5))
        t = transpose(n)
        self.failUnless(t.owner.__class__ is Transpose)
        f = Function([n], [t])
        tval = f(n.data)
        self.failUnless(tval.shape == n.data.shape)
        #test aliasing
        tval += 55.0
        self.failUnless(n.data[0] == 56.0)
        
    def test2(self):
        n = tinit(numpy.ones((5,3)))
        t = transpose(n)
        self.failUnless(t.owner.__class__ is Transpose)
        f = Function([n], [t])
        tval = f(n.data)
        self.failUnless(tval.shape == (3,5))
        #test aliasing
        tval += 55.0
        self.failUnless(n.data[0,0] == 56.0)

    def test3(self):
        n = tinit(numpy.ones((5,3,2)))
        t = transpose(n)
        self.failUnless(t.owner.__class__ is Transpose)
        f = Function([n], [t])
        tval = f(n.data)
        self.failUnless(tval.shape == (2,3,5))
        #test aliasing
        tval += 55.0
        self.failUnless(n.data[0,0,0] == 56.0)

class T_subtensor(unittest.TestCase):
    def test0_err_invalid(self):
        #it is impossible to retrieve a view of a 0-d tensor
        n = tinit(numpy.ones(()))
        try:
            t = n[0]
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is Subtensor.e_invalid)
    def test1_err_bounds(self):
        n = tinit(numpy.ones(3))
        t = n[7]
        self.failUnless(t.owner.__class__ is Subtensor)
        try:
            tval = eval_outputs([t])
        except Exception, e:
            if e[0] != 'index out of bounds':
                raise
    def test1_ok_range_finite(self):
        n = tinit(numpy.ones(3)*5)
        t = n[0:2]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test2_ok_range_finite(self):
        n = tinit(numpy.ones((3,4))*5)
        t = n[0:2,3]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    if 0:
        def test1_err_invalid(self):
            n = tinit(numpy.ones(1))
            try:
                t = n[0,0]
                self.fail()
            except ValueError, e:
                self.failUnless(e[0] is Subtensor.e_invalid)
        def test1_ok_elem(self):
            n = tinit(numpy.ones(1)*5)
            t = n[0]
            self.failUnless(t.owner.__class__ is Subtensor)
            tval = eval_outputs([t])
            self.failUnless(tval.shape == (1,))
            self.failUnless(tval[0] == 5.0)
        def test1_ok_range_infinite(self):
            n = tinit(numpy.ones(3)*5)
            t = n[1:]
            self.failUnless(t.owner.__class__ is Subtensor)
            tval = eval_outputs([t])
            self.failUnless(tval.shape == (2,))
            self.failUnless(tval[1] == 5.0)
        def test1_ok_strided(self):
            n = tinit(numpy.ones(5)*5)
            t = n[1::2]
            self.failUnless(t.owner.__class__ is Subtensor)
            tval = eval_outputs([t])
            self.failUnless(tval.shape == (3,))
            self.failUnless(tval[1] == 5.0)

            tval = eval_outputs([n[1:-1:2]])
            self.failUnless(tval.shape == (3,))
            self.failUnless(tval[1] == 5.0)

    def test2(self):
        raise NotImplementedError() #remember to bring back the rest of tests
    if 0:
        def test2_err_bounds0(self):
            raise NotImplementedError()
        def test2_err_bounds1(self):
            raise NotImplementedError()
        def test2_ok_elem(self):
            raise NotImplementedError()
        def test2_ok_row(self):
            raise NotImplementedError()
        def test2_ok_col(self):
            raise NotImplementedError()
        def test2_ok_rows_finite(self):
            raise NotImplementedError()
        def test2_ok_cols_infinite(self):
            raise NotImplementedError()
        def test2_ok_strided(self):
            raise NotImplementedError()

        def test3_ok_mat(self):
            raise NotImplementedError()


class T_add(unittest.TestCase):

    def test_complex_all_ops(self):
        for nbits in (64, 128):
            a = tinit(numpy.ones(3, dtype='complex%i' % nbits)+0.5j)
            b = tinit(numpy.ones(3, dtype='complex%i' % nbits)+1.5j)
            tests = (("+", lambda x,y: x+y),
                     ("-", lambda x,y: x-y),
                     ("*", lambda x,y: x*y),
                     ("/", lambda x,y: x/y))
            for s, fn in tests:
                f = Function([a,b], [fn(a, b)], linker_cls = gof.CLinker)
                self.failUnless(numpy.all(fn(a.data, b.data) == f(a.data, b.data)))


class T_abs(unittest.TestCase):
    def test_impl(self):
        t = tinit(1.0)
        check_eq(self, t, abs(t), 1.0, 1.0)
        check_eq(self, t, abs(t), -1.0, 1.0)

        for shape in (2,), (3,4):
            t = tinit(numpy.ones(shape))
            d = numpy.random.rand(*shape)*2-1.0
            check_eq(self, t, abs(t), d, abs(d))
            check_eq(self, t, abs(t), -d, abs(-d))

    def test_grad(self):
        verify_grad(self, Abs, [numpy.ones(())])
        verify_grad(self, Abs, [numpy.ones(3)])

    class AbsBadGrad(tensor._Elemwise):
        def impl(self, x):
            return numpy.abs(x)
        def grad(self, x, gz):
            return scale(gz * sgn(x),0.9)
        def c_foreach(self, (x_i, ), (z_i, )):
            return "z_i = abs(x_i);"

    def test_badgrad(self):
        try:
            verify_grad(self, T_abs.AbsBadGrad, [numpy.ones(())])
            self.fail()
        except Exception, e:
            self.failUnless(str(e) == verify_grad.E_grad, str(e))

class T_fill(unittest.TestCase):
    def test0(self):
        t = fill(numpy.asarray([1,2,3]), 9)
        self.failUnless(t.owner.__class__ == Fill)
        o = t.owner
        self.failUnless(o.inputs[0].broadcastable == (0,))
#        self.failUnless(o.inputs[0].dtype[0:3] == 'int')
        self.failUnless(o.inputs[1].broadcastable == ())
#        self.failUnless(o.inputs[1].dtype[0:3] == 'flo')
        self.failUnless(o.outputs[0].broadcastable == (0,))
#        self.failUnless(o.outputs[0].dtype[0:3] == 'flo')

class T_sum(unittest.TestCase):
    def test_impl(self):
        t = tinit(0.0)
        check_eq(self, t, Sum(t).out, 1.0, 1.0)
        check_eq(self, t, Sum(t).out, -1.0, -1.0)

        t = tinit([0.0, 0.0])
        d = numpy.asarray([-0.4, 1.2])
        check_eq(self, t, Sum(t).out, d, numpy.sum(d))
        check_eq(self, t, Sum(t).out, -d, -numpy.sum(d))

class T_mul(unittest.TestCase):
    def setUp(self):
        numpy.random.seed([1,2,3,4])

    def test_elemwise(self):
        a = tinit(0.0)
        b = tinit(0.0)
        check_eq2(self, [a,b], mul_elemwise(a,b), [3.0, 4.0], 12.0)
        check_eq2(self, [a,b], mul_elemwise(b,a), [-1.0,2.0], -2.0)
        self.failUnless(isinstance(mul(a,b).owner, Scale))

        a = tinit(numpy.ones(2))
        b = tinit(numpy.ones(2))
        aa = numpy.asarray([-0.5, 4.0])
        bb = numpy.asarray([-0.5, 2.0])
        check_eq2(self, [a,b], mul_elemwise(a,b), [aa,bb], numpy.asarray([0.25, 8.0]))
        check_eq2(self, [a,b], mul_elemwise(a,b), [bb,aa], numpy.asarray([0.25, 8.0]))
        self.failUnless(isinstance(mul(a,b).owner, MulElemwise))

    def test_scalar(self):
        r = numpy.random.rand(2,3)
        a = tinit(r)
        b = tinit(2.0)
        check_eq2(self, [a,b], scale(a,b), [r, 2.0], r*2.0)
        check_eq2(self, [a,b], scale(a,b), [r, 4.0], r*4.0)
        self.failUnless(b.data == 2.0)

    def test_operator(self):
        a = tinit([1,1])
        aa = tinit([1,1])
        b = tinit(4)
        self.failUnless(isinstance((a*b).owner, Scale))
        self.failUnless(isinstance((b*a).owner, Scale))
        self.failUnless(isinstance((a*aa).owner, MulElemwise))
        self.failUnless(isinstance((aa*a).owner, MulElemwise))

    def test_wrong_shapes(self):
        a = tinit(numpy.ones(3))
        b = tinit(numpy.ones(4))
        try:
            check_eq2(self, [a,b], MulElemwise(a,b).out,
                    [numpy.ones(3), numpy.ones(4)], 1.0)
            self.fail()
        except ValueError, e:
            self.failUnless(e[0] is tensor._assert_same_shapes.E_shape)

class T_div(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(9999)
    def test_grad_e(self):
        verify_grad(self, DivElemwise, [numpy.ones(()), numpy.ones(())])
        verify_grad(self, DivElemwise, [numpy.random.rand(3), numpy.ones(3)])
        verify_grad(self, DivElemwise, [numpy.random.rand(3,5), numpy.random.rand(3,5)+0.1])

    def test_grad_sl(self):
        verify_grad(self, DivElemwise, [numpy.ones(()), numpy.ones(())])
        verify_grad(self, DivElemwise, [numpy.random.rand(3), numpy.ones(3)])
        verify_grad(self, DivElemwise, [numpy.random.rand(3,5), numpy.random.rand(3,5)+0.1])


class T_pow(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(9999)
    def test_elemwise(self):
        verify_grad(self, DivElemwise, [numpy.random.rand(3,4), numpy.random.rand(3,4)+0.1])
        verify_grad(self, PowElemwise, [numpy.random.rand(3,4), numpy.random.rand(3,4)])
    def test_scalar_l(self):
        verify_grad(self, PowScalarL, [numpy.random.rand(3), numpy.asarray(3.0)])
    def test_scalar_r(self):
        verify_grad(self, PowScalarR, [numpy.random.rand(3), numpy.asarray(3.0)])

class _testCase_matinv:#(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(1)

    def mat_recip(self,dim):
        # symbolic program
        a = Tensor('float64', [0,0], name='a')
        b = Tensor('float64', [0,0], name='b')
        ab = a*b
        diff = ab - tinit(numpy.ones((dim,dim)))
        ssdiff = sum((diff**2.0))
        g_b = gradient.grad(ssdiff, b, tinit(numpy.ones(1),name='g_cost'))

        # compilation to function
        fn = Function([a,b], [ssdiff,g_b])

        # use the function
        w = numpy.random.rand(dim,dim)
        wi = numpy.random.rand(dim,dim)
        for i in xrange(300):
            ssd, gw = fn(w,wi)
            #print ssd
            if i == 0:
                str0 = str(ssd)
            wi -= 0.4 * gw

        return str0, str(ssd)

    def test_recip(self):
        """Matrix reciprocal by gradient descent"""
        self.assertEqual(('2.67327580893', '0.000438649434819'), self.mat_recip(3))

if __name__ == '__main__':
    unittest.main()
