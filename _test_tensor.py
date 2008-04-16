from tensor import *
import tensor # for hidden symbols

import unittest
from copy import copy
from compile import Function, eval_outputs
import gradient
import gof, gof.graph
from gof.python25 import any
import gof
from gof.utils import AbstractFunctionError

from elemwise import DimShuffle

def _numpy_checker(x, y):
    """
    Checks if x.data and y.data have the same contents.
    Used in DualLinker to compare C version with Python version.
    """
    x, y = x.data, y.data
    if x.dtype != y.dtype or x.shape != y.shape or numpy.any(abs(x - y) > 1e-10):
        raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})


def make_tester(name, op_class, expected, checks = {}, good = {}, bad_build = {}, bad_runtime = {}, grad = None):
    if grad is None:
        grad = good

    _op_class, _expected, _checks, _good, _bad_build, _bad_runtime, _grad = op_class, expected, checks, good, bad_build, bad_runtime, grad
    
    class Checker(unittest.TestCase):

        op_class = _op_class
        expected = staticmethod(_expected)
        checks = _checks
        good = _good
        bad_build = _bad_build
        bad_runtime = _bad_runtime
        grad = _grad

        def test_good(self):
            for testname, inputs in self.good.items():
                inputs = [copy(input) for input in inputs]
                try:
                    op = self.op_class(*inputs)
                except:
                    type, value, traceback = sys.exc_info()
                    err_msg = "With data %s::%s: This error occurred while trying to build a %s instance with inputs %s" \
                        % (self.op_class.__name__, testname, self.op_class, inputs)
                    value.args = value.args + (err_msg, )
                    raise type, value, traceback

                try:
                    f = Function(op.inputs, op.outputs,
                                 linker_cls = lambda env: gof.DualLinker(env, checker = _numpy_checker),
                                 unpack_single = False,
                                 optimizer = None)
                except:
                    type, value, traceback = sys.exc_info()
                    err_msg = "With data %s::%s: This error occurred while trying to make a function out of %s" \
                        % (self.op_class.__name__, testname, op)
                    value.args = value.args + (err_msg, )
                    raise type, value, traceback

                expecteds = self.expected(*inputs)
                
                try:
                    results = f(*inputs)
                except:
                    type, value, traceback = sys.exc_info()
                    err_msg = "With data %s::%s: This error occurred while calling %s on the inputs %s" \
                        % (self.op_class.__name__, testname, op, inputs)
                    value.args = value.args + (err_msg, )
                    raise type, value, traceback

                if not isinstance(expecteds, (list, tuple)):
                    expecteds = (expecteds, )
                for i, (result, expected) in enumerate(zip(results, expecteds)):
                    if result.dtype != expected.dtype or result.shape != expected.shape or numpy.any(abs(result - expected) > 1e-10):
                        self.fail("With data %s::%s: Output %s of %s gave the wrong value. With inputs %s, expected %s, got %s."
                                  % (self.op_class.__name__, testname, i, op, inputs, expected, result))

                for description, check in self.checks.items():
                    if not check(inputs, results):
                        self.fail("With data %s::%s: %s failed the following check: %s (inputs were %s)"
                                  % (self.op_class.__name__, testname, op, description, inputs))

        def test_bad_build(self):
            for testname, inputs in self.bad_build.items():
                inputs = [copy(input) for input in inputs]
                try:
                    op = self.op_class(*inputs)
                except:
                    return
                self.fail("With data %s::%s: %s was successfully instantiated on the following bad inputs: %s"
                          % (self.op_class.__name__, testname, op, inputs))

        def test_bad_runtime(self):
            for testname, inputs in self.bad_runtime.items():
                inputs = [copy(input) for input in inputs]
                try:
                    op = self.op_class(*inputs)
                except:
                    type, value, traceback = sys.exc_info()
                    err_msg = "With data %s::%s: This error occurred while trying to build a %s instance with inputs %s" \
                        % (self.op_class.__name__, testname, self.op_class, inputs)
                    value.args = value.args + (err_msg, )
                    raise type, value, traceback

                try:
                    f = Function(op.inputs, op.outputs,
                                 linker_cls = lambda env: gof.DualLinker(env, checker = _numpy_checker),
                                 unpack_single = False,
                                 optimizer = None)
                except:
                    type, value, traceback = sys.exc_info()
                    err_msg = "With data %s::%s: This error occurred while trying to make a function out of %s" \
                        % (self.op_class.__name__, testname, op)
                    value.args = value.args + (err_msg, )
                    raise type, value, traceback

                try:
                    results = f(*inputs)
                except:
                    return
                
                self.fail("With data %s::%s: %s was successfully called on the following bad inputs: %s"
                          % (self.op_class.__name__, testname, op, inputs))

        def test_grad(self):
            for testname, inputs in self.grad.items():
                inputs = [copy(input) for input in inputs]
                try:
                    verify_grad(self, self.op_class, inputs)
                except:
                    type, value, traceback = sys.exc_info()
                    err_msg = "With data %s::%s: This error occurred while computing the gradient for %s" \
                        % (self.op_class.__name__, testname, self.op_class)
                    value.args = value.args + (err_msg, )
                    raise type, value, traceback

    Checker.__name__ = name
    return Checker
            

rand = lambda *shape: 2 * numpy.random.rand(*shape) - 1
randint = lambda *shape: numpy.random.random_integers(-10, 10, shape)

def randint_notzero(*shape):
    r = numpy.random.random_integers(-10, 9, shape)
    return r + (r == 0) * 10

# randplus = numpy.random.rand
# randintplus = 

# def banzero(f):
#     def f2(*shape):
#         res = f(*shape)
#         while numpy.any(res == 0):
#             res = f(*shape)
#         return res
#     return f2

# def banneg(f):
#     def f2(*shape):
#         res = f(*shape)
#         while numpy.any(res < 0):
#             res = f(*shape)
#         return res
#     return f2


def make_broadcast_tester(op_class, expected, checks = {}, **kwargs):

    _randint = randint
    _rand = rand
    if kwargs.has_key('nonzero'):
        if kwargs['nonzero']:
            _randint = banzero(_randint)
            _rand = banzero(_rand)
        del kwargs['nonzero']

    if kwargs.has_key('positive'):
        if kwargs['positive']:
            _randint = banneg(_randint)
            _rand = banneg(_rand)
        del kwargs['positive']

    _good_broadcast = dict(same_shapes = (_rand(2, 3), _rand(2, 3)),
                           scalar = (_rand(2, 3), _rand(1, 1)),
                           row = (_rand(2, 3), _rand(1, 3)),
                           column = (_rand(2, 3), _rand(2, 1)),
                           integers = (_randint(2, 3), _randint(2, 3)),
                           dtype_mixup_1 = (_rand(2, 3), _randint(2, 3)),
                           dtype_mixup_2 = (_randint(2, 3), _rand(2, 3)))

    _bad_build_broadcast = dict(not_same_dimensions = (_rand(2), _rand(2, 2)))

    _bad_runtime_broadcast = dict(bad_shapes = (_rand(2, 3), _rand(3, 2)),
                                  bad_row = (_rand(2, 3), _rand(1, 2)))

    _grad_broadcast = dict(same_shapes = (_rand(2, 3), _rand(2, 3)),
                           scalar = (_rand(2, 3), _rand(1, 1)),
                           row = (_rand(2, 3), _rand(1, 3)),
                           column = (_rand(2, 3), _rand(2, 1)))
    
    kwargs.setdefault('good', _good_broadcast)
    kwargs.setdefault('bad_build', _bad_build_broadcast)
    kwargs.setdefault('bad_runtime', _bad_runtime_broadcast)
    kwargs.setdefault('grad', _grad_broadcast)
    name = op_class.__name__ + "Tester"
    if kwargs.has_key('inplace'):
        if kwargs['inplace']:
            _expected = expected
            expected = lambda *inputs: numpy.array(_expected(*inputs), dtype = inputs[0].dtype)
            checks = dict(checks,
                          inplace_check = lambda inputs, outputs: inputs[0] is outputs[0])
        del kwargs['inplace']
    return make_tester(name, op_class, expected, checks, **kwargs)


def make_broadcast_tester_unary(op_class, expected, checks = {}, **kwargs):

    _randint = randint
    _rand = rand
    if kwargs.has_key('nonzero'):
        if kwargs['nonzero']:
            _randint = banzero(_randint)
            _rand = banzero(_rand)
        del kwargs['nonzero']

    if kwargs.has_key('positive'):
        if kwargs['positive']:
            _randint = banneg(_randint)
            _rand = banneg(_rand)
        del kwargs['positive']

    _good_broadcast = dict(normal = (_rand(2, 3), ),
                           int = (_rand(2, 3), ))

    _bad_build_broadcast = dict()

    _bad_runtime_broadcast = dict()

    _grad_broadcast = dict(normal = (_rand(2, 3), ),
                           int = (_rand(2, 3), ))
    
    kwargs.setdefault('good', _good_broadcast)
    kwargs.setdefault('bad_build', _bad_build_broadcast)
    kwargs.setdefault('bad_runtime', _bad_runtime_broadcast)
    kwargs.setdefault('grad', _grad_broadcast)
    name = op_class.__name__ + "Tester"
    if kwargs.has_key('inplace'):
        if kwargs['inplace']:
            _expected = expected
            expected = lambda *inputs: numpy.array(_expected(*inputs), dtype = inputs[0].dtype)
            checks = dict(checks,
                          inplace_check = lambda inputs, outputs: inputs[0] is outputs[0])
        del kwargs['inplace']
    return make_tester(name, op_class, expected, checks, **kwargs)





# AddTester = make_broadcast_tester(op_class = Add,
#                                   expected = lambda x, y: x + y,
#                                   grad = {})
# AddInplaceTester = make_broadcast_tester(op_class = AddInplace,
#                                          expected = lambda x, y: x + y,
#                                          inplace = True,
#                                          grad = {})

# SubTester = make_broadcast_tester(op_class = Sub,
#                                   expected = lambda x, y: x - y)
# SubInplaceTester = make_broadcast_tester(op_class = SubInplace,
#                                          expected = lambda x, y: x - y,
#                                          inplace = True)

# MulTester = make_broadcast_tester(op_class = Mul,
#                                   expected = lambda x, y: x * y)
# MulInplaceTester = make_broadcast_tester(op_class = MulInplace,
#                                          expected = lambda x, y: x * y,
#                                          inplace = True)

# DivTester = make_broadcast_tester(op_class = Div,
#                                   expected = lambda x, y: x / y,
#                                   nonzero = True,
#                                   positive = True)
# DivInplaceTester = make_broadcast_tester(op_class = DivInplace,
#                                          expected = lambda x, y: x / y,
#                                          inplace = True,
#                                          nonzero = True,
#                                          positive = True)

# _pow_good = dict(normal = ()

# _pow_grad = 

# PowTester = make_broadcast_tester(op_class = Pow,
#                                   expected = lambda x, y: x ** y,
#                                   good = _pow_good)
# PowInplaceTester = make_broadcast_tester(op_class = PowInplace,
#                                          expected = lambda x, y: x ** y,
#                                          good = _pow_good)


AbsTester = make_broadcast_tester_unary(op_class = Abs,
                                        expected = lambda x: abs(x))
AbsInplaceTester = make_broadcast_tester_unary(op_class = AbsInplace,
                                               expected = lambda x: abs(x),
                                               inplace = True)

# ExpTester = make_broadcast_tester(op_class = Exp,
#                                   expected = lambda x: numpy.exp(x))
# ExpInplaceTester = make_broadcast_tester(op_class = ExpInplace,
#                                          expected = lambda x: numpy.exp(x),
#                                          inplace = True)



# Abs, _abs, AbsInplace, abs_inplace = broadcast(scal.Abs, 'Abs')
# Exp, exp, ExpInplace, exp_inplace = broadcast(scal.Exp, 'Exp')
# Neg, neg, NegInplace, neg_inplace = broadcast(scal.Neg, 'Neg')
# Log, log, LogInplace, log_inplace = broadcast(scal.Log, 'Log')
# Log2, log2, Log2Inplace, log2_inplace = broadcast(scal.Log2, 'Log2')
# Sgn, sgn, SgnInplace, sgn_inplace = broadcast(scal.Sgn, 'Sgn')
# Sqr, sqr, SqrInplace, sqr_inplace = broadcast(scal.Sqr, 'Sqr')
# Sqrt, sqrt, SqrtInplace, sqrt_inplace = broadcast(scal.Sqrt, 'Sqrt')


# AddTester = make_broadcast_tester(op_class = Add,
#                                   expected = lambda x, y: x + y,
#                                   checks = {},
#                                   grad = {})
# AddInplaceTester = make_broadcast_tester(op_class = AddInplace,
#                                          expected = lambda x, y: numpy.array(x + y, dtype = x.dtype),
#                                          checks = dict(inplace_check = lambda (x, y), (z, ): x is z),
#                                          grad = {})

# AddTester = make_tester(name = 'AddTester',
#                         op_class = Add,
#                         expected = lambda x, y: x + y,
#                         checks = {},
#                         good = dict(same_shapes = (rand(5, 6), rand(5, 6)),
#                                     scalar = (rand(5, 6), rand(1, 1)),
#                                     row = (rand(5, 6), rand(1, 6)),
#                                     column = (rand(5, 6), rand(5, 1)),
#                                     integers = (randint(5, 6), randint(5, 6)),
#                                     dtype_mixup = (rand(5, 6), randint(5, 6))),
#                         bad_build = dict(not_same_dimensions = (rand(5), rand(5, 5))),
#                         bad_runtime = dict(bad_shapes = (rand(5, 6), rand(6, 5)),
#                                            bad_row = (rand(5, 6), rand(1, 5))),
#                         grad = {})

# AddInplaceTester = make_tester(name = 'AddInplaceTester',
#                                op_class = AddInplace,
#                                expected = lambda x, y: numpy.array(x + y, dtype = x.dtype),
#                                checks = dict(inplace_check = lambda (x, y), (z, ): x is z),
#                                good = dict(same_shapes = (rand(5, 6), rand(5, 6)),
#                                            dtype_mixup = (randint(5, 6), rand(5, 6))),
#                                bad_build = dict(not_same_dimensions = (rand(5), rand(5, 5))),
#                                bad_runtime = dict(bad_shapes = (rand(5, 6), rand(6, 5)),
#                                                   bad_row = (rand(5, 6), rand(1, 5))),
#                                grad = {})

DotTester = make_tester(name = 'DotTester',
                        op_class = Dot,
                        expected = lambda x, y: numpy.dot(x, y),
                        checks = {},
                        good = dict(correct1 = (rand(5, 7), rand(7, 5)),
                                    correct2 = (rand(5, 7), rand(7, 9))),
                        bad_build = dict(),
                        bad_runtime = dict(bad1 = (rand(5, 7), rand(5, 7)),
                                           bad2 = (rand(5, 7), rand(8, 3))))



#TODO: consider moving this function / functionality to gradient.py
#      rationale: it's tricky, and necessary everytime you want to verify
#      gradient numerically

def verify_grad(testcase, op_cls, pt, n_tests=1, rng=numpy.random, eps=0.0000001, tol=0.0001):
    """testcase.failUnless( analytic gradient matches finite-diff gradient) """
    pt = [numpy.asarray(p) for p in pt]

    for test_num in xrange(n_tests):
        tensor_pt = [astensor(p,name='input %i'%i) for i,p in enumerate(pt)]
        o = op_cls(*[tpt.copy() for tpt in tensor_pt])
        if hasattr(o, 'outputs'):
            o_outputs = o.outputs
        else:
            o_outputs = o

        if len(o_outputs) > 1:
            raise NotImplementedError('cant (yet) autotest gradient of op with multiple outputs')
            # we could make loop over outputs making random projections R for each,
            # but this doesn't handle the case where not all the outputs are
            # differentiable... so I leave this as TODO for now -JB.
        o_fn = Function(tensor_pt, o_outputs)
        o_fn_out = o_fn(*pt)
        random_projection = rng.rand(*o_fn_out.shape)
        t_r = astensor(random_projection)

        #random projection of o onto t_r
        cost = sum(t_r * o_outputs[0])
        cost_fn = Function(tensor_pt, [cost])

        num_grad = gradient.numeric_grad(cost_fn, pt)

        symbolic_grad = gradient.grad(cost, tensor_pt,astensor(1.0,name='g_cost'))
        if 0:
            print '-------'
            print '----------'
            for op in gof.graph.io_toposort(tensor_pt, symbolic_grad):
                print op
        try:
            grad_fn = Function(tensor_pt, symbolic_grad)
        except:
            print gof.graph.as_string(tensor_pt, symbolic_grad)
            raise
        
        analytic_grad = grad_fn(*pt)
        if not isinstance(analytic_grad, (list, tuple)):
            analytic_grad = [analytic_grad]

#         if num_grad.max_err(analytic_grad) > 1.0e-4:
#             print "aaaaaaaaaa"
#             print gof.Env(tensor_pt, [cost])
#             print gof.Env(tensor_pt, symbolic_grad)
#             print analytic_grad
#             print num_grad.gf
#             print num_grad.max_err(analytic_grad)
#             print "bbbbbbbbbb"

        if num_grad.max_err(analytic_grad) > 1.0e-4:
            raise Exception(verify_grad.E_grad)
verify_grad.E_grad = 'gradient error exceeded tolerance'


#useful mostly for unit tests
def _approx_eq(a,b,eps=1.0e-9):
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    if a.shape != b.shape:
        if _approx_eq.debug:
            print a.shape, b.shape
        return False
    if numpy.max(numpy.abs(a-b)) >= eps:
        if _approx_eq.debug:
            print a, b
        return False
    return  True
_approx_eq.debug = 0

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

def check_eq2_both(self, inputs, output, args_in, arg_out):
    fn = Function(inputs, [output], linker_cls = lambda env: gof.DualLinker(env, _numpy_checker))
    val = fn(*args_in)
    self.failUnless( numpy.all(val == arg_out), (val, arg_out))

class T_argmax(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(123784)
        Argmax.debug = 0

    def test0(self):
        n = astensor(5.0)
        v,i = eval_outputs(argmax(n))
        self.failUnless(v == 5.0)
        self.failUnless(i == 0)

    def test1(self):
        n = astensor([1,2,3,2,-6])
        v,i = eval_outputs(argmax(n))
        self.failUnless(v == 3)
        self.failUnless(i == 2)

    def test2(self):
        n = astensor(numpy.random.rand(2,3))
        v,i = eval_outputs(argmax(n))
        self.failUnless(numpy.all(i == [0,1]))
    def test2b(self):
        n = astensor(numpy.random.rand(2,3))
        v,i = eval_outputs(argmax(n,axis=0))
        self.failUnless(numpy.all(i == [0,1,1]))
    def test2_invalid(self):
        n = astensor(numpy.random.rand(2,3))
        try:
            eval_outputs(argmax(n,axis=3))
        except ValueError, e:
            return
        self.fail()
    def test2_invalid_neg(self):
        n = astensor(numpy.random.rand(2,3))
        try:
            eval_outputs(argmax(n,axis=-3))
        except ValueError, e:
            return
        self.fail()
    def test2_valid_neg(self):
        n = astensor(numpy.random.rand(2,3))
        v,i = eval_outputs(argmax(n,axis=-1))
        self.failUnless(v.shape == (2,))
        v,i = eval_outputs(argmax(n,axis=-2))
        self.failUnless(v.shape == (3,))
    def test3(self):
        n = astensor(numpy.random.rand(2,3,4))
        v,i = eval_outputs(argmax(n,axis=0))
        self.failUnless(v.shape == (3,4))
        self.failUnless(i.shape == (3,4))
        v,i = eval_outputs(argmax(n,axis=1))
        self.failUnless(v.shape == (2,4))
        self.failUnless(i.shape == (2,4))
        v,i = eval_outputs(argmax(n,axis=2))
        self.failUnless(v.shape == (2,3))
        self.failUnless(i.shape == (2,3))


class T_transpose(unittest.TestCase):
    def test0(self):
        n = astensor(numpy.ones(()))
        t = transpose(n)
        self.failUnless(t.owner.__class__ is TransposeInplace)
        f = Function([n], [t])
        tval = f(n.data)
        self.failUnless(tval.shape == n.data.shape)

        #test aliasing
        tval += 55.0
        self.failUnless(n.data == 1.0)
        
    def test1(self):
        n = astensor(numpy.ones(5))
        t = transpose(n)
        self.failUnless(t.owner.__class__ is TransposeInplace)
        f = Function([n], [t])
        tval = f(n.data)
        self.failUnless(tval.shape == n.data.shape)
        #test aliasing
        tval += 55.0
        self.failUnless(n.data[0] == 1.0)
        
    def test2(self):
        n = astensor(numpy.ones((5,3)))
        t = transpose(n)
        self.failUnless(t.owner.__class__ is TransposeInplace)
        f = Function([n], [t])
        tval = f(n.data)
        self.failUnless(tval.shape == (3,5))
        #test aliasing
        tval += 55.0
        self.failUnless(n.data[0,0] == 1.0)

    def test3(self):
        """Test transpose of tensor, inplace version"""
        n = astensor(numpy.ones((5,3,2)))
        t = transpose_inplace(n)
        self.failUnless(t.owner.__class__ is TransposeInplace)
        f = Function([n], [t])
        tval = f(n.data)
        self.failUnless(tval.shape == (2,3,5))
        #test aliasing
        tval += 55.0
        self.failUnless(n.data[0,0,0] == 56.0)
    def test_grad(self):
        verify_grad(self, TransposeInplace, [numpy.random.rand(2, 3)])
        verify_grad(self, TransposeInplace, [numpy.ones(3)])

class T_subtensor(unittest.TestCase):
    def test0_err_invalid(self):
        #it is impossible to retrieve a view of a 0-d tensor
        n = astensor(numpy.ones(()))
        try:
            t = n[0]
        except ValueError, e:
            self.failUnless(e[0] is Subtensor.e_invalid)
            return
        self.fail()
    def test1_err_bounds(self):
        n = astensor(numpy.ones(3))
        t = n[7]
        self.failUnless(t.owner.__class__ is Subtensor)
        try:
            tval = eval_outputs([t])
        except Exception, e:
            if e[0] != 'index out of bounds':
                raise
            return
        self.fail()
    def test1_ok_range_finite(self):
        n = astensor(numpy.ones(3)*5)
        t = n[0:2]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test2_ok_range_finite(self):
        n = astensor(numpy.ones((3,4))*5)
        t = n[0:2,3]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test1_err_invalid(self):
        n = astensor(numpy.ones(1))
        try:
            t = n[0,0]
        except ValueError, e:
            self.failUnless(e[0] is Subtensor.e_invalid)
            return
        self.fail()
    def test1_ok_elem(self):
        n = astensor(numpy.ones(1)*5)
        t = n[0]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == ())
        self.failUnless(tval == 5.0)
    def test1_ok_range_infinite(self):
        n = astensor(numpy.ones(3)*5)
        t = n[1:]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test1_ok_strided(self):
        n = astensor(numpy.ones(5)*5)
        t = n[1::2]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)

        tval = eval_outputs([n[0:-1:2]]) #0 to 1 from the end stepping by 2
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)

    def test2_err_bounds0(self):
        n = astensor(numpy.ones((2,3))*5)
        t = n[0,4]
        self.failUnless(t.owner.__class__ is Subtensor)
        try:
            tval = eval_outputs([t])
        except IndexError, e:
            return
        self.fail()
    def test2_err_bounds1(self):
        n = astensor(numpy.ones((2,3))*5)
        t = n[4:5,2]
        self.failUnless(t.owner.__class__ is Subtensor)
        try:
            tval = eval_outputs([t])
        except Exception, e:
            if e[0] != 'index out of bounds':
                raise
    def test2_ok_elem(self):
        n = astensor(numpy.asarray(range(6)).reshape((2,3)))
        t = n[0,2]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == ())
        self.failUnless(numpy.all(tval == 2))
    def test2_ok_row(self):
        n = astensor(numpy.asarray(range(6)).reshape((2,3)))
        t = n[1]
        self.failIf(any(n.broadcastable))
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (3,))
        self.failUnless(numpy.all(tval == [3,4,5]))

    def test2_ok_col(self):
        n = astensor(numpy.ones((2,3))*5)
        t = n[:,0]
        self.failUnless(t.owner.__class__ is Subtensor)
        self.failIf(any(n.broadcastable))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(numpy.all(tval == 5.0))

    def test2_ok_rows_finite(self):
        n = astensor(numpy.ones((4,3))*5)
        t = n[1:3,0]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(numpy.all(tval == 5.0))

    def test2_ok_cols_infinite(self):
        n = astensor(numpy.asarray(range(12)).reshape((4,3)))
        t = n[1,2:]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (1,))
        self.failUnless(numpy.all(tval == 5))

    def test2_ok_strided(self):
        n = astensor(numpy.asarray(range(20)).reshape((4,5)))
        t = n[1:4:2,1:5:2]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,2))
        self.failUnless(numpy.all(tval == [[6, 8],[16, 18]]))

    def test3_ok_mat(self):
        n = astensor(numpy.asarray(range(24)).reshape((2,3,4)))
        t = n[0,0,0]
        self.failUnless(t.owner.__class__ is Subtensor)
        tval = eval_outputs([t])
        self.failUnless(tval.shape == ())
        self.failUnless(numpy.all(tval == 0))


class T_add(unittest.TestCase):

    def test_complex_all_ops(self):
        for nbits in (64, 128):
            a = astensor(numpy.ones(3, dtype='complex%i' % nbits)+0.5j)
            b = astensor(numpy.ones(3, dtype='complex%i' % nbits)+1.5j)
            tests = (("+", lambda x,y: x+y),
                     ("-", lambda x,y: x-y),
                     ("*", lambda x,y: x*y),
                     ("/", lambda x,y: x/y))
            for s, fn in tests:
                f = Function([a,b], [fn(a, b)], linker_cls = gof.CLinker)
                self.failUnless(numpy.all(fn(a.data, b.data) == f(a.data, b.data)))

    def test_grad_scalar_l(self):
        verify_grad(self, Add, [numpy.asarray([3.0]), numpy.random.rand(3)])
    def test_grad_scalar_r(self):
        verify_grad(self, Add, [numpy.random.rand(3), numpy.asarray([3.0])])
    def test_grad_row(self):
        verify_grad(self, Add, [numpy.random.rand(3, 5), numpy.random.rand(1, 5)])
    def test_grad_col(self):
        verify_grad(self, Add, [numpy.random.rand(3, 5), numpy.random.rand(3, 1)])


class T_abs(unittest.TestCase):
    def test_impl(self):
        t = astensor(1.0)
        check_eq(self, t, abs(t), 1.0, 1.0)
        check_eq(self, t, abs(t), -1.0, 1.0)

        for shape in (2,), (3,4):
            t = astensor(numpy.ones(shape))
            d = numpy.random.rand(*shape)*2-1.0
            check_eq(self, t, abs(t), d, abs(d))
            check_eq(self, t, abs(t), -d, abs(-d))

    def test_grad(self):
        verify_grad(self, Abs, [numpy.ones(())])
        verify_grad(self, Abs, [numpy.ones(3)])

    class AbsBadGrad(Abs):
        def grad(self, (x, ), (gz, )):
            return mul(gz * sgn(x),0.9),

    def test_badgrad(self):
        try:
            verify_grad(self, T_abs.AbsBadGrad, [numpy.ones(())])
        except Exception, e:
            self.failUnless(str(e) == verify_grad.E_grad, str(e))
            return
        self.fail()

class T_fill(unittest.TestCase):
    def test0(self):
        t = fill(numpy.asarray([1,2,3]), 9)
        self.failUnless(t.owner.__class__ == Fill)
        o = t.owner
        self.failUnless(o.inputs[0].broadcastable == (0,))
#        self.failUnless(o.inputs[0].dtype[0:3] == 'int')
        self.failUnless(o.inputs[1].broadcastable == (1,))
#        self.failUnless(o.inputs[1].dtype[0:3] == 'flo')
        self.failUnless(o.outputs[0].broadcastable == (0,))
#        self.failUnless(o.outputs[0].dtype[0:3] == 'flo')
        self.failUnless(numpy.all(eval_outputs([t]) == [9,9,9]))

    def test1(self):
        x = astensor(numpy.ones((4,5)))
        l = ones_like(x[:,0:1])
        r = ones_like(x[0:1,:])
        xx = x + dot(l,r)
        self.failUnless(numpy.mean(eval_outputs([xx]) == 2.0))

class T_sum(unittest.TestCase):
    def test_impl(self):
        t = astensor(0.0)
        check_eq(self, t, Sum(t).out, 1.0, 1.0)
        check_eq(self, t, Sum(t).out, -1.0, -1.0)

        t = astensor([0.0, 0.0])
        d = numpy.asarray([-0.4, 1.2])
        check_eq(self, t, Sum(t).out, d, numpy.sum(d))
        check_eq(self, t, Sum(t).out, -d, -numpy.sum(d))

class T_mul(unittest.TestCase):
    def setUp(self):
        numpy.random.seed([1,2,3,4])

    def test_elemwise(self):
        a = astensor(0.0)
        b = astensor(0.0)
        check_eq2_both(self, [a,b], mul(a,b), [3.0, 4.0], 12.0)
        check_eq2_both(self, [a,b], mul(b,a), [-1.0,2.0], -2.0)

        a = astensor(numpy.ones(2))
        b = astensor(numpy.ones(2))
        aa = numpy.asarray([-0.5, 4.0])
        bb = numpy.asarray([-0.5, 2.0])
        check_eq2_both(self, [a,b], mul(a,b), [aa,bb], numpy.asarray([0.25, 8.0]))
        check_eq2_both(self, [a,b], mul(a,b), [bb,aa], numpy.asarray([0.25, 8.0]))

    def test_scalar(self):
        r = numpy.random.rand(2,3)
        a = astensor(r)
        b = astensor(2.0)
        check_eq2_both(self, [a,b], mul(a,b), [r, 2.0], r*2.0)
        check_eq2_both(self, [a,b], mul(a,b), [r, 4.0], r*4.0)
        self.failUnless(b.data == 2.0)

    def test_rowcol(self):
        r1 = numpy.random.rand(3,5)
        r2 = numpy.random.rand(1,5)
        r3 = numpy.random.rand(3,1)
        a1, a2, a3 = astensor(r1), astensor(r2), astensor(r3)
        check_eq2_both(self, [a1,a2], mul(a1,a2), [r1, r2], r1*r2)
        check_eq2_both(self, [a1,a3], mul(a1,a3), [r1, r3], r1*r3)

    def test_grad_elemwise(self):
        verify_grad(self, Mul, [numpy.random.rand(3,4), numpy.random.rand(3,4)])
    def test_grad_scalar_l(self):
        verify_grad(self, Mul, [numpy.asarray([3.0]), numpy.random.rand(3)])
    def test_grad_scalar_r(self):
        verify_grad(self, Mul, [numpy.random.rand(3), numpy.asarray([3.0])])
    def test_grad_row(self):
        verify_grad(self, Mul, [numpy.random.rand(3, 5), numpy.random.rand(1, 5)])
    def test_grad_row2(self):
        op = lambda x, y: Mul(x, DimShuffle(y, ['x', 0]).out)
        verify_grad(self, op, [numpy.random.rand(3, 5), numpy.random.rand(5)])
    def test_grad_col(self):
        verify_grad(self, Mul, [numpy.random.rand(3, 5), numpy.random.rand(3, 1)])

    def test_wrong_shapes(self):
        a = astensor(numpy.ones(3))
        b = astensor(numpy.ones(4))
        try:
            check_eq2(self, [a,b], Mul(a,b).out,
                      [numpy.ones(3), numpy.ones(4)], 1.0)
            self.fail()
        except ValueError, e:
            self.failUnless('shape mismatch' in str(e))
        
        try:
            check_eq2_c(self, [a,b], Mul(a,b).out,
                        [numpy.ones(3), numpy.ones(4)], 1.0)
            self.fail()
        except ValueError, e:
            pass

class T_div(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(9999)
    def test_grad_e(self):
        verify_grad(self, Div, [numpy.random.rand(3), numpy.ones(3)])
        verify_grad(self, Div, [numpy.random.rand(3,5), numpy.random.rand(3,5)+0.1])
        verify_grad(self, Div, [numpy.ones(()), numpy.ones(())])

    def test_grad_sl(self):
        verify_grad(self, Div, [numpy.ones((3, 5)), numpy.ones((1, 1))])
        verify_grad(self, Div, [numpy.random.rand(3), numpy.ones((1, ))])
        verify_grad(self, Div, [numpy.random.rand(3,5), numpy.random.rand(1,1)])

class T_log2(unittest.TestCase):
    def test0(self):
        verify_grad(self, Log2, [numpy.random.rand(3,1)+0.0001])

class T_log(unittest.TestCase):
    def test0(self):
        verify_grad(self, Log, [numpy.random.rand(3,1)+0.0001])
    def test1(self):
        a = astensor(numpy.ones(2))
        b = astensor(numpy.ones(2))
        aa = numpy.asarray([0.5, 4.0])
        bb = numpy.asarray([0.5, 2.0])
        check_eq2(self, [a], log(a), [aa], numpy.log(numpy.asarray(aa)))

class T_pow(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(9999)
    def test_elemwise(self):
        verify_grad(self, Div, [numpy.random.rand(3,4), numpy.random.rand(3,4)+0.1])
        verify_grad(self, Pow, [numpy.random.rand(3,4), numpy.random.rand(3,4)])
    def test_scalar_l(self):
        verify_grad(self, Pow, [numpy.asarray([3.0]), numpy.random.rand(3)])
    def test_scalar_r(self):
        verify_grad(self, Pow, [numpy.random.rand(3), numpy.asarray([3.0])])
    def test_row(self):
        verify_grad(self, Pow, [numpy.random.rand(3, 5), numpy.random.rand(1, 5)])
    def test_col(self):
        verify_grad(self, Pow, [numpy.random.rand(3, 5), numpy.random.rand(3, 1)])

class _testCase_matinv(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(1)

    def mat_reciprocal(self,dim):
        # symbolic program
        # broadcastable=[False,False] means that the shape of matrix is two dimensional,
        # and none of the dimensions are constrained to have length 1.
        # Note that Tensor's constructor does not actually allocate any memory.
        # TODO: Make Tensor syntax more explicit, and maybe give shape or number of dimensions.
        a = Tensor('float64', broadcastable=[False,False], name='a')
        b = Tensor('float64', broadcastable=[False,False], name='b')
        ab = a*b
        # Here, astensor actually uses the data allocated by numpy.
        diff = ab - astensor(numpy.ones((dim,dim)))
        # Sum of squared errors
        ssdiff = sum((diff**2.0))

        g_b = gradient.grad(ssdiff, b)

        # compilation to function
        # [a,b] are the inputs, [ssdiff,g_b] are the outputs
        fn = Function([a,b], [ssdiff,g_b])

        # use the function
        x = numpy.random.rand(dim,dim)+0.1      # Initialized s.t. x is not too tiny
        w = numpy.random.rand(dim,dim)
        for i in xrange(300):
            ssd, gw = fn(x,w)
            #print ssd, x*w, x, w
            if i == 0:
                str0 = str(ssd)
            w -= 0.4 * gw

        return str0, str(ssd)

    def test_reciprocal(self):
        """Matrix reciprocal by gradient descent"""
        self.assertEqual(('6.10141615619', '0.00703816291711'), self.mat_reciprocal(3))

class t_dot(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)

    @staticmethod
    def rand(*args):
        return numpy.random.rand(*args)

    def cmp_dot(self,x,y):
        #x, y are matrices or numbers
        def spec(x):
            x = numpy.asarray(x)
            return type(x), x.dtype, x.shape
        nz = numpy.dot(x,y)
        tz = eval_outputs([dot(astensor(x), astensor(y))])
        self.failUnless(tz.dtype == nz.dtype)
        self.failUnless(tz.shape == nz.shape)
        self.failUnless(_approx_eq(nz, tz))

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

    def not_aligned(self, x, y):
        z = dot(x,y)
        try:
            tz = eval_outputs([z])
        except ValueError, e:
            self.failUnless(e[0].split()[1:4] == ['are', 'not', 'aligned'], e)
            return
        self.fail()

    def test_align_1_1(self): self.not_aligned(self.rand(5), self.rand(6))
    def test_align_1_2(self): self.not_aligned(self.rand(5), self.rand(6,4))
    def test_align_1_3(self): self.not_aligned(self.rand(5), self.rand(6,4,7))
    def test_align_2_1(self): self.not_aligned(self.rand(5,4), self.rand(6))
    def test_align_2_1(self): self.not_aligned(self.rand(5,4), self.rand(6,7))
    def test_align_2_3(self): self.not_aligned(self.rand(5,4), self.rand(6,7,8))
    def test_align_3_1(self): self.not_aligned(self.rand(5,4,3), self.rand(6))
    def test_align_3_2(self): self.not_aligned(self.rand(5,4,3), self.rand(6,7))
    def test_align_3_3(self): self.not_aligned(self.rand(5,4,3), self.rand(6,7,8))

    def test_grad(self):
        verify_grad(self, Dot, [self.rand(2,3), self.rand(3,2)])

class t_gemm(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)
        _approx_eq.debug = 0
        Gemm.debug = False

    @staticmethod
    def _gemm(z,a,x,y,b):
        assert a.shape == ()
        assert b.shape == ()
        return b * z + a * numpy.dot(x,y)
    @staticmethod
    def rand(*args):
        return numpy.random.rand(*args)

    def cmp(self, z, a, x, y, b):
        def cmp_linker(z, a, x, y, b, l):
            z,a,x,y,b = [numpy.asarray(p) for p in z,a,x,y,b]
            z_orig = z.copy()
            tz,ta,tx,ty,tb = [astensor(p) for p in z,a,x,y,b]

            f = Function([tz,ta,tx,ty,tb], [gemm(tz,ta,tx,ty,tb)], linker_cls=l)
            new_z = f(z,a,x,y,b)
            z_after = self._gemm(z_orig, a, x, y, b)

            self.failUnless(z is new_z)
            #print z_orig, z_after, z, type(z_orig), type(z_after), type(z)
            #_approx_eq.debug = 1
            self.failUnless(_approx_eq(z_after, z))
            if a == 0.0 and b == 1.0:
                return
            else:
                self.failIf(numpy.all(z_orig == z))

        cmp_linker(copy(z), a, x, y, b, gof.cc.OpWiseCLinker)
        #cmp_linker(copy(z), a, x, y, b, gof.cc.CLinker)
        cmp_linker(copy(z), a, x, y, b, gof.link.PerformLinker)

    def test0a(self): 
        Gemm.debug = True
        try:
            g = gemm([1.], 1., [1.], [1.], 1.)
        except ValueError, e:
            if e[0] is Gemm.E_rank:
                return
        self.fail()

    def test0(self): 
        try:
            self.cmp(1., 0., 1.0, 1.0, 1.0)
        except ValueError, e:
            if e[0] is Gemm.E_rank:
                return
        self.fail()

    def test2(self): 
        try:
            self.cmp(2., 1.0, [3,2,1.], [[1],[2],[3.]], 1.0)
        except ValueError, e:
            self.failUnless(e[0] == Gemm.E_rank)
            return
        self.fail()
    def test4(self): 
        self.cmp(self.rand(3,4), 1.0, self.rand(3,5), self.rand(5,4), 0.0)
    def test5(self): self.cmp(self.rand(3,4), 1.0,
            self.rand(3,5), self.rand(5,4), 1.0)
    def test6(self): self.cmp(self.rand(3,4), 1.0,
            self.rand(3,5), self.rand(5,4), -1.0)
    def test7(self): self.cmp(self.rand(3,4), 0.0,
            self.rand(3,5), self.rand(5,4), 0.0)
    def test8(self): self.cmp(self.rand(3,4), 0.0,
            self.rand(3,5), self.rand(5,4), 0.6)
    def test9(self): self.cmp(self.rand(3,4), 0.0,
            self.rand(3,5), self.rand(5,4), -1.0)
    def test10(self): 
        _approx_eq.debug = 1
        self.cmp(self.rand(3,4), -1.0, self.rand(3,5), self.rand(5,4), 0.0)
    def test11(self): self.cmp(self.rand(3,4), -1.0,
            self.rand(3,5), self.rand(5,4), 1.0)
    def test12(self): self.cmp(self.rand(3,4), -1.0,
            self.rand(3,5), self.rand(5,4), -1.0)

    def test_destroy_map0(self):
        """test that only first input can be overwritten"""
        Z = astensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, Z, Z, 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map1(self):
        """test that only first input can be overwritten"""
        Z = astensor(self.rand(2,2))
        A = astensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, A, transpose_inplace(Z), 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map2(self):
        """test that only first input can be overwritten"""
        Z = astensor(self.rand(2,2))
        A = astensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, transpose_inplace(Z), A, 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map3(self):
        """test that only first input can be overwritten"""
        Z = astensor(self.rand(2,2))
        A = astensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, Z, A, 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()

    def test_destroy_map4(self):
        """test that dot args can be aliased"""
        Z = astensor(self.rand(2,2))
        A = astensor(self.rand(2,2))
        eval_outputs([gemm(Z, 1.0, A, A, 1.0)])
        eval_outputs([gemm(Z, 1.0, A, A.T, 1.0)])


    def test_transposes(self):
        # three square matrices which are not contiguous
        A = self.rand(4,5)[:,:4]
        B = self.rand(4,5)[:,:4]
        C = self.rand(4,5)[:,:4]
        
        def t(z,x,y,a=1.0, b=0.0,l=gof.cc.OpWiseCLinker,dt='float64'):
            z,a,x,y,b = [numpy.asarray(p,dtype=dt) for p in z,a,x,y,b]
            z_orig = z.copy()
            z_after = self._gemm(z, a, x, y, b)

            tz,ta,tx,ty,tb = [astensor(p) for p in z,a,x,y,b]

            f = Function([tz,ta,tx,ty,tb], [gemm(tz,ta,tx,ty,tb)], linker_cls=l)
            f(z, a, x, y, b)
            self.failUnless(_approx_eq(z_after, z), (z_orig, z_after, z))
            f(z.T, a, y.T, x.T, b)
            self.failUnless(_approx_eq(z_after, z))

        t(C,A,B)
        t(C.T, A, B)
        t(C, A.T, B, dt='float32')
        t(C, A, B.T)
        t(C.T, A.T, B)
        t(C, A.T, B.T, dt='float32')
        t(C.T, A, B.T)
        t(C.T, A.T, B.T, dt='float32')

        t(C, A[:,:2], B[:2, :])
        t(C.T, A[:,:2], B[:2, :], dt='float32')
        t(C, A[:2,:].T, B[:2, :])
        t(C.T, A[:2,:].T, B[:2, :], dt='float32')
        t(C, A[:2,:].T, B[:, :2].T)
        t(C.T, A[:2,:].T, B[:, :2].T)

        try:
            t(C.T, A[:2,:], B[:, :2].T)
        except ValueError, e:
            if e[0].find('aligned') >= 0:
                return
        self.fail()


if __name__ == '__main__':
    unittest.main()
