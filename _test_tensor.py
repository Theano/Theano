import traceback
import operator

from tensor import *
import tensor # for hidden symbols

import unittest
from copy import copy
import compile
import gradient
import gof, gof.graph
from gof.python25 import any
import gof
from gof.utils import AbstractFunctionError

from elemwise import DimShuffle


default_mode = compile.Mode(optimizer = None,
                            linker = 'c&py')

def function(inputs, outputs, mode = default_mode):
    return compile.function(inputs, outputs, mode = mode, accept_inplace = True)


def eval_outputs(outputs, mode = default_mode):
    results = function([], outputs, mode = mode)()
    if len(results) == 1:
        return results[0]
    return results


def _numpy_checker(x, y):
    """
    Checks if x.data and y.data have the same contents.
    Used in DualLinker to compare C version with Python version.
    """
    x, y = x[0], y[0]
    if x.dtype != y.dtype or x.shape != y.shape or numpy.any(numpy.abs(x - y) > 1e-10):
        raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})

def safe_make_node(op, *inputs):
    """Emulate the behaviour of make_node when op is a function instead of an Op instance."""
    node = op(*inputs)
    if isinstance(node, list):
        return node[0].owner
    else:
        return node.owner

def make_tester(name, op, expected, checks = {}, good = {}, bad_build = {}, bad_runtime = {}, grad = {}):
    if grad is True:
        grad = good

    _op, _expected, _checks, _good, _bad_build, _bad_runtime, _grad = op, expected, checks, good, bad_build, bad_runtime, grad

    class Checker(unittest.TestCase):

        op = _op
        expected = staticmethod(_expected)
        checks = _checks
        good = _good
        bad_build = _bad_build
        bad_runtime = _bad_runtime
        grad = _grad

        def test_good(self):
            for testname, inputs in self.good.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [value(input) for input in inputs]
                try:
                    #node = self.op.make_node(*inputrs)
                    node = safe_make_node(self.op, *inputrs)
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while making a node with inputs %s" \
                        % (self.op, testname, inputs)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

                try:
                    f = function(inputrs, node.outputs,
                                 mode = default_mode, ##lambda env, **kwargs: gof.DualLinker(env, checker = _numpy_checker, **kwargs),
                                 )
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while trying to make a Function" \
                        % (self.op, testname)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

                expecteds = self.expected(*inputs)

                try:
                    results = f(*inputs)
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while calling the Function on the inputs %s" \
                        % (self.op, testname, inputs)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

                if not isinstance(expecteds, (list, tuple)):
                    expecteds = (expecteds, )
                for i, (result, expected) in enumerate(zip(results, expecteds)):
                    if result.dtype != expected.dtype or result.shape != expected.shape or \
                            numpy.any(numpy.abs(result - expected) > 1e-10):
                        self.fail("Test %s::%s: Output %s gave the wrong value. With inputs %s, expected %s, got %s."
                                  % (self.op, testname, i, inputs, expected, result))

                for description, check in self.checks.items():
                    if not check(inputs, results):
                        self.fail("Test %s::%s: Failed check: %s (inputs were %s)"
                                  % (self.op, testname, description, inputs))

        def test_bad_build(self):
            for testname, inputs in self.bad_build.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [value(input) for input in inputs]
                try:
                    node = safe_make_node(self.op,*inputrs)
                except:
                    return
                self.fail("Test %s::%s: %s was successfully instantiated on the following bad inputs: %s"
                          % (self.op, testname, node, inputs))

        def test_bad_runtime(self):
            for testname, inputs in self.bad_runtime.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [value(input) for input in inputs]
                try:
                    node = safe_make_node(self.op,*inputrs)
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while trying to make a node with inputs %s" \
                        % (self.op, testname, inputs)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

                try:
                    f = function(inputrs, node.outputs,
                                 mode = default_mode, #lambda env, **kwargs: gof.DualLinker(env, checker = _numpy_checker, **kwargs),
                                 )
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while trying to make a Function" \
                        % (self.op, testname)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

                try:
                    results = f(*inputs)
                except:
                    return

                self.fail("Test %s::%s: Successful call on the following bad inputs: %s"
                          % (self.op, testname, inputs))

        def test_grad(self):
            for testname, inputs in self.grad.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [value(input) for input in inputs]
                try:
                    verify_grad(self, self.op, inputs)
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while computing the gradient on the following inputs: %s" \
                        % (self.op, testname, inputs)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

    Checker.__name__ = name
    return Checker


rand = lambda *shape: 2 * numpy.random.rand(*shape) - 1
randint = lambda *shape: numpy.random.random_integers(-5, 5, shape)

def randint_nonzero(*shape):
    r = numpy.random.random_integers(-5, 4, shape)
    return r + (r == 0) * 5

def rand_ranged(min, max, shape):
    return numpy.random.rand(*shape) * (max - min) + min

def randint_ranged(min, max, shape):
    return numpy.random.random_integers(min, max, shape)


def make_broadcast_tester(op, expected, checks = {}, **kwargs):
    name = str(op) + "Tester"
    if kwargs.has_key('inplace'):
        if kwargs['inplace']:
            _expected = expected
            expected = lambda *inputs: numpy.array(_expected(*inputs), dtype = inputs[0].dtype)
            checks = dict(checks,
                          inplace_check = lambda inputs, outputs: inputs[0] is outputs[0])
        del kwargs['inplace']
    return make_tester(name, op, expected, checks, **kwargs)



_good_broadcast_binary_normal = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                     not_same_dimensions = (rand(2, 2), rand(2)),
                                     scalar = (rand(2, 3), rand(1, 1)),
                                     row = (rand(2, 3), rand(1, 3)),
                                     column = (rand(2, 3), rand(2, 1)),
                                     integers = (randint(2, 3), randint(2, 3)),
                                     dtype_mixup_1 = (rand(2, 3), randint(2, 3)),
                                     dtype_mixup_2 = (randint(2, 3), rand(2, 3)))

_bad_build_broadcast_binary_normal = dict()#not_same_dimensions = (rand(2), rand(2, 2)))

_bad_runtime_broadcast_binary_normal = dict(bad_shapes = (rand(2, 3), rand(3, 2)),
                                            bad_row = (rand(2, 3), rand(1, 2)))

_grad_broadcast_binary_normal = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                     scalar = (rand(2, 3), rand(1, 1)),
                                     row = (rand(2, 3), rand(1, 3)),
                                     column = (rand(2, 3), rand(2, 1)))


AddTester = make_broadcast_tester(op = add,
                                  expected = lambda *inputs: reduce(lambda x, y: x + y, inputs),
                                  good = dict(three_inputs_same_shapes = (rand(2, 3), rand(2, 3), rand(2, 3)),
                                              four_inputs_broadcast = (rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
                                              **_good_broadcast_binary_normal),
                                  bad_build = _bad_build_broadcast_binary_normal,
                                  bad_runtime = _bad_runtime_broadcast_binary_normal)
AddInplaceTester = make_broadcast_tester(op = tensor._add_inplace,
                                         expected = lambda x, y: x + y,
                                         good = _good_broadcast_binary_normal,
                                         bad_build = _bad_build_broadcast_binary_normal,
                                         bad_runtime = _bad_runtime_broadcast_binary_normal,
                                         inplace = True)

SubTester = make_broadcast_tester(op = sub,
                                  expected = lambda x, y: x - y,
                                  good = _good_broadcast_binary_normal,
                                  bad_build = _bad_build_broadcast_binary_normal,
                                  bad_runtime = _bad_runtime_broadcast_binary_normal,
                                  grad = _grad_broadcast_binary_normal)

SubInplaceTester = make_broadcast_tester(op = tensor._sub_inplace,
                                         expected = lambda x, y: x - y,
                                         good = _good_broadcast_binary_normal,
                                         bad_build = _bad_build_broadcast_binary_normal,
                                         bad_runtime = _bad_runtime_broadcast_binary_normal,
                                         grad = _grad_broadcast_binary_normal,
                                         inplace = True)

MulTester = make_broadcast_tester(op = mul,
                                  expected = lambda *inputs: reduce(lambda x, y: x * y, inputs),
                                  good = dict(three_inputs_same_shapes = (rand(2, 3), rand(2, 3), rand(2, 3)),
                                              four_inputs_broadcast = (rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
                                              **_good_broadcast_binary_normal),
                                  bad_build = _bad_build_broadcast_binary_normal,
                                  bad_runtime = _bad_runtime_broadcast_binary_normal,
                                  grad = dict(three_inputs_same_shapes = (rand(2, 3), rand(2, 3), rand(2, 3)),
                                              four_inputs_broadcast = (rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
                                              **_grad_broadcast_binary_normal))
MulInplaceTester = make_broadcast_tester(op = tensor._mul_inplace,
                                         expected = lambda x, y: x * y,
                                         good = _good_broadcast_binary_normal,
                                         bad_build = _bad_build_broadcast_binary_normal,
                                         bad_runtime = _bad_runtime_broadcast_binary_normal,
                                         grad = _grad_broadcast_binary_normal,
                                         inplace = True)

DivTester = make_broadcast_tester(op = div,
                                  expected = lambda x, y: x / y,
                                  good = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                              scalar = (rand(2, 3), rand(1, 1)),
                                              row = (rand(2, 3), rand(1, 3)),
                                              column = (rand(2, 3), rand(2, 1)),
                                              dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
                                              dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3)),
#                                               integers_positive = (randint_ranged(4, 10, (2, 3)), randint_ranged(1, 6, (2, 3))),
#                                               integers_known_to_fail = (numpy.array(-1), numpy.array(5))
                                              ),
#                                               integers = (randint(2, 3), randint_nonzero(2, 3)),
#                                               dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
#                                               dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3))),
                                  grad = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                              scalar = (rand(2, 3), rand(1, 1)),
                                              row = (rand(2, 3), rand(1, 3)),
                                              column = (rand(2, 3), rand(2, 1))))
DivInplaceTester = make_broadcast_tester(op = tensor._div_inplace,
                                         expected = lambda x, y: x / y,
                                         good = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                                     scalar = (rand(2, 3), rand(1, 1)),
                                                     row = (rand(2, 3), rand(1, 3)),
                                                     column = (rand(2, 3), rand(2, 1)),
                                                     dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
                                                     dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3))
                                                     ),
                                         grad = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                                     scalar = (rand(2, 3), rand(1, 1)),
                                                     row = (rand(2, 3), rand(1, 3)),
                                                     column = (rand(2, 3), rand(2, 1))),
                                         inplace = True)

ModTester = make_broadcast_tester(op = mod,
                                  expected = lambda x, y: x % y,
                                  good = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                              scalar = (rand(2, 3), rand(1, 1)),
                                              row = (rand(2, 3), rand(1, 3)),
                                              column = (rand(2, 3), rand(2, 1)),
                                              dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
                                              dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3)),
#                                               integers_positive = (randint_ranged(4, 10, (2, 3)), randint_ranged(1, 6, (2, 3))),
#                                               integers_known_to_fail = (numpy.array(-1), numpy.array(5))
                                              ),
#                                               integers = (randint(2, 3), randint_nonzero(2, 3)),
#                                               dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
#                                               dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3))),
                                  )
ModInplaceTester = make_broadcast_tester(op = tensor._mod_inplace,
                                         expected = lambda x, y: x % y,
                                         good = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                                     scalar = (rand(2, 3), rand(1, 1)),
                                                     row = (rand(2, 3), rand(1, 3)),
                                                     column = (rand(2, 3), rand(2, 1)),
                                                     dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
                                                     dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3))
                                                     ),
                                         inplace = True)

PowTester = make_broadcast_tester(op = pow,
                                  expected = lambda x, y: x ** y,
                                  good = dict(same_shapes = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 3))),
                                              scalar = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 1))),
                                              row = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 3))),
                                              column = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 1))),
                                              dtype_mixup = (rand_ranged(-3, 3, (2, 3)), randint_ranged(-3, 3, (2, 3)))),
                                  grad = dict(same_shapes = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 3))),
                                              scalar = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 1))),
                                              row = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 3))),
                                              column = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 1))))
                                  )
PowInplaceTester = make_broadcast_tester(op = tensor._pow_inplace,
                                         expected = lambda x, y: x ** y,
                                         good = dict(same_shapes = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 3))),
                                                     scalar = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 1))),
                                                     row = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 3))),
                                                     column = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 1))),
                                                     dtype_mixup = (rand_ranged(-3, 3, (2, 3)), randint_ranged(-3, 3, (2, 3)))),
                                         grad = dict(same_shapes = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 3))),
                                                     scalar = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 1))),
                                                     row = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 3))),
                                                     column = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 1)))),
                                         inplace = True)



_good_broadcast_unary_normal = dict(normal = (rand_ranged(-5, 5, (2, 3)),),
                                    integers = (randint_ranged(-5, 5, (2, 3)),))

_grad_broadcast_unary_normal = dict(normal = (rand_ranged(-5, 5, (2, 3)),))


AbsTester = make_broadcast_tester(op = tensor._abs,
                                  expected = lambda x: abs(x),
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
AbsInplaceTester = make_broadcast_tester(op = tensor.__abs_inplace,
                                         expected = lambda x: numpy.abs(x),
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

NegTester = make_broadcast_tester(op = neg,
                                  expected = lambda x: -x,
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
NegInplaceTester = make_broadcast_tester(op = tensor._neg_inplace,
                                         expected = lambda x: -x,
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

SgnTester = make_broadcast_tester(op = sgn,
                                  expected = numpy.sign,
                                  good = _good_broadcast_unary_normal)
SgnInplaceTester = make_broadcast_tester(op = tensor._sgn_inplace,
                                         expected = numpy.sign,
                                         good = _good_broadcast_unary_normal,
                                         inplace = True)

SqrTester = make_broadcast_tester(op = sqr,
                                  expected = numpy.square,
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
SqrInplaceTester = make_broadcast_tester(op = tensor._sqr_inplace,
                                         expected = numpy.square,
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

ExpTester = make_broadcast_tester(op = exp,
                                  expected = numpy.exp,
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
ExpInplaceTester = make_broadcast_tester(op = tensor._exp_inplace,
                                         expected = numpy.exp,
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)


_good_broadcast_unary_positive = dict(normal = (rand_ranged(0.001, 5, (2, 3)),),
                                      integers = (randint_ranged(1, 5, (2, 3)),))

_grad_broadcast_unary_positive = dict(normal = (rand_ranged(0.001, 5, (2, 3)),))

LogTester = make_broadcast_tester(op = log,
                                  expected = numpy.log,
                                  good = _good_broadcast_unary_positive,
                                  grad = _grad_broadcast_unary_positive)
LogInplaceTester = make_broadcast_tester(op = tensor._log_inplace,
                                         expected = numpy.log,
                                         good = _good_broadcast_unary_positive,
                                         grad = _grad_broadcast_unary_positive,
                                         inplace = True)

Log2Tester = make_broadcast_tester(op = log2,
                                   expected = numpy.log2,
                                   good = _good_broadcast_unary_positive,
                                   grad = _grad_broadcast_unary_positive)
Log2InplaceTester = make_broadcast_tester(op = tensor._log2_inplace,
                                          expected = numpy.log2,
                                          good = _good_broadcast_unary_positive,
                                          grad = _grad_broadcast_unary_positive,
                                          inplace = True)

SqrtTester = make_broadcast_tester(op = sqrt,
                                   expected = numpy.sqrt,
                                   good = _good_broadcast_unary_positive,
                                   grad = _grad_broadcast_unary_positive)
SqrtInplaceTester = make_broadcast_tester(op = tensor._sqrt_inplace,
                                          expected = numpy.sqrt,
                                          good = _good_broadcast_unary_positive,
                                          grad = _grad_broadcast_unary_positive,
                                          inplace = True)



_good_broadcast_unary_wide = dict(normal = (rand_ranged(-1000, 1000, (2, 3)),),
                                  integers = (randint_ranged(-1000, 1000, (2, 3)),))

_grad_broadcast_unary_wide = dict(normal = (rand_ranged(-1000, 1000, (2, 3)),))


SinTester = make_broadcast_tester(op = sin,
                                  expected = numpy.sin,
                                  good = _good_broadcast_unary_wide,
                                  grad = _grad_broadcast_unary_wide)
SinInplaceTester = make_broadcast_tester(op = tensor._sin_inplace,
                                         expected = numpy.sin,
                                         good = _good_broadcast_unary_wide,
                                         grad = _grad_broadcast_unary_wide,
                                         inplace = True)

CosTester = make_broadcast_tester(op = cos,
                                  expected = numpy.cos,
                                  good = _good_broadcast_unary_wide,
                                  grad = _grad_broadcast_unary_wide)
CosInplaceTester = make_broadcast_tester(op = tensor._cos_inplace,
                                         expected = numpy.cos,
                                         good = _good_broadcast_unary_wide,
                                         grad = _grad_broadcast_unary_wide,
                                         inplace = True)

TanTester = make_broadcast_tester(op = tan,
                                  expected = numpy.tan,
                                  good = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                              shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                  grad = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                              shifted = (rand_ranged(3.15, 6.28, (2, 3)),)))
TanInplaceTester = make_broadcast_tester(op = tensor._tan_inplace,
                                         expected = numpy.tan,
                                         good = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                                     shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                         grad = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                                     shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                         inplace = True)


CoshTester = make_broadcast_tester(op = cosh,
                                   expected = numpy.cosh,
                                   good = _good_broadcast_unary_normal,
                                   grad = _grad_broadcast_unary_normal)
CoshInplaceTester = make_broadcast_tester(op = tensor._cosh_inplace,
                                          expected = numpy.cosh,
                                          good = _good_broadcast_unary_normal,
                                          grad = _grad_broadcast_unary_normal,
                                          inplace = True)

SinhTester = make_broadcast_tester(op = sinh,
                                   expected = numpy.sinh,
                                   good = _good_broadcast_unary_normal,
                                   grad = _grad_broadcast_unary_normal)
SinhInplaceTester = make_broadcast_tester(op = tensor._sinh_inplace,
                                          expected = numpy.sinh,
                                          good = _good_broadcast_unary_normal,
                                          grad = _grad_broadcast_unary_normal,
                                          inplace = True)

TanhTester = make_broadcast_tester(op = tanh,
                                   expected = numpy.tanh,
                                   good = _good_broadcast_unary_normal,
                                   grad = _grad_broadcast_unary_normal)
TanhInplaceTester = make_broadcast_tester(op = tensor._tanh_inplace,
                                          expected = numpy.tanh,
                                          good = _good_broadcast_unary_normal,
                                          grad = _grad_broadcast_unary_normal,
                                          inplace = True)



DotTester = make_tester(name = 'DotTester',
                        op = dot,
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

# def check_eq(self, node_in, node_out, arg_in, arg_out):
#     fn = Function([node_in], node_out)
#     self.failUnless( numpy.all(fn(arg_in) == arg_out), (arg_in, arg_out))

# def check_eq2(self, inputs, output, args_in, arg_out):
#     fn = Function(inputs, output)
#     val = fn(*args_in)
#     self.failUnless( numpy.all(val == arg_out), (val, arg_out))

# def check_eq2_c(self, inputs, output, args_in, arg_out):
#     fn = Function(inputs, [output], linker_cls = gof.CLinker)
#     val = fn(*args_in)
#     self.failUnless( numpy.all(val == arg_out), (val, arg_out))

# def check_eq2_both(self, inputs, output, args_in, arg_out):
#     fn = Function(inputs, [output], linker_cls = lambda env: gof.DualLinker(env, _numpy_checker))
#     val = fn(*args_in)
#     self.failUnless( numpy.all(val == arg_out), (val, arg_out))

class T_Shape(unittest.TestCase):
    def test_basic0(self):
        s = shape(numpy.ones((5, 3)))
        self.failUnless((eval_outputs([s]) == [5, 3]).all())
    def test_basic1(self):
        s = shape(numpy.ones((2)))
        self.failUnless((eval_outputs([s]) == [2]).all())
    def test_basic2(self):
        s = shape(numpy.ones((5, 3, 10)))
        self.failUnless((eval_outputs([s]) == [5, 3, 10]).all())

class T_Cast(unittest.TestCase):
    def test_basic(self):
        for type1 in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            x = Tensor(dtype = type1, broadcastable = (False, )).make_result()
            for type2, converter in zip(['int8', 'int16', 'int32', 'int64', 'float32', 'float64'],
                                        [convert_to_int8, convert_to_int16, convert_to_int32, convert_to_int64,
                                         convert_to_float32, convert_to_float64]):
                y = converter(x)
                f = function([compile.In(x, strict = True)], y, mode = default_mode)
                a = numpy.arange(10, dtype = type1)
                b = f(a)
                self.failUnless(numpy.all(b == numpy.arange(10, dtype = type2)))

class T_max_and_argmax(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(123784)
        MaxAndArgmax.debug = 0

    def test0(self):
        n = as_tensor(5.0)
        v,i = eval_outputs(max_and_argmax(n))
        self.failUnless(v == 5.0)
        self.failUnless(i == 0)

    def test1(self):
        n = as_tensor([1,2,3,2,-6])
        v,i = eval_outputs(max_and_argmax(n))
        self.failUnless(v == 3)
        self.failUnless(i == 2)

    def test2(self):
        n = as_tensor(numpy.random.rand(2,3))
        v,i = eval_outputs(max_and_argmax(n))
        self.failUnless(numpy.all(i == [0,1]))
    def test2b(self):
        n = as_tensor(numpy.random.rand(2,3))
        v,i = eval_outputs(max_and_argmax(n,0))
        self.failUnless(numpy.all(i == [0,1,1]))
    def test2_invalid(self):
        n = as_tensor(numpy.random.rand(2,3))
        try:
            eval_outputs(max_and_argmax(n,3))
        except ValueError, e:
            return
        self.fail()
    def test2_invalid_neg(self):
        n = as_tensor(numpy.random.rand(2,3))
        try:
            eval_outputs(max_and_argmax(n,-3))
        except ValueError, e:
            return
        self.fail()
    def test2_valid_neg(self):
        n = as_tensor(numpy.random.rand(2,3))
        v,i = eval_outputs(max_and_argmax(n,-1))
        self.failUnless(v.shape == (2,))
        v,i = eval_outputs(max_and_argmax(n,-2))
        self.failUnless(v.shape == (3,))
    def test3(self):
        n = as_tensor(numpy.random.rand(2,3,4))
        v,i = eval_outputs(max_and_argmax(n,0))
        self.failUnless(v.shape == (3,4))
        self.failUnless(i.shape == (3,4))
        v,i = eval_outputs(max_and_argmax(n,1))
        self.failUnless(v.shape == (2,4))
        self.failUnless(i.shape == (2,4))
        v,i = eval_outputs(max_and_argmax(n,2))
        self.failUnless(v.shape == (2,3))
        self.failUnless(i.shape == (2,3))


class T_transpose(unittest.TestCase):
    def test0(self):
        n = as_tensor(numpy.ones(()))
        t = transpose(n)
        self.failUnless(t.owner.op == tensor._transpose_inplace)
        f = function([n], t)
        tval = f(n.data)
        self.failUnless(tval.shape == n.data.shape)

        #test aliasing
        tval += 55.0
        self.failUnless(n.data == 1.0)

    def test1(self):
        n = as_tensor(numpy.ones(5))
        t = transpose(n)
        self.failUnless(t.owner.op == tensor._transpose_inplace)
        f = function([n], t)
        tval = f(n.data)
        self.failUnless(tval.shape == n.data.shape)
        #test aliasing
        tval += 55.0
        self.failUnless(n.data[0] == 1.0)

    def test2(self):
        n = as_tensor(numpy.ones((5,3)))
        t = transpose(n)
        self.failUnless(t.owner.op == tensor._transpose_inplace)
        f = function([n], t)
        tval = f(n.data)
        self.failUnless(tval.shape == (3,5))
        #test aliasing
        tval += 55.0
        self.failUnless(n.data[0,0] == 1.0)

    def test3(self):
        """Test transpose of tensor, inplace version"""
        n = as_tensor(numpy.ones((5,3,2)))
        t = tensor._transpose_inplace(n)
        self.failUnless(t.owner.op == tensor._transpose_inplace)
        f = function([n], t)
        tval = f(n.data)
        self.failUnless(tval.shape == (2,3,5))
        #test aliasing
        tval += 55.0
        self.failUnless(n.data[0,0,0] == 56.0)
    def test_grad(self):
        verify_grad(self, tensor._transpose_inplace, [numpy.random.rand(2, 3)])
        verify_grad(self, tensor._transpose_inplace, [numpy.ones(3)])

class T_subtensor(unittest.TestCase):
    def setUp(self):
        Subtensor.debug = False
        numpy.random.seed(12353123)

    def test0_err_invalid(self):
        #it is impossible to retrieve a view of a 0-d tensor
        n = as_tensor(numpy.ones(()))
        try:
            t = n[0]
        except ValueError, e:
            self.failUnless(e[0] is Subtensor.e_invalid)
            return
        self.fail()

    def test1_err_bounds(self):
        n = as_tensor(numpy.ones(3))
        t = n[7]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        try:
            tval = eval_outputs([t])
        except Exception, e:
            if e[0] != 'index out of bounds':
                raise
            return
        self.fail()
    def test1_err_subslice(self):
        n = as_tensor(numpy.ones(3))
        try:
            t = n[slice(0,slice(1,2,None),None)]
        except Exception, e:
            if e[0] != Subtensor.e_indextype:
                raise
            return
        self.fail()

    def test1_ok_range_finite(self):
        n = as_tensor(numpy.ones(3)*5)
        t = n[0:2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test2_ok_range_finite(self):
        n = as_tensor(numpy.ones((3,4))*5)
        t = n[0:2,3]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test1_err_invalid(self):
        n = as_tensor(numpy.ones(1))
        try:
            t = n[0,0]
        except ValueError, e:
            self.failUnless(e[0] is Subtensor.e_invalid)
            return
        self.fail()
    def test1_ok_elem(self):
        n = as_tensor(numpy.ones(1)*5)
        t = n[0]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == ())
        self.failUnless(tval == 5.0)
    def test1_ok_range_infinite(self):
        #Subtensor.debug = True
        n = as_tensor(numpy.ones(3)*5)
        t = n[1:]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test1_ok_strided(self):
        n = as_tensor(numpy.ones(5)*5)
        t = n[1::2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)

        tval = eval_outputs([n[0:-1:2]]) #0 to 1 from the end stepping by 2
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)

    def test2_err_bounds0(self):
        n = as_tensor(numpy.ones((2,3))*5)
        t = n[0,4]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        try:
            tval = eval_outputs([t])
        except IndexError, e:
            return
        self.fail()
    def test2_err_bounds1(self):
        n = as_tensor(numpy.ones((2,3))*5)
        t = n[4:5,2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        try:
            tval = eval_outputs([t])
        except Exception, e:
            if e[0] != 'index out of bounds':
                raise
    def test2_ok_elem(self):
        n = as_tensor(numpy.asarray(range(6)).reshape((2,3)))
        t = n[0,2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == ())
        self.failUnless(numpy.all(tval == 2))
    def test2_ok_row(self):
        n = as_tensor(numpy.asarray(range(6)).reshape((2,3)))
        t = n[1]
        self.failIf(any(n.type.broadcastable))
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (3,))
        self.failUnless(numpy.all(tval == [3,4,5]))

    def test2_ok_col(self):
        n = as_tensor(numpy.ones((2,3))*5)
        t = n[:,0]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        self.failIf(any(n.type.broadcastable))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(numpy.all(tval == 5.0))

    def test2_ok_rows_finite(self):
        n = as_tensor(numpy.ones((4,3))*5)
        t = n[1:3,0]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(numpy.all(tval == 5.0))

    def test2_ok_cols_infinite(self):
        n = as_tensor(numpy.asarray(range(12)).reshape((4,3)))
        t = n[1,2:]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (1,))
        self.failUnless(numpy.all(tval == 5))

    def test2_ok_strided(self):
        n = as_tensor(numpy.asarray(range(20)).reshape((4,5)))
        t = n[1:4:2,1:5:2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,2))
        self.failUnless(numpy.all(tval == [[6, 8],[16, 18]]))

    def test3_ok_mat(self):
        n = as_tensor(numpy.asarray(range(24)).reshape((2,3,4)))
        t = n[0,0,0]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == ())
        self.failUnless(numpy.all(tval == 0))


    def test_grad_1d(self):
        n = as_tensor(numpy.random.rand(2,3))
        z = scal.constant(0)
        t = n[z:,z]
        gn = grad(sum(exp(t)), n)
        gval = eval_outputs([gn])
        s0 = 'array([ 2.05362099,  0.        ,  0.        ])'
        s1 = 'array([ 1.55009327,  0.        ,  0.        ])'
        self.failUnless(repr(gval[0,:]) == s0)
        self.failUnless(repr(gval[1,:]) == s1)

    def test_grad_0d(self):
        n = as_tensor(numpy.random.rand(2,3))
        t = n[1,0]
        gn = grad(sum(exp(t)), n)
        gval = eval_outputs([gn])
        g0 = repr(gval[0,:])
        g1 = repr(gval[1,:])
        s0 = 'array([ 0.,  0.,  0.])'
        s1 = 'array([ 1.55009327,  0.        ,  0.        ])'
        self.failUnless(g0 == s0, (g0, s0))
        self.failUnless(g1 == s1, (g1, s1))



class T_Join_and_Split(unittest.TestCase):
    """
    Split is tested by each verify_grad method.
    """

    class Join1(Op):
        def make_node(self, *inputs):
            inputs = [as_tensor(t) for t in inputs]
            outputs = [lscalar()] + [i.type() for i in inputs]
            return Apply(self, inputs, outputs)
        def perform(self, node, inputs, outputs):
            outputs[0][0] = 1
            for i,o in zip(inputs, outputs[1:]):
                o[0] = i.copy()
        def grad(self, inputs, g_outputs):
            return g_outputs[1:]

    def setUp(self):
        Join.debug = False

    def test_join_scalar(self):
        a = as_tensor(1)
        b = as_tensor(2)
        try:
            s = join(0, a, b)
        except:
            return
        self.fail()

    def test_stack_scalar(self):
        a = as_tensor(1)
        b = as_tensor(2)
        c = as_tensor(3)
        s = stack(a, b, c)

        want = numpy.array([1, 2, 3])
        self.failUnless((eval_outputs([s]) == want).all())


    def test_join_vector(self):
        a = as_tensor(numpy.array([1, 2, 3]))
        b = as_tensor(numpy.array([7, 8, 9]))

        s = join(0, a, b)
        want = numpy.array([1, 2, 3, 7, 8, 9])
        self.failUnless((eval_outputs([s]) == want).all())

    def test_stack_vector(self):
        a = as_tensor(numpy.array([1, 2, 3]))
        b = as_tensor(numpy.array([7, 8, 9]))

        s = stack(a, b)
        want = numpy.array([[1, 2, 3],[ 7, 8, 9]])
        self.failUnless((eval_outputs([s]) == want).all())

    def test_join_matrix0(self):
        a = as_tensor(numpy.array([[1, 2, 3], [4, 5, 6]]))
        b = as_tensor(numpy.array([[7, 8, 9]]))
        s = join(0, a, b)

        want = numpy.array([[1, 2, 3],[4,5,6],[7, 8, 9]])
        self.failUnless((eval_outputs([s]) == want).all())

    def test_join_matrix1(self):
        av=numpy.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
        bv= numpy.array([[7], [8]],dtype='float32')
        a = as_tensor(av)
        b = as_tensor(bv)
        s = join(1, a, b)
        want = numpy.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype='float32')
        self.failUnless((eval_outputs([s]) == want).all())

        verify_grad(self, lambda a, b: join(1,a,b), [av, bv], eps=1.0e-4, tol=1.0e-3)

    def test_join_matrixV(self):
        """variable join axis"""
        v = numpy.array([[1., 2., 3.], [4., 5., 6.]])
        a = as_tensor(v.copy())
        b = as_tensor(v.copy())
        ax = lscalar()
        s = join(ax, a, b)

        f = function([ax], [s])

        want = numpy.array([[1, 2, 3], [4, 5, 6] ,[1, 2, 3], [4, 5, 6]])
        got = f(0)
        self.failUnless((got == want).all(), (got, want))

        want = numpy.array([[ 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]])
        got = f(1)
        self.failUnless((got == want).all(), (got, want))

        verify_grad(self, lambda a, b: join(0,a,b), [v, 2*v])
        verify_grad(self, lambda a, b: join(1,a,b), [v, 2*v])



class _test_comparison(unittest.TestCase):
    def test_gt(self):
        x, y = fvector(), fvector()
        fn = function([x,y], x > y)
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l > r)), (v, (l>r)))

    def test_lt(self):
        x, y = fvector(), fvector()
        fn = function([x,y], x < y)
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l < r)), (v, (l<r)))

    def test_le(self):
        x, y = fvector(), fvector()
        fn = function([x,y], x <= y)
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l <= r)), (v, (l<=r)))

    def test_ge(self):
        x, y = fvector(), fvector()
        fn = function([x,y], x >= y)
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l >= r)), (v, (l>=r)))

    def test_eq(self):
        x, y = fvector(), fvector()
        fn = function([x,y], eq(x,y))
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l == r)), (v, (l==r)))

    def test_neq(self):
        x, y = fvector(), fvector()
        fn = function([x,y], neq(x, y))
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l != r)), (v, (l!=r)))

class _test_bitwise(unittest.TestCase):
    def test_or(self):
        x, y = bvector(), bvector()
        fn = function([x,y], x|y)
        l = numpy.asarray([0,0,1,1], dtype = 'int8')
        r = numpy.asarray([0,1,0,1], dtype = 'int8')
        v = fn(l, r)
        self.failUnless(numpy.all(v == (operator.or_(l, r))), (l, r, v))

    def test_xor(self):
        x, y = bvector(), bvector()
        fn = function([x,y], x^y)
        ix = x
        ix ^= y
        gn = function([x,y], ix)
        l = numpy.asarray([0,0,1,1], dtype = 'int8')
        r = numpy.asarray([0,1,0,1], dtype = 'int8')
        v = fn(l, r)
        self.failUnless(numpy.all(v == (operator.xor(l, r))), (l, r, v))
        v = gn(l, r)
        #test the in-place stuff
        self.failUnless(numpy.all(l == numpy.asarray([0,1,1,0])), l)

    def test_and(self):
        x, y = bvector(), bvector()
        fn = function([x,y], x&y)
        l = numpy.asarray([0,0,1,1], dtype = 'int8')
        r = numpy.asarray([0,1,0,1], dtype = 'int8')
        v = fn(l, r)
        self.failUnless(numpy.all(v == (operator.and_(l, r))), (l, r, v))

    def test_inv(self):
        x, y = bvector(), bvector()
        fn = function([x,y], ~x)
        l = numpy.asarray([0,0,1,1], dtype = 'int8')
        r = numpy.asarray([0,1,0,1], dtype = 'int8')
        v = fn(l, r)
        self.failUnless(numpy.all(v == (~l)), (l, r, v))



class T_add(unittest.TestCase):

    def test_complex_all_ops(self):
        for nbits in (64, 128):
            a = value(numpy.ones(3, dtype='complex%i' % nbits)+0.5j)
            b = value(numpy.ones(3, dtype='complex%i' % nbits)+1.5j)
            tests = (("+", lambda x,y: x+y),
                     ("-", lambda x,y: x-y),
                     ("*", lambda x,y: x*y),
                     ("/", lambda x,y: x/y))
            for s, fn in tests:
                f = function([a,b], fn(a, b), mode = compile.Mode(optimizer = None, linker = 'c'))
                self.failUnless(numpy.all(fn(a.data, b.data) == f(a.data, b.data)))

    def test_grad_scalar_l(self):
        verify_grad(self, add, [numpy.asarray([3.0]), numpy.random.rand(3)])
    def test_grad_scalar_r(self):
        verify_grad(self, add, [numpy.random.rand(3), numpy.asarray([3.0])])
    def test_grad_row(self):
        verify_grad(self, add, [numpy.random.rand(3, 5), numpy.random.rand(1, 5)])
    def test_grad_col(self):
        verify_grad(self, add, [numpy.random.rand(3, 5), numpy.random.rand(3, 1)])

class T_exp(unittest.TestCase):

    def test_grad_0(self):
        verify_grad(self, exp, [
            numpy.asarray([[ 1.5089518 ,  1.48439076, -4.7820262 ],
            [ 2.04832468,  0.50791564, -1.58892269]])])
    def test_grad_1(self):
        verify_grad(self, tensor._exp_inplace, [
            numpy.asarray([[ 1.5089518 ,  1.48439076, -4.7820262 ],
            [ 2.04832468,  0.50791564, -1.58892269]])])

# class T_abs(unittest.TestCase):
#     def test_impl(self):
#         t = as_tensor(1.0)
#         check_eq(self, t, abs(t), 1.0, 1.0)
#         check_eq(self, t, abs(t), -1.0, 1.0)

#         for shape in (2,), (3,4):
#             t = as_tensor(numpy.ones(shape))
#             d = numpy.random.rand(*shape)*2-1.0
#             check_eq(self, t, abs(t), d, abs(d))
#             check_eq(self, t, abs(t), -d, abs(-d))

#     def test_grad(self):
#         verify_grad(self, Abs, [numpy.ones(())])
#         verify_grad(self, Abs, [numpy.ones(3)])

#     class AbsBadGrad(Abs):
#         def grad(self, (x, ), (gz, )):
#             return mul(gz * sgn(x),0.9),

#     def test_badgrad(self):
#         try:
#             verify_grad(self, T_abs.AbsBadGrad, [numpy.ones(())])
#         except Exception, e:
#             self.failUnless(str(e) == verify_grad.E_grad, str(e))
#             return
#         self.fail()

# class T_fill(unittest.TestCase):
#     def test0(self):
#         t = fill(numpy.asarray([1,2,3]), 9)
#         self.failUnless(t.owner.__class__ == Fill)
#         o = t.owner
#         self.failUnless(o.inputs[0].broadcastable == (0,))
# #        self.failUnless(o.inputs[0].dtype[0:3] == 'int')
#         self.failUnless(o.inputs[1].broadcastable == (1,))
# #        self.failUnless(o.inputs[1].dtype[0:3] == 'flo')
#         self.failUnless(o.outputs[0].broadcastable == (0,))
# #        self.failUnless(o.outputs[0].dtype[0:3] == 'flo')
#         self.failUnless(numpy.all(eval_outputs([t]) == [9,9,9]))

#     def test1(self):
#         x = as_tensor(numpy.ones((4,5)))
#         l = ones_like(x[:,0:1])
#         r = ones_like(x[0:1,:])
#         xx = x + dot(l,r)
#         self.failUnless(numpy.mean(eval_outputs([xx]) == 2.0))

# class T_sum(unittest.TestCase):
#     def test_impl(self):
#         t = as_tensor(0.0)
#         check_eq(self, t, Sum(t).out, 1.0, 1.0)
#         check_eq(self, t, Sum(t).out, -1.0, -1.0)

#         t = as_tensor([0.0, 0.0])
#         d = numpy.asarray([-0.4, 1.2])
#         check_eq(self, t, Sum(t).out, d, numpy.sum(d))
#         check_eq(self, t, Sum(t).out, -d, -numpy.sum(d))

# class T_mul(unittest.TestCase):
#     def setUp(self):
#         numpy.random.seed([1,2,3,4])

#     def test_elemwise(self):
#         a = as_tensor(0.0)
#         b = as_tensor(0.0)
#         check_eq2_both(self, [a,b], mul(a,b), [3.0, 4.0], 12.0)
#         check_eq2_both(self, [a,b], mul(b,a), [-1.0,2.0], -2.0)

#         a = as_tensor(numpy.ones(2))
#         b = as_tensor(numpy.ones(2))
#         aa = numpy.asarray([-0.5, 4.0])
#         bb = numpy.asarray([-0.5, 2.0])
#         check_eq2_both(self, [a,b], mul(a,b), [aa,bb], numpy.asarray([0.25, 8.0]))
#         check_eq2_both(self, [a,b], mul(a,b), [bb,aa], numpy.asarray([0.25, 8.0]))

#     def test_scalar(self):
#         r = numpy.random.rand(2,3)
#         a = as_tensor(r)
#         b = as_tensor(2.0)
#         check_eq2_both(self, [a,b], mul(a,b), [r, 2.0], r*2.0)
#         check_eq2_both(self, [a,b], mul(a,b), [r, 4.0], r*4.0)
#         self.failUnless(b.data == 2.0)

#     def test_rowcol(self):
#         r1 = numpy.random.rand(3,5)
#         r2 = numpy.random.rand(1,5)
#         r3 = numpy.random.rand(3,1)
#         a1, a2, a3 = as_tensor(r1), as_tensor(r2), as_tensor(r3)
#         check_eq2_both(self, [a1,a2], mul(a1,a2), [r1, r2], r1*r2)
#         check_eq2_both(self, [a1,a3], mul(a1,a3), [r1, r3], r1*r3)

#     def test_grad_elemwise(self):
#         verify_grad(self, Mul, [numpy.random.rand(3,4), numpy.random.rand(3,4)])
#     def test_grad_scalar_l(self):
#         verify_grad(self, Mul, [numpy.asarray([3.0]), numpy.random.rand(3)])
#     def test_grad_scalar_r(self):
#         verify_grad(self, Mul, [numpy.random.rand(3), numpy.asarray([3.0])])
#     def test_grad_row(self):
#         verify_grad(self, Mul, [numpy.random.rand(3, 5), numpy.random.rand(1, 5)])
#     def test_grad_row2(self):
#         op = lambda x, y: Mul(x, DimShuffle(y, ['x', 0]).out)
#         verify_grad(self, op, [numpy.random.rand(3, 5), numpy.random.rand(5)])
#     def test_grad_col(self):
#         verify_grad(self, Mul, [numpy.random.rand(3, 5), numpy.random.rand(3, 1)])

#     def test_wrong_shapes(self):
#         a = as_tensor(numpy.ones(3))
#         b = as_tensor(numpy.ones(4))
#         try:
#             check_eq2(self, [a,b], Mul(a,b).out,
#                       [numpy.ones(3), numpy.ones(4)], 1.0)
#             self.fail()
#         except ValueError, e:
#             self.failUnless('shape mismatch' in str(e))
#         try:
#             check_eq2_c(self, [a,b], Mul(a,b).out,
#                         [numpy.ones(3), numpy.ones(4)], 1.0)
#             self.fail()
#         except ValueError, e:
#             pass

# class T_div(unittest.TestCase):
#     def setUp(self):
#         numpy.random.seed(9999)
#     def test_grad_e(self):
#         verify_grad(self, Div, [numpy.random.rand(3), numpy.ones(3)])
#         verify_grad(self, Div, [numpy.random.rand(3,5), numpy.random.rand(3,5)+0.1])
#         verify_grad(self, Div, [numpy.ones(()), numpy.ones(())])

#     def test_grad_sl(self):
#         verify_grad(self, Div, [numpy.ones((3, 5)), numpy.ones((1, 1))])
#         verify_grad(self, Div, [numpy.random.rand(3), numpy.ones((1, ))])
#         verify_grad(self, Div, [numpy.random.rand(3,5), numpy.random.rand(1,1)])

# class T_log2(unittest.TestCase):
#     def test0(self):
#         verify_grad(self, Log2, [numpy.random.rand(3,1)+0.0001])

# class T_log(unittest.TestCase):
#     def test0(self):
#         verify_grad(self, Log, [numpy.random.rand(3,1)+0.0001])
#     def test1(self):
#         a = as_tensor(numpy.ones(2))
#         b = as_tensor(numpy.ones(2))
#         aa = numpy.asarray([0.5, 4.0])
#         bb = numpy.asarray([0.5, 2.0])
#         check_eq2(self, [a], log(a), [aa], numpy.log(numpy.asarray(aa)))

# class T_pow(unittest.TestCase):
#     def setUp(self):
#         numpy.random.seed(9999)
#     def test_elemwise(self):
#         verify_grad(self, Div, [numpy.random.rand(3,4), numpy.random.rand(3,4)+0.1])
#         verify_grad(self, Pow, [numpy.random.rand(3,4), numpy.random.rand(3,4)])
#     def test_scalar_l(self):
#         verify_grad(self, Pow, [numpy.asarray([3.0]), numpy.random.rand(3)])
#     def test_scalar_r(self):
#         verify_grad(self, Pow, [numpy.random.rand(3), numpy.asarray([3.0])])
#     def test_row(self):
#         verify_grad(self, Pow, [numpy.random.rand(3, 5), numpy.random.rand(1, 5)])
#     def test_col(self):
#         verify_grad(self, Pow, [numpy.random.rand(3, 5), numpy.random.rand(3, 1)])

class _testCase_matinv(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(1)

    def mat_reciprocal(self,dim):
        # symbolic program
        # broadcastable=[False,False] means that the shape of matrix is two dimensional,
        # and none of the dimensions are constrained to have length 1.
        # Note that Tensor's constructor does not actually allocate any memory.
        # TODO: Make Tensor syntax more explicit, and maybe give shape or number of dimensions.
        a, b = matrices('ab')
        ab = a*b
        # Here, as_tensor actually uses the data allocated by numpy.
        diff = ab - as_tensor(numpy.ones((dim,dim)))
        # Sum of squared errors
        ssdiff = sum((diff**2.0))

        g_b = grad(ssdiff, b)

        # compilation to function
        # [a,b] are the inputs, [ssdiff,g_b] are the outputs
        fn = function([a,b], [ssdiff,g_b])

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
        tz = eval_outputs([dot(as_tensor(x), as_tensor(y))])
        self.failUnless(tz.dtype == nz.dtype)
        self.failUnless(tz.shape == nz.shape)
        self.failUnless(_approx_eq(nz, tz))

    #def test_dot_0d_0d(self): self.cmp_dot(1.1, 2.2)
    #def test_dot_0d_1d(self): self.cmp_dot(1.1, self.rand(5))
    #def test_dot_0d_2d(self): self.cmp_dot(3.0, self.rand(6,7))
    #def test_dot_0d_3d(self): self.cmp_dot(3.0, self.rand(8,6,7))
    #def test_dot_1d_0d(self): self.cmp_dot(self.rand(5), 1.1 )
    def test_dot_1d_1d(self): self.cmp_dot(self.rand(5), self.rand(5))
    def test_dot_1d_2d(self): self.cmp_dot(self.rand(6), self.rand(6,7))
    #def test_dot_1d_3d(self): self.cmp_dot(self.rand(6), self.rand(8,6,7))
    #def test_dot_2d_0d(self): self.cmp_dot(self.rand(5,6), 1.0)
    def test_dot_2d_1d(self): self.cmp_dot(self.rand(5,6), self.rand(6))
    def test_dot_2d_2d(self): self.cmp_dot(self.rand(5,6), self.rand(6,7))
    #def test_dot_2d_3d(self): self.cmp_dot(self.rand(5,6), self.rand(8,6,7))
    #def test_dot_3d_0d(self): self.cmp_dot(self.rand(4,5,6), 1.0)
    #def test_dot_3d_1d(self): self.cmp_dot(self.rand(4,5,6), self.rand(6))
    #def test_dot_3d_2d(self): self.cmp_dot(self.rand(4,5,6), self.rand(6,7))
    #def test_dot_3d_3d(self): self.cmp_dot(self.rand(4,5,6), self.rand(8,6,7))

    def not_aligned(self, x, y):
        z = dot(x,y)
        try:
            tz = eval_outputs([z], mode = compile.Mode(optimizer = None, linker = 'py'))
        except ValueError, e:
            self.failUnless(e[0].split()[1:4] == ['are', 'not', 'aligned'], e)
            return
        self.fail()

    def test_align_1_1(self): self.not_aligned(self.rand(5), self.rand(6))
    def test_align_1_2(self): self.not_aligned(self.rand(5), self.rand(6,4))
    #def test_align_1_3(self): self.not_aligned(self.rand(5), self.rand(6,4,7))
    def test_align_2_1(self): self.not_aligned(self.rand(5,4), self.rand(6))
    def test_align_2_1(self): self.not_aligned(self.rand(5,4), self.rand(6,7))
    #def test_align_2_3(self): self.not_aligned(self.rand(5,4), self.rand(6,7,8))
    #def test_align_3_1(self): self.not_aligned(self.rand(5,4,3), self.rand(6))
    #def test_align_3_2(self): self.not_aligned(self.rand(5,4,3), self.rand(6,7))
    #def test_align_3_3(self): self.not_aligned(self.rand(5,4,3), self.rand(6,7,8))

    def test_grad(self):
        #verify_grad(self, dot, [self.rand(2,3,4), self.rand(4)])
        verify_grad(self, dot, [self.rand(2,3), self.rand(3,2)])
        verify_grad(self, dot, [self.rand(2), self.rand(2,3)])
        verify_grad(self, dot, [self.rand(3,2), self.rand(2)])
        verify_grad(self, dot, [self.rand(2), self.rand(2)])
        #verify_grad(self, dot, [self.rand(), self.rand(2)])
        #verify_grad(self, dot, [self.rand(), self.rand(2,5)])

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
            tz,ta,tx,ty,tb = [as_tensor(p).type() for p in z,a,x,y,b]

            f = function([tz,ta,tx,ty,tb], gemm(tz,ta,tx,ty,tb), mode=compile.Mode(optimizer = None, linker = l))
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

        cmp_linker(copy(z), a, x, y, b, 'c|py')
        cmp_linker(copy(z), a, x, y, b, 'c')
        cmp_linker(copy(z), a, x, y, b, 'py')

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
        Z = as_tensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, Z, Z, 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map1(self):
        """test that only first input can be overwritten"""
        Z = as_tensor(self.rand(2,2))
        A = as_tensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, A, tensor._transpose_inplace(Z), 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map2(self):
        """test that only first input can be overwritten"""
        Z = as_tensor(self.rand(2,2))
        A = as_tensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, tensor._transpose_inplace(Z), A, 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()
    def test_destroy_map3(self):
        """test that only first input can be overwritten"""
        Z = as_tensor(self.rand(2,2))
        A = as_tensor(self.rand(2,2))
        try:
            gemm(Z, 1.0, Z, A, 1.0)
        except ValueError, e:
            if e[0] == Gemm.E_z_uniq:
                return
        self.fail()

    def test_destroy_map4(self):
        """test that dot args can be aliased"""
        Z = value(self.rand(2,2))
        A = value(self.rand(2,2))
        eval_outputs([gemm(Z, 1.0, A, A, 1.0)])
        eval_outputs([gemm(Z, 1.0, A, A.T, 1.0)])


    def test_transposes(self):
        # three square matrices which are not contiguous
        A = self.rand(4,5)[:,:4]
        B = self.rand(4,5)[:,:4]
        C = self.rand(4,5)[:,:4]

        def t(z,x,y,a=1.0, b=0.0,l='c|py',dt='float64'):
            z,a,x,y,b = [numpy.asarray(p,dtype=dt) for p in z,a,x,y,b]
            z_orig = z.copy()
            z_after = self._gemm(z, a, x, y, b)

            tz,ta,tx,ty,tb = [value(p) for p in z,a,x,y,b]

            f = function([tz,ta,tx,ty,tb], gemm(tz,ta,tx,ty,tb), mode = compile.Mode(optimizer = None, linker=l))
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

class T_tensorfromscalar(unittest.TestCase):
    def test0(self):
        s = scal.constant(56)
        t = tensor_from_scalar(s)
        self.failUnless(t.owner.op is tensor_from_scalar)
        self.failUnless(t.type.broadcastable == (), t.type.broadcastable)
        self.failUnless(t.type.ndim == 0, t.type.ndim)
        self.failUnless(t.type.dtype == s.type.dtype)

        v = eval_outputs([t])

        self.failUnless(v == 56, v)
        self.failUnless(isinstance(v, numpy.ndarray))
        self.failUnless(v.shape == (), v.shape)

    def test1(self):
        s = scal.constant(56)
        t = as_tensor(s)
        self.failUnless(t.owner.op is tensor_from_scalar)
        self.failUnless(t.type.broadcastable == (), t.type.broadcastable)
        self.failUnless(t.type.ndim == 0, t.type.ndim)
        self.failUnless(t.type.dtype == s.type.dtype)

        v = eval_outputs([t])

        self.failUnless(v == 56, v)
        self.failUnless(isinstance(v, numpy.ndarray))
        self.failUnless(v.shape == (), v.shape)


# def _tensor(data, broadcastable=None, name=None):
#     """Return a Tensor containing given data"""
#     data = numpy.asarray(data)
#     if broadcastable is None:
#         broadcastable = [s==1 for s in data.shape]
#     elif broadcastable in [0, 1]:
#         broadcastable = [broadcastable] *  len(data.shape)
#     rval = Tensor(data.dtype, broadcastable, name)
#     rval.data = data # will raise if broadcastable was mis-specified
#     return rval



# class T_tensor(unittest.TestCase):
#     def test0(self): # allocate from a scalar float
#         t = _tensor(1.0)
#         self.failUnless(isinstance(t, Tensor))
#         self.failUnless(t.dtype == 'float64')
#         self.failUnless(t.broadcastable == ())
#         self.failUnless(t.role == None)
#         self.failUnless(isinstance(t.data, numpy.ndarray))
#         self.failUnless(str(t.data.dtype) == 'float64')
#         self.failUnless(t.data == 1.0)
#     def test0_int(self): # allocate from a scalar float
#         t = _tensor(1)
#         self.failUnless(isinstance(t, Tensor))
#         self.failUnless(t.dtype == 'int64' or t.dtype == 'int32')
#     def test1(self): # allocate from a vector of ints, not broadcastable
#         t = _tensor(numpy.ones(5,dtype='int32'))
#         self.failUnless(isinstance(t, Tensor))
#         self.failUnless(t.dtype == 'int32')
#         self.failUnless(t.broadcastable == (0,))
#         self.failUnless(isinstance(t.data, numpy.ndarray))
#         self.failUnless(str(t.data.dtype) == 'int32')
#     def test2(self): # allocate from a column matrix of complex with name
#         t = _tensor(numpy.ones((5,1),dtype='complex64'),name='bart')
#         self.failUnless(isinstance(t, Tensor))
#         self.failUnless(t.dtype == 'complex64')
#         self.failUnless(t.broadcastable == (0,1))
#         self.failUnless(isinstance(t.data, numpy.ndarray))
#         self.failUnless(t.name == 'bart')
#     def test2b(self): # allocate from a column matrix, not broadcastable
#         t = _tensor(numpy.ones((5,1),dtype='complex64'),broadcastable=0)
#         self.failUnless(isinstance(t, Tensor))
#         self.failUnless(t.dtype == 'complex64')
#         self.failUnless(t.broadcastable == (0,0))
#         self.failUnless(isinstance(t.data, numpy.ndarray))
#         f = Function([t], [t], linker_cls=gof.CLinker)
#         self.failUnless(numpy.all(t.data == f(t.data)))
#     def test_data_normal(self): #test that assigning to .data works when it should
#         t = _tensor(numpy.ones((5,1),dtype='complex64'), broadcastable=0)
#         o27 = numpy.ones((2,7), dtype='complex64')
#         t.data = o27
#         lst = t._data
#         self.failUnless(t.data.shape == (2,7))
#         self.failUnless(t.data is o27)
#         self.failUnless(t._data is lst)
#     def test_data_badrank0(self):
#         t = _tensor(numpy.ones((5,1),dtype='complex64'), broadcastable=0)
#         try:
#             t.data = numpy.ones((2,7,1))
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is Tensor.filter.E_rank)
#         try:
#             t.data = numpy.ones(1)
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is Tensor.filter.E_rank)
#     def test_data_badrank1(self):
#         t = _tensor(numpy.ones((1,1),dtype='complex64'), broadcastable=1)
#         try:
#             t.data = numpy.ones((1,1,1))
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is Tensor.filter.E_rank)
#         try:
#             t.data = numpy.ones(1)
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is Tensor.filter.E_rank)
#     def test_data_badshape0(self):
#         t = _tensor(numpy.ones((1,1),dtype='complex64'), broadcastable=1)
#         try:
#             t.data = numpy.ones((1,2))
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is Tensor.filter.E_shape)
#         try:
#             t.data = numpy.ones((0,1))
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is Tensor.filter.E_shape)

#     def test_cast0(self):
#         t = Tensor('float32', [0])
#         t.data = numpy.random.rand(4) > 0.5
#         self.failUnless(str(t.data.dtype) == t.dtype)

# class T_stdlib(unittest.TestCase):
#     def test0(self):
#         t = _tensor(1.0)
#         tt = t.clone(False)
#         self.failUnless(t.dtype == tt.dtype)
#         self.failUnless(t.broadcastable is tt.broadcastable)
#         self.failUnless(tt.data is None)
#         self.failUnless(t.data == 1.0)
#     def test0b(self):
#         t = _tensor(1.0)
#         tt = t.clone()
#         self.failUnless(t.dtype == tt.dtype)
#         self.failUnless(t.broadcastable is tt.broadcastable)
#         self.failUnless(tt.data is None)
#         self.failUnless(t.data == 1.0)

#     def test1(self):
#         t = _tensor(1.0)
#         tt = t.clone(True)
#         self.failUnless(t.dtype == tt.dtype)
#         self.failUnless(t.broadcastable is tt.broadcastable)
#         self.failUnless(tt.data == 1.0)
#         self.failUnless(t.data == 1.0)
#         self.failUnless(t.data is not tt.data)
#     def test1b(self):
#         t = _tensor(1.0)
#         tt = copy(t)
#         self.failUnless(t.dtype == tt.dtype)
#         self.failUnless(t.broadcastable is tt.broadcastable)
#         self.failUnless(tt.data == 1.0)
#         self.failUnless(t.data == 1.0)
#         self.failUnless(t.data is not tt.data)


class _test_grad(unittest.TestCase):
    class O(gof.op.Op):
        def __init__(self):
            self.gval0 = scalar('e')
            self.gval1 = scalar('f')
        def make_node(self):
            inputs = [scalar('a'),scalar('c')]
            outputs = [scalar('b'),scalar('d')]
            return gof.Apply(self, inputs, outputs)
        def grad(self, (x0,x1), (gz0,gz1)):
            return self.gval0, self.gval1

    def test_1param(self):
        """grad: Test passing a single result param"""
        o = _test_grad.O()
        a1 = o.make_node()
        self.failUnless(o.gval0 is grad(a1.outputs[0], a1.inputs[0]))

    def test_Nparam(self):
        """grad: Test passing multiple result params"""
        o = _test_grad.O()
        a1 = o.make_node()
        g0,g1 = grad(a1.outputs[0], a1.inputs)
        self.failUnless(o.gval0 is g0)
        self.failUnless(o.gval1 is g1)

    def test_1None_rval(self):
        """grad: Test returning a single None from grad"""
        o = _test_grad.O()
        a1 = o.make_node()
        g = grad(a1.outputs[0], a1.outputs[1])
        self.failUnless(isinstance(g, TensorConstant))
        self.failUnless(g.data == 0)
        try:
            grad(a1.outputs[0], 'wtf')
        except AttributeError, e:
            return
        self.fail()

    def test_NNone_rval(self):
        """grad: Test returning some Nones from grad"""
        o = _test_grad.O()
        a1 = o.make_node()
        g0,g1,g2 = grad(a1.outputs[0], a1.inputs + [scalar('z')])
        self.failUnless(o.gval0 is g0)
        self.failUnless(o.gval1 is g1)
        self.failUnless(isinstance(g2, TensorConstant))
        self.failUnless(g2.data == 0)

class T_op_cache(unittest.TestCase):

    def test0(self):
        """trigger bug in ticket #162"""
        lr = constant(0.011)
        v = matrix()
        v.name = 'v'
        gv = fill(v/v, 1.0)/v - (fill(v/v, 1.0) * v) / (v*v)
        fn_py = function([v], gv, mode = compile.Mode(optimizer = None, linker = 'py'))
        fn_c_or_py = function([v], gv, compile.Mode(optimizer = None, linker = 'c|py'))

        a = numpy.random.rand(5,2)
        self.failUnless(numpy.all(fn_py(a) == fn_c_or_py(a)))

if __name__ == '__main__':
    if 1:
        unittest.main()
    else:
        testcase =  AbsInplaceTester

        suite = unittest.TestLoader()
        suite = suite.loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(suite)


