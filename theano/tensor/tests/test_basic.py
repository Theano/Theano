import traceback, StringIO
import operator

from theano.tensor import *
from theano.tensor import basic as tensor # for hidden symbols
from theano.tensor import inplace

import unittest
from copy import copy
from theano import compile
from theano import gradient
from theano import gof
from theano.gof.python25 import any, all
from theano import gof

from theano.tensor.elemwise import DimShuffle
from theano.compile.mode import get_default_mode
from theano import function
from theano.tests import unittest_tools as utt

### seed random number generator so that unittests are deterministic ###
utt.seed_rng()

def inplace_func(inputs, outputs, mode=get_default_mode()):
    return function(inputs, outputs, mode=mode, accept_inplace=True)


def eval_outputs(outputs):
    variables = inplace_func([], outputs)()
    if len(variables) == 1:
        return variables[0]
    return variables

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

def makeTester(name, op, expected, checks = {}, good = {}, bad_build = {}, bad_runtime = {}, grad = {}):
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
                    f = inplace_func(inputrs, node.outputs)
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while trying to make a Function" \
                        % (self.op, testname)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

                expecteds = self.expected(*inputs)

                try:
                    variables = f(*inputs)
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while calling the Function on the inputs %s" \
                        % (self.op, testname, inputs)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

                if not isinstance(expecteds, (list, tuple)):
                    expecteds = (expecteds, )
                for i, (variable, expected) in enumerate(zip(variables, expecteds)):
                    if variable.dtype != expected.dtype or variable.shape != expected.shape or \
                            numpy.any(numpy.abs(variable - expected) > 1e-10):
                        self.fail("Test %s::%s: Output %s gave the wrong value. With inputs %s, expected %s, got %s."
                                  % (self.op, testname, i, inputs, expected, variable))

                for description, check in self.checks.items():
                    if not check(inputs, variables):
                        self.fail("Test %s::%s: Failed check: %s (inputs were %s, outputs were %s)"
                                  % (self.op, testname, description, inputs, variables))

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
                    f = inplace_func(inputrs, node.outputs)
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while trying to make a Function" \
                        % (self.op, testname)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

                try:
                    variables = f(*inputs)
                except:
                    return

                self.fail("Test %s::%s: Successful call on the following bad inputs: %s"
                          % (self.op, testname, inputs))

        def test_grad(self):
            for testname, inputs in self.grad.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [value(input) for input in inputs]
                try:
                    utt.verify_grad(self.op, inputs)
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


def makeBroadcastTester(op, expected, checks = {}, **kwargs):
    name = str(op) + "Tester"
    if kwargs.has_key('inplace'):
        if kwargs['inplace']:
            _expected = expected
            expected = lambda *inputs: numpy.array(_expected(*inputs), dtype = inputs[0].dtype)
            def inplace_check(inputs, outputs):
                # this used to be inputs[0] is output[0]
                # I changed it so that it was easier to satisfy by the DebugMode
                return numpy.all(inputs[0] == outputs[0])
            checks = dict(checks, inplace_check=inplace_check) #lambda inputs, outputs: numpy.all(inputs[0] == outputs[0]))
        del kwargs['inplace']
    return makeTester(name, op, expected, checks, **kwargs)



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


AddTester = makeBroadcastTester(op = add,
                                  expected = lambda *inputs: reduce(lambda x, y: x + y, inputs),
                                  good = dict(three_inputs_same_shapes = (rand(2, 3), rand(2, 3), rand(2, 3)),
                                              four_inputs_broadcast = (rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
                                              **_good_broadcast_binary_normal),
                                  bad_build = _bad_build_broadcast_binary_normal,
                                  bad_runtime = _bad_runtime_broadcast_binary_normal)


AddInplaceTester = makeBroadcastTester(op = inplace.add_inplace,
                                         expected = lambda x, y: x + y,
                                         good = _good_broadcast_binary_normal,
                                         bad_build = _bad_build_broadcast_binary_normal,
                                         bad_runtime = _bad_runtime_broadcast_binary_normal,
                                         inplace = True)

SubTester = makeBroadcastTester(op = sub,
                                  expected = lambda x, y: x - y,
                                  good = _good_broadcast_binary_normal,
                                  bad_build = _bad_build_broadcast_binary_normal,
                                  bad_runtime = _bad_runtime_broadcast_binary_normal,
                                  grad = _grad_broadcast_binary_normal)

SubInplaceTester = makeBroadcastTester(op = inplace.sub_inplace,
                                         expected = lambda x, y: x - y,
                                         good = _good_broadcast_binary_normal,
                                         bad_build = _bad_build_broadcast_binary_normal,
                                         bad_runtime = _bad_runtime_broadcast_binary_normal,
                                         grad = _grad_broadcast_binary_normal,
                                         inplace = True)

MaximumTester = makeBroadcastTester(op = maximum,
                                  expected = numpy.maximum,
                                  good = _good_broadcast_binary_normal,
                                  bad_build = _bad_build_broadcast_binary_normal,
                                  bad_runtime = _bad_runtime_broadcast_binary_normal,
                                  grad = _grad_broadcast_binary_normal)

MaximumInplaceTester = makeBroadcastTester(op = inplace.maximum_inplace,
                                         expected = numpy.maximum,
                                         good = _good_broadcast_binary_normal,
                                         bad_build = _bad_build_broadcast_binary_normal,
                                         bad_runtime = _bad_runtime_broadcast_binary_normal,
                                         grad = _grad_broadcast_binary_normal,
                                         inplace = True)

MinimumTester = makeBroadcastTester(op = minimum,
                                  expected = numpy.minimum,
                                  good = _good_broadcast_binary_normal,
                                  bad_build = _bad_build_broadcast_binary_normal,
                                  bad_runtime = _bad_runtime_broadcast_binary_normal,
                                  grad = _grad_broadcast_binary_normal)

MinimumInplaceTester = makeBroadcastTester(op = inplace.minimum_inplace,
                                         expected = numpy.minimum,
                                         good = _good_broadcast_binary_normal,
                                         bad_build = _bad_build_broadcast_binary_normal,
                                         bad_runtime = _bad_runtime_broadcast_binary_normal,
                                         grad = _grad_broadcast_binary_normal,
                                         inplace = True)

MulTester = makeBroadcastTester(op = mul,
                                  expected = lambda *inputs: reduce(lambda x, y: x * y, inputs),
                                  good = dict(three_inputs_same_shapes = (rand(2, 3), rand(2, 3), rand(2, 3)),
                                              four_inputs_broadcast = (rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
                                              **_good_broadcast_binary_normal),
                                  bad_build = _bad_build_broadcast_binary_normal,
                                  bad_runtime = _bad_runtime_broadcast_binary_normal,
                                  grad = dict(three_inputs_same_shapes = (rand(2, 3), rand(2, 3), rand(2, 3)),
                                              four_inputs_broadcast = (rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
                                              **_grad_broadcast_binary_normal))
MulInplaceTester = makeBroadcastTester(op = inplace.mul_inplace,
                                         expected = lambda x, y: x * y,
                                         good = _good_broadcast_binary_normal,
                                         bad_build = _bad_build_broadcast_binary_normal,
                                         bad_runtime = _bad_runtime_broadcast_binary_normal,
                                         grad = _grad_broadcast_binary_normal,
                                         inplace = True)

DivTester = makeBroadcastTester(op = true_div,
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
DivInplaceTester = makeBroadcastTester(op = inplace.true_div_inplace,
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

ModTester = makeBroadcastTester(op = mod,
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
ModInplaceTester = makeBroadcastTester(op = inplace.mod_inplace,
                                         expected = lambda x, y: x % y,
                                         good = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                                     scalar = (rand(2, 3), rand(1, 1)),
                                                     row = (rand(2, 3), rand(1, 3)),
                                                     column = (rand(2, 3), rand(2, 1)),
                                                     dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
                                                     dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3))
                                                     ),
                                         inplace = True)

PowTester = makeBroadcastTester(op = pow,
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
PowInplaceTester = makeBroadcastTester(op = inplace.pow_inplace,
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


AbsTester = makeBroadcastTester(op = tensor.abs_,
                                  expected = lambda x: abs(x),
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
AbsInplaceTester = makeBroadcastTester(op = inplace.abs__inplace,
                                         expected = lambda x: numpy.abs(x),
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

NegTester = makeBroadcastTester(op = neg,
                                  expected = lambda x: -x,
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
NegInplaceTester = makeBroadcastTester(op = inplace.neg_inplace,
                                         expected = lambda x: -x,
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

SgnTester = makeBroadcastTester(op = sgn,
                                  expected = numpy.sign,
                                  good = _good_broadcast_unary_normal)
SgnInplaceTester = makeBroadcastTester(op = inplace.sgn_inplace,
                                         expected = numpy.sign,
                                         good = _good_broadcast_unary_normal,
                                         inplace = True)

SqrTester = makeBroadcastTester(op = sqr,
                                  expected = numpy.square,
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
SqrInplaceTester = makeBroadcastTester(op = inplace.sqr_inplace,
                                         expected = numpy.square,
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

ExpTester = makeBroadcastTester(op = exp,
                                  expected = numpy.exp,
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
ExpInplaceTester = makeBroadcastTester(op = inplace.exp_inplace,
                                         expected = numpy.exp,
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)


_good_broadcast_unary_positive = dict(normal = (rand_ranged(0.001, 5, (2, 3)),),
                                      integers = (randint_ranged(1, 5, (2, 3)),))

_grad_broadcast_unary_positive = dict(normal = (rand_ranged(0.001, 5, (2, 3)),))

LogTester = makeBroadcastTester(op = log,
                                  expected = numpy.log,
                                  good = _good_broadcast_unary_positive,
                                  grad = _grad_broadcast_unary_positive)
LogInplaceTester = makeBroadcastTester(op = inplace.log_inplace,
                                         expected = numpy.log,
                                         good = _good_broadcast_unary_positive,
                                         grad = _grad_broadcast_unary_positive,
                                         inplace = True)

Log2Tester = makeBroadcastTester(op = log2,
                                   expected = numpy.log2,
                                   good = _good_broadcast_unary_positive,
                                   grad = _grad_broadcast_unary_positive)
Log2InplaceTester = makeBroadcastTester(op = inplace.log2_inplace,
                                          expected = numpy.log2,
                                          good = _good_broadcast_unary_positive,
                                          grad = _grad_broadcast_unary_positive,
                                          inplace = True)

Log10Tester = makeBroadcastTester(op = log10,
                                   expected = numpy.log10,
                                   good = _good_broadcast_unary_positive,
                                   grad = _grad_broadcast_unary_positive)
Log10InplaceTester = makeBroadcastTester(op = inplace.log10_inplace,
                                          expected = numpy.log10,
                                          good = _good_broadcast_unary_positive,
                                          grad = _grad_broadcast_unary_positive,
                                          inplace = True)

Log1pTester = makeBroadcastTester(op = log1p,
                                  expected = numpy.log1p,
                                  good = _good_broadcast_unary_positive,
                                  grad = _grad_broadcast_unary_positive)
Log1pInplaceTester = makeBroadcastTester(op = inplace.log1p_inplace,
                                         expected = numpy.log1p,
                                         good = _good_broadcast_unary_positive,
                                         grad = _grad_broadcast_unary_positive,
                                         inplace = True)


SqrtTester = makeBroadcastTester(op = sqrt,
                                   expected = numpy.sqrt,
                                   good = _good_broadcast_unary_positive,
                                   grad = _grad_broadcast_unary_positive)
SqrtInplaceTester = makeBroadcastTester(op = inplace.sqrt_inplace,
                                          expected = numpy.sqrt,
                                          good = _good_broadcast_unary_positive,
                                          grad = _grad_broadcast_unary_positive,
                                          inplace = True)



_good_broadcast_unary_wide = dict(normal = (rand_ranged(-1000, 1000, (2, 3)),),
                                  integers = (randint_ranged(-1000, 1000, (2, 3)),))

_grad_broadcast_unary_wide = dict(normal = (rand_ranged(-1000, 1000, (2, 3)),))


SinTester = makeBroadcastTester(op = sin,
                                  expected = numpy.sin,
                                  good = _good_broadcast_unary_wide,
                                  grad = _grad_broadcast_unary_wide)
SinInplaceTester = makeBroadcastTester(op = inplace.sin_inplace,
                                         expected = numpy.sin,
                                         good = _good_broadcast_unary_wide,
                                         grad = _grad_broadcast_unary_wide,
                                         inplace = True)

CosTester = makeBroadcastTester(op = cos,
                                  expected = numpy.cos,
                                  good = _good_broadcast_unary_wide,
                                  grad = _grad_broadcast_unary_wide)
CosInplaceTester = makeBroadcastTester(op = inplace.cos_inplace,
                                         expected = numpy.cos,
                                         good = _good_broadcast_unary_wide,
                                         grad = _grad_broadcast_unary_wide,
                                         inplace = True)

TanTester = makeBroadcastTester(op = tan,
                                  expected = numpy.tan,
                                  good = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                              shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                  grad = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                              shifted = (rand_ranged(3.15, 6.28, (2, 3)),)))
TanInplaceTester = makeBroadcastTester(op = inplace.tan_inplace,
                                         expected = numpy.tan,
                                         good = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                                     shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                         grad = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                                     shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                         inplace = True)


CoshTester = makeBroadcastTester(op = cosh,
                                   expected = numpy.cosh,
                                   good = _good_broadcast_unary_normal,
                                   grad = _grad_broadcast_unary_normal)
CoshInplaceTester = makeBroadcastTester(op = inplace.cosh_inplace,
                                          expected = numpy.cosh,
                                          good = _good_broadcast_unary_normal,
                                          grad = _grad_broadcast_unary_normal,
                                          inplace = True)

SinhTester = makeBroadcastTester(op = sinh,
                                   expected = numpy.sinh,
                                   good = _good_broadcast_unary_normal,
                                   grad = _grad_broadcast_unary_normal)
SinhInplaceTester = makeBroadcastTester(op = inplace.sinh_inplace,
                                          expected = numpy.sinh,
                                          good = _good_broadcast_unary_normal,
                                          grad = _grad_broadcast_unary_normal,
                                          inplace = True)

TanhTester = makeBroadcastTester(op = tanh,
                                   expected = numpy.tanh,
                                   good = _good_broadcast_unary_normal,
                                   grad = _grad_broadcast_unary_normal)
TanhInplaceTester = makeBroadcastTester(op = inplace.tanh_inplace,
                                          expected = numpy.tanh,
                                          good = _good_broadcast_unary_normal,
                                          grad = _grad_broadcast_unary_normal,
                                          inplace = True)



DotTester = makeTester(name = 'DotTester',
                        op = dot,
                        expected = lambda x, y: numpy.dot(x, y),
                        checks = {},
                        good = dict(correct1 = (rand(5, 7), rand(7, 5)),
                                    correct2 = (rand(5, 7), rand(7, 9)),
                                    correct3 = (rand(5, 7), rand(7))),
                        bad_build = dict(),
                        bad_runtime = dict(bad1 = (rand(5, 7), rand(5, 7)),
                                           bad2 = (rand(5, 7), rand(8, 3))))



#TODO: consider moving this function / functionality to gradient.py
#      rationale: it's tricky, and necessary everytime you want to verify
#      gradient numerically



#useful mostly for unit tests
def _approx_eq(a,b,eps=1.0e-4):
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    if a.shape != b.shape:
        if _approx_eq.debug:
            print a.shape, b.shape
        return False
    abs_rel_err = numeric_grad.abs_rel_err(a,b)
    if numpy.max(abs_rel_err) >= eps:
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

class T_max_and_argmax(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test0(self):
        n = as_tensor_variable(5.0)
        v,i = eval_outputs(max_and_argmax(n))
        self.failUnless(v == 5.0)
        self.failUnless(i == 0)
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v)==0

    def test1(self):
        n = as_tensor_variable([1,2,3,2,-6])
        v,i = eval_outputs(max_and_argmax(n))
        self.failUnless(v == 3)
        self.failUnless(i == 2)
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v)==0

    def test2(self):
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)
        v,i = eval_outputs(max_and_argmax(n))
        self.failUnless(numpy.all(v == numpy.max(data,-1)))
        self.failUnless(numpy.all(i == numpy.argmax(data,-1)))
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert v==(2)

    def test2b(self):
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)
        v,i = eval_outputs(max_and_argmax(n,0))
        self.failUnless(numpy.all(v == numpy.max(data,0)))
        self.failUnless(numpy.all(i == numpy.argmax(data,0)))
        v = eval_outputs(max_and_argmax(n,0)[0].shape)
        assert v==(3)
        v = eval_outputs(max_and_argmax(n,1)[0].shape)
        assert v==(2)
#        v = eval_outputs(max_and_argmax(n,[0,1])[0].shape)
#        assert v==()

    def test2_invalid(self):
        n = as_tensor_variable(numpy.random.rand(2,3))
        # Silence expected error messages
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.getEffectiveLevel()
        _logger.setLevel(logging.CRITICAL)
        try:
            try:
                eval_outputs(max_and_argmax(n,3))
                assert False
            except ValueError, e:
                pass
        finally:
            _logger.setLevel(oldlevel)
    def test2_invalid_neg(self):
        n = as_tensor_variable(numpy.random.rand(2,3))
        old_stderr = sys.stderr
        sys.stderr = StringIO.StringIO()
        try:
            try:
                eval_outputs(max_and_argmax(n,-3))
                assert False
            except ValueError, e:
                pass
        finally:
            sys.stderr = old_stderr
    def test2_valid_neg(self):
        n = as_tensor_variable(numpy.random.rand(2,3))
        v,i = eval_outputs(max_and_argmax(n,-1))
        self.failUnless(v.shape == (2,))
        v,i = eval_outputs(max_and_argmax(n,-2))
        self.failUnless(v.shape == (3,))
        v = eval_outputs(max_and_argmax(n,-1)[0].shape)
        assert v==(2)
        v = eval_outputs(max_and_argmax(n,-2)[0].shape)
        assert v==(3)

    def test3(self):
        n = as_tensor_variable(numpy.random.rand(2,3,4))
        v,i = eval_outputs(max_and_argmax(n,0))
        self.failUnless(v.shape == (3,4))
        self.failUnless(i.shape == (3,4))
        v,i = eval_outputs(max_and_argmax(n,1))
        self.failUnless(v.shape == (2,4))
        self.failUnless(i.shape == (2,4))
        v,i = eval_outputs(max_and_argmax(n,2))
        self.failUnless(v.shape == (2,3))
        self.failUnless(i.shape == (2,3))
        v = eval_outputs(max_and_argmax(n,0)[0].shape)
        assert tuple(v)==(3,4)
        v = eval_outputs(max_and_argmax(n,1)[0].shape)
        assert tuple(v)==(2,4)
        v = eval_outputs(max_and_argmax(n,2)[0].shape)
        assert tuple(v)==(2,3)


class T_subtensor(unittest.TestCase):
    def setUp(self):
        Subtensor.debug = False
        utt.seed_rng()

    def test0_err_invalid(self):
        #it is impossible to retrieve a view of a 0-d tensor
        n = as_tensor_variable(numpy.ones(()))
        try:
            t = n[0]
        except ValueError, e:
            self.failUnless(e[0] is Subtensor.e_invalid)
            return
        self.fail()

    def test1_err_bounds(self):
        n = as_tensor_variable(numpy.ones(3))
        t = n[7]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        # Silence expected error messages
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.getEffectiveLevel()
        _logger.setLevel(logging.CRITICAL)
        try:
            try:
                tval = eval_outputs([t])
                assert 0
            except Exception, e:
                if e[0] != 'index out of bounds':
                    raise
        finally:
            _logger.setLevel(oldlevel)
    def test1_err_subslice(self):
        n = as_tensor_variable(numpy.ones(3))
        try:
            t = n[slice(0,slice(1,2,None),None)]
        except Exception, e:
            ### Relax constraint on the type of Exception,
            ### since this might be handled by AvancedSubtensor
            #if e[0] != Subtensor.e_indextype:
            #    raise
            return
        self.fail()

    def test1_ok_range_finite(self):
        n = as_tensor_variable(numpy.ones(3)*5)
        t = n[0:2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test2_ok_range_finite(self):
        n = as_tensor_variable(numpy.ones((3,4))*5)
        t = n[0:2,3]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test1_err_invalid(self):
        n = as_tensor_variable(numpy.ones(1))
        try:
            t = n[0,0]
        except ValueError, e:
            self.failUnless(e[0] is Subtensor.e_invalid)
            return
        self.fail()
    def test1_ok_elem(self):
        n = as_tensor_variable(numpy.ones(1)*5)
        t = n[0]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == ())
        self.failUnless(tval == 5.0)
    def test1_ok_range_infinite(self):
        #Subtensor.debug = True
        n = as_tensor_variable(numpy.ones(3)*5)
        t = n[1:]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)
    def test1_ok_strided(self):
        n = as_tensor_variable(numpy.ones(5)*5)
        t = n[1::2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)

        tval = eval_outputs([n[0:-1:2]]) #0 to 1 from the end stepping by 2
        self.failUnless(tval.shape == (2,))
        self.failUnless(tval[1] == 5.0)

    def test2_err_bounds0(self):
        n = as_tensor_variable(numpy.ones((2,3))*5)
        t = n[0,4]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        # Silence expected warnings
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.getEffectiveLevel()
        _logger.setLevel(logging.CRITICAL)
        try:
            try:
                tval = eval_outputs([t])
                assert 0
            except IndexError, e:
                pass
        finally:
            _logger.setLevel(oldlevel)
    def test2_err_bounds1(self):
        n = as_tensor_variable(numpy.ones((2,3))*5)
        t = n[4:5,2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        old_stderr = sys.stderr
        sys.stderr = StringIO.StringIO()
        try:
            try:
                tval = eval_outputs([t])
            except Exception, e:
                if e[0] != 'index out of bounds':
                    raise
        finally:
            sys.stderr = old_stderr
    def test2_ok_elem(self):
        n = as_tensor_variable(numpy.asarray(range(6)).reshape((2,3)))
        t = n[0,2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == ())
        self.failUnless(numpy.all(tval == 2))
    def test2_ok_row(self):
        n = as_tensor_variable(numpy.asarray(range(6)).reshape((2,3)))
        t = n[1]
        self.failIf(any(n.type.broadcastable))
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (3,))
        self.failUnless(numpy.all(tval == [3,4,5]))

    def test2_ok_col(self):
        n = as_tensor_variable(numpy.ones((2,3))*5)
        t = n[:,0]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        self.failIf(any(n.type.broadcastable))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(numpy.all(tval == 5.0))

    def test2_ok_rows_finite(self):
        n = as_tensor_variable(numpy.ones((4,3))*5)
        t = n[1:3,0]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,))
        self.failUnless(numpy.all(tval == 5.0))

    def test2_ok_cols_infinite(self):
        n = as_tensor_variable(numpy.asarray(range(12)).reshape((4,3)))
        t = n[1,2:]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (1,))
        self.failUnless(numpy.all(tval == 5))

    def test2_ok_strided(self):
        n = as_tensor_variable(numpy.asarray(range(20)).reshape((4,5)))
        t = n[1:4:2,1:5:2]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == (2,2))
        self.failUnless(numpy.all(tval == [[6, 8],[16, 18]]))

    def test3_ok_mat(self):
        n = as_tensor_variable(numpy.asarray(range(24)).reshape((2,3,4)))
        t = n[0,0,0]
        self.failUnless(isinstance(t.owner.op, Subtensor))
        tval = eval_outputs([t])
        self.failUnless(tval.shape == ())
        self.failUnless(numpy.all(tval == 0))

    def test_grad_1d(self):
        subi = 0
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)
        z = scal.constant(subi)
        t = n[z:,z]
        gn = grad(sum(exp(t)), n)
        gval = eval_outputs([gn])
        good = numpy.zeros_like(data)
        good[subi:,subi] = numpy.exp(data[subi:,subi])
        self.failUnless(numpy.all(gval == good), (gval, good))

    def test_grad_0d(self):
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)
        t = n[1,0]
        gn = grad(sum(exp(t)), n)
        f = function([], gn, mode=None)
        print 'toposort', f.maker.env.toposort()
        gval = f()
        print gval
        good = numpy.zeros_like(data)
        good[1,0] = numpy.exp(data[1,0])
        self.failUnless(numpy.allclose(gval, good), (gval, good))


class T_Join_and_Split(unittest.TestCase):
    """
    Split is tested by each verify_grad method.
    """

    class Join1(Op):
        def make_node(self, *inputs):
            inputs = [as_tensor_variable(t) for t in inputs]
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
        a = as_tensor_variable(1)
        b = as_tensor_variable(2)
        try:
            s = join(0, a, b)
        except:
            return
        self.fail()

    def test_stack_mixed_type_constants(self):
        a = as_tensor_variable(1)
        b = as_tensor_variable(2.0)
        c = as_tensor_variable(3.0)
        s = stack(a, b, c)

        want = numpy.array([1, 2, 3])
        self.failUnless((eval_outputs([s]) == want).all())

    def test_stack_scalar(self):
        a = as_tensor_variable(1)
        b = as_tensor_variable(2)
        c = as_tensor_variable(3)
        s = stack(a, b, c)

        want = numpy.array([1, 2, 3])
        self.failUnless((eval_outputs([s]) == want).all())

    def test_stack_scalar_make_vector(self):
        '''Test that calling stack() on scalars instantiates MakeVector,
        not Join.'''
        a = tensor.scalar('a')
        b = tensor.scalar('b')
        s = stack(a, b, a, b)
        f = function([a,b], s)
        val = f(1,2)
        print val
        self.failUnless(numpy.all(val == [1,2,1,2]))
        e = f.maker.env.toposort()
        assert len([n for n in e if n.op == opt.make_vector]) > 0
        assert len([n for n in e if isinstance(n, Join)]) == 0

    def test_join_vector(self):
        a = as_tensor_variable(numpy.array([1, 2, 3]))
        b = as_tensor_variable(numpy.array([7, 8, 9]))

        s = join(0, a, b)
        want = numpy.array([1, 2, 3, 7, 8, 9])
        self.failUnless((eval_outputs([s]) == want).all())

    def test_stack_vector(self):
        a = as_tensor_variable(numpy.array([1, 2, 3]))
        b = as_tensor_variable(numpy.array([7, 8, 9]))

        s = stack(a, b)
        want = numpy.array([[1, 2, 3],[ 7, 8, 9]])
        self.failUnless((eval_outputs([s]) == want).all())

    def test_join_matrix0(self):
        a = as_tensor_variable(numpy.array([[1, 2, 3], [4, 5, 6]]))
        b = as_tensor_variable(numpy.array([[7, 8, 9]]))
        s = join(0, a, b)

        want = numpy.array([[1, 2, 3],[4,5,6],[7, 8, 9]])
        self.failUnless((eval_outputs([s]) == want).all())

    def test_join_matrix1(self):
        av=numpy.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
        bv= numpy.array([[7], [8]],dtype='float32')
        a = as_tensor_variable(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = numpy.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype='float32')
        self.failUnless((eval_outputs([s]) == want).all())

        utt.verify_grad(lambda a, b: join(1,a,b), [av, bv], eps=1.0e-4, tol=1.0e-3)

    def test_join_matrix1_using_vertical_stack(self):
        a = as_tensor_variable(numpy.array([[1, 2, 3], [4, 5, 6]]))
        b = as_tensor_variable(numpy.array([[7, 8, 9]]))
        c = as_tensor_variable(numpy.array([[9, 8, 7]]))
        s = vertical_stack(a, b, c)

        want = numpy.array([[1, 2, 3],[4,5,6],[7, 8, 9], [9, 8, 7]])
        self.failUnless((eval_outputs([s]) == want).all())

    def test_join_matrix1_using_horizontal_stack(self):
        av=numpy.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
        bv=numpy.array([[7], [8]],dtype='float32')
        cv=numpy.array([[3, 2, 1], [6, 5, 4]], dtype='float32')
        a = as_tensor_variable(av)
        b = as_tensor_variable(bv)
        c = as_tensor_variable(cv)
        s = horizontal_stack(a, b, c)
        want = numpy.array([[1, 2, 3, 7, 3, 2, 1], [4, 5, 6, 8, 6, 5, 4]], dtype='float32')
        self.failUnless((eval_outputs([s]) == want).all())

        utt.verify_grad(lambda a, b: join(1,a,b), [av, bv], eps=1.0e-4, tol=1.0e-3)

    def test_join_matrixV(self):
        """variable join axis"""
        v = numpy.array([[1., 2., 3.], [4., 5., 6.]])
        a = as_tensor_variable(v.copy())
        b = as_tensor_variable(v.copy())
        ax = lscalar()
        s = join(ax, a, b)

        f = inplace_func([ax], [s])

        want = numpy.array([[1, 2, 3], [4, 5, 6] ,[1, 2, 3], [4, 5, 6]])
        got = f(0)
        self.failUnless((got == want).all(), (got, want))

        want = numpy.array([[ 1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]])
        got = f(1)
        self.failUnless((got == want).all(), (got, want))

        utt.verify_grad(lambda a, b: join(0,a,b), [v, 2*v])
        utt.verify_grad(lambda a, b: join(1,a,b), [v, 2*v])

    def test_vector_len(self):
        x = lscalar('x')
        y = dscalar('y')

        triple = as_tensor_variable((x, y, 9.0))
        assert 3 == get_vector_length(triple)

        a,b,c = triple
        f = function([x,y], [b,c,a])
        assert numpy.allclose(f(4, 5), [5, 9, 4])

class test_comparison(unittest.TestCase):
    def test_gt(self):
        x, y = fvector(), fvector()
        fn = inplace_func([x,y], x > y)
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l > r)), (v, (l>r)))

    def test_lt(self):
        x, y = fvector(), fvector()
        fn = inplace_func([x,y], x < y)
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l < r)), (v, (l<r)))

    def test_le(self):
        x, y = fvector(), fvector()
        fn = inplace_func([x,y], x <= y)
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l <= r)), (v, (l<=r)))

    def test_ge(self):
        x, y = fvector(), fvector()
        fn = inplace_func([x,y], x >= y)
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l >= r)), (v, (l>=r)))

    def test_eq(self):
        x, y = fvector(), fvector()
        fn = inplace_func([x,y], eq(x,y))
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l == r)), (v, (l==r)))

    def test_neq(self):
        x, y = fvector(), fvector()
        fn = inplace_func([x,y], neq(x, y))
        l = numpy.asarray([0.,-1.,1.])
        r = numpy.asarray([0.,1.,-1.])
        v = fn(l, r)
        self.failUnless(numpy.all(v == (l != r)), (v, (l!=r)))

class test_bitwise(unittest.TestCase):
    def test_or(self):
        x, y = bvector(), bvector()
        fn = inplace_func([x,y], x|y)
        l = theano._asarray([0,0,1,1], dtype = 'int8')
        r = theano._asarray([0,1,0,1], dtype = 'int8')
        v = fn(l, r)
        self.failUnless(numpy.all(v == (operator.or_(l, r))), (l, r, v))

    def test_xor(self):
        x, y = bvector(), bvector()
        fn = inplace_func([x,y], x^y)
        ix = x
        ix = inplace.xor_inplace(ix, y)
        gn = inplace_func([x,y], ix)
        l = theano._asarray([0,0,1,1], dtype = 'int8')
        r = theano._asarray([0,1,0,1], dtype = 'int8')
        v = fn(l, r)
        self.failUnless(numpy.all(v == (operator.xor(l, r))), (l, r, v))
        v = gn(l, r)
        #test the in-place stuff
        self.failUnless(numpy.all(l == numpy.asarray([0,1,1,0])), l)

    def test_and(self):
        x, y = bvector(), bvector()
        fn = inplace_func([x,y], x&y)
        l = theano._asarray([0,0,1,1], dtype = 'int8')
        r = theano._asarray([0,1,0,1], dtype = 'int8')
        v = fn(l, r)
        self.failUnless(numpy.all(v == (operator.and_(l, r))), (l, r, v))

    def test_inv(self):
        x, y = bvector(), bvector()
        fn = inplace_func([x,y], ~x)
        l = theano._asarray([0,0,1,1], dtype = 'int8')
        r = theano._asarray([0,1,0,1], dtype = 'int8')
        v = fn(l, r)
        self.failUnless(numpy.all(v == (~l)), (l, r, v))

    def test_eye(self):
        n = iscalar()
        m = iscalar()
        k = iscalar()
        fn = theano.function([m,n,k],eye(m,n,k) )
        self.failUnless(numpy.all(fn(5,6,1) == numpy.eye(5,6,1)))


class T_add(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_complex_all_ops(self):
        for nbits in (64, 128):
            a = value(numpy.ones(3, dtype='complex%i' % nbits)+0.5j)
            b = value(numpy.ones(3, dtype='complex%i' % nbits)+1.5j)
            tests = (("+", lambda x,y: x+y),
                     ("-", lambda x,y: x-y),
                     ("*", lambda x,y: x*y),
                     ("/", lambda x,y: x/y))
            for s, fn in tests:
                f = inplace_func([a,b], fn(a, b))
                print 'valid output:', fn(a.data, b.data)
                print 'theano output:', f(a.data, b.data)
                self.failUnless(a.type.values_eq_approx(fn(a.data, b.data), f(a.data, b.data)))

    def test_grad_scalar_l(self):
        utt.verify_grad(add, [numpy.asarray([3.0]), numpy.random.rand(3)])
    def test_grad_scalar_r(self):
        utt.verify_grad(add, [numpy.random.rand(3), numpy.asarray([3.0])])
    def test_grad_row(self):
        utt.verify_grad(add, [numpy.random.rand(3, 5), numpy.random.rand(1, 5)])
    def test_grad_col(self):
        utt.verify_grad(add, [numpy.random.rand(3, 5), numpy.random.rand(3, 1)])

class T_ceil(unittest.TestCase):
    def test_complex(self):
        self.assertRaises(TypeError, ceil, zvector())

class T_exp(unittest.TestCase):
    def test_grad_0(self):
        utt.verify_grad(exp, [
            numpy.asarray([[ 1.5089518 ,  1.48439076, -4.7820262 ],
            [ 2.04832468,  0.50791564, -1.58892269]])])
    def test_grad_1(self):
        utt.verify_grad(inplace.exp_inplace, [
            numpy.asarray([[ 1.5089518 ,  1.48439076, -4.7820262 ],
            [ 2.04832468,  0.50791564, -1.58892269]])])

    def test_int(self):
        x = ivector()
        f = function([x], exp(x))
        exp_3 = f([3])
        assert exp_3.dtype == 'float64'

    def test_complex(self):
        x = zvector()
        assert exp(x).dtype == 'complex128'
        f = function([x], exp(x))
        exp_3 = f([3+2j])
        assert numpy.allclose(exp_3, numpy.exp(3+2j))

class T_divimpl(unittest.TestCase):
    def test_impls(self):
        i = iscalar()
        ii = lscalar()
        d = dscalar()
        f = fscalar()
        c = cscalar()

        assert numpy.allclose(function([i, ii, d, f, c], i/d)(5, 3, 7.0, 11.0, complex(5,3)),
                (5.0/7.0))
        assert numpy.allclose(function([i, ii, d, f, c], d/i)(5, 3, 7.0, 11.0, complex(5,3)),
                (7.0/5.0))
        assert numpy.allclose(function([i, ii, d, f, c], i/f)(5, 3, 7.0, 11.0, complex(5,3)),
                (5.0/11.0))
        assert numpy.allclose(function([i, ii, d, f, c], f/i)(5, 3, 7.0, 11.0, complex(5,3)),
                (11.0/5.0))
        assert numpy.allclose(function([i, ii, d, f, c], i/ii)(5, 3, 7.0, 11.0, complex(5,3)),
                (5/3))
        assert numpy.allclose(function([i, ii, d, f, c], ii/i)(5, 3, 7.0, 11.0, complex(5,3)),
                (3/5))
        assert numpy.allclose(function([i, ii, d, f, c], true_div(i,ii))(5, 3, 7.0, 11.0, complex(5,3)),
                (5./3.))
        assert numpy.allclose(function([i, ii, d, f, c], true_div(ii,i))(5, 3, 7.0, 11.0, complex(5,3)),
                (3./5.))

# class T_abs(unittest.TestCase):
#     def test_impl(self):
#         t = as_tensor_variable(1.0)
#         check_eq(self, t, abs(t), 1.0, 1.0)
#         check_eq(self, t, abs(t), -1.0, 1.0)

#         for shape in (2,), (3,4):
#             t = as_tensor_variable(numpy.ones(shape))
#             d = numpy.random.rand(*shape)*2-1.0
#             check_eq(self, t, abs(t), d, abs(d))
#             check_eq(self, t, abs(t), -d, abs(-d))

#     def test_grad(self):
#         utt.verify_grad(Abs, [numpy.ones(())])
#         utt.verify_grad(Abs, [numpy.ones(3)])

#     class AbsBadGrad(Abs):
#         def grad(self, (x, ), (gz, )):
#             return mul(gz * sgn(x),0.9),

#     def test_badgrad(self):
#         try:
#             utt.verify_grad(T_abs.AbsBadGrad, [numpy.ones(())])
#         except Exception, e:
#             self.failUnless(str(e) == utt.verify_grad.E_grad, str(e))
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
#         x = as_tensor_variable(numpy.ones((4,5)))
#         l = ones_like(x[:,0:1])
#         r = ones_like(x[0:1,:])
#         xx = x + dot(l,r)
#         self.failUnless(numpy.mean(eval_outputs([xx]) == 2.0))

# class T_sum(unittest.TestCase):
#     def test_impl(self):
#         t = as_tensor_variable(0.0)
#         check_eq(self, t, Sum(t).out, 1.0, 1.0)
#         check_eq(self, t, Sum(t).out, -1.0, -1.0)

#         t = as_tensor_variable([0.0, 0.0])
#         d = numpy.asarray([-0.4, 1.2])
#         check_eq(self, t, Sum(t).out, d, numpy.sum(d))
#         check_eq(self, t, Sum(t).out, -d, -numpy.sum(d))

# class T_mul(unittest.TestCase):
#     def setUp(self):
#         utt.seed_rng()

#     def test_elemwise(self):
#         a = as_tensor_variable(0.0)
#         b = as_tensor_variable(0.0)
#         check_eq2_both(self, [a,b], mul(a,b), [3.0, 4.0], 12.0)
#         check_eq2_both(self, [a,b], mul(b,a), [-1.0,2.0], -2.0)

#         a = as_tensor_variable(numpy.ones(2))
#         b = as_tensor_variable(numpy.ones(2))
#         aa = numpy.asarray([-0.5, 4.0])
#         bb = numpy.asarray([-0.5, 2.0])
#         check_eq2_both(self, [a,b], mul(a,b), [aa,bb], numpy.asarray([0.25, 8.0]))
#         check_eq2_both(self, [a,b], mul(a,b), [bb,aa], numpy.asarray([0.25, 8.0]))

#     def test_scalar(self):
#         r = numpy.random.rand(2,3)
#         a = as_tensor_variable(r)
#         b = as_tensor_variable(2.0)
#         check_eq2_both(self, [a,b], mul(a,b), [r, 2.0], r*2.0)
#         check_eq2_both(self, [a,b], mul(a,b), [r, 4.0], r*4.0)
#         self.failUnless(b.data == 2.0)

#     def test_rowcol(self):
#         r1 = numpy.random.rand(3,5)
#         r2 = numpy.random.rand(1,5)
#         r3 = numpy.random.rand(3,1)
#         a1, a2, a3 = as_tensor_variable(r1), as_tensor_variable(r2), as_tensor_variable(r3)
#         check_eq2_both(self, [a1,a2], mul(a1,a2), [r1, r2], r1*r2)
#         check_eq2_both(self, [a1,a3], mul(a1,a3), [r1, r3], r1*r3)

#     def test_grad_elemwise(self):
#         utt.verify_grad(Mul, [numpy.random.rand(3,4), numpy.random.rand(3,4)])
#     def test_grad_scalar_l(self):
#         utt.verify_grad(Mul, [numpy.asarray([3.0]), numpy.random.rand(3)])
#     def test_grad_scalar_r(self):
#         utt.verify_grad(Mul, [numpy.random.rand(3), numpy.asarray([3.0])])
#     def test_grad_row(self):
#         utt.verify_grad(Mul, [numpy.random.rand(3, 5), numpy.random.rand(1, 5)])
#     def test_grad_row2(self):
#         op = lambda x, y: Mul(x, DimShuffle(y, ['x', 0]).out)
#         utt.verify_grad(op, [numpy.random.rand(3, 5), numpy.random.rand(5)])
#     def test_grad_col(self):
#         utt.verify_grad(Mul, [numpy.random.rand(3, 5), numpy.random.rand(3, 1)])

#     def test_wrong_shapes(self):
#         a = as_tensor_variable(numpy.ones(3))
#         b = as_tensor_variable(numpy.ones(4))
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
#         utt.seed_rng()
#     def test_grad_e(self):
#         utt.verify_grad(Div, [numpy.random.rand(3), numpy.ones(3)])
#         utt.verify_grad(Div, [numpy.random.rand(3,5), numpy.random.rand(3,5)+0.1])
#         utt.verify_grad(Div, [numpy.ones(()), numpy.ones(())])

#     def test_grad_sl(self):
#         utt.verify_grad(Div, [numpy.ones((3, 5)), numpy.ones((1, 1))])
#         utt.verify_grad(Div, [numpy.random.rand(3), numpy.ones((1, ))])
#         utt.verify_grad(Div, [numpy.random.rand(3,5), numpy.random.rand(1,1)])

# class T_log2(unittest.TestCase):
#     def setUp(self):
#         utt.seed_rng()
#     def test0(self):
#         utt.verify_grad(Log2, [numpy.random.rand(3,1)+0.0001])

# class T_log(unittest.TestCase):
#     def setUp(self):
#         utt.seed_rng()
#     def test0(self):
#         utt.verify_grad(Log, [numpy.random.rand(3,1)+0.0001])
#     def test1(self):
#         a = as_tensor_variable(numpy.ones(2))
#         b = as_tensor_variable(numpy.ones(2))
#         aa = numpy.asarray([0.5, 4.0])
#         bb = numpy.asarray([0.5, 2.0])
#         check_eq2(self, [a], log(a), [aa], numpy.log(numpy.asarray(aa)))

# class T_pow(unittest.TestCase):
#     def setUp(self):
#         utt.seed_rng()
#     def test_elemwise(self):
#         utt.verify_grad(Div, [numpy.random.rand(3,4), numpy.random.rand(3,4)+0.1])
#         utt.verify_grad(Pow, [numpy.random.rand(3,4), numpy.random.rand(3,4)])
#     def test_scalar_l(self):
#         utt.verify_grad(Pow, [numpy.asarray([3.0]), numpy.random.rand(3)])
#     def test_scalar_r(self):
#         utt.verify_grad(Pow, [numpy.random.rand(3), numpy.asarray([3.0])])
#     def test_row(self):
#         utt.verify_grad(Pow, [numpy.random.rand(3, 5), numpy.random.rand(1, 5)])
#     def test_col(self):
#         utt.verify_grad(Pow, [numpy.random.rand(3, 5), numpy.random.rand(3, 1)])

class test_matinv(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def mat_reciprocal(self,dim):
        # symbolic program
        # broadcastable=[False,False] means that the shape of matrix is two dimensional,
        # and none of the dimensions are constrained to have length 1.
        # Note that TensorType's constructor does not actually allocate any memory.
        # TODO: Make TensorType syntax more explicit, and maybe give shape or number of dimensions.

        utt.seed_rng()

        a, b = matrices('ab')
        ab = a*b
        # Here, as_tensor_variable actually uses the data allocated by numpy.
        diff = ab - as_tensor_variable(numpy.ones((dim,dim)))
        # Sum of squared errors
        ssdiff = sum((diff**2.0))

        g_b = grad(ssdiff, b)

        # compilation to function
        # [a,b] are the inputs, [ssdiff,g_b] are the outputs
        fn = inplace_func([a,b], [ssdiff,g_b])

        # use the function
        x = numpy.random.rand(dim,dim)+0.1      # Initialized s.t. x is not too tiny
        w = numpy.random.rand(dim,dim)
        for i in xrange(100):
            ssd, gw = fn(x,w)
            #print ssd, x*w, x, w
            if i == 0:
                ssd0 = ssd
            w -= 0.4 * gw

        return ssd0, ssd

    def test_reciprocal(self):
        """Matrix reciprocal by gradient descent"""
        ssd0,ssd = self.mat_reciprocal(3)

        utt.seed_rng()
        # hand-coded numpy implementation for verification
        x = numpy.random.rand(3,3)+0.1
        w = numpy.random.rand(3,3)
        myssd0 = numpy.sum((x*w - numpy.ones((3,3)))**2.0)
        # we want at least a test that is not too fast. So we make one here.
        for i in xrange(100):
            gw = 2*(x*w - numpy.ones((3,3)))*x  # derivative of dMSE/dw
            myssd = numpy.sum((x*w - numpy.ones((3,3)))**2)
            w -= 0.4 * gw
        self.failUnlessAlmostEqual(ssd0, myssd0)
        self.failUnlessAlmostEqual(ssd, myssd)

class t_dot(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
    @staticmethod
    def rand(*args):
        return numpy.random.rand(*args)

    def cmp_dot(self,x,y):
        #x, y are matrices or numbers
        def spec(x):
            x = numpy.asarray(x)
            return type(x), x.dtype, x.shape
        nz = numpy.dot(x,y)
        tz = eval_outputs([dot(as_tensor_variable(x), as_tensor_variable(y))])
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
        # constant folding will complain to _logger that things are not aligned
        # this is normal, testers are not interested in seeing that output.
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.getEffectiveLevel()
        _logger.setLevel(logging.CRITICAL)
        try:
            try:
                tz = eval_outputs([z])
                assert False # should have raised exception
            except ValueError, e:
                self.failUnless(
                    e[0].split()[1:4] == ['are', 'not', 'aligned'] or # reported by numpy
                    e[0].split()[0:2] == ['Shape', 'mismatch:'], e) # reported by blas return self.fail()
        finally:
            _logger.setLevel(oldlevel)

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
        #utt.verify_grad(dot, [self.rand(2,3,4), self.rand(4)])
        utt.verify_grad(dot, [self.rand(2,3), self.rand(3,2)])
        utt.verify_grad(dot, [self.rand(2), self.rand(2,3)])
        utt.verify_grad(dot, [self.rand(3,2), self.rand(2)])
        utt.verify_grad(dot, [self.rand(2), self.rand(2)])
        #utt.verify_grad(dot, [self.rand(), self.rand(2)])
        #utt.verify_grad(dot, [self.rand(), self.rand(2,5)])

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
        t = as_tensor_variable(s)
        self.failUnless(t.owner.op is tensor_from_scalar)
        self.failUnless(t.type.broadcastable == (), t.type.broadcastable)
        self.failUnless(t.type.ndim == 0, t.type.ndim)
        self.failUnless(t.type.dtype == s.type.dtype)

        v = eval_outputs([t])

        self.failUnless(v == 56, v)
        self.failUnless(isinstance(v, numpy.ndarray))
        self.failUnless(v.shape == (), v.shape)

        g = grad(t, s)
        self.failUnless(eval_outputs([g])==1)

class T_scalarfromtensor(unittest.TestCase):
    def test0(self):
        tt = constant(56)#scal.constant(56)
        ss = scalar_from_tensor(tt)
        self.failUnless(ss.owner.op is scalar_from_tensor)
        self.failUnless(ss.type.dtype == tt.type.dtype)

        v = eval_outputs([ss])

        self.failUnless(v == 56, v)
        self.failUnless(isinstance(v, numpy.int8))
        self.failUnless(v.shape == (), v.shape)
        tt = lscalar()
        ss = scalar_from_tensor(tt)
        g = ss.owner.op.grad([tt],[ss])
        fff=function([tt],ss)
        v = fff(numpy.asarray(5))
        self.failUnless(v == 5, v)
        self.failUnless(isinstance(v, numpy.int64))
        self.failUnless(v.shape == (),v.shape)

# def _tensor(data, broadcastable=None, name=None):
#     """Return a TensorType containing given data"""
#     data = numpy.asarray(data)
#     if broadcastable is None:
#         broadcastable = [s==1 for s in data.shape]
#     elif broadcastable in [0, 1]:
#         broadcastable = [broadcastable] *  len(data.shape)
#     rval = TensorType(data.dtype, broadcastable, name)
#     rval.data = data # will raise if broadcastable was mis-specified
#     return rval



# class T_tensor(unittest.TestCase):
#     def setUp(self):
#         utt.seed_rng()
#     def test0(self): # allocate from a scalar float
#         t = _tensor(1.0)
#         self.failUnless(isinstance(t, TensorType))
#         self.failUnless(t.dtype == 'float64')
#         self.failUnless(t.broadcastable == ())
#         self.failUnless(t.role == None)
#         self.failUnless(isinstance(t.data, numpy.ndarray))
#         self.failUnless(str(t.data.dtype) == 'float64')
#         self.failUnless(t.data == 1.0)
#     def test0_int(self): # allocate from a scalar float
#         t = _tensor(1)
#         self.failUnless(isinstance(t, TensorType))
#         self.failUnless(t.dtype == 'int64' or t.dtype == 'int32')
#     def test1(self): # allocate from a vector of ints, not broadcastable
#         t = _tensor(numpy.ones(5,dtype='int32'))
#         self.failUnless(isinstance(t, TensorType))
#         self.failUnless(t.dtype == 'int32')
#         self.failUnless(t.broadcastable == (0,))
#         self.failUnless(isinstance(t.data, numpy.ndarray))
#         self.failUnless(str(t.data.dtype) == 'int32')
#     def test2(self): # allocate from a column matrix of complex with name
#         t = _tensor(numpy.ones((5,1),dtype='complex64'),name='bart')
#         self.failUnless(isinstance(t, TensorType))
#         self.failUnless(t.dtype == 'complex64')
#         self.failUnless(t.broadcastable == (0,1))
#         self.failUnless(isinstance(t.data, numpy.ndarray))
#         self.failUnless(t.name == 'bart')
#     def test2b(self): # allocate from a column matrix, not broadcastable
#         t = _tensor(numpy.ones((5,1),dtype='complex64'),broadcastable=0)
#         self.failUnless(isinstance(t, TensorType))
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
#             self.failUnless(e[0] is TensorType.filter.E_rank)
#         try:
#             t.data = numpy.ones(1)
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is TensorType.filter.E_rank)
#     def test_data_badrank1(self):
#         t = _tensor(numpy.ones((1,1),dtype='complex64'), broadcastable=1)
#         try:
#             t.data = numpy.ones((1,1,1))
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is TensorType.filter.E_rank)
#         try:
#             t.data = numpy.ones(1)
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is TensorType.filter.E_rank)
#     def test_data_badshape0(self):
#         t = _tensor(numpy.ones((1,1),dtype='complex64'), broadcastable=1)
#         try:
#             t.data = numpy.ones((1,2))
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is TensorType.filter.E_shape)
#         try:
#             t.data = numpy.ones((0,1))
#             self.fail()
#         except ValueError, e:
#             self.failUnless(e[0] is TensorType.filter.E_shape)

#     def test_cast0(self):
#         t = TensorType('float32', [0])
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


class test_grad(unittest.TestCase):
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
        """grad: Test passing a single variable param"""
        o = test_grad.O()
        a1 = o.make_node()
        self.failUnless(o.gval0 is grad(a1.outputs[0], a1.inputs[0]))

    def test_Nparam(self):
        """grad: Test passing multiple variable params"""
        o = test_grad.O()
        a1 = o.make_node()
        g0,g1 = grad(a1.outputs[0], a1.inputs)
        self.failUnless(o.gval0 is g0)
        self.failUnless(o.gval1 is g1)

    def test_1None_rval(self):
        """grad: Test returning a single zero value from grad"""
        o = test_grad.O()
        a1 = o.make_node()
        g = grad(a1.outputs[0], a1.outputs[1])
        self.failUnless(g.owner.op == fill)
        self.failUnless(g.owner.inputs[1].data == 0)
        try:
            grad(a1.outputs[0], 'wtf')
        except AttributeError, e:
            return
        self.fail()

    def test_NNone_rval(self):
        """grad: Test returning some zero value from grad"""
        o = test_grad.O()
        a1 = o.make_node()
        g0,g1,g2 = grad(a1.outputs[0], a1.inputs + [scalar('z')])
        self.failUnless(o.gval0 is g0)
        self.failUnless(o.gval1 is g1)
        self.failUnless(g2.owner.op == fill)
        self.failUnless(g2.owner.inputs[1].data == 0)

    def test_zero_gradient_shape(self):
        """Ensure that a zero gradient has the proper shape."""
        x = dmatrix()
        f = theano.function([x], grad(dscalar(), x))
        a = numpy.ones((3, 7))
        self.failUnless((f(a) == 0).all())  # Zero gradient.
        self.failUnless(a.shape == f(a).shape)  # With proper shape.

class T_op_cache(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
    def test0(self):
        """trigger bug in ticket #162"""
        lr = constant(0.011)
        v = matrix()
        v.name = 'v'
        gv = fill(v/v, 1.0)/v - (fill(v/v, 1.0) * v) / (v*v)
        fn_py = inplace_func([v], gv)
        fn_c_or_py = inplace_func([v], gv)

        a = numpy.random.rand(5,2)
        self.failUnless(numpy.all(fn_py(a) == fn_c_or_py(a)))


def test_reshape():

    a = dvector()
    b = dmatrix()

    c = reshape(a, [2,3])

    #basic
    f = inplace_func([a], c)
    assert numpy.all(f(numpy.asarray([0,1,2,3,4,5])) == numpy.asarray([[0,1,2], [3,4,5]]))

    #test that it works without inplace operations
    a_val = numpy.asarray([0,1,2,3,4,5])
    a_val_copy = numpy.asarray([0,1,2,3,4,5])
    b_val = numpy.asarray([[0,1,2],[3,4,5]])

    f_sub = inplace_func([a,b], c-b)
    assert numpy.all(f_sub(a_val, b_val) == 0.0)
    assert numpy.all(a_val == a_val_copy)

    #test that it works with inplace operations
    a_val = theano._asarray([0,1,2,3,4,5], dtype='float64')
    a_val_copy = theano._asarray([0,1,2,3,4,5], dtype='float64')
    b_val = theano._asarray([[0,1,2],[3,4,5]], dtype='float64')

    f_sub = inplace_func([a,b], c-b)
    assert numpy.all(f_sub(a_val, b_val) == 0.0)
    assert numpy.all(a_val == a_val_copy)

    # verify gradient
    def just_vals(v):
        return Reshape(2)(v, theano._asarray([2,3], dtype='int32'))
    utt.verify_grad(just_vals, [a_val])


def test_make_column_matrix_broadcastable():
    # The goal of the operation made by `b` is to ensure the second dimension
    # of the column matrix is broadcastable.
    a = dmatrix()
    b = a.reshape((a.shape[0], )).dimshuffle(0, 'x')
    f = function([a], b)
    assert (f(numpy.zeros((3, 1))) + numpy.ones(2) == numpy.ones((3, 2))).all()

def test_flatten_outdimNone():
    """ Flatten always returns a copy of the array. There is no danger with in-place
    operations and thus no need to test it."""

    a = dmatrix()
    c = flatten(a)
    f = inplace_func([a], c)
    a_val = theano._asarray([[0,1,2],[3,4,5]], dtype='float64')
    c_val = theano._asarray([0,1,2,3,4,5], dtype='float64')
    assert numpy.all(f(a_val)==c_val)
    f = inplace_func([a], c)
    assert numpy.all(f(a_val)==c_val)

    utt.verify_grad(Flatten(), [a_val])

def test_flatten_scalar():
    a = dscalar()
    c = flatten(a)
    f = inplace_func([a], c)
    a_val = theano._asarray(3.0, dtype='float64')
    c_val = theano._asarray([3.0], dtype='float64')
    assert numpy.all(f(a_val)==c_val)
    f = inplace_func([a], c)
    assert numpy.all(f(a_val)==c_val)

    #utt.verify_grad(Flatten(), [a_val]) #TODO: fix verify_grd to work on scalars

def test_flatten_outdim1():
    a = dmatrix()
    c = flatten(a, 1)
    f = inplace_func([a], c)
    a_val = theano._asarray([[0,1,2],[3,4,5]], dtype='float64')
    c_val = theano._asarray([0,1,2,3,4,5], dtype='float64')
    assert numpy.all(f(a_val)==c_val)
    f = inplace_func([a], c)
    assert numpy.all(f(a_val)==c_val)

    utt.verify_grad(Flatten(1), [a_val])

def test_flatten_outdim2():
    a = dmatrix()
    c = flatten(a, 2)
    f = inplace_func([a], c)
    a_val = theano._asarray([[0,1,2],[3,4,5]], dtype='float64')
    assert numpy.all(f(a_val)==a_val)
    f = inplace_func([a], c)
    assert numpy.all(f(a_val)==a_val)

    utt.verify_grad(Flatten(2), [a_val])

def test_flatten_outdim2_of_3():
    a = TensorType('float64', (False, False, False))()
    c = flatten(a, 2)
    f = inplace_func([a], c)
    a_val = theano._asarray([[[0,1],[2,3]], [[4,5],[6,7]]], dtype='float64')
    c_val = theano._asarray([[0,1,2,3], [4,5,6,7]], dtype='float64')
    assert numpy.all(f(a_val)==c_val)
    f = inplace_func([a], c)
    assert numpy.all(f(a_val)==c_val)

    utt.verify_grad(Flatten(2), [a_val])

def test_flatten_outdim_invalid():
    a = dmatrix()
    try:
        c = flatten(a, 3)
        assert False
    except ValueError:
        pass
    try:
        c = flatten(a, 0)
        assert False
    except ValueError:
        pass

# TODO: write test case for Tile Op
def test_tile():
    print >> sys.stderr, "WARNING: No testcase for Tile"
    pass 


class TestARange(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_Op_integers(self):
        """Test behaviour of ARange Op on integer inputs"""
        start, stop, step = iscalars('start', 'stop', 'step')
        out = ARange(start.type.dtype)(start, stop, step)
        f = function([start, stop, step], out)

        assert numpy.all(f(0,5,1) == numpy.arange(0,5,1))
        assert numpy.all(f(2,11,4) == numpy.arange(2,11,4))
        assert numpy.all(f(-5,1,1) == numpy.arange(-5,1,1))
        assert numpy.all(f(10,2,-2) == numpy.arange(10,2,-2))
        assert numpy.all(f(10,2,2) == numpy.arange(10,2,2))
        assert numpy.all(f(0,0,1) == numpy.arange(0,0,1))

    def test_integers(self):
        """Test arange constructor, on integer outputs"""
        start, stop, step = iscalars('start', 'stop', 'step')
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        assert out.dtype == start.type.dtype
        assert numpy.all(f(0,5,1) == numpy.arange(0,5,1))
        assert numpy.all(f(2,11,4) == numpy.arange(2,11,4))
        assert numpy.all(f(-5,1,1) == numpy.arange(-5,1,1))
        assert numpy.all(f(10,2,-2) == numpy.arange(10,2,-2))
        assert numpy.all(f(10,2,2) == numpy.arange(10,2,2))
        assert numpy.all(f(0,0,1) == numpy.arange(0,0,1))

    def test_float32(self):
        """Test arange constructor, on integer outputs"""
        start, stop, step = fscalars('start', 'stop', 'step')
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        assert out.dtype == start.type.dtype
        assert numpy.all(f(0,5,1) == numpy.arange(0,5,1, dtype=start.type.dtype))
        assert numpy.all(f(2,11,4) == numpy.arange(2,11,4, dtype=start.type.dtype))
        assert numpy.all(f(-5,1.1,1.2) == numpy.arange(-5,1.1,1.2, dtype=start.type.dtype))
        assert numpy.all(f(1.3,2,-2.1) == numpy.arange(1.3,2,-2.1, dtype=start.type.dtype))
        assert numpy.all(f(10,2,2) == numpy.arange(10,2,2, dtype=start.type.dtype))

    def test_float64(self):
        """Test arange constructor, on integer outputs"""
        start, stop, step = dscalars('start', 'stop', 'step')
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        assert out.dtype == start.type.dtype
        assert numpy.all(f(0,5,1) == numpy.arange(0,5,1, dtype=start.type.dtype))
        assert numpy.all(f(2,11,4) == numpy.arange(2,11,4, dtype=start.type.dtype))
        assert numpy.all(f(-5,1.1,1.2) == numpy.arange(-5,1.1,1.2, dtype=start.type.dtype))
        assert numpy.all(f(1.3,2,-2.1) == numpy.arange(1.3,2,-2.1, dtype=start.type.dtype))
        assert numpy.all(f(10,2,2) == numpy.arange(10,2,2, dtype=start.type.dtype))

    def test_default_step(self):
        """Test that arange constructor uses the correct default step"""
        start, stop = iscalars('start', 'stop')
        out = arange(start, stop)
        f = function([start, stop], out)

        assert out.dtype == start.type.dtype
        assert numpy.all(f(0,5) == numpy.arange(0,5))
        assert numpy.all(f(-5,1) == numpy.arange(-5,1))
        assert numpy.all(f(0,0) == numpy.arange(0,0))

        dstart, dstop = dscalars('start', 'stop')
        dout = arange(dstart, dstop)
        df = function([dstart, dstop], dout)

        assert dout.dtype == dstart.type.dtype
        print df(0.2, 5.3)
        print numpy.arange(0.2, 5.3)
        assert numpy.all(df(0.2, 5.3) == numpy.arange(0.2, 5.3))
        assert numpy.all(df(0.8, 5.3) == numpy.arange(0.8, 5.3))
        assert numpy.all(df(-0.7, 5.3) == numpy.arange(-0.7, 5.3))

    def test_default_start(self):
        """Test that arange constructor uses the correct default start"""
        stop = iscalar('stop')
        out = arange(stop)
        f = function([stop], out)

        assert out.dtype == stop.type.dtype
        assert numpy.all(f(8) == numpy.arange(8))
        assert numpy.all(f(-2) == numpy.arange(-2))

        fstop = fscalar('stop')
        fout = arange(fstop)
        ff = function([fstop], fout)

        assert fout.dtype == fstop.type.dtype
        assert numpy.all(ff(0.2) == numpy.arange(0.2))
        assert numpy.all(ff(-0.7) == numpy.arange(-0.7))
        assert numpy.all(ff(8.5) == numpy.arange(8.5))

    def test_upcast(self):
        """Test that arange compute output type adequately"""
        assert arange(iscalar()).dtype == iscalar().dtype
        assert arange(fscalar()).dtype == fscalar().dtype
        assert arange(dscalar()).dtype == dscalar().dtype

        # int32 + float32 -> float64
        assert arange(iscalar(), fscalar()).dtype == dscalar().dtype
        assert arange(iscalar(), dscalar()).dtype == dscalar().dtype
        assert arange(fscalar(), dscalar()).dtype == dscalar().dtype

        assert arange(iscalar(), fscalar(), dscalar()).dtype == dscalar().dtype

    def test_dtype_cache(self):
        """Checks that the same Op is returned on repeated calls to arange
        using the same dtype, but not for different dtypes."""

        start, stop, step = iscalars('start', 'stop', 'step')
        out1 = arange(start, stop, step)
        out2 = arange(start, stop, step, dtype=start.type.dtype)
        out3 = arange(start, stop, 2., dtype=start.type.dtype)
        out4 = arange(start, stop, 2.)

        assert out1.owner.op is out2.owner.op
        assert out2.owner.op is out3.owner.op
        assert out3.owner.op is not out4.owner.op

    def test_infer_shape(self):
        start, stop, step = iscalars('start', 'stop', 'step')
        out = arange(start, stop, step)
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        mode = compile.mode.get_mode(mode).excluding('fusion')
        f = function([start, stop, step], out.shape, mode=mode)
        assert len(f.maker.env.toposort())==7
#7 [Elemwise{sub,no_inplace}(stop, start), Elemwise{Cast{float64}}(Elemwise{sub,no_inplace}.0), Elemwise{TrueDiv{output_types_preference=transfer_type{0}}}[(0, 0)](Elemwise{Cast{float64}}.0, step), Elemwise{Ceil{output_types_preference=transfer_type{0}}}[(0, 0)](Elemwise{TrueDiv{output_types_preference=transfer_type{0}}}[(0, 0)].0), Elemwise{Cast{int64}}(Elemwise{Ceil{output_types_preference=transfer_type{0}}}[(0, 0)].0), Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)](Elemwise{Cast{int64}}.0, 0), MakeVector(Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)].0)]

        assert out.dtype == start.type.dtype
        assert numpy.all(f(0,5,1) == len(numpy.arange(0,5,1)))
        assert numpy.all(f(2,11,4) == len(numpy.arange(2,11,4)))
        assert numpy.all(f(-5,1,1) == len(numpy.arange(-5,1,1)))
        assert numpy.all(f(10,2,-2) == len(numpy.arange(10,2,-2)))
        assert numpy.all(f(10,2,2) == len(numpy.arange(10,2,2)))
        assert numpy.all(f(0,0,1) == len(numpy.arange(0,0,1)))

        out = arange(start, stop, 1)
        f = function([start, stop], out.shape, mode=mode)
        assert len(f.maker.env.toposort())==4
#4 [Elemwise{sub,no_inplace}(stop, start), Elemwise{Cast{int64}}(Elemwise{sub,no_inplace}.0), Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)](Elemwise{Cast{int64}}.0, 0), MakeVector(Elemwise{Maximum{output_types_preference=transfer_type{0}}}[(0, 0)].0)]
        assert out.dtype == start.type.dtype
        assert numpy.all(f(0,5) == len(numpy.arange(0,5)))
        assert numpy.all(f(2,11) == len(numpy.arange(2,11)))
        assert numpy.all(f(-5,1) == len(numpy.arange(-5,1)))
        assert numpy.all(f(10,2) == len(numpy.arange(10,2)))
        assert numpy.all(f(10,2) == len(numpy.arange(10,2)))
        assert numpy.all(f(0,0) == len(numpy.arange(0,0)))

        out = arange(0, stop, 1)
        f = function([stop], out.shape, mode=mode)
        assert len(f.maker.env.toposort())==2
        #[Elemwise{Cast{int64}}(stop), MakeVector(Elemwise{Cast{int64}}.0)]
        
        assert out.dtype == start.type.dtype
        assert numpy.all(f(5) == len(numpy.arange(0,5)))
        assert numpy.all(f(11) == len(numpy.arange(0,11)))
        assert numpy.all(f(1) == len(numpy.arange(0,1)))
        assert numpy.all(f(2) == len(numpy.arange(0,2)))
        assert numpy.all(f(2) == len(numpy.arange(0,2)))
        assert numpy.all(f(0) == len(numpy.arange(0,0)))

class TestInversePermutation(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_dim1(self):
        """Test the inversion of one permutation (int vector)"""
        p = ivector()
        inv = inverse_permutation(p)
        f_inverse = function([p], inv)

        # Generate a random permutation
        rng = numpy.random.RandomState(utt.fetch_seed())
        p_val = rng.permutation(10)
        inv_val = f_inverse(p_val)

        # Check that the inverse of the inverse is the original permutation
        assert numpy.all(f_inverse(inv_val) == p_val)
        # Check that permutation(inverse) == inverse(permutation) = identity
        assert numpy.all(p_val[inv_val] == numpy.arange(10))
        assert numpy.all(inv_val[p_val] == numpy.arange(10))

    def test_dim2(self):
        """Test the inversion of several permutations at a time"""
        # Each row of p is a different permutation to inverse
        p = imatrix()
        inv = inverse_permutation(p)
        f_inverse = function([p], inv)

        rng = numpy.random.RandomState(utt.fetch_seed())
        # Generate 10 random permutations
        p_val = numpy.asarray([rng.permutation(10) for i in range(7)])
        inv_val = f_inverse(p_val)

        # Check that the inverse of the inverse is the original permutation list
        assert numpy.all(f_inverse(inv_val) == p_val)
        # Check that, for each permutation,
        # permutation(inverse) == inverse(permutation) = identity
        for p_row, i_row in zip(p_val, inv_val):
            assert numpy.all(p_row[i_row] == numpy.arange(10))
            assert numpy.all(i_row[p_row] == numpy.arange(10))


class TestPermuteRowElements(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_1_1(self):
        """Test PermuteRowElements(vector, vector)"""
        input = dvector()
        p = ivector()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = numpy.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(5,))
        p_val = rng.permutation(5)
        out_val = permute(input_val, p_val)

        # Should be equivalent to advanced indexing
        out_bis = input_val[p_val]
        assert numpy.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            """Auxiliary op defined to get rid of gradient wrt p_val"""
            return permute_row_elements(s_input, p_val)
        utt.verify_grad(permute_fixed, [input_val])

    def test_2_1(self):
        """Test broadcasting in PermuteRowElements(matrix, vector)"""
        input = dmatrix()
        p = ivector()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = numpy.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(3,5))
        p_val = rng.permutation(5)
        out_val = permute(input_val, p_val)

        # The same permutation should be applied to every row of the input matrix.
        out_bis = numpy.asarray([row[p_val] for row in input_val])
        assert numpy.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            """Auxiliary op defined to get rid of gradient wrt p_val"""
            return permute_row_elements(s_input, p_val)
        utt.verify_grad(permute_fixed, [input_val])

    def test_2_2(self):
        """Test PermuteRowElements(matrix, matrix)"""
        input = dmatrix()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = numpy.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(3,5))
        p_val = numpy.asarray([rng.permutation(5) for i in range(3)])
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to the corresponding
        # row of input
        out_bis = numpy.asarray([i_row[p_row] for i_row, p_row in zip(input_val, p_val)])
        assert numpy.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            """Auxiliary op defined to get rid of gradient wrt p_val"""
            return permute_row_elements(s_input, p_val)
        utt.verify_grad(permute_fixed, [input_val])

    def test_1_2(self):
        """Test PermuteRowElements(vector, matrix)
        Different permutations will be applied to the same input vector"""
        input = dvector()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = numpy.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(5,))
        p_val = numpy.asarray([rng.permutation(5) for i in range(3)])
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to the input vector
        out_bis = numpy.asarray([input_val[p_row] for p_row in p_val])
        assert numpy.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            """Auxiliary op defined to get rid of gradient wrt p_val"""
            return permute_row_elements(s_input, p_val)
        utt.verify_grad(permute_fixed, [input_val])

    def test_3b_2(self):
        """Test permute_row_elements on a more complex broadcasting pattern:
        input.type.broadcastable = (False, True, False),
        p.type.broadcastable = (False, False)."""

        input = TensorType('float64', (False, True, False))()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = numpy.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(4,1,5))
        p_val = numpy.asarray([rng.permutation(5) for i in range(3)])
        out_val = permute(input_val, p_val)

        # Each row of p contains a permutation to apply to each row
        # of the input tensor
        out_bis = numpy.asarray([[in_mat[0,p_row] for p_row in p_val] for in_mat in input_val])
        assert numpy.all(out_val == out_bis)

        # Verify gradient
        def permute_fixed(s_input):
            """Auxiliary op defined to get rid of gradient wrt p_val"""
            return permute_row_elements(s_input, p_val)
        utt.verify_grad(permute_fixed, [input_val])


class test_tensordot(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test0(self):

        # test vector-vector
        avec = dvector()
        bvec = dvector()
        axes = ((0,),(0,))
        c = tensordot(axes)(avec, bvec)
        f1 = inplace_func([avec,bvec],c)
        aval = numpy.random.rand(5);
        bval = numpy.random.rand(5);
        self.failUnless(numpy.tensordot(aval,bval,axes) == \
                        f1(aval,bval))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # test matrix-vector
        bmat = dmatrix()
        axes = ((0,),(1,))
        c = tensordot(axes)(avec, bmat)
        f2 = inplace_func([avec,bmat],c)
        aval = numpy.random.rand(5);
        bval = numpy.random.rand(8,5);
        self.failUnless(numpy.all(numpy.tensordot(aval,bval,axes) == \
                                  f2(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # test matrix-matrix
        amat = dmatrix()
        axes = ((1,),(0,))
        c = tensordot(axes)(amat, bmat)
        f3 = inplace_func([amat,bmat],c)
        aval = numpy.random.rand(4,7);
        bval = numpy.random.rand(7,9);
        self.failUnless(numpy.all(numpy.tensordot(aval,bval,axes) == \
                                  f3(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # test ndarray-matrix, sum over one dim of matrix
        atens = TensorType('float64', broadcastable=(False,)*4)()
        axes = ((2,),(1,))
        c = tensordot(axes)(atens, bmat)
        f4 = inplace_func([atens,bmat],c)
        aval = numpy.random.rand(1,2,3,4);
        bval = numpy.random.rand(2,3);
        self.failUnless(numpy.all(numpy.tensordot(aval,bval,axes) == \
                                  f4(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # test ndarray-ndarray
        atens = TensorType('float64', broadcastable=(False,)*4)()
        btens = TensorType('float64', broadcastable=(False,)*3)()
        axes = ((1,3),(0,2))
        c = tensordot(axes)(atens, btens)
        f5 = inplace_func([atens,btens],c)
        aval = numpy.random.rand(4,3,5,2);
        bval = numpy.random.rand(3,4,2);
        self.failUnless(numpy.all(numpy.tensordot(aval,bval,axes) == \
                                  f5(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])
        
        axes = (axes[1],axes[0])
        c = tensordot(axes)(btens, atens)
        f6 = inplace_func([btens,atens],c)
        self.failUnless(numpy.all(numpy.tensordot(bval,aval,axes) == \
                                  f6(bval,aval)))
        utt.verify_grad(TensorDot(axes), [bval,aval])

def test_smallest_stack():
    sx, sy = dscalar(), dscalar()

    rval = inplace_func([sx,sy], stack(sx,sy))(-4.0, -2.0)
    assert type(rval) == numpy.ndarray
    assert [-4, -2] == list(rval)


def test_smallest():
    x = dvector()
    y = dvector()
    z = dvector()
    f1 = inplace_func([x], smallest(x))
    assert numpy.all([1,2,3] == f1([1,2,3]))
    f3 = inplace_func([x,y,z], smallest(x,y,z))
    assert numpy.all([1,2,3] == f3([1,3,9], [7,7,7], [8,2,3]))

    sx, sy = dscalar(), dscalar()

    assert -4 == inplace_func([sx,sy], smallest(sx,sy))(-4.0, -2.0)

def test_reshape_member_fn():
    x = dmatrix()
    y = x.reshape((4,5,6))
    assert y.owner.op == Reshape(3)

def test_var():
    a = Tensor(dtype='float64', broadcastable=[False,False,False])()
    f = function([a], var(a))

    a_val = numpy.arange(60).reshape(3,4,5)
    print numpy.var(a_val)
    print f(a_val)
    assert numpy.allclose(numpy.var(a_val), f(a_val))

    f = function([a], var(a, axis=0))
    assert numpy.allclose(numpy.var(a_val, axis=0), f(a_val))

    f = function([a], var(a, axis=1))
    assert numpy.allclose(numpy.var(a_val, axis=1), f(a_val))

    f = function([a], var(a, axis=2))
    assert numpy.allclose(numpy.var(a_val, axis=2), f(a_val))

def test_sum_overflow():
    """Ensure that overflow errors are a little bit harder to get"""
    a = Tensor(dtype='int8', broadcastable=[False])()
    f = function([a], sum(a))
    assert f([1]*300) == 300

def test_default():
    x, y = scalars('xy')
    z = default(x, y)
    f = function([x, y], z)
    assert f(1, 2) == 1
    assert f(None, 2) == 2
    assert f(1, None) == 1

def test_default_state():
    x, y = scalars('xy')
    print config.floatX
    print x.type
    print y.type
    z = default(x, 3.8)
    new_x = y + z
    f = function([y, compile.In(x, update = new_x, value = 12.0)], new_x)
    assert f(3) == 15
    f['x'] = None
    assert numpy.allclose(f(1), 4.8)
    assert numpy.allclose(f(2.2), 7)

def test_autocast():
    orig_autocast = autocast_float.dtypes

    # test that autocast_float_as sets the autocast dtype correctly
    try: #ghetto 2.4 version of with
        ac = autocast_float_as('float32')
        ac.__enter__()
        assert autocast_float.dtypes == ('float32',)
    finally:
        ac.__exit__()
    assert autocast_float.dtypes == orig_autocast
    try: #ghetto 2.4 version of with
        ac = autocast_float_as('float64')
        ac.__enter__()
        assert autocast_float.dtypes == ('float64',)
    finally:
        ac.__exit__()
    assert autocast_float.dtypes == orig_autocast
    # test that we can set it back to something, and nest it
    try: #ghetto 2.4 version of with
        ac = autocast_float_as('float32')
        ac.__enter__()
        assert autocast_float.dtypes == ('float32',)
        try: #ghetto 2.4 version of with
            ac2 = autocast_float_as('float64')
            ac2.__enter__()
            assert autocast_float.dtypes == ('float64',)
        finally:
            ac2.__exit__()
        assert autocast_float.dtypes == ('float32',)
    finally:
        ac.__exit__()
    assert autocast_float.dtypes == orig_autocast

    # test that the autocasting dtype is used correctly in expression-building
    try: #ghetto 2.4 version of with
        ac = autocast_float_as('float32')
        ac.__enter__()
        assert (dvector()+ 1.1).dtype == 'float64'
        assert (fvector()+ 1.1).dtype == 'float32'
        assert (fvector()+ theano._asarray(1.1,dtype='float64')).dtype == 'float64'
        assert (fvector()+ theano._asarray(1.1,dtype='float32')).dtype == 'float32'

        assert (dvector()+ 1).dtype == 'float64'
        assert (fvector()+ 1).dtype == 'float32'
    finally:
        ac.__exit__()

    # test that the autocasting dtype is used correctly in expression-building
    try: #ghetto 2.4 version of with
        ac = autocast_float_as('float64')
        ac.__enter__()
        assert (dvector()+ 1.1).dtype == 'float64'
        assert (fvector()+ 1.1).dtype == 'float64'
        assert (fvector()+ 1.0).dtype == 'float64'
        assert (fvector()+ theano._asarray(1.1,dtype='float64')).dtype == 'float64'
        assert (fvector()+ theano._asarray(1.1,dtype='float32')).dtype == 'float32'

        assert (dvector()+ 1).dtype == 'float64'
        assert (fvector()+ 1).dtype == 'float32'
    finally:
        ac.__exit__()

    # test that the autocasting dtype is used correctly in expression-building
    try: #ghetto 2.4 version of with
        ac = autocast_float_as('float32', 'float64')
        ac.__enter__()
        assert (dvector()+ 1.1).dtype == 'float64'
        assert (fvector()+ 1.1).dtype == theano.config.floatX
        assert (fvector()+ 1.0).dtype == 'float32'
        try: #ghetto 2.4 version of with
            ac2 = autocast_float_as('float64')
            ac2.__enter__()
            assert (fvector()+ 1.0).dtype == 'float64'
        finally:
            ac2.__exit__()
    finally:
        ac.__exit__()

if __name__ == '__main__':
    if 1:
        unittest.main()
    else:
        testcase =  T_Join_and_Split

        suite = unittest.TestLoader()
        suite = suite.loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(suite)


