import operator
import StringIO
import sys
import unittest

import numpy
from numpy.testing import dec

from theano.tensor import *
from theano.tensor import basic as tensor # for hidden symbols
from theano.tensor import inplace

from copy import copy
from theano import compile, config
from theano import gof
from theano.gof.python25 import any, all

from theano.compile.mode import get_default_mode
from theano import function
from theano.tests import unittest_tools as utt


imported_scipy_special = False
mode_no_scipy = get_default_mode()
try:
    import scipy.special
    imported_scipy_special = True
except ImportError:
    if config.mode=="FAST_COMPILE":
        mode_no_scipy = "FAST_RUN"

### seed random number generator so that unittests are deterministic ###
utt.seed_rng()

def inplace_func(inputs, outputs, mode=get_default_mode(),
        allow_input_downcast=False):
    return function(inputs, outputs,
            mode=mode,
            allow_input_downcast=allow_input_downcast,
            accept_inplace=True)

def eval_outputs(outputs):
    variables = inplace_func([], outputs)()
    if isinstance(variables,(tuple,list)) and len(variables) == 1:
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

def makeTester(name, op, expected, checks = {}, good = {}, bad_build = {}, bad_runtime = {}, grad = {}, mode = None, grad_rtol=None, eps = 1e-10):
    if grad is True:
        grad = good

    _op, _expected, _checks, _good, _bad_build, _bad_runtime, _grad, _mode, _grad_rtol, _eps = op, expected, checks, good, bad_build, bad_runtime, grad, mode, grad_rtol, eps

    class Checker(unittest.TestCase):

        op = _op
        expected = staticmethod(_expected)
        checks = _checks
        good = _good
        bad_build = _bad_build
        bad_runtime = _bad_runtime
        grad = _grad
        mode = _mode

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
                    f = inplace_func(inputrs, node.outputs, mode = mode)
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while trying to make a Function" \
                        % (self.op, testname)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback
                if isinstance(self.expected,dict) and testname in self.expected:
                    expecteds = self.expected[testname]
                    #with numpy version, when we print a number and read it back, we don't get exactly the same result
                    #So we accept rounding error in that case.
                    eps = 5e-9
                else:
                    expecteds = self.expected(*inputs)
                    eps = 1e-10

                if any([i.dtype=='float32' for i in inputs]):
                    eps=8e-6#1e-6
                eps = numpy.max([eps,_eps])

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
                            numpy.any(numpy.abs(variable - expected) > eps):
                        self.fail("Test %s::%s: Output %s gave the wrong value. With inputs %s, expected %s, got %s. numpy.allclose return %s %s"
                                  % (self.op, testname, i, inputs, expected, variable, numpy.allclose(variable,expected,atol=eps), numpy.allclose(variable,expected)))

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
                    utt.verify_grad(self.op, inputs, mode=self.mode, rel_tol=_grad_rtol)
                except:
                    type, exc_value, traceback = sys.exc_info()
                    err_msg = "Test %s::%s: Error occurred while computing the gradient on the following inputs: %s" \
                        % (self.op, testname, inputs)
                    exc_value.args = exc_value.args + (err_msg, )
                    raise type, exc_value, traceback

    Checker.__name__ = name
    return Checker


rand = lambda *shape: 2 * numpy.asarray(numpy.random.rand(*shape), dtype=config.floatX) - 1
randint = lambda *shape: numpy.random.random_integers(-5, 5, shape)
randcomplex = lambda *shape: numpy.complex128(2 * numpy.asarray(numpy.random.rand(*shape), dtype=config.floatX) - 1)

def randint_nonzero(*shape):
    r = numpy.random.random_integers(-5, 4, shape)
    return r + (r == 0) * 5

def rand_ranged(min, max, shape):
    return numpy.asarray(numpy.random.rand(*shape) * (max - min) + min, dtype=config.floatX)

def randint_ranged(min, max, shape):
    return numpy.random.random_integers(min, max, shape)

def randc128_ranged(min, max, shape):
    return numpy.asarray(numpy.random.rand(*shape) * (max - min) + min, dtype='complex128')

def makeBroadcastTester(op, expected, checks = {}, **kwargs):
    name = str(op) + "Tester"
    if kwargs.has_key('inplace'):
        if kwargs['inplace']:
            _expected = expected
            if not isinstance(_expected,dict):
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
                                     dtype_mixup_2 = (randint(2, 3), rand(2, 3)),
                                     complex1 = (randcomplex(2,3),randcomplex(2,3)),
                                     complex2 = (randcomplex(2,3),rand(2,3)),
                                     # Disabled as we test the case where we reuse the same output as the first inputs.
                                     # complex3 = (rand(2,3),randcomplex(2,3)),
                                     empty = (numpy.asarray([]),numpy.asarray([1])),
                                     )

_bad_build_broadcast_binary_normal = dict()#not_same_dimensions = (rand(2), rand(2, 2)))

_bad_runtime_broadcast_binary_normal = dict(bad_shapes = (rand(2, 3), rand(3, 2)),
                                            bad_row = (rand(2, 3), rand(1, 2)))

_grad_broadcast_binary_normal = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                     scalar = (rand(2, 3), rand(1, 1)),
                                     row = (rand(2, 3), rand(1, 3)),
                                     column = (rand(2, 3), rand(2, 1)),
                                     #This don't work as verify grad don't support that
                                     #empty = (numpy.asarray([]), numpy.asarray([1]))
                                     #complex1 = (randcomplex(2,3),randcomplex(2,3)),
                                     #complex2 = (randcomplex(2,3),rand(2,3)),
                                     # Disabled as we test the case where we reuse the same output as the first inputs.
                                     #complex3 = (rand(2,3),randcomplex(2,3)),
                                     )


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

_good_broadcast_div_mod_normal_float_inplace = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                         scalar = (rand(2, 3), rand(1, 1)),
                                         row = (rand(2, 3), rand(1, 3)),
                                         column = (rand(2, 3), rand(2, 1)),
                                         dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
                                         dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3)),
                                         #integers_positive = (randint_ranged(4, 10, (2, 3)), randint_ranged(1, 6, (2, 3))),
                                         #integers_known_to_fail = (numpy.array(-1), numpy.array(5))
                                         complex1 = (randcomplex(2,3),randcomplex(2,3)),
                                         complex2 = (randcomplex(2,3),rand(2,3)),
                                         #complex3 = (rand(2,3),randcomplex(2,3)),# Inplace on the first element. Must have the same type.
                                         empty1 = (numpy.asarray([]), numpy.asarray([1])),
                                         #empty2 = (numpy.asarray([0]), numpy.asarray([])),
                                         )
_good_broadcast_div_mod_normal_float = dict(empty2 = (numpy.asarray([0]), numpy.asarray([])),
                                            **_good_broadcast_div_mod_normal_float_inplace
                                            )
_grad_broadcast_div_mod_normal = dict(same_shapes = (rand(2, 3), rand(2, 3)),
                                      scalar = (rand(2, 3), rand(1, 1)),
                                      row = (rand(2, 3), rand(1, 3)),
                                      column = (rand(2, 3), rand(2, 1)),
                                      #complex1 = (randcomplex(2,3),randcomplex(2,3)),
                                      #complex2 = (randcomplex(2,3),rand(2,3)),
                                      #complex3 = (rand(2,3),randcomplex(2,3)),
                                      #dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
                                      #dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3)),
                                      #empty1 = (numpy.asarray([]), numpy.asarray([1.])),
                                      #empty2 = (numpy.asarray([0]), numpy.asarray([])),
                                   )

div_grad_rtol=None
if config.floatX=='float32':
    #We raise the relative tolerence for the grad as their is error in float32
    #This is probably caused by our way of computing the gradient error.
    div_grad_rtol=0.025
DivTester = makeBroadcastTester(op = true_div,
                                  expected = lambda x, y: x / y,
                                  good = _good_broadcast_div_mod_normal_float,
#                                               integers = (randint(2, 3), randint_nonzero(2, 3)),
#                                               dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
#                                               dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3))),
                                  grad = _grad_broadcast_div_mod_normal,
                                  grad_rtol=div_grad_rtol,
                                )
DivInplaceTester = makeBroadcastTester(op = inplace.true_div_inplace,
                                         expected = lambda x, y: x / y,
                                         good = _good_broadcast_div_mod_normal_float_inplace,
                                         grad = _grad_broadcast_div_mod_normal,
                                         grad_rtol=div_grad_rtol,
                                         inplace = True)

ModTester = makeBroadcastTester(op = mod,
                                  expected = lambda x, y: numpy.asarray(x % y, dtype=theano.scalar.basic.upcast(x.dtype, y.dtype)),
                                  good = _good_broadcast_div_mod_normal_float,
#                                               integers = (randint(2, 3), randint_nonzero(2, 3)),
#                                               dtype_mixup_1 = (rand(2, 3), randint_nonzero(2, 3)),
#                                               dtype_mixup_2 = (randint_nonzero(2, 3), rand(2, 3))),
                                  )
ModInplaceTester = makeBroadcastTester(op = inplace.mod_inplace,
                                         expected = lambda x, y: numpy.asarray(x % y, dtype=theano.scalar.basic.upcast(x.dtype, y.dtype)),
                                         good = _good_broadcast_div_mod_normal_float_inplace,
                                         inplace = True)

_good_broadcast_pow_normal_float = dict(same_shapes = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 3))),
                                        scalar = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 1))),
                                        row = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 3))),
                                        column = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 1))),
                                        dtype_mixup = (rand_ranged(-3, 3, (2, 3)), randint_ranged(-3, 3, (2, 3))),
                                        complex1 = (randcomplex(2,3),randcomplex(2,3)),
                                        complex2 = (randcomplex(2,3),rand(2,3)),
                                        #complex3 = (rand(2,3),randcomplex(2,3)), # Inplace on the first element.
                                        empty1 = (numpy.asarray([]), numpy.asarray([1])),
                                        empty2 = (numpy.asarray([0]), numpy.asarray([])),)
_grad_broadcast_pow_normal = dict(same_shapes = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 3))),
                                  scalar = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 1))),
                                  row = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (1, 3))),
                                  column = (rand_ranged(1, 5, (2, 3)), rand_ranged(-3, 3, (2, 1))),
                                  #complex1 = (randcomplex(2,3),randcomplex(2,3)),
                                  #complex2 = (randcomplex(2,3),rand(2,3)),
                                  #complex3 = (rand(2,3),randcomplex(2,3)),
                                  #empty1 = (numpy.asarray([]), numpy.asarray([1])),
                                  #empty2 = (numpy.asarray([0]), numpy.asarray([])),
                                  )
#empty2 case is not supported by numpy.
_good_broadcast_pow_normal_float_pow = copy(_good_broadcast_pow_normal_float)
del _good_broadcast_pow_normal_float_pow["empty2"]

PowTester = makeBroadcastTester(op = pow,
                                  expected = lambda x, y: x ** y,
                                  good = _good_broadcast_pow_normal_float,
                                  grad = _grad_broadcast_pow_normal)
PowInplaceTester = makeBroadcastTester(op = inplace.pow_inplace,
                                       expected = lambda x, y: x ** y,
                                       good = _good_broadcast_pow_normal_float_pow,
                                       grad = _grad_broadcast_pow_normal,
                                       inplace = True)

#Those are corner case when rounding. Their is many rounding algo.
#c round() fct and numpy round are not the same!
corner_case = numpy.asarray([-2.5, -2., -1.5, -1., -0.5, -.51, -.49, 0, 0.49, 0.5, 0.9, 1, 1.5, 2, 2.5], dtype=config.floatX)
#we remove 0 here as the grad is not always computable numerically.
corner_case_grad = numpy.asarray([-2.5, -2., -1.5, -1., -0.5, -.51, -.49, 0.49, 0.5, 0.9, 1, 1.5, 2, 2.5], dtype=config.floatX)
_good_broadcast_unary_normal_float = dict(normal = (rand_ranged(-5, 5, (2, 3)),),
                                          corner_case = (corner_case,),
                                          complex = (randcomplex(2,3),),
                                          empty = (numpy.asarray([]),))

_good_broadcast_unary_normal_float_no_empty = copy(_good_broadcast_unary_normal_float)
del _good_broadcast_unary_normal_float_no_empty['empty']
_good_broadcast_unary_normal_float_no_empty_no_complex = copy(_good_broadcast_unary_normal_float_no_empty)
del _good_broadcast_unary_normal_float_no_empty_no_complex['complex']
_good_broadcast_unary_normal_float_no_complex = copy(_good_broadcast_unary_normal_float)
del _good_broadcast_unary_normal_float_no_complex['complex']

_good_broadcast_unary_normal = dict(normal = (numpy.asarray(rand_ranged(-5, 5, (2, 3)),dtype=config.floatX),),
                                    integers = (randint_ranged(-5, 5, (2, 3)),),
                                    corner_case = (corner_case,),
                                    complex = (randcomplex(2,3),),
                                    empty = (numpy.asarray([]),))

_good_broadcast_unary_normal_no_complex = dict(normal = (numpy.asarray(rand_ranged(-5, 5, (2, 3)),dtype=config.floatX),),
                                               integers = (randint_ranged(-5, 5, (2, 3)),),
                                               corner_case = (corner_case,),
                                               #complex = (randcomplex(2,3),),
                                               empty = (numpy.asarray([]),))

_grad_broadcast_unary_normal = dict(normal = (numpy.asarray(rand_ranged(-5, 5, (2, 3)),dtype=config.floatX),),
                                    corner_case = (corner_case_grad,),
                                    #complex = (randcomplex(2,3),),
                                    #empty = (numpy.asarray([]),)
                                    )



AbsTester = makeBroadcastTester(op = tensor.abs_,
                                  expected = lambda x: abs(x),
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
_good_broadcast_unary_normal_abs = copy(_good_broadcast_unary_normal)
# Can't do inplace on Abs as the input/output are not of the same type!
del _good_broadcast_unary_normal_abs['complex']
AbsInplaceTester = makeBroadcastTester(op = inplace.abs__inplace,
                                         expected = lambda x: numpy.abs(x),
                                         good = _good_broadcast_unary_normal_abs,
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
                                good = _good_broadcast_unary_normal_no_complex,
                                grad = _grad_broadcast_unary_normal,)
SgnInplaceTester = makeBroadcastTester(op = inplace.sgn_inplace,
                                       expected = numpy.sign,
                                       good = _good_broadcast_unary_normal_no_complex,
                                       grad = _grad_broadcast_unary_normal,
                                       inplace = True)
CeilTester = makeBroadcastTester(op = ceil,
                                  expected = lambda a: numpy.asarray(numpy.ceil(a),a.dtype),
                                  good = _good_broadcast_unary_normal_no_complex,
                                  grad = _grad_broadcast_unary_normal)
CeilInplaceTester = makeBroadcastTester(op = inplace.ceil_inplace,
                                         expected = lambda a: numpy.asarray(numpy.ceil(a),a.dtype),
                                         good = _good_broadcast_unary_normal_no_complex,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

FloorTester = makeBroadcastTester(op = floor,
                                  expected = lambda a: numpy.asarray(numpy.floor(a),a.dtype),
                                  good = _good_broadcast_unary_normal_no_complex,
                                  grad = _grad_broadcast_unary_normal)
FloorInplaceTester = makeBroadcastTester(op = inplace.floor_inplace,
                                         expected = lambda a: numpy.asarray(numpy.floor(a),a.dtype),
                                         good = _good_broadcast_unary_normal_no_complex,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

RoundHalfToEvenTester = makeBroadcastTester(op = round_half_to_even,
                                  expected = numpy.round,
                                  good = _good_broadcast_unary_normal_float_no_complex)
# TODO: Why complex are accepted in the next one?
RoundHalfToEvenInplaceTester = makeBroadcastTester(op = inplace.round_half_to_even_inplace,
                                         expected = numpy.round,
                                         good = _good_broadcast_unary_normal_float,
                                         inplace = True)

#numpy.vectorize don't handle correctly empty ndarray.
#see in their file numpy/lib/function_base.py in class vectorize.__call__
#This happen in float32 mode.
RoundHalfAwayFromZeroTester = makeBroadcastTester(op = round_half_away_from_zero,
                                  expected = theano.scalar.basic.round_half_away_from_zero_vec,
                                  good = _good_broadcast_unary_normal_float_no_empty_no_complex)#_good_broadcast_unary_normal_float)
RoundHalfAwayFromZeroInplaceTester = makeBroadcastTester(op = inplace.round_half_away_from_zero_inplace,
                                         expected = theano.scalar.basic.round_half_away_from_zero_vec,
                                         good = _good_broadcast_unary_normal_float_no_empty_no_complex,
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
                                      integers = (randint_ranged(1, 5, (2, 3)),),
                                      complex = (randc128_ranged(1, 5, (2,3)),),
                                      empty = (numpy.asarray([]),),
                                      )

_grad_broadcast_unary_positive = dict(normal = (rand_ranged(0.001, 5, (2, 3)),),
                                      #complex = (randc128_ranged(1, 5, (2,3)),),
                                      #empty = (numpy.asarray([]),),
                                      )

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
                                  integers = (randint_ranged(-1000, 1000, (2, 3)),),
                                  complex = (randc128_ranged(-1000, 1000, (2, 3)),),
                                  empty = (numpy.asarray([]),),)

_grad_broadcast_unary_wide = dict(normal = (rand_ranged(-1000, 1000, (2, 3)),),
                                  #complex = (randc128_ranged(-1000, 1000, (2, 3)),),
                                  #empty = (numpy.asarray([]),),
                                  )


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

tan_grad_rtol = None
if config.floatX=='float32':
#We raise the relative tolerence for the grad as their is error in float32
#This is probably caused by our way of computing the gradient error.
    tan_grad_rtol = 0.052
TanTester = makeBroadcastTester(op = tan,
                                  expected = numpy.tan,
                                  good = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                              shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                  grad = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                              shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                  grad_rtol=tan_grad_rtol)
TanInplaceTester = makeBroadcastTester(op = inplace.tan_inplace,
                                         expected = numpy.tan,
                                         good = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                                     shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                         grad = dict(normal = (rand_ranged(-3.14, 3.14, (2, 3)),),
                                                     shifted = (rand_ranged(3.15, 6.28, (2, 3)),)),
                                         grad_rtol=tan_grad_rtol,
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

#inplace ops when the input is integer and the output is float*
# don't have a well defined behavior. We don't test that case.
_good_broadcast_unary_normal_no_int_no_complex = _good_broadcast_unary_normal_no_complex.copy()
del _good_broadcast_unary_normal_no_int_no_complex['integers']
_good_broadcast_unary_normal_no_int = _good_broadcast_unary_normal.copy()
del _good_broadcast_unary_normal_no_int['integers']

if imported_scipy_special:
    # We can't test it if scipy is not installed!
    # Precomputing the result is brittle(it have been broken!)
    # As if we do any modification to random number here,
    # The input random number will change and the output!
    expected = scipy.special.erf
    ErfTester = makeBroadcastTester(op = erf,
                                    expected = scipy.special.erf,
                                    good = _good_broadcast_unary_normal,
                                    grad = _grad_broadcast_unary_normal,
                                    eps = 2e-10,
                                    mode = mode_no_scipy)
    ErfInplaceTester = makeBroadcastTester(op = inplace.erf_inplace,
                                           expected = scipy.special.erf,
                                           good = _good_broadcast_unary_normal_no_int,
                                           grad = _grad_broadcast_unary_normal,
                                           mode = mode_no_scipy,
                                           eps = 2e-10,
                                           inplace = True)

    ErfcTester = makeBroadcastTester(op = erfc,
                                     expected = scipy.special.erfc,
                                     good = _good_broadcast_unary_normal_no_int_no_complex,
                                     grad = _grad_broadcast_unary_normal,
                                     eps = 2e-10,
                                     mode = mode_no_scipy)
    ErfcInplaceTester = makeBroadcastTester(op = inplace.erfc_inplace,
                                            expected = scipy.special.erfc,
                                            good = _good_broadcast_unary_normal_no_int_no_complex,
                                            grad = _grad_broadcast_unary_normal,
                                            eps = 2e-10,
                                            mode = mode_no_scipy,
                                            inplace = True)


DotTester = makeTester(name = 'DotTester',
                        op = dot,
                        expected = lambda x, y: numpy.dot(x, y),
                        checks = {},
                        good = dict(correct1 = (rand(5, 7), rand(7, 5)),
                                    correct2 = (rand(5, 7), rand(7, 9)),
                                    correct3 = (rand(5, 7), rand(7)),
                                    complex1 = (randcomplex(5, 7), randcomplex(7)),
                                    complex2 = (rand(5, 7), randcomplex(7)),
                                    complex3 = (randcomplex(5, 7), rand(7)),
                                    empty = (numpy.asarray([]),numpy.asarray([])),),
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

def test_tensor_values_eq_approx():
    #test, inf, -inf and nan equal themself
    a=numpy.asarray([-numpy.inf,-1,0,1,numpy.inf,numpy.nan])
    assert TensorType.values_eq_approx(a,a)

    #test inf, -inf don't equal themself
    b=numpy.asarray([numpy.inf,-1,0,1,numpy.inf,numpy.nan])
    assert not TensorType.values_eq_approx(a,b)
    b=numpy.asarray([-numpy.inf,-1,0,1,-numpy.inf,numpy.nan])
    assert not TensorType.values_eq_approx(a,b)

    #test allow_remove_inf
    b=numpy.asarray([numpy.inf,-1,0,1,5,numpy.nan])
    assert TensorType.values_eq_approx(a,b,allow_remove_inf=True)
    b=numpy.asarray([numpy.inf,-1,0,1,5,6])
    assert not TensorType.values_eq_approx(a,b,allow_remove_inf=True)

    #test allow_remove_nan
    b=numpy.asarray([numpy.inf,-1,0,1,5,numpy.nan])
    assert not TensorType.values_eq_approx(a,b,allow_remove_nan=False)
    b=numpy.asarray([-numpy.inf,-1,0,1,numpy.inf,6])
    assert not TensorType.values_eq_approx(a,b,allow_remove_nan=False)

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
        v = eval_outputs(max_and_argmax(n)[1].shape)
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
        v,i = eval_outputs(max_and_argmax(n,-1))
        self.failUnless(numpy.all(v == numpy.max(data,-1)))
        self.failUnless(numpy.all(i == numpy.argmax(data,-1)))
        v = eval_outputs(max_and_argmax(n,-1)[0].shape)
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
        #currently not supported
        #v = eval_outputs(max_and_argmax(n,[0,1])[0].shape)
        #assert v.size==0

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
        self.failUnless(i.shape == (2,))
        self.failUnless(numpy.all(v == numpy.max(n.value,-1)))
        self.failUnless(numpy.all(i == numpy.argmax(n.value,-1)))
        v,i = eval_outputs(max_and_argmax(n,-2))
        self.failUnless(v.shape == (3,))
        self.failUnless(i.shape == (3,))
        self.failUnless(numpy.all(v == numpy.max(n.value,-2)))
        self.failUnless(numpy.all(i == numpy.argmax(n.value,-2)))
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

    def test_grad(self):
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)

        def check_grad_max(data, max_grad_data, axis=None):
            #This work only for axis in [0,None]
            assert axis in [0,None]
            z = numpy.zeros_like(data)
            z = z.flatten()
            argmax=numpy.argmax(data,axis=axis)
            if argmax.ndim==0:
                z[numpy.argmax(data,axis=axis)]+=1
            else:
                for id,v in enumerate(argmax):
                    z[v*numpy.prod(data.shape[data.ndim-1:axis:-1])+id]+=1

            z = z.reshape(data.shape)
            assert numpy.all(max_grad_data == z)

        #test grad of max
        #axis is the last one
        utt.verify_grad(lambda v: max_and_argmax(v,axis=-1)[0], [data])
        utt.verify_grad(lambda v: max_and_argmax(v,axis=-1)[1], [data])

        utt.verify_grad(lambda v: max_and_argmax(v,axis=[0])[0], [data])
        utt.verify_grad(lambda v: max_and_argmax(v,axis=[0])[1], [data])
        check_grad_max(data,eval_outputs(grad(max_and_argmax(n,axis=0)[0].sum(),n)),axis=0)

        utt.verify_grad(lambda v: max_and_argmax(v,axis=[1])[0], [data])
        utt.verify_grad(lambda v: max_and_argmax(v,axis=[1])[1], [data])
        #check_grad_max(data,eval_outputs(grad(max_and_argmax(n,axis=1)[0],n)),axis=1)

        utt.verify_grad(lambda v: max_and_argmax(v.flatten())[0], [data])
        utt.verify_grad(lambda v: max_and_argmax(v.flatten())[1], [data])
        check_grad_max(data,eval_outputs(grad(max_and_argmax(n.flatten())[0],n)))

class T_argmin_argmax(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test0(self):
        for fct in [argmin,argmax]:
            n = as_tensor_variable(5.0)
            i = eval_outputs(fct(n))
            self.failUnless(i == 0)
            v = eval_outputs(fct(n).shape)
            assert len(v)==0

    def test1(self):
        n = as_tensor_variable([1,2,3,2,-6])
        i = eval_outputs(argmin(n))
        self.failUnless(i == 4)
        v = eval_outputs(argmin(n).shape)
        assert len(v)==0

        n = as_tensor_variable([1,2,3,2,-6])
        i = eval_outputs(argmax(n))
        self.failUnless(i == 2)
        v = eval_outputs(argmax(n).shape)
        assert len(v)==0

    def test2(self):
        for fct,nfct in [(argmax,numpy.argmax),(argmin,numpy.argmin)]:
            data = numpy.random.rand(2,3)
            n = as_tensor_variable(data)
            i = eval_outputs(fct(n,-1))
            self.failUnless(numpy.all(i == nfct(data,-1)))
            v = eval_outputs(fct(n,-1).shape)
            assert v==(2)

    def test2b(self):
        for fct,nfct in [(argmax,numpy.argmax),(argmin,numpy.argmin)]:
            data = numpy.random.rand(2,3)
            n = as_tensor_variable(data)
            i = eval_outputs(fct(n,0))
            self.failUnless(numpy.all(i == nfct(data,0)))
            v = eval_outputs(fct(n,0).shape)
            assert v==(3)
            v = eval_outputs(fct(n,1).shape)
            assert v==(2)
            #currently not supported
            #v = eval_outputs(fct(n,[0,1]).shape)
            #assert v.size==0

    def test2_invalid(self):
        for fct,nfct in [(argmax,numpy.argmax),(argmin,numpy.argmin)]:
            n = as_tensor_variable(numpy.random.rand(2,3))
            # Silence expected error messages
            _logger = logging.getLogger('theano.gof.opt')
            oldlevel = _logger.getEffectiveLevel()
            _logger.setLevel(logging.CRITICAL)
            try:
                try:
                    eval_outputs(fct(n,3))
                    assert False
                except ValueError, e:
                    pass
            finally:
                _logger.setLevel(oldlevel)

    def test2_invalid_neg(self):
        for fct,nfct in [(argmax,numpy.argmax),(argmin,numpy.argmin)]:
            n = as_tensor_variable(numpy.random.rand(2,3))
            old_stderr = sys.stderr
            sys.stderr = StringIO.StringIO()
            try:
                try:
                    eval_outputs(fct(n,-3))
                    assert False
                except ValueError, e:
                    pass
            finally:
                sys.stderr = old_stderr

    def test2_valid_neg(self):
        for fct,nfct in [(argmax,numpy.argmax),(argmin,numpy.argmin)]:
            n = as_tensor_variable(numpy.random.rand(2,3))
            i = eval_outputs(fct(n,-1))
            self.failUnless(i.shape == (2,))
            self.failUnless(numpy.all(i == nfct(n.value,-1)))
            i = eval_outputs(fct(n,-2))
            self.failUnless(i.shape == (3,))
            self.failUnless(numpy.all(i == nfct(n.value,-2)))

            v = eval_outputs(fct(n,-1).shape)
            assert v==(2)
            v = eval_outputs(fct(n,-2).shape)
            assert v==(3)

    def test3(self):
        for fct,nfct in [(argmax,numpy.argmax),(argmin,numpy.argmin)]:
            n = as_tensor_variable(numpy.random.rand(2,3,4))
            i = eval_outputs(fct(n,0))
            self.failUnless(i.shape == (3,4))
            self.failUnless(numpy.all(i == nfct(n.value,0)))
            i = eval_outputs(fct(n,1))
            self.failUnless(i.shape == (2,4))
            self.failUnless(numpy.all(i == nfct(n.value,1)))
            i = eval_outputs(fct(n,2))
            self.failUnless(i.shape == (2,3))
            self.failUnless(numpy.all(i == nfct(n.value,2)))

            v = eval_outputs(fct(n,0).shape)
            assert tuple(v)==(3,4)
            v = eval_outputs(fct(n,1).shape)
            assert tuple(v)==(2,4)
            v = eval_outputs(fct(n,2).shape)
            assert tuple(v)==(2,3)

    def test_grad_argmin(self):
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)

        #test grad of argmin
        utt.verify_grad(lambda v: argmin(v,axis=-1), [data])

        utt.verify_grad(lambda v: argmin(v,axis=[0]), [data])

        utt.verify_grad(lambda v: argmin(v,axis=[1]), [data])

        utt.verify_grad(lambda v: argmin(v.flatten()), [data])

        try:
            grad(argmin(n,axis=-1),n)
            raise Exception('Expected an error')
        except TypeError:
            pass

    def test_grad_argmax(self):
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)

        #test grad of argmax
        utt.verify_grad(lambda v: argmax(v, axis=-1), [data])

        utt.verify_grad(lambda v: argmax(v,axis=[0]), [data])

        utt.verify_grad(lambda v: argmax(v,axis=[1]), [data])

        utt.verify_grad(lambda v: argmax(v.flatten()), [data])

        try:
            grad(argmax(n, axis=-1),n)
            raise Exception('Expected an error')
        except TypeError:
            pass

class T_min_max(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test0(self):
        for fct in [max,min]:
            n = as_tensor_variable(5.0)
            v = eval_outputs(fct(n))
            self.failUnless(v == 5.0)

            v = eval_outputs(fct(n).shape)
            assert len(v)==0

    def test1(self):
        for fct,nfct in [(max,numpy.max),(min,numpy.min)]:
            n = as_tensor_variable([1,2,3,2,-6])
            v = eval_outputs([fct(n)])
            self.failUnless(v == nfct(n.value))

            v = eval_outputs(fct(n).shape)
            assert len(v)==0

    def test2(self):
        for fct,nfct in [(max,numpy.max),(min,numpy.min)]:
            data = numpy.random.rand(2,3)
            n = as_tensor_variable(data)
            v = eval_outputs(fct(n,-1))
            self.failUnless(numpy.all(v == nfct(data,-1)))

            v = eval_outputs(fct(n,-1).shape)
            assert v==(2)

    def test2b(self):
        for fct,nfct in [(max,numpy.max),(min,numpy.min)]:
            data = numpy.random.rand(2,3)
            n = as_tensor_variable(data)
            v = eval_outputs(fct(n,0))
            self.failUnless(numpy.all(v == nfct(data,0)))

            v = eval_outputs(fct(n,0).shape)
            assert v==(3)
            v = eval_outputs(fct(n,1).shape)
            assert v==(2)
            v = eval_outputs(fct(n,[0,1]).shape)
            assert v.size==0

    def test2_invalid(self):
        for fct in [max,min]:
            n = as_tensor_variable(numpy.random.rand(2,3))
            # Silence expected error messages
            _logger = logging.getLogger('theano.gof.opt')
            oldlevel = _logger.getEffectiveLevel()
            _logger.setLevel(logging.CRITICAL)
            try:
                try:
                    eval_outputs(fct(n,3))
                    assert False
                except ValueError, e:
                    pass
            finally:
                _logger.setLevel(oldlevel)
    def test2_invalid_neg(self):
        for fct in [max,min]:
            n = as_tensor_variable(numpy.random.rand(2,3))
            old_stderr = sys.stderr
            sys.stderr = StringIO.StringIO()
            try:
                try:
                    eval_outputs(fct(n,-3))
                    assert False
                except ValueError, e:
                    pass
            finally:
                sys.stderr = old_stderr
    def test2_valid_neg(self):
        for fct,nfct in [(max,numpy.max),(min,numpy.min)]:
            n = as_tensor_variable(numpy.random.rand(2,3))
            v = eval_outputs(fct(n,-1))
            self.failUnless(v.shape == (2,))
            self.failUnless(numpy.all(v == nfct(n.value,-1)))
            v = eval_outputs(fct(n,-2))
            self.failUnless(v.shape == (3,))
            self.failUnless(numpy.all(v == nfct(n.value,-2)))

            v = eval_outputs(fct(n,-1).shape)
            assert v==(2)
            v = eval_outputs(fct(n,-2).shape)
            assert v==(3)

    def test3(self):
        for fct,nfct in [(max,numpy.max),(min,numpy.min)]:
            n = as_tensor_variable(numpy.random.rand(2,3,4))
            v = eval_outputs(fct(n,0))
            self.failUnless(v.shape == (3,4))
            self.failUnless(numpy.all(v == nfct(n.value,0)))
            v = eval_outputs(fct(n,1))
            self.failUnless(v.shape == (2,4))
            self.failUnless(numpy.all(v == nfct(n.value,1)))
            v = eval_outputs(fct(n,2))
            self.failUnless(v.shape == (2,3))
            self.failUnless(numpy.all(v == nfct(n.value,2)))
            v = eval_outputs(fct(n,[0,1]))
            self.failUnless(v.shape == (4,))
            self.failUnless(numpy.all(v == nfct(nfct(n.value,1),0)))
            v = eval_outputs(fct(n,[0,2]))
            self.failUnless(v.shape == (3,))
            self.failUnless(numpy.all(v == nfct(nfct(n.value,2),0)))
            v = eval_outputs(fct(n,[1,2]))
            self.failUnless(v.shape == (2,))
            self.failUnless(numpy.all(v == nfct(nfct(n.value,2),1)))
            v = eval_outputs(fct(n,[0,1,2]))
            self.failUnless(v.shape == ())

            v = eval_outputs(fct(n,0).shape)
            assert tuple(v)==(3,4)
            v = eval_outputs(fct(n,1).shape)
            assert tuple(v)==(2,4)
            v = eval_outputs(fct(n,2).shape)
            assert tuple(v)==(2,3)
            v = eval_outputs(fct(n,[0,1]).shape)
            self.failUnless(v == (4,))
            v = eval_outputs(fct(n,[0,2]).shape)
            self.failUnless(v == (3,))
            v = eval_outputs(fct(n,[1,2]).shape)
            self.failUnless(v == (2,))
            v = eval_outputs(fct(n,[0,1,2]).shape)
            self.failUnless(v.size == 0)

    def test_grad_max(self):
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)

        def check_grad_max(data, max_grad_data, axis=None):
            #This work only for axis in [0,None]
            assert axis in [0,None]
            z = numpy.zeros_like(data)
            z = z.flatten()
            argmax=numpy.argmax(data,axis=axis)
            if argmax.ndim==0:
                z[numpy.argmax(data,axis=axis)]+=1
            else:
                for id,v in enumerate(argmax):
                    z[v*numpy.prod(data.shape[data.ndim-1:axis:-1])+id]+=1

            z = z.reshape(data.shape)
            assert numpy.all(max_grad_data == z)

        #test grad of max
        #axis is the last one
        utt.verify_grad(lambda v: max(v,axis=-1), [data])

        utt.verify_grad(lambda v: max(v,axis=[0]), [data])
        check_grad_max(data,eval_outputs(grad(max(n,axis=0).sum(),n)),axis=0)

        utt.verify_grad(lambda v: max(v,axis=[1]), [data])
        #check_grad_max(data,eval_outputs(grad(max(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: max(v.flatten()), [data])
        check_grad_max(data,eval_outputs(grad(max(n.flatten()),n)))

    def test_grad_min(self):
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)

        def check_grad_min(data, min_grad_data, axis=None):
            #This work only for axis in [0,None]
            assert axis in [0,None]
            z = numpy.zeros_like(data)
            z = z.flatten()
            argmin=numpy.argmin(data,axis=axis)
            if argmin.ndim==0:
                z[numpy.argmin(data,axis=axis)]+=1
            else:
                for id,v in enumerate(argmin):
                    z[v*numpy.prod(data.shape[data.ndim-1:axis:-1])+id]+=1

            z = z.reshape(data.shape)
            assert numpy.all(min_grad_data == z)

        #test grad of min
        #axis is the last one
        utt.verify_grad(lambda v: min(v,axis=-1), [data])

        utt.verify_grad(lambda v: min(v,axis=[0]), [data])
        check_grad_min(data,eval_outputs(grad(min(n,axis=0).sum(),n)),axis=0)

        utt.verify_grad(lambda v: min(v,axis=[1]), [data])
        #check_grad_min(data,eval_outputs(grad(min(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: min(v.flatten()), [data])
        check_grad_min(data,eval_outputs(grad(min(n.flatten()),n)))

    def _grad_list(self):
        """
        Test the gradient when we have multiple axis at the same time.

        This not implemented, so we disable the test. See ticket: http://trac-hg.assembla.com/theano/ticket/511
        """
        data = numpy.random.rand(2,3)
        n = as_tensor_variable(data)
        for fct in [max_and_argmax,max,min]:
            utt.verify_grad(lambda v: fct(v,axis=[0,1]), [data])
        #check_grad_max(data,eval_outputs(grad(max_and_argmax(n,axis=1)[0],n)),axis=1)

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
            self.failUnless(hasattr(e,'subtensor_invalid'))
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
            self.failUnless(hasattr(e,'subtensor_invalid'))
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
        not Join. Test that the floatX dtype stay floatX, not down casted to int64'''
        a = tensor.scalar('a')
        b = tensor.scalar('b')
        s = stack(a, b, a, b)
        f = function([a,b], s)
        val = f(1,2)
        print val
        self.failUnless(numpy.all(val == [1,2,1,2]))
        e = f.maker.env.toposort()
        assert len([n for n in e if isinstance(n.op,opt.MakeVector)]) > 0
        assert len([n for n in e if isinstance(n, Join)]) == 0
        assert f.maker.env.outputs[0].dtype == config.floatX

    def test_stack_scalar_make_vector_dtype(self):
        '''Test that calling stack() on scalars instantiates MakeVector,
        event when the scalar don't have the same dtype.'''
        a = tensor.iscalar('a')
        b = tensor.lscalar('b')
        s = stack(a, b, a, b)
        f = function([a,b], s)
        val = f(1,2)
        self.failUnless(numpy.all(val == [1,2,1,2]))
        e = f.maker.env.toposort()
        assert len([n for n in e if isinstance(n.op,opt.MakeVector)]) > 0
        assert len([n for n in e if isinstance(n, Join)]) == 0
        assert f.maker.env.outputs[0].dtype == 'int64'

    def test_stack_scalar_make_vector_constant(self):
        '''Test that calling stack() on scalars instantiates MakeVector,
        event when the scalar are simple int type.'''
        a = tensor.iscalar('a')
        b = tensor.lscalar('b')
        #test when the constant is the first element.
        #The first element is used in a special way
        s = stack(10,a,b, numpy.int8(3))
        f = function([a,b], s)
        val = f(1,2)
        self.failUnless(numpy.all(val == [10,1,2,3]))
        e = f.maker.env.toposort()
        assert len([n for n in e if isinstance(n.op,opt.MakeVector)]) > 0
        assert len([n for n in e if isinstance(n, Join)]) == 0
        assert f.maker.env.outputs[0].dtype == 'int64'

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

        utt.verify_grad(lambda a, b: join(1,a,b), [av, bv], eps=1.0e-4, rel_tol=1.0e-3)

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

        utt.verify_grad(lambda a, b: join(1,a,b), [av, bv], eps=1.0e-4, rel_tol=1.0e-3)

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

    def test_broadcastable_flag_assignment_mixed_otheraxes(self):
        """
        Test that the broadcastable flags for the output of
        a join operation on non-join axes are True if one or
        more inputs is broadcastable on that dimension.
        """
        a = TensorType(dtype='int8', broadcastable=[0, 0, 1])()
        b = TensorType(dtype='int8', broadcastable=[1, 0, 1])()
        c = join(1, a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

    def test_broadcastable_flag_assignment_mixed_thisaxes(self):
        """
        Test that the broadcastable flag of the join axis
        is False when some inputs are broadcastable on that
        dimension.
        """
        a = TensorType(dtype='int8', broadcastable=[0, 0, 1])()
        b = TensorType(dtype='int8', broadcastable=[1, 0, 1])()
        c = join(0, a, b)
        assert not c.type.broadcastable[0]

    def test_broadcastable_flags_all_broadcastable_on_joinaxis(self):
        """
        Test that joining together several inputs which are all
        broadcastable on the join dimension results in the output
        being non-broadcastable on the join dimension.
        """
        a = TensorType(dtype='int8', broadcastable=[1, 0, 1])()
        b = TensorType(dtype='int8', broadcastable=[1, 0, 1])()
        c = join(0, a, b)
        assert not c.type.broadcastable[0]

    def test_broadcastable_single_input_broadcastable_dimension(self):
        """
        Test that all broadcastable flags are preserved by a
        single-input join.
        """
        a = join(0, TensorType(dtype='int8', broadcastable=[1, 0, 1])())
        assert a.type.broadcastable[0]
        assert a.type.broadcastable[2]
        assert not a.type.broadcastable[1]

    def test_broadcastable_flags_many_dims_and_inputs(self):
        """
        Test that the right broadcastable flags get set for a  join
        with many inputs and many input dimensions.
        """
        a = TensorType(dtype='int8', broadcastable=[1, 0, 1, 0, 0, 0])()
        b = TensorType(dtype='int8', broadcastable=[1, 1, 1, 0, 0, 0])()
        c = TensorType(dtype='int8', broadcastable=[1, 0, 0, 0, 0, 0])()
        d = TensorType(dtype='int8', broadcastable=[1, 0, 1, 1, 0, 1])()
        e = TensorType(dtype='int8', broadcastable=[1, 0, 1, 0, 0, 1])()
        f = join(0, a, b, c, d, e)
        fb = f.type.broadcastable
        assert not fb[0] and fb[1] and fb[2] and fb[3] and not fb[4] and fb[5]
        g = join(1, a, b, c, d, e)
        gb = g.type.broadcastable
        assert gb[0] and not gb[1] and gb[2] and gb[3] and not gb[4] and gb[5]
        h = join(4, a, b, c, d, e)
        hb = h.type.broadcastable
        assert hb[0] and hb[1] and hb[2] and hb[3] and not hb[4] and hb[5]

class test_comparison(unittest.TestCase):
    def test_gt(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x > y)
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.failUnless(numpy.all(v == (l > r)), (v, (l>r)))

    def test_lt(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x < y)
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.failUnless(numpy.all(v == (l < r)), (v, (l<r)))

    def test_le(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x <= y)
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.failUnless(numpy.all(v == (l <= r)), (v, (l<=r)))

    def test_ge(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x >= y)
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.failUnless(numpy.all(v == (l >= r)), (v, (l>=r)))

    def test_eq(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], eq(x,y))
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.failUnless(numpy.all(v == (l == r)), (v, (l==r)))

    def test_neq(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], neq(x, y))
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
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

        assert numpy.allclose(function([i, ii, d, f, c], i/d)(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (5.0/7.0))
        assert numpy.allclose(function([i, ii, d, f, c], d/i)(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (7.0/5.0))
        assert numpy.allclose(function([i, ii, d, f, c], i/f)(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (5.0/11.0))
        assert numpy.allclose(function([i, ii, d, f, c], f/i)(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (11.0/5.0))
        assert numpy.allclose(function([i, ii, d, f, c], i/ii)(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (5/3))
        assert numpy.allclose(function([i, ii, d, f, c], ii/i)(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (3/5))
        assert numpy.allclose(function([i, ii, d, f, c], true_div(i,ii))(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (5./3.))
        assert numpy.allclose(function([i, ii, d, f, c], true_div(ii,i))(5, 3, 7.0, 11.0, numpy.complex(5,3)),
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
        diff = ab - as_tensor_variable(numpy.ones((dim,dim), dtype=config.floatX))
        # Sum of squared errors
        ssdiff = sum((diff**2.0))

        g_b = grad(ssdiff, b)

        # compilation to function
        # [a,b] are the inputs, [ssdiff,g_b] are the outputs
        fn = inplace_func([a,b], [ssdiff,g_b])

        # use the function
        x = numpy.random.rand(dim,dim)+0.1      # Initialized s.t. x is not too tiny
        w = numpy.random.rand(dim,dim)
        x = numpy.asarray(x, dtype=config.floatX)
        w = numpy.asarray(w, dtype=config.floatX)

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
        x = numpy.asarray(x, dtype=config.floatX)
        w = numpy.asarray(w, dtype=config.floatX)
        ones = numpy.ones((3,3), dtype=config.floatX)

        myssd0 = numpy.sum((x*w - ones)**2.0)
        # we want at least a test that is not too fast. So we make one here.
        for i in xrange(100):
            gw = 2*(x*w - ones)*x  # derivative of dMSE/dw
            myssd = numpy.sum((x*w - ones)**2)
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

    def test_broadcastable_patterns(self):

        #
        # These examples hsould all work because we broadcastable or no, all dimensions of all
        # results have size 1.
        #
        def val_for(r):

            if r.ndim == 0:
                return numpy.asarray(1.1, dtype=r.dtype)
            if r.ndim == 1:
                return numpy.asarray([1.2], dtype=r.dtype)
            elif r.ndim == 2:
                return numpy.asarray([[1.3]], dtype=r.dtype)
            raise ValueError()

        for dtype0 in ('float32', 'float64', 'complex64', 'complex128'):
            for dtype1 in ('float32', 'float64', 'complex64', 'complex128'):
                for bc0 in ((True,), (False,), (True, True), (True, False), (False, True),
                        (False,False)):
                    for bc1 in ((True,), (False,), (True, True), (True, False), (False, True),
                            (False,False)):

                        x = TensorType(dtype=dtype0, broadcastable=bc0)()
                        y = TensorType(dtype=dtype1, broadcastable=bc1)()
                        z = dot(x,y)
                        t = TensorType(dtype=dtype0, broadcastable=z.broadcastable)()

                        rval =  z * 3 + 2*t
                        f = function([x,y,t], rval)
                        xval = val_for(x)
                        yval = val_for(y)
                        tval = val_for(t)

                        f(xval, yval, tval) #debugmode checks result


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

    @dec.knownfailureif(
            isinstance(get_default_mode(),theano.compile.debugmode.DebugMode),
            ("This test fails in DEBUG_MODE, but the generated code is OK. "
             "It is actually a problem of DEBUG_MODE, see #624."))
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
    @dec.knownfailureif(
        isinstance(get_default_mode(),theano.compile.debugmode.DebugMode),
            ("This test fails in DEBUG_MODE, but the generated code is OK. "
             "It is actually a problem of DEBUG_MODE, see #625."))
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

    def test_cost_is_scalar(self):
        '''grad: Test that a non-scalar cost raises a TypeError'''
        s = scalar()
        v = vector()
        m = matrix()
        # grad(v,...) and grad(m,...) should fail
        self.assertRaises(TypeError, grad, v, s)
        self.assertRaises(TypeError, grad, v, m)
        self.assertRaises(TypeError, grad, m, s)
        self.assertRaises(TypeError, grad, m, v)

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

        a = numpy.random.rand(5,2).astype(config.floatX)
        self.failUnless(numpy.all(fn_py(a) == fn_c_or_py(a)))


def test_reshape():

    a = dvector()
    b = dmatrix()
    d = dmatrix()

    #basic to 1 dim(without list)
    c = reshape(b, as_tensor_variable(6), ndim=1)
    f = inplace_func([b], c)
    assert numpy.all(f(numpy.asarray([[0,1,2],[3,4,5]])) == numpy.asarray([0,1,2,3,4,5]))
    print f.maker.env.toposort()
    #check that we remove the useless reshape

    #basic to 1 dim(with list)
    c = reshape(b, (as_tensor_variable(6),), ndim=1)
    f = inplace_func([b], c)
    assert numpy.all(f(numpy.asarray([[0,1,2],[3,4,5]])) == numpy.asarray([0,1,2,3,4,5]))
    print f.maker.env.toposort()
    #check that we remove the useless reshape

    #basic to shape object of same ndim
    c = reshape(b,d.shape)
    f = inplace_func([b,d], c)
    assert numpy.all(f(numpy.asarray([[0,1,2],[3,4,5]]),[[0,1],[2,3],[4,5]]) == numpy.asarray([[0,1],[2,3],[4,5]]))

    #basic to 2 dims
    c = reshape(a, [2,3])
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

    #test infer_shape
    f_sub = function([a,b], (c-b).shape)
    if config.mode=="FAST_COMPILE":
        assert len(f_sub.maker.env.toposort())==3
    else:
        assert len(f_sub.maker.env.toposort())==0
        #assert numpy.all(f_sub(a_val,numpy.asarray([[0,1],[2,3],[4,5]]))==[2,3])#work in FAST_RUN, but fail on other!
        #assert numpy.all(f_sub(a_val,numpy.asarray([[0,1],[2,3],[4,5],[6,7]]))==[2,3])#work in FAST_RUN, but fail on other!

    assert numpy.all(f_sub(a_val,b_val)==[2,3])

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
# See Ticket #619
#def test_tile():
#    print >> sys.stderr, "WARNING: No testcase for Tile"
#    pass


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
        """Test arange constructor, on float32 outputs"""
        start, stop, step = fscalars('start', 'stop', 'step')
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        assert out.dtype == start.type.dtype
        arg_vals = [ (0,5,1), (2,11,4), (-5,1.1,1.2), (1.3,2,-2.1), (10,2,2) ]
        for arg_v in arg_vals:
            start_v, stop_v, step_v = arg_v
            start_v_, stop_v_, step_v_ = numpy.asarray(arg_v, dtype=start.type.dtype)
            assert numpy.all(f(start_v_, stop_v_, step_v_) == \
                    numpy.arange(start_v, stop_v, step_v, dtype=start.type.dtype))

    def test_float64(self):
        """Test arange constructor, on float64 outputs"""
        start, stop, step = dscalars('start', 'stop', 'step')
        out = arange(start, stop, step)
        f = function([start, stop, step], out)

        assert out.dtype == start.type.dtype
        arg_vals = [ (0,5,1), (2,11,4), (-5,1.1,1.2), (1.3,2,-2.1), (10,2,2) ]
        for arg_v in arg_vals:
            start_v, stop_v, step_v = arg_v
            start_v_, stop_v_, step_v_ = numpy.asarray(arg_v, dtype=start.type.dtype)
            assert numpy.all(f(start_v_, stop_v_, step_v_) == \
                    numpy.arange(start_v, stop_v, step_v, dtype=start.type.dtype))

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
        fstop_values = [0.2, -0.7, 8.5]
        for fstop_v in fstop_values:
            fstop_v32 = numpy.float32(fstop_v)
            assert numpy.all(ff(fstop_v32) == numpy.arange(fstop_v))

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
        assert inv.dtype == p.dtype
        f_inverse = function([p], inv)

        # Generate a random permutation
        rng = numpy.random.RandomState(utt.fetch_seed())
        p_val = rng.permutation(10).astype('int32')
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
        p_val = numpy.asarray([rng.permutation(10) for i in range(7)],
                              dtype='int32')
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
        p_val = rng.permutation(5).astype('int32')
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
        p_val = rng.permutation(5).astype('int32')
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
        p_val = numpy.asarray([rng.permutation(5) for i in range(3)],
                              dtype='int32')
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
        p_val = numpy.asarray([rng.permutation(5) for i in range(3)], dtype='int32')
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
        p_val = numpy.asarray([rng.permutation(5) for i in range(3)],
                              dtype='int32')
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

    def rand(self, *shape):
        return numpy.asarray(numpy.random.rand(*shape), dtype=config.floatX)

    def test0(self):

        # Test vector-vector
        avec = vector()
        bvec = vector()
        axes = ((0,),(0,))
        c = tensordot(avec, bvec, axes)
        f1 = inplace_func([avec,bvec],c)
        aval = self.rand(5);
        bval = self.rand(5);
        self.failUnless(numpy.tensordot(aval,bval,axes) == \
                        f1(aval,bval))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test matrix-vector
        bmat = matrix()
        axes = ((0,),(1,))
        c = tensordot(avec, bmat, axes)
        f2 = inplace_func([avec,bmat],c)
        aval = self.rand(5);
        bval = self.rand(8,5);
        self.failUnless(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f2(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test matrix-matrix
        amat = matrix()
        axes = ((1,),(0,))
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat,bmat],c)
        aval = self.rand(4,7);
        bval = self.rand(7,9);
        self.failUnless(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f3(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test ndarray-matrix, sum over one dim of matrix
        atens = tensor4()
        axes = ((2,),(1,))
        c = tensordot(atens, bmat, axes)
        f4 = inplace_func([atens,bmat],c)
        aval = self.rand(1,2,3,4);
        bval = self.rand(2,3);
        self.failUnless(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f4(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test ndarray-ndarray
        atens = tensor4()
        btens = tensor3()
        axes = ((1,3),(0,2))
        c = tensordot(atens, btens, axes)
        f5 = inplace_func([atens,btens],c)
        aval = self.rand(4,3,5,2);
        bval = self.rand(3,4,2);
        self.failUnless(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f5(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        axes = (axes[1],axes[0])
        c = tensordot(btens, atens, axes)
        f6 = inplace_func([btens,atens],c)
        self.failUnless(numpy.allclose(numpy.tensordot(bval,aval,axes),
                                       f6(bval,aval)))
        utt.verify_grad(TensorDot(axes), [bval,aval])

    def test_raise_error(self):
        amat = matrix()
        bmat = matrix()
        bvec = vector()

        # Test invalid length for axes
        try:
            c = tensordot(amat, bmat, (0,1,2))
            assert False
        except ValueError:
            pass

        # Test axes of uneven length
        try:
            c = tensordot(amat, bmat, ((0,1),(0)))
            assert False
        except ValueError:
            pass

        # Test invalid len(axes) given inputs are matrices
        try:
            c = tensordot(amat, bmat, ((0,1,2),(0,1,2)))
            assert False
        except ValueError:
            pass

        # Test invalid axes[1] given that y is a vector
        try:
            c = tensordot(amat, bvec, (0,1))
            assert False
        except ValueError:
            pass

        # Test invalid scalar axes given inputs are matrices
        try:
            c = tensordot(amat, bvec, 2)
            assert False
        except ValueError:
            pass


    def test_weird_valid_axes(self):
        # Test matrix-matrix
        amat = matrix()
        bmat = matrix()
        for axes in 0, (1,0), [1,0], (1,(0,)), ((1,),0), ([1],[0]):
            c = tensordot(amat, bmat, axes)
            f3 = inplace_func([amat,bmat],c)
            aval = self.rand(4,7);
            bval = self.rand(7,9);
            self.failUnless(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                           f3(aval,bval)))
            utt.verify_grad(TensorDot(axes), [aval,bval])

    def test_scalar_axes(self):
        # Test matrix-matrix
        amat = fmatrix()
        bmat = dmatrix()# We let at float64 to test mix of float32 and float64.
        axes = 1
        aval = self.rand(4,5).astype('float32')
        bval = numpy.random.rand(5,3)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat,bmat],c)
        self.failUnless(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f3(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test tensor-tensor
        amat = tensor3()
        bmat = tensor3()
        axes = 2
        aval = self.rand(3,4,5)
        bval = self.rand(4,5,3)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat,bmat],c)
        self.failUnless(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f3(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

    def test_scalar0(self):
        # Test tensor-tensor
        amat = matrix()
        bmat = matrix()
        axes = 0
        aval = self.rand(4,5)
        bval = self.rand(5,4)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat,bmat],c)
        self.failUnless(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f3(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

    def test_tensordot_grad(self):
        # We test it manually as we recreate the op in the make_node

        amat = matrix()
        bmat = matrix()
        gzmat = matrix()
        axes = 1
        aval = self.rand(4,5)
        bval = self.rand(5,3)
        gzval = self.rand(4,3)
        f1 = inplace_func([amat,bmat,gzmat],tensordot_grad(axes)(amat, bmat, gzmat))
        f2 = inplace_func([amat,bmat,gzmat],tensordot_grad(((1,),(0,)))(amat, bmat, gzmat))
        o1=f1(aval,bval,gzval)
        o2=f2(aval,bval,gzval)
        self.failUnless(numpy.allclose(o1[0],o2[0]))
        self.failUnless(numpy.allclose(o1[1],o2[1]))

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

@dec.knownfailureif(
        isinstance(get_default_mode(),theano.compile.debugmode.DebugMode),
        ("This test fails in DEBUG_MODE, but the generated code is OK. "
         "It is actually a problem of DEBUG_MODE, see #626."))
def test_default():
    x, y = scalars('xy')
    z = default(x, y)
    f = function([x, y], z)
    assert f(1, 2) == 1
    assert f(None, 2) == 2
    assert f(1, None) == 1

@dec.knownfailureif(
        isinstance(get_default_mode(),theano.compile.debugmode.DebugMode),
        ("This test fails in DEBUG_MODE, but the generated code is OK. "
         "It is actually a problem of DEBUG_MODE, see #626."))
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
    assert numpy.allclose(f(numpy.asarray(2.2, dtype=config.floatX)), 7)

def test_autocast():
    orig_autocast = autocast_float.dtypes

    # Test that autocast_float_as sets the autocast dtype correctly
    try: # ghetto 2.4 version of with
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
    # Test that we can set it back to something, and nest it
    try: # ghetto 2.4 version of with
        ac = autocast_float_as('float32')
        ac.__enter__()
        assert autocast_float.dtypes == ('float32',)
        try: # ghetto 2.4 version of with
            ac2 = autocast_float_as('float64')
            ac2.__enter__()
            assert autocast_float.dtypes == ('float64',)
        finally:
            ac2.__exit__()
        assert autocast_float.dtypes == ('float32',)
    finally:
        ac.__exit__()
    assert autocast_float.dtypes == orig_autocast

    # Test that the autocasting dtype is used correctly in expression-building
    try: # ghetto 2.4 version of with
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

    # Test that the autocasting dtype is used correctly in expression-building
    try: # ghetto 2.4 version of with
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

    # Test that the autocasting dtype is used correctly in expression-building
    try: # ghetto 2.4 version of with
        ac = autocast_float_as('float32', 'float64')
        ac.__enter__()
        assert (dvector()+ 1.1).dtype == 'float64'
        assert (fvector()+ 1.1).dtype == theano.config.floatX
        assert (fvector()+ 1.0).dtype == 'float32'
        assert (dvector()+ numpy.float32(1.1)).dtype == 'float64'
        assert (dvector()+ numpy.float64(1.1)).dtype == 'float64'
        assert (dvector()+ numpy.float(1.1)).dtype   == 'float64'
        assert (fvector()+ numpy.float32(1.1)).dtype == 'float32'
        assert (fvector()+ numpy.float64(1.1)).dtype == 'float64'
        assert (fvector()+ numpy.float(1.1)).dtype   == theano.config.floatX
        assert (lvector()+ numpy.int64(1)).dtype == 'int64'
        assert (lvector()+ numpy.int32(1)).dtype == 'int64'
        assert (lvector()+ numpy.int16(1)).dtype == 'int64'
        assert (lvector()+ numpy.int8(1)).dtype == 'int64'
        assert (ivector()+ numpy.int8(1)).dtype == 'int32'
        assert (wvector()+ numpy.int8(1)).dtype == 'int16'
        assert (bvector()+ numpy.int8(1)).dtype == 'int8'
        try: # ghetto 2.4 version of with
            ac2 = autocast_float_as('float64')
            ac2.__enter__()
            assert (fvector()+ 1.0).dtype == 'float64'
        finally:
            ac2.__exit__()
    finally:
        ac.__exit__()

class test_broadcast(unittest.TestCase):
    def test_broadcast_bigdim(self):
        def f():
            x = matrix()
            addbroadcast(x,2)
        self.failUnlessRaises(ValueError, f)

    def test_unbroadcast_addbroadcast(self):
        """
        test that the unbroadcast fct don't insert not needed broadcast
        and fuse consecutive Rebroadcast op
        """

        x=matrix()
        assert unbroadcast(x,0) is x
        assert unbroadcast(x,1) is x
        assert unbroadcast(x,1,0) is x
        assert unbroadcast(x,0,1) is x

        assert addbroadcast(x,0) is not x
        assert addbroadcast(x,1) is not x
        assert addbroadcast(x,1,0).owner.inputs[0] is x

        assert unbroadcast(addbroadcast(x,0),0) is x
        assert addbroadcast(unbroadcast(x,0),0) is not x
        x=row()
        assert unbroadcast(x,0) is not x
        assert unbroadcast(x,1) is x
        assert unbroadcast(x,1,0) is not x
        assert unbroadcast(x,0,1) is not x

        assert addbroadcast(x,0) is x
        assert addbroadcast(x,1).owner.inputs[0] is x
        assert addbroadcast(x,1,0).owner.inputs[0] is x
        assert addbroadcast(x,0,1).owner.inputs[0] is x

        assert unbroadcast(addbroadcast(x,1),1) is x
        assert addbroadcast(unbroadcast(x,1),1) is not x

        # The first broadcast is remove the broadcast, so the second
        # should not make one
        assert unbroadcast(unbroadcast(x,0),0).owner.inputs[0] is x

        # Test that consecutive Rebroadcast op are fused
        x=TensorType(dtype = 'float64', broadcastable = (True,True))()
        assert unbroadcast(unbroadcast(x,1),0).owner.inputs[0] is x
        assert addbroadcast(unbroadcast(x,1),0).owner.inputs[0] is x
        assert addbroadcast(unbroadcast(x,0),0) is x

def test_mod():
    """
    We add this test as not all language and C implementation give the same
    signe to the result. This check that the c_code of `Mod` is implemented
    as Python. That is what we want.
    """
    x, y = fscalars('xy')
    fn = gof.DualLinker().accept(gof.Env([x,y], [x%y])).make_function()
    for a,b in ((0,1), (1,1), (0,-1), (1,-1), (-1,-1),
                (1,2), (-1,2), (1,-2), (-1,-2),
                (5,3), (-5,3), (5,-3), (-5,-3)
                ):
        assert fn(a,b) == a%b, (a,)

def test_mod_compile():
    """
    This test generate an Elemwise of Composite as:
        Elemwise{Composite{Composite{Composite{Composite{mod,EQ},Switch},mul},add}}

    The c_code generated is not compiling as of 30 June 2010. I fix the compilation in the same commit.
    """

    x = tensor.vector()
    y = tensor.vector()
    shape = x.shape
    out = tensor.switch(tensor.eq(3%x.shape[0],0),y,y[:-1])

    f = theano.function([x,y],out)

def test_unalign():
    if config.floatX == 'float64':
        dtype="b1,f8"
    else:
        dtype="b1,f4"

    a = numpy.empty(1e4, dtype=dtype)['f1']
    b = numpy.empty(1e4, dtype=dtype)['f1']
    assert not a.flags.aligned
    assert not b.flags.aligned
    a[:] = numpy.random.rand(len(a))
    b[:] = numpy.random.rand(len(b))
    out_numpy = 2*a + 3*b

    av,bv = tensor.vectors('ab')
    f = theano.function([av,bv],2*av+3*bv)
    f.maker.env.toposort()
    # FAST_COMPILE use the python code that support unaligned data
    # The DebugMode make a copy of the inputs, so they will be aligned.
    should_raise = theano.config.mode not in ["FAST_COMPILE","DebugMode", "DEBUG_MODE"]
    try:
        out_theano = f(a,b)
        assert not a.flags.aligned
        assert not b.flags.aligned
        assert numpy.allclose(out_numpy,out_theano)
        if should_raise:
            raise Exception("Expected an error from Theano!")
    except NotImplementedError, e:
        if not should_raise:
            raise Exception("Theano raised an exception when none was expected")

def test_dimshuffle_duplicate():
    x = theano.tensor.vector()

    success = False

    try:
        y = theano.tensor.DimShuffle((False, ), (0, 0))(x)
    except ValueError, e:
        assert str(e).find("may not appear twice") != -1
        success = True

    assert success

class T_get_constant_value(unittest.TestCase):

    def test_get_constant_value(self):
        a = tensor.stack(1,2,3)
        assert get_constant_value(a[0])==1
        assert get_constant_value(a[1])==2
        assert get_constant_value(a[2])==3

        b = tensor.iscalar()
        a = tensor.stack(b,2,3)
        self.assertRaises(TypeError, get_constant_value, a[0])
        assert get_constant_value(a[1])==2
        assert get_constant_value(a[2])==3

        # For now get_constant_value goes through only MakeVector and Join of
        # scalars.
        v = tensor.ivector()
        a = tensor.stack(v,2,3)
        self.assertRaises(TypeError, get_constant_value, a[0])
        self.assertRaises(TypeError, get_constant_value, a[1])
        self.assertRaises(TypeError, get_constant_value, a[2])

        # Test the case SubTensor(Shape(v)) when the dimensions
        # is broadcastable.
        v = tensor.row()
        assert get_constant_value(v.shape[0])==1

    def test_subtensor_of_constant(self):
        c = constant(numpy.random.rand(5))
        for i in range(c.value.shape[0]):
            assert get_constant_value(c[i]) == c.value[i]
        c = constant(numpy.random.rand(5,5))
        for i in range(c.value.shape[0]):
            for j in range(c.value.shape[1]):
                assert get_constant_value(c[i,j]) == c.value[i,j]

if __name__ == '__main__':
    if 1:
        unittest.main()
    else:
        testcase =  T_Join_and_Split

        suite = unittest.TestLoader()
        suite = suite.loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(suite)
