import itertools
import logging
import operator
import StringIO
import sys
import unittest
import warnings
from copy import copy, deepcopy
# Import builtin min to be able to use it after importing the tensor version.
import __builtin__
builtin_min = __builtin__.min

from nose.plugins.skip import SkipTest
import numpy
from numpy.testing import dec
from numpy.testing.noseclasses import KnownFailureTest

import theano
from theano import compile, config, function, gof, tensor, shared
from theano.compile.mode import get_default_mode
from theano.gof.python25 import any, all, combinations
from theano.tensor import (_shared, wvector, bvector, autocast_float_as,
        argmin, max_and_argmax, cscalar, Subtensor, ctensor3, join,
        horizontal_stack, vertical_stack, argmax, get_vector_length,
        fscalar, zeros_like, sum, tensor3, vector, izip, add, addbroadcast,
        alloc, as_tensor_variable, tensor_from_scalar, ARange, autocast_float,
        basic, clip, constant, default, dot, inc_subtensor, set_subtensor,
        dmatrix, dscalar, dvector, eq, eye, fill, flatten, inverse_permutation,
        tensor4, permute_row_elements, Flatten, fmatrix, fscalars, grad,
        inplace, iscalar, matrix, minimum, matrices, maximum, mul, neq,
        Reshape, row, scalar, scalars, second, smallest, stack, sub, Tensor,
        tensor_copy, tensordot, tensordot_grad,  TensorType, unbroadcast,
        var, value, Join, shape, MaxAndArgmax, lscalar, zvector, exp,
        get_constant_value, ivector, reshape, scalar_from_tensor, scal,
        iscalars, arange,  dscalars, fvector, imatrix, numeric_grad,
        opt, ComplexError, TensorDot, lvector, true_div, max, min, Split, roll,
        tile)
from theano.tests import unittest_tools as utt


imported_scipy_special = False
mode_no_scipy = get_default_mode()
try:
    import scipy.special
    imported_scipy_special = True
except ImportError:
    if config.mode == "FAST_COMPILE":
        mode_no_scipy = "FAST_RUN"
floatX = config.floatX

### seed random number generator so that unittests are deterministic ###
utt.seed_rng()


def inplace_func(inputs, outputs, mode=None, allow_input_downcast=False):
    if mode is None:
        mode = get_default_mode()
    return function(inputs, outputs,
            mode=mode,
            allow_input_downcast=allow_input_downcast,
            accept_inplace=True)


def eval_outputs(outputs):
    variables = inplace_func([], outputs)()
    if isinstance(variables, (tuple, list)) and len(variables) == 1:
        return variables[0]
    return variables


def get_numeric_subclasses(cls=numpy.number, ignore=None):
    """
    Return subclasses of `cls` in the numpy scalar hierarchy.

    We only return subclasses that correspond to unique data types.
    The hierarchy can be seen here:
        http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
    """
    if ignore is None:
        ignore = []
    rval = []
    dtype = numpy.dtype(cls)
    dtype_num = dtype.num
    if dtype_num not in ignore:
        # Safety check: we should be able to represent 0 with this data type.
        numpy.array(0, dtype=dtype)
        rval.append(cls)
        ignore.append(dtype_num)
    for sub in cls.__subclasses__():
        rval += [c for c in get_numeric_subclasses(sub, ignore=ignore)]
    return rval


def get_numeric_types(with_int=True, with_float=True, with_complex=False,
                      only_theano_types=True):
    """
    Return numpy numeric data types.

    :param with_int: Whether to include integer types.

    :param with_float: Whether to include floating point types.

    :param with_complex: Whether to include complex types.

    :param only_theano_types: If True, then numpy numeric data types that are
    not supported by Theano are ignored (i.e. those that are not declared in
    scalar/basic.py).

    :returns: A list of unique data type objects. Note that multiple data types
    may share the same string representation, but can be differentiated through
    their `num` attribute.

    Note that when `only_theano_types` is True we could simply return the list
    of types defined in the `scalar` module. However with this function we can
    test more unique dtype objects, and in the future we may use it to
    automatically detect new data types introduced in numpy.
    """
    if only_theano_types:
        theano_types = [d.dtype for d in theano.scalar.all_types]
    rval = []

    def is_within(cls1, cls2):
        # Return True if scalars defined from `cls1` are within the hierarchy
        # starting from `cls2`.
        # The third test below is to catch for instance the fact that
        # one can use ``dtype=numpy.number`` and obtain a float64 scalar, even
        # though `numpy.number` is not under `numpy.floating` in the class
        # hierarchy.
        return (cls1 is cls2 or
                issubclass(cls1, cls2) or
                isinstance(numpy.array([0], dtype=cls1)[0], cls2))

    for cls in get_numeric_subclasses():
        dtype = numpy.dtype(cls)
        if ((not with_complex and is_within(cls, numpy.complexfloating)) or
            (not with_int and is_within(cls, numpy.integer)) or
            (not with_float and is_within(cls, numpy.floating)) or
            (only_theano_types and dtype not in theano_types)):
            # Ignore this class.
            continue
        rval.append([str(dtype), dtype, dtype.num])
    # We sort it to be deterministic, then remove the string and num elements.
    return [x[1] for x in sorted(rval, key=str)]


def _numpy_checker(x, y):
    """
    Checks if x.data and y.data have the same contents.
    Used in DualLinker to compare C version with Python version.
    """
    x, y = x[0], y[0]
    if (x.dtype != y.dtype or x.shape != y.shape
        or numpy.any(numpy.abs(x - y) > 1e-10)):
        raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})


def safe_make_node(op, *inputs):
    """ Emulate the behaviour of make_node when op is a function.

    Normally op in an instead of the Op class.
    """
    node = op(*inputs)
    if isinstance(node, list):
        return node[0].owner
    else:
        return node.owner


def makeTester(name, op, expected, checks={}, good={}, bad_build={},
               bad_runtime={}, grad={}, mode=None, grad_rtol=None,
               eps=1e-10, skip=False):
    if grad is True:
        grad = good

    _op, _expected, _checks, _good = op, expected, checks, good
    _bad_build, _bad_runtime, _grad = bad_build, bad_runtime, grad
    _mode, _grad_rtol, _eps, skip_ = mode, grad_rtol, eps, skip

    class Checker(unittest.TestCase):

        op = staticmethod(_op)
        expected = staticmethod(_expected)
        checks = _checks
        good = _good
        bad_build = _bad_build
        bad_runtime = _bad_runtime
        grad = _grad
        mode = _mode
        skip = skip_

        def test_good(self):
            if skip:
                raise SkipTest(skip)
            for testname, inputs in self.good.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [value(input) for input in inputs]
                try:
                    #node = self.op.make_node(*inputrs)
                    node = safe_make_node(self.op, *inputrs)
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occurred while"
                            " making a node with inputs %s") % (
                                    self.op, testname, inputs)
                    exc.args += (err_msg,)
                    raise

                try:
                    f = inplace_func(inputrs, node.outputs, mode=mode)
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occurred while"
                        " trying to make a Function") % (self.op, testname)
                    exc.args += (err_msg,)
                    raise
                if (isinstance(self.expected, dict)
                        and testname in self.expected):
                    expecteds = self.expected[testname]
                    # with numpy version, when we print a number and read it
                    # back, we don't get exactly the same result #So we accept
                    # rounding error in that case.
                    eps = 5e-9
                else:
                    expecteds = self.expected(*inputs)
                    eps = 1e-10

                if any([i.dtype == 'float32' for i in inputs]):
                    eps = 8e-6  # 1e-6
                eps = numpy.max([eps, _eps])

                try:
                    variables = f(*inputs)
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occurred while calling"
                    " the Function on the inputs %s") % (
                            self.op, testname, inputs)
                    exc.args += (err_msg,)
                    raise

                if not isinstance(expecteds, (list, tuple)):
                    expecteds = (expecteds, )

                for i, (variable, expected) in enumerate(
                        izip(variables, expecteds)):
                    if (variable.dtype != expected.dtype
                            or variable.shape != expected.shape
                            or numpy.any(abs(variable - expected) > eps)):
                        self.fail(("Test %s::%s: Output %s gave the wrong"
                            " value. With inputs %s, expected %s (dtype %s),"
                            " got %s (dtype %s)."
                            " numpy.allclose returns %s %s") % (
                                self.op,
                                testname,
                                i,
                                inputs,
                                expected,
                                expected.dtype,
                                variable,
                                variable.dtype,
                                numpy.allclose(variable, expected, atol=eps),
                                numpy.allclose(variable, expected)))

                for description, check in self.checks.items():
                    if not check(inputs, variables):
                        self.fail(("Test %s::%s: Failed check: %s (inputs"
                            " were %s, outputs were %s)") % (
                                self.op, testname, description,
                                inputs, variables))

        def test_bad_build(self):
            if skip:
                raise SkipTest(skip)
            for testname, inputs in self.bad_build.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [value(input) for input in inputs]
                self.assertRaises(Exception,
                    safe_make_node, self.op, *inputrs)
                # The old error string was ("Test %s::%s: %s was successfully
                # instantiated on the following bad inputs: %s"
                # % (self.op, testname, node, inputs))

        def test_bad_runtime(self):
            if skip:
                raise SkipTest(skip)
            for testname, inputs in self.bad_runtime.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [value(input) for input in inputs]
                try:
                    node = safe_make_node(self.op, *inputrs)
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occurred while trying"
                        " to make a node with inputs %s") % (
                            self.op, testname, inputs)
                    exc.args += (err_msg,)
                    raise

                try:
                    f = inplace_func(inputrs, node.outputs, mode=mode)
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occurred while trying"
                        " to make a Function") % (self.op, testname)
                    exc.args += (err_msg,)
                    raise

                # Add tester return a ValueError. Should we catch only this
                # one?
                # TODO: test that only this one is raised and catch only this
                # one or the subset that get raised.
                self.assertRaises(Exception, f, *inputs)

        def test_grad(self):
            if skip:
                raise SkipTest(skip)
            # Disable old warning that may be triggered by this test.
            backup = config.warn.sum_div_dimshuffle_bug
            config.warn.sum_div_dimshuffle_bug = False
            try:
                for testname, inputs in self.grad.items():
                    inputs = [copy(input) for input in inputs]
                    inputrs = [value(input) for input in inputs]
                    try:
                        utt.verify_grad(self.op, inputs,
                                mode=self.mode,
                                rel_tol=_grad_rtol)
                    except Exception, exc:
                        err_msg = ("Test %s::%s: Error occurred while"
                            " computing the gradient on the following"
                            " inputs: %s" ) % (self.op, testname, inputs)
                        exc.args += (err_msg,)
                        raise
            finally:
                config.warn.sum_div_dimshuffle_bug = backup

    Checker.__name__ = name
    return Checker


def rand(*shape):
    r = numpy.asarray(numpy.random.rand(*shape), dtype=config.floatX)
    return r * 2 - 1


def randint(*shape):
    return numpy.random.random_integers(-5, 5, shape)


# XXX: this so-called complex random array as all-zero imaginary parts
def randcomplex(*shape):
    r = numpy.asarray(numpy.random.rand(*shape), dtype=config.floatX)
    return numpy.complex128(2 * r - 1)


def randint_nonzero(*shape):
    r = numpy.random.random_integers(-5, 4, shape)
    return r + (r == 0) * 5


def rand_ranged(min, max, shape):
    return numpy.asarray(numpy.random.rand(*shape) * (max - min) + min,
                         dtype=config.floatX)


def randint_ranged(min, max, shape):
    return numpy.random.random_integers(min, max, shape)


def randc128_ranged(min, max, shape):
    return numpy.asarray(numpy.random.rand(*shape) * (max - min) + min,
                         dtype='complex128')


def rand_of_dtype(shape, dtype):
    if 'int' in dtype:
        return randint(*shape).astype(dtype)
    elif 'float' in dtype:
        return rand(*shape).astype(dtype)
    elif 'complex' in dtype:
        return randcomplex(*shape).astype(dtype)
    else:
        raise TypeError()


def makeBroadcastTester(op, expected, checks={}, name=None, **kwargs):
    if name is None:
        name = str(op)
    # Here we ensure the test name matches the name of the variable defined in
    # this script. This is needed to properly identify the test e.g. with the
    # --with-id option of nosetests, or simply to rerun a specific test that
    # failed.
    capitalize = False
    if name.startswith('Elemwise{') and name.endswith(',no_inplace}'):
        # For instance: Elemwise{add,no_inplace} -> Add
        name = name[9:-12]
        capitalize = True
    elif name.endswith('_inplace'):
        # For instance: sub_inplace -> SubInplace
        capitalize = True
    if capitalize:
        name = ''.join([x.capitalize() for x in name.split('_')])
    # Some tests specify a name that already ends with 'Tester', while in other
    # cases we need to add it manually.
    if not name.endswith('Tester'):
        name += "Tester"
    if kwargs.has_key('inplace'):
        if kwargs['inplace']:
            _expected = expected
            if not isinstance(_expected, dict):
                expected = lambda *inputs: numpy.array(_expected(*inputs),
                                                       dtype=inputs[0].dtype)

            def inplace_check(inputs, outputs):
                # this used to be inputs[0] is output[0]
                # I changed it so that it was easier to satisfy by the
                # DebugMode
                return numpy.all(inputs[0] == outputs[0])

            checks = dict(checks, inplace_check=inplace_check)
        del kwargs['inplace']
    return makeTester(name, op, expected, checks, **kwargs)


_good_broadcast_binary_normal = dict(same_shapes=(rand(2, 3), rand(2, 3)),
                                     not_same_dimensions=(rand(2, 2), rand(2)),
                                     scalar=(rand(2, 3), rand(1, 1)),
                                     row=(rand(2, 3), rand(1, 3)),
                                     column=(rand(2, 3), rand(2, 1)),
                                     integers=(randint(2, 3), randint(2, 3)),
                                     dtype_mixup_1=(rand(2, 3), randint(2, 3)),
                                     dtype_mixup_2=(randint(2, 3), rand(2, 3)),
                                     complex1=(randcomplex(2, 3), randcomplex(2, 3)),
                                     complex2=(randcomplex(2, 3), rand(2, 3)),
                                     # Disabled as we test the case where we reuse the same output as the first inputs.
                                     # complex3=(rand(2,3),randcomplex(2,3)),
                                     empty=(numpy.asarray([]), numpy.asarray([1])),
                                     )

_bad_build_broadcast_binary_normal = dict()  # not_same_dimensions = (rand(2), rand(2, 2)))

_bad_runtime_broadcast_binary_normal = dict(bad_shapes=(rand(2, 3), rand(3, 2)),
                                            bad_row=(rand(2, 3), rand(1, 2)))

_grad_broadcast_binary_normal = dict(same_shapes=(rand(2, 3), rand(2, 3)),
                                     scalar=(rand(2, 3), rand(1, 1)),
                                     row=(rand(2, 3), rand(1, 3)),
                                     column=(rand(2, 3), rand(2, 1)),
                                     #This don't work as verify grad don't support that
                                     #empty=(numpy.asarray([]), numpy.asarray([1]))
                                     #complex1=(randcomplex(2,3),randcomplex(2,3)),
                                     #complex2=(randcomplex(2,3),rand(2,3)),
                                     # Disabled as we test the case where we reuse the same output as the first inputs.
                                     #complex3=(rand(2,3),randcomplex(2,3)),
                                     )


def check_floatX(inputs, rval):
    """
    :param inputs: Inputs to a function that returned `rval` with these inputs.

    :param rval: Value returned by a function with inputs set to `inputs`.

    :returns: Either `rval` unchanged, or `rval` cast in float32. The idea is
    that when a numpy function would have returned a float64, Theano may prefer
    to return a float32 instead when `config.cast_policy` is set to
    'numpy+floatX' and config.floatX to 'float32', and there was no float64
    input.
    """
    if (isinstance(rval, numpy.ndarray) and
        rval.dtype == 'float64' and
        config.cast_policy == 'numpy+floatX'
        and config.floatX == 'float32' and
        all(x.dtype != 'float64' for x in inputs)):
        # Then we expect float32 instead of float64.
        return rval.astype('float32')
    else:
        return rval


AddTester = makeBroadcastTester(op = add,
                                  expected = lambda *inputs: check_floatX(inputs, reduce(lambda x, y: x + y, inputs)),
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
                                  expected = lambda x, y: check_floatX((x, y), x - y),
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
                                  expected = lambda *inputs: check_floatX(inputs, numpy.maximum(*inputs)),
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
                                  expected = lambda *inputs: check_floatX(inputs, numpy.minimum(*inputs)),
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
                                  expected = lambda *inputs: check_floatX(inputs, reduce(lambda x, y: x * y, inputs)),
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

def copymod(dct, without=[], **kwargs):
    """Return dct but with the keys named by args removed, and with
    kwargs added.
    """
    rval = copy(dct)
    for a in without:
        if a in rval:
            del rval[a]
    for kw, val in kwargs.items():
        rval[kw] = val
    return rval

_good_broadcast_div_mod_normal_float_no_complex = dict(
    same_shapes=(rand(2, 3), rand(2, 3)),
    scalar=(rand(2, 3), rand(1, 1)),
    row=(rand(2, 3), rand(1, 3)),
    column=(rand(2, 3), rand(2, 1)),
    dtype_mixup_1=(rand(2, 3), randint_nonzero(2, 3)),
    dtype_mixup_2=(randint_nonzero(2, 3), rand(2, 3)),
    integer=(randint(2, 3), randint_nonzero(2, 3)),
    uinteger=(randint(2, 3).astype("uint8"),
              randint_nonzero(2, 3).astype("uint8")),
    # This empty2 doesn't work for some tests. I don't remember why
    #empty2=(numpy.asarray([0]), numpy.asarray([])),
    )

_good_broadcast_div_mod_normal_float_inplace = copymod(
    _good_broadcast_div_mod_normal_float_no_complex,
    empty1=(numpy.asarray([]), numpy.asarray([1])),
    complex1=(randcomplex(2, 3), randcomplex(2, 3)),
    complex2=(randcomplex(2, 3), rand(2, 3)),
    # Inplace on the first element. Must have the same type.
    #complex3=(rand(2, 3) ,randcomplex(2, 3)),
    )

_good_broadcast_div_mod_normal_float = copymod(
    _good_broadcast_div_mod_normal_float_inplace,
    empty2=(numpy.asarray([0]), numpy.asarray([]))
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
    # We raise the relative tolerance for the grad as there can be errors in
    # float32.
    # This is probably caused by our way of computing the gradient error.
    div_grad_rtol=0.025


def _numpy_true_div(x, y):
    """Performs true division, and cast the result in the type we expect.

    We define that function so we can use it in TrueDivTester.expected,
    because simply calling numpy.true_divide could cause a dtype mismatch.
    """
    out = numpy.true_divide(x, y)
    # Use floatX as the result of int / int
    if x.dtype in tensor.discrete_dtypes and y.dtype in tensor.discrete_dtypes:
        out = theano._asarray(out, dtype=config.floatX)
    return out

TrueDivTester = makeBroadcastTester(
        op=tensor.true_div,
        expected=_numpy_true_div,
        good=_good_broadcast_div_mod_normal_float,
        grad=_grad_broadcast_div_mod_normal,
        grad_rtol=div_grad_rtol,
        )

TrueDivInplaceTester = makeBroadcastTester(
        op=inplace.true_div_inplace,
        expected=_numpy_true_div,
        good=copymod(
            _good_broadcast_div_mod_normal_float_inplace,
            # The output is now in float, we cannot work inplace on an int.
            without=['integer', 'uinteger']),
        grad=_grad_broadcast_div_mod_normal,
        grad_rtol=div_grad_rtol,
        inplace=True)


CeilIntDivTester = makeBroadcastTester(
    op=tensor.ceil_intdiv,
    expected=lambda x, y: check_floatX((x, y), (x // y) + ((x % y) != 0)),
    good=_good_broadcast_div_mod_normal_float_no_complex,
    name='CeilIntDiv',
    # As we implement this function with neq, the gradient returned is always 0.
#    grad=_grad_broadcast_div_mod_normal,
#    grad_rtol=div_grad_rtol,
    )

ModTester = makeBroadcastTester(
    op=tensor.mod,
    expected=lambda x, y: numpy.asarray(
        x % y, dtype=theano.scalar.basic.upcast(x.dtype, y.dtype)),
    good=copymod(_good_broadcast_div_mod_normal_float,
                 ['complex1', 'complex2']),
    )


ModInplaceTester = makeBroadcastTester(
    op=inplace.mod_inplace,
    expected=lambda x, y: numpy.asarray(
        x % y, dtype=theano.scalar.basic.upcast(x.dtype, y.dtype)),
    good=copymod(_good_broadcast_div_mod_normal_float_inplace,
                 ["complex1", "complex2"]),
    inplace=True)

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

PowTester = makeBroadcastTester(
        op=pow,
        expected=lambda x, y: check_floatX((x, y), x ** y),
        good=_good_broadcast_pow_normal_float,
        grad= _grad_broadcast_pow_normal,
        name='Pow')

PowInplaceTester = makeBroadcastTester(op = inplace.pow_inplace,
                                       expected = lambda x, y: x ** y,
                                       good = _good_broadcast_pow_normal_float_pow,
                                       grad = _grad_broadcast_pow_normal,
                                       inplace = True)

#Those are corner case when rounding. Their is many rounding algo.
#c round() fct and numpy round are not the same!
corner_case = numpy.asarray(
        [-2.5, -2., -1.5, -1., -0.5, -.51, -.49, 0,
            0.49, 0.5, 0.9, 1, 1.5, 2, 2.5],
        dtype=floatX)

#we remove 0 here as the grad is not always computable numerically.
corner_case_grad = numpy.asarray(
        [-2.5, -2., -1.5, -1., -0.5, -.51, -.49,
            0.49, 0.5, 0.9, 1, 1.5, 2, 2.5],
        dtype=floatX)

_good_broadcast_unary_normal_float = dict(
        normal=[rand_ranged(-5, 5, (2, 3))],
        corner_case=[corner_case],
        complex=[randcomplex(2,3)],
        empty=[numpy.asarray([])])

_good_broadcast_unary_normal_float_no_empty = copymod(
        _good_broadcast_unary_normal_float,
        without=['empty'])

_good_broadcast_unary_normal_float_no_empty_no_complex = copymod(
        _good_broadcast_unary_normal_float_no_empty,
        without=['complex'])

_good_broadcast_unary_normal_float_no_complex = copymod(
        _good_broadcast_unary_normal_float,
        without=['complex'])

_good_broadcast_unary_normal = dict(
        normal=[numpy.asarray(rand_ranged(-5, 5, (2, 3)),dtype=config.floatX)],
        integers=[randint_ranged(-5, 5, (2, 3))],
        corner_case=[corner_case],
        complex=[randcomplex(2,3)],
        empty=[numpy.asarray([])],
        )

_good_broadcast_unary_normal_no_complex = dict(
        normal=[numpy.asarray(rand_ranged(-5, 5, (2, 3)), dtype=floatX)],
        integers=[randint_ranged(-5, 5, (2, 3))],
        corner_case=[corner_case],
        empty=[numpy.asarray([])],
        )

_grad_broadcast_unary_normal = dict(
        normal=[numpy.asarray(rand_ranged(-5, 5, (2, 3)), dtype=floatX)],
        corner_case = [corner_case_grad],
        #empty = [numpy.asarray([])] # XXX: should this be included?
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

NegTester = makeBroadcastTester(op = tensor.neg,
                                  expected = lambda x: -x,
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
NegInplaceTester = makeBroadcastTester(op = inplace.neg_inplace,
                                         expected = lambda x: -x,
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

SgnTester = makeBroadcastTester(op = tensor.sgn,
                                expected = numpy.sign,
                                good = _good_broadcast_unary_normal_no_complex,
                                grad = _grad_broadcast_unary_normal,)
SgnInplaceTester = makeBroadcastTester(op = inplace.sgn_inplace,
                                       expected = numpy.sign,
                                       good = _good_broadcast_unary_normal_no_complex,
                                       grad = _grad_broadcast_unary_normal,
                                       inplace = True)


IntDivTester = makeBroadcastTester(
    op=tensor.int_div,
    expected=lambda x, y: check_floatX((x, y), x // y),
    good=_good_broadcast_div_mod_normal_float,
    # I don't test the grad as the output is always an integer
    # (this is not a continuous output).
#    grad=_grad_broadcast_div_mod_normal,
    )


IntDivInplaceTester = makeBroadcastTester(
    op=inplace.int_div_inplace,
    expected=lambda x, y: check_floatX((x, y), x // y),
    good=_good_broadcast_div_mod_normal_float_inplace,
    # I don't test the grad as the output is always an integer
    # (this is not a continuous output).
#    grad=_grad_broadcast_div_mod_normal,
    inplace=True
    )


CeilTester = makeBroadcastTester(op=tensor.ceil,
        expected=lambda a: numpy.asarray(
            numpy.ceil(a),
            a.dtype),
        good=_good_broadcast_unary_normal_no_complex,
        grad=copymod(_grad_broadcast_unary_normal,
            without=['corner_case'],
            # corner_case includes ints where ceil is not differentiable
            extra=[numpy.asarray([-2.5, -1.5, -1.51, 0.49, .98, 1.02],
                dtype=floatX)]))

CeilInplaceTester = makeBroadcastTester(op=inplace.ceil_inplace,
        expected=lambda a: numpy.asarray(numpy.ceil(a), a.dtype),
        good=_good_broadcast_unary_normal_no_complex,
        # corner cases includes a lot of integers: points where Ceil is not
        # continuous (not differentiable)
        grad=copymod(_grad_broadcast_unary_normal,
            without=['corner_case'],
            # corner_case includes ints where ceil is not differentiable
            extra=[numpy.asarray([-2.5, -1.5, -1.51, 0.49, .98, 1.02],
                dtype=floatX)]),
        inplace=True)

FloorTester = makeBroadcastTester(op=tensor.floor,
        expected=lambda a: numpy.asarray(numpy.floor(a), a.dtype),
        good=_good_broadcast_unary_normal_no_complex,
        # XXX: why does grad of floor not give huge values at
        #      the integer points in the 'corner_case' in
        #      _grad_broadcast_unary_normal?  It seems this test should fail,
        #      yet it does not...
        grad=_grad_broadcast_unary_normal)

FloorInplaceTester = makeBroadcastTester(op=inplace.floor_inplace,
        expected=lambda a: numpy.asarray(numpy.floor(a), a.dtype),
        good=_good_broadcast_unary_normal_no_complex,
        grad=_grad_broadcast_unary_normal,
        inplace = True)

RoundHalfToEvenTester = makeBroadcastTester(op = tensor.round_half_to_even,
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
RoundHalfAwayFromZeroTester = makeBroadcastTester(op = tensor.round_half_away_from_zero,
                                  expected = theano.scalar.basic.round_half_away_from_zero_vec,
                                  good = _good_broadcast_unary_normal_float_no_empty_no_complex)#_good_broadcast_unary_normal_float)
RoundHalfAwayFromZeroInplaceTester = makeBroadcastTester(op = inplace.round_half_away_from_zero_inplace,
                                         expected = theano.scalar.basic.round_half_away_from_zero_vec,
                                         good = _good_broadcast_unary_normal_float_no_empty_no_complex,
                                         inplace = True)

SqrTester = makeBroadcastTester(op = tensor.sqr,
                                  expected = numpy.square,
                                  good = _good_broadcast_unary_normal,
                                  grad = _grad_broadcast_unary_normal)
SqrInplaceTester = makeBroadcastTester(op = inplace.sqr_inplace,
                                         expected = numpy.square,
                                         good = _good_broadcast_unary_normal,
                                         grad = _grad_broadcast_unary_normal,
                                         inplace = True)

ExpTester = makeBroadcastTester(op = tensor.exp,
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

LogTester = makeBroadcastTester(op = tensor.log,
                                  expected = numpy.log,
                                  good = _good_broadcast_unary_positive,
                                  grad = _grad_broadcast_unary_positive)
LogInplaceTester = makeBroadcastTester(op = inplace.log_inplace,
                                         expected = numpy.log,
                                         good = _good_broadcast_unary_positive,
                                         grad = _grad_broadcast_unary_positive,
                                         inplace = True)

Log2Tester = makeBroadcastTester(op = tensor.log2,
                                   expected = numpy.log2,
                                   good = _good_broadcast_unary_positive,
                                   grad = _grad_broadcast_unary_positive)
Log2InplaceTester = makeBroadcastTester(op = inplace.log2_inplace,
                                          expected = numpy.log2,
                                          good = _good_broadcast_unary_positive,
                                          grad = _grad_broadcast_unary_positive,
                                          inplace = True)

Log10Tester = makeBroadcastTester(op = tensor.log10,
                                   expected = numpy.log10,
                                   good = _good_broadcast_unary_positive,
                                   grad = _grad_broadcast_unary_positive)
Log10InplaceTester = makeBroadcastTester(op = inplace.log10_inplace,
                                          expected = numpy.log10,
                                          good = _good_broadcast_unary_positive,
                                          grad = _grad_broadcast_unary_positive,
                                          inplace = True)

Log1pTester = makeBroadcastTester(op = tensor.log1p,
                                  expected = numpy.log1p,
                                  good = _good_broadcast_unary_positive,
                                  grad = _grad_broadcast_unary_positive)
Log1pInplaceTester = makeBroadcastTester(op = inplace.log1p_inplace,
                                         expected = numpy.log1p,
                                         good = _good_broadcast_unary_positive,
                                         grad = _grad_broadcast_unary_positive,
                                         inplace = True)


SqrtTester = makeBroadcastTester(op = tensor.sqrt,
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

_good_broadcast_unary_arccos = dict(normal = (rand_ranged(-1.+1e-7, 1.-1e-7, (2, 3)),),
                                  integers = (randint_ranged(-1.+1e-7, 1-1e-7, (2, 3)),),
                                  complex = (randc128_ranged(-1.+1e-7, 1-1e-7, (2, 3)),),
                                  empty = (numpy.asarray([]),),)

_grad_broadcast_unary_arccos = dict(normal = (rand_ranged(-1.+1e-7, 1-1e-7, (2, 3)),),
                                  #complex = (randc128_ranged(-1000, 1000, (2, 3)),),
                                  #empty = (numpy.asarray([]),),
                                  )


SinTester = makeBroadcastTester(op = tensor.sin,
                                  expected = numpy.sin,
                                  good = _good_broadcast_unary_wide,
                                  grad = _grad_broadcast_unary_wide)
SinInplaceTester = makeBroadcastTester(op = inplace.sin_inplace,
                                         expected = numpy.sin,
                                         good = _good_broadcast_unary_wide,
                                         grad = _grad_broadcast_unary_wide,
                                         inplace = True)

CosTester = makeBroadcastTester(op = tensor.cos,
                                  expected = numpy.cos,
                                  good = _good_broadcast_unary_wide,
                                  grad = _grad_broadcast_unary_wide)
CosInplaceTester = makeBroadcastTester(op = inplace.cos_inplace,
                                         expected = numpy.cos,
                                         good = _good_broadcast_unary_wide,
                                         grad = _grad_broadcast_unary_wide,
                                         inplace = True)
ArccosTester = makeBroadcastTester(op = tensor.arccos,
                                  expected = numpy.arccos,
                                  good = _good_broadcast_unary_arccos,
                                  grad = _grad_broadcast_unary_arccos)
ArccosInplaceTester = makeBroadcastTester(op = inplace.arccos_inplace,
                                         expected = numpy.arccos,
                                         good = _good_broadcast_unary_arccos,
                                         grad = _grad_broadcast_unary_arccos,
                                         inplace = True)

tan_grad_rtol = None
if config.floatX=='float32':
#We raise the relative tolerence for the grad as their is error in float32
#This is probably caused by our way of computing the gradient error.
    tan_grad_rtol = 0.052
TanTester = makeBroadcastTester(op = tensor.tan,
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


CoshTester = makeBroadcastTester(op = tensor.cosh,
                                   expected = numpy.cosh,
                                   good = _good_broadcast_unary_normal,
                                   grad = _grad_broadcast_unary_normal)
CoshInplaceTester = makeBroadcastTester(op = inplace.cosh_inplace,
                                          expected = numpy.cosh,
                                          good = _good_broadcast_unary_normal,
                                          grad = _grad_broadcast_unary_normal,
                                          inplace = True)

SinhTester = makeBroadcastTester(op = tensor.sinh,
                                   expected = numpy.sinh,
                                   good = _good_broadcast_unary_normal,
                                   grad = _grad_broadcast_unary_normal)
SinhInplaceTester = makeBroadcastTester(op = inplace.sinh_inplace,
                                          expected = numpy.sinh,
                                          good = _good_broadcast_unary_normal,
                                          grad = _grad_broadcast_unary_normal,
                                          inplace = True)

TanhTester = makeBroadcastTester(op = tensor.tanh,
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

# We can't test it if scipy is not installed!
# Precomputing the result is brittle(it have been broken!)
# As if we do any modification to random number here,
# The input random number will change and the output!
if imported_scipy_special:
    expected_erf = scipy.special.erf
    expected_erfc = scipy.special.erfc
    skip_scipy = False
else:
    expected_erf = []
    expected_erfc = []
    skip_scipy = "scipy is not present"

ErfTester = makeBroadcastTester(op = tensor.erf,
                                expected = expected_erf,
                                good = _good_broadcast_unary_normal,
                                grad = _grad_broadcast_unary_normal,
                                eps = 2e-10,
                                mode = mode_no_scipy,
                                skip = skip_scipy)
ErfInplaceTester = makeBroadcastTester(op = inplace.erf_inplace,
                                       expected = expected_erf,
                                       good = _good_broadcast_unary_normal_no_int,
                                       grad = _grad_broadcast_unary_normal,
                                       mode = mode_no_scipy,
                                       eps = 2e-10,
                                       inplace = True,
                                       skip = skip_scipy)

ErfcTester = makeBroadcastTester(op = tensor.erfc,
                                 expected = expected_erfc,
                                 good = _good_broadcast_unary_normal_no_int_no_complex,
                                 grad = _grad_broadcast_unary_normal,
                                 eps = 2e-10,
                                 mode = mode_no_scipy,
                                 skip = skip_scipy)
ErfcInplaceTester = makeBroadcastTester(op = inplace.erfc_inplace,
                                        expected = expected_erfc,
                                        good = _good_broadcast_unary_normal_no_int_no_complex,
                                        grad = _grad_broadcast_unary_normal,
                                        eps = 2e-10,
                                        mode = mode_no_scipy,
                                        inplace = True,
                                        skip = skip_scipy)

ZerosLikeTester = makeBroadcastTester(
        op=tensor.zeros_like,
        expected=numpy.zeros_like,
        good=_good_broadcast_unary_normal,
        grad=_grad_broadcast_unary_normal,
        name='ZerosLike')

OnesLikeTester = makeBroadcastTester(
        op=tensor.ones_like,
        expected=numpy.ones_like,
        good=_good_broadcast_unary_normal,
        grad=_grad_broadcast_unary_normal,
        name='OnesLike')

DotTester = makeTester(name = 'DotTester',
                        op = dot,
                        expected = lambda x, y: numpy.dot(x, y),
                        checks = {},
                        good = dict(correct1 = (rand(5, 7), rand(7, 5)),
                                    correct2 = (rand(5, 7), rand(7, 9)),
                                    correct3 = (rand(5, 7), rand(7)),
                                    correct4 = (rand(5), rand(5, 7)),
                                    mixed1 = (rand(5).astype('float32'),
                                        rand(5, 7)),
                                    mixed2 = (rand(5).astype('float64'),
                                        rand(5, 7)),
                                    complex1 = (randcomplex(5, 7), randcomplex(7)),
                                    complex2 = (rand(5, 7), randcomplex(7)),
                                    complex3 = (randcomplex(5, 7), rand(7)),
                                    empty1 = (numpy.asarray([]),numpy.asarray([])),
                                    empty2 = (rand(5,0),rand(0,2)),
                                    empty3 = (rand(0,5),rand(5,0)),
                                    ),
                        bad_build = dict(),
                        bad_runtime = dict(bad1 = (rand(5, 7), rand(5, 7)),
                                           bad2 = (rand(5, 7), rand(8, 3))))

def _numpy_second(x, y):
    return numpy.broadcast_arrays(x, y)[1]

ALL_DTYPES = ('int8', 'int16', 'int32', 'int64',
              'float32', 'float64', 'complex64', 'complex128')
REAL_DTYPES = ALL_DTYPES[:-2]
COMPLEX_DTYPES = ALL_DTYPES[-2:]

def multi_dtype_checks(shape1, shape2, dtypes=ALL_DTYPES, nameprefix=''):
    for dtype1, dtype2 in combinations(dtypes, 2):
        name1 = '%s_%s_%s' % (nameprefix, dtype1, dtype2)
        name2 = '%s_%s_%s' % (nameprefix, dtype2, dtype1)
        obj1 = rand_of_dtype(shape1, dtype1)
        obj2 = rand_of_dtype(shape2, dtype2)
        yield (name1, (obj1, obj2))
        yield (name2, (obj2, obj1))

def multi_dtype_cast_checks(shape, dtypes=ALL_DTYPES, nameprefix=''):
    for dtype1, dtype2 in combinations(dtypes, 2):
        name1 = '%s_%s_%s' % (nameprefix, dtype1, dtype2)
        name2 = '%s_%s_%s' % (nameprefix, dtype2, dtype1)
        obj1 = rand_of_dtype(shape, dtype1)
        obj2 = rand_of_dtype(shape, dtype2)
        yield (name1, (obj1, dtype2))
        yield (name2, (obj2, dtype1))

SecondBroadcastTester = makeTester(
                            name='SecondBroadcastTester',
                            op=second,
                            expected=_numpy_second,
                            good=dict(itertools.chain(
                                multi_dtype_checks((4, 5), (5,)),
                                multi_dtype_checks((2, 3, 2), (3, 2)),
                                multi_dtype_checks((2, 3, 2), (2,)),
                            )),
                            # I can't think of any way to make this fail at
                            # build time
                            # Just some simple smoke tests
                            bad_runtime=dict(
                                fail1=(rand(5, 4), rand(5)),
                                fail2=(rand(3, 2, 3), rand(6, 9)),
                                fail3=(randint(6, 2, 9), rand(3, 2)),
                            )
                        )

# We exclude local_fill_to_alloc because it optimizes the "second" node
# away from the graph.
SecondSameRankTester = makeTester(
                            name='SecondSameRankTester',
                            op=second,
                            expected=_numpy_second,
                            good=dict(itertools.chain(
                                multi_dtype_checks((4, 5), (4, 5)),
                                multi_dtype_checks((1, 2), (3, 2)),
                                multi_dtype_checks((3, 2), (1, 2)),
                            )),
                            # These sizes are not broadcastable to one another
                            # and SHOULD raise an error, but currently don't.
                            bad_runtime=dict(itertools.chain(
                                multi_dtype_checks((4, 5), (5, 4)),
                                multi_dtype_checks((1, 5), (5, 4)),
                            )),
                            mode=get_default_mode().excluding('local_fill_to_alloc')
                        )

### Alloc
AllocTester = makeBroadcastTester(
        name = 'AllocTester',
        op = alloc,
        expected = (lambda x, *shp: numpy.zeros(shp, dtype=x.dtype) + x),
        good = dict(
            correct01 = (rand(), numpy.int32(7)),
            correct01_bcast = (rand(1), numpy.int32(7)),
            correct02 = (rand(), numpy.int32(4), numpy.int32(7)),
            correct12 = (rand(7), numpy.int32(4), numpy.int32(7)),
            correct13 = (rand(7), numpy.int32(2), numpy.int32(4), numpy.int32(7)),
            correct23 = (rand(4,7), numpy.int32(2), numpy.int32(4), numpy.int32(7)),
            ),
        bad_runtime = dict(
            bad_shape12 = (rand(7), numpy.int32(7), numpy.int32(5)),
            too_big32 = (rand(6,2,4), numpy.int32(6), numpy.int32(2)),
            too_big32b = (rand(6,2,4), numpy.int32(2), numpy.int32(4)),
            ),
        )

# Since not all inputs of Alloc are differentiable, we need different testers
s1, s2, s3 = randint_ranged(1, 13, (3,))
# alloc a scalar into a vector
Alloc01GradTester = makeBroadcastTester(
        name = 'Alloc01GradTester',
        #op = (lambda self, x: alloc(x, s1)),
        op = (lambda x: alloc(x, s1)),
        expected = (lambda x: numpy.zeros((s1,), dtype=x.dtype) + x),
        grad = dict(
            x1 = (rand(),),
            x2 = (rand(),),
            x3 = (rand(),),
            ),
        )

# alloc a vector into a tensor3
Alloc13GradTester = makeBroadcastTester(
        name = 'Alloc13GradTester',
        #op = (lambda self, x: alloc(x, s1, s2, s3)),
        op = (lambda x: alloc(x, s1, s2, s3)),
        expected = (lambda x: numpy.zeros((s1, s2, s3), dtype=x.dtype) + x),
        grad = dict(
            x1 = (rand(s3),),
            x2 = (rand(s3),),
            x3 = (rand(s3),),
            ),
        )

def test_eye():
    def check(dtype, N, M_=None, k=0):
        # Theano does not accept None as a tensor.
        # So we must use a real value.
        M = M_
        # Currently DebugMode does not support None as inputs even if this is
        # allowed.
        if M is None and theano.config.mode in ['DebugMode', 'DEBUG_MODE']:
            M = N
        N_symb = tensor.iscalar()
        M_symb = tensor.iscalar()
        k_symb = tensor.iscalar()
        f = function([N_symb, M_symb, k_symb],
                     eye(N_symb, M_symb, k_symb, dtype=dtype))
        result = f(N, M, k)
        assert numpy.allclose(result, numpy.eye(N, M_, k, dtype=dtype))
        assert result.dtype == numpy.dtype(dtype)
    for dtype in ALL_DTYPES:
        yield check, dtype, 3
        # M != N, k = 0
        yield check, dtype, 3, 5
        yield check, dtype, 5, 3
        # N == M, k != 0
        yield check, dtype, 3, 3, 1
        yield check, dtype, 3, 3, -1
        # N < M, k != 0
        yield check, dtype, 3, 5, 1
        yield check, dtype, 3, 5, -1
        # N > M, k != 0
        yield check, dtype, 5, 3, 1
        yield check, dtype, 5, 3, -1

def test_identity():
    def check(dtype):
        obj = rand_of_dtype((2,), dtype)
        sym = tensor.vector(dtype=dtype)
        f = function([sym], tensor_copy(sym))
        assert numpy.all(obj == f(obj))
        assert obj.dtype == f(obj).dtype
        topo = f.maker.env.toposort()
        assert len(topo)==1
        if theano.config.mode != 'FAST_COMPILE':
            assert isinstance(topo[0].op, theano.compile.function_module.DeepCopyOp)

    for dtype in ALL_DTYPES:
        yield check, dtype

class CastTester(unittest.TestCase):
    def test_good_between_real_types(self):
        expected = lambda x, y: x.astype(y),
        good = itertools.chain(
                    multi_dtype_cast_checks((2,), dtypes=REAL_DTYPES),
                    # Casts from foo to foo
                    [('%s_%s' % (rand_of_dtype((2,), dtype), dtype),
                      (rand_of_dtype((2,), dtype), dtype))
                     for dtype in ALL_DTYPES])
        for testname, (obj, dtype) in good:
            inp = tensor.vector(dtype=obj.dtype)
            out = tensor.cast(inp, dtype=dtype)
            f = function([inp], out)
            assert f(obj).dtype == numpy.dtype(dtype)

    def test_cast_from_real_to_complex(self):
        for real_dtype in REAL_DTYPES:
            for complex_dtype in COMPLEX_DTYPES:
                inp = tensor.vector(dtype=real_dtype)
                out = tensor.cast(inp, dtype=complex_dtype)
                f = function([inp], out)
                obj = rand_of_dtype((2, ), real_dtype)
                assert f(obj).dtype == numpy.dtype(complex_dtype)

    def test_cast_from_complex_to_real_raises_error(self):
        for real_dtype in REAL_DTYPES:
            for complex_dtype in COMPLEX_DTYPES:
                inp = tensor.vector(dtype=real_dtype)
                self.assertRaises(TypeError, tensor.cast(inp, dtype=complex_dtype))

ClipTester = makeTester(name='ClipTester',
                        op=clip,
                        expected=lambda x, y, z: numpy.clip(x, y, z),
                        good = dict(correct1=((5 * rand(5, 5)).astype('float32'),
                                          numpy.array(-1, dtype='float32'),
                                          numpy.array(1, dtype='float32')),
                                    correct2=((5 * rand(5, 5)).astype('float64'),
                                          numpy.array(-1, dtype='float64'),
                                          numpy.array(1, dtype='float64')),
                                    correct3=(randint(5, 5).astype('int8'),
                                          numpy.array(-1, dtype='int8'),
                                          numpy.array(1, dtype='int8')),
                                    correct4=(randint(5, 5).astype('int16'),
                                          numpy.array(-1, dtype='int16'),
                                          numpy.array(1, dtype='int16')),
                                    correct5=(randint(5, 5).astype('int32'),
                                          numpy.array(-1, dtype='int32'),
                                          numpy.array(1, dtype='int32')),
                                    correct6=(randint(5, 5).astype('int64'),
                                          numpy.array(-1, dtype='int64'),
                                          numpy.array(1, dtype='int64')),
                                    # min > max. messed up behaviour, but
                                    # should be same as NumPy's
                                    correct7=((5 * rand(5, 5)).astype('float64'),
                                          numpy.array(1, dtype='float64'),
                                          numpy.array(-1, dtype='float64')))
                       )
                        # I can't think of any way to make this fail at runtime

class T_Clip(unittest.TestCase):
    def test_complex_value(self):
        for dtype in ['complex64', 'complex128']:
            a = tensor.vector(dtype=dtype)
            b = tensor.scalar()
            c = tensor.scalar()
            self.assertRaises(TypeError, clip, a, b, c)

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
    # numpy.max don't like empty ndarray.
    if a.size == b.size == 0:
        return True
    if numpy.max(abs_rel_err) >= eps:
        if _approx_eq.debug:
            print a, b
        return False
    return  True
_approx_eq.debug = 0


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


def test_nan_inf_constant_signature():
    # Test that the signature of a constant tensor containing NaN and Inf
    # values is correct.
    test_constants = [
            [numpy.nan, numpy.inf, 0, 1],
            [numpy.nan, numpy.inf, -numpy.inf, 1],
            [0, numpy.inf, -numpy.inf, 1],
            [0, 3, -numpy.inf, 1],
            [0, 3, numpy.inf, 1],
            [numpy.nan, 3, 4, 1],
            [0, 3, 4, 1],
            numpy.nan,
            numpy.inf,
            -numpy.inf,
            0,
            1,
            ]
    n = len(test_constants)
    # We verify that signatures of two rows i, j in the matrix above are
    # equal if and only if i == j.
    for i in xrange(n):
        for j in xrange(n):
            x = constant(test_constants[i])
            y = constant(test_constants[j])
            assert (x.signature() == y.signature()) == (i == j)

    # Also test that nan !=0 and nan != nan.
    x = tensor.scalar()
    mode = get_default_mode()
    if isinstance(mode, theano.compile.debugmode.DebugMode):
        # Disable the check preventing usage of NaN / Inf values.
        # We first do a copy of the mode to avoid side effects on other tests.
        mode = copy(mode)
        mode.check_isfinite = False
    f = theano.function([x], eq(x, numpy.nan), mode=mode)

    assert f(0) == 0
    assert f(numpy.nan) == 0


class T_Shape(unittest.TestCase):
    def test_basic0(self):
        s = shape(numpy.ones((5, 3)))
        self.assertTrue((eval_outputs([s]) == [5, 3]).all())
    def test_basic1(self):
        s = shape(numpy.ones((2)))
        self.assertTrue((eval_outputs([s]) == [2]).all())
    def test_basic2(self):
        s = shape(numpy.ones((5, 3, 10)))
        self.assertTrue((eval_outputs([s]) == [5, 3, 10]).all())


class T_max_and_argmax(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test0(self):
        n = as_tensor_variable(5.0)
        v, i = eval_outputs(max_and_argmax(n))
        self.assertTrue(v == 5.0)
        self.assertTrue(i == 0)
        assert i.dtype == 'int64'
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v) == 0
        v = eval_outputs(max_and_argmax(n)[1].shape)
        assert len(v) == 0

    def test1(self):
        n = as_tensor_variable([1, 2, 3, 2, -6])
        v, i = eval_outputs(max_and_argmax(n))
        self.assertTrue(v == 3)
        self.assertTrue(i == 2)
        assert i.dtype == 'int64'
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v) == 0

    def test2(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for (axis, np_axis)  in [(-1, -1), (0, 0), (1, 1), (None, None),
                                 ([0, 1], None), ([1, 0], None)]:
            v, i = eval_outputs(max_and_argmax(n, axis))
            assert i.dtype == 'int64'
            self.assertTrue(numpy.all(v == numpy.max(data, np_axis)))
            self.assertTrue(numpy.all(i == numpy.argmax(data, np_axis)))
            v_shape = eval_outputs(max_and_argmax(n, axis)[0].shape)
            assert tuple(v_shape) == numpy.max(data, np_axis).shape

    def test2_invalid(self):
        n = as_tensor_variable(rand(2, 3))
        # Silence expected error messages
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.level
        _logger.setLevel(logging.CRITICAL)
        try:
            try:
                eval_outputs(max_and_argmax(n, 3))
                assert False
            except ValueError, e:
                pass
        finally:
            _logger.setLevel(oldlevel)

    def test2_invalid_neg(self):
        n = as_tensor_variable(rand(2, 3))
        old_stderr = sys.stderr
        sys.stderr = StringIO.StringIO()
        try:
            try:
                eval_outputs(max_and_argmax(n, -3))
                assert False
            except ValueError, e:
                pass
        finally:
            sys.stderr = old_stderr

    def test2_valid_neg(self):
        n = as_tensor_variable(rand(2, 3))
        v, i = eval_outputs(max_and_argmax(n, -1))
        assert i.dtype == 'int64'
        self.assertTrue(v.shape == (2,))
        self.assertTrue(i.shape == (2,))
        self.assertTrue(numpy.all(v == numpy.max(n.value, -1)))
        self.assertTrue(numpy.all(i == numpy.argmax(n.value, -1)))
        v, i = eval_outputs(max_and_argmax(n, -2))
        assert i.dtype == 'int64'
        self.assertTrue(v.shape == (3,))
        self.assertTrue(i.shape == (3,))
        self.assertTrue(numpy.all(v == numpy.max(n.value, -2)))
        self.assertTrue(numpy.all(i == numpy.argmax(n.value, -2)))
        v = eval_outputs(max_and_argmax(n, -1)[0].shape)
        assert v == (2)
        v = eval_outputs(max_and_argmax(n, -2)[0].shape)
        assert v == (3)

    def test3(self):
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for (axis, np_axis)  in [(-1, -1), (0, 0), (1, 1), (None, None),
                                 ([0, 1, 2], None), ([1, 2, 0], None)]:
            v, i = eval_outputs(max_and_argmax(n, axis))
            assert i.dtype == 'int64'
            self.assertTrue(numpy.all(v == numpy.max(data, np_axis)))
            self.assertTrue(numpy.all(i == numpy.argmax(data, np_axis)))
            v = eval_outputs(max_and_argmax(n, axis)[0].shape)
            assert tuple(v) == numpy.max(data, np_axis).shape

    def test_grad(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        def safe_verify_grad(func, data):
            """
            Wrapper around 'verify_grad' that picks a proper value for epsilon.

            This is needed because 'verify_grad' may fail when its epsilon is
            too large, due to the fact the argmax is not continuous.
            We make sure epsilon is less than the minimum absolute value found
            in the matrix of pairwise differences between all elements in the
            data. This way, the argmax will not change when adding epsilon.
            """
            # 'data' is a one-element list.
            data_tensor, = data
            # Flatten it into a 1D vector.
            data_vector = data_tensor.flatten()
            # Compute pairwise absolute differences.
            diff = numpy.abs(data_vector.reshape((-1, 1)) - data_vector)
            # Alter the diagonal to avoid a zero minimum.
            for i in xrange(len(diff)):
                diff[i, i] = 1
            # Find an appropriate epsilon.
            eps = builtin_min(numeric_grad.type_eps[config.floatX],
                              diff.min() / 2)
            # Run gradient verification.
            utt.verify_grad(func, data, eps=eps)

        def check_grad_max(data, max_grad_data, axis=None):
            """
            Why this is needed? verify_grad is not enough?
            """
            # This works only for axis in [0, None].
            assert axis in [0, None]
            z = numpy.zeros_like(data)
            z = z.flatten()
            argmax = numpy.argmax(data, axis=axis)
            if argmax.ndim == 0:
                z[argmax] += 1
            else:
                for id, v in enumerate(argmax):
                    z[v * numpy.prod(data.shape[data.ndim - 1:axis:-1])
                      + id] += 1

            z = z.reshape(data.shape)
            assert numpy.all(max_grad_data == z)

        for axis in (-1, 0, 1, None):
            for j in xrange(2):
                safe_verify_grad(lambda v: max_and_argmax(v, axis=axis)[j],
                                [data])
                if axis != 1:
                    safe_verify_grad(lambda v: max_and_argmax(v.flatten(),
                                                             axis=axis)[j],
                                    [data])
            if axis in (0, None):
                check_grad_max(data, eval_outputs(grad(
                    max_and_argmax(n, axis=axis)[0].sum(), n)), axis=axis)
            check_grad_max(data, eval_outputs(grad(
                max_and_argmax(n.flatten())[0], n)))

        # Test 3d inner dimensions
        data = rand(3, 4, 5)

        for i in [0, 1, 2]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[1], [data])

        # Test 4d inner dimensions
        data = rand(2, 3, 4, 5)

        for i in [0, 1, 2, 3]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[1], [data])


class T_argmin_argmax(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test_scalar(self):
        for fct in [argmin, argmax]:
            n = as_tensor_variable(5.0)
            i = eval_outputs(fct(n))
            self.assertTrue(i == 0)
            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test_list(self):
        n = as_tensor_variable([1, 2, 3, 2, -6])
        i = eval_outputs(argmin(n))
        self.assertTrue(i == 4)
        v = eval_outputs(argmin(n).shape)
        assert len(v) == 0

        n = as_tensor_variable([1, 2, 3, 2, -6])
        i = eval_outputs(argmax(n))
        self.assertTrue(i == 2)
        v = eval_outputs(argmax(n).shape)
        assert len(v) == 0

    def test2(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for fct, nfct in [(argmax, numpy.argmax), (argmin, numpy.argmin)]:
            for (axis, np_axis)  in [(-1, -1), (0, 0), (1, 1), (None, None),
                                     ([0, 1], None), ([1, 0], None)]:
                v = eval_outputs(fct(n, axis))
                self.assertTrue(numpy.all(v == nfct(data, np_axis)))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_invalid(self):
        for fct, nfct in [(argmax, numpy.argmax), (argmin, numpy.argmin)]:
            n = as_tensor_variable(rand(2, 3))
            # Silence expected error messages
            _logger = logging.getLogger('theano.gof.opt')
            oldlevel = _logger.level
            _logger.setLevel(logging.CRITICAL)
            try:
                try:
                    eval_outputs(fct(n, 3))
                    assert False
                except ValueError, e:
                    pass
            finally:
                _logger.setLevel(oldlevel)

    def test2_invalid_neg(self):
        for fct, nfct in [(argmax, numpy.argmax), (argmin, numpy.argmin)]:
            n = as_tensor_variable(rand(2, 3))
            old_stderr = sys.stderr
            sys.stderr = StringIO.StringIO()
            try:
                try:
                    eval_outputs(fct(n, -3))
                    assert False
                except ValueError, e:
                    pass
            finally:
                sys.stderr = old_stderr

    def test2_valid_neg(self):
        for fct, nfct in [(argmax, numpy.argmax), (argmin, numpy.argmin)]:
            n = as_tensor_variable(rand(2, 3))
            i = eval_outputs(fct(n, -1))
            self.assertTrue(i.shape == (2,))
            self.assertTrue(numpy.all(i == nfct(n.value, -1)))
            i = eval_outputs(fct(n, -2))
            self.assertTrue(i.shape == (3,))
            self.assertTrue(numpy.all(i == nfct(n.value, -2)))

            v = eval_outputs(fct(n, -1).shape)
            assert v == (2)
            v = eval_outputs(fct(n, -2).shape)
            assert v == (3)

    def test3(self):
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(argmax, numpy.argmax), (argmin, numpy.argmin)]:
            for (axis, np_axis)  in [(-1, -1), (0, 0), (1, 1), (2, 2),
                                     (None, None), ([0, 1, 2], None),
                                     ([1, 0, 2], None)]:
                v = eval_outputs(fct(n, axis))
                self.assertTrue(numpy.all(v == nfct(data, np_axis)))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test_grad_argmin(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        #test grad of argmin
        utt.verify_grad(lambda v: argmin(v, axis=-1), [data])

        utt.verify_grad(lambda v: argmin(v, axis=[0]), [data])

        utt.verify_grad(lambda v: argmin(v, axis=[1]), [data])

        utt.verify_grad(lambda v: argmin(v.flatten()), [data])

        try:
            grad(argmin(n, axis=-1), n)
            raise Exception('Expected an error')
        except TypeError:
            pass

    def test_grad_argmax(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        #test grad of argmax
        utt.verify_grad(lambda v: argmax(v, axis=-1), [data])

        utt.verify_grad(lambda v: argmax(v, axis=[0]), [data])

        utt.verify_grad(lambda v: argmax(v, axis=[1]), [data])

        utt.verify_grad(lambda v: argmax(v.flatten()), [data])

        try:
            grad(argmax(n, axis=-1), n)
            raise Exception('Expected an error')
        except TypeError:
            pass


class T_min_max(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test_scalar(self):
        for fct in [max, min]:
            n = as_tensor_variable(5.0)
            v = eval_outputs(fct(n))
            self.assertTrue(v == 5.0)

            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test_list(self):
        for fct, nfct in [(max, numpy.max), (min, numpy.min)]:
            n = as_tensor_variable([1, 2, 3, 2, -6])
            v = eval_outputs([fct(n)])
            self.assertTrue(v == nfct(n.value))

            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test2(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, numpy.max), (min, numpy.min)]:
            for (axis, np_axis)  in [(-1, -1), (0, 0), (1, 1), (None, None),
                                     ([0, 1], None), ([1, 0], None)]:
                v = eval_outputs(fct(n, axis))
                self.assertTrue(numpy.all(v == nfct(data, np_axis)))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_invalid(self):
        for fct in [max, min]:
            n = as_tensor_variable(rand(2, 3))
            # Silence expected error messages
            _logger = logging.getLogger('theano.gof.opt')
            oldlevel = _logger.level
            _logger.setLevel(logging.CRITICAL)
            try:
                try:
                    eval_outputs(fct(n, 3))
                    assert False
                except ValueError, e:
                    pass
            finally:
                _logger.setLevel(oldlevel)

    def test2_invalid_neg(self):
        for fct in [max, min]:
            n = as_tensor_variable(rand(2, 3))
            old_stderr = sys.stderr
            sys.stderr = StringIO.StringIO()
            try:
                try:
                    eval_outputs(fct(n, -3))
                    assert False
                except ValueError, e:
                    pass
            finally:
                sys.stderr = old_stderr

    def test2_valid_neg(self):
        for fct, nfct in [(max, numpy.max), (min, numpy.min)]:
            n = as_tensor_variable(rand(2, 3))
            v = eval_outputs(fct(n, -1))
            self.assertTrue(v.shape == (2,))
            self.assertTrue(numpy.all(v == nfct(n.value, -1)))
            v = eval_outputs(fct(n, -2))
            self.assertTrue(v.shape == (3,))
            self.assertTrue(numpy.all(v == nfct(n.value, -2)))

            v = eval_outputs(fct(n, -1).shape)
            assert v == (2)
            v = eval_outputs(fct(n, -2).shape)
            assert v == (3)

    def test3(self):
        # Test with 1 axis or all axis out of 3 dims
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, numpy.max), (min, numpy.min)]:
            for (axis, np_axis)  in [(-1, -1), (0, 0), (1, 1), (2, 2),
                                     (None, None), ([0, 1, 2], None),
                                     ([1, 0, 2], None)]:
                v = eval_outputs(fct(n, axis))
                self.assertTrue(numpy.all(v == nfct(data, np_axis)))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test3b(self):
        # Test with 2 axis out of 3 dims
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, numpy.max), (min, numpy.min)]:
            for axis in [[0, 1], [1, 2], [0, 2]]:
                v = eval_outputs(fct(n, axis))
                np_v = nfct(nfct(data, axis[1]), axis[0])
                self.assertTrue(numpy.all(v == np_v))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == np_v.shape

    def test_grad_max(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        def check_grad_max(data, max_grad_data, axis=None):
            #This work only for axis in [0,None]
            assert axis in [0, None]
            z = numpy.zeros_like(data)
            z = z.flatten()
            argmax = numpy.argmax(data, axis=axis)
            if argmax.ndim == 0:
                z[numpy.argmax(data, axis=axis)] += 1
            else:
                for id, v in enumerate(argmax):
                    z[v * numpy.prod(data.shape[data.ndim - 1:axis:-1])
                      + id] += 1

            z = z.reshape(data.shape)
            assert numpy.all(max_grad_data == z)

        #test grad of max
        #axis is the last one
        utt.verify_grad(lambda v: max(v, axis=-1), [data])

        utt.verify_grad(lambda v: max(v, axis=[0]), [data])
        check_grad_max(data, eval_outputs(grad(max(n, axis=0).sum(), n)),
                       axis=0)

        utt.verify_grad(lambda v: max(v, axis=[1]), [data])
        #check_grad_max(data,eval_outputs(grad(max(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: max(v.flatten()), [data])
        check_grad_max(data, eval_outputs(grad(max(n.flatten()), n)))

    def test_grad_min(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        def check_grad_min(data, min_grad_data, axis=None):
            #This work only for axis in [0, None]
            assert axis in [0, None]
            z = numpy.zeros_like(data)
            z = z.flatten()
            argmin = numpy.argmin(data, axis=axis)
            if argmin.ndim == 0:
                z[numpy.argmin(data, axis=axis)] += 1
            else:
                for id, v in enumerate(argmin):
                    z[v * numpy.prod(data.shape[data.ndim - 1:axis:-1])
                      + id] += 1

            z = z.reshape(data.shape)
            assert numpy.all(min_grad_data == z)

        #test grad of min
        #axis is the last one
        utt.verify_grad(lambda v: min(v, axis=-1), [data])

        utt.verify_grad(lambda v: min(v, axis=[0]), [data])
        check_grad_min(data, eval_outputs(grad(min(n, axis=0).sum(), n)),
                       axis=0)

        utt.verify_grad(lambda v: min(v, axis=[1]), [data])
        #check_grad_min(data,eval_outputs(grad(min(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: min(v.flatten()), [data])
        check_grad_min(data, eval_outputs(grad(min(n.flatten()), n)))

    def _grad_list(self):
        """
        Test the gradient when we have multiple axis at the same time.

        This not implemented, so we disable the test. See ticket:
        http://www.assembla.com/spaces/theano/tickets/511
        """
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for fct in [max_and_argmax, max, min]:
            utt.verify_grad(lambda v: fct(v, axis=[0, 1]), [data])
        #check_grad_max(data, eval_outputs(grad(max_and_argmax(n,
        #axis=1)[0], n)),axis=1)


class T_subtensor(unittest.TestCase):
    """
    This is build in a way that allow to reuse it to test the
    equivalent gpu op.
    """
    def __init__(self, name, shared=_shared,
                 sub=tensor.Subtensor,
                 inc_sub=tensor.IncSubtensor,
                 adv_sub1=tensor.AdvancedSubtensor1,
                 adv_incsub1=tensor.AdvancedIncSubtensor1,
                 mode=None,
                 dtype=theano.config.floatX,
                 ignore_topo=(theano.compile.function_module.DeepCopyOp)):
        self.shared = shared
        self.sub = sub
        self.inc_sub = inc_sub
        self.adv_sub1 = adv_sub1
        self.adv_incsub1 = adv_incsub1
        if mode is None:
            mode = theano.compile.mode.get_default_mode()
        self.mode = mode
        self.dtype = dtype
        self.ignore_topo = ignore_topo
        self.fast_compile = theano.config.mode == 'FAST_COMPILE'
        return super(T_subtensor, self).__init__(name)

    def setUp(self):
        Subtensor.debug = False
        utt.seed_rng()

    def eval_output_and_check(self, t, list=False):
        f = inplace_func([], t, mode=self.mode)
        topo = f.maker.env.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
        assert len(topo_)==1
        if not list:
            assert isinstance(topo_[0].op, self.sub)
        else:
            assert isinstance(topo_[0].op, self.adv_sub1)
        tval = f()
        return tval

    def test0_err_invalid(self):
        #it is impossible to retrieve a view of a 0-d tensor
        n = self.shared(numpy.ones((), dtype=self.dtype))
        try:
            t = n[0]
        except ValueError, e:
            self.assertTrue(hasattr(e,'subtensor_invalid'))
            return
        self.fail()

    def test1_err_bounds(self):
        n = self.shared(numpy.ones(3, dtype=self.dtype))
        ctv_backup = config.compute_test_value
        config.compute_test_value = 'off'
        try:
            t = n[7]
        finally:
            config.compute_test_value = ctv_backup
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        # Silence expected error messages
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.level
        _logger.setLevel(logging.CRITICAL)
        try:
            try:
                self.eval_output_and_check(t)
                assert 0
            except Exception, e:
                if e[0] != 'index out of bounds':
                    raise
        finally:
            _logger.setLevel(oldlevel)
    def test1_err_subslice(self):
        n = self.shared(numpy.ones(3, dtype=self.dtype))
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
        n = self.shared(numpy.ones(3, dtype=self.dtype)*5)
        t = n[0:2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        f = inplace_func([], t, mode=self.mode)
        topo = f.maker.env.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
        assert len(topo_)==1
        assert isinstance(topo_[0].op, self.sub)
        tval = f()
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(tval[1] == 5.0)

    def test2_ok_range_finite(self):
        n = self.shared(numpy.ones((3,4), dtype=self.dtype)*5)
        # Also check negative index
        for idx in [(slice(0,2),3),((slice(0,2),-1)),(slice(0,2),-4)]:
            t = n[idx]#l]#0:2,3]
            self.assertTrue(isinstance(t.owner.op, Subtensor))
            f = inplace_func([], t, mode=self.mode)
            topo = f.maker.env.toposort()
            topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
            assert len(topo_)==1
            assert isinstance(topo_[0].op, self.sub)
            tval = f()
            self.assertTrue(tval.shape == (2,))
            self.assertTrue(numpy.allclose(tval, n.get_value()[idx]))

    def test1_0_dims(self):
        n = self.shared(numpy.ones((), dtype=self.dtype))
        t = theano.tensor.Subtensor([])(n)
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        mode = self.mode
        self.mode = mode.excluding("local_useless_subtensor")
        try:
            self.eval_output_and_check(t)
        finally:
            self.mode = mode

    def test1_err_invalid(self):
        n = self.shared(numpy.ones(1, dtype=self.dtype))
        try:
            t = n[0,0]
        except ValueError, e:
            self.assertTrue(hasattr(e,'subtensor_invalid'))
            return
        self.fail()

    def test1_ok_elem(self):
        n = self.shared(numpy.ones(1, dtype=self.dtype)*5)
        t = n[0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        f = inplace_func([], t, mode=self.mode)
        topo = f.maker.env.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
        assert len(topo_)==1
        assert isinstance(topo_[0].op, self.sub)
        tval = f()
        self.assertTrue(tval.shape == ())
        self.assertTrue(tval == 5.0)
    def test1_ok_range_infinite(self):
        #Subtensor.debug = True
        n = self.shared(numpy.ones(3, dtype=self.dtype)*5)
        t = n[1:]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        f = inplace_func([], t, mode=self.mode)
        topo = f.maker.env.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
        assert len(topo_)==1
        assert isinstance(topo_[0].op, self.sub)
        tval = f()
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(tval[1] == 5.0)

    def test1_ok_strided(self):
        n = self.shared(numpy.ones(5, dtype=self.dtype)*5)
        t = n[1::2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(tval[1] == 5.0)

        t = n[0:-1:2] #0 to 1 from the end stepping by 2
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(tval[1] == 5.0)

    def test2_err_bounds0(self):
        n = self.shared(numpy.ones((2,3), dtype=self.dtype)*5)
        ctv_backup = config.compute_test_value
        config.compute_test_value = 'off'
        try:
            for idx in [(0,4),(0,-4)]:
                t = n[idx]
                self.assertTrue(isinstance(t.owner.op, Subtensor))
                # Silence expected warnings
                _logger = logging.getLogger('theano.gof.opt')
                oldlevel = _logger.level
                _logger.setLevel(logging.CRITICAL)
                try:
                    try:
                        tval = self.eval_output_and_check([t])
                        assert 0
                    except IndexError, e:
                        pass
                finally:
                    _logger.setLevel(oldlevel)
        finally:
            config.compute_test_value = ctv_backup

    def test2_err_bounds1(self):
        n = self.shared((numpy.ones((2,3), dtype=self.dtype)*5))
        t = n[4:5,2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        old_stderr = sys.stderr
        sys.stderr = StringIO.StringIO()
        try:
            try:
                tval = self.eval_output_and_check([t])
            except Exception, e:
                if e[0] != 'index out of bounds':
                    raise
        finally:
            sys.stderr = old_stderr
    def test2_ok_elem(self):
        n = self.shared(numpy.asarray(range(6), dtype=self.dtype).reshape((2,3)))
        t = n[0,2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == ())
        self.assertTrue(numpy.all(tval == 2))
    def test2_ok_row(self):
        n = self.shared(numpy.asarray(range(6), dtype=self.dtype).reshape((2,3)))
        t = n[1]
        self.assertFalse(any(n.type.broadcastable))
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (3,))
        self.assertTrue(numpy.all(tval == [3,4,5]))

    def test2_ok_col(self):
        n = self.shared(numpy.ones((2,3), dtype=self.dtype)*5)
        t = n[:,0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        self.assertFalse(any(n.type.broadcastable))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(numpy.all(tval == 5.0))

    def test2_ok_rows_finite(self):
        n = self.shared(numpy.ones((4,3), dtype=self.dtype)*5)
        t = n[1:3,0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(numpy.all(tval == 5.0))

    def test2_ok_cols_infinite(self):
        n = self.shared(numpy.asarray(range(12), dtype=self.dtype).reshape((4,3)))
        t = n[1,2:]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (1,))
        self.assertTrue(numpy.all(tval == 5))

    def test2_ok_strided(self):
        n = self.shared(numpy.asarray(range(20), dtype=self.dtype).reshape((4,5)))
        t = n[1:4:2,1:5:2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,2))
        self.assertTrue(numpy.all(tval == [[6, 8],[16, 18]]))

    def test3_ok_mat(self):
        n = self.shared(numpy.asarray(range(24), dtype=self.dtype).reshape((2,3,4)))
        t = n[0,0,0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == ())
        self.assertTrue(numpy.all(tval == 0))

    def test_grad_1d(self):
        subi = 0
        data = numpy.asarray(rand(2,3), dtype=self.dtype)
        n = self.shared(data)
        z = scal.constant(subi)
        t = n[z:,z]
        gn = grad(sum(exp(t)), n)

        f = inplace_func([], gn, mode=self.mode)
        topo = f.maker.env.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
        if not self.fast_compile:
            assert len(topo_)==6
        assert numpy.sum([isinstance(node.op, self.inc_sub) for node in topo_])==1
        assert numpy.sum([isinstance(node.op, self.sub) for node in topo_])==1
        gval = f()

        good = numpy.zeros_like(data)
        good[subi:,subi] = numpy.exp(data[subi:,subi])
        self.assertTrue(numpy.allclose(gval, good), (gval, good))

    def test_grad_0d(self):
        data = numpy.asarray(rand(2,3), dtype=self.dtype)
        n = self.shared(data)
        t = n[1,0]
        gn = grad(sum(exp(t)), n)
        f = function([], gn, mode=self.mode)
        topo = f.maker.env.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
        if not self.fast_compile:
            assert len(topo_)==6
        assert numpy.sum([isinstance(node.op, self.inc_sub) for node in topo_])==1
        assert numpy.sum([isinstance(node.op, self.sub) for node in topo_])==1

        gval = f()
        good = numpy.zeros_like(data)
        good[1,0] = numpy.exp(data[1,0])
        self.assertTrue(numpy.allclose(gval, good), (gval, good))

    def test_ok_list(self):
        for data, idx in [(rand(4), [1,0]),
                          (rand(4,5), [2,3]),
                          (rand(4,2,3), [0,3]),
                          (rand(4,2,3), [3,3,1,1,2,2,0,0]),
                          (rand(4,2,3), [3,3,1,1,2,2,0,0,-1,-2,-3,-4]),
                          # Test 4 dims as gpu code use another algo in that case
                          # This new algo is not as much optimized for that case.
                          (rand(4,4,2,3), [3,3,1,1,2,2,0,0,-1,-2,-3,-4]),
                          # Test with TensorConstant index.
                          (rand(4,2,3), constant([3,3,1,1,2,2,0,0])),
                          ]:
            data = numpy.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[idx]

            # We test again AdvancedSubtensor1 as we transfer data to the cpu.
            self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))

            val = self.eval_output_and_check(t, list=True)
            if isinstance(idx, list):
                good = data[idx]
            else:
                good = data[idx.data]
            self.assertTrue(val.ndim == data.ndim)
            self.assertTrue(numpy.allclose(val, good), (val, good))

            # Test reuse of output memory
            if isinstance(self.adv_sub1,tensor.AdvancedSubtensor1):
                op = self.adv_sub1()
                # When idx is a TensorConstant.
                if hasattr(idx, "data"):
                    idx = idx.data
                test_out = [[None]]
                op.perform(None, [data, idx],test_out)
                out1 = test_out[0][0]
                op.perform(None, [data, idx],test_out)
                out2 = test_out[0][0]
                assert out1 is out2

    def test_err_invalid_list(self):
        n = self.shared(numpy.asarray(5, dtype=self.dtype))
        self.assertRaises(TypeError, n.__getitem__, [0,0])

    def test_err_invalid_2list(self):
        # TODO the error message is not clear
        n = self.shared(numpy.ones((3,3), dtype=self.dtype)*5)
        self.assertRaises(TypeError, n.__getitem__, ([0,0],[1,1]))

    def test_err_bound_list(self):
        n = self.shared(numpy.ones((2,3),dtype=self.dtype)*5)
        l = lvector()
        t = n[l]
        # We test again AdvancedSubtensor1 as we transfer data to the cpu.
        self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))

        f = function([l], t, mode=self.mode)
        topo = f.maker.env.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
        assert len(topo_)==1
        self.assertTrue(isinstance(topo_[0].op, self.adv_sub1))
        for shp in [[0,4],[0,-3], [-10]]:
            self.assertRaises(IndexError, f, shp)

    def test_adv_sub1_broadcast(self):
        ones = numpy.ones((1,3), dtype=self.dtype)
        n = self.shared(ones*5, broadcastable=(True, False))
        idx = tensor.lvector()
        t = n[idx]
        self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))

        f = function([idx], t, mode=self.mode)
        topo = f.maker.env.toposort()
        topo_ = [node for node in topo if not isinstance(node.op, self.ignore_topo)]
        assert len(topo_)==1
        self.assertTrue(isinstance(topo_[0].op, self.adv_sub1))
        self.assertTrue(numpy.allclose(f([0]),ones[0]*5))
        self.assertRaises(IndexError, f, [0,1])

    def test_shape_i_const(self):
        # Each axis is treated independently by shape_i/shape operators

        mode_opt = config.mode
        if mode_opt == 'FAST_COMPILE':
            mode_opt = 'FAST_RUN'
        mode_opt = compile.mode.get_mode(mode_opt)

        data = self.shared(numpy.array(numpy.arange(5),dtype=self.dtype))
        for start in [None]+ [-8,-5,-1,0,1,5,8]:
            outs   = []
            shapes = []
            for stop in [None] + [-8,-5,-1,0,1,5,8]:
                for step in [None]+[-3,-1,2]:
                    outs += [ data[start:stop:step].shape ]
                    shapes += [data.get_value(borrow=True)[start:stop:step].shape ]
            f = function([], outs, mode = mode_opt)
            t_shapes = f()
            for t_shape, shape in zip(t_shapes,shapes):
                assert numpy.all(t_shape == shape)
            assert tensor.Subtensor not in [ x.op for x in
                                           f.maker.env.toposort() ]

    def test_shape_i_scalar(self):
        # Each axis is treated independently by shape_i/shape operators

        mode_opt = config.mode
        if mode_opt == 'FAST_COMPILE':
            mode_opt = 'FAST_RUN'
        mode_opt = compile.mode.get_mode(mode_opt)
        v_data = numpy.array(numpy.arange(5), dtype=self.dtype)
        t_data = self.shared(v_data)
        start  = tensor.iscalar('b')
        stop   = tensor.iscalar('e')
        step   = tensor.iscalar('s')
        f = function([start,stop,step], t_data[start:stop:step].shape, mode = mode_opt)
        f2 = function([start,stop,step],t_data[start:stop:step])
        assert tensor.Subtensor not in [x.op for x in f.maker.env.toposort()]
        for start in [-8,-5,-4,-1,0,1,4,5,8]:
            for stop in [-8,-5,-4,-1,0,1,4,5,8]:
                for step in [-3,-1,2,5]:
                    assert numpy.all(
                            f(start,stop,step) == v_data[start:stop:step].shape)


    def test_slice_canonical_form_0(self):
        start  = tensor.iscalar('b')
        stop   = tensor.iscalar('e')
        step   = tensor.iscalar('s')
        length = tensor.iscalar('l')
        cnf = tensor.get_canonical_form_slice(slice(start,stop,step), length)
        f = function([start,stop,step, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1]) ])

        length = 5
        a = numpy.arange(length)
        for start in [ -8,-5,-4,-1,0,1,4,5,8]:
            for stop in  [ -8,-5,-4,-1,0,1,4,5,8]:
                for step in [-6,-3,-1,2,5]:
                    out = f(start,stop,step,length)
                    t_out = a[ out[0]:out[1]:out[2]][::out[3]]
                    v_out = a[start:stop:step]
                    assert numpy.all(t_out == v_out)
                    assert numpy.all(t_out.shape == v_out.shape)


    def test_slice_canonical_form_1(self):
        stop   = tensor.iscalar('e')
        step   = tensor.iscalar('s')
        length = tensor.iscalar('l')
        cnf = tensor.get_canonical_form_slice(slice(None,stop,step), length)
        f = function([stop,step, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1]) ])

        length = 5
        a = numpy.arange(length)
        for stop in  [ -8,-5,-4,-1,0,1,4,5,8]:
            for step in [-6,-3,-1,2,5]:
                out = f(stop,step,length)
                t_out = a[ out[0]:out[1]:out[2]][::out[3]]
                v_out = a[:stop:step]
                assert numpy.all(t_out == v_out)
                assert numpy.all(t_out.shape == v_out.shape)


    def test_slice_canonical_form_2(self):
        start  = tensor.iscalar('b')
        step   = tensor.iscalar('s')
        length = tensor.iscalar('l')
        cnf = tensor.get_canonical_form_slice(slice(start,None,step), length)
        f = function([start,step, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1]) ])

        length = 5
        a = numpy.arange(length)
        for start in [ -8,-5,-4,-1,0,1,4,5,8]:
            for step in [-6,-3,-1,2,5]:
                out = f(start,step,length)
                t_out = a[ out[0]:out[1]:out[2]][::out[3]]
                v_out = a[start:None:step]
                assert numpy.all(t_out == v_out)
                assert numpy.all(t_out.shape == v_out.shape)


    def test_slice_canonical_form_3(self):
        start  = tensor.iscalar('b')
        stop   = tensor.iscalar('e')
        length = tensor.iscalar('l')
        cnf = tensor.get_canonical_form_slice(slice(start,stop,None), length)
        f = function([start,stop, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1]) ])

        length = 5
        a = numpy.arange(length)
        for start in [ -8,-5,-4,-1,0,1,4,5,8]:
            for stop in  [ -8,-5,-4,-1,0,1,4,5,8]:
                out = f(start,stop,length)
                t_out = a[ out[0]:out[1]:out[2]][::out[3]]
                v_out = a[start:stop:None]
                assert numpy.all(t_out == v_out)
                assert numpy.all(t_out.shape == v_out.shape)

    def test_slice_canonical_form_4(self):
        step   = tensor.iscalar('s')
        length = tensor.iscalar('l')
        cnf = tensor.get_canonical_form_slice(slice(None,None,step), length)
        f = function([step, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1]) ])

        length = 5
        a = numpy.arange(length)
        for step in [-6,-3,-1,2,5]:
            out = f(step,length)
            t_out = a[ out[0]:out[1]:out[2]][::out[3]]
            v_out = a[None:None:step]
            assert numpy.all(t_out == v_out)
            assert numpy.all(t_out.shape == v_out.shape)


    def test_slice_canonical_form_5(self):
        start  = tensor.iscalar('b')
        length = tensor.iscalar('l')
        cnf = tensor.get_canonical_form_slice(slice(start,None,None), length)
        f = function([start, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1]) ])

        length = 5
        a = numpy.arange(length)
        for start in [ -8,-5,-4,-1,0,1,4,5,8]:
            out = f(start,length)
            t_out = a[ out[0]:out[1]:out[2]][::out[3]]
            v_out = a[start:None:None]
            assert numpy.all(t_out == v_out)
            assert numpy.all(t_out.shape == v_out.shape)

    def test_slice_canonical_form_6(self):
        stop   = tensor.iscalar('e')
        length = tensor.iscalar('l')
        cnf = tensor.get_canonical_form_slice(slice(None,stop,None), length)
        f = function([stop, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1]) ])

        length = 5
        a = numpy.arange(length)
        for stop in  [ -8,-5,-4,-1,0,1,4,5,8]:
            out = f(stop,length)
            t_out = a[ out[0]:out[1]:out[2]][::out[3]]
            v_out = a[None:stop:None]
            assert numpy.all(t_out == v_out)
            assert numpy.all(t_out.shape == v_out.shape)

    def grad_list_(self, idxs, data):
        n = self.shared(data)

        for idx in idxs:
            # Should stay on the cpu.
            idx_ = _shared(numpy.asarray(idx))
            t = n[idx_]
            gn = grad(sum(exp(t)), n)
            f = function([], [gn, gn.shape], mode=self.mode)
            topo = f.maker.env.toposort()
            if not self.fast_compile:
                assert any([isinstance(node.op, self.adv_incsub1) and node.op.inplace for node in topo])
            else:
                assert any([isinstance(node.op, self.adv_incsub1) for node in topo])
            assert any([isinstance(node.op, self.adv_sub1) for node in topo])
            gval, gshape = f()
            good = numpy.zeros_like(data)
            # good[idx] += numpy.exp(data[idx]) don't work when the same index is used many time
            for i in idx:
                good[i] += numpy.exp(data[i])
            self.assertTrue(gval.ndim == data.ndim)
            self.assertTrue(numpy.allclose(gval, good), (gval, good))
            self.assertTrue(numpy.allclose(gshape, data.shape))

            def fct(t):
                return sum(t[idx_])
            utt.verify_grad(fct, [data])

            # Test the grad of the grad (e.i. AdvancedIncSubtensor1.grad)
            def fct(t):
                return grad(sum(t[idx_]),t)
            utt.verify_grad(fct, [data])

            # Test shape of AdvancedIncSubtensor1 and AdvancedSubtensor1
            if idx is idxs[0]:
                f = function([], [gn.shape, n[idx_].shape], mode=self.mode)
                topo = f.maker.env.toposort()
                if not self.fast_compile:
                    self.assertTrue(not any([isinstance(node.op, self.adv_incsub1) for node in topo]))
                    self.assertTrue(not any([isinstance(node.op, self.adv_sub1) for node in topo]))
                f()

    def test_wrong_exception_regression(self):
        a = fscalar()
        b = fscalar()
        c = vector()
        try:
            c[a:b]
        except NotImplementedError:
            self.fail()
        except TypeError:
            pass
        try:
            c[a:]
        except NotImplementedError:
            self.fail()
        except TypeError:
            pass
        try:
            c[:b]
        except NotImplementedError:
            self.fail()
        except TypeError:
            pass

    def test_grad_list(self):
        data = rand(4)
        data = numpy.asarray(data, dtype=self.dtype)
        idxs = [[i] for i in range(data.shape[0])]
        for i in range(data.shape[0]):
            for j in range(0,data.shape[0],2):
                idxs.append([i,j,(i+1)%data.shape[0]])
        self.grad_list_(idxs, data)

        data = rand(4,3)
        data = numpy.asarray(data, dtype=self.dtype)
        self.grad_list_(idxs, data)

        data = rand(4,3,2)
        data = numpy.asarray(data, dtype=self.dtype)
        self.grad_list_(idxs, data)

    def test_shape_list(self):
        #TODO for all type of subtensor shape
        for data, idx in [(rand(4), [1,0]),
                          (rand(4,2), [2,3]),
                          (rand(4,2,3), [0,3]),
                          (rand(4,2,3), [3,3,1,2,2,]),
                          ]:
            data = numpy.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[idx]
            f = function([], t.shape, mode=None)
            val = f()
            self.assertTrue(numpy.allclose(val, data[idx].shape))

    def test_grad_advanced_inc_subtensor(self):
        def inc_slice(*s):
            def just_numeric_args(a,b):
                cost = (a[s] + b).sum()
                cost_wrt_a = grad(cost, a)
                cost_wrt_b = grad(cost, b)
                grads = cost_wrt_a.sum() + cost_wrt_b.sum()
                return grads
            return just_numeric_args

        # vector
        utt.verify_grad(
            inc_slice(slice(2, 4, None)),
            (numpy.asarray([0, 1, 2, 3, 4, 5.]), numpy.asarray([9, 9.]),))

        # matrix
        utt.verify_grad(
            inc_slice(slice(1, 2, None), slice(None, None, None)),
            (numpy.asarray([[0, 1], [2, 3], [4, 5.]]),
             numpy.asarray([[9, 9.]]),))

        #single element
        utt.verify_grad(
            inc_slice(2, 1),
            (numpy.asarray([[0, 1],[2, 3],[4, 5.]]), numpy.asarray(9.),))


class TestIncSubtensor1(unittest.TestCase):
    # test inc_subtensor
    # also tests set_subtensor

    def setUp(self):
        self.s = iscalar()
        self.v = fvector()
        self.m = dmatrix()
        self.t = ctensor3()

        self.adv1q = lvector() # advanced 1d query

    def test_cant_adv_idx_into_scalar(self):
        self.assertRaises(TypeError, lambda : self.s[self.adv1q])

    def test_index_into_vec_w_vec(self):
        a = self.v[self.adv1q]
        assert a.type == self.v.type

    def test_1d_set_adv_selection(self):
        a = set_subtensor(self.v[self.adv1q], self.v[self.adv1q])

        assert a.type == self.v.type

        #TODO: compile a function and verify that the subtensor is removed
        #      completely, because the whole expression is redundant.

        f = theano.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([.4, .9, .1], [1,2])
        assert numpy.allclose(aval, [.4, 0.9, 0.1])


    def test_1d_inc_adv_selection(self):
        a = inc_subtensor(self.v[self.adv1q], self.v[self.adv1q])

        assert a.type == self.v.type
        f = theano.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([.4, .9, .1], [1,2])
        assert numpy.allclose(aval, [.4, 1.8, 0.2])


    def test_1d_inc_adv_selection_w_broadcasting(self):
        a = inc_subtensor(self.v[self.adv1q], 3.0)

        assert a.type == self.v.type
        f = theano.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([.4, .9, .1], [1,2])
        assert numpy.allclose(aval, [.4, 3.9, 3.1])

    def test_assigning_matrix_to_vector_selection(self):
        self.assertRaises(TypeError,
                lambda : inc_subtensor(self.v[self.adv1q], fmatrix()))


class T_Join_and_Split(unittest.TestCase):
    """
    Split is tested by each verify_grad method.
    """
    def setUp(self):
        Join.debug = False
        utt.seed_rng()
        self.mode = theano.compile.get_default_mode().excluding(
            'constant_folding'
        )
        self.join_op = Join
        self.split_op = Split
        self.make_vector_op = opt.MakeVector
        self.floatX = config.floatX
        self.hide_error = theano.config.mode not in ['DebugMode',
                                                     'DEBUG_MODE',
                                                     'FAST_COMPILE']
        self.shared = shared

    def eval_outputs_and_check_join(self, outputs):
        f = theano.function([], outputs, self.mode)
        topo = f.maker.env.toposort()
        assert [True for node in topo if isinstance(node.op, self.join_op)]
        variables = f()
        if isinstance(variables, (tuple, list)) and len(variables) == 1:
            return variables[0]
        return variables

    def eval_outputs_and_check_vector(self, outputs,
                                      make_vector_op=None):
        if make_vector_op is None:
            make_vector_op = self.make_vector_op
        f = theano.function([], outputs, self.mode)
        topo = f.maker.env.toposort()
        assert [True for node in topo if isinstance(node.op, make_vector_op)]
        variables = f()
        if isinstance(variables, (tuple, list)) and len(variables) == 1:
            return variables[0]
        return variables

    def test_join_scalar(self):
        a = as_tensor_variable(1)
        b = as_tensor_variable(2)
        try:
            s = join(0, a, b)
        except TypeError:
            return
        self.fail()

    def test_stack_mixed_type_constants(self):
        # tested only on cpu as gpu support only float32
        a = as_tensor_variable(1)
        b = as_tensor_variable(2.0)
        c = tensor._shared(numpy.asarray(3.0, dtype=self.floatX))
        s = stack(a, b, c)
        want = numpy.array([1, 2, 3])
        out = self.eval_outputs_and_check_vector([s], opt.MakeVector)
        self.assertTrue((out == want).all())

    def test_stack_scalar(self):
        a = self.shared(numpy.asarray(1., dtype=self.floatX))
        b = as_tensor_variable(2.)
        c = as_tensor_variable(3.)
        s = stack(a, b, c)

        want = numpy.array([1, 2, 3])
        out = self.eval_outputs_and_check_vector([s])
        self.assertTrue((out == want).all())

    def test_stack_scalar_make_vector(self):
        """Test that calling stack() on scalars instantiates MakeVector,
        not Join. Test that the floatX dtype stay floatX, not downcasted
        to int64"""
        a = tensor.scalar('a', dtype=self.floatX)
        b = tensor.scalar('b', dtype=self.floatX)
        s = stack(a, b, a, b)
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        print val
        self.assertTrue(numpy.all(val == [1, 2, 1, 2]))
        topo = f.maker.env.toposort()
        assert len([n for n in topo if isinstance(n.op, opt.MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, self.join_op)]) == 0
        assert f.maker.env.outputs[0].dtype == self.floatX

    def test_stack_scalar_make_vector_dtype(self):
        '''Test that calling stack() on scalars instantiates MakeVector,
        event when the scalar don't have the same dtype.'''
        a = tensor.iscalar('a')
        b = tensor.lscalar('b')
        s = stack(a, b, a, b)
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        self.assertTrue(numpy.all(val == [1, 2, 1, 2]))
        topo = f.maker.env.toposort()
        assert len([n for n in topo if isinstance(n.op, opt.MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, self.join_op)]) == 0
        assert f.maker.env.outputs[0].dtype == 'int64'

    def test_stack_scalar_make_vector_constant(self):
        '''Test that calling stack() on scalars instantiates MakeVector,
        event when the scalar are simple int type.'''
        a = tensor.iscalar('a')
        b = tensor.lscalar('b')
        #test when the constant is the first element.
        #The first element is used in a special way
        s = stack(10, a, b, numpy.int8(3))
        f = function([a, b], s, mode=self.mode)
        val = f(1, 2)
        self.assertTrue(numpy.all(val == [10, 1, 2, 3]))
        topo = f.maker.env.toposort()
        assert len([n for n in topo if isinstance(n.op, opt.MakeVector)]) > 0
        assert len([n for n in topo if isinstance(n, self.join_op)]) == 0
        assert f.maker.env.outputs[0].dtype == 'int64'

    def test_join_concatenate_one_element(self):
        ''' Fast test of concatenate as this is an alias for join.
        also test that we remove the Join op if there is only 1 input'''
        m = tensor.fmatrix()
        c = tensor.concatenate([m])
        f = theano.function(inputs=[m], outputs=[c],
                            mode=self.mode.including('local_join_1'))
        topo = f.maker.env.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, theano.compile.DeepCopyOp)

    def test_join_vector(self):
        a = self.shared(numpy.array([1, 2, 3], dtype=self.floatX))
        b = as_tensor_variable(numpy.array([7, 8, 9], dtype=self.floatX))

        s = join(0, a, b)
        want = numpy.array([1, 2, 3, 7, 8, 9])
        out = self.eval_outputs_and_check_join([s])
        self.assertTrue((out == want).all())

    def test_roll(self):

        # Test simple 1D example
        a = self.shared(numpy.array([1, 2, 3, 4, 5, 6], dtype=self.floatX))
        b = roll(a, 2)
        want = numpy.array([5, 6, 1, 2, 3, 4])
        out = theano.function([], b)()

        assert (out == want).all()

        # Test simple 1D example with explicit 0 axis
        b = roll(a, -1, 0)
        want = numpy.array([2, 3, 4, 5, 6, 1])
        out = theano.function([], b)()

        assert (out == want).all()

        # Test 2D example - ensure that behavior matches numpy.roll behavior
        a = self.shared(numpy.arange(21).reshape((3, 7)).astype(self.floatX))
        b = roll(a, -2, 1)

        want = numpy.roll(a.get_value(borrow=True), -2, 1)
        out = theano.function([], b)()

        assert (out == want).all()

        # Test rolling on axis 0
        want = numpy.roll(a.get_value(borrow=True), -2, 0)
        b = roll(a, -2, 0)
        out = theano.function([], b)()

        assert (out == want).all()

        # Test rolling on default axis with ndim > 1
        want = numpy.roll(a.get_value(borrow=True), 2)
        b = roll(a, 2)
        out = theano.function([], b)()

        assert (out == want).all()


    def test_stack_vector(self):
        a = self.shared(numpy.array([1, 2, 3], dtype=self.floatX))
        b = as_tensor_variable(numpy.array([7, 8, 9], dtype=self.floatX))

        s = stack(a, b)
        want = numpy.array([[1, 2, 3], [7, 8, 9]])
        out = self.eval_outputs_and_check_join([s])
        self.assertTrue((out == want).all())

    def test_join_matrix0(self):
        a = self.shared(numpy.array([[1, 2, 3], [4, 5, 6]],
                                    dtype=self.floatX))
        b = as_tensor_variable(numpy.array([[7, 8, 9]], dtype=self.floatX))
        s = join(0, a, b)

        want = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = self.eval_outputs_and_check_join([s])
        self.assertTrue((out == want).all())

    def test_join_matrix1(self):
        av = numpy.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
        bv = numpy.array([[7], [8]], dtype='float32')
        a = self.shared(av)
        b = as_tensor_variable(bv)
        s = join(1, a, b)
        want = numpy.array([[1, 2, 3, 7], [4, 5, 6, 8]], dtype='float32')
        out = self.eval_outputs_and_check_join([s])
        self.assertTrue((out == want).all())

#        assert tensor.grad(join(1,a,b), a
        utt.verify_grad(lambda a, b: join(1, a, b), [av, bv],
                        eps=1.0e-4, rel_tol=1.0e-3)

    def test_join_matrix1_using_vertical_stack(self):
        a = self.shared(numpy.array([[1, 2, 3], [4, 5, 6]], dtype=self.floatX))
        b = as_tensor_variable(numpy.array([[7, 8, 9]], dtype=self.floatX))
        c = as_tensor_variable(numpy.array([[9, 8, 7]], dtype=self.floatX))
        s = vertical_stack(a, b, c)

        want = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [9, 8, 7]])
        out = self.eval_outputs_and_check_join([s])
        self.assertTrue((out == want).all())

    def test_join_matrix1_using_horizontal_stack(self):
        av = numpy.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
        bv = numpy.array([[7], [8]], dtype='float32')
        cv = numpy.array([[3, 2, 1], [6, 5, 4]], dtype='float32')
        a = self.shared(av)
        b = as_tensor_variable(bv)
        c = as_tensor_variable(cv)
        s = horizontal_stack(a, b, c)
        want = numpy.array([[1, 2, 3, 7, 3, 2, 1], [4, 5, 6, 8, 6, 5, 4]],
                           dtype='float32')
        out = self.eval_outputs_and_check_join([s])
        self.assertTrue((out == want).all())

        utt.verify_grad(lambda a, b: join(1, a, b), [av, bv],
                        eps=1.0e-4, rel_tol=1.0e-3)

    def test_join_matrixV(self):
        """variable join axis"""
        v = numpy.array([[1., 2., 3.], [4., 5., 6.]], dtype=self.floatX)
        a = self.shared(v.copy())
        b = as_tensor_variable(v.copy())
        ax = lscalar()
        s = join(ax, a, b)

        f = inplace_func([ax], [s], mode=self.mode)
        topo = f.maker.env.toposort()
        assert [True for node in topo if isinstance(node.op, self.join_op)]

        want = numpy.array([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])
        got = f(0)
        self.assertTrue((got == want).all(), (got, want))

        want = numpy.array([[1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]])
        got = f(1)
        self.assertTrue((got == want).all(), (got, want))

        utt.verify_grad(lambda a, b: join(0, a, b), [v, 2 * v])
        utt.verify_grad(lambda a, b: join(1, a, b), [v, 2 * v])

    def test_vector_len(self):
        x = lscalar('x')
        y = dscalar('y')

        triple = as_tensor_variable((x, y, 9.0))
        assert 3 == get_vector_length(triple)

        a, b, c = triple
        f = function([x, y], [b, c, a], mode=self.mode)
        topo = f.maker.env.toposort()
        assert [True for node in topo if isinstance(node.op, opt.MakeVector)]

        assert numpy.allclose(f(4, 5), [5, 9, 4])

    def test_broadcastable_flag_assignment_mixed_otheraxes(self):
        """
        Test that the broadcastable flags for the output of
        a join operation on non-join axes are True if one or
        more inputs is broadcastable on that dimension.
        """
        rng = numpy.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(1, 4, 1).astype(self.floatX)
        b_val = rng.rand(1, 3, 1).astype(self.floatX)

        a = self.shared(a_val, broadcastable=(False, False, True))
        b = self.shared(b_val, broadcastable=(True, False, True))
        c = self.join_op()(1, a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

        # Opt can remplace the int by a Theano constant
        c = self.join_op()(theano.tensor.constant(1), a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

        # In case futur opt insert other useless stuff
        c = self.join_op()(theano.tensor.cast(theano.tensor.constant(1),
                                              dtype="int32"),
                 a, b)
        assert c.type.broadcastable[0] and c.type.broadcastable[2]
        assert not c.type.broadcastable[1]

        f = function([], c, mode=self.mode)
        topo = f.maker.env.toposort()
        assert [True for node in topo if isinstance(node.op, self.join_op)]

        f()
        utt.verify_grad((lambda a, b: join(1, a, b)), [a_val, b_val], rng=rng)

        # Should raise an error if dimension 0 does not match
        a.set_value(rng.rand(2, 4, 1).astype(self.floatX))
        self.assertRaises(ValueError, f)

    def test_broadcastable_flag_assignment_mixed_thisaxes(self):
        """
        Test that the broadcastable flag of the join axis
        is False when some inputs are broadcastable on that
        dimension.
        """
        rng = numpy.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(2, 4, 1).astype(self.floatX)
        b_val = rng.rand(1, 4, 1).astype(self.floatX)

        a = self.shared(a_val, broadcastable=(False, False, True))
        b = self.shared(b_val, broadcastable=(True, False, True))
        c = self.join_op()(0, a, b)
        assert not c.type.broadcastable[0]

        f = function([], c, mode=self.mode)
        topo = f.maker.env.toposort()
        assert [True for node in topo if isinstance(node.op, self.join_op)]

        f()
        utt.verify_grad((lambda a, b: join(0, a, b)), [a_val, b_val], rng=rng)
        # Should raise an error if b_val.shape[0] is not 1
        # We can't set the value|
        self.assertRaises(TypeError, b.set_value,
                          rng.rand(3, 4, 1).astype(self.floatX))
        a = TensorType(dtype=self.floatX, broadcastable=[0, 0, 1])()
        b = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1])()
        c = join(0, a, b)
        f = function([a, b], c, mode=self.mode)
        bad_b_val = rng.rand(3, 4, 1).astype(self.floatX)
        self.assertRaises(TypeError, f, a_val, bad_b_val)

    def test_broadcastable_flags_all_broadcastable_on_joinaxis(self):
        """
        Test that joining together several inputs which are all
        broadcastable on the join dimension results in the output
        being non-broadcastable on the join dimension.
        """
        rng = numpy.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(1, 4, 1).astype(self.floatX)
        b_val = rng.rand(1, 4, 1).astype(self.floatX)

        a = self.shared(a_val, broadcastable=(True, False, True))
        b = self.shared(b_val, broadcastable=(True, False, True))
        c = self.join_op()(0, a, b)
        assert not c.type.broadcastable[0]

        f = function([], c, mode=self.mode)
        topo = f.maker.env.toposort()
        assert [True for node in topo if isinstance(node.op, self.join_op)]

        f()
        utt.verify_grad((lambda a, b: join(0, a, b)), [a_val, b_val], rng=rng)

    def test_broadcastable_single_input_broadcastable_dimension(self):
        """
        Test that all broadcastable flags are preserved by a
        single-input join.
        """
        rng = numpy.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(1, 4, 1).astype(self.floatX)
        a = self.shared(a_val, broadcastable=(True, False, True))
        b = self.join_op()(0, a)
        assert b.type.broadcastable[0]
        assert b.type.broadcastable[2]
        assert not b.type.broadcastable[1]

        f = function([], b, mode=self.mode)
        topo = f.maker.env.toposort()
        if theano.config.mode != 'FAST_COMPILE':
            assert not [True for node in topo if isinstance(node.op, self.join_op)]

        f()
        utt.verify_grad((lambda a: join(0, a)), [a_val], rng=rng)
        # Should raise an error if length of dimension 0 is not 1
        self.assertRaises(TypeError, a.set_value,
                          rng.rand(2, 4, 1).astype(self.floatX))
        #self.assertRaises(TypeError, f, bad_a_val)

    def test_broadcastable_flags_many_dims_and_inputs(self):
        """
        Test that the right broadcastable flags get set for a  join
        with many inputs and many input dimensions.
        """
        a = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1, 0, 0, 0])()
        b = TensorType(dtype=self.floatX, broadcastable=[1, 1, 1, 0, 0, 0])()
        c = TensorType(dtype=self.floatX, broadcastable=[1, 0, 0, 0, 0, 0])()
        d = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1, 1, 0, 1])()
        e = TensorType(dtype=self.floatX, broadcastable=[1, 0, 1, 0, 0, 1])()
        f = join(0, a, b, c, d, e)
        fb = f.type.broadcastable
        assert not fb[0] and fb[1] and fb[2] and fb[3] and not fb[4] and fb[5]
        g = join(1, a, b, c, d, e)
        gb = g.type.broadcastable
        assert gb[0] and not gb[1] and gb[2] and gb[3] and not gb[4] and gb[5]
        h = join(4, a, b, c, d, e)
        hb = h.type.broadcastable
        assert hb[0] and hb[1] and hb[2] and hb[3] and not hb[4] and hb[5]

        f = function([a, b, c, d, e], f, mode=self.mode)
        topo = f.maker.env.toposort()
        assert [True for node in topo if isinstance(node.op, self.join_op)]

        rng = numpy.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        b_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        c_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        d_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        e_val = rng.rand(1, 1, 1, 1, 2, 1).astype(self.floatX)
        f(a_val, b_val, c_val, d_val, e_val)
        utt.verify_grad((lambda a, b, c, d, e: join(0, a, b, c, d, e)),
                        [a_val, b_val, c_val, d_val, e_val], rng=rng)
        # Should raise an error if length of dimension 0 is not 1
        bad_val = rng.rand(2, 1, 1, 1, 2, 1).astype(self.floatX)
        self.assertRaises(TypeError, g, bad_val, b_val, c_val, d_val, e_val)
        self.assertRaises(TypeError, g, a_val, bad_val, c_val, d_val, e_val)
        self.assertRaises(TypeError, g, a_val, b_val, bad_val, d_val, e_val)
        self.assertRaises(TypeError, g, a_val, b_val, c_val, bad_val, e_val)
        self.assertRaises(TypeError, g, a_val, b_val, c_val, d_val, bad_val)
        # Should raise an error if any dimension other than 4 has length != 1
        bad_a_val = rng.rand(1, 2, 1, 1, 2, 1).astype(self.floatX)
        bad_b_val = rng.rand(1, 1, 1, 1, 2, 2).astype(self.floatX)
        bad_c_val = rng.rand(1, 1, 2, 1, 2, 1).astype(self.floatX)
        bad_d_val = rng.rand(1, 2, 1, 1, 2, 1).astype(self.floatX)
        bad_e_val = rng.rand(1, 1, 1, 2, 2, 1).astype(self.floatX)
        self.assertRaises(ValueError, f, bad_a_val, b_val, c_val, d_val, e_val)
        self.assertRaises(ValueError, f, a_val, bad_b_val, c_val, d_val, e_val)
        self.assertRaises(ValueError, f, a_val, b_val, bad_c_val, d_val, e_val)
        self.assertRaises(ValueError, f, a_val, b_val, c_val, bad_d_val, e_val)
        self.assertRaises(ValueError, f, a_val, b_val, c_val, d_val, bad_e_val)

    def test_infer_shape_join(self):
        x1 = matrix()
        x2 = matrix()
        x3 = matrix()

        def get_mat(s1, s2):
            return numpy.asarray(numpy.random.uniform(size=(s1, s2)),
                                 dtype=self.floatX)

        # Test dim 0
        z = join(0, x1, x2, x3)
        f = theano.function([x1, x2, x3], z.shape, mode=self.mode)
        topo = f.maker.env.toposort()

        out = f(get_mat(3, 4), get_mat(2, 4), get_mat(1, 4))
        assert (out == [6, 4]).all()

        if theano.config.mode != 'FAST_COMPILE':
            for node in f.maker.env.toposort():
                assert not isinstance(node.op, tensor.Join)

        # Test dim 1
        z = join(1, x1, x2, x3)
        f = theano.function([x1, x2, x3], z.shape, mode=self.mode)
        topo = f.maker.env.toposort()

        out = f( get_mat(3, 4), get_mat(3, 4), get_mat(3, 5))
        assert (out == [3, 13]).all()

        if theano.config.mode != 'FAST_COMPILE':
            for node in f.maker.env.toposort():
                assert not isinstance(node.op, tensor.Join)

        # Test hide error
        if not self.hide_error:
            self.assertRaises(ValueError, f, get_mat(3, 4), get_mat(3, 4),
                              get_mat(2, 5))
        else:
            f(get_mat(3, 4), get_mat(3, 4), get_mat(2, 5))


class test_comparison(unittest.TestCase):
    def test_gt(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x > y)
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.assertTrue(numpy.all(v == (l > r)), (v, (l>r)))

    def test_lt(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x < y)
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.assertTrue(numpy.all(v == (l < r)), (v, (l<r)))

    def test_le(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x <= y)
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.assertTrue(numpy.all(v == (l <= r)), (v, (l<=r)))

    def test_ge(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x >= y)
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.assertTrue(numpy.all(v == (l >= r)), (v, (l>=r)))

    def test_eq(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], eq(x,y))
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.assertTrue(numpy.all(v == (l == r)), (v, (l==r)))

    def test_neq(self):
        for dtype in ['float64', 'float32', 'complex64', 'complex128']:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], neq(x, y))
            l = numpy.asarray([0.,-1.,1.], dtype=dtype)
            r = numpy.asarray([0.,1.,-1.], dtype=dtype)
            v = fn(l, r)
            self.assertTrue(numpy.all(v == (l != r)), (v, (l!=r)))

class test_bitwise(unittest.TestCase):
    dtype = ['int8', 'int16', 'int32', 'int64',]

    def test_or(self):
        for dtype in self.dtype:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x|y)
            l = theano._asarray([0,0,1,1], dtype = dtype)
            r = theano._asarray([0,1,0,1], dtype = dtype)
            v = fn(l, r)
            self.assertTrue(numpy.all(v == (operator.or_(l, r))), (l, r, v))

    def test_xor(self):
        for dtype in self.dtype:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x^y)
            ix = x
            ix = inplace.xor_inplace(ix, y)
            gn = inplace_func([x,y], ix)
            l = theano._asarray([0,0,1,1], dtype = dtype)
            r = theano._asarray([0,1,0,1], dtype = dtype)
            v = fn(l, r)
            self.assertTrue(numpy.all(v == (operator.xor(l, r))), (l, r, v))
            v = gn(l, r)
            #test the in-place stuff
            self.assertTrue(numpy.all(l == numpy.asarray([0,1,1,0])), l)

    def test_and(self):
        for dtype in self.dtype:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x,y], x&y)
            l = theano._asarray([0,0,1,1], dtype = dtype)
            r = theano._asarray([0,1,0,1], dtype = dtype)
            v = fn(l, r)
            self.assertTrue(numpy.all(v == (operator.and_(l, r))), (l, r, v))

    def test_inv(self):
        for dtype in self.dtype:
            x = vector(dtype=dtype)
            fn = inplace_func([x], ~x)
            for l in [[0,0,1,1],[0,1,0,1],
                      [0,0,1,1],[0,1,0,1],
                      [-1,2**16, 2**16-1]
                      ]:
                l = theano._asarray([0,0,1,1], dtype = dtype)
                v = fn(l)
                self.assertTrue(numpy.all(v == (~l)), (l, v))

    def test_eye(self):
        n = iscalar()
        m = iscalar()
        k = iscalar()
        fn = theano.function([m,n,k],eye(m,n,k) )
        self.assertTrue(numpy.all(fn(5,6,1) == numpy.eye(5,6,1)))


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
                self.assertTrue(a.type.values_eq_approx(fn(a.data, b.data), f(a.data, b.data)))

    def test_grad_scalar_l(self):
        utt.verify_grad(add, [numpy.asarray([3.0]), rand(3)])
    def test_grad_scalar_r(self):
        utt.verify_grad(add, [rand(3), numpy.asarray([3.0])])
    def test_grad_row(self):
        utt.verify_grad(add, [rand(3, 5), rand(1, 5)])
    def test_grad_col(self):
        utt.verify_grad(add, [rand(3, 5), rand(3, 1)])

class T_ceil(unittest.TestCase):
    def test_complex(self):
        self.assertRaises(TypeError, tensor.ceil, tensor.zvector())

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
        assert numpy.allclose(function([i, ii, d, f, c], i//ii)(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (5/3))
        assert numpy.allclose(function([i, ii, d, f, c], ii//i)(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (3/5))
        assert numpy.allclose(function([i, ii, d, f, c], true_div(i,ii))(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (5./3.))
        assert numpy.allclose(function([i, ii, d, f, c], true_div(ii,i))(5, 3, 7.0, 11.0, numpy.complex(5,3)),
                (3./5.))


class T_mean(unittest.TestCase):
    def test_regression_mean_of_ndarray_failure(self):
        try:
            tensor.mean(numpy.zeros(1))
        except AttributeError:
            self.fail()

    def test0(self):
        #Simple test...
        x = tensor.vector()
        f = theano.function([x],tensor.mean(x))
        data = rand(50)
        assert numpy.allclose(f(data), numpy.mean(data))


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
        x = rand(dim,dim)+0.1      # Initialized s.t. x is not too tiny
        w = rand(dim,dim)
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
        x = rand(3,3)+0.1
        w = rand(3,3)
        x = numpy.asarray(x, dtype=config.floatX)
        w = numpy.asarray(w, dtype=config.floatX)
        ones = numpy.ones((3,3), dtype=config.floatX)

        myssd0 = numpy.sum((x*w - ones)**2.0)
        # we want at least a test that is not too fast. So we make one here.
        for i in xrange(100):
            gw = 2*(x*w - ones)*x  # derivative of dMSE/dw
            myssd = numpy.sum((x*w - ones)**2)
            w -= 0.4 * gw
        self.assertAlmostEqual(ssd0, myssd0)
        self.assertAlmostEqual(ssd, myssd)

class t_dot(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def cmp_dot(self,x,y):
        #x, y are matrices or numbers
        def spec(x):
            x = numpy.asarray(x)
            return type(x), x.dtype, x.shape
        nz = numpy.dot(x,y)
        tz = eval_outputs([dot(as_tensor_variable(x), as_tensor_variable(y))])
        self.assertTrue(tz.dtype == nz.dtype)
        self.assertTrue(tz.shape == nz.shape)
        self.assertTrue(_approx_eq(nz, tz))

    #def test_dot_0d_0d(self): self.cmp_dot(1.1, 2.2)
    #def test_dot_0d_1d(self): self.cmp_dot(1.1, rand(5))
    #def test_dot_0d_2d(self): self.cmp_dot(3.0, rand(6,7))
    #def test_dot_0d_3d(self): self.cmp_dot(3.0, rand(8,6,7))
    #def test_dot_1d_0d(self): self.cmp_dot(rand(5), 1.1 )
    def test_dot_1d_1d(self): self.cmp_dot(rand(5), rand(5))
    def test_dot_1d0_1d0(self): self.cmp_dot(rand(0), rand(0))
    #numpy return matrix not aligned...
    #def test_dot_1d_1d0(self): self.cmp_dot(rand(5), rand(0))
    #numpy return matrix not aligned...
    #def test_dot_1d0_1d(self): self.cmp_dot(rand(0), rand(5))
    def test_dot_1d_2d(self): self.cmp_dot(rand(6), rand(6,7))
    def test_dot_1d0_2d(self): self.cmp_dot(rand(0), rand(0,7))
    def test_dot_1d_2d0(self): self.cmp_dot(rand(6), rand(6,0))
    def test_dot_1d0_2d0(self): self.cmp_dot(rand(0), rand(0,0))
    #def test_dot_1d_3d(self): self.cmp_dot(rand(6), rand(8,6,7))
    #def test_dot_2d_0d(self): self.cmp_dot(rand(5,6), 1.0)
    def test_dot_2d_1d(self): self.cmp_dot(rand(5,6), rand(6))
    def test_dot_2d0_1d(self): self.cmp_dot(rand(0,6), rand(6))
    def test_dot_2d_1d0(self): self.cmp_dot(rand(5,0), rand(0))
    def test_dot_2d0_1d0(self): self.cmp_dot(rand(0,0), rand(0))
    def test_dot_2d_2d(self): self.cmp_dot(rand(5,6), rand(6,7))
    def test_dot_2d0_2d(self): self.cmp_dot(rand(0,6), rand(6,7))
    def test_dot_2d_2d0(self): self.cmp_dot(rand(5,6), rand(6,0))
    def test_dot_2d0_2d0(self): self.cmp_dot(rand(0,6), rand(6,0))
    def test_dot_2d_0_2d(self): self.cmp_dot(rand(5,0), rand(0,7))
    def test_dot_2d0_0_2d0(self): self.cmp_dot(rand(0,6), rand(6,0))
    #def test_dot_2d_3d(self): self.cmp_dot(rand(5,6), rand(8,6,7))
    #def test_dot_3d_0d(self): self.cmp_dot(rand(4,5,6), 1.0)
    #def test_dot_3d_1d(self): self.cmp_dot(rand(4,5,6), rand(6))
    #def test_dot_3d_2d(self): self.cmp_dot(rand(4,5,6), rand(6,7))
    #def test_dot_3d_3d(self): self.cmp_dot(rand(4,5,6), rand(8,6,7))

    def not_aligned(self, x, y):
        ctv_backup = config.compute_test_value
        config.compute_test_value = 'off'
        try:
            z = dot(x,y)
        finally:
            config.compute_test_value = ctv_backup
        # constant folding will complain to _logger that things are not aligned
        # this is normal, testers are not interested in seeing that output.
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.level
        _logger.setLevel(logging.CRITICAL)
        try:
            try:
                tz = eval_outputs([z])
                assert False    # should have raised exception
            except ValueError, e:
                self.assertTrue(
                    # Reported by numpy.
                    e[0].split()[1:4] == ['are', 'not', 'aligned'] or
                    # Reported by blas or Theano.
                    e[0].split()[0:2] == ['Shape', 'mismatch:'] or
                    # Reported by Theano when 'exception_verbosity' is set
                    # to 'high'.
                    e[0].split()[0:3] == ['dot', 'product', 'failed.'],
                    e)
        finally:
            _logger.setLevel(oldlevel)

    def test_align_1_1(self):
        self.not_aligned(rand(5), rand(6))

    def test_align_1_2(self):
        self.not_aligned(rand(5), rand(6,4))

    #def test_align_1_3(self): self.not_aligned(rand(5), rand(6,4,7))

    def test_align_2_1(self):
        self.not_aligned(rand(5,4), rand(6))

    def test_align_2_1(self):
        self.not_aligned(rand(5,4), rand(6,7))

    #def test_align_2_3(self): self.not_aligned(rand(5,4), rand(6,7,8))
    #def test_align_3_1(self): self.not_aligned(rand(5,4,3), rand(6))
    #def test_align_3_2(self): self.not_aligned(rand(5,4,3), rand(6,7))
    #def test_align_3_3(self): self.not_aligned(rand(5,4,3), rand(6,7,8))

    def test_grad(self):
        #utt.verify_grad(dot, [rand(2,3,4), rand(4)])
        utt.verify_grad(dot, [rand(2,3), rand(3,2)])
        utt.verify_grad(dot, [rand(2), rand(2,3)])
        utt.verify_grad(dot, [rand(3,2), rand(2)])
        utt.verify_grad(dot, [rand(2), rand(2)])
        #utt.verify_grad(dot, [rand(), rand(2)])
        #utt.verify_grad(dot, [rand(), rand(2,5)])

    def test_broadcastable_patterns(self):

        #
        # These examples hsould all work because we broadcastable or no, all dimensions of all
        # results have size 1.
        #
        def val_for(r):
            if r.dtype.startswith('complex'):
                # We want to test complex at the same time, so we give a value
                # To the imaginary component.
                # This stange way to doing thing is the only way that worked on
                # numpy 1.4.1
                if r.ndim == 0:
                    return numpy.asarray(numpy.complex(1.1,2.1), dtype=r.dtype)
                if r.ndim == 1:
                    if r.dtype == 'complex64':
                        return numpy.complex64([numpy.complex(1.2,2.2)])
                    elif r.dtype == 'complex128':
                        return numpy.complex128([numpy.complex(1.2,2.2)])
                elif r.ndim == 2:
                    if r.dtype == 'complex64':
                        return numpy.complex64([[numpy.complex(1.3,2.3)]])
                    elif r.dtype == 'complex128':
                        return numpy.complex128([[numpy.complex(1.3,2.3)]])

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
        self.assertTrue(t.owner.op is tensor_from_scalar)
        self.assertTrue(t.type.broadcastable == (), t.type.broadcastable)
        self.assertTrue(t.type.ndim == 0, t.type.ndim)
        self.assertTrue(t.type.dtype == s.type.dtype)

        v = eval_outputs([t])

        self.assertTrue(v == 56, v)
        self.assertTrue(isinstance(v, numpy.ndarray))
        self.assertTrue(v.shape == (), v.shape)

    def test1(self):
        s = scal.constant(56)
        t = as_tensor_variable(s)
        self.assertTrue(t.owner.op is tensor_from_scalar)
        self.assertTrue(t.type.broadcastable == (), t.type.broadcastable)
        self.assertTrue(t.type.ndim == 0, t.type.ndim)
        self.assertTrue(t.type.dtype == s.type.dtype)

        v = eval_outputs([t])

        self.assertTrue(v == 56, v)
        self.assertTrue(isinstance(v, numpy.ndarray))
        self.assertTrue(v.shape == (), v.shape)

        g = grad(t, s)
        self.assertTrue(eval_outputs([g])==1)

class T_scalarfromtensor(unittest.TestCase):
    def test0(self):
        tt = constant(56)#scal.constant(56)
        ss = scalar_from_tensor(tt)
        self.assertTrue(ss.owner.op is scalar_from_tensor)
        self.assertTrue(ss.type.dtype == tt.type.dtype)

        v = eval_outputs([ss])

        self.assertTrue(v == 56, v)
        if config.cast_policy == 'custom':
            self.assertTrue(isinstance(v, numpy.int8))
        elif config.cast_policy in ('numpy', 'numpy+floatX'):
            self.assertTrue(isinstance(
                v, getattr(numpy, str(numpy.asarray(56).dtype))))
        else:
            raise NotImplementedError(config.cast_policy)
        self.assertTrue(v.shape == (), v.shape)
        tt = lscalar()
        ss = scalar_from_tensor(tt)
        g = ss.owner.op.grad([tt],[ss])
        fff=function([tt],ss)
        v = fff(numpy.asarray(5))
        self.assertTrue(v == 5, v)
        self.assertTrue(isinstance(v, numpy.int64))
        self.assertTrue(v.shape == (),v.shape)


class test_grad(unittest.TestCase):
    class O(gof.op.Op):
        def __init__(self):
            self.gval0 = scalar('e')
            self.gval1 = scalar('f')
        def make_node(self):
            inputs = [scalar('a'),scalar('c')]
            outputs = [scalar('b'),scalar('d')]
            return gof.Apply(self, inputs, outputs)
        def grad(self, inp, grads):
            x0, x1 = inp
            gz0, gz1 = grads
            return self.gval0, self.gval1

    def test_1param(self):
        """grad: Test passing a single variable param"""
        o = test_grad.O()
        a1 = o.make_node()
        self.assertTrue(o.gval0 is tensor.grad(a1.outputs[0], a1.inputs[0]))

    def test_Nparam(self):
        """grad: Test passing multiple variable params"""
        o = test_grad.O()
        a1 = o.make_node()
        g0,g1 = grad(a1.outputs[0], a1.inputs)
        self.assertTrue(o.gval0 is g0)
        self.assertTrue(o.gval1 is g1)


    def test_grad_keep_type(self):
        """Tests that the theano grad method returns a list if it is passed a list
        and a single variable if it is passed a single variable.
        pylearn2 depends on theano behaving this way. This functionality has been
        added three times and erroneously removed twice. If you do anything that
        requires changing this test or making it fail you are almost certainly
        making a common mistake, NOT fixing something. """

        X = tensor.matrix()
        y = X.sum()

        G = tensor.grad(y, [X])

        assert isinstance(G,list)

        G = tensor.grad(y, X)

        assert not isinstance(G,list)


    def test_1None_rval(self):
        """grad: Test returning a single zero value from grad"""
        o = test_grad.O()
        a1 = o.make_node()
        g = grad(a1.outputs[0], a1.outputs[1],
                 disconnected_inputs='ignore')
        self.assertTrue(g.owner.op == fill)
        self.assertTrue(g.owner.inputs[1].data == 0)
        self.assertRaises(ValueError, grad, a1.outputs[0], 'wtf')

    def test_NNone_rval(self):
        """grad: Test returning some zero value from grad"""
        o = test_grad.O()
        a1 = o.make_node()
        g0,g1,g2 = grad(a1.outputs[0], a1.inputs + [scalar('z')],
                        disconnected_inputs='ignore')
        self.assertTrue(o.gval0 is g0)
        self.assertTrue(o.gval1 is g1)
        self.assertTrue(g2.owner.op == fill)
        self.assertTrue(g2.owner.inputs[1].data == 0)

    def test_zero_gradient_shape(self):
        """Ensure that a zero gradient has the proper shape."""
        x = dmatrix()
        f = theano.function([x], grad(dscalar(), x,
                                      disconnected_inputs='ignore'))
        a = numpy.ones((3, 7))
        self.assertTrue((f(a) == 0).all())  # Zero gradient.
        self.assertTrue(a.shape == f(a).shape)  # With proper shape.

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

        a = rand(5,2).astype(config.floatX)
        self.assertTrue(numpy.all(fn_py(a) == fn_c_or_py(a)))

class T_reshape(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_reshape(self):
        a = dvector()
        b = dmatrix()
        d = dmatrix()

        #basic to 1 dim(without list)
        c = reshape(b, as_tensor_variable(6), ndim=1)
        f = inplace_func([b], c)
        assert numpy.all(f(numpy.asarray([[0,1,2],[3,4,5]])) == numpy.asarray([0,1,2,3,4,5]))
        #print f.maker.env.toposort()
        #check that we remove the useless reshape

        #basic to 1 dim(with list)
        c = reshape(b, (as_tensor_variable(6),), ndim=1)
        f = inplace_func([b], c)
        assert numpy.all(f(numpy.asarray([[0,1,2],[3,4,5]])) == numpy.asarray([0,1,2,3,4,5]))
        #print f.maker.env.toposort()
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
            topo = f_sub.maker.env.toposort()
            assert len(topo)==1
            topo[0].op == theano.compile.function_module.deep_copy_op
            #assert numpy.all(f_sub(a_val,numpy.asarray([[0,1],[2,3],[4,5]]))==[2,3])#work in FAST_RUN, but fail on other!
            #assert numpy.all(f_sub(a_val,numpy.asarray([[0,1],[2,3],[4,5],[6,7]]))==[2,3])#work in FAST_RUN, but fail on other!

        # test broadcast flag for constant value of 1
        c = reshape(b, (b.shape[0],b.shape[1],1))
        f = inplace_func([b], c)
        assert numpy.all(f(numpy.asarray([[0,1,2],[3,4,5]])) == numpy.asarray([[[0],[1],[2]],[[3],[4],[5]]]))
        assert f.maker.env.toposort()[-2].outputs[0].type.broadcastable==(False, False, True)

        assert numpy.all(f_sub(a_val,b_val)==[2,3])

    def test_infer_shape(self):
        a = matrix('a')
        shapes = ivector('shapes')
        ndim = 2

        r = a.reshape(shapes, ndim=2)
        z = zeros_like(r)

        f = function([a, shapes], z.shape)

        rng = numpy.random.RandomState(seed=utt.fetch_seed())
        a_val = rng.uniform(size=(3, 4)).astype(config.floatX)

        self.assertTrue((f(a_val, [4, 3]) == [4, 3]).all())
        self.assertTrue((f(a_val, [-1, 3]) == [4, 3]).all())
        self.assertTrue((f(a_val, [4, -1]) == [4, 3]).all())
        self.assertRaises(ValueError, f, a_val, [-1, 5])
        self.assertRaises(ValueError, f, a_val, [7, -1])
        self.assertRaises(ValueError, f, a_val, [7, 5])
        self.assertRaises(ValueError, f, a_val, [-1, -1])


def test_make_column_matrix_broadcastable():
    # The goal of the operation made by `b` is to ensure the second dimension
    # of the column matrix is broadcastable.
    a = tensor.dmatrix()
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

def test_tile():
    # Test the one-dimensional case.
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = vector()
    f = function([x], tile(x, (2,)))
    x_ = rng.randn(5).astype(config.floatX)
    assert numpy.all(f(x_) == numpy.tile(x_, (2,)))

    # Test the two-dimensional case.
    x = matrix()
    f = function([x], tile(x, (2, 3)))
    x_ = rng.randn(2, 4).astype(config.floatX)
    assert numpy.all(f(x_) == numpy.tile(x_, (2, 3)))

    # Test the three-dimensional case.
    x = tensor3()
    f = function([x], tile(x, (2, 3, 4)))
    x_ = rng.randn(2, 4, 3).astype(config.floatX)
    assert numpy.all(f(x_) == numpy.tile(x_, (2, 3, 4)))


# XXX: It turns out that almost no cases of the tile gradient actually work.
# This is a test that should pass if the proper implementation is filled in.
def test_tile_grad_3d():
    # N.B.: we should also use verify_grad in this test.
    raise SkipTest()  # Remove me when this is implemented.
    rng = numpy.random.RandomState(utt.fetch_seed())
    w = rng.randn(3, 4, 2)
    w_tiled = numpy.tile(w, (2, 3, 4))
    x = tensor.tensor3()
    c = (as_tensor_variable(w_tiled) * tile(x,  (2, 3, 4))).sum()
    f = function([x], grad(c, x))
    x_ = rng.randn(3, 4, 2)
    # The gradient should be w, multiplied by its tiling dimensions (since
    # the gradients are additive through the tiling operation)
    assert numpy.all(f(x_) == 2 * 3 * 4 * w)



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

        if config.cast_policy == 'custom':
            assert out.dtype == start.type.dtype
        elif config.cast_policy in ('numpy', 'numpy+floatX'):
            numpy_dtype = numpy.arange(numpy.array(1, dtype='int32')).dtype
            assert out.dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)
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

        if config.cast_policy == 'custom':
            assert out.dtype == start.type.dtype
        elif config.cast_policy == 'numpy':
            numpy_dtype = numpy.arange(numpy.array(0, dtype=start.dtype),
                                       numpy.array(1, dtype=stop.dtype),
                                       numpy.array(1, dtype=step.dtype)).dtype
            assert out.dtype == numpy_dtype
        elif config.cast_policy == 'numpy+floatX':
            assert out.dtype == config.floatX
        else:
            raise NotImplementedError(config.cast_policy)
        arg_vals = [ (0,5,1), (2,11,4), (-5,1.1,1.2), (1.3,2,-2.1), (10,2,2) ]
        for arg_v in arg_vals:
            start_v, stop_v, step_v = arg_v
            start_v_, stop_v_, step_v_ = numpy.asarray(arg_v, dtype=start.type.dtype)
            f_val = f(start_v_, stop_v_, step_v_)
            if config.cast_policy == 'custom':
                expected_val = numpy.arange(start_v, stop_v, step_v,
                                            dtype=start.type.dtype)
            elif config.cast_policy in ('numpy', 'numpy+floatX'):
                expected_val = numpy.arange(start_v_, stop_v_, step_v_,
                                            dtype=out.dtype)
            else:
                raise NotImplementedError(config.cast_policy)
            assert numpy.all(f_val == expected_val)

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
            f_val = f(start_v_, stop_v_, step_v_)
            if config.cast_policy == 'custom':
                expected_val = numpy.arange(start_v, stop_v, step_v,
                                            dtype=start.type.dtype)
            elif config.cast_policy in ('numpy', 'numpy+floatX'):
                expected_val = numpy.arange(start_v_, stop_v_, step_v_)
            else:
                raise NotImplementedError(config.cast_policy)
            assert numpy.all(f_val == expected_val)

    def test_default_step(self):
        """Test that arange constructor uses the correct default step"""
        start, stop = iscalars('start', 'stop')
        out = arange(start, stop)
        f = function([start, stop], out)

        if config.cast_policy == 'custom':
            assert out.dtype == start.type.dtype
        elif config.cast_policy in ('numpy', 'numpy+floatX'):
            assert out.dtype == numpy.arange(numpy.int32(0),
                                             numpy.int32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)
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

        if config.cast_policy == 'custom':
            assert out.dtype == stop.type.dtype
        elif config.cast_policy in ('numpy', 'numpy+floatX'):
            assert out.dtype == numpy.arange(numpy.int32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)
        assert numpy.all(f(8) == numpy.arange(8))
        assert numpy.all(f(-2) == numpy.arange(-2))

        fstop = fscalar('stop')
        fout = arange(fstop)
        ff = function([fstop], fout)

        if config.cast_policy == 'custom':
            assert fout.dtype == fstop.type.dtype
        elif config.cast_policy == 'numpy':
            assert fout.dtype == numpy.arange(numpy.float32(1)).dtype
        elif config.cast_policy == 'numpy+floatX':
            if config.floatX == 'float32':
                assert fout.dtype == 'float32'
            else:
                assert fout.dtype == numpy.arange(numpy.float32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)

        fstop_values = [0.2, -0.7, 8.5]
        for fstop_v in fstop_values:
            fstop_v32 = numpy.float32(fstop_v)
            assert numpy.all(ff(fstop_v32) == numpy.arange(fstop_v))

    def test_upcast(self):
        """Test that arange computes output type adequately"""
        if config.cast_policy == 'custom':
            assert arange(iscalar()).dtype == iscalar().dtype
            assert arange(fscalar()).dtype == fscalar().dtype
            assert arange(dscalar()).dtype == dscalar().dtype

            # int32 + float32 -> float64
            assert arange(iscalar(), fscalar()).dtype == dscalar().dtype
            assert arange(iscalar(), dscalar()).dtype == dscalar().dtype
            assert arange(fscalar(), dscalar()).dtype == dscalar().dtype

            assert arange(iscalar(), fscalar(), dscalar()).dtype == dscalar().dtype
        elif config.cast_policy in ('numpy', 'numpy+floatX'):
            for dtype in get_numeric_types():
                # Test with a single argument.
                arange_dtype = arange(scalar(dtype=str(dtype))).dtype
                numpy_dtype = numpy.arange(numpy.array(1, dtype=dtype)).dtype
                if (dtype != 'float64' and
                    numpy_dtype == 'float64' and
                    config.cast_policy == 'numpy+floatX' and
                    config.floatX == 'float32'):
                    # We want a float32 arange.
                    assert arange_dtype == 'float32'
                else:
                    # Follow numpy.
                    assert arange_dtype == numpy_dtype

                # Test with two arguments.
                for stop_dtype in get_numeric_types():
                    arange_dtype = arange(
                            start=scalar(dtype=str(dtype)),
                            stop=scalar(dtype=str(stop_dtype))).dtype
                    numpy_dtype = numpy.arange(
                            start=numpy.array(0, dtype=dtype),
                            stop=numpy.array(1, dtype=stop_dtype)).dtype
                    if (dtype != 'float64' and
                        stop_dtype != 'float64' and
                        numpy_dtype == 'float64' and
                        config.cast_policy == 'numpy+floatX' and
                        config.floatX == 'float32'):
                        # We want a float32 arange.
                        assert arange_dtype == 'float32'
                    else:
                        # Follow numpy.
                        assert arange_dtype == numpy_dtype

                    # Test with three arguments.
                    for step_dtype in get_numeric_types():
                        arange_dtype = arange(
                                start=scalar(dtype=str(dtype)),
                                stop=scalar(dtype=str(stop_dtype)),
                                step=scalar(dtype=str(step_dtype))).dtype
                        numpy_dtype = numpy.arange(
                                start=numpy.array(0, dtype=dtype),
                                stop=numpy.array(1, dtype=stop_dtype),
                                step=numpy.array(1, dtype=step_dtype)).dtype
                        if (dtype != 'float64' and
                            stop_dtype != 'float64' and
                            step_dtype != 'float64' and
                            numpy_dtype == 'float64' and
                            config.cast_policy == 'numpy+floatX' and
                            config.floatX == 'float32'):
                            # We want a float32 arange.
                            assert arange_dtype == 'float32'
                        else:
                            # Follow numpy.
                            assert arange_dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)

    def test_dtype_cache(self):
        """Checks that the same Op is returned on repeated calls to arange
        using the same dtype, but not for different dtypes."""

        start, stop, step = iscalars('start', 'stop', 'step')
        out1 = arange(start, stop, step)
        out2 = arange(start, stop, step, dtype=out1.dtype)
        out3 = arange(start, stop, 2., dtype=out1.dtype)
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

        if config.cast_policy == 'custom':
            assert out.dtype == start.type.dtype
        elif config.cast_policy in ('numpy', 'numpy+floatX'):
            numpy_dtype = numpy.arange(numpy.array(0, dtype=start.dtype),
                                       numpy.array(1, dtype=stop.dtype),
                                       numpy.array(1, dtype=step.dtype)).dtype
            assert out.dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)

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
        if config.cast_policy == 'custom':
            assert out.dtype == start.type.dtype
        elif config.cast_policy in ('numpy', 'numpy+floatX'):
            assert out.dtype == numpy.arange(
                    numpy.int32(0), numpy.int32(1), numpy.int32(1)).dtype
        else:
            raise NotImplementedError(config.cast_policy)
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

        if config.cast_policy == 'custom':
            assert out.dtype == start.type.dtype
        elif config.cast_policy in ('numpy', 'numpy+floatX'):
            numpy_dtype = numpy.arange(0,
                                       numpy.array(1, dtype=stop.dtype),
                                       1).dtype
            assert out.dtype == numpy_dtype
        else:
            raise NotImplementedError(config.cast_policy)

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
        input = matrix()
        p = ivector()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = numpy.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(3, 5)).astype(config.floatX)
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
        input = matrix()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = numpy.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(3, 5)).astype(config.floatX)
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
        input = vector()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = numpy.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(5,)).astype(config.floatX)
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

        input = TensorType('floatX', (False, True, False))()
        p = imatrix()
        out = permute_row_elements(input, p)
        permute = function([input, p], out)

        rng = numpy.random.RandomState(utt.fetch_seed())
        input_val = rng.uniform(size=(4, 1, 5)).astype(config.floatX)
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

    def test0(self):

        # Test vector-vector
        avec = vector()
        bvec = vector()
        axes = ((0,),(0,))
        c = tensordot(avec, bvec, axes)
        f1 = inplace_func([avec,bvec],c)
        aval = rand(5)
        bval = rand(5)
        self.assertTrue(numpy.tensordot(aval,bval,axes) == \
                        f1(aval,bval))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test matrix-vector
        bmat = matrix()
        axes = ((0,),(1,))
        c = tensordot(avec, bmat, axes)
        f2 = inplace_func([avec,bmat],c)
        aval = rand(5)
        bval = rand(8,5)
        self.assertTrue(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f2(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test matrix-matrix
        amat = matrix()
        axes = ((1,),(0,))
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat,bmat],c)
        aval = rand(4,7)
        bval = rand(7,9)
        self.assertTrue(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f3(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test ndarray-matrix, sum over one dim of matrix
        atens = tensor4()
        axes = ((2,),(1,))
        c = tensordot(atens, bmat, axes)
        f4 = inplace_func([atens,bmat],c)
        aval = rand(1,2,3,4)
        bval = rand(2,3)
        self.assertTrue(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f4(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test ndarray-ndarray
        atens = tensor4()
        btens = tensor3()
        axes = ((1,3),(0,2))
        c = tensordot(atens, btens, axes)
        f5 = inplace_func([atens,btens],c)
        aval = rand(4,3,5,2)
        bval = rand(3,4,2)
        self.assertTrue(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f5(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        axes = (axes[1],axes[0])
        c = tensordot(btens, atens, axes)
        f6 = inplace_func([btens,atens],c)
        self.assertTrue(numpy.allclose(numpy.tensordot(bval,aval,axes),
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
            aval = rand(4,7)
            bval = rand(7,9)
            self.assertTrue(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                           f3(aval,bval)))
            utt.verify_grad(TensorDot(axes), [aval,bval])

    def test_scalar_axes(self):
        # Test matrix-matrix
        amat = fmatrix()
        bmat = dmatrix()# We let at float64 to test mix of float32 and float64.
        axes = 1
        aval = rand(4,5).astype('float32')
        bval = rand(5,3)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat,bmat],c)
        self.assertTrue(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f3(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

        # Test tensor-tensor
        amat = tensor3()
        bmat = tensor3()
        axes = 2
        aval = rand(3,4,5)
        bval = rand(4,5,3)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat,bmat],c)
        self.assertTrue(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f3(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

    def test_scalar0(self):
        # Test tensor-tensor
        amat = matrix()
        bmat = matrix()
        axes = 0
        aval = rand(4,5)
        bval = rand(5,4)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat,bmat],c)
        self.assertTrue(numpy.allclose(numpy.tensordot(aval,bval,axes),
                                       f3(aval,bval)))
        utt.verify_grad(TensorDot(axes), [aval,bval])

    def test_tensordot_grad(self):
        # We test it manually as we recreate the op in the make_node

        amat = matrix()
        bmat = matrix()
        gzmat = matrix()
        axes = 1
        aval = rand(4,5)
        bval = rand(5,3)
        gzval = rand(4,3)
        f1 = inplace_func([amat,bmat,gzmat],tensordot_grad(axes)(amat, bmat, gzmat))
        f2 = inplace_func([amat,bmat,gzmat],tensordot_grad(((1,),(0,)))(amat, bmat, gzmat))
        o1=f1(aval,bval,gzval)
        o2=f2(aval,bval,gzval)
        self.assertTrue(numpy.allclose(o1[0],o2[0]))
        self.assertTrue(numpy.allclose(o1[1],o2[1]))

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
    backup_config = config.cast_policy
    # Call test functions for all possible values of `config.cast_policy`.
    for autocast_cfg in (
            'custom',
            #'numpy', # Commented out until it is implemented properly.
            'numpy+floatX',
            ):
        config.cast_policy = autocast_cfg
        try:
            eval('_test_autocast_' + autocast_cfg.replace('+', '_'))()
        finally:
            config.cast_policy = backup_config

def _test_autocast_custom():
    """Called from `test_autocast`."""
    assert config.cast_policy == 'custom'
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


def _test_autocast_numpy():
    """Called from `test_autocast`."""
    assert config.cast_policy == 'numpy'
    # Go through some typical scalar values.
    def ok(z):
        assert tensor.constant(z).dtype == numpy.asarray(z).dtype
    for x in ([2**i for i in xrange(63)] +
              [0] +
              [0., 1., 1.1, 1.5]):
        n_x = numpy.asarray(x)
        # Make sure the data type is the same as the one found by numpy.
        ok(x)
        ok(-x)
        ok(x - 1)
        ok(-x + 1)
        ok(n_x)


def _test_autocast_numpy_floatX():
    """Called from `test_autocast`."""
    assert config.cast_policy == 'numpy+floatX'
    backup_floatX = config.floatX
    def ok(z, floatX):
        if (isinstance(z, float) and
            floatX == 'float32' and
            not hasattr(z, 'dtype')):
            # Special case where we use 'float32' instead of 'float64'.
            assert tensor.constant(z).dtype == 'float32'
        else:
            assert tensor.constant(z).dtype == numpy.asarray(z).dtype
    try:
        # Test with various values of `config.floatX`.
        for floatX in ('float32', 'float64'):
            config.floatX = floatX
            # Go through some typical scalar values.
            # Note that we only consider integer values that Python considers
            # to be 'int', because 'long' is not supported by Theano (due to
            # the fact it is unbounded).
            for x in ([2**i for i in xrange(64) if type(2**i) == int] +
                      [0] +
                      [0., 1., 1.1, 1.5]):
                ok(x, floatX)
                ok(-x, floatX)
                ok(x - 1, floatX)
                ok(-x + 1, floatX)
                ok(numpy.asarray(x), floatX)
                ok(numpy.float64(x), floatX)
    finally:
        config.floatX = backup_floatX


class test_arithmetic_cast(unittest.TestCase):

    """
    Test output types of basic arithmeric operations (* / + - //).

    We only test the behavior for `config.cast_policy` set to either 'numpy' or
    'numpy+floatX': the 'custom' behavior is (at least partially) tested in
    `_test_autocast_custom`.
    """

    def test_arithmetic_cast(self):
        backup_config = config.cast_policy
        dtypes = get_numeric_types(with_complex=True)
        # Here:
        # scalar == scalar stored as a 0d array
        # array == 1d array
        # i_scalar == scalar type used internally by Theano
        theano_scalar = lambda dtype: tensor.scalar(dtype=str(dtype))
        numpy_scalar = lambda dtype: numpy.array(1, dtype=dtype)
        theano_array = lambda dtype: tensor.vector(dtype=str(dtype))
        numpy_array = lambda dtype: numpy.array([1], dtype=dtype)
        theano_i_scalar = lambda dtype: theano.scalar.Scalar(str(dtype))()
        numpy_i_scalar = numpy_scalar
        if config.int_division == 'int':
            # Avoid deprecation warning during tests.
            warnings.filterwarnings('ignore', message='Division of two integer',
                                    category=DeprecationWarning)
        try:
            for cfg in ('numpy+floatX', ): # Used to test 'numpy' as well.
                config.cast_policy = cfg
                for op in (operator.add, operator.sub, operator.mul,
                           operator.div, operator.floordiv):
                    for a_type in dtypes:
                        for b_type in dtypes:
                            # Note that we do not test division between
                            # integers if it is forbidden.
                            # Theano deals with integer division in its own
                            # special way (depending on `config.int_division`).
                            is_int_division = (
                                    op is operator.div and
                                    a_type in tensor.discrete_dtypes and
                                    b_type in tensor.discrete_dtypes)
                            # We will test all meaningful combinations of
                            # scalar and array operations.
                            for combo in (
                                          ('scalar', 'scalar'),
                                          ('array', 'array'),
                                          ('scalar', 'array'),
                                          ('array', 'scalar'),
                                          ('i_scalar', 'i_scalar'),
                                          ):

                                theano_args = map(eval,
                                        ['theano_%s' % c for c in combo])
                                numpy_args = map(eval,
                                        ['numpy_%s' % c for c in combo])
                                try:
                                    theano_dtype = op(
                                        theano_args[0](a_type),
                                        theano_args[1](b_type)).type.dtype
                                    # Should have crashed if it is an integer
                                    # division and `config.int_division` does
                                    # not allow it.
                                    assert not (is_int_division and
                                                config.int_division == 'raise')
                                except theano.scalar.IntegerDivisionError:
                                    assert (is_int_division and
                                            config.int_division == 'raise')
                                    # This is the expected behavior.
                                    continue
                                # For numpy we have a problem:
                                #   http://projects.scipy.org/numpy/ticket/1827
                                # As a result we only consider the highest data
                                # type that numpy may return.
                                numpy_dtypes = [
                                        op(numpy_args[0](a_type),
                                           numpy_args[1](b_type)).dtype,
                                        op(numpy_args[1](b_type),
                                           numpy_args[0](a_type)).dtype]
                                numpy_dtype = theano.scalar.upcast(
                                        *map(str, numpy_dtypes))
                                if numpy_dtype == theano_dtype:
                                    # Same data type found, all is good!
                                    continue
                                if (cfg == 'numpy+floatX' and
                                    config.floatX == 'float32' and
                                    a_type != 'float64' and
                                    b_type != 'float64' and
                                    numpy_dtype == 'float64'):
                                    # We should keep float32.
                                    assert theano_dtype == 'float32'
                                    continue
                                if 'array' in combo and 'scalar' in combo:
                                    # For mixed scalar / array operations,
                                    # Theano may differ from numpy as it does
                                    # not try to prevent the scalar from
                                    # upcasting the array.
                                    array_type, scalar_type = (
                                            (a_type, b_type)[
                                                        list(combo).index(arg)]
                                            for arg in ('array', 'scalar'))
                                    up_type = theano.scalar.upcast(array_type,
                                                                   scalar_type)
                                    if (
                                        # The two data types are different.
                                        scalar_type != array_type and
                                        # The array type is not enough to hold
                                        # the scalar type as well.
                                        array_type != up_type and
                                        # Theano upcasted the result array.
                                        theano_dtype == up_type and
                                        # But Numpy kept its original type.
                                        # (not an equality because of numpy bug
                                        # mentioned above).
                                        array_type in numpy_dtypes):
                                        # Then we accept this difference in
                                        # behavior.
                                        continue
                                if (is_int_division and
                                    config.int_division == 'floatX'):
                                    assert theano_dtype == config.floatX
                                    continue
                                if (cfg == 'numpy+floatX' and
                                    a_type == 'complex128' and
                                    b_type == 'float32' and
                                    combo == ('scalar', 'array') and
                                    numpy.__version__.startswith('1.6.') and
                                    theano_dtype == 'complex128' and
                                    numpy_dtypes == ['complex64',
                                                     'complex64']):
                                    # In numpy 1.6.x adding a complex128 with
                                    # a float32 may result in a complex64. This
                                    # may be a bug (investigation is currently
                                    # in progress), so in the meantime we just
                                    # mark this test as a known failure.
                                    raise KnownFailureTest('Known issue with '
                                            'numpy 1.6.x, see #761')

                                # In any other situation: something wrong is
                                # going on!
                                assert False
        finally:
            config.cast_policy = backup_config
            if config.int_division == 'int':
                # Restore default deprecation warning behavior.
                warnings.filterwarnings(
                        'default',
                        message='Division of two integer',
                        category=DeprecationWarning)


class test_broadcast(unittest.TestCase):
    def test_broadcast_bigdim(self):
        def f():
            x = matrix()
            addbroadcast(x,2)
        self.assertRaises(ValueError, f)

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

    def test_infer_shape(self):
        x = matrix()
        y = addbroadcast(x, 0)
        f = theano.function([x], y.shape)
        assert (f(numpy.zeros((1,5), dtype=config.floatX)) == [1,5]).all()
        topo = f.maker.env.toposort()
        if theano.config.mode != 'FAST_COMPILE':
            assert len(topo) == 2
            assert isinstance(topo[0].op, opt.Shape_i)
            assert isinstance(topo[1].op, opt.MakeVector)

        x = matrix()
        y = unbroadcast(x, 0)
        f = theano.function([x], y.shape)
        assert (f(numpy.zeros((2,5), dtype=config.floatX)) == [2,5]).all()
        topo = f.maker.env.toposort()
        if theano.config.mode != 'FAST_COMPILE':
            assert len(topo) == 3
            assert isinstance(topo[0].op, opt.Shape_i)
            assert isinstance(topo[1].op, opt.Shape_i)
            assert isinstance(topo[2].op, opt.MakeVector)

        x = row()
        y = unbroadcast(x, 0)
        f = theano.function([x], y.shape)
        assert (f(numpy.zeros((1,5), dtype=config.floatX)) == [1,5]).all()
        topo = f.maker.env.toposort()
        if theano.config.mode != 'FAST_COMPILE':
            assert len(topo) == 2
            assert isinstance(topo[0].op, opt.Shape_i)
            assert isinstance(topo[1].op, opt.MakeVector)


def test_len():
    for shape in [(5,), (3, 4), (7, 4, 6)]:
        x = tensor.tensor(dtype='floatX', broadcastable=(False,)*len(shape))
        try:
            len(x)
            assert False, "Expected an error"
        except TypeError:
            pass


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
        Elemwise{
            Composite{
                Composite{
                    Composite{
                        Composite{mod,EQ},
                        Switch},
                    mul},
                add}}

    The c_code generated is not compiling as of 30 June 2010. I fix the
    compilation in the same commit.
    """

    x = tensor.vector()
    y = tensor.vector()
    shape = x.shape
    out = tensor.switch(tensor.eq(3 % x.shape[0], 0), y, y[:-1])

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
    a[:] = rand(len(a))
    b[:] = rand(len(b))
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
    x = tensor.vector()

    success = False

    try:
        y = tensor.DimShuffle((False, ), (0, 0))(x)
    except ValueError, e:
        assert str(e).find("may not appear twice") != -1
        success = True

    assert success


class T_get_constant_value(unittest.TestCase):
    def test_get_constant_value(self):
        a = tensor.stack(1, 2, 3)
        assert get_constant_value(a[0]) == 1
        assert get_constant_value(a[1]) == 2
        assert get_constant_value(a[2]) == 3

        b = tensor.iscalar()
        a = tensor.stack(b, 2, 3)
        self.assertRaises(TypeError, get_constant_value, a[0])
        assert get_constant_value(a[1]) == 2
        assert get_constant_value(a[2]) == 3

        # For now get_constant_value goes through only MakeVector and Join of
        # scalars.
        v = tensor.ivector()
        a = tensor.stack(v, 2, 3)
        self.assertRaises(TypeError, get_constant_value, a[0])
        self.assertRaises(TypeError, get_constant_value, a[1])
        self.assertRaises(TypeError, get_constant_value, a[2])

        # Test the case SubTensor(Shape(v)) when the dimensions
        # is broadcastable.
        v = tensor.row()
        assert get_constant_value(v.shape[0]) == 1

    def test_subtensor_of_constant(self):
        c = constant(rand(5))
        for i in range(c.value.shape[0]):
            assert get_constant_value(c[i]) == c.value[i]
        c = constant(rand(5, 5))
        for i in range(c.value.shape[0]):
            for j in range(c.value.shape[1]):
                assert get_constant_value(c[i, j]) == c.value[i, j]


class T_as_tensor_variable(unittest.TestCase):
    """
    We test that ticket #649 stay fixed.
    We should not allow as_tensor_variable to accept True or False
    But it should upcast an ndrarray of bool to uint8
    """

    def test_bool(self):
        self.assertRaises(TypeError, as_tensor_variable, True)
        self.assertRaises(TypeError, as_tensor_variable, False)

    def test_ndarray_bool(self):
        ten = as_tensor_variable(numpy.array([True, False, False, True, True]))
        assert ten.type.dtype == 'uint8'


class test_complex_mod(unittest.TestCase):
    """Make sure % fails on complex numbers."""

    def test_fail(self):
        x = vector(dtype='complex64')
        try:
            x % 5
            assert False
        except ComplexError:
            pass


class test_size(unittest.TestCase):
    """
    Ensure the `size` attribute of tensors behaves as in numpy.
    """

    def test_matrix(self):
        x = tensor.matrix()
        y = numpy.zeros((5, 7), dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_vector(self):
        x = tensor.vector()
        y = numpy.zeros(7, dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_scalar(self):
        x = tensor.scalar()
        y = numpy.array(7, dtype=config.floatX)
        assert y.size == function([x], x.size)(y)

    def test_shared(self):
        # NB: we also test higher order tensors at the same time.
        y = numpy.zeros((1, 2, 3, 4), dtype=config.floatX)
        x = theano.shared(y)
        assert y.size == function([], x.size)()


class test_numpy_assumptions(unittest.TestCase):
    """
    Verify that some assumptions Theano makes on Numpy's behavior still hold.
    """

    def test_ndarray_copy(self):
        """
        A copy or deepcopy of the ndarray type should not create a new object.

        This is because Theano makes some comparisons of the form:
            if type(x) is numpy.ndarray
        """
        assert copy(numpy.ndarray) is numpy.ndarray
        assert deepcopy(numpy.ndarray) is numpy.ndarray

    def test_dtype_equality(self):
        """
        Ensure dtype string comparisons are consistent.

        Theano often uses string representations of dtypes (e.g. 'float32'). We
        need to make sure that comparing the string representations is the same
        as comparing the dtype objects themselves.
        """
        dtypes = get_numeric_types(with_complex=True)
        # Perform all pairwise comparisons of dtypes, making sure comparing
        # their string representation yields the same result.
        for dtype1_idx, dtype1 in enumerate(dtypes):
            for dtype2 in dtypes[dtype1_idx + 1:]:
                assert (dtype1 == dtype2) == (str(dtype1) == str(dtype2))


def test_transpose():
    x1 = tensor.dvector()
    x2 = tensor.dmatrix()
    x3 = tensor.dtensor3()

    x1v = numpy.arange(24)
    x2v = numpy.arange(24).reshape(2, 12)
    x3v = numpy.arange(24).reshape(2, 3, 4)

    f = theano.function([x1, x2, x3], [
        tensor.transpose(x1),
        tensor.transpose(x2),
        tensor.transpose(x3),
        x1.transpose(),
        x2.transpose(),
        x3.transpose(),
        x2.transpose(0, 1),
        x3.transpose((0, 2, 1)),
        tensor.transpose(x2, [0, 1]),
        tensor.transpose(x3, [0, 2, 1]),
        ])

    t1, t2, t3, t1b, t2b, t3b, t2c, t3c, t2d, t3d = f(x1v, x2v, x3v)
    assert t1.shape == numpy.transpose(x1v).shape
    assert t2.shape == numpy.transpose(x2v).shape
    assert t3.shape == numpy.transpose(x3v).shape
    assert numpy.all(t1 == numpy.transpose(x1v))
    assert numpy.all(t2 == numpy.transpose(x2v))
    assert numpy.all(t3 == numpy.transpose(x3v))
    assert numpy.all(t1b == x1v.transpose())
    assert numpy.all(t2b == x2v.transpose())
    assert numpy.all(t3b == x3v.transpose())
    assert t2c.shape == (2, 12)
    assert t3c.shape == (2, 4, 3)
    assert numpy.all(t2c == x2v.transpose([0, 1]))
    assert numpy.all(t3c == x3v.transpose([0, 2, 1]))
    assert t2d.shape == (2, 12)
    assert t3d.shape == (2, 4, 3)
    assert numpy.all(t2d == numpy.transpose(x2v, [0, 1]))
    assert numpy.all(t3d == numpy.transpose(x3v, [0, 2, 1]))


if __name__ == '__main__':
    if 0:
        unittest.main()
    else:
        testcase = FloorInplaceTester

        suite = unittest.TestLoader()
        suite = suite.loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(suite)
