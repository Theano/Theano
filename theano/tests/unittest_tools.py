from __future__ import absolute_import, print_function, division
from copy import copy, deepcopy
from functools import wraps
import logging
import sys
import unittest
from parameterized import parameterized
from nose.tools import assert_raises

from six import integer_types
from six.moves import StringIO

try:
    from nose.plugins.attrib import attr
except ImportError:
    # This is an old version of nose
    def attr(tag):
        def func(f):
            return f
        return func
import numpy as np

import theano
import theano.tensor as T
from theano import config
try:
    from nose.plugins.skip import SkipTest
except ImportError:
    class SkipTest(Exception):
        """
        Skip this test
        """
_logger = logging.getLogger("theano.tests.unittest_tools")


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


def fetch_seed(pseed=None):
    """
    Returns the seed to use for running the unit tests.
    If an explicit seed is given, it will be used for seeding numpy's rng.
    If not, it will use config.unittest.rseed (its default value is 666).
    If config.unittest.rseed is set to "random", it will seed the rng with
    None, which is equivalent to seeding with a random seed.

    Useful for seeding RandomState objects.
    >>> rng = np.random.RandomState(unittest_tools.fetch_seed())
    """

    seed = pseed or config.unittests.rseed
    if seed == 'random':
        seed = None

    try:
        if seed:
            seed = int(seed)
        else:
            seed = None
    except ValueError:
        print(('Error: config.unittests.rseed contains ' 'invalid seed, using None instead'), file=sys.stderr)
        seed = None

    return seed


def seed_rng(pseed=None):
    """
    Seeds numpy's random number generator with the value returned by fetch_seed.
    Usage: unittest_tools.seed_rng()
    """

    seed = fetch_seed(pseed)
    if pseed and pseed != seed:
        print('Warning: using seed given by config.unittests.rseed=%i' 'instead of seed %i given as parameter' % (seed, pseed), file=sys.stderr)
    np.random.seed(seed)
    return seed


def verify_grad(op, pt, n_tests=2, rng=None, *args, **kwargs):
    """
    Wrapper for gradient.py:verify_grad
    Takes care of seeding the random number generator if None is given
    """
    if rng is None:
        seed_rng()
        rng = np.random
    T.verify_grad(op, pt, n_tests, rng, *args, **kwargs)

#
# This supports the following syntax:
#
# try:
#     verify_grad(...)
# except verify_grad.E_grad, e:
#     print e.num_grad.gf
#     print e.analytic_grad
#     raise
#
verify_grad.E_grad = T.verify_grad.E_grad


# A helpful class to check random values close to the boundaries
# when designing new tests
class MockRandomState:
    def __init__(self, val):
        self.val = val

    def rand(self, *shape):
        return np.zeros(shape, dtype='float64') + self.val

    def randint(self, minval, maxval=None, size=1):
        if maxval is None:
            minval, maxval = 0, minval
        out = np.zeros(size, dtype='int64')
        if self.val == 0:
            return out + minval
        else:
            return out + maxval - 1


class TestOptimizationMixin(object):

    def assertFunctionContains(self, f, op, min=1, max=sys.maxsize):
        toposort = f.maker.fgraph.toposort()
        matches = [node for node in toposort if node.op == op]
        assert (min <= len(matches) <= max), (toposort, matches,
                                              str(op), len(matches), min, max)

    def assertFunctionContains0(self, f, op):
        return self.assertFunctionContains(f, op, min=0, max=0)

    def assertFunctionContains1(self, f, op):
        return self.assertFunctionContains(f, op, min=1, max=1)

    def assertFunctionContainsN(self, f, op, N):
        return self.assertFunctionContains(f, op, min=N, max=N)

    def assertFunctionContainsClass(self, f, op, min=1, max=sys.maxsize):
        toposort = f.maker.fgraph.toposort()
        matches = [node for node in toposort if isinstance(node.op, op)]
        assert (min <= len(matches) <= max), (toposort, matches,
                                              str(op), len(matches), min, max)

    def assertFunctionContainsClassN(self, f, op, N):
        return self.assertFunctionContainsClass(f, op, min=N, max=N)

    def SkipTest(self, msg='Skip this test'):
        raise SkipTest(msg)


# This object name should not start with Test.
# Otherwise nosetests will execute it!
class T_OpContractMixin(object):
    # self.ops should be a list of instantiations of an Op class to test.
    # self.other_op should be an op which is different from every op
    other_op = T.add

    def copy(self, x):
        return copy(x)

    def deepcopy(self, x):
        return deepcopy(x)

    def clone(self, op):
        raise NotImplementedError('return new instance like `op`')

    def test_eq(self):
        for i, op_i in enumerate(self.ops):
            assert op_i == op_i
            assert op_i == self.copy(op_i)
            assert op_i == self.deepcopy(op_i)
            assert op_i == self.clone(op_i)
            assert op_i != self.other_op
            for j, op_j in enumerate(self.ops):
                if i == j:
                    continue
                assert op_i != op_j

    def test_hash(self):
        for i, op_i in enumerate(self.ops):
            h_i = hash(op_i)
            assert h_i == hash(op_i)
            assert h_i == hash(self.copy(op_i))
            assert h_i == hash(self.deepcopy(op_i))
            assert h_i == hash(self.clone(op_i))
            assert h_i != hash(self.other_op)
            for j, op_j in enumerate(self.ops):
                if i == j:
                    continue
                assert op_i != hash(op_j)

    def test_name(self):
        for op in self.ops:
            s = str(op)    # show that str works
            assert s       # names should not be empty


class InferShapeTester(unittest.TestCase):

    def setUp(self):
        seed_rng()
        # Take into account any mode that may be defined in a child class
        # and it can be None
        mode = getattr(self, 'mode', None)
        if mode is None:
            mode = theano.compile.get_default_mode()
        # This mode seems to be the minimal one including the shape_i
        # optimizations, if we don't want to enumerate them explicitly.
        self.mode = mode.including("canonicalize")

    def _compile_and_check(self, inputs, outputs, numeric_inputs, cls,
                           excluding=None, warn=True, check_topo=True):
        """This tests the infer_shape method only

        When testing with input values with shapes that take the same
        value over different dimensions (for instance, a square
        matrix, or a tensor3 with shape (n, n, n), or (m, n, m)), it
        is not possible to detect if the output shape was computed
        correctly, or if some shapes with the same value have been
        mixed up. For instance, if the infer_shape uses the width of a
        matrix instead of its height, then testing with only square
        matrices will not detect the problem. If warn=True, we emit a
        warning when testing with such values.

        :param check_topo: If True, we check that the Op where removed
            from the graph. False is useful to test not implemented case.

        """
        mode = self.mode
        if excluding:
            mode = mode.excluding(*excluding)
        if warn:
            for var, inp in zip(inputs, numeric_inputs):
                if isinstance(inp, (integer_types, float, list, tuple)):
                    inp = var.type.filter(inp)
                if not hasattr(inp, "shape"):
                    continue
                # remove broadcasted dims as it is sure they can't be
                # changed to prevent the same dim problem.
                if hasattr(var.type, "broadcastable"):
                    shp = [inp.shape[i] for i in range(inp.ndim)
                           if not var.type.broadcastable[i]]
                else:
                    shp = inp.shape
                if len(set(shp)) != len(shp):
                    _logger.warn(
                        "While testing shape inference for %r, we received an"
                        " input with a shape that has some repeated values: %r"
                        ", like a square matrix. This makes it impossible to"
                        " check if the values for these dimensions have been"
                        " correctly used, or if they have been mixed up.",
                        cls, inp.shape)
                    break

        outputs_function = theano.function(inputs, outputs, mode=mode)
        shapes_function = theano.function(inputs, [o.shape for o in outputs],
                                          mode=mode)
        # theano.printing.debugprint(shapes_function)
        # Check that the Op is removed from the compiled function.
        if check_topo:
            topo_shape = shapes_function.maker.fgraph.toposort()
            assert not any(isinstance(t.op, cls) for t in topo_shape)
        topo_out = outputs_function.maker.fgraph.toposort()
        assert any(isinstance(t.op, cls) for t in topo_out)
        # Check that the shape produced agrees with the actual shape.
        numeric_outputs = outputs_function(*numeric_inputs)
        numeric_shapes = shapes_function(*numeric_inputs)
        for out, shape in zip(numeric_outputs, numeric_shapes):
            assert np.all(out.shape == shape), (out.shape, shape)


def str_diagnostic(expected, value, rtol, atol):
    """Return a pretty multiline string representating the cause
    of the exception"""
    sio = StringIO()

    try:
        ssio = StringIO()
        print("           : shape, dtype, strides, min, max, n_inf, n_nan:", file=ssio)
        print("  Expected :", end=' ', file=ssio)
        print(expected.shape, end=' ', file=ssio)
        print(expected.dtype, end=' ', file=ssio)
        print(expected.strides, end=' ', file=ssio)
        print(expected.min(), end=' ', file=ssio)
        print(expected.max(), end=' ', file=ssio)
        print(np.isinf(expected).sum(), end=' ', file=ssio)
        print(np.isnan(expected).sum(), end=' ', file=ssio)
        # only if all succeeds to we add anything to sio
        print(ssio.getvalue(), file=sio)
    except Exception:
        pass
    try:
        ssio = StringIO()
        print("  Value    :", end=' ', file=ssio)
        print(value.shape, end=' ', file=ssio)
        print(value.dtype, end=' ', file=ssio)
        print(value.strides, end=' ', file=ssio)
        print(value.min(), end=' ', file=ssio)
        print(value.max(), end=' ', file=ssio)
        print(np.isinf(value).sum(), end=' ', file=ssio)
        print(np.isnan(value).sum(), end=' ', file=ssio)
        # only if all succeeds to we add anything to sio
        print(ssio.getvalue(), file=sio)
    except Exception:
        pass

    print("  expected    :", expected, file=sio)
    print("  value    :", value, file=sio)

    try:
        ov = np.asarray(expected)
        nv = np.asarray(value)
        ssio = StringIO()
        absdiff = np.absolute(nv - ov)
        print("  Max Abs Diff: ", np.max(absdiff), file=ssio)
        print("  Mean Abs Diff: ", np.mean(absdiff), file=ssio)
        print("  Median Abs Diff: ", np.median(absdiff), file=ssio)
        print("  Std Abs Diff: ", np.std(absdiff), file=ssio)
        reldiff = np.absolute(nv - ov) / np.absolute(ov)
        print("  Max Rel Diff: ", np.max(reldiff), file=ssio)
        print("  Mean Rel Diff: ", np.mean(reldiff), file=ssio)
        print("  Median Rel Diff: ", np.median(reldiff), file=ssio)
        print("  Std Rel Diff: ", np.std(reldiff), file=ssio)
        # only if all succeeds to we add anything to sio
        print(ssio.getvalue(), file=sio)
    except Exception:
        pass
    atol_, rtol_ = T.basic._get_atol_rtol(expected, value)
    if rtol is not None:
        rtol_ = rtol
    if atol is not None:
        atol_ = atol
    print("  rtol, atol:", rtol_, atol_, file=sio)
    return sio.getvalue()


class WrongValue(Exception):

    def __init__(self, expected_val, val, rtol, atol):
        Exception.__init__(self)  # to be compatible with python2.4
        self.val1 = expected_val
        self.val2 = val
        self.rtol = rtol
        self.atol = atol

    def __str__(self):
        s = "WrongValue\n"
        return s + str_diagnostic(self.val1, self.val2, self.rtol, self.atol)


def assert_allclose(expected, value, rtol=None, atol=None):
    if not T.basic._allclose(expected, value, rtol, atol):
        raise WrongValue(expected, value, rtol, atol)


class AttemptManyTimes:
    """Decorator for unit tests that forces a unit test to be attempted
    multiple times. The test needs to pass a certain number of times for it to
    be considered to have succeeded. If it doesn't pass enough times, it is
    considered to have failed.

    Warning : care should be exercised when using this decorator. For some
    tests, the fact that they fail randomly could point to important issues
    such as race conditions, usage of uninitialized memory region, etc. and
    using this decorator could hide these problems.

    Usage:
        @AttemptManyTimes(n_attempts=5, n_req_successes=3)
        def fct(args):
            ...
    """

    def __init__(self, n_attempts, n_req_successes=1):
        assert n_attempts >= n_req_successes
        self.n_attempts = n_attempts
        self.n_req_successes = n_req_successes

    def __call__(self, fct):

        # Wrap fct in a function that will attempt to run it multiple
        # times and return the result if the test passes enough times
        # of propagate the raised exception if it doesn't.
        @wraps(fct)
        def attempt_multiple_times(*args, **kwargs):

            # Keep a copy of the current seed for unittests so that we can use
            # a different seed for every run of the decorated test and restore
            # the original after
            original_seed = config.unittests.rseed
            current_seed = original_seed

            # If the decorator has received only one, unnamed, argument
            # and that argument has an attribute _testMethodName, it means
            # that the unit test on which the decorator is used is in a test
            # class. This means that the setup() method of that class will
            # need to be called before any attempts to execute the test in
            # case it relies on data randomly generated in the class' setup()
            # method.
            if (len(args) == 1 and hasattr(args[0], "_testMethodName")):
                test_in_class = True
                class_instance = args[0]
            else:
                test_in_class = False

            n_fail = 0
            n_success = 0

            # Attempt to call the test function multiple times. If it does
            # raise any exception for at least one attempt, it passes. If it
            # raises an exception at every attempt, it fails.
            for i in range(self.n_attempts):
                try:
                    # Attempt to make the test use the current seed
                    config.unittests.rseed = current_seed
                    if test_in_class and hasattr(class_instance, "setUp"):
                        class_instance.setUp()

                    fct(*args, **kwargs)

                    n_success += 1
                    if n_success == self.n_req_successes:
                        break

                except Exception:
                    n_fail += 1

                    # If there is not enough attempts remaining to achieve the
                    # required number of successes, propagate the original
                    # exception
                    if n_fail + self.n_req_successes > self.n_attempts:
                        raise

                finally:
                    # Clean up after the test
                    config.unittests.rseed = original_seed
                    if test_in_class and hasattr(class_instance, "tearDown"):
                        class_instance.tearDown()

                    # Update the current_seed
                    if current_seed not in [None, "random"]:
                        current_seed = str(int(current_seed) + 1)

        return attempt_multiple_times


def assertFailure_fast(f):
    """A Decorator to handle the test cases that are failing when
    THEANO_FLAGS =cycle_detection='fast'.
    """
    if theano.config.cycle_detection == 'fast':
        def test_with_assert(*args, **kwargs):
            with assert_raises(Exception):
                f(*args, **kwargs)
        return test_with_assert
    else:
        return f
