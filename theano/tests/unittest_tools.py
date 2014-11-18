from copy import copy, deepcopy
import logging
from StringIO import StringIO
import sys
import unittest

import numpy

import theano
import theano.tensor as T
from theano.configparser import config, AddConfigVar, StrParam
from theano.gof.python25 import any
try:
    from nose.plugins.skip import SkipTest
except ImportError:
    class SkipTest(Exception):
        """
        Skip this test
        """
_logger = logging.getLogger("theano.tests.unittest_tools")


def good_seed_param(seed):
    if seed == "random":
        return True
    try:
        int(seed)
    except Exception:
        return False
    return True

AddConfigVar('unittests.rseed',
             "Seed to use for randomized unit tests. "
             "Special value 'random' means using a seed of None.",
             StrParam(666, is_valid=good_seed_param),
             in_c_key=False)


def fetch_seed(pseed=None):
    """
    Returns the seed to use for running the unit tests.
    If an explicit seed is given, it will be used for seeding numpy's rng.
    If not, it will use config.unittest.rseed (its default value is 666).
    If config.unittest.rseed is set to "random", it will seed the rng with
    None, which is equivalent to seeding with a random seed.

    Useful for seeding RandomState objects.
    >>> rng = numpy.random.RandomState(unittest_tools.fetch_seed())
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
        print >> sys.stderr, ('Error: config.unittests.rseed contains '
                              'invalid seed, using None instead')
        seed = None

    return seed


def seed_rng(pseed=None):
    """
    Seeds numpy's random number generator with the value returned by fetch_seed.
    Usage: unittest_tools.seed_rng()
    """

    seed = fetch_seed(pseed)
    if pseed and pseed != seed:
        print >> sys.stderr, 'Warning: using seed given by config.unittests.rseed=%i'\
                'instead of seed %i given as parameter' % (seed, pseed)
    numpy.random.seed(seed)
    return seed


def verify_grad(op, pt, n_tests=2, rng=None, *args, **kwargs):
    """
    Wrapper for tensor/basic.py:verify_grad
    Takes care of seeding the random number generator if None is given
    """
    if rng is None:
        seed_rng()
        rng = numpy.random
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


class TestOptimizationMixin(object):
    def assertFunctionContains(self, f, op, min=1, max=sys.maxint):
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

    def assertFunctionContainsClass(self, f, op, min=1, max=sys.maxint):
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
                if isinstance(inp, (int, float, list, tuple)):
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
                        "While testing the shape inference, we received an"
                        " input with a shape that has some repeated values: %s"
                        ", like a square matrix. This makes it impossible to"
                        " check if the values for these dimensions have been"
                        " correctly used, or if they have been mixed up.",
                        str(inp.shape))
                    break

        outputs_function = theano.function(inputs, outputs, mode=mode)
        shapes_function = theano.function(inputs, [o.shape for o in outputs],
                                          mode=mode)
        #theano.printing.debugprint(shapes_function)
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
            assert numpy.all(out.shape == shape)


def str_diagnostic(expected, value, rtol, atol):
    """Return a pretty multiline string representating the cause
    of the exception"""
    sio = StringIO()

    try:
        ssio = StringIO()
        print >> ssio, "           : shape, dtype, strides, min, max, n_inf, n_nan:"
        print >> ssio, "  Expected :",
        print >> ssio, expected.shape,
        print >> ssio, expected.dtype,
        print >> ssio, expected.strides,
        print >> ssio, expected.min(),
        print >> ssio, expected.max(),
        print >> ssio, numpy.isinf(expected).sum(),
        print >> ssio, numpy.isnan(expected).sum(),
        # only if all succeeds to we add anything to sio
        print >> sio, ssio.getvalue()
    except Exception:
        pass
    try:
        ssio = StringIO()
        print >> ssio, "  Value    :",
        print >> ssio, value.shape,
        print >> ssio, value.dtype,
        print >> ssio, value.strides,
        print >> ssio, value.min(),
        print >> ssio, value.max(),
        print >> ssio, numpy.isinf(value).sum(),
        print >> ssio, numpy.isnan(value).sum(),
        # only if all succeeds to we add anything to sio
        print >> sio, ssio.getvalue()
    except Exception:
        pass

    print >> sio, "  expected    :", expected
    print >> sio, "  value    :", value

    try:
        ov = numpy.asarray(expected)
        nv = numpy.asarray(value)
        ssio = StringIO()
        absdiff = numpy.absolute(nv - ov)
        print >> ssio, "  Max Abs Diff: ", numpy.max(absdiff)
        print >> ssio, "  Mean Abs Diff: ", numpy.mean(absdiff)
        print >> ssio, "  Median Abs Diff: ", numpy.median(absdiff)
        print >> ssio, "  Std Abs Diff: ", numpy.std(absdiff)
        reldiff = numpy.absolute(nv - ov) / (numpy.absolute(nv) +
                                             numpy.absolute(ov))
        print >> ssio, "  Max Rel Diff: ", numpy.max(reldiff)
        print >> ssio, "  Mean Rel Diff: ", numpy.mean(reldiff)
        print >> ssio, "  Median Rel Diff: ", numpy.median(reldiff)
        print >> ssio, "  Std Rel Diff: ", numpy.std(reldiff)
        # only if all succeeds to we add anything to sio
        print >> sio, ssio.getvalue()
    except Exception:
        pass
    #Use the same formula as in _allclose to find the tolerance used
    narrow = 'float32', 'complex64'
    if ((str(expected.dtype) in narrow) or
        (str(value.dtype) in narrow)):
        atol_ = T.basic.float32_atol
        rtol_ = T.basic.float32_rtol
    else:
        atol_ = T.basic.float64_atol
        rtol_ = T.basic.float64_rtol
    if rtol is not None:
        rtol_ = rtol
    if atol is not None:
        atol_ = atol
    print >> sio, "  rtol, atol:", rtol_, atol_
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


def assert_allclose(val1, val2, rtol=None, atol=None):
    if not T.basic._allclose(val1, val2, rtol, atol):
        raise WrongValue(val1, val2, rtol, atol)
