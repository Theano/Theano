import unittest
from itertools import izip
from copy import copy, deepcopy

import numpy
import theano
import theano.tensor as T
from theano.compile import DeepCopyOp
from theano.tensor.tests.test_basic import safe_make_node
from theano.tests.unittest_tools import SkipTest
from numpy.testing.noseclasses import KnownFailureTest

import theano.sandbox.gpuarray

import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available and not theano.sandbox.gpuarray.pygpu_activated:
    if not cuda_ndarray.use.device_number:
        cuda_ndarray.use('gpu')
    theano.sandbox.gpuarray.init_dev('cuda')

if not theano.sandbox.gpuarray.pygpu_activated:
    raise SkipTest("pygpu disabled")

from theano.sandbox.gpuarray.type import (GpuArrayType,
                                          gpuarray_shared_constructor)
from theano.sandbox.gpuarray.basic_ops import (host_from_gpu, gpu_from_host,
                                               gpu_alloc, gpu_from_cuda,
                                               cuda_from_gpu)

from theano.tests import unittest_tools as utt
utt.seed_rng()
rng = numpy.random.RandomState(seed=utt.fetch_seed())

from pygpu import gpuarray

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpuarray')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpuarray'\
)
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpuarray')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpuarray')


def may_fail(msg, EClass):
    """Mark a test that requires very specific conditions to work to
       mask a specific exception class."""
    def test_decorator(f):
        def wrapper():
            try:
                f()
            except Exception, e:
                if isinstance(e, EClass):
                    raise KnownFailureTest(msg, e)
                raise
        wrapper.__name__ = f.__name__
        return wrapper
    return test_decorator

def inplace_func(inputs, outputs, mode=None, allow_input_downcast=False,
                 on_unused_input='raise', name=None):
    if mode is None:
        mode = mode_with_gpu
    return theano.function(inputs, outputs, mode=mode,
                           allow_input_downcast=allow_input_downcast,
                           accept_inplace=True,
                           on_unused_input=on_unused_input, name=name)


def fake_shared(value, name=None, strict=False, allow_downcast=None, **kwargs):
    from theano.tensor.sharedvar import tensor_constructor, scalar_constructor
    for c in (gpuarray_shared_constructor, tensor_constructor,
              scalar_constructor):
        try:
            return c(value, name=name, strict=strict,
                     allow_downcast=allow_downcast, **kwargs)
        except TypeError:
            continue

def rand_gpuarray(*shape, **kwargs):
    r = rng.rand(*shape) * 2 - 1
    dtype = kwargs.pop('dtype', theano.config.floatX)
    if len(kwargs) != 0:
        raise TypeError('Unexpected argument %s', kwargs.keys()[0])
    return gpuarray.array(r, dtype=dtype)


def makeTester(name, op, expected, good=None, bad_build=None, checks=None,
               bad_runtime=None, mode=None, skip=False, eps=1e-10):
    if good is None:
        good = {}
    if bad_build is None:
        bad_build = {}
    if bad_runtime is None:
        bad_runtime = {}
    if checks is None:
        checks = {}

    _op = op
    _expected = expected
    _good = good
    _bad_build = bad_build
    _bad_runtime = bad_runtime
    _skip = skip
    _checks = checks

    class Checker(unittest.TestCase):
        op = staticmethod(_op)
        expected = staticmethod(_expected)
        good = _good
        bad_build = _bad_build
        bad_runtime = _bad_runtime
        skip = _skip
        checks = _checks

        def setUp(self):
            eval(self.__class__.__module__ + '.' + self.__class__.__name__)

        def test_good(self):
            if skip:
                raise SkipTest(skip)

            for testname, inputs in good.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [fake_shared(input) for input in inputs]

                try:
                    node = safe_make_node(self.op, *inputrs)
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occured while making "
                               "a node with inputs %s") % (self.op, testname,
                                                           inputs)
                    exc.args += (err_msg,)
                    raise

                try:
                    f = inplace_func([], node.outputs, mode=mode,
                                     name='test_good')
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occured while trying to "
                               "make a Function") % (self.op, testname)
                    exc.args += (err_msg,)
                    raise

                if isinstance(self.expected, dict) and \
                        testname in self.expected:
                    expecteds = self.expected[testname]
                else:
                    expecteds = self.expected(*inputs)

                if not isinstance(expecteds, (list, tuple)):
                    expecteds = (expecteds,)

                try:
                    variables = f()
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occured while calling "
                               "the Function on the inputs %s") % (self.op,
                                                                   testname,
                                                                   inputs)
                    exc.args += (err_msg,)
                    raise

                for i, (variable, expected) in \
                        enumerate(izip(variables, expecteds)):
                    if variable.dtype != expected.dtype or \
                            variable.shape != expected.shape or \
                            not GpuArrayType.values_eq_approx(variable,
                                                             expected):
                        self.fail(("Test %s::%s: Output %s gave the wrong "
                                   "value. With inputs %s, expected %s "
                                   "(dtype %s), got %s (dtype %s).") % (
                                self.op, testname, i, inputs, expected,
                                expected.dtype, variable, variable.dtype))

                for description, check in self.checks.items():
                    if not check(inputs, variables):
                        self.fail(("Test %s::%s: Failed check: %s "
                                   "(inputs were %s, ouputs were %s)") %
                                  (self.op, testname, description,
                                   inputs, variables))

        def test_bad_build(self):
            if skip:
                raise SkipTest(skip)
            for testname, inputs in self.bad_build.items():
                inputs = [copy(input) for input in inputs]
                inputrs = [fake_shared(input) for input in inputs]
                self.assertRaises(Exception, safe_make_node, self.op, *inputrs)

        def test_bad_runtime(self):
            if skip:
                raise SkipTest(skip)
            for testname, inputs in self.bad_runtime.items():
                inputrs = [fake_shared(input) for input in inputs]
                try:
                    node = safe_make_node(self.op, *inputrs)
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occured while trying to "
                               "make a node with inputs %s") % (self.op,
                                                                testname,
                                                                inputs)
                    exc.args += (err_msg,)
                    raise

                try:
                    f = inplace_func([], node.outputs, mode=mode,
                                     name="test_bad_runtime")
                except Exception, exc:
                    err_msg = ("Test %s::%s: Error occured while trying to "
                               "make a Function") % (self.op, testname)
                    exc.args += (err_msg,)
                    raise

                self.assertRaises(Exception, f, [])

    Checker.__name__ = name
    return Checker


def test_transfer_cpu_gpu():
    a = T.fmatrix('a')
    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')
    
    av = numpy.asarray(rng.rand(5, 4), dtype='float32')
    gv = gpuarray.array(av)
    
    f = theano.function([a], gpu_from_host(a))
    fv = f(av)
    assert GpuArrayType.values_eq(fv, gv)

    f = theano.function([g], host_from_gpu(g))
    fv = f(gv)
    assert numpy.all(fv == av)


def test_transfer_strided():
    # This is just to ensure that it works in theano
    # compyte has a much more comprehensive suit of tests to ensure correctness
    a = T.fmatrix('a')
    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')

    av = numpy.asarray(rng.rand(5, 8), dtype='float32')
    gv = gpuarray.array(av)

    av = av[:,::2]
    gv = gv[:,::2]

    f = theano.function([a], gpu_from_host(a))
    fv = f(av)
    assert GpuArrayType.values_eq(fv, gv)

    f = theano.function([g], host_from_gpu(g))
    fv = f(gv)
    assert numpy.all(fv == av)


@may_fail("Op fails if both contexts are not the same and it's rare "
          "that the tests will be run this way", ValueError)
def test_transfer_cuda_gpu():
    import theano.sandbox.cuda as cuda_ndarray
    if cuda_ndarray.cuda_available == False:
        raise SkipTest("Can't test interaction with cuda if cuda not present")
    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')
    c = cuda_ndarray.CudaNdarrayType((False, False))('c')

    av = theano._asarray(rng.rand(5, 4), dtype='float32')
    gv = gpuarray.array(av)
    cv = cuda_ndarray.CudaNdarray(av)
    gvs = gv[:,::-2]
    cvs = cv[:,::-2]

    f = theano.function([c], gpu_from_cuda(c))
    fv = f(cv)
    assert GpuArrayType.values_eq_approx(fv, gv)

    fvs = f(cvs)
    assert GpuArrayType.values_eq_approx(fvs, gvs)

    f = theano.function([g], cuda_from_gpu(g))
    fv = f(gv)
    assert cuda_ndarray.CudaNdarrayType.values_eq_approx(fv, cv)

    fvs = f(gvs)
    assert cuda_ndarray.CudaNdarrayType.values_eq_approx(fvs, cvs)


def gpu_alloc_expected(x, *shp):
    g = gpuarray.empty(shp, dtype=x.dtype)
    g[:] = x
    return g

GpuAllocTester = makeTester(
    name="GpuAllocTester",
    op=gpu_alloc,
    expected=gpu_alloc_expected,
    good=dict(
        correct01=(rand_gpuarray(), numpy.int32(7)),
        correct01_bcast=(rand_gpuarray(1), numpy.int32(7)),
        correct02=(rand_gpuarray(), numpy.int32(4), numpy.int32(7)),
        correct12=(rand_gpuarray(7), numpy.int32(4), numpy.int32(7)),
        correct13=(rand_gpuarray(7), numpy.int32(2), numpy.int32(4),
                   numpy.int32(7)),
        correct23=(rand_gpuarray(4, 7), numpy.int32(2), numpy.int32(4),
                   numpy.int32(7))
        ),
    bad_runtime=dict(
        bad_shape12=(rand_gpuarray(7), numpy.int32(7), numpy.int32(5)),
        )
)

def test_deep_copy():
    a = rand_gpuarray(20, dtype='float32')
    g = GpuArrayType(dtype='float32', broadcastable=(False,))('g')

    f = theano.function([g], g)

    assert isinstance(f.maker.fgraph.toposort()[0].op, DeepCopyOp)

    res = f(a)

    assert GpuArrayType.values_eq(res, a)
