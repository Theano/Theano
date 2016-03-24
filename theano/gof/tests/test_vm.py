from __future__ import absolute_import, print_function, division
import gc
import sys
import time
import unittest

from nose.plugins.skip import SkipTest
import numpy
from six import itervalues

from theano import function
from theano.gof import vm
from theano.gof import OpWiseCLinker
from six.moves import xrange
from theano.compile import Mode

from theano import tensor
from theano.ifelse import ifelse
import theano


class TestCallbacks(unittest.TestCase):
    """
    Test the VM_Linker's callback argument, which can be useful for debugging.
    """
    def setUp(self):
        self.n_callbacks = {}

    def callback(self, node, thunk, storage_map, compute_map):
        key = node.op.__class__.__name__
        self.n_callbacks.setdefault(key, 0)
        self.n_callbacks[key] += 1

    def test_callback(self):
        a, b, c = tensor.scalars('abc')
        f = function([a, b, c], (a + b) + c,
                     mode=Mode(
                         optimizer=None,
                         linker=vm.VM_Linker(callback=self.callback)))

        f(1, 2, 3)
        assert sum(self.n_callbacks.values()) == len(f.maker.fgraph.toposort())
        f(1, 2, 3)
        assert (sum(self.n_callbacks.values()) ==
                len(f.maker.fgraph.toposort()) * 2)

    def test_callback_with_ifelse(self):
        a, b, c = tensor.scalars('abc')
        f = function([a, b, c], ifelse(a, 2 * b, 2 * c),
                     mode=Mode(
                         optimizer=None,
                         linker=vm.VM_Linker(callback=self.callback)))

        f(1, 2, 3)
        assert self.n_callbacks['IfElse'] == 2


def test_c_thunks():
    a = tensor.scalars('a')
    b, c = tensor.vectors('bc')
    cases = [False]
    if theano.config.cxx:
        cases.append(True)
    for c_thunks in cases:
        f = function([a, b, c], ifelse(a, a * b, b * c),
                     mode=Mode(
                         optimizer=None,
                         linker=vm.VM_Linker(c_thunks=c_thunks,
                                             use_cloop=False)))
        f(1, [2], [3, 2])
        from nose.tools import assert_raises
        assert_raises(ValueError, f, 0, [2], [3, 4])
        assert any([hasattr(t, 'cthunk') for t in f.fn.thunks]) == c_thunks


def test_speed():
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")

    def build_graph(x, depth=5):
        z = x
        for d in range(depth):
            z = (z + z)
        return z

    def numpy_version(x, depth):
        z = x
        for d in xrange(depth):
            z = (z + z)
        return z

    def time_numpy():
        steps_a = 5
        steps_b = 100
        x = numpy.asarray([2.0, 3.0], dtype=theano.config.floatX)

        numpy_version(x, steps_a)
        t0 = time.time()
        # print numpy_version(x, steps_a)
        t1 = time.time()
        t2 = time.time()
        # print numpy_version(x, steps_b)
        t3 = time.time()
        t_a = t1 - t0
        t_b = t3 - t2

        print("%s takes %f s/Kop" % (
            'numpy',
            (1000 * (t_b - t_a) / (steps_b - steps_a))))

    def time_linker(name, linker):
        steps_a = 5
        steps_b = 100
        x = tensor.vector()
        a = build_graph(x, steps_a)
        b = build_graph(x, steps_b)

        f_a = function([x], a,
                       mode=Mode(optimizer=None, linker=linker()))
        f_b = function([x], b,
                       mode=Mode(optimizer=None, linker=linker()))

        f_a([2.0, 3.0])
        t0 = time.time()
        f_a([2.0, 3.0])
        t1 = time.time()

        f_b([2.0, 3.0])

        t2 = time.time()
        f_b([2.0, 3.0])
        t3 = time.time()

        t_a = t1 - t0
        t_b = t3 - t2

        print("%s takes %f s/Kop" % (
            name,
            (1000 * (t_b - t_a) / (steps_b - steps_a))))

    time_linker('c|py', OpWiseCLinker)
    time_linker('vmLinker', vm.VM_Linker)
    time_linker('vmLinker_nogc', lambda: vm.VM_Linker(allow_gc=False))
    if theano.config.cxx:
        time_linker('vmLinker_CLOOP', lambda: vm.VM_Linker(allow_gc=False,
                                                           use_cloop=True))
    time_numpy()


def test_speed_lazy():

    def build_graph(x, depth=5):
        z = x
        for d in range(depth):
            z = ifelse(z[0] > 0, -z, z)
        return z

    def time_linker(name, linker):
        steps_a = 10
        steps_b = 100
        x = tensor.vector()
        a = build_graph(x, steps_a)
        b = build_graph(x, steps_b)

        f_a = function([x], a,
                       mode=Mode(optimizer=None,
                                 linker=linker()))
        f_b = function([x], b,
                       mode=Mode(optimizer=None,
                                 linker=linker()))

        f_a([2.0])
        t0 = time.time()
        f_a([2.0])
        t1 = time.time()

        f_b([2.0])

        t2 = time.time()
        f_b([2.0])
        t3 = time.time()

        t_a = t1 - t0
        t_b = t3 - t2

        print("%s takes %f s/Kop" % (
            name,
            (1000 * (t_b - t_a) / (steps_b - steps_a))))

    time_linker('vmLinker', vm.VM_Linker)
    time_linker('vmLinker_nogc', lambda: vm.VM_Linker(allow_gc=False))
    if theano.config.cxx:
        time_linker('vmLinker_C', lambda: vm.VM_Linker(allow_gc=False,
                                                       use_cloop=True))


def test_allow_gc_cvm():
    mode = theano.config.mode
    if mode in ['DEBUG_MODE', 'DebugMode']:
        mode = "FAST_RUN"

    v = theano.tensor.vector()
    f = theano.function([v], v + 1, mode=mode)

    f([1])
    n = list(f.maker.fgraph.apply_nodes)[0].outputs[0]
    assert f.fn.storage_map[n][0] is None
    assert f.fn.allow_gc is True

    f.fn.allow_gc = False
    assert f.fn.allow_gc is False
    f([1])
    assert f.fn.storage_map[n][0] is not None
    f.fn.allow_gc = True
    assert f.fn.allow_gc is True
    f([1])
    assert f.fn.storage_map[n][0] is None


run_memory_usage_tests = False
if run_memory_usage_tests:
    # these are not normal unit tests, do not run them as part of standard
    # suite.  I ran them while looking at top, and stopped when memory usage
    # was stable.
    def test_leak2():
        import theano.sandbox.cuda as cuda
        for i in xrange(1000000):
            n = numpy.asarray([2.3, 4.5], dtype='f')
            c = sys.getrefcount(n)
            a = cuda.CudaNdarray(n)
            a.sum()
            assert c == sys.getrefcount(n)
            del a
            if not i % 1000:
                print('.', end=' ')
                print(gc.collect(), end=' ')
                print(gc.collect())
            sys.stdout.flush()

    def test_no_leak_many_graphs():
        # Verify no memory leaks when creating and deleting a lot of functions

        # This isn't really a unit test, you have to run it and look at top to
        # see if there's a leak
        for i in xrange(10000):
            x = tensor.vector()
            z = x
            for d in range(10):
                z = tensor.sin(-z + 1)

            f = function([x], z, mode=Mode(optimizer=None, linker='cvm'))
            if not i % 100:
                print(gc.collect())
            sys.stdout.flush()

            gc.collect()
            if 1:
                f([2.0])
                f([3.0])
                f([4.0])
                f([5.0])

    def test_no_leak_many_call_lazy():
        # Verify no memory leaks when calling a function a lot of times

        # This isn't really a unit test, you have to run it and look at top to
        # see if there's a leak

        def build_graph(x, depth=5):
            z = x
            for d in range(depth):
                z = ifelse(z.mean() > 0.5, -z, z)
            return z

        def time_linker(name, linker):
            steps_a = 10
            x = tensor.dvector()
            a = build_graph(x, steps_a)

            f_a = function([x], a,
                           mode=Mode(optimizer=None,
                                     linker=linker()))
            inp = numpy.random.rand(1000000)
            for i in xrange(100):
                f_a(inp)
            if 0:  # this doesn't seem to work, prints 0 for everything
                import resource
                pre = resource.getrusage(resource.RUSAGE_SELF)
                post = resource.getrusage(resource.RUSAGE_SELF)
                print(pre.ru_ixrss, post.ru_ixrss)
                print(pre.ru_idrss, post.ru_idrss)
                print(pre.ru_maxrss, post.ru_maxrss)
        print(1)
        time_linker('vmLinker_C',
                    lambda: vm.VM_Linker(allow_gc=False, use_cloop=True))
        print(2)
        time_linker('vmLinker',
                    lambda: vm.VM_Linker(allow_gc=False, use_cloop=False))

    def test_no_leak_many_call_nonlazy():
        # Verify no memory leaks when calling a function a lot of times

        # This isn't really a unit test, you have to run it and look at top to
        # see if there's a leak.

        def build_graph(x, depth=5):
            z = x
            for d in range(depth):
                z = tensor.sin(-z + 1)
            return z

        def time_linker(name, linker):
            steps_a = 10
            x = tensor.dvector()
            a = build_graph(x, steps_a)

            f_a = function([x], a,
                           mode=Mode(optimizer=None,
                                     linker=linker()))
            inp = numpy.random.rand(1000000)
            for i in xrange(500):
                f_a(inp)
        print(1)
        time_linker('vmLinker_C',
                    lambda: vm.VM_Linker(allow_gc=False, use_cloop=True))
        print(2)
        time_linker('vmLinker',
                    lambda: vm.VM_Linker(allow_gc=False, use_cloop=False))


class RunOnce(theano.Op):

    __props__ = ("nb_run",)

    def __init__(self):
        self.nb_run = 0

    def make_node(self, x):
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        assert self.nb_run == 0
        self.nb_run += 1
        outputs[0][0] = inputs[0].copy()


def test_vm_gc():
    """This already caused a bug in the trunk of Theano.

    The bug was introduced in the trunk on July 5th, 2012 and fixed on
    July 30th.

    """
    x = theano.tensor.vector()
    p = RunOnce()(x)
    mode = theano.Mode(linker=theano.gof.vm.VM_Linker(lazy=True))
    f = theano.function([theano.In(x, mutable=True)], [p + 1, p + 2],
                        mode=mode)
    f([1, 2, 3])

    p = RunOnce()(x)
    pp = p + p
    f = theano.function([x], [pp + pp],
                        mode=mode)
    f([1, 2, 3])


def test_reallocation():
    x = tensor.scalar('x')
    y = tensor.scalar('y')
    z = tensor.tanh(3 * x + y) + tensor.cosh(x + 5 * y)
    # The functinality is currently implement for non lazy and non c VM only.
    for l in [vm.VM_Linker(allow_gc=False, lazy=False, use_cloop=False),
              vm.VM_Linker(allow_gc=True, lazy=False, use_cloop=False)]:
        m = theano.compile.get_mode(theano.Mode(linker=l))
        m = m.excluding('fusion', 'inplace')

        f = theano.function([x, y], z, name="test_reduce_memory",
                            mode=m)
        output = f(1, 2)
        assert output
        storage_map = f.fn.storage_map

        def check_storage(storage_map):
            from theano.tensor.var import TensorConstant
            for i in storage_map:
                if not isinstance(i, TensorConstant):
                    keys_copy = list(storage_map.keys())[:]
                    keys_copy.remove(i)
                    for o in keys_copy:
                        if (storage_map[i][0] and
                                storage_map[i][0] is storage_map[o][0]):
                            return [True, storage_map[o][0]]
            return [False, None]

        assert check_storage(storage_map)[0]
        assert len(set(id(v) for v in
                       itervalues(storage_map))) < len(storage_map)
