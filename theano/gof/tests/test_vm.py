import gc
import sys
import time
import unittest
try:
    import line_profiler
except ImportError:
    pass
import numpy

from theano import function
from theano.gof import vm
from theano.gof import link
from theano.gof import OpWiseCLinker
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
        f = function([a,b,c], (a + b) + c,
                mode=Mode(
                    optimizer=None,
                    linker=vm.VM_Linker(callback=self.callback)))

        f(1, 2, 3)
        assert sum(self.n_callbacks.values()) == len(f.maker.env.toposort())
        f(1, 2, 3)
        assert sum(self.n_callbacks.values()) == len(f.maker.env.toposort()) * 2


    def test_callback_with_ifelse(self):
        a, b, c = tensor.scalars('abc')
        f = function([a,b,c], ifelse(a, 2*b, 2*c),
                mode=Mode(
                    optimizer=None,
                    linker=vm.VM_Linker(callback=self.callback)))

        f(1, 2, 3)
        assert self.n_callbacks['IfElse'] == 2


def test_speed():

    def build_graph(x, depth=5):
        z = x
        for d in range(depth):
            z = (z + z)
        return z

    def numpy_version(x, depth):
        z = x
        for d in xrange(depth):
            z = (z+z)
        return z
    def time_numpy():
        steps_a = 5
        steps_b = 100
        x = numpy.asarray([2.0, 3.0], dtype=theano.config.floatX)

        numpy_version(x, steps_a)
        t0 = time.time()
        print numpy_version(x, steps_a)
        t1 = time.time()
        t2 = time.time()
        print numpy_version(x, steps_b)
        t3 = time.time()
        t_a = t1 - t0
        t_b = t3 - t2

        print "%s takes %f s/Kop" % (
                'numpy',
                (1000*(t_b-t_a) / (steps_b - steps_a)))

    def time_linker(name, linker):
        steps_a = 5
        steps_b = 100
        x = tensor.vector()
        a = build_graph(x,steps_a)
        b = build_graph(x,steps_b)


        f_a = function([x], a,
                mode=Mode(optimizer=None, linker=linker()),
                #profile='f_a speed test %s'%name,
                )
        f_b = function([x], b,
                mode=Mode(optimizer=None, linker=linker()),
                #profile='f_b speed test %s'%name,
                )

        print f_a([2.0, 3.0])
        t0 = time.time()
        print f_a([2.0, 3.0])
        t1 = time.time()

        print f_b([2.0, 3.0])

        t2 = time.time()
        print f_b([2.0, 3.0])
        t3 = time.time()

        t_a = t1 - t0
        t_b = t3 - t2

        print "%s takes %f s/Kop" % (
                name,
                (1000*(t_b-t_a) / (steps_b - steps_a)))

    time_linker('c|py', OpWiseCLinker)
    time_linker('vmLinker', vm.VM_Linker)
    time_linker('vmLinker_nogc', lambda : vm.VM_Linker(allow_gc=False))
    time_linker('vmLinker_CLOOP', lambda : vm.VM_Linker(allow_gc=False,
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
                    linker=linker()),
                #profile='f_a lazy ifelse %s'%name,
                )
        f_b = function([x], b,
                mode=Mode(optimizer=None,
                    linker=linker()),
                #profile='f_b lazy ifelse %s'%name,
                )

        print f_a([2.0])
        t0 = time.time()
        print f_a([2.0])
        t1 = time.time()

        print f_b([2.0])

        t2 = time.time()
        print f_b([2.0])
        t3 = time.time()

        t_a = t1 - t0
        t_b = t3 - t2

        print "%s takes %f s/Kop" % (
                name,
                (1000*(t_b-t_a) / (steps_b - steps_a)))

    time_linker('vmLinker', vm.VM_Linker)
    time_linker('vmLinker_nogc', lambda : vm.VM_Linker(allow_gc=False))
    time_linker('vmLinker_C', lambda : vm.VM_Linker(allow_gc=False,
        use_cloop=True))

run_memory_usage_tests = False
if run_memory_usage_tests:
    # these are not normal unit tests, do not run them as part of standard
    # suite.  I ran them while looking at top, and stopped when memory usage was
    # stable.
    def test_leak2():
        import theano.sandbox.cuda as cuda
        for i in xrange(1000000):
            n = numpy.asarray([2.3, 4.5], dtype='f')
            c = sys.getrefcount(n)
            a = cuda.CudaNdarray(n)
            assert c == sys.getrefcount(n)
            if not i % 1000:
                print '.',
                print gc.collect(),
                print gc.collect()
            sys.stdout.flush()

    def test_no_leak_many_graphs():
        # Verify no memory leaks when creating and deleting a lot of functions

        # This isn't really a unit test, you have to run it and look at top to see
        # if there's a leak
        for i in xrange(10000):
            x = tensor.vector()
            z = x
            for d in range(10):
                z = tensor.sin(-z+ 1)

            f = function([x], z, mode=Mode(optimizer=None, linker='cvm'))
            if not i % 100:
                print gc.collect()
            sys.stdout.flush()

            gc.collect()
            if 1:
                f([2.0])
                f([3.0])
                f([4.0])
                f([5.0])

    def test_no_leak_many_call_lazy():
        # Verify no memory leaks when calling a function a lot of times

        # This isn't really a unit test, you have to run it and look at top to see
        # if there's a leak

        def build_graph(x, depth=5):
            z = x
            for d in range(depth):
                z = ifelse(z> 0, -z, z)
            return z

        def time_linker(name, linker):
            steps_a = 10
            x = tensor.vector()
            a = build_graph(x, steps_a)

            f_a = function([x], a,
                    mode=Mode(optimizer=None,
                        linker=linker()))

            for i in xrange(100000):
                f_a([2.0])
            if 0: # this doesn't seem to work, prints 0 for everything
                import resource
                pre = resource.getrusage(resource.RUSAGE_SELF)
                post = resource.getrusage(resource.RUSAGE_SELF)
                print pre.ru_ixrss, post.ru_ixrss
                print pre.ru_idrss, post.ru_idrss
                print pre.ru_maxrss, post.ru_maxrss

        time_linker('vmLinker_C', lambda : vm.VM_Linker(allow_gc=False, use_cloop=True))

    def test_no_leak_many_call_nonlazy():
        # Verify no memory leaks when calling a function a lot of times

        # This isn't really a unit test, you have to run it and look at top to see
        # if there's a leak

        def build_graph(x, depth=5):
            z = x
            for d in range(depth):
                z = tensor.sin(-z+1)
            return z

        def time_linker(name, linker):
            steps_a = 10
            x = tensor.vector()
            a = build_graph(x,steps_a)

            f_a = function([x], a,
                    mode=Mode(optimizer=None,
                        linker=linker()))

            for i in xrange(500000):
                f_a([2.0])

        time_linker('vmLinker_C', lambda : vm.VM_Linker(allow_gc=False, use_cloop=True))


