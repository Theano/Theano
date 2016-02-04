from __future__ import absolute_import, print_function, division
import copy
import gc

import numpy as np

import theano
from theano import tensor
from theano.sandbox import cuda
from theano import ifelse

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
if cuda.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')

# The GC need to be enabled for those tests to work correctly.
if not getattr(mode_with_gpu.linker, 'allow_gc', False):
    mode_with_gpu.linker = copy.copy(mode_with_gpu.linker)
    mode_with_gpu.linker.allow_gc = True


def freemem(extra_alloc=0):
    """
    Return the free memory on the gpu in megabytes.
    """
    gc.collect()
    gc.collect()
    gc.collect()
    n_mallocs = cuda.cuda_ndarray.cuda_ndarray.outstanding_mallocs()

    if hasattr(cuda.cuda_ndarray.cuda_ndarray, "theano_allocated"):
        theano_alloc = cuda.cuda_ndarray.cuda_ndarray.theano_allocated()
        return ("(n malloc/theano mem allocated in KB)",
                n_mallocs + extra_alloc,
                int(theano_alloc / 1024) + extra_size)

    return ("n malloc on the gpu", n_mallocs + extra_alloc)
    # I don't use the following by default as if there is other stuff running
    # on the GPU, this won't work.
    mem_info = cuda.cuda_ndarray.cuda_ndarray.mem_info()
    gpu_used = (mem_info[1] - mem_info[0]) / 1024 ** 2
    mem_info_msg = "(n malloc/gpu mem used in MB)"
    return (mem_info_msg, n_mallocs, int(gpu_used))


def test_memory():
    """
    We test that we do not keep link to memory between Theano function call
    and during Theano compilation

    The origin of this code come from Aaron Vandenoord and Sander Dieleman.
    I have their autorisation to put this in Theano with the Theano license.

    note::
        This test can fail if there is other process running on the gpu.
    """
    shapes = (200, 100)
    # more_alloc1 was different for each dtype in the past.
    # more_alloc2 is still currently not the same for both dtype.
    # when dtype is float32, the computation is done on the gpu.
    # This insert constant on the gpu during compilation
    # that raise the number of alloc.
    # When dtype is float64, only the shared is on the gpu and it is transferd
    # to the cpu for computation. So no extra alloc after compilation.
    # more_alloc1 if after the first compilation, more_alloc2 after the second.
    for dtype, more_alloc1, more_alloc2 in [("float32", 0, 3),
                                            ("float64", 0, 0)]:
        print(dtype)
        test_params = np.asarray(np.random.randn(np.prod(shapes)), dtype)

        some_vector = tensor.vector('some_vector', dtype=dtype)
        some_matrix = some_vector.reshape(shapes)

        mem1 = freemem()
        print("Before shared variable", mem1)
        variables = cuda.shared_constructor(np.ones((shapes[1],),
                                                    dtype='float32'))
        derp = tensor.sum(tensor.dot(some_matrix[:shapes[0]], variables))
        print("Shared took ", np.prod(variables.get_value(
                borrow=True,
                return_internal_type=True).shape) * 4 / 1024, "kB")

        mem2 = freemem()
        print("Before compilation", mem2)
        mem2_1 = freemem(extra_alloc=more_alloc1)
        mem2_2 = freemem(extra_alloc=more_alloc2)
        obj = theano.function([some_vector], derp, mode=mode_with_gpu)
        mem3 = freemem()
        print("After function compilation 1", mem3)
        assert mem2_1 == mem3, (mem2_1, mem3, dtype)

        grad_derp = tensor.grad(derp, some_vector)
        grad = theano.function([some_vector], grad_derp, mode=mode_with_gpu)
        mem4 = freemem()
        print("After function compilation 2", mem4)
        assert mem2_2 == mem4, (mem2_2, mem4, dtype)

        for i in range(3):
            obj(test_params)
            print("After function evaluation 1", freemem())
            assert mem2_2 == freemem(), (mem2_2, freemem())
            grad(test_params)
            print("After function evaluation 2", freemem())
            assert mem2_2 == freemem(), (mem2_2, freemem())

        del obj
        # print "After deleting function 1", freemem()
        #assert mem2 == freemem(), (mem2, freemem())

        del grad
        print("After deleting function 2", freemem())
        assert mem2 == freemem(), (mem2, freemem())

        del derp, variables, grad_derp
        print("After deleting shared variable and ref to it", freemem())
        assert mem1 == freemem(), (mem1, freemem())


@theano.configparser.change_flags(**{'vm.lazy': True})
def test_memory_lazy():
    """As test_memory, but with the ifelse op.

    We need to test it as the ifelse op with the [c]vm create op not
    executed in the graph. This mess with [c]vm gc implementation.
    """
    shapes = (50, 100)
    # more_alloc1 is not the same for both dtype.
    # when dtype is float32, the computation is done on the gpu.
    # This insert constant on the gpu during compilation
    # that raise the number of alloc.
    # When dtype is float64, only the shared is on the gpu and it is transferd
    # to the cpu for computation. So no extra alloc after compilation.
    # more_alloc1 if after the first compilation
    for dtype, more_alloc1 in [("float32", 1),
                               ("float64", 0)]:
        print(dtype)
        test_params = np.asarray(np.random.randn(np.prod(shapes)), dtype)

        some_vector = tensor.vector('some_vector', dtype=dtype)
        some_matrix = some_vector.reshape(shapes)
        branch_select = tensor.iscalar()

        mem1 = freemem()
        print("Before shared variable", mem1)
        variables = cuda.shared_constructor(np.ones((shapes[1],),
                                                    dtype='float32'))
        derp = tensor.sum(tensor.dot(some_matrix[:shapes[0]], variables))
        derp = ifelse.IfElse(1)(branch_select,
                                derp, some_matrix[:shapes[0]].sum())
        derp += 1
        print("Shared took ", np.prod(variables.get_value(
                borrow=True,
                return_internal_type=True).shape) * 4 / 1024, "kB")

        mem2 = freemem()
        print("Before compilation", mem2)
        mem2_1 = freemem(extra_alloc=more_alloc1)
        obj = theano.function([some_vector, branch_select], derp,
                              mode=mode_with_gpu)
        #theano.printing.debugprint(obj, print_type=True)
        mem3 = freemem()
        print("After function compilation 1", mem3)
        assert mem2_1 == mem3, (mem2_1, mem3)

        for i in range(3):
            obj(test_params, 1)
            print("After function evaluation branch true", freemem())
            assert mem2_1 == freemem(), (mem2_1, freemem())
            obj(test_params, 0)
            print("After function evaluation branch false", freemem())
            assert mem2_1 == freemem(), (mem2_1, freemem())

        del obj
        print("After deleting function 1", freemem())
        assert mem2 == freemem(), (mem2, freemem())

        del derp, variables
        print("After deleting shared variable and ref to it", freemem())
        assert mem1 == freemem(), (mem1, freemem())
