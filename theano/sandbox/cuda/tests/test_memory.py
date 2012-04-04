import gc

import numpy as np

import theano
from theano import tensor
from theano.sandbox import cuda

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
if cuda.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')


if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


def freemem():
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
                n_mallocs, int(theano_alloc / 1024))

    return ("n malloc on the gpu", n_mallocs)
    # I don't use the following by default as if there is other stuff running
    # on the GPU, this won't work.
    mem_info = cuda.cuda_ndarray.cuda_ndarray.mem_info()
    gpu_used = (mem_info[1] - mem_info[0]) / 1024 ** 2
    mem_info_msg = "(n malloc/gpu mem used in MB)"
    return ("(n malloc/gpu mem used in MB)", n_mallocs, int(gpu_used))


def test_memory():
    """
    We test that we do not keep link to memory between Theano function call
    and during Theano compilation

    The origin of this code come from Aaron Vandenoord and Sander Dieleman.
    I have their autorisation to put this in Theano with the Theano license.

    note::
        This test can fail if there is other process running on the gpu.
    """
    shapes = (6000, 5000)
    test_params = np.asarray(np.random.randn(np.prod(shapes)), 'float32')

    some_vector = tensor.vector('some_vector')
    some_matrix = some_vector.reshape(shapes)

    mem1 = freemem()
    print "Before shared variable", mem1
    variables = cuda.shared_constructor(np.ones((shapes[1],), dtype='float32'))
    derp = tensor.sum(tensor.dot(some_matrix[:shapes[0]], variables))
    print "Shared took ", np.prod(variables.get_value(
            borrow=True,
            return_internal_type=True).shape) * 4 / 1024, "kB"

    mem2 = freemem()
    print "Before compilation", mem2
    obj = theano.function([some_vector], derp, mode=mode_with_gpu)
    mem3 = freemem()
    print "After function compilation 1", mem3
    assert mem2 == mem3, (mem2, mem3)

    grad_derp = tensor.grad(derp, some_vector)
    grad = theano.function([some_vector], grad_derp, mode=mode_with_gpu)
    mem4 = freemem()
    print "After function compilation 2", mem4
    assert mem2 == mem4, (mem2, mem4)

    for i in range(3):
        obj(test_params)
        print "After function evaluation 1", freemem()
        assert mem2 == freemem(), (mem2, freemem())
        grad(test_params)
        print "After function evaluation 2", freemem()
        assert mem2 == freemem(), (mem2, freemem())

    del obj
    print "After deleting function 1", freemem()
    assert mem2 == freemem(), (mem2, freemem())

    del grad
    print "After deleting function 2", freemem()
    assert mem2 == freemem(), (mem2, freemem())

    del derp, variables, grad_derp
    print "After deleting shared variable and ref to it", freemem()
    assert mem1 == freemem(), (mem1, freemem())
