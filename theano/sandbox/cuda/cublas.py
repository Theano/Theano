import pkg_resources
import atexit

import theano
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable

try:
    from theano.sandbox.cuda import cuda_ndarray
    dimshuffle = cuda_ndarray.cuda_ndarray.dimshuffle
except ImportError:
    pass

cublas_available = False
cublas_initialized = False
cublas_handle = None

try:
    from skcuda import cublas
    cublas_available = True
except (ImportError, OSError, RuntimeError,
        pkg_resources.DistributionNotFound):
    pass


def initialize_cublas():
    """ Initializes CUBLAS handle and ensures resource release on exit. """
    global cublas_initialized
    global cublas_handle
    cublas_handle = cublas.cublasCreate()
    cublas_initialized = True
    atexit.register(release_cublas_resources)


def release_cublas_resources():
    global cublas_initialized
    if cublas_initialized:
        cublas.cublasDestroy(cublas_handle)
        cublas_initialized = False


class GpuTriangularSolve(GpuOp):
    """
    CUBLAS GPU triangular linear system solver OP.
    Parameters
    ----------
    trans
        Whether to take the transpose of the input matrix or not.
    lower
        Whether system is lower-triangular (True) or upper-triangular (False).
    """

    __props__ = ('trans', 'lower',)

    def __init__(self, trans='N', lower=True):
        self.trans = trans
        self.lower = lower
        super(GpuTriangularSolve, self).__init__()

    def output_type(self, inp):
        return CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_node(self, inp1, inp2):
        inp1 = as_cuda_ndarray_variable(inp1)
        inp2 = as_cuda_ndarray_variable(inp2)

        assert inp1.ndim == 2
        assert inp2.ndim in [1, 2]
        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def make_thunk(self,
                   node,
                   storage_map, _,
                   no_recycling=[]):

        # Initialize CULA the first time it is needed

        if not cublas_available:
            raise RuntimeError('Cublas is not available and '
                               'GpuTriangularSolve Op can not be constructed.')

        global cublas_initialized

        if not cublas_initialized:
            initialize_cublas()

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():

            # Solution vector(s) of system to return as output.
            x = outputs[0]

            # Matrix
            A = inputs[0][0]

            # Vector(s) to solve system for.
            b = inputs[1][0]

            # A is not explicitly converted between C and F order, instead we
            # switch the "transpose" and "lower" flags
            if self.trans in ('T', 'C'):
                trans = 'N'
            else:
                trans = 'T'
            lower = not self.lower

            # If b is one-dimensional reshape to two-dimensional array with
            # singleton second dimension
            if b.ndim == 1:
                b_2d = b.reshape((b.shape[0], 1))
            else:
                b_2d = b

            # Convert b to F-order from c-order.
            b_cpy = dimshuffle(b_2d, (1, 0)).reshape((b_2d.shape[0],
                                                      b_2d.shape[1]))

            # This copy forces allocation of a new C-contiguous buffer
            # and returns it.
            A_cpy = A.copy()
            b_cpy = b_cpy.copy()

            def cublas_gpu_triangular_solve(A_, b_, trans='T', lower=False):

                A_shape = A_.shape
                b_shape = b_.shape

                assert(len(A_shape) == 2)
                assert(len(b_shape) == 2)

                if trans in ['T', 'C']:
                    l, n = A_shape
                    k, m = b_shape
                    if n != k:
                        raise ValueError('A and b must be aligned.')
                elif trans in ['N']:
                    n, l = A_shape
                    k, m = b_shape
                    if l != m:
                        raise ValueError('A and b must be aligned.')
                else:
                    raise ValueError('Invalid value for trans')

                if l != n:
                    raise ValueError('A must be square.')

                lda = max(1, n)
                ldb = max(1, n)

                # construct pointer arrays needed for culaDeviceSgels
                # Cula requires you to pass a pointer for A and b.
                A_ptr = A_.gpudata
                b_ptr = b_.gpudata

                alpha = 1.0  # unit scalar used for multiplicatio
                side = 'l'  # indicates matrix A is on right of B
                uplo = 'l' if lower else 'u'  # set whether upper or lower
                                              # part of matrix A stored
                diag = 'n'  # indicates elements on diagonal of matrix A may
                            # not be unity

                cublas.cublasStrsm(cublas_handle, side, uplo, trans, diag,
                                   n, m, alpha, A_ptr, lda, b_ptr, ldb)

                return A_, b_

            A_pycuda, b_pycuda = cublas_gpu_triangular_solve(
                A_cpy, b_cpy, trans, lower)

            # Convert b to F-order from c-order:
            b_cpy = b_cpy.reshape(b_2d.shape[::-1])
            b_cpy = dimshuffle(b_cpy, (1, 0))

            # If b was originally a one-dimensional array reshape back
            if b.ndim == 1:
                b_cpy = b_cpy.reshape(b.shape[0])

            # Assign result to output
            x[0] = b_cpy

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

gpu_lower_triangular_solve = GpuTriangularSolve(trans='N', lower=True)
gpu_upper_triangular_solve = GpuTriangularSolve(trans='N', lower=False)
