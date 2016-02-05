import pkg_resources

import theano
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable

try:
    from theano.sandbox.cuda import cuda_ndarray
    dimshuffle = cuda_ndarray.cuda_ndarray.dimshuffle
except ImportError:
    pass

cula_available = False

try:
    from skcuda import cula
    from skcuda import linalg
    cula_available = True
except (ImportError, OSError, RuntimeError,
        pkg_resources.DistributionNotFound):
    pass

cula_initialized = False


class GpuSolve(GpuOp):
    """
    CULA GPU solver OP.

    Parameters
    ----------
    trans
        Whether to take the transpose of the input matrix or not.

    """

    __props__ = ('trans',)

    def __init__(self, trans='N'):
        self.trans = trans
        super(GpuSolve, self).__init__()

    def output_type(self, inp):
        return CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_node(self, inp1, inp2):
        inp1 = as_cuda_ndarray_variable(inp1)
        inp2 = as_cuda_ndarray_variable(inp2)

        assert inp1.ndim == 2
        assert inp2.ndim == 2
        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def make_thunk(self,
                   node,
                   storage_map, _,
                   no_recycling=[]):

        # Initialize CULA the first time it is needed
        global cula_initialized

        if not cula_available:
            raise RuntimeError('Cula is not available and '
                               'GpuSolve Op can not be constructed.')

        if not cula_initialized:
            cula.culaInitialize()
            cula_initialized = True

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            # size of the matrices to invert
            z = outputs[0]

            # Matrix
            A = inputs[0][0]

            # Solution vectors
            b = inputs[1][0]

            # A is not explicitly converted between C and F order, instead we
            # switch the "transpose" flag
            if self.trans in ('T', 'C'):
                trans = 'N'
            else:
                trans = 'T'

            # Convert b to F-order from c-order.
            b_cpy = dimshuffle(b, (1, 0)).reshape((b.shape[0], b.shape[1]))

            # This copy forces allocation of a new C-contiguous buffer
            # and returns it.
            A_cpy = A.copy()
            b_cpy = b_cpy.copy()

            def cula_gpu_solve(A_, b_, trans='T'):

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

                lda = max(1, n)
                ldb = max(1, n, l)

                # construct pointer arrays needed for culaDeviceSgels
                # Cula requires you to pass a pointer for A and b.
                A_ptr = A_.gpudata
                b_ptr = b_.gpudata

                cula.culaDeviceSgels(trans, n, l, m, A_ptr, lda, b_ptr, ldb)
                return A_, b_

            A_pycuda, b_pycuda = cula_gpu_solve(A_cpy, b_cpy, trans)

            # Convert b to F-order from c-order and assign it to output:
            b_cpy = b_cpy.reshape(b.shape[::-1])
            b_cpy = dimshuffle(b_cpy, (1, 0))
            z[0] = b_cpy

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

gpu_solve = GpuSolve()


class GpuCholesky(GpuOp):
    """
    CULA GPU Choleksy factorisation op.

    Given a real positive definite matrix `A` returns either a lower
    triangular matrix `L` such that `A == dot(L, L.T)` if `lower == True`
    else returns an upper triangular matrix `U` such that `A == dot(U.T, U)`
    if `lower == False`.

    Parameters
    ----------
    lower
        Whether to return a lower rather than upper triangular decomposition.

    """

    __props__ = ('lower',)

    def __init__(self, lower=True):
        self.lower = lower
        super(GpuCholesky, self).__init__()

    def output_type(self, inp):
        return CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_node(self, inp):
        inp = as_cuda_ndarray_variable(inp)

        assert inp.ndim == 2
        return theano.Apply(self, [inp], [self.output_type(inp)()])

    def make_thunk(self,
                   node,
                   storage_map, _,
                   no_recycling=[]):

        # Initialize CULA the first time it is needed
        global cula_initialized

        if not cula_available:
            raise RuntimeError('Cula is not available and '
                               'GpuCholesky Op can not be constructed.')

        if not cula_initialized:
            cula.culaInitialize()
            cula_initialized = True

        # import util function here to avoid circular import errors
        from theano.misc.pycuda_utils import to_gpuarray

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():

            # Matrix to decompose is first (and only) input
            A = inputs[0][0]

            # Cholesky decomposition of matrix is first (and only) output
            A_chol = outputs[0]

            # CULA Cholesky function is destructive as it assigns calculated
            # decomposition to lower / upper triangle of input array therefore
            # force copy of array with allocation of a new C-contiguous buffer.
            # CULA operation assumes Fortran ordering however input matrix must
            # be symmetric therefore no need to alter order.
            A_cpy = A.copy()

            def cula_gpu_cholesky(A_, lower=True):

                A_shape = A_.shape

                # Decomposition only valid for matrix inputs.
                if len(A_shape) != 2:
                    raise ValueError('A must be two-dimensional.')

                m, n = A_shape

                # Decomposition only exists for square matrix inputs.
                if m != n:
                    raise ValueError('A must be square.')

                # Get pointer to A array data needed for culaDeviceSpotrf.
                A_ptr = A_.gpudata

                # culaDeviceSpotrf takes upper-lower argument as string.
                uplo = 'L' if lower else 'U'

                # Leading dimension of input array A.
                lda = max(1, n)

                # CULA may raise exception if input matrix found not to be
                # positive-definite therefore make sure buffers freed even
                # in error cases.
                try:
                    cula.culaDeviceSpotrf(uplo, n, A_ptr, lda)
                finally:
                    cula.culaFreeBuffers()

            # Decomposition assigned in-place to A_cpy
            cula_gpu_cholesky(A_cpy, self.lower)

            # CULA assumes Fortran ordering and assigns the calculated
            # decomposition to the relevant triangle of A_cpy (i.e. lower
            # triangle if lower=True and upper otherwise) in Fortran ordering.
            # The remaining elements in A_cpy are left as the original values
            # in A i.e. A_cpy is not enforced to be lower / upper triangular.
            # Therefore use skcuda.linalg triu/tril functions to extract
            # relevant triangle elements only, others set to zero. These
            # require a pycuda GPUArray type as input therefore use util
            # function to convert CudaNdArray -> GPUArray, applying this
            # before any transpose operation to ensure A_cpy buffer is still
            # C-contiguous as so can be shared with GPUArray object without
            # any copy required this meaning triu/tril operations occur in
            # place on A_cpy buffer.
            if self.lower:
                # extract only upper triangle in C-ordering i.e. lower triangle
                # in F-ordering
                linalg.triu(to_gpuarray(A_cpy), overwrite=True)
            else:
                # extract only lower triangle in C-ordering i.e. upper triangle
                # in F-ordering
                linalg.tril(to_gpuarray(A_cpy), overwrite=True)
            # Assign output as transposed array to move from F to C ordering.
            A_chol[0] = dimshuffle(A_cpy, (1, 0))

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

gpu_cholesky = lambda A, lower=True: GpuCholesky(lower)(A)
