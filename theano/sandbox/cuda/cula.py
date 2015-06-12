import pkg_resources

import theano
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable

try:
    from theano.sandbox.cuda import cuda_ndarray, CudaNdarray
    dimshuffle = cuda_ndarray.cuda_ndarray.dimshuffle
except ImportError:
    pass

cula_available = False

try:
    from scikits.cuda import cula
    cula_available = True
except (ImportError, OSError, pkg_resources.DistributionNotFound):
    pass

cula_initialized = False


class CulaOp(GpuOp):
    def make_thunk(self, *args, **kwargs):
        # Initialize CULA the first time it is needed
        global cula_initialized

        if not cula_available:
            raise RuntimeError('Cula is not available and '
                               'GpuSolve Op can not be constructed.')

        if not cula_initialized:
            cula.culaInitialize()
            cula_initialized = True

        return self._make_thunk(*args, **kwargs)


class GpuSolve(CulaOp):
    """
    CULA GPU solver OP.

    :param trans: Whether to take the transpose of the input matrix
    or not.
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

    def _make_thunk(self, node, storage_map, _, no_recycling=[]):
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


class GpuSyev(CulaOp):
    """
    CULA symmetric real matrix eigendecomposition (LAPACK's SYEV).

    :param: compute_v: Whether to compute the eigenvectors or not.
    :param: lower: Whether to store the lower or upper triangular.

    """
    __props__ = ('jobz', 'uplo')

    def __init__(self, jobz=1, uplo=0):
        self.jobz = ('N', 'V')[jobz]
        # LAPACK expects Fortran order
        self.uplo = ('U', 'L')[1 - uplo]

    def output_type(self):
        return [CudaNdarrayType(broadcastable=[False])(),
                CudaNdarrayType(broadcastable=[False, False])()]

    def make_node(self, inp):
        inp = as_cuda_ndarray_variable(inp)

        assert inp.ndim == 2
        return theano.Apply(self, [inp], self.output_type())

    def _make_thunk(self, node, storage_map, _, no_recycling=[]):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            # Matrix
            A = inputs[0][0]
            n, m = A.shape
            assert m == n
            lda = max(1, n)

            A_copy = A.copy()

            # Allocate array for the eigenvectors
            w = CudaNdarray.zeros((n,))

            cula.culaDeviceSsyev(self.jobz, self.uplo, n, A_copy.gpudata,
                                 lda, w.gpudata)

            # Assign outputs
            outputs[0][0] = w
            outputs[1][0] = dimshuffle(A_copy, (1, 0))

        return thunk


def gpu_syev(a, compute_v=1, lower=0):
    return GpuSyev(compute_v, lower)(a)
