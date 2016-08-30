from __future__ import absolute_import, division, print_function

import pkg_resources
import theano

from theano.sandbox.cuda import GpuOp, cuda_available
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
from theano.sandbox.cuda.type import CudaNdarrayType

if cuda_available:
    from theano.sandbox.cuda import CudaNdarray

try:
    from theano.sandbox.cuda import cuda_ndarray
    dimshuffle = cuda_ndarray.cuda_ndarray.dimshuffle
except ImportError:
    pass

cusolver_available = False

try:
    from scikits.cuda import cusolver
    cusolver_available = True
except (ImportError, OSError, RuntimeError, pkg_resources.DistributionNotFound):
    pass

cusolver_handle = None


class GpuCusolverSolve(GpuOp):
    """
    CUSOLVER GPU solver OP.

    Parameters
    ----------
    trans
        Whether to take the transpose of the input matrix or not.

    """

    __props__ = ('trans',)

    def __init__(self, trans='N'):
        self.trans = trans
        super(GpuCusolverSolve, self).__init__()

    def make_node(self, inp1, inp2):
        inp1 = as_cuda_ndarray_variable(inp1)
        inp2 = as_cuda_ndarray_variable(inp2)

        assert inp1.ndim == 2
        assert inp2.ndim == 2
        return theano.Apply(
            self, [inp1, inp2],
            [CudaNdarrayType(broadcastable=[False] * inp1.type.ndim)()])

    def make_thunk(self,
                   node,
                   storage_map, _,
                   no_recycling=[]):
        if not cusolver_available:
            raise RuntimeError('CUSOLVER is not available and '
                               'GpuCusolverSolve Op can not be constructed.')

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            global cusolver_handle

            # Size of the matrices to invert.
            z = outputs[0]

            # Matrix.
            A = inputs[0][0]

            # Solution vectors.
            b = inputs[1][0]

            # A is not explicitly converted between C and F order, instead we
            # switch the "transpose" flag.
            if self.trans in ('T', 'C'):
                trans = 'N'
            else:
                trans = 'T'

            # Convert b to F-order from C-order.
            b_cpy = dimshuffle(b, (1, 0)).reshape((b.shape[0], b.shape[1]))

            # This copy forces allocation of a new C-contiguous buffer
            # and returns it.
            A_cpy = A.copy()
            b_cpy = b_cpy.copy()

            assert(len(A.shape) == 2)
            assert(len(b.shape) == 2)

            if trans in ['T', 'C']:
                trans = 1
                l, n = A.shape
                k, m = b.shape
                if n != k:
                    raise ValueError('A and b must be aligned.')
            elif trans in ['N']:
                trans = 0
                n, l = A.shape
                k, m = b.shape
                if l != m:
                    raise ValueError('A and b must be aligned.')
            else:
                raise ValueError('Invalid value for trans')

            lda = max(1, n)
            ldb = max(1, n, l)

            A_ptr = A_cpy.gpudata
            b_ptr = b_cpy.gpudata

            if cusolver_handle is None:
                cusolver_handle = cusolver.cusolverDnCreate()

            workspace_size = cusolver.cusolverDnSgetrf_bufferSize(
                cusolver_handle, m, n, A_ptr, lda)

            if (thunk.workspace is None or
                    thunk.workspace.size != workspace_size):
                thunk.workspace = CudaNdarray.zeros((workspace_size,))

            if thunk.pivots is None or thunk.pivots.size != min(m, n):
                thunk.pivots = CudaNdarray.zeros((min(m, n),))

            if thunk.dev_info is None:
                thunk.dev_info = CudaNdarray.zeros((1,))

            workspace_ptr = thunk.workspace.gpudata
            pivots_ptr = thunk.pivots.gpudata
            dev_info_ptr = thunk.dev_info.gpudata

            cusolver.cusolverDnSgetrf(
                cusolver_handle, n, l, A_ptr, lda, workspace_ptr,
                pivots_ptr, dev_info_ptr)

            cusolver.cusolverDnSgetrs(
                cusolver_handle, trans, n, m, A_ptr, lda,
                pivots_ptr, b_ptr, ldb, dev_info_ptr)

            # Convert b to F-order from C-order and assign it to output.
            b_cpy = b_cpy.reshape(b.shape[::-1])
            b_cpy = dimshuffle(b_cpy, (1, 0))
            z[0] = b_cpy

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        thunk.workspace = None
        thunk.pivots = None
        thunk.dev_info = None

        return thunk

gpu_solve = GpuCusolverSolve()
