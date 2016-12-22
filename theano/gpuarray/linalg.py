from __future__ import absolute_import, division, print_function

import pkg_resources
import theano

from theano import Op
from theano.gpuarray import basic_ops, GpuArrayType

try:
    from pygpu import gpuarray
except ImportError:
    pass

cusolver_available = False
try:
    from skcuda import cusolver
    cusolver_available = True
except (ImportError, OSError, RuntimeError, pkg_resources.DistributionNotFound):
    pass

cusolver_handle = None


class GpuCusolverSolve(Op):
    """
    CUSOLVER GPU solver OP.

    Parameters
    ----------
    trans
        Whether to take the transpose of the input matrix or not.

    """

    __props__ = ('trans',)

    def __init__(self, trans='N', inplace=False):
        self.trans = trans
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0, 1]}
        super(GpuCusolverSolve, self).__init__()

    def make_node(self, inp1, inp2):
        self.context = basic_ops.infer_context_name(inp1, inp2)

        inp1 = basic_ops.as_gpuarray_variable(inp1, self.context)
        inp2 = basic_ops.as_gpuarray_variable(inp2, self.context)

        inp1 = basic_ops.gpu_contiguous(inp1)
        inp2 = basic_ops.gpu_contiguous(inp2)

        # this op can only operate on float32 matrices
        assert inp1.ndim == 2
        assert inp2.ndim == 2
        assert inp1.dtype == 'float32'
        assert inp2.dtype == 'float32'

        return theano.Apply(
            self, [inp1, inp2],
            [GpuArrayType('float32',
                          broadcastable=inp1.broadcastable,
                          context_name=self.context)()])

    def make_thunk(self,
                   node,
                   storage_map, _,
                   no_recycling=[],
                   impl=None):
        if not cusolver_available:
            raise RuntimeError('CUSOLVER is not available and '
                               'GpuCusolverSolve Op can not be constructed.')

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        global cusolver_handle
        if cusolver_handle is None:
            cusolver_handle = cusolver.cusolverDnCreate()

        def thunk():
            context = inputs[0][0].context

            # Size of the matrices to invert.
            z = outputs[0]

            # Matrix.
            A = inputs[0][0]

            # Solution vectors.
            b = inputs[1][0]

            assert(len(A.shape) == 2)
            assert(len(b.shape) == 2)

            if self.trans in ['T', 'C']:
                trans = 1
                l, n = A.shape
                k, m = b.shape
            elif self.trans == 'N':
                trans = 0
                n, l = A.shape
                k, m = b.shape
            else:
                raise ValueError('Invalid value for trans')
            if l != n:
                raise ValueError('A must be a square matrix')
            if n != k:
                raise ValueError('A and b must be aligned.')

            lda = max(1, n)
            ldb = max(1, k, m)

            # We copy A and b as cusolver operates inplace
            b = gpuarray.array(b, copy=True, order='F')
            if not self.inplace:
                A = gpuarray.array(A, copy=True)
            A_ptr = A.gpudata
            b_ptr = b.gpudata

            # cusolver expects a F ordered matrix, but A is not explicitly
            # converted between C and F order, instead we switch the
            # "transpose" flag.
            if A.flags['C_CONTIGUOUS']:
                trans = 1 - trans

            workspace_size = cusolver.cusolverDnSgetrf_bufferSize(
                cusolver_handle, n, n, A_ptr, lda)

            if (thunk.workspace is None or
                    thunk.workspace.size != workspace_size):
                thunk.workspace = gpuarray.zeros((workspace_size,),
                                                 dtype='float32',
                                                 context=context)

            if thunk.pivots is None or thunk.pivots.size != min(n, n):
                thunk.pivots = gpuarray.zeros((min(n, n),),
                                              dtype='float32',
                                              context=context)

            if thunk.dev_info is None:
                thunk.dev_info = gpuarray.zeros((1,),
                                                dtype='float32',
                                                context=context)

            workspace_ptr = thunk.workspace.gpudata
            pivots_ptr = thunk.pivots.gpudata
            dev_info_ptr = thunk.dev_info.gpudata

            cusolver.cusolverDnSgetrf(
                cusolver_handle, n, n, A_ptr, lda, workspace_ptr,
                pivots_ptr, dev_info_ptr)

            cusolver.cusolverDnSgetrs(
                cusolver_handle, trans, n, m, A_ptr, lda,
                pivots_ptr, b_ptr, ldb, dev_info_ptr)

            z[0] = b

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        thunk.workspace = None
        thunk.pivots = None
        thunk.dev_info = None

        return thunk


def gpu_solve(A, b, trans='N'):
    return GpuCusolverSolve(trans)(A, b)
