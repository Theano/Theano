import ctypes
import theano
from theano.sandbox.cuda import (CudaNdarrayType, as_cuda_ndarray_variable,
                                 GpuOp, cuda_ndarray, CudaNdarray)
try:
    from scikits.cuda.cusolver import (_libcusolver, cusolverDnCreate,
                                       cusolverCheckStatus)
    from scikits.cuda.cublas import _CUBLAS_SIDE_MODE, _CUBLAS_OP
    cusolver_available = True
except (ImportError, OSError):
    cusolver_available = False
    raise

dimshuffle = cuda_ndarray.cuda_ndarray.dimshuffle


class CuSolverOp(GpuOp):
    """cuSOLVER ops lazily raise an error when cuSOLVER is not available."""
    def make_thunk(self, *args, **kwargs):
        if not cusolver_available:
            raise RuntimeError('cuSOLVER not available')
        return self._make_thunk(*args, **kwargs)


class GpuGeqrf(CuSolverOp):
    """Wrapper of cuSOLVER's implementation of the geqrf LAPACK method.

    Notes
    -----
    Wrapping of the methods is done as in `scikits.cuda.cusolver`. The
    interface is intendend to mimick `scipy.linalg.lapack`.

    """
    __props__ = ()

    def output_type(self):
        return [CudaNdarrayType(broadcastable=[False, False])(),
                CudaNdarrayType(broadcastable=[False])(),
                CudaNdarrayType(broadcastable=[False])(),
                CudaNdarrayType(broadcastable=[])()]

    def make_node(self, inp):
        inp = as_cuda_ndarray_variable(inp)
        assert inp.ndim == 2

        return theano.Apply(self, [inp], self.output_type())

    def _make_thunk(self, node, storage_map, _, no_recycling=[]):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            A = inputs[0][0]
            m, n = A.shape
            # LAPACK routines expect Fortran order
            A_T = dimshuffle(A, (1, 0))
            A_copy = A_T.copy()

            lda = max(1, m)
            TAU = CudaNdarray.zeros((min(m, n),))
            Lwork = cusolverDnSgeqrf_bufferSize(handle, m, n, A_copy.gpudata,
                                                lda)
            Work = CudaNdarray.zeros((max(1, Lwork),))
            devInfo = CudaNdarray.zeros((1,))
            cusolverDnSgeqrf(handle, m, n, A_copy.gpudata, lda, TAU.gpudata,
                             Work.gpudata, Lwork, devInfo.gpudata)
            outputs[0][0] = dimshuffle(A_copy, (1, 0))
            outputs[1][0] = TAU
            outputs[2][0] = Work
            outputs[3][0] = devInfo

        return thunk


gpu_geqrf = GpuGeqrf()


class GpuOrmqr(CuSolverOp):
    """Wrapper of cuSOLVER's implementation of the ormqr LAPACK method.

    Notes
    -----
    Wrapping of the methods is done as in `scikits.cuda.cusolver`. The
    interface is intendend to mimick `scipy.linalg.lapack`.

    """
    __props__ = ()

    def __init__(self, side, trans, Lwork):
        self.side = side
        self.trans = trans
        self.Lwork = Lwork
        super(GpuOrmqr, self).__init__()

    def output_type(self):
        return [CudaNdarrayType(broadcastable=[False, False])(),
                CudaNdarrayType(broadcastable=[False])(),
                CudaNdarrayType(broadcastable=[])()]

    def make_node(self, inp1, inp2, inp3):
        inp1 = as_cuda_ndarray_variable(inp1)
        inp2 = as_cuda_ndarray_variable(inp2)
        inp3 = as_cuda_ndarray_variable(inp3)

        assert inp1.ndim == 2
        assert inp2.ndim == 1
        assert inp3.ndim == 2

        return theano.Apply(self, [inp1, inp2, inp3], self.output_type())

    def _make_thunk(self, node, storage_map, _, no_recycling=[]):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            A = inputs[0][0]
            TAU = inputs[1][0]
            C = inputs[2][0]
            lda, k = A.shape
            m, n = C.shape
            ldc = max(1, m)

            # LAPACK routines expect Fortran order
            A_T = dimshuffle(A, (1, 0))
            A_copy = A_T.copy()
            C_T = dimshuffle(C, (1, 0))
            C_copy = C_T.copy()

            assert self.side in ('L', 'R')
            assert self.trans in ('N', 'T')
            assert ldc >= max(1, m)
            if self.side == 'L':
                assert lda >= max(1, m)
                assert self.Lwork >= max(1, n)
                assert m >= k >= 0
            else:
                assert lda >= max(1, n)
                assert self.Lwork >= max(1, m)
                assert n >= k >= 0

            Work = CudaNdarray.zeros((self.Lwork,))
            devInfo = CudaNdarray.zeros((1,))

            cusolverDnSormqr(handle, _CUBLAS_SIDE_MODE[self.side],
                             _CUBLAS_OP[self.trans], m, n, k,
                             A_copy.gpudata, lda, TAU.gpudata, C_copy.gpudata,
                             ldc, Work.gpudata, self.Lwork, devInfo.gpudata)

            outputs[0][0] = dimshuffle(C_copy, (1, 0))
            outputs[1][0] = Work
            outputs[2][0] = devInfo

        return thunk


def gpu_ormqr(side, trans, a, tau, c, lwork):
    return GpuOrmqr(side, trans, lwork)(a, tau, c)


if cusolver_available:
    # Creating handle is costly, so re-use it for all ops
    # NOTE We don't destroy it using cusolverDnDestroy
    handle = cusolverDnCreate()

    _libcusolver.cusolverDnSgeqrf.restype = int
    _libcusolver.cusolverDnSgeqrf.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p]

    def cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Work, Lwork, devInfo):
        status = _libcusolver.cusolverDnSgeqrf(handle, m, n, int(A), lda,
                                               int(TAU), int(Work),
                                               Lwork, int(devInfo))
        cusolverCheckStatus(status)

    _libcusolver.cusolverDnSgeqrf_bufferSize.restype = int
    _libcusolver.cusolverDnSgeqrf_bufferSize.argtypes = [ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p,
                                                         ctypes.c_int,
                                                         ctypes.c_void_p]

    def cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda):
        Lwork = ctypes.c_int()
        status = _libcusolver.cusolverDnSgeqrf_bufferSize(handle, m, n, int(A),
                                                          lda,
                                                          ctypes.byref(Lwork))
        cusolverCheckStatus(status)
        return Lwork.value

    _libcusolver.cusolverDnSormqr.restype = int
    _libcusolver.cusolverDnSormqr.argtypes = [ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p,
                                              ctypes.c_int,
                                              ctypes.c_void_p]

    def cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc,
                         work, lwork, devInfo):
        status = _libcusolver.cusolverDnSormqr(handle, side, trans, m, n, k,
                                               int(A), lda, int(tau), int(C),
                                               ldc, int(work), lwork,
                                               int(devInfo))
        cusolverCheckStatus(status)
