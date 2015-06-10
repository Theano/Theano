import ctypes
import theano
from theano.sandbox.cuda import (CudaNdarrayType, as_cuda_ndarray_variable,
                                 GpuOp, cuda_ndarray, CudaNdarray)
from theano.tensor import reshape
try:
    from scikits.cuda.cusolver import (_libcusolver, cusolverDnCreate,
                                       cusolverCheckStatus)
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
    interface is made to match that of `scipy.linalg.lapac. The interface
    is intendend to mimick `scipy.linalg.lapack`.

    """
    __props__ = ()

    def output_type(self, inp):
        return [CudaNdarrayType(broadcastable=[False] * inp.type.ndim)(),
                CudaNdarrayType(broadcastable=[False])(),
                CudaNdarrayType(broadcastable=[False])(),
                CudaNdarrayType(broadcastable=[])()]

    def make_node(self, inp):
        inp = as_cuda_ndarray_variable(inp)
        assert inp.ndim == 2

        return theano.Apply(self, [inp], self.output_type(inp))

    def _make_thunk(self, node, storage_map, _, no_recycling=[]):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            A = inputs[0][0]
            m, n = A.shape
            # LAPACK routines expect Fortran order
            A_T = dimshuffle(A, (1, 0)).reshape((m, n))
            A_copy = A_T.copy()

            lda = max(1, m)
            TAU = CudaNdarray.zeros((min(m, n),))
            Lwork = cusolverDnSgeqrf_bufferSize(handle, m, n, A_copy.gpudata,
                                                lda)
            Work = CudaNdarray.zeros((max(1, Lwork),))
            devInfo = CudaNdarray.zeros((1,))
            cusolverDnSgeqrf(handle, m, n, A_copy.gpudata, lda, TAU.gpudata,
                             Work.gpudata, Lwork, devInfo.gpudata)
            outputs[0][0] = reshape(A_copy, (n, m)).dimshuffle((1, 0))
            outputs[1][0] = TAU
            outputs[2][0] = Work
            outputs[3][0] = devInfo

        return thunk


geqrf = GpuGeqrf()


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
