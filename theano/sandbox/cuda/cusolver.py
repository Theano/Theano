import ctypes
import theano
from theano.sandbox.cuda import (CudaNdarrayType, as_cuda_ndarray_variable,
                                 GpuOp, cuda_ndarray, CudaNdarray)
from scikits.cuda.cusolver import (_libcusolver, cusolverDnCreate,
                                   cusolverCheckStatus, cusolverDnDestroy)

dimshuffle = cuda_ndarray.cuda_ndarray.dimshuffle


class GpuGeqrf(GpuOp):
    __props__ = ()

    def output_type(self, inp):
        return [CudaNdarrayType(broadcastable=[False] * inp.type.ndim)(),
                CudaNdarrayType(broadcastable=[False])()]

    def make_node(self, inp):
        inp = as_cuda_ndarray_variable(inp)
        assert inp.ndim == 2

        return theano.Apply(self, [inp], self.output_type(inp))

    def make_thunk(self, node, storage_map, _, no_recycling=[]):
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
            handle = cusolverDnCreate()
            Lwork = cusolverDnSgeqrf_bufferSize(handle, m, n, A_copy.gpudata,
                                                lda)
            Work = CudaNdarray.zeros((max(1, Lwork),))
            devInfo = CudaNdarray.zeros((1,))
            cusolverDnSgeqrf(handle, m, n, A_copy.gpudata, lda, TAU.gpudata,
                             Work.gpudata, Lwork, devInfo.gpudata)
            cusolverDnDestroy(handle)
            outputs[0][0] = A_copy
            outputs[1][0] = TAU

        return thunk


geqrf = GpuGeqrf()


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
                                                      lda, ctypes.byref(Lwork))
    cusolverCheckStatus(status)
    return Lwork.value
