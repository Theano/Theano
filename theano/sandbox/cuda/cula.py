import theano
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp, CudaNdarray

from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)

from theano.tensor import as_tensor_variable
from scikits.cuda import cula

try:
    from scikits.cuda import cula
    scikits_cuda_available = True
except ImportError:
    scikits_cuda_available = False

if cula is not None:
    cula.culaInitialize()

import numpy

class GpuSolve(GpuOp):
    """
    CULA GPU solver OP.

    """
    def __init__(self, trans='N'):
        self.trans = trans
        super(GpuSolve, self).__init__()

    def __eq__(self, other):
        return (type(other) == type(self))

    def __hash__(self):
        return hash(type(self))

    def output_type(self, inp):
        return CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_node(self, inp1, inp2):
        inp1 = as_cuda_ndarray_variable(inp1)
        inp2 = as_cuda_ndarray_variable(inp2)

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 2
        assert inp2.ndim == 2
        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def make_thunk(self, node, storage_map, _, no_recycling=[]):
        from theano.misc.pycuda_utils import to_gpuarray

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            input_shape = inputs[1][0].shape

            #size of the matrices to invert
            z = outputs[0]

            #Matrix
            A = inputs[0][0]

            #Solution vectors
            b = inputs[1][0]
            A_cpy = A.copy()
            b_cpy = b.copy()

            A_pycuda = to_gpuarray(A_cpy)
            b_pycuda = to_gpuarray(b_cpy)

            def cula_gpu_solve(A, b):

                A_shape = A.shape
                b_shape = b.shape
                assert(len(A_shape) == 2)
                assert(len(b_shape) == 2)

                if A_shape[0] != A_shape[1]:
                    raise ValueError('Coefficient matrix should be a square matrix.')

                n = A_shape[0]
                nrhs = b_shape[1]

                #Create the integer pivot vector to store the indices for
                #permutation matrix.
                ipiv = CudaNdarray.zeros((n,))
                ipiv = to_gpuarray(ipiv)

                import string
                lda = max(1, n)
                ldb = max(1, n)

                # construct pointer arrays needed for culaDeviceSgels
                # Cula requires you to pass a pointer for A and b.
                A_ptr = A_cpy.gpudata
                b_ptr = b_cpy.gpudata
                ipiv_ptr = ipiv.gpudata

                cula.culaDeviceSgesv(n, nrhs, A_ptr, lda, ipiv_ptr, b_ptr, ldb)
                return A, b

            A_pycuda, b_pycuda = cula_gpu_solve(A_pycuda, b_pycuda)
            z[0] = b

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

gpu_solve = GpuSolve()
