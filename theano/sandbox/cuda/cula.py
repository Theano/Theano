import theano
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           gpu_contiguous)
from theano.tensor import as_tensor_variable
from scikits.cuda import cula


def cula_gpu_solve(A, b, trans='N'):
    cula.culaInitialize()
    A_shape = A.shape
    b_shape = b.shape

    assert(len(A_shape) == 2)
    assert(len(b_shape) == 2)

    import string

    if trans in ['T', 'C']:
        l, n = A_shape
        m, k = b_shape
    elif trans in ['N']:
        n, l = A_shape
        k, m = b_shape
    else:
        raise ValueError('Invalid value for trans')

    if n != k:
        raise ValueError('A and b must be aligned.')

    if trans == 'n':
        lda = max(1, n)
    else:
        lda = max(1, l)

    ldb = max(1, k)

    # construct pointer arrays needed for culaDeviceSgels
    # Cula requires you to pass a pointer for A and b.
    A_ptr = A.gpudata
    b_ptr = b.gpudata

    cula.culaDeviceSgels(trans, n, l, m, A_ptr, lda, b_ptr, ldb)
    return A, b

class GpuSolve(GpuOp):
    """
    Cula Gpu solver OP.
    """
    def __init__(self, trans='N'):
        self.trans = trans
        super(GpuSolve, self).__init__()

    def __eq__(self, other):
        return (type(other) == type(self))

    def __hash__(self):
        return hash(type(self))

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False] * inp.type.ndim)

    def make_node(self, inp1, inp2):
        inp1 = gpu_contiguous(as_cuda_ndarray_variable(inp1))
        inp2 = gpu_contiguous(as_cuda_ndarray_variable(inp2))

        assert inp1.dtype == "float32"
        assert inp2.dtype == "float32"
        assert inp1.ndim == 2
        assert inp2.ndim == 2
        return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

    def make_thunk(self, node, storage_map, _, _2):
        from theano.misc.pycuda_utils import to_gpuarray
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            input_shape = inputs[0][0].shape

            #size of the matrices to invert
            size = input_shape[1]

            z = outputs[0]

            if z[0] is None or z[0].shape != input_shape:
                z[0] = cuda.CudaNdarray.zeros(input_shape)

            #Matrix
            A = inputs[0][0]

            #Solution vectors
            b = inputs[0][1]

            A_pycuda = to_gpuarray(A)
            b_pycuda = to_gpuarray(b)

            cula_gpu_solve(A_pycuda, b_pycuda, self.trans)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

gpu_solve = GpuSolve()
