from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar
import StringIO

import cuda_ndarray
from .type import CudaNdarrayType

class GpuCrossentropySoftmaxArgmax1HotWithBias (Op):
    nin=3
    nout=3
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return self.__class__.__name__
    def make_node(self, x, b, y_idx):
        nll = y_idx.type() #N.B. won't work when we don't cast y_idx to float anymore
        sm = x.type()
        am = y_idx.type()
        return Apply(self, [x, b, y_idx], [nll, sm, am])

    def c_support_code(self):
        return """
        __global__ void k_xent_sm_1hot_bias(int M, int N,
            const float * x_data, int xs0, int xs1,
            const float * b, int bs0,
            const float * y_idx_data, int y_idxs0,
            float * nll_data, int nlls0,
            float * sm_data, int sms0, int sms1,
            float * am_data, int ams0)
        {
            const int row = blockIdx.x;

            const float * x = x_data + xs0 * row;
            const int y_idx = (int)y_idx_data[row * y_idxs0];
            float * sm = sm_data + sms0 * row;

            float sum = 0.0;
            int row_max_j = 0;
            float row_max = x[0] + b[0];
            for (int j = 1; j < N; ++j)
            {
                float row_ij = x[j*xs1] + b[j*bs0];
                //todo: store to shared memory
                row_max_j = (row_ij > row_max) ? j : row_max_j;
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }
            //compute the exp
            for (int j = 0; j < N; ++j)
            {
                float row_ij = x[j*xs1] + b[j*bs0];
                float sm_ij = exp(row_ij - row_max);
                sum += sm_ij;
                sm[j * sms1] = sm_ij;
            }
            float sum_inv = 1.0 / sum;
            for (int j = 0; j < N; ++j)
            {
                sm[j * sms1] *= sum_inv;
            }
            if ((y_idx >= N) || (y_idx < 0))
            {
                //TODO: set raise an error bit in a global var?
                nll_data[row*nlls0] = 0.0; // raise some suspicion at least...
            }
            else
            {
                nll_data[row*nlls0] = - x[y_idx*xs1]
                           - b[y_idx*bs0]
                           + row_max
                           + log(sum);
            }
            am_data[row*ams0] = row_max_j;
        }

        """

    def c_code(self, node, nodename, (x, b, y_idx), (nll, sm, am), sub):
        classname=self.__class__.__name__
        fail = sub['fail']
        sio = StringIO.StringIO()
        print >> sio, """
        if (cnda_%(y_idx)s->nd != 1)
        {
            PyErr_SetString(PyExc_ValueError, "y_idx not 1d tensor");
            %(fail)s;
        }
        if (cnda_%(x)s->nd != 2)
        {
            PyErr_SetString(PyExc_ValueError, "x not 2d tensor");
            %(fail)s;
        }
        if (cnda_%(b)s->nd != 1)
        {
            PyErr_SetString(PyExc_ValueError, "b not 1d tensor");
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(cnda_%(x)s)[0] != CudaNdarray_HOST_DIMS(cnda_%(y_idx)s)[0])
        {
            PyErr_SetString(PyExc_ValueError, "dimension mismatch in x,y_idx arguments");
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(cnda_%(x)s)[1] != CudaNdarray_HOST_DIMS(cnda_%(b)s)[0])
        {
            PyErr_SetString(PyExc_ValueError, "dimension mismatch in x,b arguments");
            %(fail)s;
        }
        if ((NULL == cnda_%(nll)s) //initial condition
            || (CudaNdarray_HOST_DIMS(cnda_%(nll)s)[0] != CudaNdarray_HOST_DIMS(cnda_%(y_idx)s)[0]))
        {
            Py_XDECREF(cnda_%(nll)s);
            cnda_%(nll)s = (CudaNdarray*)CudaNdarray_NewDims(1, CudaNdarray_HOST_DIMS(cnda_%(y_idx)s));
            if(!cnda_%(nll)s)
            {
                %(fail)s;
            }
        }
        if ((NULL == cnda_%(sm)s)
            || (CudaNdarray_HOST_DIMS(cnda_%(sm)s)[0] != CudaNdarray_HOST_DIMS(cnda_%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(cnda_%(sm)s)[1] != CudaNdarray_HOST_DIMS(cnda_%(x)s)[1]))
        {
            Py_XDECREF(cnda_%(sm)s);
            cnda_%(sm)s = (CudaNdarray*) CudaNdarray_NewDims(2, CudaNdarray_HOST_DIMS(cnda_%(x)s));
            if(!cnda_%(sm)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc sm output");
                // no need to decref cnda_nll, the cleanup code should pick it up.
                %(fail)s;
            }
        }
        if ((NULL == cnda_%(am)s)
            || (CudaNdarray_HOST_DIMS(cnda_%(am)s)[0] != CudaNdarray_HOST_DIMS(cnda_%(y_idx)s)[0]))
        {
            Py_XDECREF(cnda_%(am)s);
            cnda_%(am)s = (CudaNdarray*) CudaNdarray_NewDims(1, CudaNdarray_HOST_DIMS(cnda_%(y_idx)s));
            if(!cnda_%(am)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc am output");
                // no need to decref nll amd sm, the cleanup code should pick it up.
                %(fail)s;
            }
        }
        {
            int n_blocks = CudaNdarray_HOST_DIMS(cnda_%(sm)s)[0];
            int n_threads = 1; //TODO: launch more threads per row and do parallel sum and max reductions.
            int n_shared_bytes = 0; //n_threads * sizeof(float);

            k_xent_sm_1hot_bias<<<n_blocks, n_threads, n_shared_bytes>>>(
                CudaNdarray_HOST_DIMS(cnda_%(x)s)[0],
                CudaNdarray_HOST_DIMS(cnda_%(x)s)[1],
                CudaNdarray_DEV_DATA(cnda_%(x)s), CudaNdarray_HOST_STRIDES(cnda_%(x)s)[0], CudaNdarray_HOST_STRIDES(cnda_%(x)s)[1], 
                CudaNdarray_DEV_DATA(cnda_%(b)s), CudaNdarray_HOST_STRIDES(cnda_%(b)s)[0], 
                CudaNdarray_DEV_DATA(cnda_%(y_idx)s), CudaNdarray_HOST_STRIDES(cnda_%(y_idx)s)[0], 
                CudaNdarray_DEV_DATA(cnda_%(nll)s), CudaNdarray_HOST_STRIDES(cnda_%(nll)s)[0], 
                CudaNdarray_DEV_DATA(cnda_%(sm)s), CudaNdarray_HOST_STRIDES(cnda_%(sm)s)[0], CudaNdarray_HOST_STRIDES(cnda_%(sm)s)[1], 
                CudaNdarray_DEV_DATA(cnda_%(am)s), CudaNdarray_HOST_STRIDES(cnda_%(am)s)[0]);
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if (cudaSuccess != err) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %(classname)s %(nodename)s: %%s.\\n", cudaGetErrorString(err));
                // no need to decref output vars the cleanup code should pick them up.
                %(fail)s;
            }
        }
        """ % locals()
        return sio.getvalue()

    def c_code_cache_version(self):
        return ()
        return (1,0)


class GpuCrossentropySoftmax1HotWithBiasDx (Op):
    nin=3
    nout=1
    """Gradient wrt x of the CrossentropySoftmax1Hot Op"""
    def __init__(self, **kwargs):
        Op.__init__(self,**kwargs)
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return self.__class__.__name__
    def make_node(self, dy, sm, y_idx):
        return Apply(self, [dy, sm, y_idx],[sm.type()])
    def perform(self, node, input_storage, output_storage):
        assert False
        raise NotImplementedError('only C is implemented')

    def c_code_cache_version(self):
        return ()
    def c_code(self, node, nodename, (dnll, sm, y_idx), (dx,), sub):
        fail = sub['fail']
        return """
        if ((cnda_%(dnll)s->nd != 1)
            || (cnda_%(sm)s->nd != 2)
            || (cnda_%(y_idx)s->nd != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(cnda_%(dnll)s)[0] != CudaNdarray_HOST_DIMS(cnda_%(sm)s)[0])
        {
            PyErr_Format(PyExc_ValueError, "dnll.shape[0] == %%i, but sm.shape[0] == %%i",
            CudaNdarray_HOST_DIMS(cnda_%(dnll)s)[0],CudaNdarray_HOST_DIMS(cnda_%(sm)s)[0]);
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(cnda_%(dnll)s)[0] != CudaNdarray_HOST_DIMS(cnda_%(y_idx)s)[0])
        {
            PyErr_SetString(PyExc_ValueError, "dnll.shape[0] != y_idx.shape[0]");
            %(fail)s;
        }
        if ((NULL == cnda_%(dx)s)
            || (CudaNdarray_HOST_DIMS(cnda_%(dx)s)[0] != CudaNdarray_HOST_DIMS(cnda_%(sm)s)[0])
            || (CudaNdarray_HOST_DIMS(cnda_%(dx)s)[1] != CudaNdarray_HOST_DIMS(cnda_%(sm)s)[1]))
        {
            Py_XDECREF(cnda_%(dx)s);
            cnda_%(dx)s = (CudaNdarray*)CudaNdarray_new_null();
            if ((NULL == cnda_%(dx)s)
                || CudaNdarray_alloc_contiguous(cnda_%(dx)s, 2, CudaNdarray_HOST_DIMS(cnda_%(sm)s)))
            {
                Py_XDECREF(cnda_%(dx)s);
                cnda_%(dx)s = NULL;
                %(fail)s;
            }
        }
        {
            kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s
                <<<
                    CudaNdarray_HOST_DIMS(cnda_%(dx)s)[0],
                    CudaNdarray_HOST_DIMS(cnda_%(dx)s)[1]
                >>>(
                        CudaNdarray_HOST_DIMS(cnda_%(dx)s)[0],
                        CudaNdarray_HOST_DIMS(cnda_%(dx)s)[1], 

                        CudaNdarray_DEV_DATA(cnda_%(dnll)s),
                        CudaNdarray_HOST_STRIDES(cnda_%(dnll)s)[0],

                        CudaNdarray_DEV_DATA(cnda_%(sm)s),
                        CudaNdarray_HOST_STRIDES(cnda_%(sm)s)[0],
                        CudaNdarray_HOST_STRIDES(cnda_%(sm)s)[1],

                        CudaNdarray_DEV_DATA(cnda_%(y_idx)s),
                        CudaNdarray_HOST_STRIDES(cnda_%(y_idx)s)[0],

                        CudaNdarray_DEV_DATA(cnda_%(dx)s)       //guaranteed c-contiguous
                );
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n", "kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s", cudaGetErrorString(err));
                %(fail)s;
            }                         
        }
        assert(cnda_%(dx)s);
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        return """
        __global__ void kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s(
           int N, int K,
           const float * dnll, const int dnll_s0,
           const float * sm, const int sm_s0, const int sm_s1,
           const float * y_idx, const int y_idx_s0,
           float * dx)
        {
            for (int i = blockIdx.x; i < N; i += gridDim.x)
            {
                float dnll_i = dnll[i * dnll_s0];
                int y_i = (int)y_idx[i * y_idx_s0];

                for (int j = threadIdx.x; j < K; j += blockDim.x)
                {
                    if (y_i == j)
                    {
                        dx[i * K + j] = dnll_i * (sm[i * sm_s0 + j * sm_s1]-1.0);
                    }
                    else
                    {
                        dx[i * K + j] = dnll_i * sm[i * sm_s0 + j * sm_s1];
                    }
                    //dx[i * K + j] = dnll_i * sm[i * sm_s0 + j * sm_s1];
                    //dx[i*K+j] = 0;
                }
            }
        }
        """ % locals()

