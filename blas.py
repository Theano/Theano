from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar
import StringIO

import cuda_ndarray
from .type import CudaNdarrayType

class GpuDot22(Op):
    def __str__(self):
        return 'GpuDot22'
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, y):
        if x.type.ndim != 2:
            raise TypeError(x)
        if y.type.ndim != 2:
            raise TypeError(y)
        return Apply(self, [x,y], [x.type()])

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, nodename, inputs, outputs, sub):
        x, y = inputs
        z, = outputs
        fail = sub['fail']
        return """
        if (cnda_%(x)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(x)==%%i must be 2", cnda_%(x)s->nd);
            %(fail)s;
        }
        if (cnda_%(y)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(y)==%%i must be 2", cnda_%(y)s->nd);
            %(fail)s;
        }
        if ((NULL == cnda_%(z)s)
            || (CudaNdarray_HOST_DIMS(cnda_%(z)s)[0] != CudaNdarray_HOST_DIMS(cnda_%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(cnda_%(z)s)[1] != CudaNdarray_HOST_DIMS(cnda_%(y)s)[1]))
        {
            if (cnda_%(z)s) Py_DECREF(cnda_%(z)s);
            npy_intp dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(cnda_%(x)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(cnda_%(y)s)[1];
            cnda_%(z)s = (CudaNdarray*)CudaNdarray_new_null();
            if ((NULL == cnda_%(z)s) || CudaNdarray_alloc_contiguous(cnda_%(z)s, 2, dims))
            {
                if (cnda_%(z)s)
                {
                    Py_DECREF(cnda_%(z)s);
                    cnda_%(z)s = NULL;
                }
                %(fail)s;
            }
        }
        if (CudaNdarray_gemm(1.0f, cnda_%(x)s, cnda_%(y)s, 0.0f, cnda_%(z)s))
        {
            if (cnda_%(z)s)
            {
                Py_DECREF(cnda_%(z)s);
                cnda_%(z)s = NULL;
            }
            %(fail)s;
        }
        """ % locals()
gpu_dot22 = GpuDot22()

class GpuGemm(Op):
    destroy_map = {0:[0]}
    def __str__(self):
        return 'GpuGemm'
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, z, a, x, y, b):
        # the more complicated error checking performed by tensor.gemm is assumed to already
        # have been done
        return Apply(self, [z, a, x, y, b], [z.type()])

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, name, inputs, outputs, sub):
        z_in, a, x, y, b = inputs
        z_out, = outputs
        fail = sub['fail']
        return """

        #define REAL float
        float %(name)s_a = (%(a)s->descr->type_num == PyArray_FLOAT) 
        ? (REAL)(((float*)%(a)s->data)[0])
        : (REAL)(((double*)%(a)s->data)[0]);

        float %(name)s_b = (%(b)s->descr->type_num == PyArray_FLOAT) ?
        (REAL)(((float*)%(b)s->data)[0])
        : (REAL)(((double*)%(b)s->data)[0]);
        #undef REAL

        if (CudaNdarray_gemm(%(name)s_a, cnda_%(x)s, cnda_%(y)s, %(name)s_b, cnda_%(z_in)s))
        {
            %(fail)s;
        }
        cnda_%(z_out)s = cnda_%(z_in)s;
        Py_INCREF(cnda_%(z_out)s);
        """ % locals()
gpu_gemm = GpuGemm()

##
# Not really a BLAS operation, but whatever.
#
class GpuConv(Op):
    @staticmethod
    def logical_output_shape_2d(imshp, kshp, mode):
        if mode == 'valid':
            return imshp[0] - kshp[0] + 1, imshp[1] - kshp[1] + 1
        if mode == 'full':
            return imshp[0] + kshp[0] - 1, imshp[1] + kshp[1] - 1
        raise ValueError(mode)

    def __init__(self, border_mode, 
            subsample=(1,1), 
            logical_img_hw=None, 
            logical_kern_hw=None,
            logical_kern_align_top=True):
        self.border_mode = border_mode
        self.subsample = subsample
        if logical_img_hw is not None:
            h,w = logical_img_hw
            #TODO: reconsider this... since shapes are not given in constructor,
            # maybe a multiplier + offset is a more appropriate way of passing this logical
            # grid
        self.logical_img_hw = tuple(logical_img_hw)
        if logical_kern_hw is not None:
            h,w = logical_kern_hw
            #TODO: reconsider this... since shapes are not given in constructor,
            # maybe a multiplier + offset is a more appropriate way of passing this logical
            # grid
        self.logical_kern_hw = tuple(logical_kern_hw)
        self.logical_kern_align_top = logical_kern_align_top

    def __eq__(self, other):
        return type(self) == type(other) \
            and self.border_mode == other.border_mode \
            and self.subsample == other.subsample \
            and self.logical_img_hw == other.logical_img_hw \
            and self.logical_kern_hw == other.logical_kern_hw \
            and self.logical_kern_align_top == other.logical_kern_align_top

    def __hash__(self):
        return hash(type(self)) \
            ^ hash(self.border_mode) \
            ^ hash(self.subsample) \
            ^ hash(self.logical_img_hw) \
            ^ hash(self.logical_kern_hw) \
            ^ hash(self.logical_kern_align_top)

    def __str__(self):
        return '%s{%s, %s, %s, %s, %s}' %(self.__class__.__name__,
                self.border_mode,
                str(self.subsample),
                str(self.logical_img_hw),
                str(self.logical_kern_hw),
                str(self.logical_kern_align_top))

    def make_node(self, img, kern):
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if img.type != kern.type:
            raise TypeError('img and kern must have same type')
        return Apply(self, [img, kern], [img.type()])

    def perform(self, node, (img, kern), (out,)):
        out[0] = cuda_ndarray.conv(img, kern, 
                mode=self.border_mode, 
                subsample=self.subsample,
                logical_img_shape=self.logical_img_hw,
                logical_kern_shape=self.logical_kern_hw,
                kern_align=self.logical_kern_align_top)


class GpuCrossentropySoftmaxArgmax1HotWithBias(Op):
    nin=3
    nout=3
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
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

