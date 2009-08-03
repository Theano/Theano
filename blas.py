from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar
import StringIO

import cuda_ndarray

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

