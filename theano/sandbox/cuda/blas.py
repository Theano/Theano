import copy
import os

import theano
from theano import Apply
from theano import tensor
from theano.compat.six import StringIO
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import GpuOp


class GpuDot22(GpuOp):
    """
    Implement dot(2d, 2d) on the gpu.
    """
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
        otype = CudaNdarrayType(
                (x.type.broadcastable[0], y.type.broadcastable[1]))
        return Apply(self, [x, y], [otype()])

    def c_code_cache_version(self):
        return (1, 2)

    def c_code(self, node, nodename, inputs, outputs, sub):
        x, y = inputs
        z, = outputs
        fail = sub['fail']
        return """
        if (%(x)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(x)==%%i must be 2", %(x)s->nd);
            %(fail)s;
        }
        if (%(y)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(y)==%%i must be 2", %(y)s->nd);
            %(fail)s;
        }
        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] !=
                  CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] !=
                  CudaNdarray_HOST_DIMS(%(y)s)[1])
            || (CudaNdarray_HOST_STRIDES(%(z)s)[0] < 0)
            || (CudaNdarray_HOST_STRIDES(%(z)s)[1] < 0)
            || ((CudaNdarray_HOST_DIMS(%(z)s)[0] > 1)
                && (CudaNdarray_HOST_STRIDES(%(z)s)[0] != 1)
                && (CudaNdarray_HOST_DIMS(%(z)s)[1] > 1)
                && (CudaNdarray_HOST_STRIDES(%(z)s)[1] != 1)))
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(%(x)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(%(y)s)[1];
            %(z)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(z)s) ||
                CudaNdarray_alloc_contiguous(%(z)s, 2, dims))
            {
                if (%(z)s)
                {
                    Py_DECREF(%(z)s);
                    %(z)s = NULL;
                }
                %(fail)s;
            }
        }
        if (CudaNdarray_gemm(1.0f, %(x)s, %(y)s, 0.0f, %(z)s))
        {
            if (%(z)s)
            {
                Py_DECREF(%(z)s);
                %(z)s = NULL;
            }
            %(fail)s;
        }
        """ % locals()
gpu_dot22 = GpuDot22()


class GpuDot22Scalar(GpuOp):
    """
    Implement dot(2d, 2d) * scalar on the gpu.
    """
    def __str__(self):
        return 'GpuDot22Scalar'

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, y, a):
        if x.type.ndim != 2:
            raise TypeError(x)
        if y.type.ndim != 2:
            raise TypeError(y)
        if not tensor.blas._as_scalar(a):
            raise TypeError(a)
        otype = CudaNdarrayType(
                (x.type.broadcastable[0], y.type.broadcastable[1]))
        return Apply(self, [x, y, a], [otype()])

    def c_code_cache_version(self):
        return (1, 2)

    def c_code(self, node, name, inputs, outputs, sub):
        x, y, a = inputs
        z, = outputs
        fail = sub['fail']
        return """
        #define REAL float
        float %(name)s_a = (PyArray_TYPE(%(a)s) == NPY_FLOAT)
        ? (REAL)(((float*)%(a)s->data)[0])
        : (REAL)(((double*)%(a)s->data)[0]);
        #undef REAL
        if (%(x)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(x)==%%i must be 2", %(x)s->nd);
            %(fail)s;
        }
        if (%(y)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(y)==%%i must be 2", %(y)s->nd);
            %(fail)s;
        }

        if ((NULL == %(z)s) ||
            (CudaNdarray_HOST_DIMS(%(z)s)[0] !=
              CudaNdarray_HOST_DIMS(%(x)s)[0]) ||
            (CudaNdarray_HOST_DIMS(%(z)s)[1] !=
              CudaNdarray_HOST_DIMS(%(y)s)[1])
            || (CudaNdarray_HOST_STRIDES(%(z)s)[0] < 0)
            || (CudaNdarray_HOST_STRIDES(%(z)s)[1] < 0)
            || ((CudaNdarray_HOST_DIMS(%(z)s)[0] > 1)
                && (CudaNdarray_HOST_STRIDES(%(z)s)[0] != 1)
                && (CudaNdarray_HOST_DIMS(%(z)s)[1] > 1)
                && (CudaNdarray_HOST_STRIDES(%(z)s)[1] != 1)))
        {
            //if (%(z)s) Py_DECREF(%(z)s);
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(%(x)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(%(y)s)[1];
            %(z)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(z)s) ||
                CudaNdarray_alloc_contiguous(%(z)s, 2, dims))
            {
                if (%(z)s)
                {
                    Py_DECREF(%(z)s);
                    %(z)s = NULL;
                }
                %(fail)s;
            }
        }
        if (CudaNdarray_gemm(%(name)s_a, %(x)s, %(y)s, 0.0f, %(z)s))
        {
            if (%(z)s)
            {
                Py_DECREF(%(z)s);
                %(z)s = NULL;
            }
            %(fail)s;
        }
        """ % locals()
gpu_dot22scalar = GpuDot22Scalar()


class GpuGemm(GpuOp):
    """
    implement the gemm on the gpu.

    """
    def __init__(self, inplace):
        self.__setstate__({'inplace': inplace})

    def __str__(self):
        if self.inplace:
            return 'GpuGemm{inplace}'
        else:
            return 'GpuGemm{no_inplace}'

    def __eq__(self, other):
        return (type(self) == type(other)\
                and self.inplace == other.inplace)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def __setstate__(self, dct):
        inplace = dct.get('inplace', True)
        if inplace:
            self.destroy_map = {0: [0]}
        self.inplace = inplace

    def __getstate__(self):
        return dict(inplace=self.inplace)

    def make_node(self, z, a, x, y, b):
        # the more complicated error checking performed by tensor.gemm
        # is assumed to already have been done
        return Apply(self, [z, a, x, y, b], [z.type()])

    def c_code_cache_version(self):
        return (4,)

    def c_code(self, node, name, inputs, outputs, sub):
        #z_out = alpha * dot(x,y) + beta * z_in
        #inplace version, set set z_out = z_in
        #not inplace version, we copy z_in to z_out.
        z_in, a, x, y, b = inputs
        z_out, = outputs
        inplace = int(self.inplace)
        fail = sub['fail']
        sio = StringIO()

        print >> sio, """

        #define REAL float
        float %(name)s_a = (PyArray_TYPE(%(a)s) == NPY_FLOAT)
        ? (REAL)(((float*)%(a)s->data)[0])
        : (REAL)(((double*)%(a)s->data)[0]);

        float %(name)s_b = (PyArray_TYPE(%(b)s) == NPY_FLOAT) ?
        (REAL)(((float*)%(b)s->data)[0])
        : (REAL)(((double*)%(b)s->data)[0]);
        #undef REAL

        if (%(inplace)s
            && (CudaNdarray_HOST_STRIDES(%(z_in)s)[0] >= 0)
            && (CudaNdarray_HOST_STRIDES(%(z_in)s)[1] >= 0)
            && ((CudaNdarray_HOST_DIMS(%(z_in)s)[0] <= 1)
                || (CudaNdarray_HOST_STRIDES(%(z_in)s)[0] == 1)
                || (CudaNdarray_HOST_DIMS(%(z_in)s)[1] <= 1)
                || (CudaNdarray_HOST_STRIDES(%(z_in)s)[1] == 1)))
        {
            // The input has an appropriate layout, we work inplace
            Py_XDECREF(%(z_out)s);
            %(z_out)s = %(z_in)s;
            Py_INCREF(%(z_out)s);
        }
        else if (%(z_out)s
                && (%(z_out)s->nd == 2)
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[0]
                    == CudaNdarray_HOST_DIMS(%(z_in)s)[0])
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[1]
                    == CudaNdarray_HOST_DIMS(%(z_in)s)[1])
                && (CudaNdarray_HOST_STRIDES(%(z_out)s)[0] >= 0)
                && (CudaNdarray_HOST_STRIDES(%(z_out)s)[1] >= 0)
                && ((CudaNdarray_HOST_DIMS(%(z_out)s)[0] <= 1)
                    || (CudaNdarray_HOST_STRIDES(%(z_out)s)[0] == 1)
                    || (CudaNdarray_HOST_DIMS(%(z_out)s)[1] <= 1)
                    || (CudaNdarray_HOST_STRIDES(%(z_out)s)[1] == 1)))
        {
            // The existing output has an appropriate layout,
            // copy the input data into it, then work inplace
            if (CudaNdarray_CopyFromCudaNdarray(%(z_out)s, %(z_in)s))
            {
                %(fail)s;
            }
        }
        else
        {
            // Copy the input, use the copy as output
            Py_XDECREF(%(z_out)s);
            %(z_out)s = (CudaNdarray*)CudaNdarray_Copy(%(z_in)s);
            if (!%(z_out)s)
            {
                %(fail)s;
            }
        }

        if (CudaNdarray_gemm(%(name)s_a, %(x)s, %(y)s, %(name)s_b, %(z_out)s))
        {
            %(fail)s;
        }
        """

        return sio.getvalue() % locals()
gpu_gemm_no_inplace = GpuGemm(inplace=False)
gpu_gemm_inplace = GpuGemm(inplace=True)


class GpuGemv(GpuOp):
    """
    implement gemv on the gpu.

    """
    def __init__(self, inplace):
        self.__setstate__({'inplace': inplace})

    def __str__(self):
        if self.inplace:
            return 'GpuGemv{inplace}'
        else:
            return 'GpuGemv{no_inplace}'

    def __eq__(self, other):
        return (type(self) == type(other)\
                and self.inplace == other.inplace)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def __setstate__(self, dct):
        inplace = dct.get('inplace', True)
        if inplace:
            self.destroy_map = {0: [0]}
        self.inplace = inplace

    def __getstate__(self):
        return dict(inplace=self.inplace)

    def make_node(self, z, a, x, y, b):
        # the more complicated error checking performed by tensor.gemv
        # is assumed to already have been done
        return Apply(self, [z, a, x, y, b], [z.type()])

    def c_code_cache_version(self):
        return (3,)

    def c_code(self, node, name, inputs, outputs, sub):
        #z_out = alpha * dot(x,y) + beta * z_in
        #inplace version, set set z_out = z_in
        #not inplace version, we copy z_in to z_out.
        z_in, a, x, y, b = inputs
        z_out, = outputs
        inplace = int(self.inplace)
        fail = sub['fail']
        sio = StringIO()

        print >> sio, """
        float %(name)s_alpha = ((dtype_%(a)s*)(%(a)s->data))[0];
        float %(name)s_beta = ((dtype_%(b)s*)(%(b)s->data))[0];

        if (%(inplace)s
            && ((CudaNdarray_HOST_STRIDES(%(z_in)s)[0] > 0)
                || ((CudaNdarray_HOST_STRIDES(%(z_in)s)[0] == 0)
                    && (CudaNdarray_HOST_DIMS(%(z_in)s)[0] == 1))))
        {
            // Work inplace on the input
            Py_XDECREF(%(z_out)s);
            %(z_out)s = %(z_in)s;
            Py_INCREF(%(z_out)s);
        }
        else if (%(z_out)s
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[0] ==
                    CudaNdarray_HOST_DIMS(%(z_in)s)[0])
                && ((CudaNdarray_HOST_STRIDES(%(z_out)s)[0] > 0)
                    || ((CudaNdarray_HOST_STRIDES(%(z_out)s)[0] == 0)
                        && (CudaNdarray_HOST_DIMS(%(z_out)s)[0] == 1))))
        {
            // Work on the output
            if (CudaNdarray_CopyFromCudaNdarray(%(z_out)s, %(z_in)s))
            {
                %(fail)s;
            }
        }
        else
        {
            // Copy
            Py_XDECREF(%(z_out)s);
            %(z_out)s = (CudaNdarray*)CudaNdarray_Copy(%(z_in)s);
            if (!%(z_out)s)
            {
                %(fail)s;
            }
        }

        if (CudaNdarray_sgemv(%(name)s_alpha, %(x)s, %(y)s,
                              %(name)s_beta, %(z_out)s))
        {
            %(fail)s;
        }
        """
        return sio.getvalue() % locals()
gpu_gemv_no_inplace = GpuGemv(inplace=False)
gpu_gemv_inplace = GpuGemv(inplace=True)


class GpuGer(GpuOp):
    """
    implement ger on the gpu.

    """
    def __init__(self, inplace):
        self.__setstate__({'inplace': inplace})

    def __str__(self):
        if self.inplace:
            return 'GpuGer{inplace}'
        else:
            return 'GpuGer{no_inplace}'

    def __eq__(self, other):
        return (type(self) == type(other)\
                and self.inplace == other.inplace)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def __setstate__(self, dct):
        inplace = dct.get('inplace', True)
        if inplace:
            self.destroy_map = {0: [0]}
        self.inplace = inplace

    def __getstate__(self):
        return dict(inplace=self.inplace)

    def make_node(self, z, a, x, y):
        # the more complicated error checking performed by tensor.ger is
        # assumed to already have been done
        return Apply(self, [z, a, x, y], [z.type()])

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inputs, outputs, sub):
        #z_out = alpha * dot(x,y) + beta * z_in
        #inplace version, set set z_out = z_in
        #not inplace version, we copy z_in to z_out.
        z_in, a, x, y = inputs
        z_out, = outputs
        inplace = int(self.inplace)
        fail = sub['fail']
        sio = StringIO()

        print >> sio, """
        float %(name)s_alpha = ((dtype_%(a)s*)(%(a)s->data))[0];

        if (%(inplace)s
            && (CudaNdarray_HOST_STRIDES(%(z_in)s)[0] >= 0)
            && (CudaNdarray_HOST_STRIDES(%(z_in)s)[1] >= 0)
            && ((CudaNdarray_HOST_DIMS(%(z_in)s)[0] <= 1)
                || (CudaNdarray_HOST_STRIDES(%(z_in)s)[0] == 1)
                || (CudaNdarray_HOST_DIMS(%(z_in)s)[1] <= 1)
                || (CudaNdarray_HOST_STRIDES(%(z_in)s)[1] == 1)))
        {
            // The input has an appropriate layout, we work inplace
            Py_XDECREF(%(z_out)s);
            %(z_out)s = %(z_in)s;
            Py_INCREF(%(z_out)s);
        }
        else if (%(z_out)s
                && (%(z_out)s->nd == 2)
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[0]
                    == CudaNdarray_HOST_DIMS(%(z_in)s)[0])
                && (CudaNdarray_HOST_DIMS(%(z_out)s)[1]
                    == CudaNdarray_HOST_DIMS(%(z_in)s)[1])
                && (CudaNdarray_HOST_STRIDES(%(z_out)s)[0] >= 0)
                && (CudaNdarray_HOST_STRIDES(%(z_out)s)[1] >= 0)
                && ((CudaNdarray_HOST_DIMS(%(z_out)s)[0] <= 1)
                    || (CudaNdarray_HOST_STRIDES(%(z_out)s)[0] == 1)
                    || (CudaNdarray_HOST_DIMS(%(z_out)s)[1] <= 1)
                    || (CudaNdarray_HOST_STRIDES(%(z_out)s)[1] == 1)))
        {
            // The existing output has an appropriate layout,
            // copy the input data into it, then work inplace
            if (CudaNdarray_CopyFromCudaNdarray(%(z_out)s, %(z_in)s))
            {
                %(fail)s;
            }
        }
        else
        {
            // Copy the input, use the copy as output
            Py_XDECREF(%(z_out)s);
            %(z_out)s = (CudaNdarray*)CudaNdarray_Copy(%(z_in)s);
            if (!%(z_out)s)
            {
                %(fail)s;
            }
        }

        if (CudaNdarray_sger(%(name)s_alpha, %(x)s, %(y)s, %(z_out)s))
        {
            %(fail)s;
        }
        """
        return sio.getvalue() % locals()
gpu_ger_no_inplace = GpuGer(inplace=False)
gpu_ger_inplace = GpuGer(inplace=True)


##
# Not really a BLAS operation, but whatever.
#
class GpuConv(GpuOp):
    """
    Implement the batched and stacked 2d convolution on the gpu.
    """
    @staticmethod
    def logical_output_shape_2d(imshp, kshp, mode):
        if mode == 'valid':
            return imshp[0] - kshp[0] + 1, imshp[1] - kshp[1] + 1
        if mode == 'full':
            return imshp[0] + kshp[0] - 1, imshp[1] + kshp[1] - 1
        raise ValueError(mode)

    def __init__(self, border_mode,
            subsample=(1, 1),
            logical_img_hw=None,
            logical_kern_hw=None,
            logical_kern_align_top=True,
            version=-1,
            verbose=0,
            kshp=None,
            imshp=None,
            max_threads_dim0=None):
        """
        :param version: each version of c_code implement many kernel for the
                        convolution. By default we try to guess the best one.
                        You can force one version with this parameter. This
                        parameter is used by the tests.
        :param verbose: for value of 1,2 and 3. Print more information during
                        the execution of the convolution. Mostly used for
                        optimization or debugging.
        :param kshp:    The size of the kernel. If provided, can genera
                        faster code. If the GpuConv op is automatically
                        inserted,
                        we take its value automatically from the Conv op.
        :param imshp:   The size of the image. Not used for code generation but
                        allow to select an experimental new version in another
                        repo.
        :param max_threads_dim0: The maximum number of thread for the
                        block size dimensions 0 (blockDim.x) used by the
                        GPU function.

        """
        self.border_mode = border_mode
        self.subsample = subsample
        if logical_img_hw is not None:
            h, w = logical_img_hw
            #TODO: reconsider this... since shapes are not given in
            # constructor, maybe a multiplier + offset is a more
            # appropriate way of passing this logical grid
            logical_img_hw = tuple(logical_img_hw)
        self.logical_img_hw = logical_img_hw
        if logical_kern_hw is not None:
            h, w = logical_kern_hw
            #TODO: reconsider this... since shapes are not given in
            # constructor, maybe a multiplier + offset is a more
            # appropriate way of passing this logical grid
            logical_kern_hw = tuple(logical_kern_hw)
        self.logical_kern_hw = logical_kern_hw
        self.logical_kern_align_top = logical_kern_align_top
        self.version = version
        self.verbose = verbose
        self.kshp = kshp
        self.imshp = imshp
        self.max_threads_dim0 = max_threads_dim0

    def __eq__(self, other):
        return type(self) == type(other) \
            and self.border_mode == other.border_mode \
            and self.subsample == other.subsample \
            and self.logical_img_hw == other.logical_img_hw \
            and self.logical_kern_hw == other.logical_kern_hw \
            and self.logical_kern_align_top == other.logical_kern_align_top \
            and self.version == other.version \
            and self.verbose == other.verbose \
            and self.kshp == other.kshp\
            and self.imshp == other.imshp\
            and self.max_threads_dim0 == other.max_threads_dim0

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "imshp"):
            self.imshp = None
        if not hasattr(self, "max_threads_dim0"):
            self.max_threads_dim0 = None

    def __hash__(self):
        # don't use hash(self.version) as hash(-1)==-2 and
        # hash(-2)==-2 in python!
        return hash(type(self)) \
            ^ hash(self.border_mode) \
            ^ hash(self.subsample) \
            ^ hash(self.logical_img_hw) \
            ^ hash(self.logical_kern_hw) \
            ^ hash(self.logical_kern_align_top) \
            ^ self.version \
            ^ hash(self.verbose) \
            ^ hash(self.kshp)\
            ^ hash(self.imshp)\
            ^ hash(self.max_threads_dim0)

    def __str__(self):
        return '%s{%s, %s, %s, %s, %s, %s, %s}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample),
            str(self.logical_img_hw),
            str(self.logical_kern_hw),
            str(self.logical_kern_align_top),
            str(self.imshp),
            str(self.kshp))

    def make_node(self, img, kern):
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        return Apply(self, [img, kern], [CudaNdarrayType(broadcastable)()])

    def flops(self, inputs, outputs):
        """ Useful with the hack in profilemode to print the MFlops"""
        images, kerns = inputs
        out, = outputs
        assert images[1] == kerns[1]
        flops = 0
        if self.out_mode == "valid":
            # nb mul and add by output pixel
            flops = kerns[2] * kerns[3] * 2
            # nb flops by output image
            flops *= out[2] * out[3]
            # nb patch multiplied
            flops *= images[1] * kerns[0] * images[0]
        else:
            flops = (images[0] * kerns[0] * images[1] *
                     kerns[2] * kerns[3] *
                     images[2] * images[3] * 2)
        return flops

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_ = copy.copy(node)
        assert node.op is node_.op
        if node_.op.max_threads_dim0 is None:
            cuda = theano.sandbox.cuda
            device_id = cuda.use.device_number
            if device_id is None:
                cuda.use("gpu",
                         force=False,
                         default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False,
                         test_driver=True)
                device_id = cuda.use.device_number
            cuda_ndarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
            prop = cuda_ndarray.device_properties(device_id)
            node_.op.max_threads_dim0 = prop['maxThreadsDim0']
        return super(GpuConv, node_.op).make_thunk(node_, storage_map,
                                                   compute_map, no_recycling)

    def c_compile_args(self):
        nb = 0
        if self.kshp is not None:
            nb = self.kshp[1]
        return ['-DTHEANO_KERN_WID=' + str(nb)]  # ,'-g','-G']

    def c_headers(self):
        return ['cuda_ndarray.cuh', '<stdio.h>']

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 20)

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['conv_kernel.cu', 'conv_full_kernel.cu', 'conv.cu']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                for f in files]
        return reduce(str.__add__, codes)

    def c_code(self, node, nodename, inp, out_, sub):
        img, kern = inp
        out, = out_
        dx = self.subsample[0]
        dy = self.subsample[1]
        border_mode = self.border_mode
        version = self.version
        verbose = self.verbose
        sub = sub.copy()
        max_threads_dim0 = self.max_threads_dim0
        if max_threads_dim0 is None:
            raise NotImplementedError("GpuConv.c_code should not be called "
                                      "directly. It should be called by "
                                      "make_thunk() that add some information "
                                      "related to the selected GPU.")
        sub.update(locals())
        return """
    //Mandatory args
    const char *mode_str = "%(border_mode)s";

    //Optional args
    int version = %(version)s;
    int verbose = %(verbose)s;
    int dx = %(dx)s;
    int dy = %(dy)s;

    int mode;
    if (strcmp(mode_str, "full") == 0)
    {
        mode = ConvMode_FULL;
    }
    else if (strcmp(mode_str, "valid") == 0)
    {
        mode = ConvMode_VALID;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError,
                        "mode must be one of 'full' or 'valid'");
        return NULL;
    }

    // TODO, make out be decref before we alloc out2!
    CudaNdarray * out2 = (CudaNdarray *)CudaNdarray_Conv(%(img)s, %(kern)s,
                                                         %(out)s, mode,
                                                         dx, dy,
                                                         version, verbose,
                                                         %(max_threads_dim0)s);
    Py_XDECREF(%(out)s);
    %(out)s = out2;

    if (%(out)s==NULL){
        %(fail)s
    }
""" % sub


class GpuDownsampleFactorMax(GpuOp):
    """
    Implement downsample with max on the gpu.
    """
    def __init__(self, ds, ignore_border=False):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.ds == other.ds and
                self.ignore_border == other.ignore_border)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__,
                              self.ds,
                              self.ignore_border)

    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError()
        if not x.type.ndim == 4:
            raise TypeError()
        return Apply(self, [x], [x.type()])

    #def perform(self, node, input_storage, output_storage):
        #raise NotImplementedError('only C is implemented')
    def c_code_cache_version(self):
        return (6)

    def c_code(self, node, nodename, inp, out, sub):
        x, = inp
        z, = out
        fail = sub['fail']
        ds0, ds1 = self.ds
        ignore_border = int(self.ignore_border)
        return """
        int dims[4], xdim2, xdim3;
        if (%(x)s->nd != 4)
        {
            PyErr_SetString(PyExc_ValueError,
                            "GpuDownsampleFactorMax: rank error");
            %(fail)s;
        }
        xdim2 = CudaNdarray_HOST_DIMS(%(x)s)[2];
        xdim3 = CudaNdarray_HOST_DIMS(%(x)s)[3];
        dims[0] = CudaNdarray_HOST_DIMS(%(x)s)[0];
        dims[1] = CudaNdarray_HOST_DIMS(%(x)s)[1];
        dims[2] = xdim2 / %(ds0)s;
        dims[3] = xdim3 / %(ds1)s;
        if (! %(ignore_border)s)
        {
            dims[2] += (xdim2%%(%(ds0)s)?1:0);
            dims[3] += (xdim3%%(%(ds1)s)?1:0);
        }
        if(dims[3]>512){
            PyErr_Format(PyExc_ValueError,
                         "GpuDownsampleFactorMax: last dimention size of %%d"
                         " is bigger then 512. This case is not implemented.",
                         dims[3]);
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] != dims[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] != dims[1])
            || (CudaNdarray_HOST_DIMS(%(z)s)[2] != dims[2])
            || (CudaNdarray_HOST_DIMS(%(z)s)[3] != dims[3]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(z)s)
                || CudaNdarray_alloc_contiguous(%(z)s, 4, dims))
            {
                Py_XDECREF(%(z)s);
                %(z)s = NULL;
                PyErr_SetString(PyExc_ValueError,
                                "GpuDownsampleFactorMax:"
                                "Was not able to allocate output!");
                %(fail)s;
            }
        }
        {
            dim3 grid(std::min(dims[0] * dims[1], 65535),
                      dims[2]);
            //dim3 block(std::min(dims[3], 512));
            //TODO: implement this by supporting more outputs than threads
            dim3 block(dims[3]);
            if ((grid.x*grid.y) && dims[3])
            kMaxPool_%(nodename)s<%(ds0)s, %(ds1)s> <<<grid, block,
                                                       xdim3*sizeof(float)>>>(
                dims[0], dims[1], dims[2], dims[3], xdim2, xdim3,
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3]);
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %%s: %%s. (grid: %%i x %%i;"
                    " block: %%i x %%i x %%i)\\n",
                    "kMaxPool_%(nodename)s",
                    cudaGetErrorString(err),
                    grid.x,
                    grid.y,
                    block.x,
                    block.y,
                    block.z);
                %(fail)s;
            }
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        ignore_border = int(self.ignore_border)
        return """
        template<int pf2, int pf3>
        __global__ void kMaxPool_%(nodename)s(
           int D0, int D1, int D2, int D3, int xD2, int xD3,
           const float * x, int xS0, int xS1, int xS2, int xS3,
           float *z, int zS0, int zS1, int zS2, int zS3)
        {
            float cur_max, cur_x;
            // Cast threadIdx.x into a signed int, to avoid problems with
            // indexing with negative offsets.
            int tx = threadIdx.x;
            for(int block_x_idx = blockIdx.x;
                block_x_idx < D0 * D1;
                block_x_idx += gridDim.x){

                int i0 = block_x_idx %% D0;
                int i1 = block_x_idx / D0;
                int i2 = blockIdx.y;

                extern __shared__ float xbuf[]; //size [xD3]

                for (int r2 = 0;
                     (r2 < pf2) && (%(ignore_border)s || (r2 + i2*pf2 < xD2));
                     ++r2)
                {
                    __syncthreads();
                    // load the current row of the image into shared memory
                    for (int j = tx; j < xD3; j += blockDim.x)
                    {
                        xbuf[j] = x[i0*xS0 + i1*xS1 + (i2*pf2+r2)*xS2 + j*xS3];
                    }
                    __syncthreads();

                    // initialize our max if this is the
                    // first row we're loading
                    cur_max = (r2 == 0) ? xbuf[tx*pf3] : cur_max;

                    // do a mini-reduction over the pf3 relevant elements
                    // in the current row

                    if (%(ignore_border)s)
                    {
                        for (int k = 0; k < pf3; ++k)
                        {
                            cur_x = xbuf[tx*pf3+k];
                            cur_max = (cur_x > cur_max) ? cur_x : cur_max;
                        }
                    }
                    else
                    {
                        for (int k = 0; k < pf3; ++k)
                        {
                            if (tx*pf3 + k < xD3)
                            {
                                cur_x = xbuf[tx*pf3+k];
                                cur_max = (cur_x > cur_max) ? cur_x : cur_max;
                            }
                        }
                    }
                }

                z[i0*zS0 + i1*zS1 + i2*zS2 + tx*zS3] = cur_max;
            }
        }
        """ % locals()


class GpuDownsampleFactorMaxGrad(GpuOp):
    """
    Implement the grad of downsample with max on the gpu.
    """
    def __init__(self, ds, ignore_border):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.ds == other.ds and
                self.ignore_border == other.ignore_border)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__,
                              self.ds,
                              self.ignore_border)

    def make_node(self, x, z, gz):
        return Apply(self, [x, z, gz], [x.type()])

    def c_code_cache_version(self):
        return (9,)

    def c_code(self, node, nodename, inp, out, sub):
        x, z, gz = inp
        gx, = out
        fail = sub['fail']
        ds0, ds1 = self.ds
        ignore_border = int(self.ignore_border)
        return """
        if (%(x)s->nd != 4
            || %(z)s->nd != 4
            || %(gz)s->nd != 4)
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if ((NULL == %(gx)s)
            || (CudaNdarray_HOST_DIMS(%(gx)s)[0] !=
                CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[1] !=
                CudaNdarray_HOST_DIMS(%(x)s)[1])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[2] !=
                CudaNdarray_HOST_DIMS(%(x)s)[2])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[3] !=
                CudaNdarray_HOST_DIMS(%(x)s)[3]))
        {
            Py_XDECREF(%(gx)s);
            %(gx)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(gx)s)
                || CudaNdarray_alloc_contiguous(%(gx)s, 4,
                                                CudaNdarray_HOST_DIMS(%(x)s)))
            {
                Py_XDECREF(%(gx)s);
                %(gx)s = NULL;
                %(fail)s;
            }
        }
        {
            //TODO: supporting more output columns than threads
            // make sure we cover every x row when ignore border isset and
            // there's a border present to be ignored
            int needs_extra_z_col = %(ignore_border)s && (CudaNdarray_HOST_DIMS(%(x)s)[2] %% %(ds0)s);
            dim3 grid(std::min(CudaNdarray_HOST_DIMS(%(z)s)[0], 65535),
                      CudaNdarray_HOST_DIMS(%(z)s)[2] + (needs_extra_z_col ? 1 : 0));
            dim3 block(std::min(CudaNdarray_HOST_DIMS(%(x)s)[3], 512));

            kDownsampleMaxGrad_%(nodename)s<%(ds0)s, %(ds1)s> <<<grid, block>>>(
                CudaNdarray_HOST_DIMS(%(z)s)[0],
                CudaNdarray_HOST_DIMS(%(z)s)[1],
                CudaNdarray_HOST_DIMS(%(z)s)[2],
                CudaNdarray_HOST_DIMS(%(z)s)[3],
                CudaNdarray_HOST_DIMS(%(x)s)[2],
                CudaNdarray_HOST_DIMS(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3],
                CudaNdarray_DEV_DATA(%(gz)s),
                CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                CudaNdarray_DEV_DATA(%(gx)s),
                CudaNdarray_HOST_STRIDES(%(gx)s)[0],
                CudaNdarray_HOST_STRIDES(%(gx)s)[1],
                CudaNdarray_HOST_STRIDES(%(gx)s)[2],
                CudaNdarray_HOST_STRIDES(%(gx)s)[3]);
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
    "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kDownsampleMaxGrad_%(nodename)s",
                    cudaGetErrorString(err),
                    grid.x,
                    grid.y,
                    block.x,
                    block.y,
                    block.z);
                %(fail)s;
            }
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        # This code considers every position in the output z, andthen
        # computes the gradient for the input pixels that were
        # downsampled to that z-position. It does so by running along
        # every z row (sometimes plus one, to make sure every gx row
        # gets totally filled), and by running along every x col. This
        # code is not sensitive to the ignore_border flag along the
        # row dimension (since it runs for every position in the
        # output z), but it is sensitive along the col dimension.
        ignore_border = int(self.ignore_border)

        return """
        // ds0 is the downsampling factor in rows, ds1 in columns
        template<int ds0, int ds1>
        __global__ void kDownsampleMaxGrad_%(nodename)s(
           int D0, int D1, int D2, int D3, int xD2, int xD3,
           const float * x, int xS0, int xS1, int xS2, int xS3,
           const float * z, int zS0, int zS1, int zS2, int zS3,
           const float * gz, int gzS0, int gzS1, int gzS2, int gzS3,
           float *gx, int gxS0, int gxS1, int gxS2, int gxS3)
        {
            //  D0: number of image rows
            //  D1: number of image cols
            //  D2: number of z rows
            //  D3: number of z cols
            // xD2: number of x rows
            // xD3: number of x cols
            // various .S. variables are strides

            float cur_max, cur_x, my_z, my_gz;
            // Cast threadIdx.x into a signed int, to avoid problems with
            // indexing with negative offsets.
            int tx = threadIdx.x;
            int bdimx = blockDim.x;

            for(int i0 = blockIdx.x;
                i0 < D0;
                i0 += gridDim.x){

                int i1 = 0;                // image col
                // row wrt z and/or gz, ranges from 0 to D2 - 1 OR D2
                // (as needed to cover all x rows)
                int i2 = blockIdx.y;
                int x_col = tx;            // col wrt x, ranges from 0 to xD3 - 1
                int z_col = x_col/ds1;     // z_col corresponding to this x_col


                //TODO: raise occupancy.  Use threadIdx.y to run several
                //      iterations of this i1 loop in parallel

                for (i1 = 0; i1 < D1; ++i1) // loop over images (same for z and x)
                {
                    for(int col_iter = 0;
                        (tx + col_iter * bdimx < xD3) ; col_iter++){

                        //The if inside is to don't do the division if we
                        // need only 1 col_iter

                        if(tx + bdimx < xD3)
                        {
                            x_col = tx + col_iter * bdimx;
                            z_col = x_col/ds1;
                        }

                        if (%(ignore_border)s && ((x_col >= ds1 * D3) || (i2 >= D2)))
                        {
                            // This happens only if x_col, or i2*ds0, was ignored
                            // (via ignore_border)
                            // TODO: if ignore_border is False, this is impossible
                            //        and we don't even need to generate this code.

                            my_gz = 0.0f;

                            //any fp number suffices for my_z, so we don't even
                            //need to set it to anything in particular.

                        }
                        else
                        {
                            // this is effectively:
                            // my_gz = gz[image_row][image_col][z_row][z_col]
                            // my_z  = z[image_row][image_col][z_row][z_col]
                            my_gz = gz[i0 * gzS0 + i1 * gzS1 + i2 * gzS2 +
                                       z_col*gzS3];
                            my_z =   z[i0 *  zS0 + i1 *  zS1 + i2 *  zS2 +
                                       z_col* zS3];
                        }
                        for (int x_row = i2*ds0;
                              (x_row < i2*ds0+ds0) && (x_row < xD2); ++x_row)
                        {
                            // this is effectively:
                            // gx[image_row][image_col][x_row][x_col]
                            //   = (my_z == x[image_row][image_col][
                            //                x_row][x_col]) ? my_gz : 0.0f;
                            gx[i0*gxS0 + i1*gxS1 + x_row*gxS2 + x_col*gxS3]
                               = (my_z == x[i0*xS0 + i1*xS1 + x_row*xS2 +
                                            x_col*xS3]) ? my_gz : 0.0f;
                        }

                    }
                }
            }
        }
        """ % locals()
