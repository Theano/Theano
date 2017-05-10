from __future__ import absolute_import, print_function, division

import numpy
import warnings

from six import integer_types

import theano
from theano.tensor.blas import ldflags
from theano import tensor, Apply, Variable
from theano.gradient import DisconnectedType
from theano.sandbox.mkl.basic_ops import MKLOp
from theano.sandbox.mkl.mkl_helper import header_text


class PoolBase(MKLOp):
    def __init__(self, ignore_border=False, mode='max', ndim=2):
        self.mkl_ver = theano.sandbox.mkl.mkl_version()

        mkl_pool_modes = ['min', 'max', 'average_exc_pad']
        mkl_ignore_border = [False]
        if isinstance(self.mkl_ver, integer_types) and (self.mkl_ver >= 20170206):
            mkl_pool_modes.append('average_inc_pad')
            mkl_ignore_border.append(True)

        if mode not in mkl_pool_modes:
            if 'average_inc_pad' == mode:
                raise ValueError("'average_inc_pad' is supported by MKL newer than 20170206, "
                                 "Current MKL version: %s" % self.mkl_ver)
            else:
                raise ValueError(
                    "Pool mode parameter only support \'%s\' by MKL. Got %s" %
                    (', '.join(mkl_pool_modes), mode))

        if ignore_border not in mkl_ignore_border:
            if ignore_border:
                raise ValueError("'ignore_border=True' is supported by MKL newer than 20170206, "
                                 "Current MKL version: %s" % self.mkl_ver)
            else:
                raise ValueError(
                    "Pool ignore_border only support \'%s\' by MKL. Got %s" %
                    (', '.join(map(str, mkl_ignore_border)), ignore_border))

        self.ndim = ndim
        self.ignore_border = ignore_border
        self.mode = mode

    @staticmethod
    def out_shape(imgshape, ws=None, ignore_border=False, stride=None, pad=None,
                  ndim=2, ds=None, st=None, padding=None):
        """
        Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple, list, or similar of integer or scalar Theano variable
            The shape of a tensor of images. The last N elements are
            interpreted as the number of rows, and the number of cols.
        ws : list or tuple of N ints
            Downsample factor over rows and column.
            ws indicates the pool region size.
        ignore_border : bool
            If ws doesn't divide imgshape, do we include an extra row/col/slice
            of partial downsampling (False) or ignore it (True).
        stride : list or tuple of N ints or None
            Stride size, which is the number of shifts over rows/cols/slices to get the
            next pool region. If stride is None, it is considered equal to ws
            (no overlap on pooling regions).
        pad : tuple of N ints or None
            For each downsampling dimension, this specifies the number of zeros to
            add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
            size of the top and bottom margins, pad_w specifies the size of the left and
            right margins. No padding is added if pad is None.
        ndim : int
            The number of pooling dimensions N.
            The default is 2.
        ds
            *deprecated*, use parameter ws instead.
        st
            *deprecated*, use parameter st instead.
        padding
            *deprecated*, use parameter pad instead.

        Returns
        -------
        list
            The shape of the output from this op, for input of given shape.
            This will have the same length as imgshape, but with last N
            elements reduced as per the downsampling & ignore_border flags.

        """
        # check for deprecated parameter names
        if ds is not None:
            if ws is not None:
                raise ValueError(
                    "You can't provide a tuple value to both 'ws' and 'ds'."
                    " Please provide a value only to 'ws'."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'ds' parameter is not going to exist"
                    " anymore as it is going to be replaced by the parameter"
                    " 'ws'.",
                    stacklevel=2
                )
                ws = ds
        elif ds is None and ws is None:
            raise ValueError(
                "You must provide a tuple value for the window size."
            )

        if st is not None:
            if stride is not None:
                raise ValueError(
                    "You can't provide a tuple value to both 'st and 'stride'."
                    " Please provide a value only to 'stride'."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'st' parameter is not going to exist"
                    " anymore as it is going to be replaced by the parameter"
                    " 'stride'.",
                    stacklevel=2
                )
                stride = st

        if padding is not None:
            zero_pad = (0,) * ndim
            if pad not in {None, zero_pad}:
                raise ValueError(
                    "You can't provide a tuple value to both 'padding' and pad."
                    "  Please provide a value only to pad."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'padding' parameter is not going to"
                    " exist anymore as it is going to be replaced by the"
                    " parameter 'pad'.",
                    stacklevel=2
                )
                pad = padding

        if ndim is None:
            ndim = 2
        assert ndim > 0
        if len(imgshape) < ndim:
            raise TypeError('imgshape must have at least {} dimensions'.format(ndim))

        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * ndim
        patch_shape = tuple(tensor.extract_constant(imgshape[-ndim + i]) + pad[i] * 2
                            for i in xrange(ndim))

        def compute_out(v, downsample, stride):
            if ignore_border:
                if downsample == stride:
                    return v // stride
                else:
                    out = (v - downsample) // stride + 1
                    if isinstance(out, theano.Variable):
                        return tensor.maximum(out, 0)
                    else:
                        return numpy.maximum(out, 0)
            else:
                if isinstance(v, theano.Variable):
                    return tensor.switch(tensor.ge(stride, downsample),
                                         (v - 1) // stride + 1,
                                         tensor.maximum(0, (v - 1 - downsample) //
                                                        stride + 1) + 1)
                elif stride >= downsample:
                    return (v - 1) // stride + 1
                else:
                    return max(0, (v - 1 - downsample + stride) // stride) + 1

        out_shape = [compute_out(patch_shape[i], ws[i], stride[i]) for i in xrange(ndim)]

        rval = list(imgshape[:-ndim]) + out_shape
        return rval

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(PoolBase, self).c_compile_args()
        return compile_args

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_headers(self):
        headers = ['<stdio.h>', '<fstream>']
        headers += super(PoolBase, self).c_headers()
        return headers

    def c_support_code(self):
        ccode = header_text()
        ccode += """
        #define DIMENSION (4)
        #define CHECK_ERR(f, err) \\
            do { \\
                (err) = (f); \\
                if ((err) != E_SUCCESS) { \\
                    printf("Error in file [%s:%d], err code (%d)", \\
                           __FILE__, __LINE__, err); \\
                    exit(1); \\
                } \\
            } while(0)
        """
        return ccode

    def c_support_code_struct(self, node, name):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        sub = {}
        if dtype == 'float32':
            sub['dtype'] = 'float'
            sub['precision'] = 'F32'
        else:
            sub['dtype'] = 'double'
            sub['precision'] = 'F64'
        sub['name'] = name

        ccode = """
        int first_run;
        size_t inputSize[DIMENSION] = {0};
        size_t inputStrides[DIMENSION] = {0};
        size_t outputSize[DIMENSION] = {0};
        size_t outputStrides[DIMENSION] = {0};
        size_t kernelSize[2] = {0};
        size_t kernelStride[2] = {0};
        int inputOffset[2] = {0};

        void *x_internal_buffer = NULL;
        void *x_internal_buffer_get_from_previous_op = NULL;
        void *x_internal_buffer_to_previous = NULL;
        void *z_internal_buffer = NULL;
        void *gz_internal_buffer_get_from_previous_op = NULL;
        void *gz_internal_buffer = NULL;
        void *workspace_buffer = NULL;

        dnnError_t err;
        dnnPrimitive_t pPoolingFwd = NULL;
        dnnPrimitive_t pPoolingBwd = NULL;
        void *pool_res[dnnResourceNumber] = {0};
        int input_buffer_size = 0;

        size_t input_bytes;
        size_t output_bytes;
        size_t workspace_bytes;

        dnnLayout_t x_internal_layout = NULL;
        dnnLayout_t *x_internal_layout_ptr = NULL;
        dnnLayout_t x_internal_layout_get_from_previous_op = NULL;
        dnnLayout_t z_internal_layout = NULL;
        dnnLayout_t gz_internal_layout_get_from_previous_op = NULL;
        dnnLayout_t gz_internal_layout = NULL;
        dnnLayout_t workspace_internal_layout = NULL;
        dnnPrimitive_t convert_gz_to_internal = NULL;
        dnnPrimitive_t convert_x_to_internal = NULL;
        """ % sub
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
        first_run = 1;
        """
        return ccode

    '''
    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"

        ccode = """
        dnnDelete_%(precision)s(convert_gz_to_internal);
        dnnLayoutDelete_%(precision)s(x_internal_layout);
        dnnLayoutDelete_%(precision)s(z_internal_layout);
        dnnLayoutDelete_%(precision)s(workspace_internal_layout);
        """ % locals()
        return ccode
    '''

    def connection_pattern(self, node):
        return [[1], [0], [0], [0]]


class Pool(PoolBase):
    """
    For N-dimensional tensors, consider that the last two dimensions span
    images. This Op downsamples these images by taking the max, sum or average
    over different patch.

    The constructor takes the max, sum or average or different input patches.

    Parameters
    ----------
    ds : list or tuple of two ints
        Downsample factor over rows and column.
        ds indicates the pool region size.
    ignore_border : bool
        If ds doesn't divide imgshape, do we include an extra row/col
        of partial downsampling (False) or ignore it (True).
    st : list or tuple of two ints or None
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding: tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the images,
        pad_h is the size of the top and bottom margins, and pad_w is the size
        of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        ('average_exc_pad' excludes the padding from the count)

    """
    __props__ = ('ignore_border', 'mode')

    '''
    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) == 1:
            # Old interface
            self.ndim = len(node.op.ds)
            self.mode = node.op.mode
            ws = theano.tensor.constant(node.op.ds)
            st = theano.tensor.constant(node.op.st)
            pad = theano.tensor.constant(node.op.padding)
            node.inputs.append(ws)
            node.inputs.append(st)
            node.inputs.append(pad)
            if isinstance(ws, theano.Constant):
                storage_map[ws] = [ws.data]
                compute_map[ws] = [True]
            else:
                storage_map[ws] = [None]
                compute_map[ws] = [False]
            if isinstance(st, theano.Constant):
                storage_map[st] = [st.data]
                compute_map[st] = [True]
            else:
                storage_map[st] = [None]
                compute_map[st] = [False]
            if isinstance(pad, theano.Constant):
                storage_map[pad] = [pad.data]
                compute_map[pad] = [True]
            else:
                storage_map[pad] = [None]
                compute_map[pad] = [False]
    '''

    def make_node(self, x, ws, stride=None, pad=None):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise NotImplementedError("MKL Pool only supports 4D tensor!")

        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        elif isinstance(pad, (tuple, list)):
            if isinstance(ws, (tuple, list)):
                if any(pad[i] >= ws[i] for i in range(nd)):
                    raise NotImplementedError(
                        'padding must be smaller than strides')
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.ndim == 1
        assert stride.ndim == 1
        assert pad.ndim == 1
        if x.type.ndim < nd:
            raise TypeError()
        if ws.dtype not in tensor.int_dtypes:
            raise TypeError('Pool downsample parameters must be ints.')
        if stride.dtype not in tensor.int_dtypes:
            raise TypeError('Stride parameters must be ints.')
        if pad.dtype not in tensor.int_dtypes:
            raise TypeError('Padding parameters must be ints.')
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:-nd] + (False,) * nd
        out = tensor.TensorType(x.dtype, broad)
        return Apply(self, [x, ws, stride, pad], [out()])

    def infer_shape(self, node, in_shapes):
        ws, stride, pad = [node.inputs[1], node.inputs[2], node.inputs[3]]
        shp = self.out_shape(in_shapes[0], ws, self.ignore_border, stride,
                             pad, self.ndim)
        return [shp]

    def grad(self, inp, grads):
        x, ws, stride, pad = inp
        z = self(*inp)
        gz, = grads
        disc = [DisconnectedType()() for i in inp[1:]]

        return [PoolGrad(ignore_border=self.ignore_border,
                         mode=self.mode)(x, z, gz, ws, stride, pad)] + disc

    def c_code(self, node, name, inp, out, sub):
        x, ws, stride, pad = inp
        z, = out

        if 'max' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingMax"
        elif 'min' == self.mode:
            sub['algo'] = 'dnnAlgorithmPoolingMin'
        elif 'average_exc_pad' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingAvgExcludePadding"
        elif 'average_inc_pad' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingAvgIncludePadding"
        else:
            raise ValueError("mode must be one of 'max', 'min', "
                             "'average_exc_pad', and 'average_inc_pad'")

        if self.ignore_border:
            sub['borderType'] = 'dnnBorderZerosAsymm'
            sub['ignore_border'] = 1
        else:
            sub['borderType'] = 'dnnBorderZeros'
            sub['ignore_border'] = 0

        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'
        else:
            raise TypeError('input must be float32 or float64')

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        #ifdef _MKL_DEBUG_
            std::cout<<"pool start"<<std::endl;
        #endif

        if (1 == first_run) {
            size_t kernel_h = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
            size_t kernel_w = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
            size_t stride_h = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
            size_t stride_w = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
            size_t pad_h = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
            size_t pad_w = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));

            kernelSize[0] = kernel_w;
            kernelSize[1] = kernel_h;
            kernelStride[0] = stride_w;
            kernelStride[1] = stride_h;
            inputOffset[0] = -pad_w;
            inputOffset[1] = -pad_h;

            int out_h, out_w; // shape of the output
            int in_h, in_w; // shape of the padded_input
            in_h = PyArray_DIMS(%(x)s)[2];
            in_w = PyArray_DIMS(%(x)s)[3];

            if (%(ignore_border)s) {
                out_h = floor((float)(in_h + 2 * pad_h - kernel_h)/stride_h) + 1;
                out_w = floor((float)(in_w + 2 * pad_w - kernel_w)/stride_w) + 1;
            } else {
                out_h = ceil((float)(in_h + 2 * pad_h - kernel_h)/stride_h) + 1;
                out_w = ceil((float)(in_w + 2 * pad_w - kernel_w)/stride_w) + 1;
            }
            if (pad_h || pad_w) {
                if ((out_h - 1) * stride_h >= (in_h + pad_h)) {
                    --out_h;
                }
                if ((out_w - 1) * stride_w >= (in_w + pad_w)) {
                    --out_w;
                }
                assert((out_h - 1) * stride_h < in_h + pad_h);
                assert((out_w - 1) * stride_w < in_w + pad_w);
            }

            inputSize[0] = PyArray_DIMS(%(x)s)[3];  //w
            inputSize[1] = PyArray_DIMS(%(x)s)[2];  //h
            inputSize[2] = PyArray_DIMS(%(x)s)[1];  //c
            inputSize[3] = PyArray_DIMS(%(x)s)[0];  //n
            inputStrides[0] = 1;
            inputStrides[1] = inputSize[0];
            inputStrides[2] = inputSize[0] * inputSize[1];
            inputStrides[3] = inputSize[0] * inputSize[1] * inputSize[2];

            outputSize[0] = out_w;
            outputSize[1] = out_h;
            outputSize[2] = inputSize[2];
            outputSize[3] = inputSize[3];
            outputStrides[0] = 1;
            outputStrides[1] = outputSize[0];
            outputStrides[2] = outputSize[0] * outputSize[1];
            outputStrides[3] = outputSize[0] * outputSize[1] * outputSize[2];
        }
        #ifdef _MKL_DEBUG_
            std::cout << "inputSize: " << inputSize[3] << "x" << inputSize[2] << "x" << inputSize[1] << "x" << inputSize[0] << std::endl;
            std::cout << "outputSize: " << outputSize[3] << "x" << outputSize[2] << "x" << outputSize[1] << "x" << outputSize[0] << std::endl;
            std::cout << "pooling region: " << kernelSize[0] << "x" << kernelSize[1] << std::endl;
            std::cout << "pooling stride: " << kernelStride[0] << "x" << kernelStride[1] << std::endl;
            std::cout << "padding: " << inputOffset[0] << "x" << inputOffset[1] << std::endl;
            std::cout << "ignore_border: " << %(ignore_border)s << std::endl;
        #endif

        x_internal_layout_get_from_previous_op = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];
        x_internal_buffer_get_from_previous_op = ((void **)PyArray_DATA(%(x)s))[1];

        if (NULL == pPoolingFwd) {
            CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&pPoolingFwd, NULL,
                       %(algo)s, x_internal_layout_get_from_previous_op, kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );
        }

        if (NULL == x_internal_layout) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &x_internal_layout, pPoolingFwd, dnnResourceSrc), err );
        }
        if (NULL == z_internal_layout) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &z_internal_layout, pPoolingFwd, dnnResourceDst), err );
        }
        if (NULL == workspace_internal_layout) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &workspace_internal_layout, pPoolingFwd, dnnResourceWorkspace), err );
        }

        if (NULL == z_internal_buffer) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&z_internal_buffer, z_internal_layout) , err );
        }
        if (NULL == workspace_buffer) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&workspace_buffer, workspace_internal_layout) , err );
        }

        pool_res[dnnResourceWorkspace] = workspace_buffer;

        npy_intp out_dim[4];
        out_dim[0] = outputSize[3];
        out_dim[1] = outputSize[2];
        out_dim[2] = outputSize[1];
        out_dim[3] = outputSize[0];
        // Prepare output array
        if ( !(%(z)s
            && PyArray_NDIM(%(z)s)==4
            && PyArray_DIMS(%(z)s)[0]==out_dim[0]
            && PyArray_DIMS(%(z)s)[1]==out_dim[1]
            && PyArray_DIMS(%(z)s)[2]==out_dim[2]
            && PyArray_DIMS(%(z)s)[3]==out_dim[3])) {

            if (%(z)s) Py_XDECREF(%(z)s);

            %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION,
                                                  out_dim,
                                                  PyArray_TYPE(%(x)s),
                                                  0);
            if (NULL == %(z)s) {
                PyErr_Format(PyExc_RuntimeError,
                            "Pool: Failed to allocate output of %%lld x %%lld x %%lld x %%lld",
                            (long long)out_dim[0], (long long)out_dim[1], (long long)out_dim[2], (long long)out_dim[3]);
                %(fail)s
            }
        }

        if (!dnnLayoutCompare_%(precision)s(x_internal_layout_get_from_previous_op, x_internal_layout)) {
            #ifdef _MKL_DEBUG_
                std::cout<<"pool forward, x layout from previous op is not equal to internal layout" <<std::endl;
            #endif
            if (NULL == convert_x_to_internal) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_x_to_internal, x_internal_layout_get_from_previous_op, x_internal_layout), err );
            }
        }
        if (convert_x_to_internal) {
            if (NULL == x_internal_buffer) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&x_internal_buffer, x_internal_layout), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(convert_x_to_internal, x_internal_buffer_get_from_previous_op, x_internal_buffer), err );
            x_internal_layout_ptr = &x_internal_layout;
        } else {
            x_internal_buffer = x_internal_buffer_get_from_previous_op;
            x_internal_layout_ptr = &x_internal_layout_get_from_previous_op;
        }

        pool_res[dnnResourceSrc] = x_internal_buffer;
        pool_res[dnnResourceDst] = z_internal_buffer;

        #ifdef _MKL_DEBUG_
            input_bytes = dnnLayoutGetMemorySize_%(precision)s(*x_internal_layout_ptr);
            output_bytes = dnnLayoutGetMemorySize_%(precision)s(z_internal_layout);
            workspace_bytes = dnnLayoutGetMemorySize_%(precision)s(workspace_internal_layout);
            std::cout << " input_bytes = " << input_bytes << std::endl;
            std::cout << " output_bytes = " << output_bytes << std::endl;
            std::cout << " workspace_bytes =  " << workspace_bytes << std::endl;
            std::cout << "pool_res[dnnResourceSrc] = @" << pool_res[dnnResourceSrc] << std::endl;
            std::cout << "pool_res[dnnResourceDst] = @" << pool_res[dnnResourceDst] << std::endl;
            std::cout << "pool_res[dnnResourceWorkspace] = @" << pool_res[dnnResourceWorkspace] << std::endl;
        #endif

        CHECK_ERR( dnnExecute_%(precision)s(pPoolingFwd, (void**)pool_res), err );

        ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = z_internal_layout;
        ((void**)PyArray_DATA(%(z)s))[1] = z_internal_buffer;
        // pass workspace buffer to backward Op
        ((void **)PyArray_DATA(%(z)s))[2] = workspace_buffer;
        ((dnnLayout_t *)PyArray_DATA(%(z)s))[3] = workspace_internal_layout;

        first_run = 0;
        #ifdef _MKL_DEBUG_
            std::cout<<"pool forward, z_internal_buffer: @"<<z_internal_buffer<<", output layout: @"<<z_internal_layout<<std::endl;
            std::cout<<"pool end\\n"<<std::endl;
        #endif
        """ % sub

        return ccode

    def c_code_cache_version(self):
        return (1, 0)


class PoolGrad(PoolBase):
    __props__ = ('ignore_border', 'mode')

    '''
    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) < 5:  # 5 for AveragePoolGrad, 6 for MaxPoolGrad
            # Old interface
            self.ndim = len(node.op.ds)
            self.mode = node.op.mode
            ws = theano.tensor.constant(node.op.ds)
            st = theano.tensor.constant(node.op.st)
            pad = theano.tensor.constant(node.op.padding)
            node.inputs.append(ws)
            node.inputs.append(st)
            node.inputs.append(pad)
            if isinstance(ws, theano.Constant):
                storage_map[ws] = [ws.data]
                compute_map[ws] = [True]
            else:
                storage_map[ws] = [None]
                compute_map[ws] = [False]
            if isinstance(st, theano.Constant):
                storage_map[st] = [st.data]
                compute_map[st] = [True]
            else:
                storage_map[st] = [None]
                compute_map[st] = [False]
            if isinstance(pad, theano.Constant):
                storage_map[pad] = [pad.data]
                compute_map[pad] = [True]
            else:
                storage_map[pad] = [None]
                compute_map[pad] = [False]
    '''

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, x, z, gz, ws, stride=None, pad=None):
        x = tensor.as_tensor_variable(x)
        z = tensor.as_tensor_variable(z)
        gz = tensor.as_tensor_variable(gz)

        if x.type.ndim != 4 or gz.type.ndim != 4:
            raise NotImplementedError("MKL Pool only supports 4D tensor!")

        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert isinstance(x, Variable) and x.ndim >= nd
        assert isinstance(gz, Variable) and gz.ndim >= nd
        assert isinstance(ws, Variable) and ws.ndim == 1
        assert isinstance(stride, Variable) and stride.ndim == 1
        assert isinstance(pad, Variable) and pad.ndim == 1
        assert x.ndim == gz.ndim >= nd
        if ws.dtype not in tensor.int_dtypes:
            raise TypeError('Pool downsample parameters must be ints.')
        if stride.dtype not in tensor.int_dtypes:
            raise TypeError('Stride parameters must be ints.')
        if pad.dtype not in tensor.int_dtypes:
            raise TypeError('Padding parameters must be ints.')

        return Apply(self, [x, z, gz, ws, stride, pad], [x.type()])

    def c_code(self, node, name, inp, out, sub):
        x, z, gz, ws, stride, pad = inp
        gx, = out

        if 'max' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingMax"
        elif 'min' == self.mode:
            sub['algo'] = 'dnnAlgorithmPoolingMin'
        elif 'average_exc_pad' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingAvgExcludePadding"
        elif 'average_inc_pad' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingAvgIncludePadding"
        else:
            raise ValueError("mode must be one of 'max', 'min', "
                             "'average_exc_pad', and 'average_inc_pad'")

        if self.ignore_border:
            sub['borderType'] = 'dnnBorderZerosAsymm'
            sub['ignore_border'] = 1
        else:
            sub['borderType'] = 'dnnBorderZeros'
            sub['ignore_border'] = 0

        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        #ifdef _MKL_DEBUG_
            std::cout<<"poolgrad start"<<std::endl;
        #endif

        if (1 == first_run) {
            size_t kernel_h = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
            size_t kernel_w = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
            size_t stride_h = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
            size_t stride_w = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
            size_t pad_h = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
            size_t pad_w = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));

            kernelSize[0] = kernel_w;
            kernelSize[1] = kernel_h;
            kernelStride[0] = stride_w;
            kernelStride[1] = stride_h;
            inputOffset[0] = -pad_w;
            inputOffset[1] = -pad_h;

            inputSize[0] = PyArray_DIMS(%(x)s)[3];  //w
            inputSize[1] = PyArray_DIMS(%(x)s)[2];  //h
            inputSize[2] = PyArray_DIMS(%(x)s)[1];  //c
            inputSize[3] = PyArray_DIMS(%(x)s)[0];  //n
            inputStrides[0] = 1;
            inputStrides[1] = inputSize[0];
            inputStrides[2] = inputSize[0] * inputSize[1];
            inputStrides[3] = inputSize[0] * inputSize[1] * inputSize[2];

            outputSize[0] = PyArray_DIMS(%(gz)s)[3];
            outputSize[1] = PyArray_DIMS(%(gz)s)[2];
            outputSize[2] = PyArray_DIMS(%(gz)s)[1];
            outputSize[3] = PyArray_DIMS(%(gz)s)[0];
            outputStrides[0] = 1;
            outputStrides[1] = outputSize[0];
            outputStrides[2] = outputSize[0] * outputSize[1];
            outputStrides[3] = outputSize[0] * outputSize[1] * outputSize[2];
        }
        #ifdef _MKL_DEBUG_
            std::cout << "inputSize: " << inputSize[3] << "x" << inputSize[2] << "x" << inputSize[1] << "x" << inputSize[0] << std::endl;
            std::cout << "outputSize: " << outputSize[3] << "x" << outputSize[2] << "x" << outputSize[1] << "x" << outputSize[0] << std::endl;
            std::cout << "pooling region: " << kernelSize[0] << "x" << kernelSize[1] << std::endl;
            std::cout << "pooling stride: " << kernelStride[0] << "x" << kernelStride[1] << std::endl;
            std::cout << "padding: " << inputOffset[0] << "x" << inputOffset[1] << std::endl;
            std::cout << "ignore_border: " << %(ignore_border)s << std::endl;
        #endif

        x_internal_layout_get_from_previous_op = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];

        if (NULL == pPoolingBwd) {
            CHECK_ERR( dnnPoolingCreateBackward_%(precision)s(&pPoolingBwd, NULL,
                       %(algo)s, x_internal_layout_get_from_previous_op, kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );
        }

        if (NULL == pPoolingFwd) {
            CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&pPoolingFwd, NULL,
                       %(algo)s, x_internal_layout_get_from_previous_op, kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );
        }

        if (NULL == x_internal_layout) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &x_internal_layout, pPoolingFwd, dnnResourceSrc), err );
        }
        if (NULL == gz_internal_layout) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &gz_internal_layout, pPoolingFwd, dnnResourceDst), err );
        }

        if (NULL == x_internal_buffer) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&x_internal_buffer, x_internal_layout) , err );
            input_buffer_size = dnnLayoutGetMemorySize_%(precision)s(x_internal_layout);
        }
        #pragma omp parallel for
        #pragma ivdep
        for(int i = 0 ; i < input_buffer_size/sizeof(%(dtype)s); ++i) {
             ((unsigned int*)x_internal_buffer)[i] = 0;
        }

        // Prepare output array
        npy_intp out_dim[4];
        out_dim[0] = PyArray_DIMS(%(x)s)[0];
        out_dim[1] = PyArray_DIMS(%(x)s)[1];
        out_dim[2] = PyArray_DIMS(%(x)s)[2];
        out_dim[3] = PyArray_DIMS(%(x)s)[3];
        if ( !(%(gx)s
            && PyArray_NDIM(%(gx)s)==4
            && PyArray_DIMS(%(gx)s)[0]==out_dim[0]
            && PyArray_DIMS(%(gx)s)[1]==out_dim[1]
            && PyArray_DIMS(%(gx)s)[2]==out_dim[2]
            && PyArray_DIMS(%(gx)s)[3]==out_dim[3])) {

            if (%(gx)s) Py_XDECREF(%(gx)s);

            %(gx)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION,
                                                   out_dim,
                                                   PyArray_TYPE(%(x)s),
                                                   0);
            if (NULL == %(gx)s) {
                PyErr_Format(PyExc_RuntimeError,
                            "PoolGrad: Failed to allocate gx of %%lld x %%lld x %%lld x %%lld",
                            (long long)out_dim[0], (long long)out_dim[1], (long long)out_dim[2], (long long)out_dim[3]);
                %(fail)s
            }
        }

        gz_internal_layout_get_from_previous_op = ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0];
        gz_internal_buffer_get_from_previous_op = ((void **)PyArray_DATA(%(gz)s))[1];

        pool_res[dnnResourceWorkspace] = ((void **)PyArray_DATA(%(z)s))[2];
        workspace_internal_layout = ((dnnLayout_t *)PyArray_DATA(%(z)s))[3];

        if (1 == first_run) {
            if (!dnnLayoutCompare_%(precision)s(gz_internal_layout_get_from_previous_op, gz_internal_layout)) {
            #ifdef _MKL_DEBUG_
                std::cout<<"pool backward, gz layout is not equal" <<std::endl;
            #endif
                if (NULL == convert_gz_to_internal) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_gz_to_internal, gz_internal_layout_get_from_previous_op, gz_internal_layout), err );
                 }
            }
        }

        if (convert_gz_to_internal) {
            if (NULL == gz_internal_buffer) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&gz_internal_buffer, gz_internal_layout), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(convert_gz_to_internal, gz_internal_buffer_get_from_previous_op, gz_internal_buffer), err );
        } else {
             gz_internal_buffer = gz_internal_buffer_get_from_previous_op;
        }
        pool_res[dnnResourceDiffDst] = gz_internal_buffer;
        pool_res[dnnResourceDiffSrc] = x_internal_buffer;

        #ifdef _MKL_DEBUG_
            input_bytes = dnnLayoutGetMemorySize_%(precision)s(x_internal_layout);
            output_bytes = dnnLayoutGetMemorySize_%(precision)s(gz_internal_layout);
            workspace_bytes = dnnLayoutGetMemorySize_%(precision)s(workspace_internal_layout);
            std::cout << " input_bytes = " << input_bytes << std::endl;
            std::cout << " output_bytes = " << output_bytes << std::endl;
            std::cout << " workspace_bytes =  " << workspace_bytes << std::endl;
            std::cout << "pool_res[dnnResourceDiffSrc] = @" << pool_res[dnnResourceDiffSrc] << std::endl;
            std::cout << "pool_res[dnnResourceDiffDst] = @" << pool_res[dnnResourceDiffDst] << std::endl;
            std::cout << "pool_res[dnnResourceWorkspace] = @" << pool_res[dnnResourceWorkspace] << std::endl;
        #endif

        CHECK_ERR( dnnExecute_%(precision)s(pPoolingBwd, (void**)pool_res), err );

        if (!dnnLayoutCompare_%(precision)s(x_internal_layout, x_internal_layout_get_from_previous_op)) {
            #ifdef _MKL_DEBUG_
                std::cout<<"pool backward, x layout is not equal" <<std::endl;
            #endif
            if (NULL == convert_x_to_internal) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_x_to_internal, x_internal_layout, x_internal_layout_get_from_previous_op), err );
            }
        }
        if (convert_x_to_internal) {
            if (NULL == x_internal_buffer_to_previous) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&x_internal_buffer_to_previous, x_internal_layout_get_from_previous_op ), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(convert_x_to_internal, x_internal_buffer, x_internal_buffer_to_previous), err );
         } else {
            x_internal_buffer_to_previous = x_internal_buffer;
        }

        ((dnnLayout_t*)PyArray_DATA(%(gx)s))[0] = x_internal_layout_get_from_previous_op;
        ((void**)PyArray_DATA(%(gx)s))[1] = x_internal_buffer_to_previous;

        #ifdef _MKL_DEBUG_
            std::cout<<"poolgrad end\\n"<<std::endl;
        #endif
        first_run = 0;

        """ % sub

        return ccode

    def c_code_cache_version(self):
        return (1, 0)
