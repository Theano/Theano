from __future__ import absolute_import, print_function, division

import numpy

import theano
from theano.tensor.blas import ldflags
from theano import tensor, Apply
from theano.sandbox.mkl import mkl_helper
from theano.gradient import DisconnectedType
from theano.sandbox.mkl.basic_ops import MKLOp


class PoolBase(MKLOp):
    def __init__(self, ignore_border=True, mode='max'):

        if not ignore_border:
            raise NotImplementedError(
                'ignore_border only allows to be True in MKL currently')
        self.ignore_border = ignore_border

        if mode.startswith('average'):
            mode = 'average'

        if mode not in ['max', 'min', 'average']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'min',"
                " 'average'. Got %s" % mode)
        self.mode = mode

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
        return mkl_helper.header_text()

    def c_support_code_apply(self, node, name):
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
            #define __DEBUG__ 0
            #define USER_LAYOUT 0
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

        ccode += """
            static int first_run = 1;
            static size_t inputSize[DIMENSION] = {0};
            static size_t inputStrides[DIMENSION] = {0};
            static size_t outputSize[DIMENSION] = {0};
            static size_t outputStrides[DIMENSION] = {0};
            static size_t kernelSize[2] = {0};
            static size_t kernelStride[2] = {0};
            static int inputOffset[2] = {0};

            static void *input_buffer_ptr = NULL;
            static void *input_buffer_ptr_from_previous = NULL;
            static void *input_buffer_ptr_to_previous = NULL;
            static void *output_buffer_ptr = NULL;
            static void *gz_buffer_ptr = NULL;
            static void *gz_buffer_tmp_ptr = NULL;
            static void *workspace_buffer_ptr = NULL;

            static dnnError_t err;
            static dnnPrimitive_t pPoolingFwd = NULL;
            static dnnPrimitive_t pPoolingBwd = NULL;
            static void *pool_res[dnnResourceNumber] = {0};
            static int input_buffer_size = 0;

            /////////////// only for debug usage ////////////////////
            size_t input_bytes;
            size_t output_bytes;
            size_t workspace_bytes;
            ////////////////////////////////////////////////////////

            ////FIXME, remove below definition if it's handled in conversion Op
            static dnnLayout_t usr_layout_input = NULL;
            static dnnLayout_t usr_layout_output = NULL;
            static dnnLayout_t int_layout_input = NULL;
            static dnnLayout_t *int_layout_input_ptr = NULL;
            static dnnLayout_t int_layout_input_from_previous = NULL;
            static dnnLayout_t int_layout_output = NULL;
            static dnnLayout_t gz_int_layout_from_other = NULL;
            static dnnLayout_t gz_int_layout = NULL;
            static dnnLayout_t int_layout_workspace = NULL;
            static dnnLayout_t *int_layout_workspace_p = NULL;
            static dnnPrimitive_t cvt_to_int_input = NULL;
            static dnnPrimitive_t cvt_gz_to_int = NULL;
            static dnnPrimitive_t cvt_from_int_input = NULL;
            static dnnPrimitive_t cvt_from_int_output = NULL;
            static dnnPrimitive_t convert_int2int_input = NULL;

            static void *workspace_ptr_ptr[2];
            static void *workspace_ptr = NULL;
            ////END
        """ % sub
        return ccode

    '''
    def c_support_code_struct(self, node, name):
        ccode = """
        """
        return ccode
    '''

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"

        ccode = """
            //dnnDelete_%(precision)s(cvt_to_int_input);
            //dnnDelete_%(precision)s(cvt_gz_to_int);
            //dnnDelete_%(precision)s(cvt_from_int_input);
            //dnnDelete_%(precision)s(cvt_from_int_output);
            //dnnLayoutDelete_%(precision)s(usr_layout_input);
            //dnnLayoutDelete_%(precision)s(usr_layout_output);
            //dnnLayoutDelete_%(precision)s(int_layout_input);
            //dnnLayoutDelete_%(precision)s(int_layout_output);
            //dnnLayoutDelete_%(precision)s(int_layout_workspace);
        """ % locals()
        return ccode

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
        ('average_inc_pad' excludes the padding from the count,
        'average_exc_pad' include it)

    """
    __props__ = ('ignore_border', 'mode', 'uniq_id')

    def __init__(self, ignore_border=True, mode='max', uniq_id=0):
        super(Pool, self).__init__(ignore_border, mode)
        self.uniq_id = uniq_id

    def __eq__(self, other):
        if hasattr(self, '__props__'):
            if type(self) != type(other):
                return False
            else:
                self_props = [getattr(self, p) for p in self.__props__ if p != 'uniq_id']
                other_props = [getattr(other, p) for p in other.__props__ if p != 'uniq_id']
                if self_props == other_props:
                    return True
                else:
                    return False
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.ignore_border) ^ hash(self.mode)

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
        """
        Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple, list, or similar of integer or scalar Theano variable
            The shape of a tensor of images. The last two elements are
            interpreted as the number of rows, and the number of cols.
        ds : list or tuple of two ints
            Downsample factor over rows and columns this parameter indicates
            the size of the pooling region.
        st : list or tuple of two ints
            The stride size. This is the distance between the pooling regions.
            If it's set to None, it equals ds.
        ignore_border : bool
            If ds doesn't divide imgshape, do we include an extra row/col of
            partial downsampling (False) or ignore it (True).
        padding : tuple of two ints
            (pad_h, pad_w), pad zeros to extend beyond four borders
            of the images, pad_h is the size of the top and bottom margins,
            and pad_w is the size of the left and right margins.

        Returns
        -------
        list
            The shape of the output from this op, for input of given shape.
            This will have the same length as imgshape, but with last two
            elements reduced as per the downsampling & ignore_border flags.

        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]
        r = tensor.extract_constant(r)
        c = tensor.extract_constant(c)

        # TODO CY, looks like no need to make it float then ceil
        out_r = numpy.ceil(((r + 2 * padding[0] - ds[0])) / (st[0])) + 1
        out_c = numpy.ceil(((c + 2 * padding[1] - ds[1])) / (st[1])) + 1

        if padding[0]:
            if isinstance(r, theano.Variable) or isinstance(out_r, theano.Variable):
                out_r = tensor.switch(tensor.ge(((out_r - 1) * st[0]), (r + padding[0])),
                                      out_r - 1, out_r)
                assert(tensor.lt(((out_r - 1) * st[0]), (r + padding[0])))
            else:
                if ((out_r - 1) * st[0]) >= (r + padding[0]):
                    out_r -= 1
                assert(((out_r - 1) * st[0]) < (r + padding[0]))

        if padding[1]:
            if isinstance(c, theano.Variable) or isinstance(out_c, theano.Variable):
                out_c = tensor.switch(tensor.ge(((out_c - 1) * st[1]), (c + padding[1])),
                                      out_c - 1, out_c)
                assert(tensor.lt(((out_c - 1) * st[1]), (c + padding[1])))
            else:
                if ((out_c - 1) * st[1]) >= (c + padding[1]):
                    out_c -= 1
                assert(((out_c - 1) * st[1]) < (c + padding[1]))

        if isinstance(out_r, theano.Variable):
            nr = tensor.cast(out_r, 'int32')
        else:
            nr = numpy.int(out_r)

        if isinstance(out_c, theano.Variable):
            nc = tensor.cast(out_c, 'int32')
        else:
            nc = numpy.int(out_c)

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    def make_node(self, x, ws, stride=None, pad=(0, 0)):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError()

        if stride is None:
            stride = ws

        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)

        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)

        return Apply(self, [x, ws, stride, pad], [out()])

    def infer_shape(self, node, in_shapes):
        ws, stride, pad = [node.inputs[1], node.inputs[2], node.inputs[3]]
        shp = self.out_shape(in_shapes[0], ws, self.ignore_border, stride, pad)
        return [shp]

    def grad(self, inp, grads):
        x, ws, stride, pad = inp
        gz, = grads
        disc = [DisconnectedType()() for i in inp[1:]]

        return [PoolGrad(ignore_border=self.ignore_border,
                         mode=self.mode,
                         uniq_id=self.uniq_id)(x, gz, ws, stride, pad)] + disc

    def c_code(self, node, name, inp, out, sub):
        x, ws, stride, pad = inp
        z, = out

        if 'max' == self.mode:
            algo = "dnnAlgorithmPoolingMax"
        elif 'min' == self.mode:
            algo = 'dnnAlgorithmPoolingMin'
        elif self.mode.startswith('average'):
            algo = "dnnAlgorithmPoolingAvg"
        else:
            raise ValueError("mode must be one of 'max', 'min', 'average'")

        '''
        ignore_border = int(self.ignore_border)
        if self.ignore_border:
            borderType = 'dnnBorderZeros'
        else:
            borderType = 'dnnBorderExtrapolation'
        '''
        # FIXME, current mkl only support this type
        borderType = 'dnnBorderZeros'

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
        #if __DEBUG__
        std::cout<<"pool start"<<std::endl;
        #endif
            ((void **)PyArray_DATA(%(x)s))[2] = (void*)workspace_ptr_ptr;
            //printf(\"pool workspace_ptr_ptr:%%x\\n\",workspace_ptr_ptr);

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

            out_h = ceil((float)(in_h + 2 * pad_h - kernel_h)/stride_h) + 1;
            out_w = ceil((float)(in_w + 2 * pad_w - kernel_w)/stride_w) + 1;
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
        #if __DEBUG__
            std::cout << "inputSize: " << inputSize[3] << "x" << inputSize[2] << "x" << inputSize[1] << "x" << inputSize[0] << std::endl;
            std::cout << "outputSize: " << outputSize[3] << "x" << outputSize[2] << "x" << outputSize[1] << "x" << outputSize[0] << std::endl;
            std::cout << "pooling region: " << kernelSize[0] << "x" << kernelSize[1] << std::endl;
            std::cout << "pooling stride: " << kernelStride[0] << "x" << kernelStride[1] << std::endl;
            std::cout << "padding: " << inputOffset[0] << "x" << inputOffset[1] << std::endl;
        #endif

        // get internal layout for gz from previous Op
        int_layout_input_from_previous = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];
        // get internal buffer for gz from previous op
        input_buffer_ptr_from_previous = ((void **)PyArray_DATA(%(x)s))[1];

        #if __DEBUG__
            std::cout <<"pool forward, int_layout_input_from_previous: @"<<int_layout_input_from_previous<<std::endl;
            std::cout <<"pool forward, input_buffer_ptr_from_previous: @"<<input_buffer_ptr_from_previous<<std::endl;
        #endif

        if (NULL == pPoolingFwd) {
            CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&pPoolingFwd, NULL,
                       %(algo)s, int_layout_input_from_previous, kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );
        }

        if (NULL == int_layout_input) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &int_layout_input, pPoolingFwd, dnnResourceSrc), err );
        }
        if (NULL == int_layout_output) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &int_layout_output, pPoolingFwd, dnnResourceDst), err );
        }
        if (NULL == int_layout_workspace) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &int_layout_workspace, pPoolingFwd, dnnResourceWorkspace), err );
        }

        if (NULL == output_buffer_ptr) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&output_buffer_ptr, int_layout_output) , err );
        }
        if (NULL == workspace_buffer_ptr) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&workspace_buffer_ptr, int_layout_workspace) , err );
        }
        pool_res[dnnResourceWorkspace] = workspace_buffer_ptr;
        ((dnnLayout_t**)workspace_ptr_ptr)[0] = &int_layout_workspace;
        ((void**)workspace_ptr_ptr)[1] = workspace_buffer_ptr;

        npy_intp out_dim[4];
        out_dim[0] = outputSize[3];
        out_dim[1] = outputSize[2];
        out_dim[2] = outputSize[1];
        out_dim[3] = outputSize[0];
        // Prepare output array
        int typenum;
        //if ( !(%(z)s
        //        && PyArray_NDIM(%(z)s) == 4
        //        && PyArray_IS_C_CONTIGUOUS(%(z)s)
        //        && PyArray_DIMS(%(z)s)[0] == out_dim[0]
        //        && PyArray_DIMS(%(z)s)[1] == out_dim[1]
        //        && PyArray_DIMS(%(z)s)[2] == out_dim[2]
        //        && PyArray_DIMS(%(z)s)[3] == out_dim[3])) {
        //    Py_XDECREF(%(z)s);
        if ( !(%(z)s) ) {
            typenum = PyArray_TYPE(%(x)s);
            %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION,
                                              out_dim,
                                              typenum,
                                              0);
            if (NULL == %(z)s) {
                PyErr_Format(PyExc_RuntimeError,
                            "PoolBase: Failed to allocate output of %%lld x %%lld x %%lld x %%lld",
                            (long long)out_dim[0], (long long)out_dim[1], (long long)out_dim[2], (long long)out_dim[3]);
                %(fail)s
            }
        }

        if (!dnnLayoutCompare_%(precision)s(int_layout_input_from_previous, int_layout_input)) {
            #if __DEBUG__
                std::cout<<"############ pool forward, input layout is not equal" <<std::endl;
            #endif
            if (NULL == convert_int2int_input) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_int2int_input, int_layout_input_from_previous, int_layout_input), err );
            }
        }
        if (convert_int2int_input) {
            if (NULL == input_buffer_ptr) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&input_buffer_ptr, int_layout_input), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(convert_int2int_input, input_buffer_ptr_from_previous, input_buffer_ptr), err );
            int_layout_input_ptr = &int_layout_input;
        } else {
            int_layout_input_ptr = &int_layout_input_from_previous;
            input_buffer_ptr = input_buffer_ptr_from_previous;
        }

        pool_res[dnnResourceSrc] = input_buffer_ptr;
        pool_res[dnnResourceDst] = output_buffer_ptr;

        #if __DEBUG__
        input_bytes = dnnLayoutGetMemorySize_%(precision)s(*int_layout_input_ptr);
        output_bytes = dnnLayoutGetMemorySize_%(precision)s(int_layout_output);
        workspace_bytes = dnnLayoutGetMemorySize_%(precision)s(int_layout_workspace);
        std::cout << " input_bytes = " << input_bytes << std::endl;
        std::cout << " output_bytes = " << output_bytes << std::endl;
        std::cout << " workspace_bytes =  " << workspace_bytes << std::endl;
        std::cout << "pool_res[dnnResourceSrc] = @" << pool_res[dnnResourceSrc] << std::endl;
        std::cout << "pool_res[dnnResourceDst] = @" << pool_res[dnnResourceDst] << std::endl;
        std::cout << "pool_res[dnnResourceWorkspace] = @" << pool_res[dnnResourceWorkspace] << std::endl;
        #endif

        CHECK_ERR( dnnExecute_%(precision)s(pPoolingFwd, (void**)pool_res), err );

        ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = int_layout_output;
        ((void**)PyArray_DATA(%(z)s))[1] = output_buffer_ptr;

        #if __DEBUG__
            float *out_p = (float *)workspace_buffer_ptr;
            printf(\"pool forward, workspace; %%g, %%g, %%g, %%g, %%g\\n\", out_p[0], out_p[1],out_p[2],out_p[3],out_p[4]);
            if (dnnLayoutGetMemorySize_%(precision)s(int_layout_output) != (outputSize[0] * outputSize[1] * outputSize[2] * outputSize[3] * sizeof(%(dtype)s))) {
                printf(\"ERROR: conv forward, z view size NOT equal with z_layout!!!!!!\\n\");
            }
        #endif

        first_run = 0;
        #if __DEBUG__
        std::cout<<"pool forward, output_buffer_ptr: @"<<output_buffer_ptr<<", output layout: @"<<int_layout_output<<std::endl;
        std::cout<<"pool end\\n"<<std::endl;
        #endif
        """ % sub

        return ccode

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)


class PoolGrad(PoolBase):
    __props__ = ('ignore_border', 'mode', 'uniq_id')

    def __init__(self, ignore_border=False, mode='max', uniq_id=0):
        super(PoolGrad, self).__init__(ignore_border, mode)
        self.uniq_id = uniq_id

    def __eq__(self, other):
        if hasattr(self, '__props__'):
            if type(self) != type(other):
                return False
            else:
                self_props = [getattr(self, p) for p in self.__props__
                              if p != 'uniq_id']
                other_props = [getattr(other, p) for p in other.__props__
                               if p != 'uniq_id']
                if self_props == other_props:
                    return True
                else:
                    return False
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.ignore_border) ^ hash(self.mode)

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
        """Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple of integers or scalar Theano variables
            the shape of a tensor of images. The last two elements are
            interpreted as the number of rows, and the number of cols.
        ds : tuple of two ints
            downsample factor over rows and columns this parameter
            indicates the size of the pooling region
        st : tuple of two ints
            the stride size. This is the distance between the pooling
            regions. If it's set to None, in which case it equlas ds.
        ignore_border : bool
            if ds doesn't divide imgshape, do we include an extra
            row/col of partial downsampling (False) or ignore it
            (True).
        padding : tuple of two ints
            (pad_h, pad_w), pad zeros to extend beyond four borders of
            the images, pad_h is the size of the top and bottom
            margins, and pad_w is the size of the left and right
            margins.

        Returns
        -------
        list :
            the shape of the output from this op, for input of given
            shape.  This will have the same length as imgshape, but
            with last two elements reduced as per the downsampling &
            ignore_border flags.

        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]
        r = tensor.extract_constant(r)
        c = tensor.extract_constant(c)

        # TODO CY, looks like no need to make it float then ceil
        out_r = numpy.ceil(((r + 2 * padding[0] - ds[0])) / (st[0])) + 1
        out_c = numpy.ceil(((c + 2 * padding[1] - ds[1])) / (st[1])) + 1

        if padding[0]:
            if isinstance(r, theano.Variable) or isinstance(out_r, theano.Variable):
                out_r = tensor.switch(tensor.ge(((out_r - 1) * st[0]), (r + padding[0])),
                                      out_r - 1, out_r)
                assert(tensor.lt(((out_r - 1) * st[0]), (r + padding[0])))
            else:
                if ((out_r - 1) * st[0]) >= (r + padding[0]):
                    out_r -= 1
                assert(((out_r - 1) * st[0]) < (r + padding[0]))

        if padding[1]:
            if isinstance(c, theano.Variable) or isinstance(out_c, theano.Variable):
                out_c = tensor.switch(tensor.ge(((out_c - 1) * st[1]), (c + padding[1])),
                                      out_c - 1, out_c)
                assert(tensor.lt(((out_c - 1) * st[1]), (c + padding[1])))
            else:
                if ((out_c - 1) * st[1]) >= (c + padding[1]):
                    out_c -= 1
                assert(((out_c - 1) * st[1]) < (c + padding[1]))

        if isinstance(out_r, theano.Variable):
            nr = tensor.cast(out_r, 'int32')
        else:
            nr = numpy.int(out_r)

        if isinstance(out_c, theano.Variable):
            nc = tensor.cast(out_c, 'int32')
        else:
            nc = numpy.int(out_c)

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, x, gz, ws, stride, pad):
        x = tensor.as_tensor_variable(x)
        gz = tensor.as_tensor_variable(gz)

        if x.type.ndim != 4 or gz.type.ndim != 4:
            raise TypeError()

        if stride is None:
            stride = ws

        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)

        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)

        return Apply(self, [x, gz, ws, stride, pad], [out()])

    def c_code(self, node, name, inp, out, sub):
        x, gz, ws, stride, pad = inp
        gx, = out

        if 'max' == self.mode:
            algo = "dnnAlgorithmPoolingMax"
        elif 'min' == self.mode:
            algo = 'dnnAlgorithmPoolingMin'
        elif self.mode.startswith('average'):
            algo = "dnnAlgorithmPoolingAvg"
        else:
            raise ValueError("mode must be one of 'max', 'min', 'average'")

        '''
        ignore_border = int(self.ignore_border)
        if self.ignore_border:
            borderType = 'dnnBorderZeros'
        else:
            borderType = 'dnnBorderExtrapolation'
        '''
        borderType = 'dnnBorderZeros'

        if node.inputs[0].type.dtype == "float32":  # FIXME, remove if it's defined in other place
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        #if __DEBUG__
        std::cout<<"poolgrad start"<<std::endl;
        #endif
            workspace_ptr = ((void**)PyArray_DATA(%(x)s))[2];
            //printf("workspace_ptr=0x%%x\\n", workspace_ptr);

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

            ///////////// no need to calc output h&w since we can get the shape from gz. remove it.
            //int out_h, out_w; // shape of the output
            //int in_h, in_w; // shape of the padded_input
            //in_h = PyArray_DIMS(%(x)s)[2];
            //in_w = PyArray_DIMS(%(x)s)[3];

            //out_h = ceil((float)(in_h + 2 * pad_h - kernel_h)/stride_h) + 1;
            //out_w = ceil((float)(in_w + 2 * pad_w - kernel_w)/stride_w) + 1;
            //if (pad_h || pad_w) {
            //    if ((out_h - 1) * stride_h >= (in_h + pad_h)) {
            //        --out_h;
            //    }
            //    if ((out_w - 1) * stride_w >= (in_w + pad_w)) {
            //        --out_w;
            //    }
            //    assert((out_h - 1) * stride_h < in_h + pad_h);
            //    assert((out_w - 1) * stride_w < in_w + pad_w);
            //}
            ///////////////////////////////////////////////////////////////////////////////

            //use 'x' instead of '%(x)s' will cause segment fault!!!
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

            #if __DEBUG__
            std::cout << "inputgradSize: " << inputSize[3] << "x" << inputSize[2] << "x" << inputSize[1] << "x" << inputSize[0] << std::endl;
            std::cout << "outputgradSize: " << outputSize[3] << "x" << outputSize[2] << "x" << outputSize[1] << "x" << outputSize[0] << std::endl;
            std::cout << "pooling region: " << kernelSize[0] << "x" << kernelSize[1] << std::endl;
            std::cout << "pooling stride: " << kernelStride[0] << "x" << kernelStride[1] << std::endl;
            std::cout << "padding: " << inputOffset[0] << "x" << inputOffset[1] << std::endl;
            #endif
        }

        // get internal layout for gz from previous Op
        int_layout_input_from_previous = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];

        if (NULL == pPoolingBwd) {
            CHECK_ERR( dnnPoolingCreateBackward_%(precision)s(&pPoolingBwd, NULL,
                       %(algo)s, int_layout_input_from_previous, kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );
        }

        if (NULL == pPoolingFwd) {
            CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&pPoolingFwd, NULL,
                       %(algo)s, int_layout_input_from_previous, kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );
        }

        if (NULL == int_layout_input) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &int_layout_input, pPoolingFwd, dnnResourceSrc), err );
        }
        if (NULL == gz_int_layout) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &gz_int_layout, pPoolingFwd, dnnResourceDst), err );
        }

        if (NULL == input_buffer_ptr) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&input_buffer_ptr, int_layout_input) , err );
            input_buffer_size = dnnLayoutGetMemorySize_%(precision)s(int_layout_input);
        }
        #pragma omp parallel for
        #pragma ivdep
        for(int i = 0 ; i < input_buffer_size/sizeof(%(dtype)s); ++i) {
             ((unsigned int*)input_buffer_ptr)[i] = 0;
        }
        //memset(input_buffer_ptr, 0, dnnLayoutGetMemorySize_%(precision)s(int_layout_input));
        // Prepare output array
        int typenum;
        if (!(%(gx)s)) {

            typenum = PyArray_TYPE(%(gz)s);
            %(gx)s = (PyArrayObject*)PyArray_ZEROS(4,
                                                  PyArray_DIMS(%(x)s),
                                                  typenum,
                                                  0);
            if (NULL == %(gx)s) {
                std::cout<<"allocat fail\\n";
            }
        }

        // get internal buffer for gz from previous op
        gz_int_layout_from_other = ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0];
        gz_buffer_ptr = ((void **)PyArray_DATA(%(gz)s))[1];

        int_layout_workspace_p = ((dnnLayout_t**)workspace_ptr)[0];
        int_layout_workspace = *int_layout_workspace_p;
        pool_res[dnnResourceWorkspace] = ((void**)workspace_ptr)[1];

        if(first_run ==1)
        {
            if (!dnnLayoutCompare_%(precision)s(gz_int_layout_from_other, gz_int_layout)) {
            #if __DEBUG__
                std::cout<<"############ pool backward, gz layout is not equal" <<std::endl;
            #endif
                if (NULL == cvt_gz_to_int) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&cvt_gz_to_int, gz_int_layout_from_other, gz_int_layout), err );
                 }
            }
        }

        if (cvt_gz_to_int) {
            if (NULL == gz_buffer_tmp_ptr) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&gz_buffer_tmp_ptr, gz_int_layout), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(cvt_gz_to_int, gz_buffer_ptr, gz_buffer_tmp_ptr), err );
        } else {
             gz_buffer_tmp_ptr = gz_buffer_ptr;
        }
        pool_res[dnnResourceDiffDst] = gz_buffer_tmp_ptr;
        pool_res[dnnResourceDiffSrc] = input_buffer_ptr;

        #if __DEBUG__
        input_bytes = dnnLayoutGetMemorySize_%(precision)s(int_layout_input);
        output_bytes = dnnLayoutGetMemorySize_%(precision)s(gz_int_layout);
        workspace_bytes = dnnLayoutGetMemorySize_%(precision)s(int_layout_workspace);
        std::cout << " input_bytes = " << input_bytes << std::endl;
        std::cout << " output_bytes = " << output_bytes << std::endl;
        std::cout << " workspace_bytes =  " << workspace_bytes << std::endl;
        #endif

        CHECK_ERR( dnnExecute_%(precision)s(pPoolingBwd, (void**)pool_res), err );

        if (!dnnLayoutCompare_%(precision)s(int_layout_input, int_layout_input_from_previous)) {
            #if __DEBUG__
                std::cout<<"############ pool backward, input layout is not equal" <<std::endl;
            #endif
            if (NULL == convert_int2int_input) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_int2int_input, int_layout_input, int_layout_input_from_previous), err );
            }
        }
        if (convert_int2int_input) {
            if (NULL == input_buffer_ptr_to_previous) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&input_buffer_ptr_to_previous, int_layout_input_from_previous ), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(convert_int2int_input, input_buffer_ptr, input_buffer_ptr_to_previous), err );
         } else {
            input_buffer_ptr_to_previous = input_buffer_ptr;
            //printf(\"D2: %%x\\n\",((dnnLayout_t*)PyArray_DATA(%(gx)s))[0]);
        }

        ((dnnLayout_t*)PyArray_DATA(%(gx)s))[0] = int_layout_input_from_previous;
        ((void**)PyArray_DATA(%(gx)s))[1] = input_buffer_ptr_to_previous;

        first_run = 0;

        """ % sub

        return ccode

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)
