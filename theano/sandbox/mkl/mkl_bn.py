from theano import gof, tensor, Variable
from theano.tensor import as_tensor_variable
from theano.sandbox.mkl import basic_ops, mkl_helper
from theano.gradient import DisconnectedType


class AbstractBatchNormalization(basic_ops.MKLOp):
    __props__ = ('eps', 'bias', 'term', 'inplace', 'train_stage')

    def __init__(self, eps=1e-5, bias=1, term=1, inplace=1, train_stage=1):
        self.eps = eps
        self.bias = bias
        self.term = term
        self.inplace = inplace
        self.train_stage = train_stage

    def make_node(self, x, scale, shift, mean, std):
        x = tensor.as_tensor_variable(x)
        assert x.ndim == 4
        scale = tensor.as_tensor_variable(scale)
        shift = tensor.as_tensor_variable(shift)
        mean = tensor.as_tensor_variable(mean)
        std = tensor.as_tensor_variable(std)

        return gof.Apply(self, [x, scale, shift, mean, std], [x.type()])

    def grad(self, inp, grads):
        x, scale, shift, mean, std = inp
        gz, = grads

        disc = [DisconnectedType()() for i in inp[3:]]
        AbstractBN = AbstractBatchNormalizationGrad(eps=self.eps,
                                                    bias=self.bias,
                                                    term=self.term,
                                                    inplace=self.inplace,
                                                    train_stage=self.train_stage)
        [gx, g_scale, g_shift] = AbstractBN(x, gz, scale, shift)
        return [gx, g_scale, g_shift] + disc

    def perform(self, node, inp, out):
        x, scale, shift, mean, std = inp
        z, = out

    def connection_pattern(self, node):
        return [[1], [1], [1], [0], [0]]


class AbstractBatchNormalizationGrad(basic_ops.MKLOp):
    __props__ = ('eps', 'bias', 'term', 'inplace', 'train_stage')

    def __init__(self, eps=1e-5, bias=1, term=1, inplace=1, train_stage=1):
        self.eps = eps
        self.bias = bias
        self.term = term
        self.inplace = inplace
        self.train_stage = train_stage

    def make_node(self, x, gz, scale, shift):
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        scale = as_tensor_variable(scale)
        shift = as_tensor_variable(shift)
        return gof.Apply(self, [x, gz, scale, shift], [x.type(), scale.type(), shift.type()])

    def perform(self, node, inp, out):
        x, gz, scale, shift = inp
        gx, g_scale, g_shift = out


class BatchNormalization(basic_ops.MKLOp):
    __props__ = ('eps', 'bias', 'term', 'inplace', 'train_stage')

    def __init__(self, eps=1e-5, bias=1, term=1, inplace=1, train_stage=1, uniq_id=1):
        super(BatchNormalization, self).__init__(uniq_id=uniq_id)
        self.eps = eps
        self.bias = bias
        self.term = term
        self.inplace = inplace
        self.train_stage = train_stage
        self.uniq_id = uniq_id

    def make_node(self, x, scale, shift):
        if x.type.ndim != 4:
            raise TypeError()
        x = tensor.as_tensor_variable(x)
        scale = tensor.as_tensor_variable(scale)
        shift = tensor.as_tensor_variable(shift)
        return gof.Apply(self, [x, scale, shift], [x.type()])

    def grad(self, inp, grads):
        x, scale, shift, = inp
        gz, = grads
        gx, g_scale, g_shift = BatchNormalizationGrad(uniq_id=self.uniq_id,
                                                      eps=self.eps,
                                                      bias=self.bias,
                                                      term=self.term)(x, gz, scale, shift)
        return gx, g_scale, g_shift

    def c_support_code(self):
        return mkl_helper.header_text() + """
            static float* scale_buffer_ptr = NULL;
            static float* shift_buffer_ptr = NULL;
            static dnnLayout_t bn_buffer_l = NULL;
            static dnnLayout_t fwd_top_l = NULL;
            static dnnLayout_t bwd_top_l = NULL;
            static dnnLayout_t bwd_bottom_l = NULL;
            static dnnLayout_t scaleShift_buffer_l = NULL;
            static int *bn_buffer = static_cast<int*>(NULL);
            static void *scaleShift_buffer = static_cast<void*>(NULL);
            static dnnPrimitive_t bnFwd  = static_cast<dnnPrimitive_t>(NULL);
            static int first_run=1;
            static int typenum;
            static int count=0;
            static int x_bs;
            static int x_channels;
            static int x_row;
            static int x_col;
            static int data_size;
            static size_t dim = 4;
            static size_t sizes[4];
            static size_t strides[4];
            static dnnError_t e;
            static void* bn_res[dnnResourceNumber];
            static dnnLayout_t layout_previous_layer = NULL;
            static void *input = NULL;
            static void* bp[4];
            static float* input_copy = NULL;
            static void* buffer = NULL;
            #define __DEBUG__ 0
        """

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        dtype = node.inputs[0].type.dtype
        assert dtype in ('float32', 'float64')

        if dtype is 'float32':
            precision = 'F32'
        else:
            precision = 'F64'

        return """
        dnnReleaseBuffer_%s(buffer);
        dnnReleaseBuffer_%s(bn_buffer);
        dnnReleaseBuffer_%s(scaleShift_buffer);
        mkl_free(input_copy);
        """ % precision

    def c_code(self, node, name, inp, out, sub):
        x, scale, shift, = inp
        bn_fwd_out, = out
        eps = self.eps
        bias = self.bias
        term = self.term
        uid = self.uniq_id
        inplace = self.inplace
        train_stage = self.train_stage

        dtype = node.inputs[0].type.dtype
        assert dtype in ('float32', 'float64')

        if dtype is 'float32':
            precision = 'F32'
        else:
            precision = 'F64'

        ret = """
        {
            ((void **)PyArray_DATA(%(x)s))[2] = (void*)bp;
            if(first_run) {
                typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
                x_bs = PyArray_DIMS(%(x)s)[0];
                x_channels = PyArray_DIMS(%(x)s)[1];
                x_row = PyArray_DIMS(%(x)s)[2];
                x_col = PyArray_DIMS(%(x)s)[3];
                sizes[0] = x_col;
                sizes[1] = x_row;
                sizes[2] = x_channels;
                sizes[3] = x_bs;
                data_size = x_bs * x_channels * x_row * x_col;
                strides[0] = 1;
                strides[1] = sizes[0];
                strides[2] = sizes[0]*sizes[1];
                strides[3] = sizes[0]*sizes[1]*sizes[2];
                if(%(bias)s) {
                    scale_buffer_ptr = (float*)PyArray_DATA(%(scale)s);
                    shift_buffer_ptr = (float*)PyArray_DATA(%(shift)s);
                }
            }

            if ((!%(bn_fwd_out)s) || (PyArray_DIMS(%(bn_fwd_out)s)[0] != PyArray_DIMS(%(x)s)[0]) ||
                (PyArray_DIMS(%(bn_fwd_out)s)[1] != PyArray_DIMS(%(x)s)[1])) {
                if(%(bn_fwd_out)s) Py_XDECREF(%(bn_fwd_out)s);
                npy_intp dims[4] = {0, 0, 0, 0};
                dims[0] = PyArray_DIMS(%(x)s)[0];
                dims[1] = PyArray_DIMS(%(x)s)[1];
                dims[2] = PyArray_DIMS(%(x)s)[2];
                dims[3] = PyArray_DIMS(%(x)s)[3];
                %(bn_fwd_out)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum, 0);
            }

            input = ((void **)PyArray_DATA(%(x)s))[1];
            layout_previous_layer = ((dnnLayout_t *)PyArray_DATA(%(x)s))[0];
            dtype_%(bn_fwd_out)s *output = (dtype_%(bn_fwd_out)s *)PyArray_DATA(%(bn_fwd_out)s);
            if(first_run) {
                e = dnnBatchNormalizationCreateForward_%(precision)s(&bnFwd, NULL, layout_previous_layer, %(eps)s);
                if (E_SUCCESS != e) {
                    std::cout<<"bn fwd creat fail\\n";
                    std::cout<<"layout from previous layer "<<layout_previous_layer<<std::endl;
                }

                //create fwd output internal layout
                e = dnnLayoutCreateFromPrimitive_%(precision)s(&fwd_top_l, bnFwd, dnnResourceDst);
                if (e != E_SUCCESS) {
                    std::cout<<"dnnLayoutCreateFromPrimitive fail\\n";
                }
                //create bwd input internal layout
                e = dnnLayoutCreateFromPrimitive_%(precision)s(&bwd_top_l, bnFwd, dnnResourceDst);
                if (e != E_SUCCESS) {
                    std::cout<<"dnnLayoutCreateFromPrimitive fail\\n";
                }
                //create bwd output internal layout
                e = dnnLayoutCreateFromPrimitive_%(precision)s(&bwd_bottom_l, bnFwd, dnnResourceSrc);
                if (e != E_SUCCESS) {
                    std::cout<<"dnnLayoutCreateFromPrimitive fail\\n";
                }
                e = dnnLayoutCreateFromPrimitive_%(precision)s(&bn_buffer_l, bnFwd, dnnResourceWorkspace);
                if (e != E_SUCCESS) {
                    std::cout<<"dnnLayoutCreateFromPrimitive fail\\n";
                }
                e = dnnAllocateBuffer_%(precision)s(reinterpret_cast<void **>(&bn_buffer), bn_buffer_l);
                if (e != E_SUCCESS) {
                    std::cout<<"allocate bn buffer fail with e code "<<e<<std::endl;
                }

                e = dnnLayoutCreateFromPrimitive_%(precision)s(&scaleShift_buffer_l, bnFwd, dnnResourceScaleShift);
                if (e != E_SUCCESS) {
                    std::cout<<"dnnLayoutCreateFromPrimitive fail\\n";
                }
                e = dnnAllocateBuffer_%(precision)s(reinterpret_cast<void**>(&scaleShift_buffer), scaleShift_buffer_l);
                if (e != E_SUCCESS) {
                    std::cout<<"allocate bn buffer fail with e code "<<e<<std::endl;
                }
                dnnLayoutDelete_%(precision)s(scaleShift_buffer_l);
                dnnLayoutDelete_%(precision)s(bn_buffer_l);
                ((void**)bp)[0] = bn_buffer;
                ((void**)bp)[1] = scaleShift_buffer;
                ((void**)bp)[2] = bwd_bottom_l;
                if (!%(bias)s) {
                    for (int i = 0; i < x_channels; i++) {
                        if(((float*)scaleShift_buffer)[i] != 1.0){
                            std::cout<<"scale init failed! "<<((float*)scaleShift_buffer)[i]<<std::endl;
                            exit(0);
                        }
                        if(((float*)scaleShift_buffer)[x_channels + i] != 0) {
                            std::cout<<"shift init failed!"<<std::endl;
                            exit(1);
                        }
                    }
                }
            }
            if ((NULL == buffer) || (first_run)) {
                e = dnnAllocateBuffer_%(precision)s(&buffer, layout_previous_layer);
                if (E_SUCCESS != e) {
                    std::cout<<"fwd bn allocate fail with error code "<<e<<std::endl;
                }
            }

            if (%(bias)s) {
                // Read data from bias weight and bias term buffern to ScaleShift buffer
                #if __DEBUG__
                std::cout<<"fwd copy to scaleShift buffer"<<std::endl;
                #endif
                for (int i = 0; i < x_channels; i++) {
                    ((float*)scaleShift_buffer)[i] = scale_buffer_ptr[i];
                    ((float*)scaleShift_buffer)[x_channels + i] = 0;
                    if (%(term)s) {
                        ((float*)scaleShift_buffer)[x_channels + i] = shift_buffer_ptr[i];
                    }
                }
            }
            bn_res[dnnResourceSrc] = (void*)input;
            bn_res[dnnResourceDst] = buffer;
            ((dnnLayout_t*)output)[0] = fwd_top_l;
            ((void**)output)[1] = buffer;
            bn_res[dnnResourceWorkspace] = bn_buffer;
            bn_res[dnnResourceScaleShift] = scaleShift_buffer;
            if (E_SUCCESS != dnnExecute_%(precision)s(bnFwd, bn_res)) {
                std::cout<<"bn fwd execute fail"<<std::endl;
            }
            first_run = 0;
            #if __DEBUG__
            std::cout<<"bn fwd end, output to file "<<std::endl;
            #endif
        }
        """ % locals()
        return ret

    def c_code_cache_version(self):
        return (0, 1, self.uniq_id)


class BatchNormalizationGrad(basic_ops.MKLOp):
    __props__ = ('eps', 'bias', 'term')

    def __init__(self, eps=1e-5, bias=1, term=1, uniq_id=1):
        super(BatchNormalizationGrad, self).__init__(uniq_id=uniq_id)
        self.eps = eps
        self.bias = bias
        self.term = term
        self.uniq_id = uniq_id

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        dtype = node.inputs[0].type.dtype
        assert dtype in ('float32', 'float64')

        if dtype is 'float32':
            precision = 'F32'
        else:
            precision = 'F64'

        return """
        std::cout<<"releasing buffer\\n";
        dnnReleaseBuffer_%s(buffer);
        """ % precision

    def c_support_code(self):
        return mkl_helper.header_text() + """
        static int first_run=1;
        static int typenum;
        static int x_bs;
        static int x_channels;
        static int x_row;
        static int x_col;
        static size_t dim = 4;
        static size_t sizes[4];
        static size_t strides[4];
        static dnnError_t e;
        static void *input_x=NULL;
        static void *input_gz=NULL;
        static void* buffer=NULL;
        static void* z_buffer=NULL;
        static void* bn_res[dnnResourceNumber];
        static void* BatchNormBwdScaleShift_res[dnnResourceNumber];
        static dnnPrimitive_t bnBwd  = static_cast<dnnPrimitive_t>(NULL);
        static dnnPrimitive_t batchNormBwdScaleShift = static_cast<dnnPrimitive_t>(NULL);
        static dnnPrimitive_t convert_int2int_topgrad_for_weight = NULL;
        static void* ip=NULL;
        static void* bias_weight_p=NULL;
        static void* bias_term_p=NULL;
        static void* scale_shfit_p=NULL;
        static float* g_scale_buffer_ptr=NULL;
        static float* g_shift_buffer_ptr=NULL;
        static dnnLayout_t layout_previous_layer = NULL;
        static int data_size;
        #define __DEBUG__ 0
        """

    def make_node(self, x, gz, scale, shift):
        scale = as_tensor_variable(scale)
        shift = as_tensor_variable(shift)
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        return gof.Apply(self, [x, gz, scale, shift], [x.type(), scale.type(), shift.type()])

    def c_code(self, node, name, inp, out, sub):
        x, gz, scale, shift, = inp
        z, g_scale, g_shift, = out
        eps = self.eps
        bias = self.bias
        term = self.term
        uid = self.uniq_id

        dtype = node.inputs[0].type.dtype
        assert dtype in ('float32', 'float64')
        if dtype is 'float32':
            precision = 'F32'
        else:
            precision = 'F64'

        ret = """
        {
            ip = ((void**)PyArray_DATA(%(x)s))[2];
            if(first_run) {
                typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
                x_bs = PyArray_DIMS(%(x)s)[0];
                x_channels = PyArray_DIMS(%(x)s)[1];
                x_row = PyArray_DIMS(%(x)s)[2];
                x_col = PyArray_DIMS(%(x)s)[3];
                sizes[0] = x_col;
                sizes[1] = x_row;
                sizes[2] = x_channels;
                sizes[3] = x_bs;
                strides[0] = 1;
                strides[1] = sizes[0];
                strides[2] = sizes[0]*sizes[1];
                strides[3] = sizes[0]*sizes[1]*sizes[2];
                layout_previous_layer = ((dnnLayout_t *)PyArray_DATA(%(x)s))[0];
                data_size = x_bs * x_channels * x_row * x_col;
            }

            if ((!%(z)s)
              ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
              ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])) {
                if(%(z)s) Py_XDECREF(%(z)s);
                npy_intp dims[4] = {0, 0, 0, 0};
                dims[0] = x_bs;
                dims[1] = x_channels;
                dims[2] = x_row;
                dims[3] = x_col;
                //TODO: zeros not necessary
                %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum, 0);
            }
            input_x = ((void **)PyArray_DATA(%(x)s))[1];

            #if __DEBUG__
            std::cout<<"bn bwd inputx address "<<std::hex<<input_x<<std::endl;
            #endif

            dtype_%(z)s *output = (dtype_%(z)s*)PyArray_DATA(%(z)s);
            input_gz = ((void**)PyArray_DATA(%(gz)s))[1];

            if(first_run) {
                e = dnnBatchNormalizationCreateBackwardData_%(precision)s(&bnBwd, NULL, layout_previous_layer, %(eps)s);
                if (E_SUCCESS != e) {
                    std::cout<<"bn bwd creat fail\\n";
                }
                if (%(bias)s) {
                    e = dnnBatchNormalizationCreateBackwardScaleShift_%(precision)s(&batchNormBwdScaleShift, NULL, layout_previous_layer, %(eps)s);
                    %(g_scale)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(scale)s),
                                                               PyArray_DIMS(%(scale)s),
                                                               PyArray_TYPE(%(scale)s),
                                                               0);
                    %(g_shift)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(shift)s),
                                                               PyArray_DIMS(%(shift)s),
                                                               PyArray_TYPE(%(shift)s),
                                                               0);
                    if ((NULL == %(g_shift)s)||(NULL == %(g_scale)s)) {
                        std::cout<<"Allocate g_scale buffer failed"<<std::endl;
                    }
                    //CHECK_EQ(e, E_SUCCESS);
                    g_scale_buffer_ptr = (float*)PyArray_DATA(%(g_scale)s);
                    g_shift_buffer_ptr = (float*)PyArray_DATA(%(g_shift)s);
                }
                if (NULL == buffer) {
                    e = dnnAllocateBuffer_%(precision)s(&buffer, ((dnnLayout_t*)ip)[2]);
                    if (E_SUCCESS != e) {
                        std::cout<<"bwd bn allocate fail with error code "<<e<<std::endl;
                    }
                }
            }

            bn_res[dnnResourceWorkspace] = ((void**)ip)[0];
            bn_res[dnnResourceScaleShift] = ((void**)ip)[1];
            bn_res[dnnResourceDiffDst] = (void*)input_gz;
            bn_res[dnnResourceSrc] = (void*)input_x;
            bn_res[dnnResourceDiffSrc] = buffer;

            ((dnnLayout_t*)output)[0] = layout_previous_layer;
            ((void**)output)[1] = buffer;
            e = dnnExecute_%(precision)s(bnBwd, bn_res);
            if (E_SUCCESS != e) {
                std::cout<<"theano bwd execute fail with error code "<<e<<std::endl;
            }

            if(first_run) {
                if(!dnnLayoutCompare_%(precision)s(((dnnLayout_t*)ip)[2], layout_previous_layer)) {
                    if(z_buffer==NULL) {
                        e = dnnAllocateBuffer_%(precision)s(&z_buffer, layout_previous_layer);
                    }
                    if (NULL == convert_int2int_topgrad_for_weight) {
                        dnnConversionCreate_%(precision)s(&convert_int2int_topgrad_for_weight, ((dnnLayout_t*)ip)[2], layout_previous_layer);
                    }
                    if (NULL != convert_int2int_topgrad_for_weight) {
                        dnnConversionExecute_%(precision)s(convert_int2int_topgrad_for_weight, buffer, z_buffer);
                        ((void**)output)[1] = z_buffer;
                        std::cout<<"theano bn zbuffer\\n";
                    }
                }
            }

            if (%(bias)s) {
                BatchNormBwdScaleShift_res[dnnResourceSrc] = (void*)input_x;
                BatchNormBwdScaleShift_res[dnnResourceWorkspace] = ((void**)ip)[0];
                BatchNormBwdScaleShift_res[dnnResourceDiffScaleShift] = ((void**)ip)[1];
                BatchNormBwdScaleShift_res[dnnResourceDiffDst] = (void*)input_gz;
                e = dnnExecute_%(precision)s(batchNormBwdScaleShift, BatchNormBwdScaleShift_res);

                for (int i = 0; i < x_channels; i++) {
                    g_scale_buffer_ptr[i] = (((float**)ip)[1])[i];
                    g_shift_buffer_ptr[i] = 0;
                    if (%(term)s) {
                        g_shift_buffer_ptr[i] =  (((float**)ip)[1])[x_channels + i];
                    }
                }
            }
            first_run = 0;
            #if __DEBUG__
            std::cout<<"bn bwd end\\n"<<std::endl;
            #endif
        }
        """ % locals()
        return ret

    def c_code_cache_version(self):
        return (0, 1, self.uniq_id)
