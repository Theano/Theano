from theano import gof, tensor, Variable
from theano.tensor.blas import ldflags
from theano.sandbox.mkl import mkl_helper, basic_ops


class AbstractLRN(gof.Op):
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError()
        return gof.Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [AbstractLRNGrad(alpha=self.alpha,
                                beta=self.beta,
                                k=self.k,
                                n=self.n)(x, gz)]

    def perform(self, node, inp, out):
        x, = inp
        z, = out


class AbstractLRNGrad(gof.Op):
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n

    def make_node(self, x, gz):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError()
        return gof.Apply(self, [x, gz], [x.type()])

    def perform(self, node, inp, out):
        x, gz = inp
        gx, = out


class LRN(basic_ops.MKLOp):
    """
    Local Response Normalization (Across Maps)

    Refer to the below link for the definition of LRN
        https://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_
        response_normalization_layer_(across_maps)

    The activity of a neuron is divided only by the "adjacent" of activities
    which are in the same spatial postion but in different maps (channels).
    'c' stands for current channel index.

        F[c][x,y] = (1 + (alpha*1.0/n) * sum(F[c - n/2][x,y]^2,
                    F[c + n/2][x,y]^2))^beta

    Parameters
    ----------
    alpha:
    beta :
    n    : indicates how many nearby maps to use for normalization.
    """
    __props__ = ('alpha', 'beta', 'k', 'size')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5, uniq_id=0):
        self.alpha = alpha
        self.beta = beta
        self.size = n
        self.k = k
        self.uniq_id = uniq_id

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError()
        return gof.Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [LRNGrad(uniq_id=self.uniq_id, alpha=self.alpha,
                        beta=self.beta, k=self.k, n=self.size)(x, gz)]

    def c_support_code(self):
        return mkl_helper.header_text() + """
            static dnnLayout_t fwd_bottom_data_usr_l;
            static dnnPrimitive_t fwd_bottom_convert_to_int;
            static dnnPrimitive_t fwd_bottom_convert_from_int;
            static dnnPrimitive_t fwd_bottom_convert_prv2prv;
            static dnnLayout_t fwd_top_data_usr_l;
            static dnnLayout_t fwd_top_data_int_l;
            static dnnPrimitive_t fwd_top_convert_to_int;
            static dnnPrimitive_t fwd_top_convert_from_int;
            static dnnPrimitive_t fwd_top_convert_prv2prv;
            static dnnLayout_t fwd_bottom_data_int_l=NULL;
            static dnnLayout_t lrn_buffer_l = NULL;
            static int *lrn_buffer = static_cast<int*>(NULL);
            static dnnPrimitive_t lrnFwd  = static_cast<dnnPrimitive_t>(NULL);
            static int first_run=1;
            static int typenum;
            static int count=0;
            static int x_bs;
            static int x_channels;
            static int x_row;
            static int x_col;
            static size_t dim = 4;
            static size_t sizes[4];
            static size_t strides[4];
            static dnnError_t e;
            static void* lrn_res[dnnResourceNumber];
            static dnnLayout_t layout_previous_layer = NULL;
            static void *input=NULL;
            static void* bp[4];
            static void* buffer=NULL;
            #define __DEBUG__ 0
        """

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if dtype == 'float32':
            sub['precision'] = 'F32'
        else:
            sub['precision'] = 'F64'

        ccode = """
            dnnReleaseBuffer_%s(buffer);
        """ % sub
        return ccode

    def c_headers(self):
        return ['<math.h>', '<iostream>', '<fstream>']

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        lrn_fwd_out, = out
        alpha = self.alpha
        beta = self.beta
        size = self.size
        k = self.k

        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if dtype == 'float32':
            precision = 'F32'
        else:
            precision = 'F64'

        ret = """
        {
            #if __DEBUG__
            std::cout<<"lrn fwd start\\n";
            #endif
            ((void **)PyArray_DATA(%(x)s))[2] = (void *)bp;
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
            }
            if ((!%(lrn_fwd_out)s)
              ||(PyArray_DIMS(%(lrn_fwd_out)s)[0] != PyArray_DIMS(%(x)s)[0])
              ||(PyArray_DIMS(%(lrn_fwd_out)s)[1] != PyArray_DIMS(%(x)s)[1])) {
              if(%(lrn_fwd_out)s) Py_XDECREF(%(lrn_fwd_out)s);
              npy_intp dims[4] = {0, 0, 0, 0};
              dims[0] = PyArray_DIMS(%(x)s)[0];
              dims[1] = PyArray_DIMS(%(x)s)[1];
              dims[2] = PyArray_DIMS(%(x)s)[2];
              dims[3] = PyArray_DIMS(%(x)s)[3];
              %(lrn_fwd_out)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum, 0);
            }

            input = ((void **)PyArray_DATA(%(x)s))[1];
            layout_previous_layer = ((dnnLayout_t *)PyArray_DATA(%(x)s))[0];

            dtype_%(lrn_fwd_out)s *output = (dtype_%(lrn_fwd_out)s *)PyArray_DATA(%(lrn_fwd_out)s);
            if(first_run) {
                //fake a fwd bottom internal layout, should be passed from previous layer
                if (E_SUCCESS != dnnLayoutCreate_%(precision)s(&fwd_bottom_data_int_l, dim, sizes, strides)) {
                    std::cout<<"fwd_bottom_data_int_l creat fail\\n";
                }
                // These 4 usr layouts should be inited once
                if (E_SUCCESS != dnnLayoutCreate_%(precision)s(&fwd_bottom_data_usr_l, dim, sizes, strides)) {
                    std::cout<<"fwd_bottom_data_usr_l creat fail\\n";
                }
                if (E_SUCCESS != dnnLayoutCreate_%(precision)s(&fwd_top_data_usr_l, dim, sizes, strides)) {
                    std::cout<<"fwd_top_data_usr_l creat fail\\n";
                }

                if (E_SUCCESS != dnnLRNCreateForward_%(precision)s(&lrnFwd, NULL, layout_previous_layer, %(size)s, %(alpha)s, %(beta)s, %(k)s)) {
                    std::cout<<"lrn fwd creat fail\\n";
                    std::cout<<"layout from previous layer "<<layout_previous_layer<<std::endl;
                    std::cout<<"size: "<<%(size)s<<", alpha: "<<%(alpha)s<<", beta: "<<%(beta)s<<", k: "<<%(k)s<<std::endl;
                }

                if (E_SUCCESS != dnnLayoutCreateFromPrimitive_%(precision)s(&fwd_top_data_int_l, lrnFwd, dnnResourceDst)) {
                  std::cout<<"fwd_top_data_int_l creat fail\\n";
                }

                if (fwd_top_data_int_l && !dnnLayoutCompare_%(precision)s(fwd_top_data_usr_l, fwd_top_data_int_l)) {
                    //std::cout<<"fwd layout conversion\\n";
                    e = dnnConversionCreate_%(precision)s(&fwd_top_convert_to_int, fwd_top_data_usr_l,fwd_top_data_int_l);
                    if (e != E_SUCCESS) {
                        std::cout<<"dnnConversionCreate_%(precision)s fail with e "<<e<<std::endl;
                    }
                    e = dnnConversionCreate_%(precision)s(&fwd_top_convert_from_int, fwd_top_data_int_l,fwd_top_data_usr_l);
                    if (e != E_SUCCESS) {
                        std::cout<<"dnnConversionCreate_%(precision)s i2u fail with e "<<e<<std::endl;
                    }
                }

                e = dnnLayoutCreateFromPrimitive_%(precision)s(&lrn_buffer_l, lrnFwd, dnnResourceWorkspace);
                if (e != E_SUCCESS) {
                    std::cout<<"dnnLayoutCreateFromPrimitive_%(precision)s fail\\n";
                }

                e = dnnAllocateBuffer_%(precision)s(reinterpret_cast<void **>(&lrn_buffer), lrn_buffer_l);
                if (e != E_SUCCESS) {
                    std::cout<<"allocate lrn buffer fail with e code "<<e<<std::endl;
                }

                dnnLayoutDelete_%(precision)s(lrn_buffer_l);
                ((void**)bp)[0] = lrn_buffer;
            }

            if (NULL == buffer) {
                e = dnnAllocateBuffer_%(precision)s(&buffer, layout_previous_layer);
                if (E_SUCCESS != e){
                    std::cout<<"fwd bn allocate fail with error code "<<e<<std::endl;
                }
            }

            lrn_res[dnnResourceSrc] = (void*)input;
            lrn_res[dnnResourceDst] = buffer;
            ((dnnLayout_t*)output)[0] = layout_previous_layer;
            ((void**)output)[1] = buffer;
            lrn_res[dnnResourceWorkspace] = lrn_buffer;
            if (E_SUCCESS != dnnExecute_%(precision)s(lrnFwd, lrn_res)) {
                std::cout<<"fwd execute fail"<<std::endl;
            }

            first_run=0;
            #if __DEBUG__
            std::cout<<"lrn fwd end\\n"<<std::endl;
            #endif
        }
        """ % locals()
        return ret

    def c_code_cache_version(self):
        return (0, 1, self.uniq_id)


class LRNGrad(basic_ops.MKLOp):
    """
    Grad Function of NormAcrossMap
        roOut = gz * f(x)
        f(x) = 1/(1 + (alpha/n)*sum(x*x))**beta - 2*x*alpha*beta*sum(x)/(1+(alpha/n)*sum(x*x))**(beta+1)

    Parameters
    ----------
    alpha:
    beta :
    n    : indicates how many nearby maps to use for normalization.

    """
    __props__ = ('alpha', 'beta', 'k', 'size')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5, uniq_id=0):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.size = n
        self.uniq_id = uniq_id

    def c_headers(self):
        return ['<math.h>', '<fstream>']

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if dtype == 'float32':
            sub['precision'] = 'F32'
        else:
            sub['precision'] = 'F64'

        ccode = """
            std::cout<<"releasing buffer\\n";
            dnnReleaseBuffer_%s(buffer);
        """ % sub
        return ccode

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
        static dnnLayout_t fwd_bottom_data_int_l;
        static dnnLayout_t bwd_bottom_diff_usr_l;
        static dnnLayout_t bwd_bottom_diff_int_l;
        static dnnPrimitive_t bwd_bottom_convert_to_int;
        static dnnPrimitive_t bwd_bottom_convert_from_int;
        static dnnPrimitive_t bwd_bottom_convert_prv2prv;
        static void* buffer=NULL;
        static dnnLayout_t bwd_top_diff_usr_l;
        static dnnLayout_t bwd_top_diff_int_l;
        static dnnPrimitive_t bwd_top_convert_to_int;
        static dnnPrimitive_t bwd_top_convert_from_int;
        static dnnPrimitive_t bwd_top_convert_prv2prv;
        static void* lrn_res[dnnResourceNumber];
        static dnnPrimitive_t lrnBwd  = static_cast<dnnPrimitive_t>(NULL);
        static void *ip = NULL;
        static dnnLayout_t layout_previous_layer = NULL;
        #define __DEBUG__ 0
        """

    def make_node(self, x, gz):
        if not isinstance(x, Variable) or x.ndim != 4:
            raise TypeError('Input x type error or dimension error.')
        if not isinstance(gz, Variable) or gz.ndim != 4:
            raise TypeError('Inputs gz type error or dimension error.')
        return gof.Apply(self, [x, gz], [x.type()])

    def c_code(self, node, name, inp, out, sub):
        x, gz, = inp
        z, = out
        alpha = self.alpha
        beta = self.beta
        size = self.size
        k = self.k

        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if dtype == 'float32':
            precision = 'F32'
        else:
            precision = 'F64'

        ret = """
        {
            #if __DEBUG__
            std::cout<<"lrn bwd start\\n";
            #endif
            ip = ((void**)PyArray_DATA(%(x)s))[2];
            if(first_run) {
                typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
                x_bs = PyArray_DIMS(%(x)s)[0];
                x_channels = PyArray_DIMS(%(x)s)[1];
                x_row = PyArray_DIMS(%(x)s)[2];
                x_col = PyArray_DIMS(%(x)s)[3];
                //std::cout<<"bwd x shape "<<x_bs<<" channel "<<x_channels<<" row "<<x_row<<" col "<<x_col<<std::endl;
                sizes[0] = x_col;
                sizes[1] = x_row;
                sizes[2] = x_channels;
                sizes[3] = x_bs;
                strides[0] = 1;
                strides[1] = sizes[0];
                strides[2] = sizes[0]*sizes[1];
                strides[3] = sizes[0]*sizes[1]*sizes[2];
                if (E_SUCCESS != dnnLayoutCreate_%(precision)s(&bwd_bottom_diff_usr_l, dim, sizes, strides)) {
                    std::cout<<"bwd_bottom_diff_usr_l creat fail\\n";
                }
                if (E_SUCCESS != dnnLayoutCreate_%(precision)s(&bwd_top_diff_usr_l, dim, sizes, strides)) {
                    std::cout<<"bwd_top_diff_usr_l creat fail\\n";
                }
                layout_previous_layer = ((dnnLayout_t *)PyArray_DATA(%(x)s))[0];
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
            dtype_%(z)s *output = (dtype_%(z)s *)PyArray_DATA(%(z)s);
            input_gz = ((void**)PyArray_DATA(%(gz)s))[1];
            if(first_run) {
                if (E_SUCCESS != dnnLRNCreateBackward_%(precision)s(&lrnBwd, NULL,layout_previous_layer,
                layout_previous_layer,%(size)s, %(alpha)s, %(beta)s, %(k)s)) {
                    std::cout<<"lrn bwd creat fail\\n";
                }
            }

            if (NULL == buffer) {
                e = dnnAllocateBuffer_%(precision)s(&buffer, layout_previous_layer);
                if (E_SUCCESS != e) {
                    std::cout<<"bwd bn allocate fail with error code "<<e<<std::endl;
                }
            }

            lrn_res[dnnResourceWorkspace] = ((void**)ip)[0];
            lrn_res[dnnResourceDiffDst] = (void*)input_gz;
            lrn_res[dnnResourceSrc] = (void*)input_x;
            lrn_res[dnnResourceDiffSrc] = buffer;

            ((dnnLayout_t*)output)[0] = layout_previous_layer;
            ((void**)output)[1] = buffer;

            e = dnnExecute_%(precision)s(lrnBwd, lrn_res);
            if (E_SUCCESS != e) {
                std::cout<<"bwd execute fail with error code "<<e<<std::endl;
            }
            first_run = 0;
            #if __DEBUG__
            std::cout<<"lrn bwd end\\n"<<std::endl;
            #endif
            }
            """ % locals()
        return ret

    def c_code_cache_version(self):
        return (0, 1, self.uniq_id)
