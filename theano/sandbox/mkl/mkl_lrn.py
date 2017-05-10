
from theano import gof, tensor, Variable
from theano.sandbox.mkl import basic_ops, mkl_helper


class AbstractLRN(gof.Op):
    """
    LRN: local response normalization.
    An abstract OP for LRN, called in /tensor/lrn/py.
    This OP will be optimized in local OPT with LRN OP.
    """
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        super(AbstractLRN, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('Input should be a 4-dim tensor')
        return gof.Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [AbstractLRNGrad(alpha=self.alpha,
                                beta=self.beta,
                                k=self.k,
                                n=self.n)(x, gz)]

    def perform(self, node, inp, out):
        print('AbstracLRN is a abstract OP, should not exist in graph..')
        x, = inp
        z, = out


class AbstractLRNGrad(gof.Op):
    """
    LRN: local response normalization.
    An abstract OP for LRN gradient. It will be called in AbstractLRN.grad().
    This OP will be optimized in local OPT with LRNGrad OP.
    """
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        super(AbstractLRNGrad, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n

    def make_node(self, x, gz):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('Input should be a 4-dim tensor')
        return gof.Apply(self, [x, gz], [x.type()])

    def perform(self, node, inp, out):
        print('AbstracLRNGrad is a abstract OP, should not exist in graph..')
        x, gz = inp
        gx, = out


class LRN(basic_ops.MKLOp):
    """
    LRN: local response normalization (Across Maps)

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
    alpha: hyper-parameter
    beta : hyper-parameter
    k    : hyper-parameter
    n    : hyper-parameter, indicates how many nearby maps to use for normalization.
    """
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.k = k

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('Input should be a 4-dim tensor')
        return gof.Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        z = self(*inp)
        gz, = grads
        return [LRNGrad(alpha=self.alpha, beta=self.beta, k=self.k,
                        n=self.n)(x, z, gz)]

    def c_support_code(self):
        support_code = mkl_helper.header_text()
        support_code += """
        #define DIMENSION 4

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
        return support_code

    def c_support_code_struct(self, node, name):
        support_code = """
            dnnError_t err;
            int first_run;
            void* internal_buf;
            void* user_buf;
            dnnLayout_t layout_internal;
            dnnLayout_t layout_usr;
            dnnPrimitive_t to_internal;
            dnnPrimitive_t from_internal;
            dnnPrimitive_t primitive;
            void* convert_resources[dnnResourceNumber];
            size_t bottomSize[DIMENSION];
            size_t bottomStride[DIMENSION];
            void* x_buf_previous;
            dnnLayout_t x_layout_previous;
            dnnLayout_t layout_internal_output;
            dnnLayout_t layout_internal_workspace;
            void* buf_workspace;
            void* buf_output;
            void* lrn_res[dnnResourceNumber];
        """
        return support_code

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            sub['precision'] = 'F32'
        else:
            sub['precision'] = 'F64'

        ccode = """
            // dnnReleaseBuffer_%(precision)s(buf_output);
        """ % sub
        return ccode

    def c_init_code_struct(self, node, name, sub):
        init_code = """
            first_run = 1;
            internal_buf = NULL;
            user_buf = NULL;
            layout_internal = NULL;
            layout_usr = NULL;
            to_internal = NULL;
            from_internal = NULL;
            primitive = NULL;
            x_buf_previous = NULL;
            x_layout_previous = NULL;
            layout_internal_output = NULL;
            layout_internal_workspace = NULL;
            buf_workspace = NULL;
            buf_output = NULL;
        """
        return init_code

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        alpha = self.alpha
        beta = self.beta
        size = self.n
        k = self.k

        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            precision = 'F32'
        else:
            precision = 'F64'

        fail = sub['fail']

        ccode = """
        {
            if (first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3];  // w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2];  // h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1];  // c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0];  // n

                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];
            }

            if ((!%(z)s) ||
                (PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0]) ||
                (PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])) {

                if (%(z)s) {
                    Py_XDECREF(%(z)s);
                }

                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION,
                                                      PyArray_DIMS(%(x)s),
                                                      PyArray_TYPE(%(x)s),
                                                      0);

                if (NULL == %(z)s) {
                    %(fail)s
                }
            }

            x_buf_previous = ((void **)PyArray_DATA(%(x)s))[1];
            x_layout_previous = ((dnnLayout_t *)PyArray_DATA(%(x)s))[0];

            if (first_run) {
                // primitive for LRN
                CHECK_ERR( dnnLRNCreateForward_%(precision)s(&primitive, NULL, x_layout_previous,
                                                             %(size)s, %(alpha)s, %(beta)s, %(k)s), err );

                // internal layout for input
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal, primitive, dnnResourceSrc), err );

                if (!dnnLayoutCompare_%(precision)s(x_layout_previous, layout_internal)) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, x_layout_previous, layout_internal), err );
                }

                // workspace
                if (NULL == layout_internal_workspace) {
                    CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal_workspace, primitive,
                                                                          dnnResourceWorkspace), err );
                }

                if (NULL == buf_workspace) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&buf_workspace, layout_internal_workspace), err );
                }

                dnnLayoutDelete_%(precision)s(layout_internal_workspace);

                // output
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal_output, primitive, dnnResourceDst), err );
            }

            if (NULL != to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal, x_buf_previous, internal_buf), err );
                lrn_res[dnnResourceSrc] = (void*)internal_buf;
            } else {
                lrn_res[dnnResourceSrc] = (void*)x_buf_previous;
            }

            if (NULL == buf_output) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s(&buf_output, layout_internal_output), err );
            }

            lrn_res[dnnResourceDst] = buf_output;
            lrn_res[dnnResourceWorkspace] = buf_workspace;

            CHECK_ERR( dnnExecute_%(precision)s(primitive, lrn_res), err );

            ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_internal_output;
            ((void**)PyArray_DATA(%(z)s))[1] = buf_output;
            // pass to backward
            ((void **)PyArray_DATA(%(z)s))[2] = buf_workspace;
            first_run = 0;
        }
        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (0, 1, 1)


class LRNGrad(basic_ops.MKLOp):
    """
    LRN: local response normalization
    Grad Function of NormAcrossMap
        roOut = gz * f(x)
        f(x) = 1/(1 + (alpha/n)*sum(x*x))**beta - 2*x*alpha*beta*sum(x)/(1+(alpha/n)*sum(x*x))**(beta+1)

    Parameters
    ----------
    alpha:
    beta :
    n    : indicates how many nearby maps to use for normalization.

    """
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n

    def c_support_code(self):
        support_code = mkl_helper.header_text()
        support_code += """
        #define DIMENSION 4

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
        return support_code

    def c_support_code_struct(self, node, name):
        support_code = """
            dnnError_t err;
            int first_run;
            void* internal_buf;
            void* user_buf;
            dnnLayout_t layout_internal;
            dnnLayout_t layout_usr;
            dnnPrimitive_t to_internal;
            dnnPrimitive_t from_internal;
            dnnPrimitive_t primitive;
            void* convert_resources[dnnResourceNumber];
            size_t bottomSize[DIMENSION];
            size_t bottomStride[DIMENSION];
            void* x_buf_previous;
            dnnLayout_t x_layout_previous;
            dnnLayout_t layout_internal_output;
            dnnLayout_t layout_internal_workspace;
            void* buf_diff;
            void* buf_gz;
            void* buf_output;
            void* lrn_res[dnnResourceNumber];
        """
        return support_code

    def c_init_code_struct(self, node, name, sub):
        init_code = """
            first_run = 1;
            internal_buf = NULL;
            user_buf = NULL;
            layout_internal = NULL;
            layout_usr = NULL;
            to_internal = NULL;
            from_internal = NULL;
            primitive = NULL;
            x_buf_previous = NULL;
            x_layout_previous = NULL;
            layout_internal_output = NULL;
            layout_internal_workspace = NULL;
            buf_diff = NULL;
            buf_gz = NULL;
            buf_output = NULL;
        """
        return init_code

    def c_code_cleanup_struct(self, node, name, input_names, output_names, sub):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            sub['precision'] = 'F32'
        else:
            sub['precision'] = 'F64'

        ccode = """
            // dnnReleaseBuffer_%(precision)s(buf_diff);
        """ % sub
        return ccode

    def make_node(self, x, z, gz):
        if not isinstance(x, Variable) or x.type.ndim != 4:
            raise TypeError('Input x type error or dimension error.')
        if not isinstance(z, Variable) or z.type.ndim != 4:
            raise TypeError('Input z type error or dimension error.')
        if not isinstance(gz, Variable) or gz.type.ndim != 4:
            raise TypeError('Inputs gz type error or dimension error.')
        return gof.Apply(self, [x, z, gz], [x.type()])

    def c_code(self, node, name, inp, out, sub):
        x, z, gz, = inp
        gx, = out
        alpha = self.alpha
        beta = self.beta
        size = self.n
        k = self.k

        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            precision = 'F32'
        else:
            precision = 'F64'

        fail = sub['fail']

        ccode = """
        {
            if (first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3];  // w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2];  // h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1];  // c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0];  // n

                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];
            }

            if ((!%(gx)s) ||
                (PyArray_DIMS(%(gx)s)[0] != PyArray_DIMS(%(x)s)[0]) ||
                (PyArray_DIMS(%(gx)s)[1] != PyArray_DIMS(%(x)s)[1])) {

                if (%(gx)s) {
                    Py_XDECREF(%(gx)s);
                }

                %(gx)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION,
                                                      PyArray_DIMS(%(x)s),
                                                      PyArray_TYPE(%(x)s),
                                                      0);

                if (NULL == %(gx)s) {
                    %(fail)s
                }
            }


            x_layout_previous = ((dnnLayout_t *)PyArray_DATA(%(x)s))[0];
            x_buf_previous = ((void **)PyArray_DATA(%(x)s))[1];
            buf_gz = ((void**)PyArray_DATA(%(gz)s))[1];

            if (first_run) {
                CHECK_ERR( dnnLRNCreateBackward_%(precision)s(&primitive, NULL, x_layout_previous, x_layout_previous,
                                                              %(size)s, %(alpha)s, %(beta)s, %(k)s), err );
            }

            if (NULL == buf_diff) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s(&buf_diff, x_layout_previous), err );
            }

            lrn_res[dnnResourceWorkspace] = ((void**)PyArray_DATA(%(z)s))[2];
            lrn_res[dnnResourceDiffDst] = (void*)buf_gz;
            lrn_res[dnnResourceSrc] = (void*)x_buf_previous;
            lrn_res[dnnResourceDiffSrc] = buf_diff;

            CHECK_ERR( dnnExecute_%(precision)s(primitive, lrn_res), err );

            ((dnnLayout_t*)PyArray_DATA(%(gx)s))[0] = x_layout_previous;
            ((void**)PyArray_DATA(%(gx)s))[1] = buf_diff;

            first_run = 0;
        }
        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (0, 1, 1)
