from theano.tensor import as_tensor_variable
from theano.gof import Apply
from theano.tensor.basic import Join, int_types
from theano.tensor.blas import ldflags
from theano.gradient import grad_undefined
from theano.sandbox.mkl import mkl_helper, basic_ops


class Concatenate(basic_ops.MKLOp, Join):
    __props__ = ()

    def __init__(self):
        super(Concatenate, self).__init__()

    def c_headers(self):
        return super(Concatenate, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(Concatenate, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        ccode = mkl_helper.header_text()
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
        ccode = """
        dnnError_t err;
        dnnPrimitive_t pConcat;
        dnnLayout_t z_internal_layout;
        void *z_internal_buffer;
        void *concat_res[dnnResourceNumber];
        npy_intp out_dim[4];
        """
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
        pConcat = NULL;
        z_internal_layout = NULL;
        z_internal_buffer = NULL;
        """
        return ccode

    def c_cleanup_code_struct(self, node, name):
        sub = {}
        if 'float32' == node.inputs[1].type.dtype:
            sub['type'] = 'float'
            sub['precision'] = 'F32'
        elif "float64" == node.inputs[1].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = 'F64'
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[1].type.dtype)
        return """
        if(NULL != pConcat) {
            CHECK_ERR( dnnDelete_%(precision)s(pConcat), err );
            pConcat = NULL;
        }

        if(NULL != z_internal_buffer) {
            CHECK_ERR( dnnReleaseBuffer_%(precision)s(z_internal_buffer), err );
             z_internal_buffer = NULL;
        }

        if(NULL != z_internal_layout) {
            CHECK_ERR( dnnLayoutDelete_%(precision)s(z_internal_layout), err );
             z_internal_layout = NULL;
        }
        """ % sub

    def make_node(self, *axis_and_tensors):
        return Join.make_node(self, *axis_and_tensors)

    def grad(self, inp, grads):
        axis, tensors = inp[0], inp[1:]
        gz, = grads

        rval = [grad_undefined(self, 0, axis)]

        out = ConcatenateGrad()(gz, axis, *tensors)

        if not isinstance(out, list):
            out = [out]

        rval = rval + out
        return rval

    def c_code_cache_version(self):
        return (1, 0)

    def c_code(self, node, name, inputs, out, sub):
        axis, tensors, = inputs[0], inputs[1:]
        input_1 = tensors[0]
        len_of_tensors = len(tensors)
        z, = out

        adtype = node.inputs[0].type.dtype_specs()[1]

        if 'float32' == node.inputs[1].type.dtype:
            sub['type'] = 'float'
            sub['precision'] = 'F32'
        elif "float64" == node.inputs[1].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = 'F64'
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[1].type.dtype)

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        int axis = ((%(adtype)s *)PyArray_DATA(%(axis)s))[0];
        int ndim = PyArray_NDIM(%(input_1)s);
        if (axis < -ndim) {
            PyErr_Format(PyExc_IndexError,
                         "Concatenate axis %%d out of bounds [0, %%d)", axis, ndim);
            %(fail)s
        }

        if(NULL == pConcat) {
             dnnLayout_t x_internal_layout[%(len_of_tensors)s];
             memcpy(out_dim, PyArray_DIMS(%(input_1)s), PyArray_NDIM(%(input_1)s)*sizeof(npy_intp));
             out_dim[axis] = 0;
        """ % sub

        for i, inp in enumerate(tensors):
            d = {}
            d['i'] = i
            d['inp'] = inp
            ccode += """
            x_internal_layout[%(i)s] = ((dnnLayout_t*)PyArray_DATA(%(inp)s))[0];
            concat_res[dnnResourceMultipleSrc + %(i)s] = ((void**)PyArray_DATA(%(inp)s))[1];
            out_dim[axis] = out_dim[axis] + PyArray_DIMS(%(inp)s)[axis];
            """ % d

        ccode += """
            CHECK_ERR( dnnConcatCreate_%(precision)s(&pConcat,
                       NULL, %(len_of_tensors)s, x_internal_layout), err );
        }
        if(NULL == z_internal_layout) {
            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&z_internal_layout,
                       pConcat, dnnResourceDst), err );
        }

        if ( !(%(z)s
               && PyArray_NDIM(%(z)s) == PyArray_NDIM(%(input_1)s)
               && PyArray_DIMS(%(z)s)[0] == out_dim[0]
               && PyArray_DIMS(%(z)s)[1] == out_dim[1]
               && PyArray_DIMS(%(z)s)[2] == out_dim[2]
               && PyArray_DIMS(%(z)s)[3] == out_dim[3])) {
            if (%(z)s) Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject *)PyArray_ZEROS(PyArray_NDIM(%(input_1)s),
                                                   out_dim,
                                                   PyArray_TYPE(%(input_1)s),
                                                   0);
            if(NULL == %(z)s) {
               %(fail)s
            }
        }

        if(NULL == z_internal_buffer) {
            CHECK_ERR( dnnAllocateBuffer_%(precision)s(&z_internal_buffer,
                            z_internal_layout), err);
        }

        concat_res[dnnResourceDst] = z_internal_buffer;
        CHECK_ERR( dnnExecute_%(precision)s(pConcat, concat_res), err );

        ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = z_internal_layout;
        ((void**)PyArray_DATA(%(z)s))[1] = z_internal_buffer;

        """ % sub
        return ccode


class ConcatenateGrad(basic_ops.MKLOp):
    __props__ = ()

    def __init__(self):
        super(ConcatenateGrad, self).__init__()

    def c_code_cache_version(self):
        return (1, 0)

    def c_headers(self):
        return super(ConcatenateGrad, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(ConcatenateGrad, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        ccode = mkl_helper.header_text()
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
        ccode = """
        dnnError_t err;
        dnnPrimitive_t pSplit;
        void **gx_internal_buffer;
        void *split_res[dnnResourceNumber];
        """
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
        pSplit = NULL;
        gx_internal_buffer = NULL;
        """
        return ccode

    def c_cleanup_code_struct(self, node, name):
        len_of_outputs = len(node.outputs)
        sub = {}
        sub['len_of_outputs'] = len_of_outputs
        if 'float32' == node.inputs[-1].type.dtype:
            sub['type'] = 'float'
            sub['precision'] = 'F32'
        elif 'float64' == node.inputs[-1].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = 'F64'
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[-1].type.dtype)

        return """
        if(NULL != pSplit) {
            CHECK_ERR( dnnDelete_%(precision)s(pSplit), err );
            pSplit = NULL;
        }

        if(NULL != gx_internal_buffer) {
            for(int i = 0; i < %(len_of_outputs)s; ++i) {
                CHECK_ERR( dnnReleaseBuffer_%(precision)s(gx_internal_buffer[i]), err );
            }
            free(gx_internal_buffer);
            gx_internal_buffer = NULL;
        }
        """ % sub

    def infer_shape(self, node, shapes):
        return list(shapes[2:])

    def make_node(self, gz, axis, *tensors):
        gz = as_tensor_variable(gz)
        axis = as_tensor_variable(axis)
        as_tensor_variable_args = [as_tensor_variable(x) for x in tensors]

        if axis.type not in int_types:
            raise TypeError('axis must have type lscalar', axis.type)

        inputs = [gz, axis] + as_tensor_variable_args
        outputs = [gz.type() for i in xrange(len(tensors))]

        return Apply(self, inputs, outputs)

    def c_code(self, node, name, inputs, out, sub):
        gz, axis, tensors, = inputs[0], inputs[1], inputs[2:]
        gx = out

        num_of_gx = len(tensors)
        adtype = node.inputs[1].type.dtype_specs()[1]

        if 'float32' == node.inputs[0].type.dtype:
            sub['type'] = 'float'
            sub['precision'] = 'F32'
        elif "float64" == node.inputs[0].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = 'F64'
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[1].type.dtype)

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        int axis = ((%(adtype)s *)PyArray_DATA(%(axis)s))[0];

        if (NULL == gx_internal_buffer) {
            gx_internal_buffer = (void**)malloc(%(num_of_gx)s * sizeof(void*));
            for(int i = 0; i < %(num_of_gx)s; ++i)
                gx_internal_buffer[i] = NULL;
        }

        if (NULL == pSplit) {
             size_t dstChannelSize[%(num_of_gx)s] = {0};
        """ % sub

        for i, inp in enumerate(tensors):
            d = {}
            d['index'] = i
            d['inp'] = inp
            ccode += """
            dstChannelSize[%(index)s] = (size_t)PyArray_DIMS(%(inp)s)[axis];
            """ % d

        ccode += """
        CHECK_ERR( dnnSplitCreate_%(precision)s(&pSplit, NULL, %(num_of_gx)s,
                               ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0],
                               dstChannelSize), err );
        }
        """ % sub

        for i, inp in enumerate(tensors):
            d = {}
            d['i'] = i
            d['inp'] = inp
            d['gx_i'] = gx[i]
            d['fail'] = sub['fail']
            d['precision'] = sub['precision']
            ccode += """
            if ( !(%(gx_i)s
                   && PyArray_NDIM(%(gx_i)s) == PyArray_NDIM(%(inp)s)
                   && PyArray_DIMS(%(gx_i)s)[0] == PyArray_DIMS(%(inp)s)[0]
                   && PyArray_DIMS(%(gx_i)s)[1] == PyArray_DIMS(%(inp)s)[1]
                   && PyArray_DIMS(%(gx_i)s)[2] == PyArray_DIMS(%(inp)s)[2]
                   && PyArray_DIMS(%(gx_i)s)[3] == PyArray_DIMS(%(inp)s)[3])) {
                if (%(gx_i)s) Py_XDECREF(%(gx_i)s);

                %(gx_i)s = (PyArrayObject *)PyArray_ZEROS(PyArray_NDIM(%(inp)s),
                                                          PyArray_DIMS(%(inp)s),
                                                          PyArray_TYPE(%(inp)s),
                                                          0);
                if(NULL == %(gx_i)s) {
                    %(fail)s
                }

                dnnLayout_t gx_internal_layout_%(i)s = ((dnnLayout_t*)PyArray_DATA(%(inp)s))[0];
                ((dnnLayout_t*)PyArray_DATA(%(gx_i)s))[0] = gx_internal_layout_%(i)s;

                if(NULL == gx_internal_buffer[%(i)s]) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s(&gx_internal_buffer[%(i)s],
                                                               gx_internal_layout_%(i)s), err );
                }

                split_res[dnnResourceMultipleDst + %(i)s] = gx_internal_buffer[%(i)s];
                ((void**)PyArray_DATA(%(gx_i)s))[1] = gx_internal_buffer[%(i)s];
            }
            """ % d

        ccode += """
        split_res[dnnResourceSrc] = (((void**)PyArray_DATA(%(gz)s))[1]);
        CHECK_ERR( dnnExecute_%(precision)s(pSplit, split_res), err );
        """ % sub
        return ccode
