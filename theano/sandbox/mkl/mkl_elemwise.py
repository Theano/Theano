from theano.gof import Apply
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor.basic import Join
from theano.sandbox.mkl import mkl_helper, basic_ops


class ElemwiseSum(basic_ops.MKLOp, Join):
    """
    ElemwiseSum is used to add inputs with MKL layout.

    inp_num: number of inputs
    coeff: coefficients for all inputs
    """
    __props__ = ('inp_num', 'coeff')

    def __init__(self, inp_num=1, coeff=(1.0, )):
        super(ElemwiseSum, self).__init__()
        self.inp_num = inp_num
        if isinstance(coeff, tuple):
            self.coeff = coeff
        elif isinstance(coeff, list):
            self.coeff = tuple(coeff)
        else:
            raise TypeError('Coeff should be a tuple or list.')
        if self.inp_num != len(self.coeff):
            raise ValueError('Number of ElemwiseSum inputs is not equal to \
                             number of coefficients.')

    def make_node(self, *tensors):
        # Neet to check ndim and shape of all input tensors!
        for x in tensors:
            assert x.type.ndim == 4

        node = Join.make_node(self, 1, *tensors)

        def agv(v):
            return as_tensor_variable(v)
        return Apply(self, list(map(agv, tensors)),
                     [TensorType(dtype=node.outputs[0].dtype,
                      broadcastable=node.outputs[0].broadcastable)()])

    def infer_shape(self, node, shapes):
        return list(shapes[-1:])

    def grad(self, inp, grads):
        gz, = grads
        return ElemwiseSumGrad(inp_num=self.inp_num, coeff=self.coeff)(gz, inp)

    def c_code_cache_version(self):
        return (1, 0, 1)

    def c_support_code(self):
        final_code = mkl_helper.header_text()
        final_code += """
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
        return final_code

    def c_support_code_struct(self, node, name):
        support_code = """
        dnnPrimitive_t pSum;
        void* buf_output;
        void** pbuf_inputs;
        dnnLayout_t layout_output;
        void *elemwise_res[dnnResourceNumber];
        dnnLayout_t* layout_internal;
        dnnPrimitive_t* convert_int2int_bottom;
        """
        return support_code

    def c_init_code_struct(self, node, name, sub):
        init_code = """
        pSum = NULL;
        buf_output = NULL;
        pbuf_inputs = NULL;
        layout_output = NULL;
        layout_internal = NULL;
        convert_int2int_bottom = NULL;
        """
        return init_code

    def c_cleanup_code_struct(self, node, name):
        sub = {}
        sub['len'] = len(node.inputs)
        if 'float32' == node.inputs[1].type.dtype:
            sub['type'] = "float"
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
        elif "float64" == node.inputs[1].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
        else:
            raise Exception("Type %s not implemented"
                            % node.inputs[1].type.dtype)

        return """
        int status = 0;
        if (NULL != pSum) {
            status = dnnDelete_%(precision)s(pSum);
            if (0 != status) {
                printf(\"ERROR: Free pSum\\n\");
                exit(1);
            }
            pSum = NULL;
        }

        if (NULL != buf_output) {
            status = dnnReleaseBuffer_%(precision)s(buf_output);
            if (0 != status) {
                printf(\"ERROR: Free buffer in ElemwiseSum\\n\");
                exit(1);
            }
            buf_output = NULL;
        }

        if (NULL != layout_output) {
            status = dnnLayoutDelete_%(precision)s(layout_output);
            if (0 != status) {
                printf(\"ERROR: Free layout_output in ElemwiseSum\\n\");
                exit(1);
            }
            layout_output = NULL;
        }

        if (NULL != layout_internal) {
            for (int i = 0; i < %(len)s; i++) {
                if (NULL != layout_internal[i]) {
                    status = dnnLayoutDelete_%(precision)s(layout_internal[i]);
                    if (0 != status) {
                        printf(\"ERROR: Free layout_internal[%%d] in ElemwiseSum, %%d\\n\", i, status);
                        exit(1);
                    }
                    layout_internal[i] = NULL;
                }
            }
            free(layout_internal);
            layout_internal = NULL;
        }

        if (NULL != pbuf_inputs) {
            for (int i = 0; i < %(len)s; i++) {
                if (NULL != pbuf_inputs[i]) {
                    status = dnnReleaseBuffer_%(precision)s(pbuf_inputs[i]);
                    if (0 != status) {
                        printf(\"ERROR: Free pbuf_inputs in ElemwiseSum\\n\");
                        exit(1);
                    }
                    pbuf_inputs[i] = NULL;
                }
            }
            free(pbuf_inputs);
            pbuf_inputs = NULL;
        }

        if (NULL != convert_int2int_bottom) {
            for (int i = 0; i < %(len)s; i++) {
                if (NULL != convert_int2int_bottom[i]) {
                    status = dnnDelete_%(precision)s(convert_int2int_bottom[i]);
                    if (0 != status) {
                        printf(\"ERROR: Free convert_int2int_bottom[%%d]\\n\", i);
                        exit(1);
                    }
                    convert_int2int_bottom[i] = NULL;
                }
            }
            free(convert_int2int_bottom);
            convert_int2int_bottom = NULL;
        }
        """ % sub

    def c_code(self, node, name, inp, out, sub):
        tensors = inp
        z, = out
        sub['z'] = z
        sub['len'] = self.inp_num
        if 'float32' == node.inputs[1].type.dtype:
            sub['type'] = "float"
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
        elif "float64" == node.inputs[1].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
        else:
            raise Exception("Type %s not implemented" % node.inputs[1].type.dtype)
        sub['x'] = tensors[0]
        coeff = self.coeff

        ccode = """
            %(type)s coeffs[%(len)s] = {1.0};
            """ % sub

        for i, co in enumerate(coeff):
            ccode += """
            coeffs[%s] = %s;
            """ % (i, co)

        ccode += """
            int size = 0;
            int status = 0;
            if (NULL == pSum) {
                dnnLayout_t x_int = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];
                status = dnnSumCreate_%(precision)s(&pSum, NULL, %(len)s, x_int, coeffs);
                if (0 != status) {
                    printf(\"ERROR: Create %(len)s primitive for ElemwiseSum\\n\");
                    exit(1);
                }
            }

            if (NULL == convert_int2int_bottom) {
                convert_int2int_bottom = (dnnPrimitive_t*)malloc(%(len)s * sizeof (dnnPrimitive_t));
                for (int i = 0; i < %(len)s; i++)
                    convert_int2int_bottom[i] = NULL;
            }
            if (NULL == layout_internal) {
                layout_internal = (dnnLayout_t*)malloc(%(len)s * sizeof (dnnLayout_t));
                for (int i =  0; i < %(len)s; i++)
                    layout_internal[i] = NULL;
            }
            if (NULL == pbuf_inputs) {
                pbuf_inputs = (void**)malloc(%(len)s * sizeof (void*));
                for (int i = 0; i < %(len)s; i++)
                    pbuf_inputs[i] = NULL;
            }
            """ % sub

        for i, inp in enumerate(tensors):
            d = {}
            d['i'] = i
            d['inp'] = inp
            d['precision'] = sub['precision']
            ccode += """
            if (NULL == layout_internal[%(i)s]) {
                status = dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal[%(i)s], pSum,
                                        (dnnResourceType_t)(dnnResourceMultipleSrc + %(i)s));
                if (0 != status) {
                    printf(\"ERROR: Create layout %(i)s x in ElemwiseSum\\n\");
                    exit(1);
                }
                dnnLayout_t layout_x = ((dnnLayout_t*)PyArray_DATA(%(inp)s))[0];

                //Create I2I primitive
                if (!dnnLayoutCompare_%(precision)s(layout_x, layout_internal[%(i)s])) {
                    if (NULL == convert_int2int_bottom[%(i)s]) {
                        status = dnnConversionCreate_%(precision)s(
                                            &convert_int2int_bottom[%(i)s],
                                            layout_x, layout_internal[%(i)s]);
                        if (0 != status) {
                            printf(\"ERROR: Create I2I in ElemwiseSum\\n\");
                            exit(1);
                        }
                    }
                    // Alloc memory for new x layout
                    if (NULL == pbuf_inputs[%(i)s]) {
                        status = dnnAllocateBuffer_%(precision)s(
                                                    (void**)(&pbuf_inputs[%(i)s]),
                                                    layout_internal[%(i)s]);
                        if (0 != status) {
                            printf(\"ERROR: Create internal buffer in ElemwiseSum\\n\");
                            exit(1);
                        }
                    }
                }
            }

            if (NULL != convert_int2int_bottom[%(i)s]) {
                void* prev_buf = ((void**)PyArray_DATA(%(inp)s))[1];
                status = dnnConversionExecute_%(precision)s(convert_int2int_bottom[%(i)s],
                                                            prev_buf,
                                                            pbuf_inputs[%(i)s]);
                if (0 != status) {
                    printf(\"ERROR: Execute I2I in ElemwiseSum\\n\");
                    exit(1);
                }
                elemwise_res[dnnResourceMultipleSrc + %(i)s] = (void*)(pbuf_inputs[%(i)s]);
            } else {
                elemwise_res[dnnResourceMultipleSrc + %(i)s] = (void*)(((void**)PyArray_DATA(%(inp)s))[1]);
            }
            """ % d

        ccode += """
            if (NULL == %(z)s) {
                //create PyArrayObject
                %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),
                                                      PyArray_DIMS(%(x)s),
                                                      PyArray_TYPE(%(x)s),
                                                      0);
                if (NULL == %(z)s) {
                    %(fail)s
                }
            }

            if (NULL == layout_output) {
                status = dnnLayoutCreateFromPrimitive_%(precision)s(&layout_output, pSum, dnnResourceDst);
                if(0 != status) {
                    printf(\"ERROR: Create output layout in Elemwise\\n\");
                    exit(1);
                }
            }

            if (NULL == buf_output) {
                status = dnnAllocateBuffer_%(precision)s((void **)(&buf_output), layout_output);
            }

            size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_output);
            if (size != PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0]) {
                exit(1);
            }

            elemwise_res[dnnResourceDst] = buf_output;
            status = dnnExecute_%(precision)s(pSum, elemwise_res);
            if (0 != status) {
                printf(\"ERROR: ElemwiseSum Execute\\n\");
                exit(1);
            }

            ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_output;
            ((void**)PyArray_DATA(%(z)s))[1] = buf_output;
            """ % sub
        return ccode


class ElemwiseSumGrad(basic_ops.MKLOp):
    """
    ElemwiseSumGrad is used to compute the gradients for ElemwiseSum OP.

    inp_num: number of inputs for ElemwiseSum
    coeff: Coefficients of all inputs
    """
    __props__ = ('inp_num', 'coeff')

    def __init__(self, inp_num=1, coeff=(1.0, )):
        super(ElemwiseSumGrad, self).__init__()
        self.inp_num = inp_num
        if isinstance(coeff, tuple):
            self.coeff = coeff
        elif isinstance(coeff, list):
            self.coeff = tuple(coeff)
        else:
            raise TypeError('Coeff should be a tuple or list.')
        if self.inp_num != len(self.coeff):
            raise ValueError('Number of ElemwiseSum inputs is not equal to number of coefficients.')

    def make_node(self, gz, *tensors):
        gz = as_tensor_variable(gz)

        def agv(v):
            return as_tensor_variable(v)

        def ago(v):
            return as_tensor_variable(v).type()
        return Apply(self, [gz] + list(map(agv, *tensors)), list(map(ago, *tensors)))

    def c_code_cache_version(self):
        return (1, 0, 1)

    def c_support_code(self):
        final_code = mkl_helper.header_text()
        final_code += """
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
        return final_code

    def c_support_code_struct(self, node, name):
        support_code = """
        void** internal_ptr;
        dnnPrimitive_t* convert_int2int_top;
        """
        return support_code

    def c_init_code_struct(self, node, name, sub):
        init_code = """
        internal_ptr = NULL;
        convert_int2int_top = NULL;
        """
        return init_code

    def c_cleanup_code_struct(self, node, name):
        d = {}
        d['len'] = self.inp_num
        if 'float32' == node.inputs[-1].type.dtype:
            d['type'] = 'float'
            d['precision'] = 'F32'
            d['x_item_size'] = 4
        elif 'float64' == node.inputs[-1].type.dtype:
            d['type'] = 'double'
            d['precision'] = 'F64'
            d['x_item_size'] = 8
        else:
            raise Exception('Type %s not implemented' % node.inputs[-1].type.dtype)

        return """
            int status = 0;
            if (NULL != internal_ptr) {
                for (int i = 0; i < %(len)s; i++) {
                    if (NULL != internal_ptr[i]) {
                        status = dnnReleaseBuffer_%(precision)s(internal_ptr[i]);
                        if (0 != status) {
                            printf (\"ERROR: Free buffer in ElemwiseSumGrad\\n\");
                            exit(1);
                        }
                    }
                    internal_ptr[i] = NULL;
                }
                free(internal_ptr);
                internal_ptr = NULL;
            }

            if (NULL != convert_int2int_top) {
                for (int i = 0; i < %(len)s; i++) {
                    if (NULL != convert_int2int_top[i]) {
                        status = dnnDelete_%(precision)s(convert_int2int_top[i]);
                        if (0 != status) {
                            printf (\"ERROR: Free convert_int2int_top[%%d]\\n\", i);
                            exit(1);
                        }
                    }
                }
                free(conver_int2int_top);
                convert_int2int_top = NULL;
            }
            """ % d

    def infer_shape(self, node, shapes):
        return list(shapes[1:])

    def c_code(self, node, name, inp, out, sub):
        gz, tensors, = inp[0], inp[1:]
        outputs = out
        sub['gz'] = gz
        sub['len'] = self.inp_num
        sub['x'] = tensors[0]
        if 'float32' == node.inputs[1].type.dtype:
            sub['type'] = 'float'
            sub['precision'] = 'F32'
            sub['x_item_size'] = 4
        elif 'float64' == node.inputs[1].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = 'F64'
            sub['x_item_size'] = 8
        else:
            raise Exception('Type %s not implemented' % node.inputs[1].type.dtype)

        ccode = """
            int status = 0;
            int size = 0;
            if (NULL == internal_ptr) {
                internal_ptr = (void**)malloc(%(L)s * sizeof (void*));
                for (int i = 0; i < %(len)s; i++) {
                    internal_ptr[i] = NULL;
                }
            }

            if (NULL != convert_int2int_top) {
                convert_int2int_top = (dnnPrimitive_t)*malloc(%(L)s * sizeof (dnnPrimitive_t));
                for (int i = 0; i < %(len)s; i++) {
                    convert_int2int_top[i] = NULL;
                }
            }

            void* buf_gz = ((void**)PyArray_DATA(%(gz)s))[1];
            dnnLayout_t layout_gz = ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0];
            """ % sub

        for i, x in enumerate(tensors):
            d = {}
            d['i'] = i
            d['x'] = x
            d['z'] = outputs[i]
            d['fail'] = sub['fail']
            d['precision'] = sub['precision']

            ccode += """
                if (NULL == %(z)s) {
                    %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),
                                                          PyArray_DIMS(%(x)s),
                                                          PyArray_TYPE(%(x)s),
                                                          0);
                    if (NULL == %(z)s) {
                        %(fail)s
                    }
                    dnnLayout_t layout_x = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];

                    if (NULL == internal_ptr[%(i)s]) {
                        status = dnnAllocateBuffer_%(precision)s(
                                        (void **)(&internal_ptr[%(i)s]),
                                        layout_x);
                    }

                    size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_x);
                    if (size != PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0]) {
                        printf(\"ERROR: Internal buffer Size: %%d != usr: %%d\\n\", size,
                                PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0]);
                        exit(1);
                    }

                    memset (internal_ptr[%(i)s], 0, size);
                    if (!dnnLayoutCompare_%(precision)s(layout_x, layout_gz)) {
                        if (NULL == conver_int2int_top[%(i)s]) {
                            status = dnnConversionCreate_%(precison)s(&convert_int2int_top[%(i)s],
                                                                      layout_gz, layout_x);
                            if (0 != status) {
                                printf (\"ERROR: Create I2I in ElemwiseSumGrad\\n\");
                                exit(1);
                            }
                        }
                    }
                }

                ((void**)PyArray_DATA(%(z)s))[1] = internal_ptr[%(i)s];
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];

                if (NULL != convert_int2int_top[%(i)s]) {
                    status = dnnConversionExecute_%(precison)s(convert_int2int_top[%(i)s], buf_gz, internal_ptr[%(i)s]);
                    if (0 != status) {
                        printf (\"ERROR: I2I in ElemwiseSumGrad\\n\");
                        exit(1)
                    }
                } else {
                    ((void**)PyArray_DATA(%(z)s))[1] = buf_gz;
                }
                """ % d
        return ccode
