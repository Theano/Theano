from theano.gof import Apply
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor.basic import Join
from theano.tensor.blas import ldflags
from theano.sandbox.mkl import mkl_helper, basic_ops


class ElemwiseSum(basic_ops.MKLOp, Join):
    """
    ElemwiseSum for MKL
    """
    __props__ = ('inp_num', 'coeff')

    def __init__(self, inp_num=1, coeff=[1.0, ], uniq_id=0):
        super(ElemwiseSum, self).__init__()
        self.uniq_id = uniq_id
        self.inp_num = inp_num
        self.coeff = coeff
        assert isinstance(self.coeff, list)
        if self.inp_num != len(self.coeff):
            raise ValueError('Number of ElemwiseSum inputs is not equal to number of coefficients.')

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inp_num) ^ hash(sum(self.coeff))

    def make_node(self, *tensors):
        # Neet to check ndim and shape of all input tensors!
        for x in tensors:
            assert x.type.ndim == 4

        node = Join.make_node(self, 1, *tensors)

        def agv(v):
            return as_tensor_variable(v)
        return Apply(self, list(map(agv, tensors)),
                     [TensorType(dtype=node.outputs[0].dtype, broadcastable=node.outputs[0].broadcastable)()])

    def infer_shape(self, node, shapes):
        return list(shapes[-1:])

    def grad(self, inp, grads):
        gz, = grads
        return ElemwiseSumGrad(inp_num=self.inp_num, coeff=self.coeff, uniq_id=self.uniq_id)(gz, inp)

    def c_code_cache_version(self):
        return (1, 0, hash(self.uniq_id))

    def c_headers(self):
        return super(ElemwiseSum, self).c_headers()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(ElemwiseSum, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        final_code = mkl_helper.header_text()
        final_code += """
              // #define _DEBUG_
               static dnnPrimitive_t pSum = NULL;
               static void* internal_ptr = NULL;
               static void** internal_x_ptr = NULL;
               static dnnLayout_t out_layer = NULL;
               static void *eltwise_res[dnnResourceNumber];
               static dnnLayout_t* layerout_int = NULL;
               static dnnPrimitive_t*convert_int2int_bottom = NULL;
                    """
        return final_code

    def c_cleanup_code_struct(self, node, name):
        sub = {}
        sub['uid'] = self.uniq_id
        sub['L'] = len(node.inputs)
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

        if (NULL != internal_ptr) {
            status = dnnReleaseBuffer_%(precision)s(internal_ptr);
            if (0 != status) {
                printf(\"ERROR: Free buffer in ElemwiseSum\\n\");
                exit(1);
            }
            internal_ptr = NULL;
        }

        if (NULL != out_layer) {
            status = dnnLayoutDelete_%(precision)s(out_layer);
            if (0 != status) {
                printf(\"ERROR: Free out_layer in ElemwiseSum\\n\");
                exit(1);
            }
            out_layer = NULL;
        }

        if (NULL != layerout_int) {
            for (int i = 0; i < %(L)s; i++) {
                if (NULL != layerout_int[i]) {
                    status = dnnLayoutDelete_%(precision)s(layerout_int[i]);
                    if (0 != status) {
                        printf(\"ERROR: Free layerout_int[%%d] in ElemwiseSum, %%d\\n\", i, status);
                        exit(1);
                    }
                    layerout_int[i] = NULL;
                }
            }
            free(layerout_int);
            layerout_int = NULL;
        }

        if (NULL != internal_x_ptr) {
            for (int i = 0; i < %(L)s; i++) {
                if (NULL != internal_x_ptr[i]) {
                    status = dnnReleaseBuffer_%(precision)s(internal_x_ptr[i]);
                    if (0 != status) {
                        printf(\"ERROR: Free out_layer in ElemwiseSum\\n\");
                        exit(1);
                    }
                    internal_x_ptr[i] = NULL;
                }
            }
            free(internal_x_ptr);
            internal_x_ptr = NULL;
        }

        if (NULL != convert_int2int_bottom) {
            for (int i = 0; i < %(L)s; i++) {
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
        L = len(tensors)
        sub['z'] = z
        sub["L"] = L
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
        assert L == self.inp_num

        ccode = """
            %(type)s coeffs[%(L)s] = {1.0};
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
                status = dnnSumCreate_%(precision)s(&pSum, NULL, %(L)s, x_int, coeffs);
                if (0 != status) {
                    printf(\"ERROR: Create %(L)s primitive for ElemwiseSum\\n\");
                    exit(1);
                }
            }

            if (NULL == convert_int2int_bottom) {
                convert_int2int_bottom = (dnnPrimitive_t*)malloc(%(L)s * sizeof(dnnPrimitive_t));
                for (int i = 0; i < %(L)s; i++)
                    convert_int2int_bottom[i] = NULL;
            }
            if (NULL == layerout_int) {
                layerout_int = (dnnLayout_t*)malloc(%(L)s * sizeof(dnnLayout_t));
                for (int i =  0; i < %(L)s; i++)
                    layerout_int[i] = NULL;
            }
            if (NULL == internal_x_ptr) {
                internal_x_ptr = (void**)malloc(%(L)s * sizeof(void*));
                for (int i = 0; i < %(L)s; i++)
                    internal_x_ptr[i] = NULL;
            }
            """ % sub

        for i, inp in enumerate(tensors):
            d = {}
            d['i'] = i
            d['inp'] = inp
            d['precision'] = sub['precision']
            ccode += """
            if (NULL == layerout_int[%(i)s]) {
                status = dnnLayoutCreateFromPrimitive_%(precision)s(&layerout_int[%(i)s], pSum,
                                        (dnnResourceType_t)(dnnResourceMultipleSrc + %(i)s));
                if (0 != status) {
                    printf(\"ERROR: Create layerout %(i)s x in ElemwiseSum\\n\");
                    exit(1);
                }
                dnnLayout_t x_layout = ((dnnLayout_t*)PyArray_DATA(%(inp)s))[0];

                //Create I2I primitive
                if (!dnnLayoutCompare_%(precision)s(x_layout, layerout_int[%(i)s])) {
                    if (NULL == convert_int2int_bottom[%(i)s]) {
                        status = dnnConversionCreate_%(precision)s(
                                            &convert_int2int_bottom[%(i)s],
                                            x_layout,layerout_int[%(i)s]);
                        if (0 != status) {
                            printf(\"ERROR: Create I2I in ElemwiseSum\\n\");
                            exit(1);
                        }
                    }
                    // Alloc memory for new x layerout
                    if (NULL == internal_x_ptr[%(i)s]) {
                        status = dnnAllocateBuffer_%(precision)s(
                                                    (void**)(&internal_x_ptr[%(i)s]),
                                                    layerout_int[%(i)s]);
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
                                                            internal_x_ptr[%(i)s]);
                if (0 != status) {
                    printf(\"ERROR: Execute I2I in ElemwiseSum\\n\");
                    exit(1);
                }
                eltwise_res[dnnResourceMultipleSrc + %(i)s] = (void*)(internal_x_ptr[%(i)s]);
            } else {
                eltwise_res[dnnResourceMultipleSrc + %(i)s] = (void*)(((void**)PyArray_DATA(%(inp)s))[1]);
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

            if (NULL == out_layer) {
                status = dnnLayoutCreateFromPrimitive_%(precision)s(&out_layer, pSum, dnnResourceDst);
                if(0 != status) {
                    printf(\"ERROR: Create output layerout in Elemwise\\n\");
                    exit(1);
                }
            }

            if (NULL == internal_ptr) {
                status = dnnAllocateBuffer_%(precision)s((void **)(&internal_ptr), out_layer);
            }

            size = (int)dnnLayoutGetMemorySize_%(precision)s(out_layer);
            if (size != PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0]) {
                exit(1);
            }

            eltwise_res[dnnResourceDst] = internal_ptr;
            status = dnnExecute_%(precision)s(pSum, eltwise_res);
            if (0 != status) {
                printf(\"ERROR: ElemwiseSum Execute\\n\");
                exit(1);
            }

            ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = out_layer;
            ((void**)PyArray_DATA(%(z)s))[1] = internal_ptr;
            """ % sub
        return ccode


class ElemwiseSumGrad(basic_ops.MKLOp):
    """
    ElemwiseSumGrad for MKL
    """
    __props__ = ('inp_num', 'coeff')

    def __init__(self, inp_num=1, coeff=[1.0, ], uniq_id=0):
        self.uniq_id = uniq_id
        self.inp_num = inp_num
        self.coeff = coeff
        assert isinstance(coeff, list)
        if self.inp_num != len(self.coeff):
            raise ValueError('Number of ElemwiseSum inputs is not equal to number of coefficients.')

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inp_num) ^ hash(sum(self.coeff))

    def make_node(self, gz, *tensors):
        gz = as_tensor_variable(gz)

        def agv(v):
            return as_tensor_variable(v)

        def ago(v):
            return as_tensor_variable(v).type()
        return Apply(self, [gz] + list(map(agv, *tensors)), list(map(ago, *tensors)))

    def c_code_cache_version(self):
        return (1, 0, hash(self.uniq_id))

    def c_headers(self):
        return super(ElemwiseSumGrad, self).c_headers()

    def c_lib_dirs(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(ElemwiseSumGrad, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        final_code = mkl_helper.header_text()
        final_code += """
            static void** internal_ptr = NULL;
            static dnnPrimitive_t* convert_int2int_top = NULL;
            """
        return final_code

    def c_cleanup_code_struct(self, node, name):
        L = len(node.outputs)
        d = {}
        d['L'] = L
        d['uid'] = self.uniq_id
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
                for (int i = 0; i < %(L)s; i++) {
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
                for (int i = 0; i < %(L)s; i++) {
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
        L = len(tensors)
        sub['gz'] = gz
        sub['L'] = L
        sub['uniq_id'] = self.uniq_id
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
                for (int i = 0; i < %(L)s; i++) {
                    internal_ptr[i] = NULL;
                }
            }

            if (NULL != convert_int2int_top) {
                convert_int2int_top = (dnnPrimitive_t)*malloc(%(L)s * sizeof (dnnPrimitive_t));
                for (int i = 0; i < %(L)s; i++) {
                    convert_int2int_top[i] = NULL;
                }
            }

            void* gz_buf = ((void**)PyArray_DATA(%(gz)s))[1];
            dnnLayout_t gz_layerout = ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0];
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
                    dnnLayout_t layout_int = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];

                    if (NULL == internal_ptr[%(i)s]) {
                        status = dnnAllocateBuffer_%(precision)s(
                                        (void **)(&internal_ptr[%(i)s]),
                                        layout_int);
                    }

                    size = (int)dnnLayoutGetMemorySize_%(precision)s(layout_int);
                    if (size != PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0]) {
                        printf(\"ERROR: Internal buffer Size: %%d != usr: %%d\\n\", size,
                                PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0]);
                        exit(1);
                    }

                    memset(internal_ptr[%(i)s], 0, size);
                    if (!dnnLayoutCompare_%(precision)s(layout_int, gz_layerout)) {
                        if (NULL == conver_int2int_top[%(i)s]) {
                            status = dnnConversionCreate_%(precison)s(&convert_int2int_top[%(i)s],
                                                                      gz_layerout, layout_int);
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
                    status = dnnConversionExecute_%(precison)s(convert_int2int_top[%(i)s], gz_buf, internal_ptr[%(i)s]);
                    if (0 != status) {
                        printf (\"ERROR: I2I in ElemwiseSumGrad\\n\");
                        exit(1)
                    }
                } else {
                    ((void**)PyArray_DATA(%(z)s))[1] = gz_buf;
                }
                """ % d
        return ccode
