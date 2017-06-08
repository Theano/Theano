import theano.tensor as T
from theano.gof import Apply, Op
from theano.tensor.blas import ldflags
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.contrib.mkl.mkl_helper import header_text
from theano.contrib.mkl.mkl_type import MKLNdarrayType


class MKLOp(Op):
    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(MKLOp, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        ccode = header_text()
        ccode += """
        #define DIMENSION  4

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


class BaseConvertOp(MKLOp):
    def c_support_code_struct(self, node, name):
        ccode = """
        dnnError_t err;
        int first_run;
        void* internal_buf;
        void* user_buf;
        dnnLayout_t layout_internal;
        dnnLayout_t layout_user;
        dnnPrimitive_t to_internal;
        dnnPrimitive_t from_internal;
        dnnPrimitive_t primitive;
        void *convert_resources[dnnResourceNumber];
        size_t bottomSize[DIMENSION];
        size_t bottomStride[DIMENSION];
        """
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
        first_run = 1;
        internal_buf = NULL;
        user_buf = NULL;
        layout_internal = NULL;
        layout_user = NULL;
        to_internal = NULL;
        from_internal = NULL;
        primitive = NULL;
        """
        return ccode

    def c_code_cache_version(self):
        return (1, 0)


class I2U(BaseConvertOp):
    __props__ = ()

    def make_node(self, x):
        assert isinstance(x.type, MKLNdarrayType)
        return Apply(self, [x], [T.TensorType(broadcastable=x.broadcastable, dtype=x.dtype)()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [I2UGrad()(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        fail = sub['fail']

        ccode = """
            if (%(z)s) {
                Py_XDECREF(%(z)s);
            }
            %(z)s = (PyArrayObject*)MKLNdarray_CreateArrayObj(%(x)s);
            if (!%(z)s) {
                %(fail)s;
            }
        """ % locals()

        return ccode


class U2IGrad(BaseConvertOp):
    __props__ = ()

    def make_node(self, x, gz):
        out = x.type()
        return Apply(self, [x, gz], [out])

    def c_code(self, node, name, inp, out, sub):
        x, gz, = inp
        z, = out
        sub['x'] = x
        sub['gz'] = gz
        sub['z'] = z
        sub['name'] = U2IGrad.__name__
        if 'float32' == node.inputs[0].type.dtype:
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
        elif "float64" == node.inputs[0].type.dtype:
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
        else:
            raise TypeError("Type %s not implemented" %
                            node.inputs[0].type.dtype)
        ccode = """
        int status = 0;
        size_t z_size[4] = {0};
        size_t z_stride[4] = {0};
        int ndim = 0;
        npy_intp dims[4] = {0};
        if(NULL == %(z)s) {
            dims[0] = MKLNdarray_DIMS(%(gz)s)[0];
            dims[1] = MKLNdarray_DIMS(%(gz)s)[1];
            dims[2] = MKLNdarray_DIMS(%(gz)s)[2];
            dims[3] = MKLNdarray_DIMS(%(gz)s)[3];

            %(z)s = (PyArrayObject*)PyArray_ZEROS(MKLNdarray_NDIM(%(gz)s),
                                                  dims,
                                                  MKLNdarray_TYPE(%(gz)s),
                                                  0);
            if(NULL == %(z)s) {
                %(fail)s;
            }
        }

        ndim = (int)MKLNdarray_NDIM(%(gz)s);
        assert (ndim == DIMENSION);

        for(int i=0; i<DIMENSION; i++) {
            z_size[i] = (size_t)PyArray_DIMS(%(z)s)[ndim-i-1];
            z_stride[i] = (size_t)PyArray_STRIDES(%(z)s)[ndim-i-1] / %(x_item_size)s;
        }

        //create usr layerout
        if (NULL == layout_user) {
            CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user,
                                                     ndim, z_size,
                                                     z_stride), err);
        }

        if (! dnnLayoutCompare_%(precision)s(MKLNdarray_LAYOUT(%(gz)s),
                                             layout_user)) {
            if (NULL == from_internal) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&from_internal,
                                                             MKLNdarray_LAYOUT(%(gz)s),
                                                             layout_user), err);
            }

            if (from_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(from_internal,
                                                              MKLNdarray_DATA(%(gz)s),
                                                              PyArray_DATA(%(z)s)), err);
            }
        } else {
            memcpy ((void*)PyArray_DATA(%(z)s), MKLNdarray_DATA(%(gz)s), PyArray_SIZE(%(z)s)*PyArray_ITEMSIZE(%(z)s));
        }

        """ % sub
        return ccode


class I2UGrad(BaseConvertOp):
    __props__ = ()

    def make_node(self, x, gz):
        out = x.type()
        return Apply(self, [x, gz], [out])

    def c_code(self, node, name, inp, out, sub):
        x, gz, = inp
        z, = out
        sub['x'] = x
        sub['z'] = z
        sub['gz'] = gz

        if 'float32' == node.inputs[0].type.dtype:
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
            sub['type'] = "float"
        elif "float64" == node.inputs[0].type.dtype:
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
            sub["type"] = "double"
        else:
            raise TypeError("Type %s not implemented" %
                            node.inputs[0].type.dtype)

        ccode = """
        int status = 0;
        int gz_ndim = 0;
        size_t gz_size[4] = {0};
        size_t gz_stride[4] = {0};

        gz_size[0] = PyArray_DIMS(%(gz)s)[3];  //w
        gz_size[1] = PyArray_DIMS(%(gz)s)[2];  //h
        gz_size[2] = PyArray_DIMS(%(gz)s)[1];  //c
        gz_size[3] = PyArray_DIMS(%(gz)s)[0];  //n
        gz_stride[0] = 1;
        gz_stride[1] = gz_size[0];
        gz_stride[2] = gz_size[0] * gz_size[1];
        gz_stride[3] = gz_size[0] * gz_size[1] * gz_size[2];

        if (! (%(z)s
            && MKLNdarray_Check((PyObject*)%(z)s)
            && MKLNdarray_NDIM(%(z)s) == MKLNdarray_NDIM(%(x)s)
            && MKLNdarray_DIMS(%(z)s)[0] == MKLNdarray_DIMS(%(x)s)[0]
            && MKLNdarray_DIMS(%(z)s)[1] == MKLNdarray_DIMS(%(x)s)[1]
            && MKLNdarray_DIMS(%(z)s)[2] == MKLNdarray_DIMS(%(x)s)[2]
            && MKLNdarray_DIMS(%(z)s)[3] == MKLNdarray_DIMS(%(x)s)[3] )) {

            if (%(z)s) Py_XDECREF(%(z)s);

            %(z)s = (MKLNdarray*)MKLNdarray_New(MKLNdarray_NDIM(%(x)s), MKLNdarray_TYPE(%(x)s));
            if (NULL == %(z)s) {
                %(fail)s;
            }

            status = MKLNdarray_set_structure(%(z)s, MKLNdarray_NDIM(%(x)s), MKLNdarray_DIMS(%(x)s));
            if (0 != status) {
                %(fail)s;
            }

            status = MKLNdarray_copy_layout(%(z)s, %(x)s, MNDA_DATA);
            if (0 != status) {
                %(fail)s;
            }

            status = MKLNdarray_create_buffer_from_layout(%(z)s, MNDA_DATA);
            if (0 != status) {
                %(fail)s;
            }
        }

        //create usr layerout of gz
        if (NULL == layout_user) {
            CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user,
                                                     DIMENSION, gz_size,
                                                     gz_stride), err);
        }

        if (! dnnLayoutCompare_%(precision)s(MKLNdarray_LAYOUT(%(z)s),
                                             layout_user)) {
            if (NULL == to_internal) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal,
                                                            layout_user,
                                                            MKLNdarray_LAYOUT(%(z)s)), err);
            }
            if (to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal,
                                                              PyArray_DATA(%(gz)s),
                                                              MKLNdarray_DATA(%(z)s)), err);
            }
        } else {
        size_t nn = dnnLayoutGetMemorySize_%(precision)s(MKLNdarray_LAYOUT(%(z)s));
        memcpy((void*)MKLNdarray_DATA(%(z)s),
               (void*)PyArray_DATA(%(gz)s),
               nn);
        }
        """ % sub
        return ccode


class U2IConv(BaseConvertOp):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_dilation=(1, 1)):
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        self.imshp = imshp
        self.kshp = kshp
        self.filter_dilation = filter_dilation

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        if x.type.ndim is not 4:
            raise TypeError('U2IConv: input x should be an 4-dim tensor')
        return Apply(self, [x], [MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [U2IGrad()(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        dH, dW = self.subsample

        if len(self.kshp) == 5:
            grp, k_n, k_c, k_h, k_w = self.kshp
            gkshp = [grp * k_n, grp * k_c, k_h, k_w]
        else:
            k_n, k_c, k_h, k_w = self.kshp
            grp = 1
            gkshp = self.kshp

        if None in self.imshp:
            i_n, i_c, i_h, i_w = 0, 0, 0, 0
            o_n, o_c, o_h, o_w = 0, 0, 0, 0
        else:
            i_n, i_c, i_h, i_w = self.imshp
            o_n, o_c, o_h, o_w = get_conv_output_shape(image_shape=self.imshp,
                                                       kernel_shape=self.kshp,
                                                       border_mode=self.border_mode,
                                                       filter_dilation=self.filter_dilation,
                                                       subsample=self.subsample)

        if self.border_mode == 'valid':
            padH, padW = (0, 0)
        elif self.border_mode == 'full':
            padH, padW = ((k_h - 1), (k_w - 1))
        elif self.border_mode == 'half':
            padH, padW = ((k_h / 2), (k_w / 2))
        elif isinstance(self.border_mode, tuple):
            padH, padW = self.border_mode
        else:
            raise ValueError("border_mode must have two elements")

        z, = out

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise TypeError("Type %s is not supported!" %
                            node.inputs[0].type.dtype)
        fail = sub['fail']

        ccode = """
            npy_intp* x_dims = PyArray_DIMS(%(x)s);
            int ndim = PyArray_NDIM(%(x)s);
            int dtype = PyArray_TYPE(%(x)s);

            if (1 == first_run) {
                int convPadding[2];
                size_t convStride[2], weightSize[5], weightStride[5], imageSize[4], imageStride[4], zSize[4], zStride[4];
                convStride[0] = %(dW)s;
                convStride[1] = %(dH)s;
                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                imageSize[0] = %(i_w)s;  //w
                imageSize[1] = %(i_h)s;  //h
                imageSize[2] = %(i_c)s;  //c
                imageSize[3] = %(i_n)s;  //n

                if (0 == imageSize[0] || 0 == imageSize[1] || 0 == imageSize[2] || 0 == imageSize[3]) {
                    imageSize[0] = x_dims[3];
                    imageSize[1] = x_dims[2];
                    imageSize[2] = x_dims[1];
                    imageSize[3] = x_dims[0];
                }

                imageStride[0] = 1;
                imageStride[1] = imageSize[0];
                imageStride[2] = imageSize[0] * imageSize[1];
                imageStride[3] = imageSize[0] * imageSize[1] * imageSize[2];

                weightSize[0] = %(k_w)s;
                weightSize[1] = %(k_h)s;
                weightSize[2] = %(k_c)s;
                weightSize[3] = %(k_n)s;
                weightSize[4] = %(grp)s;
                weightStride[0] = 1;
                weightStride[1] = weightSize[0];
                weightStride[2] = weightSize[0] * weightSize[1];
                weightStride[3] = weightSize[0] * weightSize[1] * weightSize[2];
                weightStride[4] = weightSize[0] * weightSize[1] * weightSize[2] * weightSize[3];

                zSize[0] = %(o_w)s;
                zSize[1] = %(o_h)s;
                zSize[2] = %(o_c)s;
                zSize[3] = %(o_n)s;

                if (0 == zSize[0] || 0 == zSize[1] || 0 == zSize[2] || 0== zSize[3]) {
                    zSize[0] = (imageSize[0] - 2 * convPadding[0] - weightSize[0]) / convStride[0] + 1;
                    zSize[1] = (imageSize[1] - 2 * convPadding[1] - weightSize[1]) / convStride[1] + 1;
                    zSize[2] = weightSize[3];
                    zSize[3] = imageSize[3];
                }

                zStride[0] = 1;
                zStride[1] = zSize[0];
                zStride[2] = zSize[0] * zSize[1];
                zStride[3] = zSize[0] * zSize[1] * zSize[2];

                const int group = %(grp)s;
                // create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, DIMENSION, imageSize, imageStride), err );
                // create convolution primitive
                CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&primitive, NULL,
                           dnnAlgorithmConvolutionDirect, group, DIMENSION, imageSize, zSize,
                           weightSize, convStride, convPadding, dnnBorderZeros), err );
            }

            assert (ndim == DIMENSION);
            size_t dims[DIMENSION] = {0};
            for (int i = 0; i < ndim; i++) {
                dims[i] = (size_t)x_dims[i];
            }

            if (!( %(z)s
                && MKLNdarray_Check((PyObject*)%(z)s)
                && MKLNdarray_NDIM(%(z)s) == ndim
                && MKLNdarray_DIMS(%(z)s)[0] == x_dims[0]
                && MKLNdarray_DIMS(%(z)s)[1] == x_dims[1]
                && MKLNdarray_DIMS(%(z)s)[2] == x_dims[2]
                && MKLNdarray_DIMS(%(z)s)[3] == x_dims[3]) ) {

                if (%(z)s) Py_XDECREF(%(z)s);
                %(z)s = (MKLNdarray*)MKLNdarray_New(ndim, dtype);
                if (!%(z)s) {
                    %(fail)s;
                }

                int status = MKLNdarray_set_structure(%(z)s, ndim, dims);
                if (status != 0) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &primitive, dnnResourceSrc);
                if (status != 0) {
                    %(fail)s;
                }
            }

            if (!dnnLayoutCompare_%(precision)s(layout_user, MKLNdarray_LAYOUT(%(z)s))) {
                if (NULL == to_internal) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal,
                                                                 layout_user,
                                                                 MKLNdarray_LAYOUT(%(z)s)), err );
                }
            }

            if (to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal,
                                                              PyArray_DATA(%(x)s),
                                                              MKLNdarray_DATA(%(z)s)), err );
            } else {
                memcpy(MKLNdarray_DATA(%(z)s), (void*)PyArray_DATA(%(x)s), %(z)s->data_size);
            }

            first_run = 0;
        """ % locals()
        return ccode
