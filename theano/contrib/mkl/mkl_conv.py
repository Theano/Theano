from __future__ import absolute_import, print_function, division
import theano
from six import integer_types
from theano import Apply
from theano import gof
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.blas import ldflags
from theano.contrib.mkl import mkl_helper, mkl_type


class MKLConvBase(gof.Op):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample')

    def __init__(self, imshp=None, kshp=None, border_mode="valid", subsample=(1, 1)):
        if (not theano.config.blas.ldflags) or ('mkl' not in theano.config.blas.ldflags):
            raise NotImplementedError("MKL Convolution requires MKL library.")

        if isinstance(border_mode, int):
            if border_mode < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'non-negative integer'.format(border_mode))
            border_mode = (border_mode, border_mode)
        if isinstance(border_mode, tuple):
            if len(border_mode) != 2 or border_mode[0] < 0 or border_mode[1] < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'pair of non-negative integers'.format(border_mode))
            pad_h, pad_w = map(int, border_mode)
            border_mode = (pad_h, pad_w)
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(border_mode))
        self.border_mode = border_mode

        if len(subsample) != 2:
            raise ValueError("subsample must have two elements")
        self.subsample = tuple(subsample)
        self.imshp = imshp
        self.kshp = kshp

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(MKLConvBase, self).c_compile_args()
        return compile_args

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_headers(self):
        headers = ['<stdio.h>']
        headers += super(MKLConvBase, self).c_headers()
        return headers

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (1, 0, 1)

    def c_support_code(self):
        ccode = mkl_helper.header_text()
        ccode += """
            #define _MKL_DEBUG_ 0
            #define DIMENSION 4
            #define CHECK_ERR(f, err) \\
                do { \\
                    (err) = (f); \\
                    if ((err) != E_SUCCESS) { \\
                        (PyExc_RuntimeError, "Error in file " \\
                            "[%s:%d], err code (%d)", __FILE__, __LINE__, err); \\
                    } \\
                } while(0)
        """
        return ccode

    def c_support_code_struct(self, node, name):
        dtype = node.inputs[0].dtype
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
            size_t imageSize[DIMENSION]; //w, h, c, n
            size_t imageStride[DIMENSION];
            size_t weightSize[DIMENSION+1]; //w, h, c, n, group
            size_t weightStride[DIMENSION+1];
            size_t zSize[DIMENSION]; //w, h, c, n
            size_t zStride[DIMENSION];
            size_t biasSize[1]; //w, h, c, n
            size_t biasStride[1];
            size_t groups;
            size_t fdimension;

            ////////// debug only //////////
            size_t _image_size;
            size_t _weight_size;
            size_t _z_size;
            ///////////////////////////////

            size_t convStride[2];
            int convPadding[2];

            void *conv_res[dnnResourceNumber];
            void *conv_res_bias[dnnResourceNumber];

            void *image_buf;
            void *image_buf_from_previous;
            void *image_buf_to_previous;

            void *z_buf;

            void *weight_buf;
            void *weight_buf_tmp;

            void *bwdf2fwd_weight_buf;
            void *bias_buf;
            void *bias_buf_tmp;

            void *gradz_buf;
            void *gradz_buf_for_weight;
            void *gradz_buf_for_bias;

            dnnError_t err;
            dnnPrimitive_t pConvolutionFwd;
            dnnPrimitive_t pConvolutionBwdData;
            dnnPrimitive_t pConvolutionBwdFilter;
            dnnPrimitive_t pConvolutionBwdBias;

            dnnPrimitive_t bwdf_weight_to_fwd_internal;
            dnnPrimitive_t bwdf_weight_to_usr;
            dnnPrimitive_t bwdd_weight_to_bwdd_internal;

            dnnLayout_t bwdf_weight_internal_layout;
            dnnLayout_t image_user_layout;
            dnnLayout_t weight_usr_layout;
            dnnLayout_t z_user_layout;
            dnnLayout_t image_internal_layout;
            dnnLayout_t *image_internal_layout_buf;
            dnnLayout_t image_internal_layout_from_previous;
            dnnLayout_t weight_internal_layout;
            dnnLayout_t z_internal_layout;
            dnnLayout_t gradz_internal_layout;
            dnnLayout_t gradz_internal_layout_for_weight;
            dnnLayout_t gradz_internal_layout_for_bias;
            dnnLayout_t fwd_weight_internal_layout;

            dnnLayout_t bias_internal_layout;
            dnnLayout_t bias_usr_layout;

            dnnPrimitive_t image_to_internal;
            dnnPrimitive_t weight_to_internal;
            dnnPrimitive_t z_to_internal;
            dnnPrimitive_t weight_from_internal;
            dnnPrimitive_t image_from_internal;
            dnnPrimitive_t z_from_internal;
            dnnPrimitive_t internal_to_internal_image;
            dnnPrimitive_t gradz_to_internal;
            dnnPrimitive_t gradz_to_internal_for_weight;
            dnnPrimitive_t gradz_to_internal_for_bias;
            dnnPrimitive_t bias_to_internal;
            dnnPrimitive_t bias_from_internal;
        """ % sub
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
            first_run = 1;
            imageSize[0] = 0; //w, h, c, n
            imageSize[1] = 0; //w, h, c, n
            imageSize[2] = 0; //w, h, c, n
            imageSize[3] = 0; //w, h, c, n

            imageStride[0] = 0;
            imageStride[1] = 0;
            imageStride[2] = 0;
            imageStride[3] = 0;

            weightSize[0] = 0; //w, h, c, n, group
            weightSize[1] = 0; //w, h, c, n, group
            weightSize[2] = 0; //w, h, c, n, group
            weightSize[3] = 0; //w, h, c, n, group
            weightSize[4] = 0;

            weightStride[0] = 0;
            weightStride[1] = 0;
            weightStride[2] = 0;
            weightStride[3] = 0;
            weightStride[4] = 0;

            zSize[0] = 0; //w, h, c, n
            zSize[1] = 0; //w, h, c, n
            zSize[2] = 0; //w, h, c, n
            zSize[3] = 0; //w, h, c, n

            zStride[0] = 0;
            zStride[1] = 0;
            zStride[2] = 0;
            zStride[3] = 0;

            biasSize[0] = 0; //w, h, c, n
            biasStride[0] = 0;

            groups = 1;
            fdimension = 0;

            convStride[0] = 0;
            convStride[1] = 0;
            convPadding[0] = 0;
            convPadding[1] = 0;

            image_buf = NULL;
            image_buf_from_previous = NULL;
            image_buf_to_previous = NULL;

            z_buf = NULL;

            weight_buf = NULL;
            weight_buf_tmp = NULL;

            bwdf2fwd_weight_buf = NULL;
            bias_buf = NULL;
            bias_buf_tmp = NULL;

            gradz_buf = NULL;
            gradz_buf_for_weight = NULL;
            gradz_buf_for_bias = NULL;

            pConvolutionFwd = NULL;
            pConvolutionBwdData = NULL;
            pConvolutionBwdFilter = NULL;
            pConvolutionBwdBias = NULL;

            bwdf_weight_to_fwd_internal = NULL;
            bwdf_weight_to_usr = NULL;
            bwdd_weight_to_bwdd_internal = NULL;

            bwdf_weight_internal_layout = NULL;
            image_user_layout = NULL;
            weight_usr_layout = NULL;
            z_user_layout = NULL;
            image_internal_layout = NULL;
            image_internal_layout_buf = NULL;
            image_internal_layout_from_previous = NULL;
            weight_internal_layout = NULL;
            z_internal_layout = NULL;
            gradz_internal_layout = NULL;
            gradz_internal_layout_for_weight = NULL;
            gradz_internal_layout_for_bias = NULL;
            fwd_weight_internal_layout = NULL;

            bias_internal_layout = NULL;
            bias_usr_layout = NULL;

            image_to_internal = NULL;
            weight_to_internal = NULL;
            z_to_internal = NULL;
            weight_from_internal = NULL;
            image_from_internal = NULL;
            z_from_internal = NULL;
            internal_to_internal_image = NULL;
            gradz_to_internal = NULL;
            gradz_to_internal_for_weight = NULL;
            gradz_to_internal_for_bias = NULL;
            bias_to_internal = NULL;
            bias_from_internal = NULL;
        """
        return ccode


class Conv2D(MKLConvBase):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_flip', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_flip=False, filter_dilation=(1, 1)):
        super(Conv2D, self).__init__(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample)
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"

        ccode = """
            // release layout
            if (image_internal_layout) {
                dnnLayoutDelete_%(precision)s(image_internal_layout);
                image_internal_layout = NULL;
            }

            if (weight_usr_layout) {
                dnnLayoutDelete_%(precision)s(weight_usr_layout);
                weight_internal_layout = NULL;
            }

            if (weight_internal_layout) {
                dnnLayoutDelete_%(precision)s(weight_internal_layout);
                weight_internal_layout = NULL;
            }

            if (bias_usr_layout) {
                dnnLayoutDelete_%(precision)s(bias_usr_layout);
                bias_usr_layout = NULL;
            }

            if (bias_internal_layout) {
                dnnLayoutDelete_%(precision)s(bias_internal_layout);
                bias_internal_layout = NULL;
            }

            // release primitive
            if (pConvolutionFwd) {
                dnnDelete_%(precision)s(pConvolutionFwd);
                pConvolutionFwd = NULL;
            }

            if (internal_to_internal_image) {
                dnnDelete_%(precision)s(internal_to_internal_image);
                internal_to_internal_image = NULL;
            }

            if (weight_to_internal) {
                dnnDelete_%(precision)s(weight_to_internal);
                weight_to_internal = NULL;
            }

            if (bias_to_internal) {
                dnnDelete_%(precision)s(bias_to_internal);
                bias_to_internal = NULL;
            }

            // release buffer
            if (image_buf) {
                dnnReleaseBuffer_%(precision)s(image_buf);
                image_buf = NULL;
            }

            if (weight_buf) {
                dnnReleaseBuffer_%(precision)s(weight_buf);
                weight_buf = NULL;
            }

            if (bias_buf) {
                dnnReleaseBuffer_%(precision)s(bias_buf);
                bias_buf = NULL;
            }
        """ % locals()
        return ccode

    def make_node(self, image, weight, bias=None):
        if not isinstance(image.type, mkl_type.MKLNdarrayType):
            raise TypeError('Conv2D: input image should be an instance of MKLNdarray')

        weight = as_tensor_variable(weight)

        if image.type.ndim != 4:
            raise TypeError('Conv2D: image should be 4D tensor')
        if weight.type.ndim not in (4, 5):
            raise TypeError('Conv2D: weight should be 4D or 5D tensor')

        if weight.type.ndim == 4:
            broadcastable = [image.type.broadcastable[0], weight.type.broadcastable[0], False, False]
        else:
            broadcastable = [image.type.broadcastable[0], weight.type.broadcastable[1], False, False]

        if bias is not None:
            bias = as_tensor_variable(bias)
            inputs = [image, weight, bias]
        else:
            inputs = [image, weight]
        return Apply(self, inputs, [mkl_type.MKLNdarrayType(image.type.dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        imshp = input_shape[0]
        gkshp = input_shape[1]

        if len(gkshp) == 5:
            kshp = [gkshp[1] * gkshp[0], gkshp[2] * gkshp[0], gkshp[3], gkshp[4]]
        else:
            kshp = [gkshp[0], gkshp[1], gkshp[2], gkshp[3]]

        res = get_conv_output_shape(imshp,
                                    kshp,
                                    self.border_mode,
                                    self.subsample)
        return [res]

    def c_code(self, node, name, inp, out_, sub):
        if len(inp) > 2:
            image, weight, bias = inp
        else:
            image, weight = inp
            bias = None

        z, = out_

        if None in self.kshp:
            raise ValueError('Conv2D: None in kernel shape')

        if None in self.imshp:
            in_n, in_c, in_h, in_w = 0, 0, 0, 0
            o_n, o_c, o_h, o_w = 0, 0, 0, 0
        else:
            in_n, in_c, in_h, in_w = self.imshp
            o_n, o_c, o_h, o_w = get_conv_output_shape(self.imshp,
                                                       self.kshp,
                                                       self.border_mode,
                                                       self.subsample)

        if node.inputs[1].type.ndim == 5:
            grp, k_n, k_c, k_h, k_w = self.kshp
        else:
            k_n, k_c, k_h, k_w = self.kshp
            grp = 1

        dH, dW = self.subsample

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

        sub['image'] = image
        sub['weight'] = weight
        sub['z'] = z
        sub['bias'] = bias

        if bias is not None:
            withBias = 1
            sub['bias'] = bias
        else:
            withBias = 0
        sub['withBias'] = withBias

        if node.inputs[0].dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'

        sub.update(locals())

        if bias is None:
            sub['bias'] = 'NULL'

        ccode = """
            int status = 0;
            if (1 == first_run) {
                convStride[0] = %(dW)s;
                convStride[1] = %(dH)s;
                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                imageSize[0] = %(in_w)s;  //w
                imageSize[1] = %(in_h)s;  //h
                imageSize[2] = %(in_c)s;  //c
                imageSize[3] = %(in_n)s;  //n

                if (0 == imageSize[0] || 0 == imageSize[1] || 0 == imageSize[2] || 0 == imageSize[3]) {
                    imageSize[0] = MKLNdarray_DIMS(%(image)s)[3];
                    imageSize[1] = MKLNdarray_DIMS(%(image)s)[2];
                    imageSize[2] = MKLNdarray_DIMS(%(image)s)[1];
                    imageSize[3] = MKLNdarray_DIMS(%(image)s)[0];
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

                if (0 == zSize[0] || 0 == zSize[1] || 0 == zSize[2] || 0 == zSize[3]) {
                    zSize[0] = (imageSize[0] - 2 * convPadding[0] - weightSize[0]) / convStride[0] + 1;
                    zSize[1] = (imageSize[1] - 2 * convPadding[1] - weightSize[1]) / convStride[1] + 1;
                    zSize[2] = weightSize[3];
                    zSize[3] = imageSize[3];
                }

                zStride[0] = 1;
                zStride[1] = zSize[0];
                zStride[2] = zSize[0] * zSize[1];
                zStride[3] = zSize[0] * zSize[1] * zSize[2];

                if(%(withBias)s) {
                    biasSize[0] = zSize[2];
                    biasStride[0] = 1;
                }

                groups = %(grp)s;
                fdimension = DIMENSION + (groups != 1);

                // Create conv forward primitive
                if (%(withBias)s) {
                    CHECK_ERR( dnnGroupsConvolutionCreateForwardBias_%(precision)s(&pConvolutionFwd, NULL,
                               dnnAlgorithmConvolutionDirect, groups, DIMENSION, imageSize,
                               zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );
                    CHECK_ERR( dnnLayoutCreate_%(precision)s(&bias_usr_layout, 1, biasSize, biasStride), err );
                    CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&bias_internal_layout,
                                                                          pConvolutionFwd,
                                                                          dnnResourceBias), err );

                    if (!dnnLayoutCompare_%(precision)s(bias_usr_layout, bias_internal_layout)) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bias_buf,
                                                                   bias_internal_layout), err );
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&bias_to_internal,
                                                                     bias_usr_layout,
                                                                     bias_internal_layout), err );
                    } else {
                        bias_to_internal = NULL;
                        bias_buf = NULL;
                    }
                } else {
                    CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&pConvolutionFwd, NULL,
                               dnnAlgorithmConvolutionDirect, groups, DIMENSION, imageSize,
                               zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );
                }

                // For internal weights
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&weight_usr_layout,
                                                         fdimension,
                                                         weightSize,
                                                         weightStride), err );
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&weight_internal_layout,
                                                                      pConvolutionFwd,
                                                                      dnnResourceFilter), err );
                if (!dnnLayoutCompare_%(precision)s(weight_usr_layout, weight_internal_layout)) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buf,
                                                               weight_internal_layout), err );
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&weight_to_internal,
                                                                 weight_usr_layout,
                                                                 weight_internal_layout), err );
                } else {
                    weight_to_internal = NULL;
                    weight_buf = NULL;
                }

                // For internal image
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&image_internal_layout,
                           pConvolutionFwd, dnnResourceSrc), err );
                if (!dnnLayoutCompare_%(precision)s(MKLNdarray_LAYOUT(%(image)s), image_internal_layout)) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&image_buf,
                                                               image_internal_layout), err );
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&internal_to_internal_image,
                                                                 MKLNdarray_LAYOUT(%(image)s),
                                                                 image_internal_layout), err );
                } else {
                    internal_to_internal_image = NULL;
                    image_buf = NULL;
                }

            }

            if (! (%(z)s
                && MKLNdarray_Check((PyObject*)%(z)s)
                && MKLNdarray_NDIM(%(z)s) == MKLNdarray_NDIM(%(image)s)
                && MKLNdarray_DIMS(%(z)s)[0] == MKLNdarray_DIMS(%(image)s)[0]
                && MKLNdarray_DIMS(%(z)s)[1] == MKLNdarray_DIMS(%(image)s)[1]
                && MKLNdarray_DIMS(%(z)s)[2] == MKLNdarray_DIMS(%(image)s)[2]
                && MKLNdarray_DIMS(%(z)s)[3] == MKLNdarray_DIMS(%(image)s)[3] )) {

                if (%(z)s) Py_XDECREF(%(z)s);

                %(z)s = (MKLNdarray*)MKLNdarray_New(MKLNdarray_NDIM(%(image)s),
                                                    MKLNdarray_TYPE(%(image)s));
                if (! %(z)s) {
                    %(fail)s;
                }

                size_t z_dims[4] = {zSize[3], zSize[2], zSize[1], zSize[0]};
                status = MKLNdarray_set_structure(%(z)s, MKLNdarray_NDIM(%(image)s), z_dims);
                if (status != 0) {
                    %(fail)s;
                }

                // create dst layout and buffer in z
                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &pConvolutionFwd, dnnResourceDst);
                if (status != 0) {
                    %(fail)s;
                }
            }   // else reuse %(z)s

            if (internal_to_internal_image) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(internal_to_internal_image,
                                                              MKLNdarray_DATA(%(image)s),
                                                              image_buf), err );
                conv_res[dnnResourceSrc] = image_buf;
            } else {
                conv_res[dnnResourceSrc] = MKLNdarray_DATA(%(image)s);
            }

            if (weight_to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(weight_to_internal,
                                                              (%(dtype)s*)PyArray_DATA(%(weight)s),
                                                              weight_buf), err );
                conv_res[dnnResourceFilter] = weight_buf;
            } else {
                conv_res[dnnResourceFilter] = (void*)PyArray_DATA(%(weight)s);
            }

            if(%(withBias)s) {
                if (bias_to_internal) {
                    CHECK_ERR( dnnConversionExecute_%(precision)s(bias_to_internal,
                                                                  (%(dtype)s*)PyArray_DATA(%(bias)s),
                                                                  bias_buf), err );
                    conv_res[dnnResourceBias] = bias_buf;
                } else {
                    conv_res[dnnResourceBias] = (void*)PyArray_DATA(%(bias)s);
                }
            } else {
                conv_res[dnnResourceBias] = NULL;
            }

            conv_res[dnnResourceDst] = MKLNdarray_DATA(%(z)s);
            //Execute convolution forward pass
            CHECK_ERR( dnnExecute_%(precision)s(pConvolutionFwd, (void**)conv_res), err );

            first_run = 0;
        """ % sub
        return ccode

    def grad(self, inp, grads):
        if len(inp) > 2:
            image, weights, bias = inp
        else:
            image, weights = inp
            bias = None

        gz, = grads
        d_images = ConvGradInputs(border_mode=self.border_mode,
                                  subsample=self.subsample,
                                  imshp=self.imshp,
                                  kshp=self.kshp,
                                  filter_flip=self.filter_flip)(image, weights, gz)

        dlist = ConvGradWeights(border_mode=self.border_mode,
                                subsample=self.subsample,
                                imshp=self.imshp,
                                kshp=self.kshp,
                                filter_flip=self.filter_flip)(image, weights, gz, bias)

        if bias is None:
            d_weights = dlist
            return d_images, d_weights
        else:
            d_weights, d_bias = dlist
            return d_images, d_weights, d_bias


class ConvGradInputs(MKLConvBase):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_flip', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_flip=True, filter_dilation=(1, 1)):
        super(ConvGradInputs, self).__init__(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample)
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"
        ccode = """
            // release layout
            if (weight_usr_layout) {
                dnnLayoutDelete_%(precision)s(weight_usr_layout);
                weight_usr_layout = NULL;
            }

            if (weight_internal_layout) {
                dnnLayoutDelete_%(precision)s(weight_internal_layout);
                weight_internal_layout = NULL;
            }

            if (gradz_internal_layout) {
                dnnLayoutDelete_%(precision)s(gradz_internal_layout);
                gradz_internal_layout = NULL;
            }

            if (image_internal_layout) {
                dnnLayoutDelete_%(precision)s(image_internal_layout);
                image_internal_layout = NULL;
            }

            // release primitive
            if (pConvolutionBwdData) {
                dnnDelete_%(precision)s(pConvolutionBwdData);
                pConvolutionBwdData = NULL;
            }

            if (weight_to_internal) {
                dnnDelete_%(precision)s(weight_to_internal);
                weight_to_internal = NULL;
            }

            if (gradz_to_internal) {
                dnnDelete_%(precision)s(gradz_to_internal);
                gradz_to_internal = NULL;
            }

            if (internal_to_internal_image) {
                dnnDelete_%(precision)s(internal_to_internal_image);
                internal_to_internal_image = NULL;
            }

            // release buffer
            if (weight_buf) {
                dnnReleaseBuffer_%(precision)s(weight_buf);
                weight_buf = NULL;
            }

            if (gradz_buf) {
                dnnReleaseBuffer_%(precision)s(gradz_buf);
                gradz_buf = NULL;
            }

            if (image_buf) {
                dnnReleaseBuffer_%(precision)s(image_buf);
                image_buf = NULL;
            }
        """ % locals()
        return ccode

    def make_node(self, image, weight, gradz):
        if not isinstance(image.type, mkl_type.MKLNdarrayType) or image.type.ndim != 4:
            raise TypeError('ConvGradInputs: input image should be 4-dim MKLNdarray')

        if not isinstance(gradz.type, mkl_type.MKLNdarrayType) or gradz.type.ndim != 4:
            raise TypeError('ConvGradInputs: input gradz should be 4-dim MKLNdarray')

        weight = as_tensor_variable(weight)
        if weight.type.ndim not in [4, 5]:
            raise TypeError('ConvGradInputs: weight should be 4D or 5D tensor')

        if weight.type.ndim == 4:
            broadcastable = [gradz.type.broadcastable[0], weight.type.broadcastable[1], False, False]
        else:
            broadcastable = [gradz.type.broadcastable[0], weight.type.broadcastable[2], False, False]

        return Apply(self, [image, weight, gradz], [mkl_type.MKLNdarrayType(image.type.dtype, broadcastable)()])

    def c_code(self, node, name, inp, out_, sub):
        image, weights, gradz = inp
        imagegrad, = out_

        if None in self.kshp:
            raise ValueError('ConvGradInputs: None in kernel shape')

        if node.inputs[1].type.ndim == 5:
            tshp = [self.kshp[1] * self.kshp[0], self.kshp[2] * self.kshp[0], self.kshp[3], self.kshp[4]]
        else:
            tshp = [self.kshp[0], self.kshp[1], self.kshp[2], self.kshp[3]]

        if None in self.imshp:
            in_n, in_c, in_h, in_w = 0, 0, 0, 0
            o_n, o_c, o_h, o_w = 0, 0, 0, 0
        else:
            in_n, in_c, in_h, in_w = self.imshp
            outshp = get_conv_output_shape(self.imshp, tshp, self.border_mode, self.subsample)
            o_n, o_c, o_h, o_w = outshp

        if node.inputs[1].type.ndim == 5:
            grp, k_n, k_c, k_h, k_w = self.kshp
        else:
            grp = 1
            k_n, k_c, k_h, k_w = self.kshp

        dH, dW = self.subsample

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

        sub['image'] = image
        sub['imagegrad'] = imagegrad
        sub['weight'] = weights
        sub['gradz'] = gradz

        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'
        sub.update(locals())

        ccode = """
            int status = 0;
            if (1 == first_run) {
                convStride[0] = %(dW)s;
                convStride[1] = %(dH)s;

                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                imageSize[0] = %(in_w)s;  //w
                imageSize[1] = %(in_h)s;  //h
                imageSize[2] = %(in_c)s;  //c
                imageSize[3] = %(in_n)s;  //n

                if (0 == imageSize[0] || 0 == imageSize[1] || 0 == imageSize[2] || 0 == imageSize[3]) {
                    imageSize[0] = MKLNdarray_DIMS(%(image)s)[3];
                    imageSize[1] = MKLNdarray_DIMS(%(image)s)[2];
                    imageSize[2] = MKLNdarray_DIMS(%(image)s)[1];
                    imageSize[3] = MKLNdarray_DIMS(%(image)s)[0];
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

                if (0 == zSize[0] || 0 == zSize[1] || 0 == zSize[2] || 0 == zSize[3]) {
                    zSize[0] = (imageSize[0] - 2 * convPadding[0] - weightSize[0]) / convStride[0] + 1;
                    zSize[0] = (imageSize[1] - 2 * convPadding[1] - weightSize[1]) / convStride[1] + 1;
                    zSize[0] = weightSize[3];
                    zSize[0] = imageSize[3];
                }

                zStride[0] = 1;
                zStride[1] = zSize[0];
                zStride[2] = zSize[0] * zSize[1];
                zStride[3] = zSize[0] * zSize[1] * zSize[2];

                groups = %(grp)s;
                fdimension = DIMENSION + (groups != 1);

                // Create conv gradInput primitive
                CHECK_ERR( dnnGroupsConvolutionCreateBackwardData_%(precision)s(&pConvolutionBwdData, NULL,
                           dnnAlgorithmConvolutionDirect, groups, DIMENSION, imageSize,
                           zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );

                // For internal weight
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&weight_internal_layout,
                           pConvolutionBwdData, dnnResourceFilter), err );
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&weight_usr_layout, fdimension, weightSize, weightStride), err);
                if (!dnnLayoutCompare_%(precision)s(weight_usr_layout, weight_internal_layout)) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buf, weight_internal_layout), err);
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&weight_to_internal,
                                                                 weight_usr_layout,
                                                                 weight_internal_layout), err );
                } else {
                    weight_to_internal = NULL;
                    weight_buf = NULL;
                }

                // For internal diff dst (gradz)
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&gradz_internal_layout,
                                                                      pConvolutionBwdData,
                                                                      dnnResourceDiffDst), err);
                if (!dnnLayoutCompare_%(precision)s(gradz_internal_layout, MKLNdarray_LAYOUT(%(gradz)s) )) {
                   CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&gradz_buf, gradz_internal_layout), err );
                   CHECK_ERR( dnnConversionCreate_%(precision)s(&gradz_to_internal,
                                                                MKLNdarray_LAYOUT(%(gradz)s),
                                                                gradz_internal_layout), err );
                } else {
                    gradz_to_internal = NULL;
                    gradz_buf = NULL;
                }

                // For internal diff src (grad image)
                // need convert grad image from internal layout to layout of input image
                // so they can be added for update
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&image_internal_layout,
                           pConvolutionBwdData, dnnResourceDiffSrc), err );
                if (! dnnLayoutCompare_%(precision)s(MKLNdarray_LAYOUT(%(image)s), image_internal_layout)) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)image_buf, image_internal_layout), err);
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&internal_to_internal_image,
                                                                 image_internal_layout,
                                                                 MKLNdarray_LAYOUT(%(image)s)), err);
                } else {
                    image_to_internal = NULL;
                    image_buf = NULL;
                }
            }

            if (! (%(imagegrad)s
                && MKLNdarray_Check((PyObject*)%(imagegrad)s)
                && MKLNdarray_NDIM(%(imagegrad)s) == MKLNdarray_NDIM(%(image)s)
                && MKLNdarray_DIMS(%(imagegrad)s)[0] == MKLNdarray_DIMS(%(image)s)[0]
                && MKLNdarray_DIMS(%(imagegrad)s)[1] == MKLNdarray_DIMS(%(image)s)[1]
                && MKLNdarray_DIMS(%(imagegrad)s)[2] == MKLNdarray_DIMS(%(image)s)[2]
                && MKLNdarray_DIMS(%(imagegrad)s)[3] == MKLNdarray_DIMS(%(image)s)[3])) {

                if (%(imagegrad)s) Py_XDECREF(%(imagegrad)s);

                %(imagegrad)s = (MKLNdarray*)MKLNdarray_New(MKLNdarray_NDIM(%(image)s),
                                                            MKLNdarray_TYPE(%(image)s));
                if (! %(imagegrad)s) {
                    %(fail)s;
                }

                status = MKLNdarray_set_structure(%(imagegrad)s,
                                                  MKLNdarray_NDIM(%(image)s),
                                                  MKLNdarray_DIMS(%(image)s));
                if (status != 0) {
                    %(fail)s;
                }

                status = MKLNdarray_copy_layout(%(imagegrad)s, %(image)s, MNDA_DATA);
                if (status != 0) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_layout(%(imagegrad)s, MNDA_DATA);
                if (status != 0) {
                    %(fail)s;
                }
            }   // else reuse %(imagegrad)s

            if (weight_to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(weight_to_internal,
                                                              PyArray_DATA(%(weight)s),
                                                              weight_buf), err );
                conv_res[dnnResourceFilter] = weight_buf;
            } else {
                conv_res[dnnResourceFilter] = PyArray_DATA(%(weight)s);
            }

            if (gradz_to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(gradz_to_internal,
                                                              MKLNdarray_DATA(%(gradz)s),
                                                              gradz_buf), err );
                conv_res[dnnResourceDiffDst] = gradz_buf;
            } else {
                conv_res[dnnResourceDiffDst] = MKLNdarray_DATA(%(gradz)s);
            }

            if (internal_to_internal_image) {
                conv_res[dnnResourceDiffSrc] = image_buf;
            } else {
                conv_res[dnnResourceDiffSrc] = MKLNdarray_DATA(%(imagegrad)s);
            }

            CHECK_ERR( dnnExecute_%(precision)s(pConvolutionBwdData, (void**)conv_res), err );
            if (internal_to_internal_image) {
                CHECK_ERR (dnnConversionExecute_%(precision)s(internal_to_internal_image,
                                                              image_buf,
                                                              MKLNdarray_DATA(%(imagegrad)s)), err );
            }

           first_run = 0;
        """ % sub
        return ccode


class ConvGradWeights(MKLConvBase):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_flip', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_flip=False, filter_dilation=(1, 1)):
        super(ConvGradWeights, self).__init__(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample)
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"
        ccode = """
            // release layout
            if (weight_usr_layout) {
                dnnLayoutDelete_%(precision)s(weight_usr_layout);
                weight_usr_layout = NULL;
            }

            if (weight_internal_layout) {
                dnnLayoutDelete_%(precision)s(weight_internal_layout);
                weight_internal_layout = NULL;
            }

            if (image_internal_layout) {
                dnnLayoutDelete_%(precision)s(image_internal_layout);
                image_internal_layout = NULL;
            }

            if (gradz_internal_layout_for_weight) {
                dnnLayoutDelete_%(precision)s(gradz_internal_layout_for_weight);
                gradz_internal_layout_for_weight = NULL;
            }

            if (gradz_internal_layout_for_bias) {
                dnnLayoutDelete_%(precision)s(gradz_internal_layout_for_bias);
                gradz_internal_layout_for_bias = NULL;
            }
            if (image_internal_layout) {
                dnnLayoutDelete_%(precision)s(image_internal_layout);
                image_internal_layout = NULL;
            }

            if (bias_usr_layout) {
                dnnLayoutDelete_%(precision)s(bias_usr_layout);
                bias_usr_layout = NULL;
            }

            if (bias_internal_layout) {
                dnnLayoutDelete_%(precision)s(bias_internal_layout);
                bias_internal_layout = NULL;
            }

            // release primitive
            if (pConvolutionBwdFilter) {
                dnnDelete_%(precision)s(pConvolutionBwdFilter);
                pConvolutionBwdFilter = NULL;
            }

            if (pConvolutionBwdBias) {
                dnnDelete_%(precision)s(pConvolutionBwdBias);
                pConvolutionBwdBias = NULL;
            }

            if (weight_from_internal) {
                dnnDelete_%(precision)s(weight_from_internal);
                weight_from_internal = NULL;
            }

            if (bias_from_internal) {
                dnnDelete_%(precision)s(bias_from_internal);
                bias_from_internal = NULL;
            }

            if (internal_to_internal_image) {
                dnnDelete_%(precision)s(internal_to_internal_image);
                internal_to_internal_image = NULL;
            }

            if (gradz_to_internal_for_weight) {
                dnnDelete_%(precision)s(gradz_to_internal_for_weight);
                gradz_to_internal_for_weight = NULL;
            }

            if (gradz_to_internal_for_bias) {
                dnnDelete_%(precision)s(gradz_to_internal_for_bias);
                gradz_to_internal_for_bias = NULL;
            }

            // release buffer
            if (weight_buf) {
                dnnReleaseBuffer_%(precision)s(weight_buf);
                weight_buf = NULL;
            }

            if (image_buf) {
                dnnReleaseBuffer_%(precision)s(image_buf);
                image_buf = NULL;
            }

            if (gradz_buf_for_weight) {
                dnnReleaseBuffer_%(precision)s(gradz_buf_for_weight);
                gradz_buf_for_weight = NULL;
            }

            if (gradz_buf_for_bias) {
                dnnReleaseBuffer_%(precision)s(gradz_buf_for_bias);
                gradz_buf_for_bias = NULL;
            }

            if (bias_buf) {
                dnnReleaseBuffer_%(precision)s(bias_buf);
                bias_buf = NULL;
            }
        """ % locals()
        return ccode

    def make_node(self, image, weight, gradz, bias=None):
        if not isinstance(image.type, mkl_type.MKLNdarrayType) or image.type.ndim is not 4:
            raise TypeError('ConvGradWeights: input image should be 4-dim MKLNdarray')

        if not isinstance(gradz.type, mkl_type.MKLNdarrayType) or gradz.type.ndim is not 4:
            raise TypeError('ConvGradWeights: input gradz should be 4-dim MKLNdarray')

        weight = as_tensor_variable(weight)
        if weight.type.ndim not in (4, 5):
            raise TypeError('ConvGradWeights: input weight should be 4D or 5D tensor')

        if weight.type.ndim == 4:
            weightbt = [gradz.type.broadcastable[1], image.type.broadcastable[1], False, False]
        else:
            weightbt = [False, gradz.type.broadcastable[1], image.type.broadcastable[1], False, False]

        dtype = image.type.dtype
        if bias is not None:
            bias = as_tensor_variable(bias)
            inputs = [image, weight, gradz, bias]
            biasbt = [gradz.type.broadcastable[1]]
            outputs = [TensorType(dtype, weightbt)(), TensorType(dtype, biasbt)()]
        else:
            inputs = [image, weight, gradz]
            outputs = [TensorType(dtype, weightbt)()]

        return Apply(self, inputs, outputs)

    def c_code(self, node, name, inp, out_, sub):
        if len(inp) > 3:
            image, weight, gradz, bias = inp
            weightgrad, biasgrad = out_
        else:
            image, weight, gradz = inp
            bias = None
            weightgrad, = out_

        if None in self.kshp:
            raise ValueError('ConvGradWeights: None in kernel shape')

        if node.inputs[1].type.ndim == 5:
            grp, k_n, k_c, k_h, k_w = self.kshp
            tshp = [self.kshp[1] * self.kshp[0], self.kshp[2] * self.kshp[0], self.kshp[3], self.kshp[4]]
        else:
            k_n, k_c, k_h, k_w = self.kshp
            grp = 1
            tshp = [self.kshp[0], self.kshp[1], self.kshp[2], self.kshp[3]]

        if None in self.imshp:
            in_n, in_c, in_h, in_w = 0, 0, 0, 0
            o_n, o_c, o_h, o_w = 0, 0, 0, 0
        else:
            in_n, in_c, in_h, in_w = self.imshp
            outshp = get_conv_output_shape(self.imshp, tshp, self.border_mode, self.subsample)
            o_n, o_c, o_h, o_w = outshp

        if bias is not None:
            sub['bias'] = bias
            sub['biasgrad'] = biasgrad
            withBias = 1
        else:
            withBias = 0
        sub['withBias'] = withBias

        dH, dW = self.subsample
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

        sub['image'] = image
        sub['weight'] = weight
        sub['weightgrad'] = weightgrad
        sub['gradz'] = gradz

        if node.inputs[0].dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'
        sub.update(locals())

        if bias is None:
            sub['bias'] = 'NULL'
            sub['biasgrad'] = 'NULL'

        ccode = """
            int status = 0;
            if (1 == first_run) {
                convStride[0] = %(dW)s;
                convStride[1] = %(dH)s;
                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                imageSize[0] = %(in_w)s;  //w
                imageSize[1] = %(in_h)s;  //h
                imageSize[2] = %(in_c)s;  //c
                imageSize[3] = %(in_n)s;  //n

                if (0 == imageSize[0] || 0 == imageSize[1] || 0 == imageSize[2] || 0 == imageSize[3]) {
                    imageSize[0] = MKLNdarray_DIMS(%(image)s)[3];
                    imageSize[1] = MKLNdarray_DIMS(%(image)s)[2];
                    imageSize[2] = MKLNdarray_DIMS(%(image)s)[1];
                    imageSize[3] = MKLNdarray_DIMS(%(image)s)[0];
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

                if (0 == zSize[0] || 0 == zSize[1] || 0 == zSize[2] || 0 == zSize[3]) {
                    zSize[0] = (imageSize[0] - 2 * convPadding[0] - weightSize[0]) / convStride[0] + 1;
                    zSize[1] = (imageSize[1] - 2 * convPadding[1] - weightSize[1]) / convStride[1] + 1;
                    zSize[2] = weightSize[3];
                    zSize[3] = imageSize[3];
                }

                zStride[0] = 1;
                zStride[1] = zSize[0];
                zStride[2] = zSize[0] * zSize[1];
                zStride[3] = zSize[0] * zSize[1] * zSize[2];

                if( %(withBias)s ) {
                    biasSize[0] = zSize[2];
                    biasStride[0] = 1;
                }

                groups = %(grp)s;
                fdimension = DIMENSION + (groups != 1);

                // Create conv backward primitive
                CHECK_ERR( dnnGroupsConvolutionCreateBackwardFilter_%(precision)s(&pConvolutionBwdFilter, NULL,
                           dnnAlgorithmConvolutionDirect, groups, DIMENSION, imageSize,
                           zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );

                // For internal weight
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&weight_internal_layout,
                           pConvolutionBwdFilter, dnnResourceDiffFilter), err );

                CHECK_ERR( dnnLayoutCreate_%(precision)s(&weight_usr_layout,
                                                         fdimension,
                                                         weightSize,
                                                         weightStride), err);

                if ( !dnnLayoutCompare_%(precision)s(weight_usr_layout, weight_internal_layout)) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buf,
                                                               weight_internal_layout), err );
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&weight_from_internal,
                                                                 weight_internal_layout,
                                                                 weight_usr_layout), err );
                } else {
                    weight_from_internal = NULL;
                    weight_buf = NULL;
                }

                // For internal image
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&image_internal_layout,
                           pConvolutionBwdFilter, dnnResourceSrc), err );

                if ( !dnnLayoutCompare_%(precision)s(image_internal_layout, MKLNdarray_LAYOUT(%(image)s))) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&image_buf,
                                                               image_internal_layout), err );
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&internal_to_internal_image,
                                                                 MKLNdarray_LAYOUT(%(image)s),
                                                                 image_internal_layout), err );
                } else {
                    internal_to_internal_image = NULL;
                    image_buf = NULL;
                }

                // For internal gradz
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&gradz_internal_layout_for_weight,
                           pConvolutionBwdFilter, dnnResourceDiffDst), err );

                if ( !dnnLayoutCompare_%(precision)s(gradz_internal_layout_for_weight,
                                                     MKLNdarray_LAYOUT(%(gradz)s))) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&gradz_buf_for_weight,
                                                               gradz_internal_layout_for_weight), err );
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&gradz_to_internal_for_weight,
                                                                 MKLNdarray_LAYOUT(%(gradz)s),
                                                                 gradz_internal_layout_for_weight), err );
                } else {
                    gradz_to_internal_for_weight = NULL;
                    gradz_buf_for_weight = NULL;
                }

                if( %(withBias)s ) {
                    CHECK_ERR( dnnGroupsConvolutionCreateBackwardBias_%(precision)s(&pConvolutionBwdBias, NULL,
                                dnnAlgorithmConvolutionDirect, groups, DIMENSION, zSize), err );

                    CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&bias_internal_layout,
                               pConvolutionBwdBias, dnnResourceDiffBias), err );

                    CHECK_ERR( dnnLayoutCreate_%(precision)s(&bias_usr_layout, 1, biasSize, biasStride), err );

                    if ( !dnnLayoutCompare_%(precision)s(bias_usr_layout, bias_internal_layout)) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bias_buf, bias_internal_layout), err);
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&bias_from_internal,
                                                                     bias_internal_layout,
                                                                     bias_usr_layout), err);
                    } else {
                        bias_from_internal = NULL;
                        bias_buf = NULL;
                    }

                    CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&gradz_internal_layout_for_bias,
                               pConvolutionBwdBias, dnnResourceDiffDst), err );

                    if ( !dnnLayoutCompare_%(precision)s(gradz_internal_layout_for_bias,
                                                         MKLNdarray_LAYOUT(%(gradz)s))) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&gradz_buf_for_bias,
                                                                   gradz_internal_layout_for_bias), err );
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&gradz_to_internal_for_bias,
                                                                     MKLNdarray_LAYOUT(%(gradz)s),
                                                                     gradz_internal_layout_for_bias), err );
                    } else {
                        gradz_to_internal_for_bias = NULL;
                        gradz_buf_for_bias = NULL;
                    }
                }
            }

            // Prepare weightgrad array
            if ( !(%(weightgrad)s) ) {
                %(weightgrad)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(weight)s),
                                                               PyArray_DIMS(%(weight)s),
                                                               PyArray_TYPE(%(weight)s),
                                                               0);
                if (NULL == %(weightgrad)s) {
                    %(fail)s;
                }
            }
            """ % sub

        if bias is not None:
            ccode += """
            if ( !%(biasgrad)s) {
                %(biasgrad)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(bias)s),
                                                             PyArray_DIMS(%(bias)s),
                                                             PyArray_TYPE(%(bias)s),
                                                             0);
                if (NULL == %(biasgrad)s) {
                    %(fail)s;
                }
            }
            """ % sub

        ccode += """
            // set conv_rest for computing gradweight
            if (internal_to_internal_image) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(internal_to_internal_image,
                                                              MKLNdarray_DATA(%(image)s),
                                                              image_buf), err );
                conv_res[dnnResourceSrc] = image_buf;
            } else {
                conv_res[dnnResourceSrc] = MKLNdarray_DATA(%(image)s);
            }

            if (gradz_to_internal_for_weight) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(gradz_to_internal_for_weight,
                                                              MKLNdarray_DATA(%(gradz)s),
                                                              gradz_buf_for_weight), err );
                conv_res[dnnResourceDiffDst] = gradz_buf_for_weight;
            } else {
                conv_res[dnnResourceDiffDst] = MKLNdarray_DATA(%(gradz)s);
            }

            if (weight_from_internal) {
                conv_res[dnnResourceDiffFilter] = weight_buf;
            } else {
                conv_res[dnnResourceDiffFilter] = PyArray_DATA(%(weightgrad)s);
            }

            CHECK_ERR( dnnExecute_%(precision)s(pConvolutionBwdFilter, (void**)conv_res), err );

            if (weight_from_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(weight_from_internal,
                                                              weight_buf,
                                                              PyArray_DATA(%(weightgrad)s)), err );
            }

            if (%(withBias)s) {
                // set conv_res_bias for computing gradbias
                if (gradz_to_internal_for_bias) {
                    CHECK_ERR( dnnConversionExecute_%(precision)s(gradz_to_internal_for_bias,
                                                                  MKLNdarray_DATA(%(gradz)s),
                                                                  gradz_buf_for_bias), err );
                    conv_res_bias[dnnResourceDiffDst] = gradz_buf_for_bias;
                } else {
                    conv_res_bias[dnnResourceDiffDst] = MKLNdarray_DATA(%(gradz)s);
                }

                if (bias_from_internal) {
                    conv_res_bias[dnnResourceDiffBias] = bias_buf;
                } else {
                    conv_res_bias[dnnResourceDiffBias] = PyArray_DATA(%(biasgrad)s);
                }

                //Execute convolution gradbias pass
                CHECK_ERR( dnnExecute_%(precision)s(pConvolutionBwdBias, (void**)conv_res_bias), err );

                if (bias_from_internal) {
                    CHECK_ERR( dnnConversionExecute_%(precision)s(bias_from_internal,
                                                                  bias_buf,
                                                                  PyArray_DATA(%(biasgrad)s)), err );
                }
            }

            first_run = 0;
        """ % sub
        return ccode


class AbstractConvGroup(gof.Op):
    __props__ = ('imshp', 'kshp', 'subsample', 'border_mode', 'filter_flip', 'filter_dilation', 'group')

    def __init__(self,
                 imshp=None, kshp=None, subsample=(1, 1),
                 border_mode='valid', filter_flip=False,
                 filter_dilation=(1, 1), group=1):

        super(AbstractConvGroup, self).__init__()
        imshp = tuple(imshp) if imshp else (None, ) * 4
        kshp = tuple(kshp) if kshp else (None, ) * 4

        if len(subsample) != 2:
            raise ValueError('subsample must have two elements')
        subsample = tuple(subsample)

        if len(filter_dilation) != 2:
            raise ValueError('filter_dilation must have two elements')
        filter_dilation = tuple(filter_dilation)

        if isinstance(border_mode, integer_types):
            border_mode = (border_mode, border_mode)
        if isinstance(border_mode, tuple):
            pad_h, pad_w = map(int, border_mode)
            border_mode = (pad_h, pad_w)
        if border_mode == (0, 0):
            border_mode = 'valid'
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(border_mode))

        self.imshp = imshp
        self.kshp = kshp
        self.subsample = subsample
        self.border_mode = border_mode
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation
        self.group = group
        if not (isinstance(group, integer_types) and group > 0):
            raise ValueError('invalid group {}, which mush be a positive integer.'.format(group))

    def make_node(self, image, weight, bias=None):
        image = as_tensor_variable(image)
        weight = as_tensor_variable(weight)

        if image.type.ndim != 4:
            raise TypeError('Input image should be a 4-dim tensor.')

        if self.group is 1:
            assert weight.type.ndim == 4
            broadcastable = [image.type.broadcastable[0], weight.type.broadcastable[0], False, False]
        else:
            assert weight.type.ndim == 5
            broadcastable = [image.type.broadcastable[0], weight.type.broadcastable[1], False, False]

        dtype = image.type.dtype

        if bias is not None:
            bias = as_tensor_variable(bias)
            inputs = [image, weight, bias]
        else:
            inputs = [image, weight]
        return gof.Apply(self, inputs, [TensorType(dtype, broadcastable)()])

    def grad(self, inp, grads):
        assert len(inp) in [2, 3]

        if len(inp) is 3:
            image, weight, bias, = inp
        else:
            image, weight, = inp
            bias = None

        gz, = grads
        grad_out = AbstractConvGroupGrad(imshp=self.imshp,
                                         kshp=self.kshp,
                                         subsample=self.subsample,
                                         border_mode=self.border_mode,
                                         filter_flip=self.filter_flip,
                                         filter_dilation=self.filter_dilation,
                                         group=self.group)(image, gz, weight, bias)
        if len(grad_out) > 2:
            grad_image, grad_weight, grad_bias, = grad_out
            return grad_image, grad_weight, grad_bias
        else:
            grad_image, grad_weight, = grad_out
            return grad_image, grad_weight

    def perform(self, node, inp, out):
        if len(inp) == 2:
            image, weight = inp
        else:
            image, weight, bias = inp
        z, = out


class AbstractConvGroupGrad(gof.Op):
    __props__ = ('imshp', 'kshp', 'subsample', 'border_mode', 'filter_flip', 'filter_dilation', 'group')

    def __init__(self,
                 imshp=None, kshp=None, subsample=(1, 1),
                 border_mode='valid', filter_flip=False,
                 filter_dilation=(1, 1), group=1):

        super(AbstractConvGroupGrad, self).__init__()
        self.imshp = imshp
        self.kshp = kshp
        self.subsample = subsample
        self.border_mode = border_mode
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation
        self.group = group

    def make_node(self, image, gz, weight, bias=None):
        if image.type.ndim != 4:
            raise TypeError('Input image should be a 4-dim tensor.')

        if self.group is 1:
            assert weight.type.ndim == 4
            w_broadcastable = [gz.type.broadcastable[1], image.type.broadcastable[1], False, False]
            i_broadcastable = [gz.type.broadcastable[0], weight.type.broadcastable[1], False, False]
        else:
            assert weight.type.ndim == 5
            w_broadcastable = [False, gz.type.broadcastable[1], image.type.broadcastable[1], False, False]
            i_broadcastable = [gz.type.broadcastable[0], weight.type.broadcastable[2], False, False]

        dtype = weight.type.dtype

        if bias is not None:
            bias = as_tensor_variable(bias)
            inputs = [image, gz, weight, bias]
            b_broadcastable = [gz.type.broadcastable[1]]
            outputs = [TensorType(dtype, i_broadcastable)(), TensorType(dtype, w_broadcastable)(),
                       TensorType(dtype, b_broadcastable)()]
        else:
            inputs = [image, gz, weight]
            outputs = [TensorType(dtype, i_broadcastable)(), TensorType(dtype, w_broadcastable)()]

        return gof.Apply(self, inputs, outputs)

    def perform(self, node, inp, out):
        if len(inp) == 3:
            image, gz, weight, = inp
            grad_image, grad_weight, = out
        else:
            image, gz, weight, bias, = inp
            grad_image, grad_weight, grad_bias, = out
