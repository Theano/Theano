"""
contains an op for convolving input images with a set of weights by using MKL
library, which is a free dnn library provided by Intel.
"""
from __future__ import absolute_import, print_function, division
import theano
from theano import Apply
from theano import gof
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.blas import ldflags
from theano.sandbox.mkl import mkl_helper


class MKLConvBase(gof.Op):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample')

    def __init__(self, imshp=None, kshp=None, border_mode="valid", subsample=(1, 1), uniq_id=0):
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
        self.uniq_id = uniq_id

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
        return (1, 0, self.uniq_id)

    def c_support_code(self):
        ccode = mkl_helper.header_text()
        ccode += """
            #define _MKL_DEBUG_ 0
            #define dimension 4
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

    def c_support_code_apply(self, node, name):
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
            static int first_run = 1;
            static size_t imageSize[dimension] = {0}; //w, h, c, n
            static size_t imageStride[dimension] = {0};
            static size_t weightSize[dimension+1] = {0}; //w, h, c, n, group
            static size_t weightStride[dimension+1] = {0};
            static size_t zSize[dimension] = {0}; //w, h, c, n
            static size_t zStride[dimension] = {0};
            static size_t biasSize[1] = {0}; //w, h, c, n
            static size_t biasStride[1] = {0};
            static size_t groups = 1;
            static size_t fdimension = 0;

            ////////// debug only //////////
            static size_t _image_size;
            static size_t _weight_size;
            static size_t _z_size;
            ///////////////////////////////

            static size_t convStride[2] = {0};
            static int convPadding[2] = {0};

            static void *conv_res[dnnResourceNumber] = {0};
            static void *conv_res_bias[dnnResourceNumber] = {0};

            static void *image_buf = NULL;
            static void *image_buf_from_previous = NULL;
            static void *image_buf_to_previous = NULL;

            static void *z_buf = NULL;

            static void *weight_buf = NULL;
            static void *weight_buf_tmp = NULL;

            static void *bwdf2fwd_weight_buf = NULL;
            static void *bias_buf = NULL;
            static void *bias_buf_tmp = NULL;

            static void *gradz_buf = NULL;
            static void *gradz_buf_for_weight = NULL;
            static void *gradz_buf_for_bias = NULL;

            static dnnError_t err;
            static dnnPrimitive_t pConvolutionFwd = NULL;
            static dnnPrimitive_t pConvolutionBwdData = NULL;
            static dnnPrimitive_t pConvolutionBwdFilter = NULL;
            static dnnPrimitive_t pConvolutionBwdBias = NULL;

            static dnnPrimitive_t bwdf_weight_to_fwd_internal = NULL;
            static dnnPrimitive_t bwdf_wegith_to_usr = NULL;
            static dnnPrimitive_t bwdd_weight_to_bwdd_internal = NULL;

            static dnnLayout_t bwdf_weight_internal_layout = NULL;
            static dnnLayout_t image_user_layout = NULL;
            static dnnLayout_t weight_usr_layout = NULL;
            static dnnLayout_t z_user_layout = NULL;
            static dnnLayout_t image_internal_layout = NULL;
            static dnnLayout_t *image_internal_layout_buf = NULL;
            static dnnLayout_t image_internal_layout_from_previous = NULL;
            static dnnLayout_t weight_internal_layout = NULL;
            static dnnLayout_t z_internal_layout = NULL;
            static dnnLayout_t gradz_internal_layout = NULL;
            static dnnLayout_t gradz_internal_layout_for_weight = NULL;
            static dnnLayout_t gradz_internal_layout_for_bias = NULL;
            static dnnLayout_t fwd_weight_internal_layout = NULL;

            static dnnLayout_t bias_internal_layout = NULL;
            static dnnLayout_t bias_usr_layout = NULL;

            static dnnPrimitive_t image_to_internal = NULL;
            static dnnPrimitive_t weight_to_internal = NULL;
            static dnnPrimitive_t z_to_internal = NULL;
            static dnnPrimitive_t weight_from_internal = NULL;
            static dnnPrimitive_t image_from_internal = NULL;
            static dnnPrimitive_t z_from_internal = NULL;
            static dnnPrimitive_t internal_to_internal_image = NULL;
            static dnnPrimitive_t internal_to_internal_gradz_for_weight = NULL;
            static dnnPrimitive_t internal_to_internal_gradz_bias = NULL;
            static dnnPrimitive_t bias_to_internal = NULL;
            static dnnPrimitive_t bias_from_internal = NULL;
        """ % sub
        return ccode


class Conv2D(MKLConvBase):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_flip', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_flip=False, filter_dilation=(1, 1), uniq_id=0):
        super(Conv2D, self).__init__(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample, uniq_id=uniq_id)
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"
        ccode = """
            //std::cout << "in c_cleanup_code_struct " << std::endl;
            //FIXME, remove below sentence if it's handled by conversion Op
            //dnnDelete_%(precision)s(image_to_internal);
            //dnnDelete_%(precision)s(weight_to_internal);
            //dnnDelete_%(precision)s(z_to_internal);
            //dnnDelete_%(precision)s(z_from_internal);
            //dnnDelete_%(precision)s(weight_from_internal);
            //dnnDelete_%(precision)s(image_from_internal);
            //dnnLayoutDelete_%(precision)s(image_user_layout);
            //dnnLayoutDelete_%(precision)s(weight_usr_layout);
            //dnnLayoutDelete_%(precision)s(z_user_layout);
            //dnnLayoutDelete_%(precision)s(image_internal_layout);
            //dnnLayoutDelete_%(precision)s(weight_internal_layout);
            //dnnLayoutDelete_%(precision)s(z_internal_layout);
            //END
        """ % locals()
        return ccode

    def make_node(self, image, weight, bias=None):
        image = as_tensor_variable(image)
        weight = as_tensor_variable(weight)

        if image.type.ndim != 4:
            raise TypeError('image must be 4D tensor')
        if weight.type.ndim not in [4, 5]:
            raise TypeError('weight must be 4D or 5D tensor')

        broadcastable = [image.type.broadcastable[0], weight.type.broadcastable[0],
                         False, False]
        dtype = image.type.dtype

        if bias is not None:
            bias = as_tensor_variable(bias)
            inputs = [image, weight, bias]
        else:
            inputs = [image, weight]
        return Apply(self, inputs, [TensorType(dtype, broadcastable)()])

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

        if self.imshp is None:
            imshp = node.inputs[0].shape
        else:
            imshp = self.imshp
        in_n, in_c, in_h, in_w = imshp

        if self.kshp is None:
            kshp = node.inputs[1].shape
        else:
            kshp = self.kshp

        if node.inputs[1].type.ndim == 5:
            grp, k_n, k_c, k_h, k_w = kshp
            assert in_c == k_c * grp
        else:
            k_n, k_c, k_h, k_w = kshp
            grp = 1
            assert in_c == k_c

        outshp = self.infer_shape(node, [imshp, kshp])
        o_n, o_c, o_h, o_w = outshp[0]

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
            #if __DEBUG__
            std::cout << "conv forward, c_code start" << std::endl;
            #endif
            if (NULL == pConvolutionFwd) {
                convStride[0] = %(dW)s;
                convStride[1] = %(dH)s;
                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                imageSize[0] = %(in_w)s;  //w
                imageSize[1] = %(in_h)s;  //h
                imageSize[2] = %(in_c)s;  //c
                imageSize[3] = %(in_n)s;  //n
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
                zStride[0] = 1;
                zStride[1] = zSize[0];
                zStride[2] = zSize[0] * zSize[1];
                zStride[3] = zSize[0] * zSize[1] * zSize[2];

                //printf("Bias = :", %(withBias)s);
                if(%(withBias)s) {
                    biasSize[0] = %(o_c)s;
                    biasStride[0] = 1;
                }

                groups = %(grp)s;
                fdimension = dimension + (groups != 1);

                // Create conv forward primitive
                if (%(withBias)s) {
                    CHECK_ERR( dnnGroupsConvolutionCreateForwardBias_%(precision)s(&pConvolutionFwd, NULL,
                               dnnAlgorithmConvolutionDirect, groups, dimension, imageSize,
                               zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );
                } else {
                    CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&pConvolutionFwd, NULL,
                               dnnAlgorithmConvolutionDirect, groups, dimension, imageSize,
                               zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );
                }
            }

            if (NULL == weight_usr_layout) {
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&weight_usr_layout, fdimension, weightSize, weightStride), err );
            }
            if (NULL == image_internal_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&image_internal_layout,
                           pConvolutionFwd, dnnResourceSrc), err );
            }
            if (NULL == weight_internal_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&weight_internal_layout,
                           pConvolutionFwd, dnnResourceFilter), err );
            }
            if (NULL == z_internal_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&z_internal_layout,
                           pConvolutionFwd, dnnResourceDst), err );
            }

            if (%(withBias)s && NULL == bias_usr_layout) {
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&bias_usr_layout, 1, biasSize, biasStride), err );
            }

            // Prepare z array, only create once for passing internal layout and
            // internal data buffer for z data.
            if ( !(%(z)s && PyArray_NDIM(%(z)s) == 4)) {
                npy_intp out_dim[4];
                out_dim[0] = zSize[3];
                out_dim[1] = zSize[2];
                out_dim[2] = zSize[1];
                out_dim[3] = zSize[0];
                %(z)s = (PyArrayObject*)PyArray_ZEROS(dimension,
                                                        out_dim,
                                                        PyArray_TYPE(%(image)s),
                                                        0);
                if (NULL == %(z)s) {
                    PyErr_Format(PyExc_RuntimeError,
                                "Conv2D: Failed to allocate z of %%lld x %%lld x %%lld x %%lld",
                                (long long)out_dim[0], (long long)out_dim[1], (long long)out_dim[2], (long long)out_dim[3]);
                    %(fail)s
                }
            }

            #if _MKL_DEBUG_
                std::cout<<"x: "<<imageSize[3]<<" x "<<imageSize[2]<<" x "<<imageSize[1]<<" x "<<imageSize[0]<<std::endl;
                std::cout<<"weight: "<<weightSize[3]<<" x "<<weightSize[2]<<" x "<<weightSize[1]<<" x "<<weightSize[0]<<std::endl;
                std::cout<<"z: "<<zSize[3]<<" x "<<zSize[2]<<" x "<<zSize[1]<<" x "<<zSize[0]<<std::endl;
                std::cout<<"stride: "<<convStride[1]<<" x "<<convStride[0]<<std::endl;
                std::cout<<"padding: "<<convPadding[1]<<" x "<<convPadding[0]<<std::endl;
            #endif

            // get internal layout for input from previous Op
            image_internal_layout_from_previous = ((dnnLayout_t*)PyArray_DATA(%(image)s))[0];
            // get internal buffer for input from previous op
            image_buf_from_previous = ((void **)PyArray_DATA(%(image)s))[1];

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(image_internal_layout_from_previous, image_internal_layout)) {
                    if (NULL == internal_to_internal_image) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&internal_to_internal_image, image_internal_layout_from_previous, image_internal_layout), err );
                    }
                }
            }
            if (internal_to_internal_image) {
                if (NULL == image_buf) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&image_buf, image_internal_layout), err );
                }
                CHECK_ERR( dnnConversionExecute_%(precision)s(internal_to_internal_image, image_buf_from_previous, image_buf), err );
                image_internal_layout_buf = &image_internal_layout;
            } else {
                image_internal_layout_buf = &image_internal_layout_from_previous;
                image_buf = image_buf_from_previous;
            }
            conv_res[dnnResourceSrc] = image_buf;

            #if _MKL_DEBUG_
                std::cout<<"x internal layout = @"<<*image_internal_layout_buf<<std::endl;
                std::cout<<"x internal buffer = @"<<image_buf<<std::endl;
            #endif

            weight_buf = (%(dtype)s*)PyArray_DATA(%(weight)s);
            if(%(withBias)s) {
                bias_buf = (%(dtype)s*)PyArray_DATA(%(bias)s);
            }

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(weight_usr_layout, weight_internal_layout)) {
                    if (NULL == weight_to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&weight_to_internal, weight_usr_layout, weight_internal_layout), err );
                    }

                    if (%(withBias)s && !dnnLayoutCompare_%(precision)s(bias_usr_layout, bias_internal_layout)) {
                        if (NULL == bias_to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&bias_to_internal, bias_usr_layout, bias_internal_layout), err );
                        }
                    }

                }
            }

            #if __SUPPORT_USER_PARAMS__
                if (weight_to_internal) {
                    if (NULL == weight_buf_tmp) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buf_tmp, weight_internal_layout), err );
                    }
                    CHECK_ERR( dnnConversionExecute_%(precision)s(weight_to_internal, weight_buf, weight_buf_tmp), err );
                    conv_res[dnnResourceFilter] = weight_buf_tmp;
                } else {
                    conv_res[dnnResourceFilter] = weight_buf;
                }

                if (%(withBias)s && bias_to_internal) {
                    if (NULL == bias_buf_tmp) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bias_buf_tmp, bias_internal_layout), err );
                    }
                    CHECK_ERR( dnnConversionExecute_%(precision)s(bias_to_internal,bias_buf, bias_buf_tmp), err );
                    conv_res[dnnResourceBias] = bias_buf_tmp;
                } else {
                    conv_res[dnnResourceBias] = bias_buf;
                }
            #else //__SUPPORT_USER_PARAMS__
                if (1 == first_run) {
                    if (weight_to_internal) {
                        if (NULL == weight_buf_tmp) {
                            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buf_tmp, weight_internal_layout), err );
                        }
                        CHECK_ERR( dnnConversionExecute_%(precision)s(weight_to_internal, weight_buf, weight_buf_tmp), err );
                        memcpy(weight_buf, weight_buf_tmp, dnnLayoutGetMemorySize_%(precision)s(weight_internal_layout));
                    }
                }

                if (%(withBias)s && bias_to_internal) {
                    if (NULL == bias_buf_tmp) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bias_buf_tmp, bias_internal_layout), err );
                    }
                    CHECK_ERR( dnnConversionExecute_%(precision)s(bias_to_internal, bias_buf, bias_buf_tmp), err );
                    memcpy(bias_buf, bias_buf_tmp, dnnLayoutGetMemorySize_%(precision)s(bias_internal_layout));
               }

               conv_res[dnnResourceFilter] = weight_buf;
               if(%(withBias)s) conv_res[dnnResourceBias] = bias_buf;
            #endif //__SUPPORT_USER_PARAMS__

            //Allocate internal buffer for z data, only apply once
            if (NULL == z_buf) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&z_buf, z_internal_layout), err );
            }
            conv_res[dnnResourceDst] = z_buf;

            #if __DEBUG__
                _image_size = dnnLayoutGetMemorySize_%(precision)s(*image_internal_layout_buf);
                _weight_size = dnnLayoutGetMemorySize_%(precision)s(weight_internal_layout);
                _z_size = dnnLayoutGetMemorySize_%(precision)s(z_internal_layout);
                bias_size = dnnLayoutGetMemorySize_%(precision)s(bias_internal_layout);
                std::cout << "forward, pConvolution = @" << pConvolutionFwd << std::endl;
                std::cout<<"x size: "<<imageSize[3]<<" x "<<imageSize[2]<<" x "<<imageSize[1]<<" x "<<imageSize[0]<<", acutal size: "<<_image_size<<std::endl;
                std::cout<<"x buffer ptr: "<<image_buf<<std::endl;
                std::cout<<"weight size: "<<weightSize[3]<<" x "<<weightSize[2]<<" x "<<weightSize[1]<<" x "<<weightSize[0]<<", actual size: "<<_weight_size<<std::endl;
                std::cout<<"weight buffer ptr: "<<weight_buf<<std::endl;
                std::cout<<"z size: "<<zSize[3]<<" x "<<zSize[2]<<" x "<<zSize[1]<<" x "<<zSize[0]<<", actual size: "<<_z_size<<std::endl;
                std::cout << "bias_size = " << bias_size << std::endl;
                std::cout<<"z buffer ptr: "<<z_buf<<std::endl;
                std::cout << "forward, pConvolution = @" << pConvolutionFwd << std::endl;
                std::cout << "forward, conv_res[Src] = @" << conv_res[dnnResourceSrc] << std::endl;
                std::cout << "forward, conv_res[Filter] = @" << conv_res[dnnResourceFilter] << std::endl;
                std::cout << "forward, conv_res[Dst] = @" << conv_res[dnnResourceDst] << std::endl;
            #endif

            //Execute convolution forward pass
            CHECK_ERR( dnnExecute_%(precision)s(pConvolutionFwd, (void**)conv_res), err );

            // return the internal layout and data buffer for z directly.
            ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = z_internal_layout;
            ((void**)PyArray_DATA(%(z)s))[1] = z_buf;

            first_run = 0;
            #if __DEBUG__
                printf(\"convFw z:%%x, %%x, %%x\\n\",%(z)s,z_internal_layout,z_buf);
                std::cout <<"conv forward, z_internal_layout: @" <<z_internal_layout<<std::endl;
                std::cout <<"conv forward, z_buf: @" <<z_buf<<std::endl;
                std::cout << "forward, c_code end\\n" << std::endl;
            #endif
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
                                  kshp=self.kshp)(image, weights, gz)

        dlist = ConvGradWeights(border_mode=self.border_mode,
                                subsample=self.subsample,
                                imshp=self.imshp,
                                kshp=self.kshp)(image, weights, gz, bias)

        if len(dlist) > 1:
            d_weights, d_bias = dlist
            return d_images, d_weights, d_bias
        else:
            d_weights = dlist
            return d_images, d_weights


class ConvGradInputs(MKLConvBase):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_flip', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_flip=True, filter_dilation=(1, 1), uniq_id=0):
        super(ConvGradInputs, self).__init__(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample, uniq_id=uniq_id)
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation

    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"
        ccode = """
            //std::cout << "in gradI c_cleanup_code_struct " << std::endl;
            //FIXME, remove below sentence if it's handled by conversion Op
            //dnnDelete_%(precision)s(image_to_internal);
            //dnnDelete_%(precision)s(weight_to_internal);
            //dnnDelete_%(precision)s(z_to_internal);
            //dnnDelete_%(precision)s(z_from_internal);
            //dnnDelete_%(precision)s(weight_from_internal);
            //dnnDelete_%(precision)s(image_from_internal);
            //dnnLayoutDelete_%(precision)s(image_user_layout);
            //dnnLayoutDelete_%(precision)s(weight_usr_layout);
            //dnnLayoutDelete_%(precision)s(z_user_layout);
            //dnnLayoutDelete_%(precision)s(image_internal_layout);
            //dnnLayoutDelete_%(precision)s(weight_internal_layout);
            //dnnLayoutDelete_%(precision)s(z_internal_layout);
            //END
        """ % locals()
        return ccode

    def make_node(self, image, weight, gradz):
        image = as_tensor_variable(image)
        weight = as_tensor_variable(weight)
        gradz = as_tensor_variable(gradz)
        if weight.type.ndim not in [4, 5]:
            raise TypeError('weight must be 4D or 5D tensor')
        if gradz.type.ndim != 4:
            raise TypeError('gradz must be 4D tensor')

        broadcastable = [gradz.type.broadcastable[0], weight.type.broadcastable[1],
                         False, False]
        dtype = weight.type.dtype
        return Apply(self, [image, weight, gradz], [TensorType(dtype, broadcastable)()])

    def c_code(self, node, name, inp, out_, sub):
        image, weights, gradz = inp
        imagegrad, = out_
        if self.imshp is None:
            imshp = node.inputs[0].shape
        else:
            imshp = self.imshp
        in_n, in_c, in_h, in_w = imshp

        if self.kshp is None:
            kshp = node.inputs[1].shape
        else:
            kshp = self.kshp

        if node.inputs[1].type.ndim == 5:
            grp, k_n, k_c, k_h, k_w = kshp
            assert in_c == k_c * grp
        else:
            grp = 1
            k_n, k_c, k_h, k_w = kshp
            assert in_c == k_c

        if node.inputs[1].type.ndim == 5:
            tshp = [kshp[1] * kshp[0], kshp[2] * kshp[0], kshp[3], kshp[4]]
        else:
            tshp = [kshp[0], kshp[1], kshp[2], kshp[3]]

        outshp = get_conv_output_shape(imshp, tshp, self.border_mode, self.subsample)

        o_n, o_c, o_h, o_w = outshp

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
            #if __DEBUG__
                std::cout << "gradInput, c_code start " << std::endl;
            #endif
            if (NULL == pConvolutionBwdData) {
                convStride[0] = %(dW)s;
                convStride[1] = %(dH)s;

                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                imageSize[0] = %(in_w)s;  //w
                imageSize[1] = %(in_h)s;  //h
                imageSize[2] = %(in_c)s;  //c
                imageSize[3] = %(in_n)s;  //n
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
                zStride[0] = 1;
                zStride[1] = zSize[0];
                zStride[2] = zSize[0] * zSize[1];
                zStride[3] = zSize[0] * zSize[1] * zSize[2];

                groups = %(grp)s;
                fdimension = dimension + (groups != 1);

                // Create conv gradInput primitive
                CHECK_ERR( dnnGroupsConvolutionCreateBackwardData_%(precision)s(&pConvolutionBwdData, NULL,
                           dnnAlgorithmConvolutionDirect, groups, dimension, imageSize,
                           zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );
            }
            if (NULL == weight_internal_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&weight_internal_layout,
                           pConvolutionBwdData, dnnResourceFilter), err );
            }
            if (NULL == image_internal_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&image_internal_layout,
                           pConvolutionBwdData, dnnResourceDiffSrc), err );
            }

            if (NULL == pConvolutionFwd) {
                // Create conv forward primitive
                    CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&pConvolutionFwd, NULL,
                               dnnAlgorithmConvolutionDirect, groups, dimension, imageSize,
                               zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );
            }
            if(NULL == fwd_weight_internal_layout) {
                CHECK_ERR(dnnLayoutCreateFromPrimitive_%(precision)s(&fwd_weight_internal_layout,
                          pConvolutionFwd, dnnResourceFilter), err );
            }

            if ( !(%(imagegrad)s)) {
                %(imagegrad)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(image)s),
                                                               PyArray_DIMS(%(image)s),
                                                               PyArray_TYPE(%(image)s),
                                                               0);
                if (NULL == %(imagegrad)s) {
                    PyErr_Format(PyExc_RuntimeError,
                                "conv_gradInput: Failed to allocate image of %%lld x %%lld x %%lld x %%lld",
                                (long long)(PyArray_DIMS(%(image)s))[0], (long long)(PyArray_DIMS(%(image)s))[1],
                                (long long)(PyArray_DIMS(%(image)s))[2], (long long)(PyArray_DIMS(%(image)s))[3]);
                    %(fail)s
                }
           }

           //weight use its own buffer
           weight_buf = (%(dtype)s*)PyArray_DATA(%(weight)s);

           //get internal layout for gradz from previous Op
           gradz_internal_layout = ((dnnLayout_t*)PyArray_DATA(%(gradz)s))[0];
           //get internal buffer for gradz from previous op
           gradz_buf = ((void **)PyArray_DATA(%(gradz)s))[1];

           conv_res[dnnResourceDiffDst] = gradz_buf;

           #if __SUPPORT_USER_PARAMS__
               if(NULL == weight_usr_layout) {
                   CHECK_ERR( dnnLayoutCreate_%(precision)s(&weight_usr_layout, fdimension, weightSize, weightStride), err );
               }
               if (weight_to_internal) {
                   if(NULL == weight_buf_tmp) {
                       CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buf_tmp, weight_internal_layout), err );
                   }
                   CHECK_ERR( dnnConversionExecute_%(precision)s(weight_to_internal, weight_buf, weight_buf_tmp), err );
               } else {
                   weight_buf_tmp = weight_buf;
               }
           #else
               if (1 == first_run) {
                   if (!dnnLayoutCompare_%(precision)s(fwd_weight_internal_layout, weight_internal_layout)) {
                       if(NULL == bwdd_weight_to_bwdd_internal) {
                           CHECK_ERR( dnnConversionCreate_%(precision)s(&bwdd_weight_to_bwdd_internal, fwd_weight_internal_layout, weight_internal_layout), err );
                       }
                   }
               }
               if (bwdd_weight_to_bwdd_internal) {
                   if(NULL == weight_buf_tmp) {
                       CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buf_tmp, weight_internal_layout), err );
                   }
                   CHECK_ERR( dnnConversionExecute_%(precision)s(bwdd_weight_to_bwdd_internal, weight_buf, weight_buf_tmp), err );
               } else {
                   weight_buf_tmp = weight_buf;
               }
           #endif

           conv_res[dnnResourceFilter] = weight_buf_tmp;

           //Allocate internal buffer for imagegrad data
           if (NULL == image_buf) {
               CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&image_buf, image_internal_layout), err );
           }
           conv_res[dnnResourceDiffSrc] = image_buf;

           //Execute convolution gradInput pass
           CHECK_ERR( dnnExecute_%(precision)s(pConvolutionBwdData, (void**)conv_res), err );

           //get image_internal_layout from forward pass, pass the data buffer match previous layout.
           image_internal_layout_from_previous = ((dnnLayout_t*)PyArray_DATA(%(image)s))[0];

           //image int2int cvt
           if (1 == first_run) {
               if (!dnnLayoutCompare_%(precision)s(image_internal_layout, image_internal_layout_from_previous)) {
                   if (NULL == internal_to_internal_image) {
                       CHECK_ERR( dnnConversionCreate_%(precision)s(&internal_to_internal_image, image_internal_layout, image_internal_layout_from_previous), err );
                   }
               }
           }
           if (internal_to_internal_image) {
               if (NULL == image_buf_to_previous) {
                   CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&image_buf_to_previous, image_internal_layout_from_previous), err );
               }
               CHECK_ERR( dnnConversionExecute_%(precision)s(internal_to_internal_image, image_buf, image_buf_to_previous), err );
           } else {
               image_buf_to_previous = image_buf;
           }

           ((dnnLayout_t*)PyArray_DATA(%(imagegrad)s))[0] = image_internal_layout_from_previous;
           ((void**)PyArray_DATA(%(imagegrad)s))[1] = image_buf_to_previous;

           first_run = 0;
        """ % sub
        return ccode


class ConvGradWeights(MKLConvBase):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_flip', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_flip=False, filter_dilation=(1, 1), uniq_id=0):
        super(ConvGradWeights, self).__init__(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample, uniq_id=uniq_id)
        self.filter_flip = filter_flip
        self.filter_dilation = filter_dilation

    def make_node(self, image, weight, gradz, bias=None):
        image = as_tensor_variable(image)
        weight = as_tensor_variable(weight)
        gradz = as_tensor_variable(gradz)

        if image.type.ndim != 4:
            raise TypeError('image must be 4D tensor')
        if weight.type.ndim not in [4, 5]:
            raise TypeError('weightmust be 4D or 5D tensor')
        if gradz.type.ndim != 4:
            raise TypeError('gradz must be 4D tensor')

        if bias is not None:
            bias = as_tensor_variable(bias)
            inputs = [image, weight, gradz, bias]
            outputs = [weight.type(), bias.type()]
        else:
            inputs = [image, weight, gradz]
            outputs = [weight.type()]

        return Apply(self, inputs, outputs)

    def c_code(self, node, name, inp, out_, sub):
        if len(inp) > 3:
            image, weight, gradz, bias = inp
            weightgrad, biasgrad = out_
        else:
            image, weight, gradz = inp
            bias = None
            weightgrad, = out_

        if self.imshp is None:
            imshp = node.inputs[0].shape
        else:
            imshp = self.imshp
        in_n, in_c, in_h, in_w = imshp

        if self.kshp is None:
            kshp = node.inputs[1].shape
        else:
            kshp = self.kshp

        if node.inputs[1].type.ndim == 5:
            grp, k_n, k_c, k_h, k_w = kshp
            tshp = [kshp[1] * kshp[0], kshp[2] * kshp[0], kshp[3], kshp[4]]
            assert in_c == k_c * grp
        else:
            k_n, k_c, k_h, k_w = kshp
            grp = 1
            tshp = [kshp[0], kshp[1], kshp[2], kshp[3]]
            assert in_c == k_c

        if bias is not None:
            sub['bias'] = bias
            sub['biasgrad'] = biasgrad
            withBias = 1
        else:
            withBias = 0
        sub['withBias'] = withBias

        outshp = get_conv_output_shape(imshp, tshp, self.border_mode, self.subsample)

        o_n, o_c, o_h, o_w = outshp

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
            ////bwdfilter related
            if (NULL == pConvolutionBwdFilter) {
                convStride[0] = %(dW)s;
                convStride[1] = %(dH)s;
                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                imageSize[0] = %(in_w)s;  //w
                imageSize[1] = %(in_h)s;  //h
                imageSize[2] = %(in_c)s;  //c
                imageSize[3] = %(in_n)s;  //n
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
                zStride[0] = 1;
                zStride[1] = zSize[0];
                zStride[2] = zSize[0] * zSize[1];
                zStride[3] = zSize[0] * zSize[1] * zSize[2];

                if( %(withBias)s ) {
                    biasSize[0] = %(o_c)s;
                    biasStride[0] = 1;
                }

                groups = %(grp)s;
                fdimension = dimension + (groups != 1);

                // Create conv backward primitive
                CHECK_ERR( dnnGroupsConvolutionCreateBackwardFilter_%(precision)s(&pConvolutionBwdFilter, NULL,
                           dnnAlgorithmConvolutionDirect, groups, dimension, imageSize,
                           zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );
            }
            if (NULL == bwdf_weight_internal_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&bwdf_weight_internal_layout,
                           pConvolutionBwdFilter, dnnResourceDiffFilter), err );
            }

            if (NULL == image_internal_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&image_internal_layout,
                           pConvolutionBwdFilter, dnnResourceSrc), err );
            }

            if (NULL == gradz_internal_layout_for_weight) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&gradz_internal_layout_for_weight,
                           pConvolutionBwdFilter, dnnResourceDiffDst), err );
            }

            // create forward primitive here to get forward internal layout
            if (NULL == pConvolutionFwd) {
                if ( %(withBias)s ) {
                    CHECK_ERR( dnnGroupsConvolutionCreateForwardBias_%(precision)s(&pConvolutionFwd, NULL,
                               dnnAlgorithmConvolutionDirect, groups, dimension, imageSize,
                               zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );
                } else {
                    CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&pConvolutionFwd, NULL,
                               dnnAlgorithmConvolutionDirect, groups, dimension, imageSize,
                               zSize, weightSize, convStride, convPadding, dnnBorderZeros), err );
                }
            }

            if (NULL == fwd_weight_internal_layout) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&fwd_weight_internal_layout,
                           pConvolutionFwd, dnnResourceFilter), err );
            }

            //bwdbias related
            if( %(withBias)s ) {
                if (NULL == pConvolutionBwdBias) {
                    CHECK_ERR ( dnnGroupsConvolutionCreateBackwardBias_%(precision)s(&pConvolutionBwdBias, NULL,
                                dnnAlgorithmConvolutionDirect, groups, dimension, zSize), err );
                }
                if (NULL == bias_internal_layout) {
                    CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&bias_internal_layout,
                               pConvolutionBwdBias, dnnResourceDiffBias), err );
                }
                if (NULL == gradz_internal_layout_for_bias) {
                    CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&gradz_internal_layout_for_bias,
                               pConvolutionBwdBias, dnnResourceDiffDst), err );
                }
            }

            //// Prepare weightgrad array
            if ( !(%(weightgrad)s) ) {
                %(weightgrad)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(weight)s),
                                                               PyArray_DIMS(%(weight)s),
                                                               PyArray_TYPE(%(weight)s),
                                                               0);
                if (NULL == %(weightgrad)s) {
                    /*
                    PyErr_Format(PyExc_RuntimeError,
                            "conv_gradWeight: Failed to allocate weight of %%lld x %%lld x %%lld x %%lld x %%lld",
                            (long long)(PyArray_DIMS(%(weight)s))[0], (long long)(PyArray_DIMS(%(weight)s))[1],
                            (long long)(PyArray_DIMS(%(weight)s))[2], (long long)(PyArray_DIMS(%(weight)s))[3]);
                    */
                }
            }

            weight_buf = (%(dtype)s*)PyArray_DATA(%(weightgrad)s);
            """ % sub

        if bias is not None:
            ccode += """
            if (NULL == %(biasgrad)s) {
                %(biasgrad)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(bias)s),
                                                             PyArray_DIMS(%(bias)s),
                                                             PyArray_TYPE(%(bias)s),
                                                             0);
                if (NULL == %(biasgrad)s) {
                    PyErr_Format(PyExc_RuntimeError, "conv_backward: Failed to allocate bias of %%lld",
                                (long long)PyArray_NDIM(%(bias)s));
                    %(fail)s
                }
            }
            bias_buf = (%(dtype)s*)PyArray_DATA(%(biasgrad)s);
            """ % sub

        ccode += """
            // get internal layout for input from previous Op
            image_internal_layout_from_previous = ((dnnLayout_t*)PyArray_DATA(%(image)s))[0];
            // get internal buffer for input from previous op
            image_buf_from_previous = ((void **)PyArray_DATA(%(image)s))[1];

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(image_internal_layout_from_previous, image_internal_layout)) {
                    if (NULL == internal_to_internal_image) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&internal_to_internal_image, image_internal_layout_from_previous, image_internal_layout), err );
                    }
                }
            }

            if (internal_to_internal_image) {
                if (NULL == image_buf) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&image_buf, image_internal_layout), err );
                }
                CHECK_ERR( dnnConversionExecute_%(precision)s(internal_to_internal_image, image_buf_from_previous, image_buf), err );
                image_internal_layout_buf = &image_internal_layout;
            } else {
                image_internal_layout_buf = &image_internal_layout_from_previous;
                image_buf = image_buf_from_previous;
            }

            // get internal layout for gradz from previous Op
            gradz_internal_layout = ((dnnLayout_t*)PyArray_DATA(%(gradz)s))[0];
            // get internal buffer for gradz from previous op
            gradz_buf = ((void **)PyArray_DATA(%(gradz)s))[1];

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(gradz_internal_layout, gradz_internal_layout_for_weight)) {
                    if (NULL == internal_to_internal_gradz_for_weight) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&internal_to_internal_gradz_for_weight, gradz_internal_layout, gradz_internal_layout_for_weight), err );
                    }
                }
            }
            if (internal_to_internal_gradz_for_weight) {
                if (NULL == gradz_buf_for_weight) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&gradz_buf_for_weight, gradz_internal_layout_for_weight), err );
                }
                CHECK_ERR( dnnConversionExecute_%(precision)s(internal_to_internal_gradz_for_weight, gradz_buf, gradz_buf_for_weight), err );
            } else {
                gradz_buf_for_weight = gradz_buf;
            }

            conv_res[dnnResourceSrc] = image_buf;
            conv_res[dnnResourceDiffDst] = gradz_buf_for_weight;

            if (%(withBias)s) {
                if (1 == first_run) {
                    if (!dnnLayoutCompare_%(precision)s(gradz_internal_layout, gradz_internal_layout_for_bias)) {
                        if (NULL == internal_to_internal_gradz_bias) {
                            CHECK_ERR( dnnConversionCreate_%(precision)s(&internal_to_internal_gradz_bias, gradz_internal_layout, gradz_internal_layout_for_bias), err );
                        }
                    }
                }
                if (internal_to_internal_gradz_bias) {
                    if (NULL == gradz_buf_for_bias) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&gradz_buf_for_bias, gradz_internal_layout_for_bias), err );
                    }
                    CHECK_ERR( dnnConversionExecute_%(precision)s(internal_to_internal_gradz_bias, gradz_buf, gradz_buf_for_bias), err );
                } else {
                    gradz_buf_for_bias = gradz_buf;
                }
            }

            //Allocate internal buffer for weightgrad data
            if (NULL == weight_buf_tmp) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&weight_buf_tmp, bwdf_weight_internal_layout), err );
            }
            conv_res[dnnResourceDiffFilter] = weight_buf_tmp;

            //Execute convolution gradweight pass
            CHECK_ERR( dnnExecute_%(precision)s(pConvolutionBwdFilter, (void**)conv_res), err );

            if (%(withBias)s) {
                conv_res_bias[dnnResourceDiffDst] = gradz_buf_for_bias;
                conv_res_bias[dnnResourceDiffBias] = bias_buf;

                //Execute convolution gradbias pass
                CHECK_ERR( dnnExecute_%(precision)s(pConvolutionBwdBias, (void**)conv_res_bias), err );
            }

            //weight bwd -> fwd cvt
            if(!dnnLayoutCompare_%(precision)s(bwdf_weight_internal_layout, fwd_weight_internal_layout)) {
                if (NULL == bwdf_weight_to_fwd_internal) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&bwdf_weight_to_fwd_internal, bwdf_weight_internal_layout, fwd_weight_internal_layout), err );
                }
            }

            #if __SUPPORT_USER_PARAMS__
                if(NULL == weight_usr_layout) {
                    dnnLayoutCreate_%(precision)s(&weight_usr_layout, fdimension, weightSize, weightStride);
                    //printf(\"gradweight, weightstride: %%d, %%d, %%d, %%d\\n\", weightStride[0], weightStride[1], weightStride[2], weightStride[3]);
                }

                if(%(withBias)s && NULL == bias_usr_layout) {
                    dnnLayoutCreate_%(precision)s(&bias_usr_layout, 1, biasSize, biasStride);
                }

                if (bwdf_weight_to_fwd_internal) {
                    if (NULL == bwdf2fwd_weight_buf) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bwdf2fwd_weight_buf, fwd_weight_internal_layout), err );
                    }
                    CHECK_ERR( dnnConversionExecute_%(precision)s(bwdf_weight_to_fwd_internal, weight_buf_tmp, bwdf2fwd_weight_buf), err );
                } else {
                    bwdf2fwd_weight_buf = weight_buf_tmp;
                }

                if (NULL == bwdf_wegith_to_usr) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&bwdf_wegith_to_usr, fwd_weight_internal_layout, weight_usr_layout), err );
                }
                dnnConversionExecute_%(precision)s(bwdf_wegith_to_usr, bwdf2fwd_weight_buf, weight_buf);

                //no need to do conversion for bias
                if (%(withBias)s) {
                    if(NULL == bias_buf_tmp) {
                        dnnAllocateBuffer_%(precision)s((void**)&bias_buf_tmp, bias_usr_layout);
                    }
                    if(NULL == bias_from_internal) {
                         dnnConversionCreate_%(precision)s(&bias_from_internal, bias_internal_layout, bias_usr_layout);
                    }
                    dnnConversionExecute_%(precision)s(bias_from_internal, bias_buf, bias_buf_tmp);
                    memcpy(bias_buf, bias_buf_tmp, dnnLayoutGetMemorySize_%(precision)s(bias_usr_layout));
                }
            #else
                if (bwdf_weight_to_fwd_internal) {
                    if (NULL == bwdf2fwd_weight_buf) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&bwdf2fwd_weight_buf, fwd_weight_internal_layout), err );
                    }
                    CHECK_ERR( dnnConversionExecute_%(precision)s(bwdf_weight_to_fwd_internal, weight_buf_tmp, weight_buf), err );
                }
                else {
                    memcpy(weight_buf, weight_buf_tmp, dnnLayoutGetMemorySize_%(precision)s(fwd_weight_internal_layout));
                }
            #endif  //__SUPPORT_USER_PARAMS__

            first_run = 0;
        """ % sub
        return ccode
