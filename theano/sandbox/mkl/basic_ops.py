import theano.tensor as T
from theano.gof import Apply, Op
from theano.tensor.blas import ldflags
from theano.sandbox.mkl import mkl_helper
from theano.gradient import DisconnectedType
from theano.tensor.nnet.abstract_conv import get_conv_output_shape


class MKLOp(Op):
    def __init__(self, uniq_id=0):
        self.uniq_id = uniq_id

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(MKLOp, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        ccode = mkl_helper.header_text()
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

            static dnnError_t err;
            static int first_run = 1;
            static void* internal_buf = NULL;
            static void* user_buf = NULL;
            static dnnLayout_t layout_internal = NULL;
            static dnnLayout_t layout_usr = NULL;
            static dnnPrimitive_t to_internal = NULL;
            static dnnPrimitive_t from_internal = NULL;
            static dnnPrimitive_t primitive = NULL;
            void *convert_resources[dnnResourceNumber];
            size_t bottomSize[DIMENSION] = {0};
            size_t bottomStride[DIMENSION] = {0};
        """
        return ccode

    def c_code_cache_version(self):
        return (1, 0, self.uniq_id)


class U2IPool(MKLOp):
    __props__ = ('ignore_border', 'mode', 'uniq_id')

    def __init__(self, ignore_border=False, mode='max', uniq_id=0):
        super(U2IPool, self).__init__(uniq_id)
        self.ignore_border = ignore_border
        self.mode = mode

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

    def make_node(self, x, ws, stride=None, pad=(0, 0)):
        x = T.as_tensor_variable(x)
        if stride is None:
            stride = ws

        ws = T.as_tensor_variable(ws)
        stride = T.as_tensor_variable(stride)
        pad = T.as_tensor_variable(pad)

        broad = x.broadcastable[:2] + (False, False)
        out = T.TensorType(x.dtype, broad)
        return Apply(self, [x, ws, stride, pad], [out()])

    def grad(self, inp, grads):
        x, ws, stride, pad = inp
        gz, = grads
        disc = [DisconnectedType()() for i in inp[1:]]

        return [U2IGrad(uniq_id=self.uniq_id)(x, gz)] + disc

    # def c_cleanup_code_struct(self, node, name):
    #    pass

    def c_code(self, node, name, inp, out, sub):
        x, ws, stride, pad = inp
        z, = out

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise Exception("Type %s is not supported!" %
                            node.inputs[0].type.dtype)

        fail = sub['fail']

        ignore_border = self.ignore_border
        if 'max' == self.mode:
            algo = "dnnAlgorithmPoolingMax"
        elif self.mode.startswith('average'):
            algo = "dnnAlgorithmPoolingAvg"
        else:
            raise NotImplementedError('%s is not implemented' % self.mode)

        ccode = """
            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3];  //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2];  //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1];  //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0];  //n
                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                size_t kernel_h = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
                size_t kernel_w = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
                size_t stride_h = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
                size_t stride_w = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
                size_t pad_h = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
                size_t pad_w = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));

                size_t kernelSize[2] = {kernel_w, kernel_h};
                size_t kernelStride[2] = {stride_w, stride_h};
                int inputOffset[2] = {pad_w, pad_h};

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr, DIMENSION, bottomSize, bottomStride), err );

                CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&primitive, NULL, %(algo)s,
                           layout_usr, kernelSize, kernelStride, inputOffset, dnnBorderZeros), err );

                //create internal layout
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal, primitive, dnnResourceSrc), err );

                if (!dnnLayoutCompare_%(precision)s(layout_usr, layout_internal)) {
                    if(NULL == to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_usr, layout_internal), err );
                    }
                }
            }

            if (NULL == %(z)s) {
                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION,
                                                      PyArray_DIMS(%(x)s),
                                                      PyArray_TYPE(%(x)s),
                                                      0);
                if(NULL == %(z)s) {
                    %(fail)s
                }

                if (NULL == internal_buf) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&internal_buf, layout_internal), err );
                }
            }

            if (to_internal) {
                convert_resources[dnnResourceFrom] = (PyArray_DATA(%(x)s));
                convert_resources[dnnResourceTo] = (void *)(internal_buf);

                CHECK_ERR( dnnExecute_%(precision)s(to_internal, convert_resources), err );
            } else {
                internal_buf = (PyArray_DATA(%(x)s));
            }

            if(layout_internal != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0])
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_internal;
            if(internal_buf != ((void**)PyArray_DATA(%(z)s))[1])
                ((void**)PyArray_DATA(%(z)s))[1] = internal_buf;

            first_run = 0;

            #ifdef _MKL_DEBUG_
                printf(\"U2IPool: from_buffer %%x to_buffer %%x\\n\",convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
            #endif
        """ % locals()
        return ccode


class I2U(MKLOp):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        super(I2U, self).__init__(uniq_id)

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
        return hash(type(self))

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        out = x.type()
        return Apply(self, [x], [out])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [I2UGrad(uniq_id=self.uniq_id)(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
            x_item_size = 4
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
            x_item_size = 8
        else:
            raise Exception("Type %s is not supported!" %
                            node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            int status = 0;
            if (NULL == %(z)s) {
                %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),
                                                      PyArray_DIMS(%(x)s),
                                                      PyArray_TYPE(%(x)s),
                                                      0);
                if (NULL == %(z)s) {
                    %(fail)s
                }

                int ndim = (int)PyArray_NDIM(%(x)s);
                size_t *bottom_size = (size_t *)malloc(ndim * sizeof(size_t));
                size_t *out_stride = (size_t *)malloc(ndim * sizeof(size_t));
                if(NULL == bottom_size || NULL == out_stride) {
                    printf(\"ERROR: malloc buffer in I2U \\n\");
                    exit(1);
                }

                npy_intp dataSize = 1;
                for(int i = 0; i < ndim; i++) {
                    bottom_size[i] = (size_t)PyArray_DIMS(%(z)s)[ndim-i-1];
                    out_stride[i] = (size_t)PyArray_STRIDES(%(z)s)[ndim-i-1] / %(x_item_size)s;
                    dataSize = dataSize * bottom_size[i];
                }

                //create usr layerout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr,
                                                     ndim, bottom_size,
                                                     out_stride), err );

                free(bottom_size);
                free(out_stride);

                //Get layerout and internal buffer from input.
                layout_internal = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0];
                internal_buf = ((void**)PyArray_DATA(%(x)s))[1];

                CHECK_ERR( dnnConversionCreate_%(precision)s(&from_internal, layout_internal, layout_usr), err );
            }

            //FIXME, compare internal and user layout, then decides whether to do the conversion

            convert_resources[dnnResourceTo] = reinterpret_cast<void *>(PyArray_DATA(%(z)s));
            convert_resources[dnnResourceFrom] = reinterpret_cast<void *>(internal_buf);

            //cvt
            CHECK_ERR( dnnExecute_%(precision)s(from_internal, convert_resources), err );
        """ % locals()

        return ccode


class U2IRelu(MKLOp):
    __props__ = ('slope', 'uniq_id')

    def __init__(self, slope=1, uniq_id=0):
        super(U2IRelu, self).__init__(uniq_id)
        self.slope = slope

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
        return hash(self.slope)

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x):
        x = T.as_tensor_variable(x)

        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [U2IGrad(uniq_id=self.uniq_id)(x, gz)]

    # def c_cleanup_code_struct(self, node, name):
    #    pass

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        slope = self.slope
        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise Exception("Type %s is not supported!" %
                            node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3];  //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2];  //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1];  //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0];  //n
                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr, DIMENSION, bottomSize, bottomStride), err );

                CHECK_ERR( dnnReLUCreateForward_%(precision)s(&primitive, NULL, layout_usr, %(slope)s), err );

                //create internal layout
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal, primitive, dnnResourceSrc), err );

                if (!dnnLayoutCompare_%(precision)s(layout_usr, layout_internal)) {
                    if(NULL == to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_usr, layout_internal), err );
                    }
                }
            }

            if (NULL == %(z)s) {
                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION,
                                                      PyArray_DIMS(%(x)s),
                                                      PyArray_TYPE(%(x)s),
                                                      0);
                if(NULL == %(z)s) {
                    %(fail)s
                }

                if (NULL == internal_buf) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&internal_buf, layout_internal), err );
                }
            }

            if (to_internal) {
                convert_resources[dnnResourceFrom] = PyArray_DATA(%(x)s);
                convert_resources[dnnResourceTo] = (void *)(internal_buf);

                CHECK_ERR( dnnExecute_%(precision)s(to_internal, convert_resources), err );
            } else {
                internal_buf = (PyArray_DATA(%(x)s));
            }

            if(layout_internal != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0])
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_internal;
            if(internal_buf != ((void**)PyArray_DATA(%(z)s))[1])
                ((void**)PyArray_DATA(%(z)s))[1] = internal_buf;

            first_run = 0;

            #ifdef _MKL_DEBUG_
                printf(\"U2IRelu: from_buffer %%x to_buffer %%x\\n\",convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
            #endif
        """ % locals()
        return ccode


class U2IGrad(MKLOp):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        super(U2IGrad, self).__init__(uniq_id)

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
        return hash(type(self))

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x, gz):
        out = x.type()
        return Apply(self, [x, gz], [out])

    def c_cleanup_code_struct(self, node, name):
        d = {}
        if 'float32' == node.inputs[0].type.dtype:
            d['precision'] = "F32"
        elif "float64" == node.inputs[0].type.dtype:
            d['precision'] = "F64"
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[0].type.dtype)
        return """
              //free buffer
              int status = 0;

              if(NULL != user_buf)
              {
                  status = dnnReleaseBuffer_%(precision)s(user_buf);
                  if(0 != status)
                  {
                        printf(\"\\nERROR:dnnReleaseBuffer_%(precision)s user_buf in U2IGrad\\n\");
                        exit(0);
                  }

                  user_buf = NULL;
              }

              if(NULL != from_internal)
              {
                  status = dnnDelete_%(precision)s(from_internal);
                  if(0 != status)
                  {
                       printf(\"\\nERROR:dnnDelete_%(precision)s to_internal\\n\");
                  }
                  from_internal = NULL;
              }

              if(NULL != layout_usr)
              {
                  status = dnnLayoutDelete_%(precision)s(layout_usr);
                  if(0 != status)
                  {
                        printf(\"\\nERROR:dnnLayoutDelete__%(precision)s layout_usr in U2IGrad\\n\");
                        exit(0);
                  }
                  layout_usr = NULL;
              }
              """ % d

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
            raise Exception("Type %s not implemented" %
                            node.inputs[0].type.dtype)
        ccode = """
        int status = 0;
        if(NULL == %(z)s)
        {
              %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(gz)s),
                                          PyArray_DIMS(%(gz)s),
                                          PyArray_TYPE(%(gz)s),
                                          0);
              if(NULL == %(z)s)
              {
                    %(fail)s
              }

              int ndim = (int)PyArray_NDIM(%(gz)s);
              size_t*bottom_size = (size_t*)malloc(ndim*sizeof(size_t));
              size_t*out_stride = (size_t*)malloc(ndim*sizeof(size_t));
              if(0 == bottom_size || 0 == out_stride)
              {
                       printf(\"ERROR: malloc buffer in U2IGrad \\n\");
                       exit(0);
              }

              npy_intp dataSize = 1;
              for(int i=0;i<ndim;i++)
              {
                      bottom_size[i] = (size_t)PyArray_DIMS(%(z)s)[ndim-i-1];
                      out_stride[i] = (size_t)PyArray_STRIDES(%(z)s)[ndim-i-1] / %(x_item_size)s;
                      dataSize = dataSize * bottom_size[i];
              }

              //create usr layerout
              status = dnnLayoutCreate_%(precision)s(&layout_usr,
                                                     ndim, bottom_size,
                                                     out_stride);

              size_t size = dnnLayoutGetMemorySize_%(precision)s(layout_usr);
              if(size != PyArray_DIMS(%(z)s)[0]*PyArray_STRIDES(%(z)s)[0])
              {
                      printf(\"ERROR:dnnLayoutCreate_%(precision)s: %%d , %%d in U2IGrad\\n\",size, dataSize);
                      exit(0);
              }
              free(bottom_size);
              free(out_stride);

              //Get layerout buffer from input.
              layout_internal = ((dnnLayout_t*)PyArray_DATA(%(gz)s))[0]; //get internal layerout
              internal_buf = ((void **)PyArray_DATA(%(gz)s))[1];

              status = dnnConversionCreate_%(precision)s(&from_internal, layout_internal, layout_usr);
              if(0 != status)
              {
                     printf(\"ERROR:dnnConversionCreate_%(precision)s\\n\");
                     exit(0);
              }
        }

        convert_resources[dnnResourceFrom] = internal_buf;
        convert_resources[dnnResourceTo] = (void*)PyArray_DATA(%(z)s);
        #ifdef _MKL_DEBUG_
            printf(\"%%x, %%x , %%x to %%x\\n\",from_internal,layout_internal,internal_buf,convert_resources[dnnResourceTo]);
        #endif

        //cvt
        status = dnnExecute_%(precision)s(from_internal, convert_resources);
        if(0 != status)
        {
                printf(\"ERROR:U2IGrad:%%x, %%x, %%x, status: %%d\\n\",from_internal,convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo],status);
                exit(0);
        }
        """ % sub
        return ccode


class I2UGrad(MKLOp):
    __props__ = ('uniq_id',)

    def __init__(self, uniq_id=0):
        super(I2UGrad, self).__init__(uniq_id)

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
        return hash(type(self))

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__,
                               ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x, gz):
        out = x.type()
        return Apply(self, [x, gz], [out])

    def c_cleanup_code_struct(self, node, name):
        d = {}
        d['name'] = I2UGrad.__name__
        if 'float32' == node.inputs[0].type.dtype:
            d['precision'] = "F32"
        elif "float64" == node.inputs[0].type.dtype:
            d['precision'] = "F64"
        else:
            raise Exception("Type %s not implemented" %
                            node.inputs[0].type.dtype)
        return """
              //free buffer
              int status = 0;
       #ifdef _DEBUG_
              printf(\"I2UGrad CleanUp\\n\");
       #endif
              if(NULL != to_internal)
              {
                    int status = dnnDelete_%(precision)s(to_internal);
                    if(0 != status)
                    {
                          printf(\"\\nERROR:dnnDelete_%(precision)s to_internal\\n\");
                    }
                    to_internal = NULL;
              }

              if(NULL != internal_buf)
              {
                     status = dnnReleaseBuffer_%(precision)s(internal_buf);
                     if(0 != status)
                     {
                             printf(\"\\nERROR:dnnReleaseBuffer_%(precision)s free internal buffer\\n\");
                      }
                      internal_buf = NULL;
              }

              if(NULL != layout_usr)
              {
                       status = dnnLayoutDelete_%(precision)s(layout_usr);
                       if(0 != status)
                       {
                               printf(\"\\nERROR:dnnLayoutDelete__%(precision)s layout_usr in %(name)s\\n\");
                               exit(0);
                       }
                       layout_usr = NULL;
              }

       #ifdef _DEBUG_
              printf(\"I2UGrad CleanUp End\\n\");
       #endif
              """ % d

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
            raise Exception("Type %s not implemented" %
                            node.inputs[0].type.dtype)

        ccode = """
      int status = 0;
      if(NULL == %(z)s)
      {
            npy_intp *dims = (npy_intp*)malloc(sizeof(npy_intp) * PyArray_NDIM(%(x)s));
            if(NULL == dims)
            {
                 printf(\"ERROR: malloc in I2UGrad\\n\");
                 exit(0);
            }

            %(z)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(x)s),//I2UGrad
                                         PyArray_DIMS(%(x)s),//owen
                                         PyArray_TYPE(%(x)s),
                                         0);
             if(NULL == %(z)s)
             {
                 %(fail)s
             }
             free(dims);

             int ndim = (int)PyArray_NDIM(%(gz)s);
             size_t*bottom_size = (size_t*)malloc(ndim*sizeof(size_t));
             size_t*out_stride = (size_t*)malloc(ndim*sizeof(size_t));
             if(0 == bottom_size || 0 == out_stride)
             {
                    printf(\"ERROR: malloc buffer in I2U \\n\");
                    exit(0);
             }

             for(int i=0;i<ndim;i++)
             {
                    bottom_size[i] = (size_t)PyArray_DIMS(%(gz)s)[ndim-i-1];
                    out_stride[i] = (size_t)PyArray_STRIDES(%(gz)s)[ndim-i-1] / %(x_item_size)s;
             }

              //create usr layerout for gz
             status = dnnLayoutCreate_%(precision)s(&layout_usr,
                                                     ndim, bottom_size,
                                                     out_stride);
             if(0 != status)
             {
                      printf(\"ERROR:dnnLayoutCreate_%(precision)s\\n\");
                      exit(0);
             }
             free(bottom_size);
             free(out_stride);

             layout_internal = ((dnnLayout_t*)PyArray_DATA(%(x)s))[0]; //get internal layerout

             //create internal buffer for gradI
             if(NULL == internal_buf)
             {
                       status = dnnAllocateBuffer_%(precision)s(
                                         reinterpret_cast<void **>(&internal_buf),
                                         layout_internal);
                       if(0 != status)
                       {
                               printf(\"ERROR:dnnAllocateBuffer_%(precision)s : %%d \\n\", status);
                               exit(0);
                       }
              }

              if(dnnLayoutGetMemorySize_%(precision)s(layout_usr) != dnnLayoutGetMemorySize_%(precision)s(layout_internal))
              {
                            printf(\"ERROR:I2UGrad: usr space: %%d not equal to internal:%%d\\n\",
                                            dnnLayoutGetMemorySize_%(precision)s(layout_usr), dnnLayoutGetMemorySize_%(precision)s(layout_internal));
                            exit(0);
              }

              status = dnnConversionCreate_%(precision)s(&to_internal, layout_usr, layout_internal);
              if(0 != status)
              {
                    printf(\"ERROR: dnnConversionCreate_%(precision)s in I2UGrad\\n\");
                    exit(0);
              }

              ((void**)PyArray_DATA(%(z)s))[1] = internal_buf;
              ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_internal;
      }

      convert_resources[dnnResourceTo] = reinterpret_cast<void *>(internal_buf);
      convert_resources[dnnResourceFrom] = reinterpret_cast<void *>(PyArray_DATA(%(gz)s));

  #ifdef _DEBUG_
      printf(\"I2UGrad:%%x, %%x, %%x to %%x\\n\",to_internal,layout_internal,convert_resources[dnnResourceFrom],convert_resources[dnnResourceTo]);
      printf(\"I2UGrad x: %%d,%%d,%%d,%%d\\n\",PyArray_DIMS(%(x)s)[0],PyArray_DIMS(%(x)s)[1],PyArray_DIMS(%(x)s)[2],PyArray_DIMS(%(x)s)[3]);
      printf(\"I2UGrad gz: %%d,%%d,%%d,%%d\\n\",PyArray_DIMS(%(gz)s)[0],PyArray_DIMS(%(gz)s)[1],PyArray_DIMS(%(gz)s)[2],PyArray_DIMS(%(gz)s)[3]);
  #endif

      if(dnnLayoutGetMemorySize_%(precision)s(layout_internal) != PyArray_DIMS(%(z)s)[0] * PyArray_STRIDES(%(z)s)[0])
      {
            printf(\"I2UGrad int %%d != usr: %%d\\n\",dnnLayoutGetMemorySize_%(precision)s(layout_internal),PyArray_DIMS(%(z)s)[0]*PyArray_STRIDES(%(z)s)[0]);
            printf(\"I2UGrad gz: %%d,%%d,%%d,%%d\\n\",PyArray_DIMS(%(gz)s)[0],PyArray_DIMS(%(gz)s)[1],PyArray_DIMS(%(gz)s)[2],PyArray_DIMS(%(gz)s)[3]);
            printf(\"I2UGrad x: %%d,%%d,%%d,%%d\\n\",PyArray_DIMS(%(x)s)[0],PyArray_DIMS(%(x)s)[1],PyArray_DIMS(%(x)s)[2],PyArray_DIMS(%(x)s)[3]);
      }

      //cvt
      status = dnnExecute_%(precision)s(to_internal, convert_resources);
      if(0 != status)
      {
                printf(\"ERROR:dnnExecute_%(precision)s\\n\");
                exit(0);
      }
     """ % sub
        return ccode


class U2ILRN(MKLOp):
    __props__ = ('alpha', 'beta', 'k', 'size')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5, uniq_id=0):
        super(U2ILRN, self).__init__(uniq_id)
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.size = n

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('Input should be a 4-dim variable.')
        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [U2IGrad(uniq_id=self.uniq_id)(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        alpha = self.alpha
        beta = self.beta
        k = self.k
        size = self.size

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise Exception('Type %s is not supported!' % node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3]; //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2]; //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1]; //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0]; //n

                bottomStride[0] = 1;
                bottomStride[1] = bottomStride[0] * bottomSize[0];
                bottomStride[2] = bottomStride[1] * bottomSize[1];
                bottomStride[3] = bottomStride[2] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr, DIMENSION, bottomSize, bottomStride), err );
                CHECK_ERR( dnnLRNCreateForward_%(precision)s(&primitive, NULL, layout_usr, %(size)s, %(alpha)s, %(beta)s, %(k)s), err );
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal, primitive, dnnResourceSrc), err );

                if (!dnnLayoutCompare_%(precision)s(layout_usr, layout_internal)) {
                    if (NULL == to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_usr, layout_internal), err );
                    }
                }
            }

            if (NULL == %(z)s) {
                //Create PyArrayObject for output
                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION, PyArray_DIMS(%(x)s), PyArray_TYPE(%(x)s), 0);

                if (NULL == %(z)s) {
                    %(fail)s
                }

                if (NULL == internal_buf) {
                    CHECK_ERR(  dnnAllocateBuffer_%(precision)s((void**)&internal_buf, layout_internal), err );
                }
            }

            if (to_internal) {
                convert_resources[dnnResourceFrom] = (PyArray_DATA(%(x)s));
                convert_resources[dnnResourceTo] = (void*)(internal_buf);
                CHECK_ERR( dnnExecute_%(precision)s(to_internal, convert_resources), err );
            } else {
                internal_buf = (PyArray_DATA(%(x)s));
            }

            if (layout_internal != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0]) {
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_internal;
            }

            if (internal_buf != ((void**)PyArray_DATA(%(z)s))[1]) {
                ((void**)PyArray_DATA(%(z)s))[1] = internal_buf;
            }

            first_run = 0;

            #ifdef _MKL_DEBUG_
                printf(\"U2ILRN: from_buffer %%x to_buffer %%x\\n\", convert_resources[dnnResouceFrom], convert_resources[dnnResourceTo]);
            #endif
        """ % locals()
        return ccode


class U2IConv(MKLOp):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_dilation=(1, 1), uniq_id=0):
        super(U2IConv, self).__init__(uniq_id)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        self.imshp = imshp
        self.kshp = kshp
        self.filter_dilation = filter_dilation

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
        return hash(self.imshp) ^ hash(self.kshp) ^ hash(self.border_mode) ^ hash(self.subsample) ^ hash(self.filter_dilation)

    def __str__(self):
        if hasattr(self, '__props__'):
            return '%s{%s}' % (self.__class__.__name__, ', '.join('%s=%r' % (p, getattr(self, p)) for p in self.__props__))
        else:
            return '%s' % (self.__class__.__name__)

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [U2IGrad(uniq_id=self.uniq_id)(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        dH, dW = self.subsample

        if self.imshp is None:
            self.imshp = x.shape

        i_n, i_c, i_h, i_w = self.imshp

        if len(self.kshp) == 5:
            grp, k_n, k_c, k_h, k_w = self.kshp
            assert i_c == k_c * grp
        else:
            k_n, k_c, k_h, k_w = self.kshp
            grp = 1

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
            raise Exception("Type %s is not supported!" %
                            node.inputs[0].type.dtype)
        fail = sub['fail']

        ccode = """
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

                const int group = %(grp)s;
                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr, DIMENSION, imageSize, imageStride), err );
                CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&primitive, NULL,
                           dnnAlgorithmConvolutionDirect, group, DIMENSION, imageSize, zSize,
                           weightSize, convStride, convPadding, dnnBorderZeros), err );
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal, primitive, dnnResourceSrc), err );
            }

            if (!dnnLayoutCompare_%(precision)s(layout_usr, layout_internal))
            {
                if (NULL == to_internal)
                {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_usr, layout_internal), err );
                }
            }

            if (NULL == %(z)s)
            {
                //Create PyArrayObject for output
                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION, PyArray_DIMS(%(x)s), PyArray_TYPE(%(x)s), 0);

                if (NULL == %(z)s)
                {
                    %(fail)s
                }
            }

            if (NULL == internal_buf)
            {
                CHECK_ERR(  dnnAllocateBuffer_%(precision)s((void**)&internal_buf, layout_internal), err );
            }

            if (to_internal)
            {
                convert_resources[dnnResourceFrom] = (PyArray_DATA(%(x)s));
                convert_resources[dnnResourceTo] = (void*)(internal_buf);
                CHECK_ERR( dnnExecute_%(precision)s(to_internal, convert_resources), err );
            }
            else
            {
                internal_buf = (PyArray_DATA(%(x)s));
            }

            if (layout_internal != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0])
            {
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_internal;
            }
            if (internal_buf != ((void**)PyArray_DATA(%(z)s))[1])
            {
                ((void**)PyArray_DATA(%(z)s))[1] = internal_buf;
            }
            first_run = 0;

            #ifdef _MKL_DEBUG_
            printf(\"U2I_Conv: from_buffer %%x to_buffer %%x\\n\", convert_resources[dnnResouceFrom], convert_resources[dnnResourceTo]);
            #endif
        """ % locals()
        return ccode


class U2IElemwiseSum(MKLOp):
    __props__ = ('inp_num', 'coeff')

    def __init__(self, inp_num=1, coeff=[1.0, ], uniq_id=0):
        super(U2IElemwiseSum, self).__init__(uniq_id)
        self.coeff = coeff
        self.inp_num = inp_num
        assert isinstance(self.coeff, (list, tuple))
        if self.inp_num != len(self.coeff):
            raise ValueError('Number of ElemwiseSum inputs is not equal to number of coefficients.')

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inp_num) ^ hash(sum(self.coeff))

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('U2IElemwiseSum inputs should be 4-dim tensor')
        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [U2IGrad(uniq_id=self.uniq_id)(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        coeff = self.coeff
        inp_num = self.inp_num

        if 'float32' == node.inputs[0].type.dtype:
            sub['type'] = 'float'
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            sub['type'] = 'double'
            precision = 'F64'
        else:
            raise Exception('Type %s is not supported!' % node.inputs[0].type.dtype)

        fail = sub['fail']
        sub['len'] = inp_num

        ccode = """
            %(type)s coeffs[%(len)s] = {1.0};
        """ % sub

        for i, co in enumerate(coeff):
            ccode = ccode + """
            coeffs[%s] = %s;
            """ % (i, co)

        ccode = ccode + """
            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3]; //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2]; //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1]; //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0]; //n

                bottomStride[0] = 1;
                bottomStride[1] = bottomStride[0] * bottomSize[0];
                bottomStride[2] = bottomStride[1] * bottomSize[1];
                bottomStride[3] = bottomStride[2] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr, DIMENSION, bottomSize, bottomStride), err );
                CHECK_ERR( dnnSumCreate_%(precision)s(&primitive, NULL, %(inp_num)s, layout_usr, coeffs), err);
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal, primitive, dnnResourceMultipleSrc), err );

                if (!dnnLayoutCompare_%(precision)s(layout_usr, layout_internal)) {
                    if (NULL == to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_usr, layout_internal), err );
                    }
                }
            }

            if (NULL == %(z)s) {
                //Create PyArrayObject for output
                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION, PyArray_DIMS(%(x)s), PyArray_TYPE(%(x)s), 0);
                if (NULL == %(z)s) {
                    %(fail)s
                }

                if (NULL == internal_buf) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&internal_buf, layout_internal), err );
                }
            }

            if (to_internal) {
                convert_resources[dnnResourceFrom] = (PyArray_DATA(%(x)s));
                convert_resources[dnnResourceTo] = (void*)(internal_buf);
                CHECK_ERR( dnnExecute_%(precision)s(to_internal, convert_resources), err );
            } else {
                internal_buf = (PyArray_DATA(%(x)s));
            }

            if (layout_internal != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0]) {
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_internal;
            }

            if (internal_buf != ((void**)PyArray_DATA(%(z)s))[1]) {
                ((void**)PyArray_DATA(%(z)s))[1] = internal_buf;
            }

            first_run = 0;
            #ifdef _MKL_DEBUG_
                printf(\"U2IElemwiseSum: From buffer %%x to buffer %%x\\n\", convert_resources[dnnResouceFrom], convert_resources[dnnResourceTo]);
            #endif
        """ % locals()
        return ccode


class U2IBatchNormalization(MKLOp):
    __props__ = ('eps',)

    def __init__(self, eps=1e-5, uniq_id=0):
        super(U2IBatchNormalization, self).__init__(uniq_id=uniq_id)
        self.uniq_id = uniq_id
        self.eps = eps

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('The input should be a 4-dim tensor.')
        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [U2IGrad(uniq_id=self.uniq_id)(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        eps = self.eps

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise Exception('Type %s is not supported!' % node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3]; //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2]; //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1]; //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0]; //n

                bottomStride[0] = 1;
                bottomStride[1] = bottomStride[0] * bottomSize[0];
                bottomStride[2] = bottomStride[1] * bottomSize[1];
                bottomStride[3] = bottomStride[2] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_usr, DIMENSION, bottomSize, bottomStride), err );
                CHECK_ERR( dnnBatchNormalizationCreateForward_%(precision)s(&primitive, NULL, layout_usr, %(eps)s), err);
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal, primitive, dnnResourceSrc), err );

                if (!dnnLayoutCompare_%(precision)s(layout_usr, layout_internal)) {
                    if (NULL == to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_usr, layout_internal), err );
                    }
                }
            }

            if (NULL == %(z)s) {
                //Create PyArrayObject for output
                %(z)s = (PyArrayObject*)PyArray_ZEROS(DIMENSION, PyArray_DIMS(%(x)s), PyArray_TYPE(%(x)s), 0);

                if (NULL == %(z)s) {
                    %(fail)s
                }

                if (NULL == internal_buf) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&internal_buf, layout_internal), err );
                }
            }

            if (to_internal) {
                convert_resources[dnnResourceFrom] = (PyArray_DATA(%(x)s));
                convert_resources[dnnResourceTo] = (void*)(internal_buf);
                CHECK_ERR( dnnExecute_%(precision)s(to_internal, convert_resources), err );
            } else {
                internal_buf = (PyArray_DATA(%(x)s));
            }

            if (layout_internal != ((dnnLayout_t*)PyArray_DATA(%(z)s))[0]) {
                ((dnnLayout_t*)PyArray_DATA(%(z)s))[0] = layout_internal;
            }

            if (internal_buf != ((void**)PyArray_DATA(%(z)s))[1]) {
                ((void**)PyArray_DATA(%(z)s))[1] = internal_buf;
            }
            first_run = 0;
            #ifdef _MKL_DEBUG_
                printf(\"U2IBatchNormalization: From buffer %%x to buffer %%x\\n\", convert_resources[dnnResouceFrom], convert_resources[dnnResourceTo]);
            #endif
        """ % locals()
        return ccode
