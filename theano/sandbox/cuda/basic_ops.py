import StringIO, sys
import numpy

from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar, config

from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import filter as type_support_filter

from theano.sandbox.cuda.elemwise import NaiveAlgo

import logging, copy
_logger_name = 'theano_cuda_ndarray.basic_ops'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler()) #TO REMOVE
def warning(*msg):
    _logger.warning(_logger_name+'WARNING: '+' '.join(str(m) for m in msg))
def info(*msg):
    _logger.info(_logger_name+'INFO: '+' '.join(str(m) for m in msg))
def debug(*msg):
    _logger.debug(_logger_name+'DEBUG: '+' '.join(str(m) for m in msg))

def as_cuda_ndarray_variable(x):
    if hasattr(x, '_as_CudaNdarrayVariable'):
        return x._as_CudaNdarrayVariable()
    tensor_x = tensor.as_tensor_variable(x)
    return GpuFromHost()(tensor_x)

class HostFromGpu(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return 'HostFromGpu'
    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x], [tensor.TensorType(dtype=x.dtype, broadcastable=x.broadcastable)()])
    def perform(self, node, (x,), (z,)):
        z[0] = numpy.asarray(x)
    def grad(self, inputs, (gz,)):
        return gz,
        #return [GpuFromHost()(gz)]
host_from_gpu = HostFromGpu()

class GpuFromHost(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return 'GpuFromHost'
    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable)()])
    def perform(self, node, (x,), (z,)):
        z[0] = type_support_filter(numpy.asarray(x, dtype='float32'), tuple([0]*x.ndim), 0)
    def grad(self, inputs, (gz,)):
        return gz,
        #return [HostFromGpu()(gz)]
gpu_from_host = GpuFromHost()

class GpuElemwise(Op):
    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)

    def __init__(self, scalar_op, inplace_pattern, sync=None):
        ##
        # TODO: implement inplace operations.  
        #       It's ok that we set the DestroyMap to something but then don't actually destroy
        #       anything.  It's just a bit of a waste of memory.
        #
        #       As current GPUs don't have cache, this probably doesn't make any difference to
        #       the amount of loading and storing to global memory that we would have to do.
        #       That's why it isn't implemented yet.
        #
        sync = config.config.getboolean('gpuelemwise.sync',sync)
        self.scalar_op = scalar_op
        self.inplace_pattern = inplace_pattern
        self.destroy_map = dict((o, [i]) for o, i in inplace_pattern.items())
        if scalar_op.nin > 0:
            self.ufunc = numpy.frompyfunc(scalar_op.impl, scalar_op.nin, scalar_op.nout)
        else:
            self.ufunc = None
        self._rehash()

        self.src_generator = NaiveAlgo(self.scalar_op, sync=sync)
        self.sync = sync

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        d.pop('ufunc')
        d.pop('__epydoc_asRoutine', None)
        d.pop('_hashval')
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
        if self.scalar_op.nin > 0:
            self.ufunc = numpy.frompyfunc(self.scalar_op.impl, self.scalar_op.nin, self.scalar_op.nout)
        else:
            self.ufunc = None
        self._rehash()

    def __eq__(self, other):
        return type(self) == type(other) and (self.scalar_op == other.scalar_op) \
                and self.inplace_pattern == other.inplace_pattern

    def _rehash(self):
        items = self.inplace_pattern.items()
        items.sort()
        tuple_items=[k for k,v in items]
        for k,v in items:
            if isinstance(v, (tuple, list)):
                tuple_items+=[tuple(v)]
            else: tuple_items+=[v]
        tuple_items = tuple(tuple_items)
        h = hash(type(self)) ^ hash(self.scalar_op) ^ hash(tuple_items)
        # don't change a code that has already been  computed for this object
        assert h == getattr(self,'_hashval', h)
        self._hashval = h

    def __hash__(self):
        return self._hashval

    def __str__(self):
        if 0:
            # TODO:
            # Current implementation does not use inplace pattern
            # although since memory on card is precious... it should!
            if self.inplace_pattern:
                items = self.inplace_pattern.items()
                items.sort()
                return "GpuElemwise{%s}%s" % (self.scalar_op.__class__.__name__, str(items))
        #return "GpuElemwise{%s}" % (self.scalar_op.__class__.__name__)
        return "GpuElemwise{%s}" % (self.scalar_op)

    def __repr__(self):
        return self.__str__()

    def make_node(self, *inputs):
        _inputs = [as_cuda_ndarray_variable(i) for i in inputs]
        if self.nin > 0 and len(_inputs) != self.nin:
            raise TypeError('Wrong argument count', (self.nin, len(_inputs)))
        for i in _inputs[1:]:
            if i.type.ndim != inputs[0].type.ndim:
                raise TypeError('different ranks among inputs')

        # output is broadcastable only along dimensions where all inputs are broadcastable
        broadcastable = []
        for d in xrange(_inputs[0].type.ndim):
            bcast_d = True
            for i in _inputs:
                if not i.type.broadcastable[d]:
                    bcast_d = False
                    break
            broadcastable.append(bcast_d)
        assert len(broadcastable) == _inputs[0].type.ndim

        otype = CudaNdarrayType(broadcastable=broadcastable)
        assert self.nout > 0
        return Apply(self, _inputs, [otype() for o in xrange(self.nout)])

    def c_support_code(self, *args, **kwargs):
        return self.src_generator.c_support_code(*args, **kwargs)

    def c_support_code_apply(self, *args, **kwargs):
        return self.src_generator.c_support_code_apply(*args, **kwargs)

    def c_code(self, *args, **kwargs):
        return self.src_generator.c_code(*args, **kwargs)

    def c_code_cache_version(self):
        return self.src_generator.cache_version

class GpuDimShuffle(Op):
    def __init__(self, input_broadcastable, new_order):
        input_broadcastable = tuple(input_broadcastable)
        self.input_broadcastable = input_broadcastable
        new_order = tuple(new_order)
        self.new_order = new_order

        # list of dimensions of the input to drop
        self.drop = []
        i2j = {} # this maps i before dropping dimensions to j after dropping dimensions so self.shuffle can be set properly later on
        j = 0
        for i, b in enumerate(input_broadcastable):
            if i not in new_order:
                # we want to drop this dimension because it's not a value in new_order
                if b == 1: # 1 aka True
                    self.drop.append(i)
                else:
                    # we cannot drop non-broadcastable dimensions
                    raise ValueError("You cannot drop a non-broadcastable dimension.", (input_broadcastable, new_order))
            else:
                i2j[i] = j
                j += 1

        # transposition of non-broadcastable dimensions
        # This is how the dimensions will be permuted, without accounting for the extra
        # 'x' broadcastable dimensions to insert.
        self.shuffle = [i2j[x] for x in new_order if x != 'x']

        # list of dimensions of the output that are broadcastable and were not in the original input
        self.augment = [i for i, x in enumerate(new_order) if x == 'x']

        self.view_map = {0: [0]}

        self._rehash()

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_hashval']
        return d
    def __setstate__(self, d):
        self.__dict__.update(d)
        self._rehash()

    def make_node(self, input):
        ib = tuple(input.type.broadcastable)
        if not ib == self.input_broadcastable:
            raise TypeError("The number of dimensions and/or broadcastable pattern of the input is incorrect for this op. Expected %s, got %s." % (self.input_broadcastable, ib))
        ob = []
        for value in self.new_order:
            if value == 'x':
                ob.append(True)
            else:
                ob.append(ib[value])
        return Apply(self, [input], [CudaNdarrayType(broadcastable=ob)()])

    def __eq__(self, other):
        # it's probably not necessary to compare input_broadcastable
        return type(self) == type(other) \
            and self.new_order == other.new_order \
            and self.input_broadcastable == other.input_broadcastable

    def _rehash(self):
        self._hashval = hash(type(self).__name__) ^ hash(type(self).__module__) \
                ^ hash(self.new_order) ^ hash(self.input_broadcastable)

    def __hash__(self):
        return self._hashval

    def __str__(self):
        return "GpuDimShuffle{%s}" % ",".join(str(x) for x in self.new_order)

    def c_code(self, node, name, (input,), (res,), sub):
        basename = input + '__view_or_copy'

        nd_in = len(self.input_broadcastable)
        nd_out = len(self.new_order)
        sio = StringIO.StringIO()
        fail = sub['fail']

        #check input
        print >> sio, """
        if (%(input)s->nd != %(nd_in)s)
        {
            PyErr_Format(PyExc_TypeError, "required nd=%(nd_in)s, got nd=%%i", %(input)s->nd);
            %(fail)s;
        }
        """ %locals()

        #alloc an output
        print >> sio, """
        if (%(res)s && (%(res)s->nd == %(nd_out)s))
        {
            //re-use previously-allocated cnda
        }
        else
        {
            if (%(res)s)
            {
                if (CudaNdarray_set_nd(%(res)s, %(nd_out)s))
                {
                    Py_DECREF(%(res)s);
                    %(res)s = NULL;
                    %(fail)s;
                }
            }
            else
            {
                %(res)s = (CudaNdarray*) CudaNdarray_New(%(nd_out)s);
                if (NULL == %(res)s)
                {
                    %(fail)s;
                }
            }
        }
        """ %locals()

        print >> sio, """
        if (CudaNdarray_set_device_data(%(res)s, CudaNdarray_DEV_DATA(%(input)s), %(input)s))
        {
            // err message set
            Py_DECREF(%(res)s);
            %(res)s = NULL;
            %(fail)s;
        }
        """ %locals()

        #reassign the dimension and strides in the host pointers
        for i, o in enumerate(self.new_order):
            if o == 'x':
                #TODO: remove this assertion
                #      the correct thing to do is to insert a run-time check 
                #      that the size in this dimension is 1
                assert node.outputs[0].type.broadcastable[i]
                print >> sio, """
        CudaNdarray_set_dim(%(res)s, %(i)s, 1);
        CudaNdarray_set_stride(%(res)s, %(i)s, 0);
                """ %locals()
            else:
                print >> sio, """
        CudaNdarray_set_dim(%(res)s, %(i)s, CudaNdarray_HOST_DIMS(%(input)s)[%(o)s]);
        CudaNdarray_set_stride(%(res)s, %(i)s, CudaNdarray_HOST_STRIDES(%(input)s)[%(o)s]);
                """ %locals()

        for i, o in enumerate(self.new_order):
                print >> sio, """
        //std::cerr << "GpuDimShuffle " << %(res)s << " str[%(i)s] = " << %(res)s->str[%(i)s] << "\\n";
                """ %locals()

        # copy the host dims and stride -> device
        if 0:
            print >> sio, """
            if (CudaNdarray_copy_structure_to_device(%(res)s))
            {
                //err msg set
                Py_DECREF(%(res)s);
                %(res)s = NULL;
                %(fail)s;
            }
            """ %locals()

        if 0: # print full code to stdout
            print '--------------------------------------'
            print 'C_CODE'
            print ''
            print self
            print "IN BROAD", self.input_broadcastable
            print "NEW ORDER", self.new_order
            print "SHUFFLE", self.shuffle
            print "AUGMENT", self.augment
            print '------------'
            print ''
            print sio.getvalue()
            print '--------------------------------------'
            if 0:
                import sys
                sys.exit()

        return sio.getvalue()
    
    def c_code_cache_version(self):
        return (1,0)

class GpuSum(Op):
    """GpuSum is a Reduction along some dimensions by summation.

    The dimensions along which to sum is specified by the `reduce_mask` that you pass to the
    constructor.  The `reduce_mask` is a tuple of booleans (actually integers 0 or 1) that
    specify for each input dimension, whether to reduce it (1) or not (0).

    For example:
    
      - reduce_mask == (1,) sums a vector to a scalar

      - reduce_mask == (1,0) computes the sum of each column in a matrix

      - reduce_mask == (0,1) computes the sum of each row in a matrix

      - reduce_mask == (1,1,1) computes the sum of all elements in a 3-tensor.

    :note: any reduce_mask of all zeros is a sort of 'copy', and may be removed during graph
    optimization

    """
    def __init__(self, reduce_mask):
        self.reduce_mask = tuple(reduce_mask)

    def __eq__(self, other):
        return type(self) == type(other) and self.reduce_mask == other.reduce_mask

    def __hash__(self):
        return hash(type(self)) ^ hash(self.reduce_mask)

    def __str__(self):
        return "GpuSum{%s}" % ','.join(str(i) for i in self.reduce_mask)

    def make_node(self, x):
        if (x.type.ndim != len(self.reduce_mask)):
            raise TypeError("x must have rank %i"%len(self.reduce_mask))
        o_broadcast = [x.type.broadcastable[i] for i in xrange(x.type.ndim) if not self.reduce_mask[i]]
        return Apply(self, [x], [CudaNdarrayType(o_broadcast)()])

    def perform(self, node, (x,), (z,)):
        z[0] = x.reduce_sum(self.reduce_mask)

    def c_code(self, node, name, (x,), (z,), sub):

        nd_in = node.inputs[0].type.ndim
        nd_out = node.outputs[0].type.ndim

        assert nd_in - nd_out == sum(self.reduce_mask)

        sio = StringIO.StringIO()
        fail = sub['fail']

        #check input
        print >> sio, """
        if (%(x)s->nd != %(nd_in)s)
        {
            PyErr_Format(PyExc_TypeError, "required nd=%(nd_in)s, got nd=%%i", %(x)s->nd);
            %(fail)s;
        }
        """ %locals()

        #
        # alloc an output if we need one
        #

        # check the basics of out output
        print >> sio, """
        if (  !%(z)s 
           || (%(z)s->nd != %(nd_out)s)
        """ % locals()

        #ensure that the output has the right non-reduced dimensions
        j = 0
        for i in xrange(nd_in):
            if not self.reduce_mask[i]: 
                print >> sio, " || (CudaNdarray_HOST_DIMS(%(z)s)[%(j)s] !=CudaNdarray_HOST_DIMS(%(x)s)[%(i)s]) " % locals()
                j += 1

        print >> sio, """
           )
        {
            """ %locals()
        print >> sio, "int new_dims[%(nd_out)s]; " % locals()

        j = 0
        for i in xrange(nd_in):
            if not self.reduce_mask[i]: 
                print >> sio, 'new_dims[%(j)s] = CudaNdarray_HOST_DIMS(%(x)s)[%(i)s];' % locals()
                j += 1

        print >> sio, """
            Py_XDECREF(%(z)s);
            %(z)s = (CudaNdarray*) CudaNdarray_NewDims(%(nd_out)s, new_dims);
            if (NULL == %(z)s)
            {
                PyErr_Format(PyExc_RuntimeError, "Failed to allocate output");
                %(fail)s;
            }
        }
        """ %locals()

        #
        # Now perform the reduction
        #
        getattr(self, 'c_code_reduce_%s'%(''.join(str(i) for i in self.reduce_mask)))(sio, node, name, x, z, fail)

        return sio.getvalue()

    def _makecall(self, node, name, x, z, fail):
        """Return a string for making a kernel call.

            The return value looks something like:

            .. code-block:: c

                if (verbose) printf("running kernel_reduce_sum_10_%(name)s\\n");
                int n_shared = sizeof(float) * n_threads.x;
                kernel_reduce_sum_10_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                        CudaNdarray_HOST_DIMS(%(x)s)[0],
                        CudaNdarray_HOST_DIMS(%(x)s)[1],
                        CudaNdarray_DEV_DATA(%(x)s),
                        CudaNdarray_HOST_STRIDES(%(x)s)[0],
                        CudaNdarray_HOST_STRIDES(%(x)s)[1],
                        CudaNdarray_DEV_DATA(%(z)s),
                        CudaNdarray_HOST_STRIDES(%(z)s)[0]
                        );
                CNDA_THREAD_SYNC;
                if (cudaSuccess != cudaGetLastError()) 
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: ... );
                    %(fail)s;
                }
        """
        sio = StringIO.StringIO()
        pattern = ''.join(str(c) for c in self.reduce_mask)
        ndim = len(pattern)
        nd_out = ndim - sum(self.reduce_mask)
        print >> sio, """
            if (verbose) printf("running kernel_reduce_sum_%(pattern)s_%(name)s\\n");
            int n_shared = sizeof(float) * n_threads.x * n_threads.y * n_threads.z;
            kernel_reduce_sum_%(pattern)s_%(name)s<<<n_blocks, n_threads, n_shared>>>(
            """ %locals()
        for i in xrange(ndim):
            print >> sio, """
                    CudaNdarray_HOST_DIMS(%(x)s)[%(i)s],
            """ %locals()
        print >> sio, """
                    CudaNdarray_DEV_DATA(%(x)s)
            """ %locals()
        for i in xrange(ndim):
            print >> sio, """
                    ,CudaNdarray_HOST_STRIDES(%(x)s)[%(i)s]
            """ %locals()
        print >> sio, """
                    ,CudaNdarray_DEV_DATA(%(z)s)
            """ %locals()
        for i in xrange(nd_out):
            print >> sio, """
                    ,CudaNdarray_HOST_STRIDES(%(z)s)[%(i)s]
            """ %locals()
        print >> sio, """
                    );
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_sum_%(pattern)s_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
        """ %locals()
        return sio.getvalue()

    def _k_decl(self, node, nodename):
        """Return a string to declare a kernel function

        .. code-block:: c

            static __global__ void kernel_reduce_sum_110_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const float *A,
                    const int sA0, 
                    const int sA1,
                    const int sA2,
                    float * Z,
                    const int sZ0)

        """ %locals()
        pattern = ''.join(str(i) for i in self.reduce_mask)
        sio = StringIO.StringIO()

        print >> sio, """
            static __global__ void kernel_reduce_sum_%(pattern)s_%(nodename)s(
        """ %locals()
        for i in xrange(len(self.reduce_mask)):
            print >> sio, """
                    const int d%(i)s,
        """ %locals()
        print >> sio, """
                    const float *A,
        """ %locals()
        for i in xrange(len(self.reduce_mask)):
            print >> sio, """
                    const int sA%(i)s,
        """ %locals()
        print >> sio, """
                    float * Z
        """ %locals()
        for i in xrange(len(self.reduce_mask) - sum(self.reduce_mask)):
            print >> sio, """
                    , const int sZ%(i)s
        """ %locals()
        print >> sio, ")"
        return sio.getvalue()

    def _k_init(self, *args):
        return """
                const int threadCount = blockDim.x * blockDim.y * blockDim.z;
                const int threadNum = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ float buf[];
                float mysum = 0.0f;

                if (warpSize != 32)
                {  
                    //TODO: set error code
                    Z[0] = -666;
                    return;
                }

        """

    def _k_reduce_buf(self, z_pos):
        return """
        buf[threadNum] = mysum;
        __syncthreads();

        // rest of function is handled by one warp
        if (threadNum < warpSize)
        {
            //round up all the partial sums into the first `warpSize` elements
            for (int i = threadNum + warpSize; i < threadCount; i += warpSize)
            {
                mysum += buf[i];
            }
            buf[threadNum] = mysum;
            // no sync because only one warp is running
            if (threadNum < 16)
            {
                //reduce so that threadNum 0 has the sum of everything
                if(threadNum + 16 < threadCount) buf[threadNum] += buf[threadNum+16];
                if(threadNum + 8 < threadCount) buf[threadNum] += buf[threadNum+8];
                if(threadNum + 4 < threadCount) buf[threadNum] += buf[threadNum+4];
                if(threadNum + 2 < threadCount) buf[threadNum] += buf[threadNum+2];
                if(threadNum + 1 < threadCount) buf[threadNum] += buf[threadNum+1];
                if (threadNum == 0)
                {
                    %(z_pos)s = buf[0];
                }
            }
        }
        """ %locals()
    
    def c_code_reduce_1(self, sio, node, name, x, z, fail):
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(1);
            if (verbose) printf("running kernel_reduce_sum_1_%(name)s\\n");
            int n_shared = sizeof(float) * n_threads.x * n_threads.y * n_threads.z;
            kernel_reduce_sum_1_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    CudaNdarray_HOST_DIMS(%(x)s)[0],
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_DEV_DATA(%(z)s));
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_sum_1_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
        }
        """ %locals()

    def c_code_reduce_11(self, sio, node, name, x, z, fail):
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[1],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            while (n_threads.y * n_threads.x <= NUM_VECTOR_OP_THREADS_PER_BLOCK) ++n_threads.y;
            n_threads.y -= 1;
            if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[0]) 
                n_threads.y = CudaNdarray_HOST_DIMS(%(x)s)[0]; 

            dim3 n_blocks(1);
            if (verbose) fprintf(stdout, "running kernel_reduce_sum_11_%(name)s\\n");
            if (verbose) fprint_CudaNdarray(stdout, %(x)s);
            int n_shared = sizeof(float) * n_threads.x * n_threads.y * n_threads.z;
            kernel_reduce_sum_11_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    CudaNdarray_HOST_DIMS(%(x)s)[0],
                    CudaNdarray_HOST_DIMS(%(x)s)[1],
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_HOST_STRIDES(%(x)s)[1],
                    CudaNdarray_DEV_DATA(%(z)s));
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_sum_11_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
        }
        """ %locals()


    def c_code_reduce_10(self, sio, node, name, x, z, fail):
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(CudaNdarray_HOST_DIMS(%(x)s)[1]);
            if (verbose) printf("running kernel_reduce_sum_10_%(name)s\\n");
            int n_shared = sizeof(float) * n_threads.x;
            kernel_reduce_sum_10_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    CudaNdarray_HOST_DIMS(%(x)s)[0],
                    CudaNdarray_HOST_DIMS(%(x)s)[1],
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_HOST_STRIDES(%(x)s)[1],
                    CudaNdarray_DEV_DATA(%(z)s),
                    CudaNdarray_HOST_STRIDES(%(z)s)[0]
                    );
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_sum_10_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
        }
        """ %locals()


    def c_code_reduce_100(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        # use threadIdx.x for i0
        # use blockIdx.x for i1
        # use blockIdx.y for i2
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(CudaNdarray_HOST_DIMS(%(x)s)[1]);
            while (n_blocks.x * n_blocks.y <= NUM_VECTOR_OP_BLOCKS)
            {
                if (n_blocks.y > CudaNdarray_HOST_DIMS(%(x)s)[2])
                    break;
                n_blocks.y += 1;
            }
            n_blocks.y -= 1;
            %(makecall)s
        }
        """ %locals()

    def c_code_reduce_110(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[1],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            while (n_threads.x*n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[0])
                    break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;

            dim3 n_blocks(CudaNdarray_HOST_DIMS(%(x)s)[2]);
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_001(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[2],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                        NUM_VECTOR_OP_BLOCKS));
            while (n_blocks.x * n_blocks.y <= NUM_VECTOR_OP_BLOCKS)
            {
                if (n_blocks.y > CudaNdarray_HOST_DIMS(%(x)s)[1])
                    break;
                n_blocks.y += 1;
            }
            n_blocks.y -= 1;
            %(makecall)s
        }
        """ % locals()
    def c_code_reduce_111(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[2],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));

            //get as many y threads as we can fit
            while (n_threads.x * n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[1])
                    break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;

            //get as many z threads as we can fit
            while (n_threads.x * n_threads.y * n_threads.z <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.z > CudaNdarray_HOST_DIMS(%(x)s)[0])
                    break;
                n_threads.z += 1;
            }
            n_threads.z -= 1;

            dim3 n_blocks(1,1,1);
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_1111(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[2],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));

            //get as many y threads as we can fit
            while (n_threads.x * n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[1])
                    break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;

            //get as many z threads as we can fit
            while (n_threads.x * n_threads.y * n_threads.z <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.z > CudaNdarray_HOST_DIMS(%(x)s)[0])
                    break;
                n_threads.z += 1;
            }
            n_threads.z -= 1;

            dim3 n_blocks(1,1,1);
            %(makecall)s
        }
        """ % locals()

    def c_code_reduce_1011(self, sio, node, name, x, z, fail):
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[3],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));

            while (n_threads.y * n_threads.x < NUM_VECTOR_OP_THREADS_PER_BLOCK) ++n_threads.y;
            n_threads.y -= 1;
            if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[2]) 
                n_threads.y = CudaNdarray_HOST_DIMS(%(x)s)[2]; 

            while (n_threads.x * n_threads.y * n_threads.z < NUM_VECTOR_OP_THREADS_PER_BLOCK) ++n_threads.z;
            n_threads.z -= 1;
            if (n_threads.z > 64)
                n_threads.z = 64;
            if (n_threads.z > CudaNdarray_HOST_DIMS(%(x)s)[0]) 
                n_threads.z = CudaNdarray_HOST_DIMS(%(x)s)[0]; 
            
            dim3 n_blocks(CudaNdarray_HOST_DIMS(%(x)s)[1]);

            if (verbose) printf("running kernel_reduce_sum_1011_%(name)s\\n");
            if (verbose) fprint_CudaNdarray(stdout, %(x)s);
            if (verbose) fprint_CudaNdarray(stdout, %(z)s);
            int n_shared = sizeof(float) * n_threads.x * n_threads.y * n_threads.z;
            kernel_reduce_sum_1011_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    CudaNdarray_HOST_DIMS(%(x)s)[0],
                    CudaNdarray_HOST_DIMS(%(x)s)[1],
                    CudaNdarray_HOST_DIMS(%(x)s)[2],
                    CudaNdarray_HOST_DIMS(%(x)s)[3],
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_HOST_STRIDES(%(x)s)[1],
                    CudaNdarray_HOST_STRIDES(%(x)s)[2],
                    CudaNdarray_HOST_STRIDES(%(x)s)[3],
                    CudaNdarray_DEV_DATA(%(z)s),
                    CudaNdarray_HOST_STRIDES(%(z)s)[0]);
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_sum_1011_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
        }
        """ %locals()

    def c_code_cache_version(self):
        #return ()
        return (8,)


    def c_support_code_apply(self, node, nodename):
        sio = StringIO.StringIO()
        if self.reduce_mask == (1,):
            #this kernel is ok for up to a few thousand elements, but 
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]')
            print >> sio, """
            static __global__ void kernel_reduce_sum_1_%(nodename)s(
                    const unsigned int d0,
                    const float *A, const int sA0,
                    float * Z)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ float buf[];
                float mysum = 0.0f;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.x; i0 < d0; i0 += blockDim.x)
                {
                    float Ai = A[i0 * sA0];
                    mysum += Ai;
                }
                %(reducebuf)s
            }
            """ %locals()
        if self.reduce_mask == (1,1):
            #this kernel is ok for up to a few thousand elements, but 
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]')
            print >> sio, """
            static __global__ void kernel_reduce_sum_11_%(nodename)s(
                    const int d0,
                    const int d1,
                    const float *A, const int sA0, const int sA1,
                    float * Z)
            {
                const int threadCount = blockDim.x * blockDim.y;
                const int threadNum = threadIdx.y*blockDim.x + threadIdx.x;
                extern __shared__ float buf[];
                float mysum = 0.0f;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.y; i0 < d0; i0 += blockDim.y)
                {
                    for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                    {
                        float Ai = A[i0 * sA0 + i1 * sA1];
                        mysum += Ai;
                    }
                }
                %(reducebuf)s
            }
            """ %locals()
        if self.reduce_mask == (1,0):
            # this kernel uses one block for each column, 
            # threads per block for each element per column.

            #TODO: This kernel is pretty inefficient in terms of reading, because if A is
            #      c_contiguous (typical case) then each warp is accessing non-contigous
            #      memory (a segment of a column).
            reducebuf = self._k_reduce_buf('Z[blockIdx.x * sZ0]')
            print >> sio, """
            static __global__ void kernel_reduce_sum_10_%(nodename)s(
                    const int d0,
                    const int d1,
                    const float *A, const int sA0, const int sA1,
                    float * Z, const int sZ0)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ float buf[];
                float mysum = 0.0f;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.x; i0 < d0; i0 += blockDim.x)
                {
                    float Ai = A[i0 * sA0 + blockIdx.x * sA1];
                    mysum += Ai;
                }
                %(reducebuf)s
            }
            """ %locals()
        if self.reduce_mask == (1,1,0):
            # this kernel uses one block for each column, 
            # threads per block for each element per column.

            #TODO: This kernel is pretty inefficient in terms of reading, because if A is
            #      c_contiguous (typical case) then each warp is accessing non-contigous
            #      memory (a segment of a column).
            reducebuf = self._k_reduce_buf('Z[blockIdx.x * sZ0]')
            print >> sio, """
            static __global__ void kernel_reduce_sum_110_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const float *A, const int sA0, const int sA1, const int sA2,
                    float * Z, const int sZ0)
            {
                const int threadCount = blockDim.x * blockDim.y;
                const int threadNum = threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ float buf[];
                float mysum = 0.0f;

                if (warpSize != 32)
                {  
                    //TODO: set error code
                    Z[blockIdx.x * sZ0] = -666;
                    return;
                }

                for (int i0 = threadIdx.y; i0 < d0; i0 += blockDim.y)
                {
                    for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                    {
                        float Ai = A[i0 * sA0 + i1 * sA1 + blockIdx.x * sA2];
                        mysum += Ai;
                    }
                }

                %(reducebuf)s
            }
            """ %locals()
        if self.reduce_mask == (1,0,0):
            reducebuf = self._k_reduce_buf('Z[i1 * sZ0 + i2 * sZ1]')
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            print >> sio, """
            %(decl)s
            {
                %(init)s
                for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                {
                    for (int i1 = blockIdx.x; i1 < d1; i1 += gridDim.x)
                    {
                        mysum = 0;
                        for (int i0 = threadIdx.x; i0 < d0; i0 += blockDim.x)
                        {
                            mysum += A[i0 * sA0 + i1 * sA1 + i2 * sA2];
                        }
                        %(reducebuf)s
                    }
                }
            }
            """ %locals()
        if self.reduce_mask == (1,1,1):
            reducebuf = self._k_reduce_buf('Z[0]')
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            print >> sio, """
            %(decl)s
            {
                %(init)s
                mysum = 0;
                for (int i0 = threadIdx.z; i0 < d0; i0 += blockDim.z)
                {
                    for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)
                    {
                        for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)
                        {
                            mysum += A[i0 * sA0 + i1 * sA1 + i2 * sA2];
                        }
                    }
                }
                %(reducebuf)s
            }
            """ %locals()
        if self.reduce_mask == (0,0,1):
            # this kernel uses one block for each row, 
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i1 * sZ1]')
            print >> sio, """
            static __global__ void kernel_reduce_sum_001_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const float *A, const int sA0, const int sA1, const int sA2,
                    float * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ float buf[];

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y)
                    {
                        float mysum = 0.0f;
                        for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)
                        {
                            mysum += A[i0 * sA0 + i1 * sA1 + i2 * sA2];
                        }
                        %(reducebuf)s
                    }
                }
            }
            """ %locals()
        if self.reduce_mask == (1,1,1,1):
            reducebuf = self._k_reduce_buf('Z[0]')
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            print >> sio, """
            %(decl)s
            {
                %(init)s
                mysum = 0;
              for (int i0 = 0; i0 < d0; i0++)
                for (int i1 = threadIdx.z; i1 < d1; i1 += blockDim.z)
                {
                    for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            mysum += A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3];
                        }
                    }
                }
                %(reducebuf)s
            }
            """ %locals()
        if self.reduce_mask == (1,0,1,1):
            reducebuf = self._k_reduce_buf('Z[blockIdx.x*sZ0]')
            print >> sio, """
            static __global__ void kernel_reduce_sum_1011_%(nodename)s(
                    const unsigned int d0,
                    const unsigned int d1,
                    const unsigned int d2,
                    const unsigned int d3,
                    const float *A, const int sA0, const int sA1, const int sA2, const int sA3,
                    float * Z, const int sZ0)
            {
                const int threadCount = blockDim.x * blockDim.y * blockDim.z;
                const int threadNum = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ float buf[];
                float mysum = 0.0f;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.z; i0 < d0; i0 += blockDim.z)
                {
                    for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            float Ai = A[i0 * sA0 + blockIdx.x * sA1 + i2 * sA2 + i3 * sA3];
                            mysum += Ai;
                        }
                    }
                }
                %(reducebuf)s
            }
            """ %locals()
        return sio.getvalue()

class GpuReshape(tensor.Reshape):
    # __hash__, __eq__, __str__ come from tensor.Subtensor
    def make_node(self, x, shp):
        host_reshaped = host_from_gpu(x).reshape(shp)
        return Apply(self, [x, shp], [CudaNdarrayType(host_reshaped.broadcastable)()])
    def perform(self, node, (x, shp), (out,)):
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to Reshape.perform has incorrect length %i'
                    ', should be %i' % (len(shp), self.ndim), shp)
        out[0] = x.reshape(tuple(shp))

class GpuSubtensor(tensor.Subtensor):
    # __hash__, __eq__, __str__ come from tensor.Subtensor
    def make_node(self, x, *inputs):
        assert isinstance(x.type, CudaNdarrayType)
        rval = tensor.Subtensor.make_node(self, x, *inputs)
        otype = CudaNdarrayType(rval.outputs[0].type.broadcastable)
        return Apply(self, [x]+rval.inputs[1:], [otype()])

    def perform(self, node, inputs, (out, )):
        x = inputs[0]
        indices = list(reversed(inputs[1:]))

        def convert(entry):
            if isinstance(entry, Type):
                return indices.pop()
            elif isinstance(entry, slice):
                return slice(convert(entry.start),
                             convert(entry.stop),
                             convert(entry.step))
            else:
                return entry

        cdata = tuple(map(convert, self.idx_list))
        if len(cdata) == 1:
            cdata = cdata[0]
        out[0] = x.__getitem__(cdata)

class GpuIncSubtensor(tensor.IncSubtensor):
    def make_node(self, x, y, *inputs):
        assert isinstance(x.type, CudaNdarrayType)
        assert isinstance(y.type, CudaNdarrayType)
        rval = tensor.IncSubtensor.make_node(self, x, y, *inputs)
        return Apply(self, [x,y]+rval.inputs[2:], [x.type()])

class GpuFlatten(tensor.Flatten):
    def make_node(self, x ):
        assert isinstance(x.type, CudaNdarrayType)
        rval = tensor.Flatten.make_node(self, x)
        host_out_broadcastable = rval.outputs[0].type.broadcastable
        out_type = CudaNdarrayType(broadcastable=host_out_broadcastable)
        return Apply(self, [x], [out_type()])

class GpuShape(tensor.Shape):
    def make_node(self, x):
        return Apply(self, [x], [tensor.lvector()])
gpu_shape = GpuShape()

