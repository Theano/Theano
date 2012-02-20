import StringIO, sys
import numpy

import theano
from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar, config

from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda import filter as type_support_filter

from theano.sandbox.cuda.elemwise import NaiveAlgo

import logging, copy

import cuda_ndarray

_logger_name = 'theano.sandbox.cuda.basic_ops'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler()) #TO REMOVE

def as_cuda_ndarray_variable(x):
    if hasattr(x, '_as_CudaNdarrayVariable'):
        return x._as_CudaNdarrayVariable()
    tensor_x = tensor.as_tensor_variable(x)
    return gpu_from_host(tensor_x)

def as_cuda_array(obj):
    if isinstance(obj, numpy.ndarray):
        return cuda_ndarray.cuda_ndarray.CudaNdarray(obj)
    elif isinstance(obj, cuda_ndarray.cuda_ndarray.CudaNdarray):
        return obj
    else:
        raise TypeError("Don't know how to cast to a CudaNdarray object")


class HostFromGpu(GpuOp):
    """
    Implement the transfer from gpu to the cpu.
    """
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
    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = numpy.asarray(x)
    def grad(self, inputs, grads):
        gz, = grads
        return [gpu_from_host(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, tensor.TensorType):
            return [gpu_from_host(ev)]
        else:
            return [ev]
    def infer_shape(self, node, xshp):
        return xshp
host_from_gpu = HostFromGpu()

class GpuFromHost(GpuOp):
    """
    Implement the transfer from cpu to the gpu.
    """
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return 'GpuFromHost'
    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable, dtype=x.dtype)()])
    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = type_support_filter(theano._asarray(x, dtype='float32'), tuple([0]*x.ndim), 0, z[0])
    def grad(self, inputs, grads):
        gz, = grads
        return [host_from_gpu(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        if isinstance(ev, CudaNdarrayType):
            return [host_from_gpu(ev)]
        else:
            return [ev]

    def infer_shape(self, node, xshp):
        return xshp
gpu_from_host = GpuFromHost()


class GpuElemwise(GpuOp):
    """
    Implement a generic elemwise on the gpu.
    """
    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)

    def __init__(self, scalar_op, inplace_pattern = {}, sync=None):
        #TODO-- this looks like a bug-- either we should use the sync argument
        # or get rid of it, we shouldn't let the client think they can control
        #sync when they can't
        sync = config.gpuelemwise.sync

        self.scalar_op = scalar_op

        self.inplace_pattern = inplace_pattern
        self.destroy_map = dict((o, [i]) for o, i in inplace_pattern.items())

        self.sync = sync

        self._rehash()

        self.src_generator = NaiveAlgo(self.scalar_op, sync=sync,
                                       inplace_pattern = self.inplace_pattern)

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        d.pop('__epydoc_asRoutine', None)
        d.pop('_hashval')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.sync = d.get('sync', True) #old objects defaulted to sync behaviour
        self._rehash()

    def __eq__(self, other):
        return type(self) == type(other) and (self.scalar_op == other.scalar_op) \
                and self.inplace_pattern == other.inplace_pattern \
                and self.sync == other.sync

    def _rehash(self):
        items = self.inplace_pattern.items()
        items.sort()
        tuple_items=[k for k,v in items]
        for k,v in items:
            if isinstance(v, (tuple, list)):
                tuple_items+=[tuple(v)]
            else: tuple_items+=[v]
        tuple_items = tuple(tuple_items)
        h = hash(type(self)) ^ hash(self.scalar_op) ^ hash(tuple_items) ^ hash(self.sync)
        # don't change a code that has already been  computed for this object
        assert h == getattr(self,'_hashval', h)
        self._hashval = h

    def __hash__(self):
        return self._hashval

    def __str__(self):
        if self.inplace_pattern:
            items = self.inplace_pattern.items()
            items.sort()
            # We need to print the scalar_op, not only the its class name
            # to have the full definition of composite op.
            return "GpuElemwise{%s}%s" % (self.scalar_op, str(items))
        return "GpuElemwise{%s,no_inplace}" % (self.scalar_op)

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

    def c_compile_args(self):
        #TODO: compile ptx file without constraint and then use the number of
        # registers required to inform the maximum number of threads per block.
        return ["--maxrregcount=32"]

    def c_code_cache_version(self):
        return self.src_generator.cache_version

class GpuDimShuffle(GpuOp):
    """
    Implement DimShuffle on the gpu.
    """
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
        if not isinstance(input.type, CudaNdarrayType):
            raise TypeError("The input of a GpuDimshuffle must be a CudaNdarray")
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

    def c_code(self, node, name, inp, out, sub):
        input, = inp
        res, = out
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

class GpuSum(GpuOp):
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

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = x.reduce_sum(self.reduce_mask)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

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
        if nd_out > 0:
            print >> sio, "int new_dims[%(nd_out)s]; " % locals()
        else:
            print >> sio, "int *new_dims=NULL; "

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

        # \begin bracket the reduction in a check that there is actually work to do
        print >> sio, """
        if (CudaNdarray_SIZE(%(z)s))
        {
        """ % locals()

        #
        # Now perform the reduction
        #

        if all(i==1 for i in self.reduce_mask):
            #check if the tensor is ccontiguous, if true, use the c_c0de_reduce_ccontig code.
            #TODO: check if we are ccontiguous when we un-dimshuffle
            #TODO: if only some dims are ccontiguous, call version with less dims.
            print >> sio, 'if(CudaNdarray_is_c_contiguous(%(x)s)){'%locals()
            self.c_code_reduce_ccontig(sio, node, name, x, z, fail)
            print >> sio, "}else{"
            getattr(self, 'c_code_reduce_%s'%(''.join(str(i) for i in self.reduce_mask)))(sio, node, name, x, z, fail)
            print >> sio, "}"
        else:
            getattr(self, 'c_code_reduce_%s'%(''.join(str(i) for i in self.reduce_mask)))(sio, node, name, x, z, fail)


        # \end bracket the reduction ...
        print >> sio, """
        }
        """ % locals()

        return sio.getvalue()

    def _makecall(self, node, name, x, z, fail, pattern=None):
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
        if pattern is None:
            pattern = ''.join(str(c) for c in self.reduce_mask)
        ndim = len(self.reduce_mask)
        nd_out = ndim - sum(self.reduce_mask)
        print >> sio, """
            if (verbose) printf("running kernel_reduce_sum_%(pattern)s_%(name)s\\n");
            int n_shared = sizeof(float) * n_threads.x * n_threads.y * n_threads.z;
            if (verbose>1) printf("n_threads.x=%%d, n_threads.y=%%d, n_threads.z=%%d, nb_threads=%%d, n_blocks.x=%%d, n_blocks.y=%%d, nb_block=%%d, n_shared=%%d\\n",
                                  n_threads.x,n_threads.y,n_threads.z,
                                  n_threads.x*n_threads.y*n_threads.z,
                                  n_blocks.x,n_blocks.y,
                                  n_blocks.x*n_blocks.y, n_shared);
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

    def _k_decl(self, node, nodename, pattern = None, ndim = None, reduce_mask = None):
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
        if reduce_mask is None:
            reduce_mask = self.reduce_mask
        if ndim is None:
            ndim = len(reduce_mask)
        if pattern is None:
            pattern = ''.join(str(i) for i in reduce_mask)
        sio = StringIO.StringIO()

        print >> sio, """
            static __global__ void kernel_reduce_sum_%(pattern)s_%(nodename)s(
        """ %locals()
        for i in xrange(ndim):
            print >> sio, """
                    const int d%(i)s,
        """ %locals()
        print >> sio, """
                    const float *A,
        """ %locals()
        for i in xrange(ndim):
            print >> sio, """
                    const int sA%(i)s,
        """ %locals()
        print >> sio, """
                    float * Z
        """ %locals()
        for i in xrange(ndim - sum(reduce_mask)):
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
        __syncthreads(); // some kernel do multiple reduction.
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
/*Comment this optimization as it don't work on Fermi GPU.
  TODO: find why it don't work or put the GPU compute capability into the version
            // no sync because only one warp is running
            if(threadCount >32)
            {
                buf[threadNum] += buf[threadNum+16];
                buf[threadNum] += buf[threadNum+8];
                buf[threadNum] += buf[threadNum+4];
                buf[threadNum] += buf[threadNum+2];
                buf[threadNum] += buf[threadNum+1];
                if (threadNum == 0)
                {
                    %(z_pos)s = buf[0];
                }

            }
            else */
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

    #Threads must be organized as: threadNum%nb_reduce correspond to the same sum
    #nb_reduce<=warpSize
    def _k_reduce_buf_multiple(self, z_pos, nb_reduce):
        return """
        __syncthreads(); // some kernel do multiple reduction.
        buf[threadNum] = mysum;
        __syncthreads();

        // rest of function is handled by one warp
        if (threadNum < %(nb_reduce)s)
        {
            //round up all the partial sums into the first `nb_reduce` elements
            for (int i = threadNum + %(nb_reduce)s; i < threadCount; i += %(nb_reduce)s)
            {
                mysum += buf[i];
            }
            %(z_pos)s = mysum;
        }
        """ %locals()

    def c_code_reduce_ccontig(self, sio, node, name, x, z, fail):
        print >> sio, """
        {
          if(CudaNdarray_SIZE(%(x)s)==0){
            cudaMemset(CudaNdarray_DEV_DATA(%(z)s),0,sizeof(float));
          }else{
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_SIZE(%(x)s),
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(1);
            if (verbose) printf("running kernel_reduce_sum_ccontig_%(name)s n_threads.x=%%d, size=%%d, ndim=%%d\\n",
                                n_threads.x,CudaNdarray_SIZE(%(x)s),%(x)s->nd);
            int n_shared = sizeof(float) * n_threads.x;
            kernel_reduce_sum_ccontig_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    CudaNdarray_SIZE(%(x)s),
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_DEV_DATA(%(z)s));
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_sum_ccontig_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
         }
        }
        """ %locals()

    def c_code_reduce_1(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(1);
            %(makecall)s
        }
        """ %locals()

    def c_code_reduce_11(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
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
            %(makecall)s
        }
        """ %locals()

    def c_code_reduce_01X(self, sio, node, name, x, z, fail, N):
        """
        :param N: the number of 1 in the pattern N=1 -> 01, N=2 -> 011 N=3 ->0111
                  Work for N=1,2,3
        """
        assert N in [1,2,3]
        makecall = self._makecall(node, name, x, z, fail)
        N_pattern = ''.join(['1']*N)
        param_dim = ",".join(["CudaNdarray_HOST_DIMS(%(x)s)[%(i)s]"%locals() for i in xrange(N+1)])
        strides_dim = ",".join(["CudaNdarray_HOST_STRIDES(%(x)s)[%(i)s]"%locals() for i in xrange(N+1)])
        threads_y = """
            //get as many y threads as we can fit
            while (n_threads.x * (n_threads.y+1) <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y < CudaNdarray_HOST_DIMS(%(x)s)[%(N)s-1])
                    n_threads.y += 1;
                else
                    break;
            }
"""%locals()
        threads_z = """
            //get as many z threads as we can fit
            while (n_threads.x * n_threads.y * (n_threads.z+1) <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.z < CudaNdarray_HOST_DIMS(%(x)s)[%(N)s-2])
                    n_threads.z += 1;
                else
                    break;
            }
"""%locals()
        if len(self.reduce_mask)==2:
            threads_y = ''
            threads_z = ''
        if len(self.reduce_mask)==3:
            threads_z = ''
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[%(N)s],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            %(threads_y)s
            %(threads_z)s
            dim3 n_blocks(std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],NUM_VECTOR_OP_BLOCKS));
            %(makecall)s
        }
        """ %locals()

    def c_code_reduce_01(self, sio, node, name, x, z, fail):
        self.c_code_reduce_01X(sio, node, name, x, z, fail, 1)
    def c_code_reduce_011(self, sio, node, name, x, z, fail):
        self.c_code_reduce_01X(sio, node, name, x, z, fail, 2)
    def c_code_reduce_0111(self, sio, node, name, x, z, fail):
        self.c_code_reduce_01X(sio, node, name, x, z, fail, 3)

    def c_code_reduce_10(self, sio, node, name, x, z, fail):
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(1,
                std::min(CudaNdarray_HOST_DIMS(%(x)s)[1],
                    NUM_VECTOR_OP_BLOCKS));
            if (verbose) {
              fprintf(stderr,
                "running kernel_reduce_sum_10_%(name)s n_blocks=(%%i,%%i)\\n",
                n_blocks.x,
                n_blocks.y);
            }
            assert( CudaNdarray_HOST_DIMS(%(x)s)[1] == CudaNdarray_HOST_DIMS(%(z)s)[0]);
            int n_shared = sizeof(float) * n_threads.x;
            kernel_reduce_sum_010_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    1,
                    CudaNdarray_HOST_DIMS(%(x)s)[0],
                    CudaNdarray_HOST_DIMS(%(x)s)[1],
                    CudaNdarray_DEV_DATA(%(x)s),
                    1,
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_HOST_STRIDES(%(x)s)[1],
                    CudaNdarray_DEV_DATA(%(z)s),
                    1,
                    CudaNdarray_HOST_STRIDES(%(z)s)[0]
                    );
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_sum_010_%(name)s",
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
    def c_code_reduce_010(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        makecall_inner = self._makecall(node, name, x, z, fail, pattern="010_inner")
        pattern = ''.join(str(i) for i in self.reduce_mask)
        print >> sio, """
        {
            //int n_summations = CudaNdarray_HOST_DIMS(%(x)s)[0] * CudaNdarray_HOST_DIMS(%(x)s)[2];

            //if ((n_summations >= 15 * 32) && (CudaNdarray_HOST_DIMS(%(x)s)[2]>=16))
            if (1) // if the alternative is less buggy, consider not using this branch
            {
                // If there are a lot of summations to do, then we can use simple parallelization -
                // use each thread to do one sum.

                // we might as well launch blocks of 32 threads because that's the warp size.
                // we could schedule more threads if we were maxing out the gridsize below, but
                // the gridsize is way more than the physical hardware and I think 32 threads
                // on a huge grid is enough to fully use the hardware.
                dim3 n_threads(32,1,1);

                // We kindof reshape the input implicitly to something 4D:
                //  the shape A,B,C    ->   A, B, D, E
                //  where C <= D*E < C+32
                //  where E==32

                int A = CudaNdarray_HOST_DIMS(%(x)s)[0];
                int B = CudaNdarray_HOST_DIMS(%(x)s)[1];
                int C = CudaNdarray_HOST_DIMS(%(x)s)[2];
                int D = C/32;
                if (32*D < C) D+= 1;
                assert ((C <= 32*D) && (32*D < C+32));

                // The gridsize would ideally be (A, D).  But we do the following logic to make
                // sure we don't ask for a grid that is too big.
                dim3 n_blocks(A,D);
                if (n_blocks.x > NUM_VECTOR_OP_BLOCKS) n_blocks.x = NUM_VECTOR_OP_BLOCKS;
                if (n_blocks.x*n_blocks.y > NUM_VECTOR_OP_BLOCKS) n_blocks.y = NUM_VECTOR_OP_BLOCKS/n_blocks.x;
                int n_shared = 0;
                kernel_reduce_sum_010_AD_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                A,B,C,D,
                        CudaNdarray_DEV_DATA(%(x)s),
                        CudaNdarray_HOST_STRIDES(%(x)s)[0],
                        CudaNdarray_HOST_STRIDES(%(x)s)[1],
                        CudaNdarray_HOST_STRIDES(%(x)s)[2],
                        CudaNdarray_DEV_DATA(%(z)s),
                        CudaNdarray_HOST_STRIDES(%(z)s)[0],
                        CudaNdarray_HOST_STRIDES(%(z)s)[1]
                        );
                CNDA_THREAD_SYNC;
                cudaError_t sts = cudaGetLastError();
                if (cudaSuccess != sts)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                        "kernel_reduce_sum_010_%(name)s",
                        cudaGetErrorString(sts),
                        n_blocks.x,
                        n_blocks.y,
                        n_threads.x,
                        n_threads.y,
                        n_threads.z);
                    %(fail)s;
                }
            }
            else
            {
                int verbose = 2;

                  dim3 n_threads(std::min(32,CudaNdarray_HOST_DIMS(%(x)s)[2]));
                  while(    (n_threads.x*(n_threads.y+1)<=NUM_VECTOR_OP_THREADS_PER_BLOCK)
                         && (n_threads.y<CudaNdarray_HOST_DIMS(%(x)s)[1])){
                      n_threads.y++;
                  }

                  dim3 n_blocks(std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                                (int)NUM_VECTOR_OP_BLOCKS));
                  n_blocks.y = std::min(
                      ceil_intdiv(CudaNdarray_HOST_DIMS(%(x)s)[2],(int)n_threads.x),
                      (int)(NUM_VECTOR_OP_BLOCKS / n_blocks.x)
                      );
                if(std::min(std::min(CudaNdarray_HOST_STRIDES(%(x)s)[0],
                                     CudaNdarray_HOST_STRIDES(%(x)s)[1]),
                            CudaNdarray_HOST_STRIDES(%(x)s)[2])
                   ==CudaNdarray_HOST_STRIDES(%(x)s)[2]
                  && n_blocks.y==ceil_intdiv(CudaNdarray_HOST_DIMS(%(x)s)[2],(int)n_threads.x)){
                  if(verbose>1)
                    printf("n_block.x.1=%%d, n_block.x.2=%%d, n_block.y.1=%%d, n_block.y.2=%%d,\\n",
                           CudaNdarray_HOST_DIMS(%(x)s)[0],NUM_VECTOR_OP_BLOCKS,
                           ceil_intdiv(CudaNdarray_HOST_DIMS(%(x)s)[2],(int)n_threads.x),
                                       (int)(NUM_VECTOR_OP_BLOCKS / n_blocks.x));
                  assert(n_threads.x<=32);
                  %(makecall_inner)s
                }else{
                  n_threads.x = std::min(CudaNdarray_HOST_DIMS(%(x)s)[1],
                                  (int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                  n_blocks.x = std::min(CudaNdarray_HOST_DIMS(%(x)s)[0], (int)NUM_VECTOR_OP_BLOCKS);
                  n_blocks.y = std::min(
                      CudaNdarray_HOST_DIMS(%(x)s)[2],
                      (int)(NUM_VECTOR_OP_BLOCKS / n_blocks.x)
                      );
                  %(makecall)s
                }
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
            }
        }
        """ %locals()

    def c_code_reduce_0101(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[3],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            while (n_threads.x * n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[1]) break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;
            dim3 n_blocks(CudaNdarray_HOST_DIMS(%(x)s)[0], CudaNdarray_HOST_DIMS(%(x)s)[2]);
            %(makecall)s
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
            while (n_blocks.x * (n_blocks.y+1) <= NUM_VECTOR_OP_BLOCKS && n_blocks.y <= CudaNdarray_HOST_DIMS(%(x)s)[2])
            {
                n_blocks.y += 1;
            }
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

    def c_code_reduce_0011(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;

            dim3 n_blocks(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                        NUM_VECTOR_OP_BLOCKS));

            while (n_blocks.x * n_blocks.y <= NUM_VECTOR_OP_BLOCKS && n_blocks.y < CudaNdarray_HOST_DIMS(%(x)s)[1])
            {
                n_blocks.y += 1;
            }

            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[3],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            while (n_threads.x * n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK
                   && n_threads.y < CudaNdarray_HOST_DIMS(%(x)s)[2]
                   && n_threads.x * n_threads.y * sizeof(float) <=(15*1024-200))
            {
                n_threads.y += 1;
            }

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
        makecall = self._makecall(node, name, x, z, fail)
        print >> sio, """
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[3],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));

            while (n_threads.x * (n_threads.y+1) <= NUM_VECTOR_OP_THREADS_PER_BLOCK) ++n_threads.y;
            if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[2])
                n_threads.y = CudaNdarray_HOST_DIMS(%(x)s)[2];

            while (n_threads.x * n_threads.y * (n_threads.z+1) <= NUM_VECTOR_OP_THREADS_PER_BLOCK) ++n_threads.z;
            if (n_threads.z > 64)
                n_threads.z = 64;
            if (n_threads.z > CudaNdarray_HOST_DIMS(%(x)s)[0])
                n_threads.z = CudaNdarray_HOST_DIMS(%(x)s)[0];

            dim3 n_blocks(CudaNdarray_HOST_DIMS(%(x)s)[1]);
            %(makecall)s
        }
        """ %locals()

    def c_code_cache_version(self):
        return (22,)

    def c_support_code_apply(self, node, nodename):
        sio = StringIO.StringIO()
        nd_in = len(self.reduce_mask)
        if all(i==1 for i in self.reduce_mask):
            #this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]')
            print >> sio, """
            static __global__ void kernel_reduce_sum_ccontig_%(nodename)s(
                    const unsigned int d0,
                    const float *A,
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
                    mysum += A[i0];
                }
                %(reducebuf)s
            }
            """ %locals()
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
        #01, 011, 0111
        if 0 == self.reduce_mask[0] and all(self.reduce_mask[1:]) and nd_in in[2,3,4]:
            # this kernel uses one block for each row.
            # threads per block for each element per row.

            N_pattern = ''.join(['1']*(nd_in-1))
            if nd_in==2:
                for_i1 = "for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)"
                for_i2="int i2=0, sA2=0;"
                for_i3="int i3=0, sA3=0;"
            if nd_in==3:
                for_i1 = "for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)"
                for_i2 = "for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)"
                for_i3="int i3=0, sA3=0;"
            if nd_in==4:
                for_i1 = "for (int i1 = threadIdx.z; i1 < d1; i1 += blockDim.z)"
                for_i2 = "for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)"
                for_i3 = "for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)"

            reducebuf = self._k_reduce_buf('Z[i0 * sZ0]')
            param_dim = ",".join(["const int d%(i)s"%locals() for i in xrange(nd_in)])
            param_strides = ",".join(["const int sA%(i)s"%locals() for i in xrange(nd_in)])
            decl = self._k_decl(node,nodename)
            init = self._k_init(node,nodename)
            print >> sio, """
            %(decl)s{
                %(init)s
                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x){
                  mysum = 0;
                  %(for_i1)s{
                    %(for_i2)s{
                      %(for_i3)s{
                        float Ai = A[i3 * sA3 + i2 * sA2 + i1 * sA1 + i0 * sA0];
                        mysum += Ai;
                      }
                    }
                  }
                  %(reducebuf)s
                }
            }
            """ %locals()
        if self.reduce_mask == (0,1,0) or self.reduce_mask == (1,0):
            # this kernel uses one block for each column,
            # threads per block for each element per column.

            #TODO: This kernel is pretty inefficient in terms of reading, because if A is
            #      c_contiguous (typical case) then each warp is accessing non-contigous
            #      memory (a segment of a column).
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i2*sZ1]')
            print >> sio, """
            static __global__ void kernel_reduce_sum_010_%(nodename)s(
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
                    for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                    {
                        float mysum = 0.0f;
                        for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                        {
                            mysum += A[i0 * sA0 + i1 * sA1 + i2 * sA2];
                        }
                        %(reducebuf)s
                    }
                }

            }
            """ %locals()
        if self.reduce_mask == (0,1,0):
            print >> sio, """
            static __global__ void kernel_reduce_sum_010_AD_%(nodename)s(
                    const int A,
                    const int B,
                    const int C,
                    const int D,
                    //const int E, // THIS is 32
                    const float *X, const int sX0, const int sX1, const int sX2,
                    float * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                float mysum = 0.0f;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int a = blockIdx.x; a < A; a += gridDim.x)
                {
                    for (int i2_D = blockIdx.y; i2_D < D; i2_D += gridDim.y)
                    {
                        int c = i2_D * 32 + threadIdx.x;
                        if (c < C)
                        {
                            mysum = 0;
                            for (int b = 0; b < B; ++b)
                            {
                                mysum += X[a * sX0 + b * sX1 + c * sX2];
                            }
                            Z[a * sZ0 + c * sZ1] = mysum;
                        }
                    }
                }

            }
            """ %locals()
        if self.reduce_mask == (0,1,0):
            #
            # This kernel is optimized when the inner most dimensions have the smallest stride.

            # this kernel uses one block for multiple column(up to 32TODO),
            # threads per block for each element per column.

#thread.x = dim 2 contiguous
#thread.y = dim 1
#block.x = dim 0
#block.y = dim 1 rest
            init = self._k_init(node,nodename)
            decl = self._k_decl(node, nodename, pattern="010_inner")
            reducebuf = self._k_reduce_buf_multiple('Z[i0 * sZ0 + i2*sZ1]','blockDim.x')
            reducebuf = self._k_reduce_buf_multiple('Z[i0 * sZ0 + i2*sZ1]','blockDim.x')
            print >> sio, """
            %(decl)s
            {
             if(warpSize<blockDim.x){
               //TODO: set error code
               Z[0] = -666;
               return;
              }

              %(init)s
              for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
              {
                for (int i2 = blockIdx.y*blockDim.x+threadIdx.x; i2 < d2; i2 += gridDim.y*blockDim.x)
                 {
                  for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)
                  {
                      mysum += A[i0 * sA0 + i1 * sA1 + i2 * sA2];
                  }
                  %(reducebuf)s
                 }
              }
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
        if self.reduce_mask == (0,0,1,1):
            # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i1 * sZ1]')
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            print >> sio, """
            %(decl)s
            {
                %(init)s

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y)
                    {
                        float mysum = 0.0f;
                    for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            mysum += A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3];
                        }
                    }
                        %(reducebuf)s
                    }
                }
            }
            """ %locals()
        if self.reduce_mask == (0,1,0,1):
            # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i2 * sZ1]')
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            print >> sio, """
            %(decl)s
            {
                %(init)s

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                    {
                        float mysum = 0.0f;
                    for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            mysum += A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3];
                        }
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

class GpuReshape(tensor.Reshape, GpuOp):
    """
    Implement Reshape on the gpu.
    """
    # __hash__, __eq__, __str__ come from tensor.Subtensor
    def make_node(self, x, shp):
        host_reshaped = host_from_gpu(x).reshape(shp,ndim=self.ndim)
        return Apply(self, [x, shp], [CudaNdarrayType(host_reshaped.broadcastable)()])
    def perform(self, node, inp, out_):
        x, shp = inp
        out, = out_
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to Reshape.perform has incorrect length %i'
                    ', should be %i' % (len(shp), self.ndim), shp)
        out[0] = x.reshape(tuple(shp))

class GpuSubtensor(tensor.Subtensor, GpuOp):
    """
    Implement subtensor on the gpu.
    """
    # __hash__, __eq__, __str__ come from tensor.Subtensor
    def make_node(self, x, *inputs):
        assert isinstance(x.type, CudaNdarrayType)
        rval = tensor.Subtensor.make_node(self, x, *inputs)
        otype = CudaNdarrayType(rval.outputs[0].type.broadcastable)
        return Apply(self, [x]+rval.inputs[1:], [otype()])

    def perform(self, node, inputs, out_):
        out, = out_
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

class GpuAdvancedSubtensor1(tensor.AdvancedSubtensor1, GpuOp):
    """
    Implement AdvancedSubtensor1 on the gpu.
    """
    def make_node(self, x, ilist):
        x_ = as_cuda_ndarray_variable(x)
        ilist_ = tensor.as_tensor_variable(ilist)
        if ilist_.type.dtype[:3] not in ('int', 'uin'):
            raise TypeError('index must be integers')
        if ilist_.type.broadcastable != (False,):
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')

        return Apply(self, [x_, ilist_], [x_.type()])

    def perform(self, node, inp, out_):
        # This don't work as CudaNdarray_Subscript() don't support it.
        #super(GpuAdvancedSubtensor1, self).perform(node, inp, out_)
        x, idx = inp
        out, = out_
        o = cuda_ndarray.cuda_ndarray.CudaNdarray.zeros((len(idx),)+x.shape[1:])
        for (j,i) in enumerate(idx):
            o[j] = x[i]
        out[0] = o

class GpuAdvancedIncSubtensor1(tensor.AdvancedIncSubtensor1, GpuOp):
    """
    Implement AdvancedIncSubtensor1 on the gpu.
    """
    def make_node(self, x, y, ilist):
        x_ = as_cuda_ndarray_variable(x)
        y_ = as_cuda_ndarray_variable(y)
        ilist_ = tensor.as_tensor_variable(ilist)

        assert x_.type.dtype == y_.type.dtype
        assert x_.type.ndim == y_.type.ndim

        if ilist_.type.dtype[:3] not in ('int', 'uin'):
            raise TypeError('index must be integers')
        if ilist_.type.broadcastable != (False,):
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        if x_.type.broadcastable[0]:
            # the caller should have made a copy of x len(ilist) times
            raise TypeError('cannot index into a broadcastable dimension')

        return Apply(self, [x_, y_, ilist_], [x_.type()])

    #def perform(self, node, inp, out_):
        # CudaNdarray_Subscript() don't support Advanced slicing.
        # so we use the parent version that loop on each indices.

class GpuIncSubtensor(tensor.IncSubtensor, GpuOp):
    """
    Implement IncSubtensor on the gpu.
    """
    def make_node(self, x, y, *inputs):
        assert isinstance(x.type, CudaNdarrayType)
        assert isinstance(y.type, CudaNdarrayType)
        rval = tensor.IncSubtensor.make_node(self, x, y, *inputs)
        return Apply(self, [x,y]+rval.inputs[2:], [x.type()])

class GpuFlatten(tensor.Flatten, GpuOp):
    """
    Implement Flatten on the gpu.
    """
    def make_node(self, x ):
        assert isinstance(x.type, CudaNdarrayType)
        rval = tensor.Flatten.make_node(self, x)
        host_out_broadcastable = rval.outputs[0].type.broadcastable
        out_type = CudaNdarrayType(broadcastable=host_out_broadcastable)
        return Apply(self, [x], [out_type()])

class GpuShape(tensor.Shape, GpuOp):
    """
    Implement Shape on the gpu.
    """
    def make_node(self, x):
        return Apply(self, [x], [tensor.lvector()])
gpu_shape = GpuShape()

class GpuJoin(tensor.Join, GpuOp):
    """
    Implement Join on the gpu.
    """
    def make_node(self, *axis_and_tensors):
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        if not tensors:
            raise ValueError('Cannot join an empty list of tensors')
        are_instances = [isinstance(x.type, CudaNdarrayType) \
                                                for x in tensors]
        assert numpy.all(are_instances)

        # no conversion needed, we just checked everything was
        # a CNDA var
        as_tensor_variable_args = tensors

        output_maker = \
                lambda bcast: CudaNdarrayType(broadcastable=bcast)()

        return tensor.Join._make_node_internal(self,
                        axis, tensors,
                        as_tensor_variable_args, output_maker)

    def perform(self, node, axis_and_tensors, out_):
        out, = out_
        axis, cndas = axis_and_tensors[0], axis_and_tensors[1:]
        # In case axis is numpy.int8 and has no __index__() method
        axis = int(axis)

        # compute size/shape
        width_sum = 0
        template_shape = cndas[0].shape
        for cnda in cndas:
            width_sum += cnda.shape[axis]
            # and while we're at it, check that shapes match
            tmp_shape = list(cnda.shape)
            # dimension in "axis" can be different, so make equal for ==
            tmp_shape[axis] = template_shape[axis]
            if tuple(tmp_shape) != template_shape:
                raise ValueError, "Shape of input CudaNdarrays must agree except for the 'axis' dimension"

        if len(template_shape) != node.outputs[0].type.ndim:
            raise ValueError, "Number of dimension of input tensors disagree with dimensions passed at graph creation time."

        # final shape must be the same as all input tensors
        # except for the "axis" dimension, so we can simply
        # copy the shape of the first one
        final_shape = list(cndas[0].shape)
        final_shape[axis] = width_sum

        # just to be explicit, check that dim=1 for broadcastable
        # dimensions
        for i, bcastable in enumerate(node.outputs[0].type.broadcastable):
            assert not bcastable or final_shape[i] == 1, "Broadcastable dimension but dim != 1, this is invalid"

        rval = cuda_ndarray.cuda_ndarray.CudaNdarray.zeros(final_shape)

        curpos = 0

        # we use a [:] (copy all) slice for all dimensions
        # except for 'axis'

        def construct_slices(curlen):
            slices = [slice(None,None,None) for i in \
                            range(len(template_shape))]
            slices[axis] = slice(curpos,curpos+curlen,None)
            return tuple(slices)

        for i, cnda in enumerate(cndas):
            curlen = cnda.shape[axis]
            rval.__setitem__(construct_slices(curlen), cnda)
            curpos += curlen

        out[0] = rval

gpu_join = GpuJoin()

class GpuAlloc(GpuOp):
    """
    Implement Alloc on the gpu.
    """
    def __init__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, value, *shape):
        #if their is unneeded transfert generated by the next line
        #the optimizer will remove them.
        v = as_cuda_ndarray_variable(value)
        sh = [tensor.as_tensor_variable(s) for s in shape]
        if v.ndim != len(shape):
            raise TypeError('GpuAlloc requires value of same dimensions as shape', value, len(shape))

        bcast = []
        for s in sh:
            if s.type.dtype[:3] not in ('int', 'uin'):
                raise TypeError('Shape arguments must be integers', s)
            # if s is constant 1, then we're broadcastable in that dim
            try:
                const_shp = tensor.get_constant_value(s)
            except TypeError:
                const_shp = None
            bcast.append(numpy.all(1 == const_shp))
        otype = CudaNdarrayType(dtype='float32', broadcastable=bcast)
        return Apply(self, [v]+sh, [otype()])

    def perform(self, node, inputs, out_):
        out, = out_
        v = inputs[0]
        sh = tuple([int(i) for i in inputs[1:]])
        if out[0] is None or out[0].shape != sh:
            out[0] = cuda_ndarray.cuda_ndarray.CudaNdarray.zeros(sh)
        out[0][...] = v # broadcast v to fill us up

    def c_code(self, node, name, inputs, out_, sub):
        out, = out_
        fail = sub['fail']
        value = inputs[0]
        shps = inputs[1:]
        nd = len(shps)
        str =  "int dims[%(nd)s];\n"%locals()
        for idx,sh in enumerate(shps):
            str += "dims[%(idx)s] = PyInt_AsLong((PyObject*)%(sh)s);\n"%locals()

        str += "if(%(out)s==NULL\n"%locals()
        for idx,sh in enumerate(shps):
            str += "||CudaNdarray_HOST_DIMS(%(out)s)[%(idx)s]!=dims[%(idx)s]"%locals()
        str+="""){
        Py_XDECREF(%(out)s);
        %(out)s= (CudaNdarray*)CudaNdarray_New();
        CudaNdarray_alloc_contiguous(%(out)s, %(nd)s, dims);
    }
    if (CudaNdarray_CopyFromCudaNdarray(%(out)s, %(value)s, true))
    {
    %(fail)s;
    }
"""%locals()
        return str

    def infer_shape(self, node, input_shapes):
        return [node.inputs[1:]]

    def grad(self, inputs, grads):
        gout, = grads
        return [None for i in inputs]

    def c_code_cache_version(self):
        return (3,)

    def do_constant_folding(self, node):
        for client in node.outputs[0].clients:
            if client[0] == 'output':
                # If the output is a constant, it will have to be deepcopied
                # each time the function is called.  So we do not fold.
                return False
            elif (not isinstance(client[0], basestring)
                    and isinstance(client[0].op, (
                        tensor.IncSubtensor,
                        tensor.AdvancedIncSubtensor1,
                        GpuIncSubtensor,
                        GpuAdvancedIncSubtensor1
                        ))):
                return False
        return True

gpu_alloc = GpuAlloc()


class GpuContiguous(GpuOp):
    """
    Always return a c contiguous output. Copy the input only if it is
    not already c contiguous.
    """
    view_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return self.__class__.__name__

    def make_node(self, input):
        input = as_cuda_ndarray_variable(input)
        return Apply(self, [input], [input.type()])

    def c_code(self, node, name, inp, out, sub):
        input, = inp
        z, = out
        fail = sub['fail']
        str = """
        {
            if (CudaNdarray_is_c_contiguous(%(input)s)){
                Py_XDECREF(%(z)s);
                %(z)s = %(input)s;
                Py_INCREF(%(z)s);

            } else if ((NULL == %(z)s)"""%locals()
        for i in xrange(len(node.inputs[0].type.broadcastable)):
            str += "\n|| (CudaNdarray_HOST_DIMS(%(input)s)[%(i)s] != CudaNdarray_HOST_DIMS(%(z)s)[%(i)s])"%locals()
        str += """)
            {
                Py_XDECREF(%(z)s);
                %(z)s = (CudaNdarray*)CudaNdarray_Copy(%(input)s);
                if (!%(z)s)
                {
                    %(fail)s;
                }
            }else if(CudaNdarray_CopyFromCudaNdarray(%(z)s,%(input)s)){
                %(fail)s;
            }
        }
        """%locals()
        return str

    def c_code_cache_version(self):
        return (1,)

gpu_contiguous = GpuContiguous()


def tensordot(a, b, axes=2):
    """
    Implementation of tensordot that reduces to a regular matrix product.

    This allows tensordot to be GPU accelerated, which isn't possible
    with the default Theano implementation (which is just a wrapper
    around numpy.tensordot). based on code from Tijmen Tieleman's gnumpy
    http://www.cs.toronto.edu/~tijmen/gnumpy.html
    """
    if numpy.isscalar(axes):
        # if 'axes' is a number of axes to multiply and sum over (trailing axes
        # of a, leading axes of b), we can just reshape and use dot.
        outshape = tensor.concatenate([a.shape[:a.ndim - axes],
                                      b.shape[axes:]])
        outndim = a.ndim + b.ndim - (2 * axes)
        a_reshaped = a.reshape((tensor.prod(a.shape[:a.ndim - axes]),
                                tensor.prod(a.shape[a.ndim - axes:])))
        b_reshaped = b.reshape((tensor.prod(b.shape[:axes]),
                                tensor.prod(b.shape[axes:])))
        assert a_reshaped.ndim == 2
        assert b_reshaped.ndim == 2
        # We use _dot22 here because:
        #   - we know that the number of dimensions will be 2
        #   - it makes it possible for the computation to be moved to GPU
        # When cuda.opt.local_gpu_tensordot is applied, it is too late
        # for the usual blas optimizations to take place.
        # This will change if we decide to get rid of tensor.tensordot,
        # and always use this version.
        return tensor.blas._dot22(a_reshaped, b_reshaped).reshape(
                outshape, ndim=outndim)
    elif len(axes) == 2:
        # if 'axes' is a pair of axis lists, we first shuffle the axes of a and
        # b to reduce this to the first case (note the recursion).
        a_other, b_other = tuple(axes[0]), tuple(axes[1])
        num_axes = len(a_other)
        a_order = (tuple(x for x in tuple(xrange(a.ndim)) if x not in a_other)
                + a_other)
        b_order = (b_other
                + tuple(x for x in tuple(xrange(b.ndim)) if x not in b_other))
        a_shuffled = a.dimshuffle(a_order)
        b_shuffled = b.dimshuffle(b_order)
        return tensordot(a_shuffled, b_shuffled, num_axes)
    else:
        raise ValueError(
            "Axes should be scalar valued or a list/tuple of len 2.",
            axes)

# Those are predifined CudaNdarrayType as done in tensor.basic
# Useful mostly for test as the gpu op are inserted automatically...

fscalar = CudaNdarrayType(dtype='float32', broadcastable=())
def scalar(name = None, dtype = None):
    """Return a symbolic scalar variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=())
    return type(name)

fvector = CudaNdarrayType(dtype='float32', broadcastable=(False, ))
def vector(name = None, dtype = None):
    """Return a symbolic vector variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(False, ))
    return type(name)

fmatrix = CudaNdarrayType(dtype='float32', broadcastable=(False, False))
def matrix(name = None, dtype = None):
    """Return a symbolic matrix variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(False, False))
    return type(name)

frow = CudaNdarrayType(dtype='float32', broadcastable=(True, False))
def row(name = None, dtype = None):
    """Return a symbolic row variable (ndim=2, broadcastable=[True,False]).
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(True, False))
    return type(name)

fcol = CudaNdarrayType(dtype='float32', broadcastable=(False, True))
def col(name = None, dtype = None):
    """Return a symbolic column variable (ndim=2, broadcastable=[False,True]).
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(False, True))
    return type(name)

ftensor3 = CudaNdarrayType(dtype='float32', broadcastable=(False,)*3)
def tensor3(name=None, dtype=None):
    """Return a symbolic 3-D variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(False, False, False))
    return type(name)

ftensor4 = CudaNdarrayType(dtype='float32', broadcastable=(False,)*4)
def tensor4(name=None, dtype=None):
    """Return a symbolic 4-D variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(False, False, False, False))
    return type(name)


@theano.compile.profilemode.register_profiler_printer
def profile_printer(fct_name, compile_time, fct_call_time, fct_call,
                    apply_time, apply_cimpl, message, outputs_size,
                    other_time):
    if any([x[1].op.__class__.__name__.lower().startswith("gpu") for x in apply_time.keys()]):
        local_time = sum(apply_time.values())
        print
        print 'Some info useful for gpu:'

        cpu=0
        gpu=0
        trans=0
        for (_,node),t  in apply_time.items():
            if isinstance(node.op.__class__.__name__,(HostFromGpu, GpuFromHost)):
                trans += t
            elif node.op.__class__.__name__.lower().startswith("gpu"):
                gpu += t
            else:
                cpu += t
        print
        print "    Spent %.3fs(%.3f%%) in cpu Op, %.3fs(%.3f%%) in gpu Op and %.3fs(%.3f%%) transfert Op"%(
            cpu, cpu/local_time*100, gpu, gpu/local_time*100, trans, trans/local_time*100)

        print
        print "    Theano function input that are float64"
        print "    <fct name> <input name> <input type> <str input>"
        for fct in fct_call.keys():
            for i in fct.input_storage:
                if hasattr(i.type, 'dtype') and i.type.dtype=='float64':
                    print '        ', fct.name, i.name, i.type, i

        print
        print "    List of apply that don't have float64 as input but have float64 in outputs"
        print "    (Useful to know if we forgot some cast when using floatX=float32 or gpu code)"
        print '    <Apply> <Apply position> <fct name> <inputs type> <outputs type>'
        for fct in fct_call.keys():
            for idx, node in enumerate(fct.maker.env.toposort()):
                if any(hasattr(i,'dtype') and i.dtype=='float64' for i in node.outputs) and not any(hasattr(i,'dtype') and i.dtype=='float64' for i in node.inputs):
                    print '        ', str(node), idx, fct.name, str([getattr(i,'dtype',None) for i in node.inputs]),str([getattr(i,'dtype',None) for i in node.outputs])
