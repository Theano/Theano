import StringIO
import numpy

from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar

from .type import CudaNdarrayType
from .type_support import filter as type_support_filter

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

    def __init__(self, scalar_op, inplace_pattern):
        self.scalar_op = scalar_op
        self.inplace_pattern = inplace_pattern
        self.destroy_map = dict((o, [i]) for o, i in inplace_pattern.items())
        if scalar_op.nin > 0:
            self.ufunc = numpy.frompyfunc(scalar_op.impl, scalar_op.nin, scalar_op.nout)
        else:
            self.ufunc = None
        self._rehash()

    def __getstate__(self):
        d = copy(self.__dict__)
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
        return type(self) == type(other) and (self.scalar_op == other.scalar_op)

    def _rehash(self):
        items = self.inplace_pattern.items()
        items.sort()
        tuple_items = tuple([k for k,v in items] + [(tuple(v) if isinstance(v, (tuple, list)) else v) for k,v in items])
        h = hash('Elemwise') ^ hash(self.scalar_op) ^ hash(tuple_items)
        assert h == getattr(self,'_hashval', h)
        self._hashval = h

    def __hash__(self):
        return self._hashval

    def __str__(self):
        if self.inplace_pattern:
            items = self.inplace_pattern.items()
            items.sort()
            return "GpuElemwise{%s}%s" % (self.scalar_op, str(items))
        else:
            return "GpuElemwise{%s}" % (self.scalar_op)

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

    def c_support_code(self):
        return """
        #define INTDIV_POW2(a, b) (a >> b)
        #define INTMOD_POW2(a, b) (a & ((1<<b)-1))
        """

    def c_src_kernel(self, node, nodename):
        nd = node.outputs[0].type.ndim
        sio = StringIO.StringIO()
        #print 'C_SRC_KERNEL', sio.getvalue()

        def _logical_scalar(x):
            return all(x.type.broadcastable)

        print >> sio, "static __global__ void kernel_%s(unsigned int numEls," %nodename
        print >> sio, "\t ", ", ".join("unsigned int log2_dim%i" % i for i in xrange(nd))
        #declare inputs
        for ipos, i in enumerate(node.inputs):
            print >> sio, "\t,", ", ".join("int i%i_str_%i" % (ipos, d) for d in xrange(nd))
            print >> sio, "\t,", "const float * i%i_data" % ipos
        #declare outputs
        for ipos, i in enumerate(node.outputs):
            print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
            print >> sio, "\t,", "float * o%i_data" % ipos
        print >> sio, "\t)\n{"
        print >> sio, "    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        print >> sio, "    const unsigned int numThreads = blockDim.x * gridDim.x;"

        # For each input that is a scalar which has been broadcasted to a tensor,
        #     load it into a local variable
        for ipos, i in enumerate(node.inputs):
            if _logical_scalar(i):
                print >> sio, "    const float ii_i%i_value = i%i_data[0];" % (ipos, ipos)

        
        #TODO: insert code to check for strides of 1, and use a different loop
        
        #loop over the elements to be treated by this kernel call
        print >> sio, "    for (unsigned int i = idx; i < numEls; i += numThreads) {"
        # calculate the data pointers for all arguments
        print >> sio, "        unsigned int ii = i;"
        for ipos, i in enumerate(node.inputs):
            if not _logical_scalar(i):
                print >> sio, "        const float * ii_i%i_data = i%i_data;" % (ipos, ipos)
        for ipos, i in enumerate(node.outputs):
            print >> sio, "        float * ii_o%i_data = o%i_data;" % (ipos, ipos)
        for d in xrange(nd-1, -1, -1):
            if d > 0:
                print >> sio, "        unsigned int pos%i = INTMOD_POW2(ii, log2_dim%i);" %(d, d)
                print >> sio, "        ii = INTDIV_POW2(ii, log2_dim%i);" %d
            else:
                print >> sio, "        unsigned int pos%i = ii;" %d
            for ipos, i in enumerate(node.inputs):
                if not _logical_scalar(i):
                    print >> sio, "        ii_i%i_data += pos%i * i%i_str_%i;" % (ipos, d, ipos, d)
            for ipos, i in enumerate(node.outputs):
                print >> sio, "        ii_o%i_data += pos%i * o%i_str_%i;" % (ipos, d, ipos, d)

        # perform the scalar operation on the input and output references
            if d == 0:
                #TODO: What if the scalar_op needs support_code??
                task_code = self.scalar_op.c_code(
                        Apply(self.scalar_op,
                            [scalar.Scalar(dtype = input.type.dtype)() for input in node.inputs],
                            [scalar.Scalar(dtype = output.type.dtype)() for output in node.outputs])
                        , nodename + '_scalar_'
                        , [('ii_i%i_value' if _logical_scalar(i) else 'ii_i%i_data[0]')%ipos for ipos, i in enumerate(node.inputs)] 
                        , ['ii_o%i_data[0]'%ipos for ipos, i in enumerate(node.outputs)] 
                        , sub=dict(fail='return;')) #TODO: set a failure code somehow!!!
                print >> sio, "       ", task_code
        print >> sio, "    }"

        #TODO: insert runtime stride checks that select the best loop order either here, or in
        # the host code that launched the  kernel (host code probably better spot)

        #indent = " "*(4*d+7)
        #for ipos, i in enumerate(node.inputs):
            #print >> sio, indent, "const float * i%i" % ipos, '= i%i_data', ''
        print >> sio, "}"

        #print sio.getvalue()
        return sio.getvalue()

    def c_support_code_apply(self, node, nodename):
        return self.c_src_kernel(node, nodename) + \
                self.c_src_callkernel(node, nodename)

    def c_src_callkernel(self, node, nodename):
        nd = node.outputs[0].type.ndim
        d = dict()
        #input_params and output_params go into the function declaration/definition
        input_params = ", ".join("const float * i%i_data, const int * i%i_str"%(ipos, ipos) 
                for ipos in xrange(len(node.inputs)))
        output_params = ", ".join("float * o%i_data, const int * o%i_str"%(ipos, ipos) 
                for ipos in xrange(len(node.outputs)))

        #input_args and output_args go into the recursive call.
        input_args = ", ".join("i%i_data, i%i_str"%(ipos, ipos) 
                for ipos in xrange(len(node.inputs)))
        output_args = ", ".join("o%i_data, o%i_str"%(ipos, ipos) 
                for ipos in xrange(len(node.outputs)))

        # kernel_call_args are used to invoke the cuda kernel
        kernel_call_args = ["numEls"]
        kernel_call_args.extend("log2_dims[%i]"%di for di in xrange(nd))
        for ipos in xrange(len(node.inputs)):
            strides = ", ".join("i%i_str[%i]"%(ipos, di) for di in xrange(nd))
            kernel_call_args.append( "%s, i%i_data" % (strides, ipos))
        for ipos in xrange(len(node.outputs)):
            strides = ", ".join("o%i_str[%i]"%(ipos, di) for di in xrange(nd))
            kernel_call_args.append( "%s, o%i_data" % (strides, ipos))
        kernel_call_args = ", ".join(kernel_call_args)

        # the data_pointer_increments are inserted after each recursive call
        data_ptr_inc = []
        for ipos in xrange(len(node.inputs)):
            data_ptr_inc.append("i%i_data += (1<< log2_dim) * i%i_str[d]" %(ipos, ipos))
        for ipos in xrange(len(node.outputs)):
            data_ptr_inc.append("o%i_data += (1<< log2_dim) * o%i_str[d]" %(ipos, ipos))
        data_ptr_inc = ";\n".join(data_ptr_inc)


        d.update(locals())
        return """

        static void callkernel_%(nodename)s(const unsigned int numEls, const int d,
            const int * dims, int * log2_dims,
            %(input_params)s,
            %(output_params)s)
        {
            if (d == %(nd)s)
            {
                int threads_per_block = std::min(numEls, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                //a ceil would be better here
                int n_blocks = std::min(numEls/threads_per_block + (numEls %% threads_per_block?1:0), (unsigned int)NUM_VECTOR_OP_BLOCKS);
                kernel_%(nodename)s<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);
                //std::cerr << "ADDCALL a str" << i0_str[0] << " "<< i0_str[1] << "\\n";
                //std::cerr << "ADDCALL a data" << i0_data << "\\n";
                //std::cerr << "ADDCALL b str" << i1_str[0] << " "<< i1_str[1] << "\\n";
                //std::cerr << "ADDCALL b data" << i1_data << "\\n";
                //std::cerr << "ADDCALL z str" << o0_str[0] << " "<< o0_str[1] << "\\n";
                //std::cerr << "ADDCALL z data" << o0_data << "\\n";
            }
            else
            {
                //std::cerr << "_ADDCALL d " << d << "\\n";
                unsigned int dim_d = dims[d];
                //std::cerr << "_ADDCALL dim_d " << dim_d << "\\n";
                int log2_dim = 0;
                while(dim_d)
                {
                        //std::cerr << "___ADDCALL d " << d << " " << dim_d << "\\n";
                        if (dim_d&1)
                        {
                            log2_dims[d] = log2_dim; 
                            //std::cerr << "___ADDCALL a str" << i0_str[0] << " "<< i0_str[1] << "\\n";
                            //std::cerr << "___ADDCALL a data" << i0_data << "\\n";
                        //std::cerr << "___ADDCALL b str" << i1_str[0] << " "<< i1_str[1] << "\\n";
                        //std::cerr << "___ADDCALL b data" << i1_data << "\\n";
                        //std::cerr << "___ADDCALL z str" << o0_str[0] << " "<< o0_str[1] << "\\n";
                        //std::cerr << "___ADDCALL z data" << o0_data << "\\n";
                        callkernel_%(nodename)s(numEls * (1<<log2_dim), d+1, dims, log2_dims, 
                            %(input_args)s,
                            %(output_args)s);

                        %(data_ptr_inc)s;
                        //i0_data += (1 << log2_dim) * i0_str[d];
                        //i1_data += (1 << log2_dim) * i1_str[d];
                        //o0_data += (1 << log2_dim) * o0_str[d];
                    }
                    log2_dim += 1;
                    dim_d >>= 1;
                }
            }
        }
        """ %d

    def c_code(self, node, nodename, inputs, outputs, sub):
        d = dict(sub)
        nd = node.outputs[0].type.ndim
        d.update(locals())
        sio = StringIO.StringIO()
        nin = len(inputs)
        nout = len(outputs)
        fail = sub['fail']
        opname = str(self.scalar_op)
        initial_dims = ','.join('1' for i in xrange(nd))
        if 1 or self.scalar_op == scalar.pow:
            print >> sio, """
        //std::cerr << "C_CODE %(opname)s START\\n";
        //standard elemwise size checks
            """ %locals()
        print >> sio, """
        int dims[%(nd)s] = {%(initial_dims)s};
        """ %locals()
        for iname in inputs:
            print >> sio, """
        //std::cerr << "C_CODE %(opname)s checking input %(iname)s\\n";
        if (%(nd)s != cnda_%(iname)s->nd)
        {
            PyErr_Format(PyExc_TypeError, "need %(nd)s dims, not %%i", cnda_%(iname)s->nd);
            %(fail)s;
        }
        for (int i = 0; i< %(nd)s; ++i)
        {
            dims[i] = (dims[i] == 1) ? cnda_%(iname)s->dim[i] : dims[i];
            if ((cnda_%(iname)s->dim[i] != 1) && (dims[i] != cnda_%(iname)s->dim[i]))
            {
                //std::cerr << "C_CODE %(opname)s checking input %(iname)s failed\\n";
                PyErr_Format(PyExc_TypeError, "GpuElemwise input has incompatible dim[%%i] == %%i, where output has size %%i",
                    i,
                    cnda_%(iname)s->dim[i],
                    dims[i]
                    );
                %(fail)s;
            }
        }
            """ %locals()

        for oname in outputs:
            print >> sio, """
        if (cnda_%(oname)s){
            //TODO: check if we can maybe use existing storage
            Py_DECREF(cnda_%(oname)s);
            cnda_%(oname)s = NULL;
        }
        if (NULL == cnda_%(oname)s)
        {
            cnda_%(oname)s = (CudaNdarray*)CudaNdarray_new_null();
            if (!cnda_%(oname)s)
            { 
                //error string already set
                %(fail)s;
            }
            if (CudaNdarray_alloc_contiguous(cnda_%(oname)s, %(nd)s, dims))
            {
                //error string already set
                Py_DECREF(cnda_%(oname)s);
                cnda_%(oname)s = NULL;
                %(fail)s;
            }
        }
        //std::cerr << "ELEMWISE NEW %(oname)s nd" << cnda_%(oname)s->nd << "\\n";
        //std::cerr << "ELEMWISE NEW %(oname)s data" << cnda_%(oname)s->devdata << "\\n";
        """ % locals()
        print >> sio, """
        { 
            //new block so that failure gotos don't skip over variable initialization
            int log2_dims[%(nd)s];
            //std::cerr << "calling callkernel\\n";
            callkernel_%(nodename)s(1, 0, dims, log2_dims
            """ % locals()
        for iname in inputs:
            print >> sio, """
                        , CudaNdarray_DEV_DATA(cnda_%(iname)s), CudaNdarray_STRIDES(cnda_%(iname)s)
            """ % locals()
        for oname in outputs:
            print >> sio, """
                        , CudaNdarray_DEV_DATA(cnda_%(oname)s), CudaNdarray_STRIDES(cnda_%(oname)s)
            """ % locals()
        print >> sio, """
                        );

            //std::cerr << "calling callkernel returned\\n";
            cudaThreadSynchronize();
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n", "Elemwise %(nodename)s", cudaGetErrorString(err));
                """ % locals()
        for oname in outputs:
            print >> sio, """
                Py_DECREF(cnda_%(oname)s);
                cnda_%(oname)s = NULL;
                """ % locals()
        print >> sio, """
                %(fail)s;
            }                         
        }
        //std::cerr << "C_CODE %(opname)s END\\n";
        """ % locals()
        return sio.getvalue()

    def c_code_cache_version(self):
        return ()

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
        if (cnda_%(input)s->nd != %(nd_in)s)
        {
            PyErr_Format(PyExc_TypeError, "required nd=%(nd_in)s, got nd=%%i", cnda_%(input)s->nd);
            %(fail)s;
        }
        """ %locals()

        #alloc an output
        print >> sio, """
        if (cnda_%(res)s && (cnda_%(res)s->nd == %(nd_out)s))
        {
            //re-use previously-allocated cnda
        }
        else
        {
            if (cnda_%(res)s)
            {
                Py_DECREF(cnda_%(res)s);
                cnda_%(res)s = NULL;
            }
            cnda_%(res)s = (CudaNdarray*) CudaNdarray_new_null();
            if (NULL == cnda_%(res)s)
            {
                PyErr_SetString(PyExc_MemoryError, "Failed to allocate result");
                %(fail)s;
            }
            if (CudaNdarray_set_nd(cnda_%(res)s, %(nd_out)s))
            {
                // err message set
                Py_DECREF(cnda_%(res)s);
                cnda_%(res)s = NULL;
                %(fail)s;
            }
        }
        """ %locals()

        print >> sio, """
        if (CudaNdarray_set_device_data(cnda_%(res)s, CudaNdarray_DEV_DATA(cnda_%(input)s), cnda_%(input)s))
        {
            // err message set
            Py_DECREF(cnda_%(res)s);
            cnda_%(res)s = NULL;
            %(fail)s;
        }
        """ %locals()

        #reassign the dimension and strides in the host pointers
        for i, o in enumerate(self.new_order):
            if o == 'x':
                assert node.outputs[0].type.broadcastable[i]
                print >> sio, """
        cnda_%(res)s->dim[%(i)s] = 1;
        cnda_%(res)s->str[%(i)s] = 0;
                """ %locals()
            else:
                assert not node.outputs[0].type.broadcastable[i]
                print >> sio, """
        cnda_%(res)s->dim[%(i)s] = cnda_%(input)s->dim[%(o)s];
        cnda_%(res)s->str[%(i)s] = cnda_%(input)s->str[%(o)s];
                """ %locals()

        for i, o in enumerate(self.new_order):
                print >> sio, """
        //std::cerr << "GpuDimShuffle " << cnda_%(res)s << " str[%(i)s] = " << cnda_%(res)s->str[%(i)s] << "\\n";
                """ %locals()

        # copy the host dims and stride -> device
        print >> sio, """
        if (CudaNdarray_copy_structure_to_device(cnda_%(res)s))
        {
            //err msg set
            Py_DECREF(cnda_%(res)s);
            cnda_%(res)s = NULL;
            %(fail)s;
        }
        """ %locals()

        if 0:
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

class GpuSum(Op):
    def __init__(self, reduce_mask):
        self.reduce_mask = tuple(reduce_mask)

    def __eq__(self, other):
        return type(self) == type(other) and self.reduce_mask == other.reduce_mask

    def __hash__(self):
        return hash(type(self)) ^ hash(self.reduce_mask)

    def __str__(self):
        return "GpuSum{%s}" % str(self.reduce_mask)

    def make_node(self, x):
        if (x.type.ndim != len(self.reduce_mask)):
            raise TypeError("x must have rank %i"%len(self.reduce_mask))
        o_broadcast = [x.type.broadcastable[i] for i in xrange(x.type.ndim) if not self.reduce_mask[i]]
        return Apply(self, [x], [CudaNdarrayType(o_broadcast)()])

    def perform(self, node, (x,), (z,)):
        z[0] = x.reduce_sum(self.reduce_mask)

