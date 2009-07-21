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
        return '<HostFromGpu@%i>' % id(self)
    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x], [tensor.TensorType(dtype=x.dtype, broadcastable=x.broadcastable)()])
    def perform(self, node, (x,), (z,)):
        z[0] = numpy.asarray(x)
    def grad(self, inputs, (gz,)):
        return [GpuFromHost()(gz)]

class GpuFromHost(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return '<GpuFromHost@%i>' % id(self)
    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable)()])
    def perform(self, node, (x,), (z,)):
        z[0] = type_support_filter(numpy.asarray(x, dtype='float32'), tuple([0]*x.ndim), 0)
    def grad(self, inputs, (gz,)):
        return [HostFromGpu()(gz)]


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
            if i.type.broadcastable != inputs[0].type.broadcastable:
                raise NotImplementedError('different bcastable')
        otype = CudaNdarrayType(broadcastable=_inputs[0].broadcastable)
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
        #TODO: optimize by passing the log2 of each dim, as well as the mask of 1s that we need 
        #      to compute the modulo

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
        #TODO: insert code to check for strides of 1, and use a different loop
        
        #loop over the elements to be treated by this kernel call
        print >> sio, "    for (unsigned int i = idx; i < numEls; i += numThreads) {"
        # calculate the data pointers for all arguments
        print >> sio, "        unsigned int ii = i;"
        for ipos, i in enumerate(node.inputs):
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
                print >> sio, "        ii_i%i_data += pos%i * i%i_str_%i;" % (ipos, d, ipos, d)
            for ipos, i in enumerate(node.outputs):
                print >> sio, "        ii_o%i_data += pos%i * o%i_str_%i;" % (ipos, d, ipos, d)

        # perform the scalar operation on the input and output references
            if d == 0:
                print >> sio, "       ", self.scalar_op.c_code(None, None, 
                        ['ii_i%i_data[0]'%ipos for ipos, i in enumerate(node.inputs)], 
                        ['ii_o%i_data[0]'%ipos for ipos, i in enumerate(node.outputs)], 
                        sub=dict(fail='return;')) #TODO: set a failure code somehow!!!
        print >> sio, "    }"

        #TODO: insert runtime stride checks that select the best loop order either here, or in
        # the host code that launched the  kernel (host code probably better spot)

        #indent = " "*(4*d+7)
        #for ipos, i in enumerate(node.inputs):
            #print >> sio, indent, "const float * i%i" % ipos, '= i%i_data', ''
        print >> sio, "}"
        if 0:
            print sio.getvalue()
        return sio.getvalue()

    def c_support_code_apply(self, node, nodename):
        return self.c_src_kernel(node, nodename) + \
                self.c_src_callkernel(node, nodename)
    def c_src_callkernel(self, node, nodename):
        nd = node.outputs[0].type.ndim
        d = dict()
        assert nd == 2
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
        kernel_call_args = ["numEls, log2_dims[0], log2_dims[1]"]
        for ipos in xrange(len(node.inputs)):
            strides = ", ".join("i%i_str[%i]"%(ipos, di) for di in xrange(nd))
            kernel_call_args.append( "%s, i%i_data" % (strides, ipos))
        for ipos in xrange(len(node.outputs)):
            strides = ", ".join("i%i_str[%i]"%(ipos, di) for di in xrange(nd))
            kernel_call_args.append( "%s, o%i_data" % (strides, ipos))
        kernel_call_args = ",".join(kernel_call_args)

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
                int n_blocks = std::min(numEls/threads_per_block + 1, (unsigned int)NUM_VECTOR_OP_BLOCKS);
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
        print >> sio, """
        std::cerr << "C_CODE %(opname)s START\\n";
        //standard elemwise size checks
        const int * dims = NULL;
        """ %locals()
        for iname in inputs:
            print >> sio, """
        if (%(nd)s != cnda_%(iname)s->nd)
        {
            PyErr_Format(PyExc_TypeError, "need %(nd)s dims, not %%i", cnda_%(iname)s->nd);
            %(fail)s;
        }
            """ %locals()
        for iname0, iname1 in zip(inputs[1:], inputs[:-1]):
            print >> sio, """
        //standard elemwise dim checks
        for (int i = 0; i< %(nd)s; ++i)
        {
            if (cnda_%(iname0)s->dim[i] != cnda_%(iname1)s->dim[i])
            {
                PyErr_SetString(PyExc_TypeError, "need same dimensions");
                %(fail)s;
            }
        }
            """ %locals()
        iname0 = inputs[0]
        print >> sio, """
        dims = cnda_%(iname0)s->dim;
        //unsigned int size = CudaNdarray_SIZE(cnda_%(iname0)s);
        //std::cerr << "ADD size " << size << "\\n";
        """ %locals()

        for oname in outputs:
            print >> sio, """
        if (cnda_%(oname)s){
            //TODO: check if we can maybe use existing storage
            Py_XDECREF(cnda_%(oname)s);
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
                Py_XDECREF(cnda_%(oname)s);
                cnda_%(oname)s = NULL;
                %(fail)s;
            }
        }
        std::cerr << "ELEMWISE NEW %(oname)s nd" << cnda_%(oname)s->nd << "\\n";
        std::cerr << "ELEMWISE NEW %(oname)s data" << cnda_%(oname)s->devdata << "\\n";
        """ % locals()
        print >> sio, """
        { 
            //new block so that failure gotos don't skip over variable initialization
            int log2_dims[%(nd)s];
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

            cudaThreadSynchronize();
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n", "Elemwise %(nodename)s", cudaGetErrorString(err));
                """ % locals()
        for oname in outputs:
            print >> sio, """
                Py_XDECREF(cnda_%(oname)s);
                cnda_%(oname)s = NULL;
                """ % locals()
        print >> sio, """
                %(fail)s;
            }                         
        }
        std::cerr << "C_CODE %(opname)s END\\n";
        """ % locals()
        return sio.getvalue()

    def c_code_cache_version(self):
        return ()

if 0:
    class GpuAdd(GpuElemwise):
        def __init__(self):
            super(GpuAdd, self).__init__(scalar.add)

        def perform(self, node, args, (z,)):
            print "GpuAdd perform"
            zval = numpy.asarray(args[0])
            for a in args[1:]:
                zval += numpy.asarray(a)
            z[0] = type_support_filter(zval, (0,)*len(zval.shape), 0)

    gpu_add = GpuAdd()
