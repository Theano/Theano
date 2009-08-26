import StringIO, sys
from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar

import logging, copy
_logger_name = 'theano_cuda_ndarray.elemwise'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler()) #TO REMOVE
def warning(*msg):
    _logger.warning(_logger_name+'WARNING: '+' '.join(str(m) for m in msg))
def info(*msg):
    _logger.info(_logger_name+'INFO: '+' '.join(str(m) for m in msg))
def debug(*msg):
    _logger.debug(_logger_name+'DEBUG: '+' '.join(str(m) for m in msg))


def _logical_scalar(x):
    return all(x.type.broadcastable)

class RecAlgo(object):
    def c_src_kernel(self, node, nodename):
        nd = node.outputs[0].type.ndim
        sio = StringIO.StringIO()
        #print 'C_SRC_KERNEL', sio.getvalue()


        for ipos, i in enumerate(node.inputs):
            print >> sio, "//    Input  ", ipos, str(i.type)
        for ipos, i in enumerate(node.outputs):
            print >> sio, "//    Output ", ipos, str(i.type)
        print >> sio, "static __global__ void kernel_%s_%s(unsigned int numEls" %(self.scalar_op.__class__.__name__,nodename)
        if (nd):
            print >> sio, "\t,", ", ".join("unsigned int log2_dim%i" % i for i in xrange(nd))
        #declare inputs
        for ipos, i in enumerate(node.inputs):
            s = ", ".join(["const float * i%i_data" % ipos] + list("int i%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print >> sio, "\t,", s
        #declare outputs
        for ipos, i in enumerate(node.outputs):
            s = ", ".join(["float * o%i_data" % ipos] + list("int o%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print >> sio, "\t,", s
            #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
            #print >> sio, "\t,", "float * o%i_data" % ipos
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
            kernel_call_args.append(
                    ", ".join(["i%i_data"%ipos] + list("i%i_str[%i]"%(ipos, di) for di in xrange(nd)))
                    )
            #strides = ", ".join("i%i_str[%i]"%(ipos, di) for di in xrange(nd))
            #kernel_call_args.append( "%s, i%i_data" % (strides, ipos))
        for ipos in xrange(len(node.outputs)):
            kernel_call_args.append(
                    ", ".join(["o%i_data"%ipos] + list("o%i_str[%i]"%(ipos, di) for di in xrange(nd)))
                    )
            #strides = ", ".join("o%i_str[%i]"%(ipos, di) for di in xrange(nd))
            #kernel_call_args.append( "%s, o%i_data" % (strides, ipos))
        kernel_call_args = ", ".join(kernel_call_args)

        # the data_pointer_increments are inserted after each recursive call
        data_ptr_inc = []
        for ipos in xrange(len(node.inputs)):
            data_ptr_inc.append("i%i_data += (1<< log2_dim) * i%i_str[d]" %(ipos, ipos))
        for ipos in xrange(len(node.outputs)):
            data_ptr_inc.append("o%i_data += (1<< log2_dim) * o%i_str[d]" %(ipos, ipos))
        data_ptr_inc = ";\n".join(data_ptr_inc)


        d.update(locals())
        d["scalar_op"]=self.scalar_op.__class__.__name__
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
                kernel_%(scalar_op)s_%(nodename)s<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);
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

    def c_support_code_apply(self, node, nodename):
        return self.c_src_kernel(node, nodename) + self.c_src_callkernel(node, nodename)

class NaiveAlgo(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.cache_version = ()

    def c_src_kernel(self, node, nodename):
        nd = node.outputs[0].type.ndim
        sio = StringIO.StringIO()
        #print 'C_SRC_KERNEL', sio.getvalue()

        def _logical_scalar(x):
            return all(x.type.broadcastable)

        for ipos, i in enumerate(node.inputs):
            print >> sio, "//    Input  ", ipos, str(i.type)
        for ipos, i in enumerate(node.outputs):
            print >> sio, "//    Output ", ipos, str(i.type)
        print >> sio, "static __global__ void kernel_%s_%s_%s(unsigned int numEls" %(self.scalar_op.__class__.__name__,nodename, id(self))
        if (nd):
            print >> sio, "\t,", ", ".join("const int dim%i" % i for i in xrange(nd))
        #declare inputs
        for ipos, i in enumerate(node.inputs):
            s = ", ".join(["const float * i%i_data" % ipos] + list("int i%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print >> sio, "\t,", s
        #declare outputs
        for ipos, i in enumerate(node.outputs):
            s = ", ".join(["float * o%i_data" % ipos] + list("int o%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print >> sio, "\t,", s
            #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
            #print >> sio, "\t,", "float * o%i_data" % ipos
        print >> sio, "\t)\n{"
        print >> sio, "    const int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        print >> sio, "    const int numThreads = blockDim.x * gridDim.x;"

        # For each input that is a scalar which has been broadcasted to a tensor,
        #     load it into a local variable
        for ipos, i in enumerate(node.inputs):
            if _logical_scalar(i):
                print >> sio, "    const float ii_i%i_value = i%i_data[0];" % (ipos, ipos)

        
        #TODO: insert code to check for strides of 1, and use a different loop
        
        #loop over the elements to be treated by this kernel call
        print >> sio, "    for (int i = idx; i < numEls; i += numThreads) {"
        # calculate the data pointers for all arguments
        print >> sio, "        int ii = i;"
        for ipos, i in enumerate(node.inputs):
            if not _logical_scalar(i):
                print >> sio, "        const float * ii_i%i_data = i%i_data;" % (ipos, ipos)
        for ipos, i in enumerate(node.outputs):
            print >> sio, "        float * ii_o%i_data = o%i_data;" % (ipos, ipos)
        for d in xrange(nd-1, -1, -1):
            if d > 0:
                print >> sio, "        int pos%i = ii %% dim%i;" %(d, d)
                print >> sio, "        ii = ii / dim%i;" %d
            else:
                print >> sio, "        int pos%i = ii;" %d

            for ipos, i in enumerate(node.inputs):
                if not _logical_scalar(i):
                    print >> sio, "        ii_i%i_data += pos%i * i%i_str_%i;" % (ipos, d, ipos, d)
            for ipos, i in enumerate(node.outputs):
                print >> sio, "        ii_o%i_data += pos%i * o%i_str_%i;" % (ipos, d, ipos, d)

        # perform the scalar operation on the input and output references
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

        print sio.getvalue()
        return sio.getvalue()

    def c_src_kernel_tiling(self, node, nodename):
        """ The kernel applies to problems with <= 5 dimensions """

        #The kernel is intended to be structured roughly like this:
        """
        static __global__ void kernel()
        {
            for (int v = blockIdx.y; v < dim0; v += gridDim.x)
            {
                for (int w = blockIdx.y; w < dim1; w += gridDim.y)
                {
                    for (int x = threadIdx.x; x < dim2; x += blockDim.x)
                    {
                        for (int y = threadIdx.y; y < dim3; y += blockDim.y)
                        {
                            for (int z = threadIdx.z; z < dim4; z += blockDim.z)
                            {
                                out[v * out_stride[0] + ...] = f(in1[...],  in2[...])
                            }
                        }
                    }
                }
            }
        }

        """

        nd = node.outputs[0].type.ndim
        sio = StringIO.StringIO()
        #print 'C_SRC_KERNEL', sio.getvalue()

        def _logical_scalar(x):
            return all(x.type.broadcastable)

        if nd in (4,):
            # print some leading comments to make the code easier to read
            for ipos, i in enumerate(node.inputs):
                print >> sio, "//    Input  ", ipos, str(i.type)
            for ipos, i in enumerate(node.outputs):
                print >> sio, "//    Output ", ipos, str(i.type)
            print >> sio, "static __global__ void kernel_%s_%s_%s_%s(unsigned int numEls" %(
                    self.scalar_op.__class__.__name__,
                    nodename, 
                    id(self),
                    'tiling%i'%nd)
            if (nd):
                print >> sio, "\t,", ", ".join("const int dim%i" % i for i in xrange(nd))
            #declare inputs
            for ipos, i in enumerate(node.inputs):
                s = ", ".join(["const float * i%i_data" % ipos] + list("int i%i_str_%i" % (ipos, d) for d in xrange(nd)))
                print >> sio, "\t,", s
            #declare outputs
            for ipos, i in enumerate(node.outputs):
                s = ", ".join(["float * o%i_data" % ipos] + list("int o%i_str_%i" % (ipos, d) for d in xrange(nd)))
                print >> sio, "\t,", s
                #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
                #print >> sio, "\t,", "float * o%i_data" % ipos
            print >> sio, "\t)\n{"

            # For each input that is a scalar which has been broadcasted to a tensor,
            #     load it into a local variable
            print >> sio, "    __shared__ float value0[%i];" % len(node.inputs)
            print >> sio, "    __shared__ int shared_dims[%(nd)s];" % locals()
            #print >> sio, "    __shared__ int shared_i_str[%(n_in)s][%(nd)s]"
            print >> sio, "    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {"
            for ipos, i in enumerate(node.inputs):
                if _logical_scalar(i):
                    print >> sio, "    value0[%i] = i%i_data[0];" % (ipos, ipos)
            for ipos in xrange(nd):
                print >> sio, "    shared_dims[%i] = dim%i;" % (ipos, ipos)
            print >> sio, "    }"
            print >> sio, "    __syncthreads();"
        

            if (nd == 4):
                print >> sio, """
                for (int pos0 = blockIdx.x; pos0 < shared_dims[0]; pos0 += gridDim.x)
                {
                    for (int pos1 = blockIdx.y; pos1 < shared_dims[1]; pos1 += gridDim.y)
                    {
                        //for (int pos2 = threadIdx.x; pos2 < shared_dims[2]; pos2 += blockDim.x)
                        for (int pos2 = threadIdx.y; pos2 < shared_dims[2]; pos2 += blockDim.y)
                        {
                            //for (int pos3 = threadIdx.y; pos3 < shared_dims[3]; pos3 += blockDim.y)
                            for (int pos3 = threadIdx.x; pos3 < shared_dims[3]; pos3 += blockDim.x)
                            {
                """
            else:
                raise NotImplementedError()
            
            for ipos, i in enumerate(node.inputs):
                if not _logical_scalar(i):
                    print >> sio, "        const float * ii_i%i_data = i%i_data;" % (ipos, ipos)
            for ipos, i in enumerate(node.outputs):
                print >> sio, "        float * ii_o%i_data = o%i_data;" % (ipos, ipos)
            for d in xrange(nd):
                for ipos, i in enumerate(node.inputs):
                    if not _logical_scalar(i):
                        print >> sio, "        ii_i%i_data += pos%i * i%i_str_%i;" % (ipos, d, ipos, d)
                for ipos, i in enumerate(node.outputs):
                    print >> sio, "        ii_o%i_data += pos%i * o%i_str_%i;" % (ipos, d, ipos, d)

            # perform the scalar operation on the input and output references
            #TODO: What if the scalar_op needs support_code??
            task_code = self.scalar_op.c_code(
                    Apply(self.scalar_op,
                        [scalar.Scalar(dtype = input.type.dtype)() for input in node.inputs],
                        [scalar.Scalar(dtype = output.type.dtype)() for output in node.outputs])
                    , nodename + '_scalar_'
                    , [('value0[%i]' if _logical_scalar(i) else 'ii_i%i_data[0]')%ipos for ipos, i in enumerate(node.inputs)] 
                    , ['ii_o%i_data[0]'%ipos for ipos, i in enumerate(node.outputs)] 
                    , sub=dict(fail='return;')) #TODO: set a failure code somehow!!!
            print >> sio, "       ", task_code

            print >> sio, "    }" * nd

            #TODO: insert runtime stride checks that select the best loop order either here, or in
            # the host code that launched the  kernel (host code probably better spot)

            #indent = " "*(4*d+7)
            #for ipos, i in enumerate(node.inputs):
                #print >> sio, indent, "const float * i%i" % ipos, '= i%i_data', ''
            print >> sio, "}"

        print sio.getvalue()
        return sio.getvalue()

    def c_src_kernel_tiling_less_registers(self, node, nodename):
        """ The kernel applies to problems with <= 5 dimensions """

        nd = node.outputs[0].type.ndim
        n_in = len(node.inputs)
        n_out = len(node.outputs)
        sio = StringIO.StringIO()

        if nd not in (4,):
            return sio.getvalue()

        # print some leading comments to make the code easier to read
        for ipos, i in enumerate(node.inputs):
            print >> sio, "//    Input  ", ipos, str(i.type)
        for ipos, i in enumerate(node.outputs):
            print >> sio, "//    Output ", ipos, str(i.type)
        print >> sio, "static __global__ void kernel_%s_%s_%s_%s(unsigned int numEls" %(
                self.scalar_op.__class__.__name__,
                nodename, 
                id(self),
                'tiling%i_less_registers'%nd)
        if (nd):
            print >> sio, "\t,", ", ".join("const int dim%i" % i for i in xrange(nd))
        #declare inputs
        for ipos, i in enumerate(node.inputs):
            s = ", ".join(["const float * i%i_data_0" % ipos] + list("int i%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print >> sio, "\t,", s
        #declare outputs
        for ipos, i in enumerate(node.outputs):
            s = ", ".join(["float * o%i_data_0" % ipos] + list("int o%i_str_%i" % (ipos, d) for d in xrange(nd)))
            print >> sio, "\t,", s
            #print >> sio, "\t,", ", ".join("int o%i_str_%i" % (ipos, d) for d in xrange(nd))
            #print >> sio, "\t,", "float * o%i_data" % ipos
        print >> sio, "\t)\n{"

        # TODO: Setting these to true makes the function fail SOMETIMES.  I don't know why yet.
        use_shared_stride = False
        use_shared_limits = False

        def decl_limits(nd):
            if use_shared_limits:
                print >> sio, "__shared__ float * limits[%(nd)s];" % locals()

        def stride(io, p, d):
            if use_shared_stride:
                return "s%s_str[%i][%i]" %(io, p, d)
            else:
                return "%s%i_str_%i" %(io, p, d)
        def limits(d):
            if use_shared_limits:
                return "limits[%i]" % d
            else:
                return "limits%i" % d

        def decl_shared_stride(nin, nout, nd):
            if not use_shared_stride:
                return
            print >> sio, """
            __shared__ int si_str[%(nin)s][%(nd)s];
            __shared__ int so_str[%(nout)s][%(nd)s];
            if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
            """ % locals()
            for i in xrange(nin):
                for d in xrange(nd):
                    print >> sio, "si_str[%(i)s][%(d)s] = i%(i)s_str_%(d)s;" %locals()
            for i in xrange(n_out):
                for d in xrange(nd):
                    print >> sio, "so_str[%(i)s][%(d)s] = o%(i)s_str_%(d)s;" %locals()
            print >> sio, "} __syncthreads();"

        def calc_limit(d):
            s = stride('o', 0, d)
            lname = limits(d)
            if use_shared_limits:
                print >> sio, "if ((threadIdx.x == 0) && (threadIdx.y == 0)) {"
                if d == 0:
                    print >> sio, "%(lname)s = o0_data_0 + dim%(d)s * %(s)s;" % locals()
                else:
                    dm1 = d - 1
                    print >> sio, "%(lname)s = o0_data_%(dm1)s + dim%(d)s * %(s)s;" % locals()
                print >> sio, "} __syncthreads();"
            else:
                if d == 0:
                    print >> sio, "const float * %(lname)s = o0_data_0 + dim%(d)s * %(s)s;" % locals()
                else:
                    dm1 = d - 1
                    print >> sio, "const float * %(lname)s = o0_data_%(dm1)s + dim%(d)s * %(s)s;" % locals()

        def decl_ptrs(d, offset):
            dm1 = d - 1
            assert dm1 >= 0
            for i in xrange(n_in):
                s = stride('i', i, d)
                print >> sio, "const float * i%(i)s_data_%(d)s = i%(i)s_data_%(dm1)s + %(offset)s * %(s)s;" %locals()
            for i in xrange(n_out):
                s = stride('o', i, d)
                print >> sio, "float * o%(i)s_data_%(d)s = o%(i)s_data_%(dm1)s + %(offset)s * %(s)s;" %locals()

        def inc_ptrs(d, amt):
            for i in xrange(n_in):
                s = stride('i', i, d)
                print >> sio, "i%(i)s_data_%(d)s += %(amt)s * %(s)s;" %locals()
            for i in xrange(n_out):
                s = stride('o', i, d)
                print >> sio, "o%(i)s_data_%(d)s += %(amt)s * %(s)s;" %locals()

        def while_limit(d):
            lname = limits(d)
            print >> sio, "while (o0_data_%(d)s < %(lname)s) { " % locals()

        def end_while(d):
            print >> sio, "}"

        def task_code(d):
            print >> sio, self.scalar_op.c_code(
                Apply(self.scalar_op,
                    [scalar.Scalar(dtype = input.type.dtype)() for input in node.inputs],
                    [scalar.Scalar(dtype = output.type.dtype)() for output in node.outputs])
                , nodename + '_scalar_'
                , ['i%i_data_%i[0]'%(ipos,d) for ipos, i in enumerate(node.inputs)] 
                , ['o%i_data_%i[0]'%(ipos,d) for ipos, i in enumerate(node.outputs)] 
                , sub=dict(fail='return;')) #TODO: set a failure code somehow!!!

        if nd == 4:
            decl_shared_stride(n_in, n_out, nd)
            decl_limits(nd)
            calc_limit(0)
            inc_ptrs(0, 'blockIdx.x')
            while_limit(0)
            if 1:
                calc_limit(1)
                decl_ptrs(1, 'blockIdx.y')
                while_limit(1)
                if 1:
                    calc_limit(2)
                    decl_ptrs(2, 'threadIdx.y')
                    while_limit(2)
                    if 1:
                        calc_limit(3)
                        decl_ptrs(3, 'threadIdx.x')
                        while_limit(3)
                        if 1:
                            task_code(3)
                            inc_ptrs(3, 'blockDim.x')
                        end_while(3)
                        inc_ptrs(2, 'blockDim.y')
                    end_while(2)
                    inc_ptrs(1, 'gridDim.y')
                end_while(1)
                inc_ptrs(0, 'gridDim.x')
            end_while(0)
            
        print >> sio, "}"
        print sio.getvalue()
        return sio.getvalue()

    def c_src_kernel_Ccontiguous(self, node, nodename):
        nd = node.outputs[0].type.ndim
        sio = StringIO.StringIO()
        #print 'C_SRC_KERNEL', sio.getvalue()

        def _logical_scalar(x):
            return all(x.type.broadcastable)

        for ipos, i in enumerate(node.inputs):
            print >> sio, "//    Input  ", ipos, str(i.type)
        for ipos, i in enumerate(node.outputs):
            print >> sio, "//    Output ", ipos, str(i.type)
        print >> sio, "static __global__ void kernel_%s_%s_Ccontiguous (unsigned int numEls" %(self.scalar_op.__class__.__name__,nodename)
        #declare inputs
        for ipos, i in enumerate(node.inputs):
            print >> sio, "\t,", "const float * i%i_data" % ipos
        #declare outputs
        for ipos, i in enumerate(node.outputs):
            print >> sio, "\t,", "float * o%i_data" % ipos
        print >> sio, "\t)\n{"
        print >> sio, "    const int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        print >> sio, "    const int numThreads = blockDim.x * gridDim.x;"
       
        # For each input that is a scalar which has been broadcasted to a tensor,
        #     load it into a local variable
        for ipos, i in enumerate(node.inputs):
            if _logical_scalar(i):
                print >> sio, "    const float ii_i%i_value = i%i_data[0];" % (ipos, ipos)


        #loop over the elements to be treated by this kernel call
        print >> sio, "    for (int i = idx; i < numEls; i += numThreads) {"
        # perform the scalar operation on the input and output references
        #TODO: What if the scalar_op needs support_code??
        task_code = self.scalar_op.c_code(
                Apply(self.scalar_op,
                    [scalar.Scalar(dtype = input.type.dtype)() for input in node.inputs],
                    [scalar.Scalar(dtype = output.type.dtype)() for output in node.outputs])
                , nodename + '_scalar_'
                #, ['i%i_data[i]'%ipos for ipos, i in enumerate(node.inputs)] 
                , [('ii_i%i_value' if _logical_scalar(i) else 'i%i_data[i]')%ipos for ipos, i in enumerate(node.inputs)] 
                , ['o%i_data[i]'%ipos for ipos, i in enumerate(node.outputs)] 
                , sub=dict(fail='return;')) #TODO: set a failure code somehow!!!
        print >> sio, "       ", task_code
        print >> sio, "    }"
        print >> sio, "}"

        print sio.getvalue()
        return sio.getvalue()

    def c_src_callkernel(self, node, nodename):
        nd = node.outputs[0].type.ndim
        id_self = id(self)
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

        prod_dims = '*'.join("dims[%i]"%di for di in xrange(nd))

        scalar_op=self.scalar_op.__class__.__name__

        sio = StringIO.StringIO()
        print >> sio, """
        static inline int 
        c_contiguous_beyond_%(nodename)s(int nd, const int * dims, const int * strides, int &size)
        {
            // return the dimension such that it and all greater dimensions are c-contiguous
            // if everything is c_contiguous then this function returns 0, and size is left
            // with the number of elements.

            size = 1;
            while (nd > 0)
            {
                if ((dims[nd-1] > 1) && (strides[nd-1] != size))
                {
                    return nd;
                }
                size = size * dims[nd-1];
                --nd;
            }
            return nd;
        }
        """ %locals()
        print >> sio, """
        static int callkernel_%(nodename)s(unsigned int numEls, const int d,
            const int * dims,
            %(input_params)s,
            %(output_params)s)
        {
            numEls = %(prod_dims)s;
            std::cerr << "calling kernel_%(scalar_op)s_%(nodename)s_%(id_self)s     w numEls" << numEls << "\\n";
        """ %locals()
        # DEBUGPRINT
        print >> sio, 'std::cerr << ' + " << ' ' <<  ".join(['"  "']+list("dims[%i]"%di
            for di in xrange(nd)) + ["'\\n';"])
        # DEBUGPRINT
        for ipos in xrange(len(node.inputs)):
            print >> sio, """
            std::cerr << "   %(ipos)s " << 
        """ %locals() + " << ' ' <<  ".join(["i%i_data"%ipos]
                + list("i%i_str[%i]"%(ipos, di) for di in xrange(nd))) + ''' << "\\n"; '''

        # collapse contiguous right-most dimensions (ignoring scalars)
        # this is a good idea because [we assume that] the output has been allocated c_contiguous

        print >> sio, "int nd_collapse = 0;" #because the outputs are assumed to be c_contiguous
        print >> sio, "int nd_collapse_size = numEls;" #because the outputs are assumed to be c_contiguous
        for ipos in xrange(len(node.inputs)):
            if not _logical_scalar(node.inputs[ipos]):
                print >> sio, """
                    int nd_collapse_size_%(ipos)s;
                    int nd_collapse_%(ipos)s = c_contiguous_beyond_%(nodename)s(%(nd)s, dims, i%(ipos)s_str, nd_collapse_size_%(ipos)s);
                    if (nd_collapse_%(ipos)s > nd_collapse)
                    {
                        nd_collapse = nd_collapse_%(ipos)s;
                        nd_collapse_size = nd_collapse_size_%(ipos)s;
                    }
                """ %locals()
        # DEBUGPRINT
        print >> sio, 'std::cerr << "  nd_collapse " << nd_collapse << " " << nd_collapse_size << "\\n";'
        for ipos in xrange(len(node.inputs)):
            print >> sio, "int local_i%(ipos)s_str[%(nd)s];"%locals()
            for d in xrange(nd):
                print >> sio, "local_i%(ipos)s_str[%(d)s] = (%(d)s == nd_collapse) ? 1 : i%(ipos)s_str[%(d)s];"%locals()
        for ipos in xrange(len(node.outputs)):
            print >> sio, "int local_o%(ipos)s_str[%(nd)s];"%locals()
            for d in xrange(nd):
                print >> sio, "local_o%(ipos)s_str[%(d)s] = (%(d)s == nd_collapse) ? 1 : o%(ipos)s_str[%(d)s];"%locals()
        print >> sio, "int local_dims[%(nd)s];"%locals()
        for d in xrange(nd):
            print >> sio, "local_dims[%(d)s] = (%(d)s == nd_collapse) ? nd_collapse_size : dims[%(d)s];"%locals()


        def launch_Ccontiguous(nodename, id_self, scalar_op):
            kernel_call_args = ["numEls"]
            for ipos in xrange(len(node.inputs)):
                kernel_call_args.append("i%i_data"%ipos)
            for ipos in xrange(len(node.outputs)):
                kernel_call_args.append("o%i_data"%ipos)
            kernel_call_args = ", ".join(kernel_call_args)
            print >> sio, """
                int threads_per_block = std::min(numEls, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                int n_blocks = std::min(numEls/threads_per_block + (numEls %% threads_per_block?1:0), (unsigned int)NUM_VECTOR_OP_BLOCKS);
                kernel_%(scalar_op)s_%(nodename)s_Ccontiguous<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);

                //std::cerr << "calling callkernel returned\\n";
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err) 
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n", "Elemwise %(nodename)s", cudaGetErrorString(err));
                    return -1;
                
                }                         
                return 0;
                """ %locals()

        def launch_tile4():
            if (False and nd == 4): # tiling kernel
                print >> sio, """
                {
                    std::cerr << "   Running tiling 4D \\n";
                    dim3 gridDim(dims[0], dims[1]);
                    dim3 blockDim;
                    if (0) {
                        blockDim.y = std::min(dims[3], NUM_VECTOR_OP_THREADS_PER_BLOCK);
                        blockDim.x = std::min(dims[2], (int)(NUM_VECTOR_OP_THREADS_PER_BLOCK/ blockDim.y));
                    }
                    else
                    {
                        blockDim.x = std::min(dims[3], NUM_VECTOR_OP_THREADS_PER_BLOCK);
                        blockDim.y = std::min(dims[2], (int)(NUM_VECTOR_OP_THREADS_PER_BLOCK/ blockDim.x));
                    }
                    if ((o0_str[0] <= 0) || (o0_str[1] <= 0) || (o0_str[2] <= 0) || (o0_str[3] <= 0))
                    {
                        kernel_%(scalar_op)s_%(nodename)s_%(id_self)s_tiling4<<<gridDim, blockDim>>>(%(kernel_call_args)s);
                    } else {
                        kernel_%(scalar_op)s_%(nodename)s_%(id_self)s_tiling4_less_registers<<<gridDim, blockDim>>>(%(kernel_call_args)s);
                    }

                    cudaError_t err = cudaGetLastError();
                    if( cudaSuccess != err) 
                    {
                        std::cerr << "   DEBUG: tiling4 call failure... falling back to general version \\n";
                        std::cerr << "   DEBUG: tiling4 call failure... " << cudaGetErrorString(err) << "\\n";
                        std::cerr << "   DEBUG: tiling4 call failure... grid" <<  gridDim.x<< " " << gridDim.y<< "\\n";
                        std::cerr << "   DEBUG: tiling4 call failure... block" <<  blockDim.x<< " " << blockDim.y<< "\\n";
                        int threads_per_block = std::min(numEls, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                        int n_blocks = std::min(numEls/threads_per_block + (numEls %% threads_per_block?1:0), (unsigned int)NUM_VECTOR_OP_BLOCKS);
                        kernel_%(scalar_op)s_%(nodename)s_%(id_self)s<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);
                        CNDA_THREAD_SYNC;
                        err = cudaGetLastError();
                        if( cudaSuccess != err) 
                        {
                            PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n", "Elemwise %(nodename)s", cudaGetErrorString(err));
                            return -1;
                        
                        }                         
                    }
                    return 0;
                }
            }
                """ %locals()

        def launch_General(nodename, id_self, scalar_op):
            # kernel_call_args are used to invoke the cuda kernel
            kernel_call_args = ["numEls"]
            kernel_call_args.extend("dims[%i]"%di for di in xrange(nd))
            for ipos in xrange(len(node.inputs)):
                kernel_call_args.append(
                        ", ".join(["i%i_data"%ipos] + list("local_i%i_str[%i]"%(ipos, di) for di in xrange(nd)))
                        )
                #strides = ", ".join("i%i_str[%i]"%(ipos, di) for di in xrange(nd))
                #kernel_call_args.append( "%s, i%i_data" % (strides, ipos))
            for ipos in xrange(len(node.outputs)):
                kernel_call_args.append(
                        ", ".join(["o%i_data"%ipos] + list("o%i_str[%i]"%(ipos, di) for di in xrange(nd)))
                        )
                #strides = ", ".join("o%i_str[%i]"%(ipos, di) for di in xrange(nd))
                #kernel_call_args.append( "%s, o%i_data" % (strides, ipos))
            kernel_call_args = ", ".join(kernel_call_args)
            print >> sio, """
                std::cerr << "   Running general version \\n";
                int threads_per_block = std::min(numEls, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                int n_blocks = std::min(numEls/threads_per_block + (numEls %% threads_per_block?1:0), (unsigned int)NUM_VECTOR_OP_BLOCKS);
                kernel_%(scalar_op)s_%(nodename)s_%(id_self)s<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);
                CNDA_THREAD_SYNC;
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err) 
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n", "Elemwise %(nodename)s", cudaGetErrorString(err));
                    return -1;
                
                }                         
                return 0;
                """ %locals()

        print >> sio, "switch (nd_collapse) {"
        print >> sio, "case 0: {"
        launch_Ccontiguous(nodename, id_self, scalar_op)
        print >> sio, "        } break;"
        #print >> sio, "case 4: {" 
        #launch_tile4()
        #print >> sio, "        } break;"
        print >> sio, "default: {" 
        launch_General(nodename, id_self, scalar_op)
        print >> sio, "        }"
        print >> sio, "}"
        print >> sio, "}"

        #N.B. cudaGetLastError is called by c_code
        return sio.getvalue()


    def c_support_code_apply(self, node, nodename):
        return self.c_src_kernel(node, nodename) \
                + self.c_src_kernel_Ccontiguous(node, nodename) \
                + self.c_src_kernel_tiling(node, nodename) \
                + self.c_src_kernel_tiling_less_registers(node, nodename) \
                + self.c_src_callkernel(node, nodename)

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
            dims[i] = (dims[i] == 1) ? CudaNdarray_HOST_DIMS(cnda_%(iname)s)[i] : dims[i];
            if ((CudaNdarray_HOST_DIMS(cnda_%(iname)s)[i] != 1) && (dims[i] != CudaNdarray_HOST_DIMS(cnda_%(iname)s)[i]))
            {
                //std::cerr << "C_CODE %(opname)s checking input %(iname)s failed\\n";
                PyErr_Format(PyExc_TypeError, "GpuElemwise input has incompatible dim[%%i] == %%i, where output has size %%i",
                    i,
                    CudaNdarray_HOST_DIMS(cnda_%(iname)s)[i],
                    dims[i]
                    );
                %(fail)s;
            }
        }
            """ %locals()

        for oname in outputs:
            print >> sio, """
        if (cnda_%(oname)s) {
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
            //std::cerr << "calling callkernel\\n";
            if (callkernel_%(nodename)s(1, 0, dims
            """ % locals()
        for iname in inputs:
            print >> sio, """
                        , CudaNdarray_DEV_DATA(cnda_%(iname)s), CudaNdarray_HOST_STRIDES(cnda_%(iname)s)
            """ % locals()
        for oname in outputs:
            print >> sio, """
                        , CudaNdarray_DEV_DATA(cnda_%(oname)s), CudaNdarray_HOST_STRIDES(cnda_%(oname)s)
            """ % locals()
        print >> sio, """
                        ))
            {
                 // error
            """
        for oname in outputs:
            print >> sio, """
                Py_DECREF(cnda_%(oname)s);
                cnda_%(oname)s = NULL;
                """ % locals()
        print >> sio, """
                %(fail)s;
            }
            else // no error
            {
            }
        }
        //std::cerr << "C_CODE %(opname)s END\\n";
        """ % locals()
        #print sio.getvalue()
        return sio.getvalue()

    def c_support_code(self):
        return """
        #define INTDIV_POW2(a, b) (a >> b)
        #define INTMOD_POW2(a, b) (a & ((1<<b)-1))
        """



class ExternAlgo(object):
    def externalgo_c_support_code_apply(self, node, nodename):
        nd = node.outputs[0].type.ndim
        n_inputs = len(node.inputs)
        n_outputs = len(node.outputs)
        id_self = id(self)
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

        prod_dims = '*'.join("dims[%i]"%di for di in xrange(nd))

        scalar_op=self.scalar_op.__class__.__name__

        apply_task_code = self.scalar_op.c_code(
                Apply(self.scalar_op,
                    [scalar.Scalar(dtype = input.type.dtype)() for input in node.inputs],
                    [scalar.Scalar(dtype = output.type.dtype)() for output in node.outputs])
                , nodename + '_scalar_'
                , ['x[%i][0]'%ipos for ipos, i in enumerate(node.inputs)] 
                , ['z[%i][0]'%ipos for ipos, i in enumerate(node.outputs)] 
                , sub=dict(fail='return;')) #TODO: set a failure code somehow!!!

        ### NOTE WELL: log2_dims is not initialized on input to this function... it is meant as
        ### storage space where the log2_dims *could* be computed and stored.
        sio = StringIO.StringIO()
        print >> sio, """
        #include "elemwise.cuh"

        template <int nx, typename Tx, int nz, typename Tz>
        class ElemwiseFn_%(scalar_op)s
        {
        public:
            static __device__ void apply(const Tx**x, Tz**z)
            {
                %(apply_task_code)s
            }  
        };

        static void callkernel_%(nodename)s(unsigned int numEls, const int d,
            const int * dims, const int * log2_dims,
            %(input_params)s,
            %(output_params)s)
        {
            const float * inputs[%(n_inputs)s];
            float * outputs[%(n_outputs)s];
            const int * input_strides[%(n_inputs)s];
            const int * output_strides[%(n_inputs)s];
            
        """ %locals()
        for ipos, i in enumerate(node.inputs):
            print >> sio, """
            inputs[%(ipos)s] = i%(ipos)s_data;
            input_strides[%(ipos)s] = i%(ipos)s_str;
            """ %locals()
        for ipos, i in enumerate(node.outputs):
            print >> sio, """
            outputs[%(ipos)s] = o%(ipos)s_data;
            output_strides[%(ipos)s] = o%(ipos)s_str;
            """ %locals()
        print >> sio, """
            cnda_elemwise<float, float, 
                    ElemwiseFn_%(scalar_op)s<%(n_inputs)s, typeof(i0_data[0]), %(n_outputs)s, typeof(o0_data[0])>
                    , %(n_inputs)s, %(n_outputs)s, %(nd)s> (
                 dims,
                 inputs,
                 input_strides,
                 outputs,
                 output_strides
            );
        }
        """ %locals()
        print sio.getvalue()
        return sio.getvalue()


