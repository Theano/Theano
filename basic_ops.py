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
    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable)()])
    def perform(self, node, (x,), (z,)):
        z[0] = type_support_filter(numpy.asarray(x, dtype='float32'), tuple([0]*x.ndim), 0)
    def grad(self, inputs, (gz,)):
        return [HostFromGpu()(gz)]

class GpuAdd(Op):
    def __eq__(self, other):
        self.scalar_op = scalar.add
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def make_node(self, a, b):
        _a = as_cuda_ndarray_variable(a)
        _b = as_cuda_ndarray_variable(b)
        if _a.type.broadcastable != _b.type.broadcastable:
            raise NotImplementedError('different bcastable')
        return Apply(self, [_a,_b], [CudaNdarrayType(broadcastable=_a.broadcastable)()])

    def perform(self, node, (a,b), (z,)):
        aval = numpy.asarray(a, dtype='float32')
        bval = numpy.asarray(b, dtype='float32')
        z[0] = type_support_filter(aval + bval, (0,)*len(zval.shape), 0)

    def grad(self, inputs, (gz,)):
        return [gz for i in inputs]
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
        print sio.getvalue()
        return sio.getvalue()

    def c_support_code_apply(self, node, nodename):
        return self.c_src_kernel(node, nodename) + \
                self.c_src_callkernel(node, nodename)
    def c_src_callkernel(self, node, nodename):
        nd = node.outputs[0].type.ndim
        d = dict()
        assert nd == 2
        kernel_call_args = ("numEls, log2_dims[0], log2_dims[1]"
                ", a_str[0], a_str[1], a_data"
                ", b_str[0], b_str[1], b_data"
                ", z_str[0], z_str[1], z_data")
        d.update(locals())
        return """

        static void callkernel_%(nodename)s(const unsigned int numEls, const int d,
        const int * dims, int * log2_dims,
        const float * a_data, const int * a_str,
        const float * b_data, const int * b_str, 
        float * z_data, const int * z_str)
        {
            if (d == %(nd)s)
            {
                int threads_per_block = std::min(numEls, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                //a ceil would be better here
                int n_blocks = std::min(numEls/threads_per_block + 1, (unsigned int)NUM_VECTOR_OP_BLOCKS);
                kernel_%(nodename)s<<<n_blocks, threads_per_block>>>(%(kernel_call_args)s);
std::cerr << "ADDCALL a str" << a_str[0] << " "<< a_str[1] << "\\n";
std::cerr << "ADDCALL a data" << a_data << "\\n";
std::cerr << "ADDCALL b str" << b_str[0] << " "<< b_str[1] << "\\n";
std::cerr << "ADDCALL b data" << b_data << "\\n";
std::cerr << "ADDCALL z str" << z_str[0] << " "<< z_str[1] << "\\n";
std::cerr << "ADDCALL z data" << z_data << "\\n";

            }
            else
            {
std::cerr << "_ADDCALL d " << d << "\\n";
                unsigned int dim_d = dims[d];
std::cerr << "_ADDCALL dim_d " << dim_d << "\\n";
                int log2_dim = 0;
                while(dim_d)
                {
std::cerr << "___ADDCALL d " << d << " " << dim_d << "\\n";
                    if (dim_d&1)
                    {
                        log2_dims[d] = log2_dim; 
std::cerr << "___ADDCALL a str" << a_str[0] << " "<< a_str[1] << "\\n";
std::cerr << "___ADDCALL a data" << a_data << "\\n";
std::cerr << "___ADDCALL b str" << b_str[0] << " "<< b_str[1] << "\\n";
std::cerr << "___ADDCALL b data" << b_data << "\\n";
std::cerr << "___ADDCALL z str" << z_str[0] << " "<< z_str[1] << "\\n";
std::cerr << "___ADDCALL z data" << z_data << "\\n";
                        callkernel_%(nodename)s(numEls * (1<<log2_dim), d+1, 
                            dims, log2_dims, 
                            a_data, a_str, 
                            b_data, b_str, 
                            z_data, z_str);
                        a_data += (1 << log2_dim) * a_str[d];
                        b_data += (1 << log2_dim) * b_str[d];
                        z_data += (1 << log2_dim) * z_str[d];
                    }
                    log2_dim += 1;
                    dim_d >>= 1;
                }
            }
        }
        """ %d

    def c_code(self, node, nodename, (a,b), (z,), sub):
        d = dict(sub)
        nd = node.outputs[0].type.ndim
        d.update(locals())
        return """
std::cerr << "ADD start\\n";
        //standard elemwise size checks
        if (cnda_%(a)s->nd != cnda_%(b)s->nd)
        {
            PyErr_SetString(PyExc_TypeError, "need same number of dims");
            return NULL;
        }
        //standard elemwise dim checks
        unsigned int size = 1;
        for (int i = 0; i< cnda_%(a)s->nd; ++i)
        {
            if (cnda_%(a)s->dim[i] != cnda_%(b)s->dim[i])
            {
                PyErr_SetString(PyExc_TypeError, "need same dimensions");
                return NULL;
            }
            size *= (unsigned int) cnda_%(a)s->dim[i];
        }
std::cerr << "ADD size " << size << "\\n";
        if (cnda_%(z)s){
            //TODO: check if we can maybe use existing storage
            Py_XDECREF(cnda_%(z)s);
            cnda_%(z)s = NULL;
std::cerr << "ADD decref z \\n";
        }
        if (NULL == cnda_%(z)s)
        {
            cnda_%(z)s = (CudaNdarray*)CudaNdarray_new_null();
            if (!cnda_%(z)s)
            {
                %(fail)s;
            }
            if (CudaNdarray_alloc_contiguous(cnda_%(z)s, cnda_%(a)s->nd, cnda_%(a)s->dim))
            {
                Py_XDECREF(cnda_%(z)s);
                cnda_%(z)s = NULL;
                %(fail)s;
            }
        }
std::cerr << "ADD z nd" << cnda_%(z)s->nd << "\\n";
std::cerr << "ADD z str" << cnda_%(z)s->str[0] << " "<< cnda_%(z)s->str[1] << "\\n";
std::cerr << "ADD z data" << cnda_%(z)s->devdata << "\\n";
        { //new block so that failure gotos don't skip over variable initialization
            int log2_dims[%(nd)s];
            callkernel_%(nodename)s(1, 0, CudaNdarray_DIMS(cnda_%(z)s), log2_dims,
                        CudaNdarray_DEV_DATA(cnda_%(a)s), CudaNdarray_STRIDES(cnda_%(a)s),
                        CudaNdarray_DEV_DATA(cnda_%(b)s), CudaNdarray_STRIDES(cnda_%(b)s),
                        CudaNdarray_DEV_DATA(cnda_%(z)s), CudaNdarray_STRIDES(cnda_%(z)s));

            cudaThreadSynchronize();
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n", "kExp", cudaGetErrorString(err));
                Py_XDECREF(cnda_%(z)s);
                cnda_%(z)s = NULL;
                %(fail)s;
            }                         
        }
        """ % d

    def c_code_cache_version(self):
        return ()

