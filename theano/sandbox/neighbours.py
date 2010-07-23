import theano
from theano import Op, Apply
import theano.tensor as T
from theano.tensor.opt import register_specialize
from theano.gof import local_optimizer
from theano.sandbox.cuda import cuda_available

if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host
    from theano.sandbox.cuda.opt import register_opt as register_gpu_opt

class Images2Neibs(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return self.__class__.__name__
    def make_node(self, ten4, neib_shape):
        ten4 = T.as_tensor_variable(ten4)
        neib_shape = T.as_tensor_variable(neib_shape)
        return Apply(self, [ten4, neib_shape], [T.matrix(dtype=ten4.type.dtype)])

    def grad(self, (pvals, unis), (gz,)):
        return [None, None]

    def c_code_cache_version(self):
        return (2,)
                
    def c_code(self, node, name, (ten4, neib_shape), (z,), sub):

        fail = sub['fail']
        return """
        {
        if (%(ten4)s->nd != 4)
        {
            PyErr_Format(PyExc_TypeError, "ten4 wrong rank");
            %(fail)s;
        }
        if (%(neib_shape)s->nd != 1)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape wrong rank");
            %(fail)s;
        }
        if ( (%(neib_shape)s->dimensions)[0] != 2)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape wrong shape ; has to contain 2 elements");
            %(fail)s;
        }
        
        const npy_intp c = (npy_intp) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
        const npy_intp d = (npy_intp) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);

        if ( (%(ten4)s->dimensions)[2] %% c != 0)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape[0] must divide ten4.shape[2]");
            %(fail)s;
        }
        if ( (%(ten4)s->dimensions)[3] %% d != 0)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape[1] must divide ten4.shape[3]");
            %(fail)s;
        }
        
        // new dimensions for z
        const npy_intp z_dim1 = c * d;
        const npy_intp z_dim0 =  (%(ten4)s->dimensions)[2] / c
                            * (%(ten4)s->dimensions)[3] / d
                            * (%(ten4)s->dimensions)[1]
                            * (%(ten4)s->dimensions)[0];
        
        if ((NULL == %(z)s)
            || ((%(z)s->dimensions)[0] != z_dim0 )
            || ((%(z)s->dimensions)[1] != z_dim1 )
        )
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = z_dim0;
            dims[1] = z_dim1;
            
            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                dims,
                type_num_%(ten4)s,
                0);
                       
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }
        }

        { // NESTED SCOPE
        
        const int nb_batch = (%(ten4)s->dimensions)[0];
        const int nb_stack = (%(ten4)s->dimensions)[1];
        const int height = (%(ten4)s->dimensions)[2];
        const int width = (%(ten4)s->dimensions)[3];
        
        // (c,d) = neib_shape
        const int c = (int) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
        const int d = (int) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);

        const int grid_c = height/c;
        const int grid_d = width/d;

        // Oh this is messed up...      
        for (int n = 0; n < nb_batch; n++)              // loop over batches
            for (int s = 0; s < nb_stack; s++)          // loop over stacks
                for (int a = 0; a < grid_c; a++)        // loop over height/c
                    for (int b = 0; b < grid_d; b++)    // loop over width/d
                    {
                        int z_row = b + grid_d*(a + grid_c*(s + nb_stack*n));
                        for (int i = 0; i < c; i++)     // loop over c
                        {
                            int ten4_2 = i + a * c;
                            for (int j = 0; j < d; j++)  // loop over d
                            {
                                
                                int ten4_3 = j + b * d;     
                                int z_col = j + d * i;
                                
                                dtype_%(z)s* curr_z = (dtype_%(z)s*) PyArray_GETPTR2(%(z)s, z_row, z_col);
                                *curr_z = *( (dtype_%(ten4)s*) PyArray_GETPTR4(%(ten4)s, n, s, ten4_2, ten4_3));
                                
                                //printf("\\n(%%i,%%i,%%i,%%i) --> (%%i,%%i)",n,s, ten4_2, ten4_3, z_row, z_col);
                                //printf("%%f ", *curr_z);
                            }
                        }
                    }
        } // END NESTED SCOPE
        """ % locals()
images2neibs = Images2Neibs()

def neibs2images(neibs, neib_shape, original_shape):
    """
    Inverse of images2neib.
    
    neibs : matrix like the one obtained by images2neib
    neib_shape : neib_shape that was used in images2neib
    original_shape : original shape of the 4d tensor given to images2neib
    
    Return a 4d tensor of shape `original_shape`.
    """
    neibs = T.as_tensor_variable(neibs)
    neib_shape = T.as_tensor_variable(neib_shape)
    original_shape = T.as_tensor_variable(original_shape)
    
    new_neib_shape = T.stack( original_shape[-1]/neib_shape[1], neib_shape[1] )
    return images2neibs(neibs.dimshuffle('x','x',0,1), new_neib_shape).reshape(original_shape)
    
   
# This is work in progress
class GpuImages2Neibs(Images2Neibs):

    def make_node(self, ten4, neib_shape):
        assert ten4.dtype == 'float32'
        #assert neib_shape.dtype == 'float32'
        if not isinstance(ten4.type, CudaNdarrayType):
            raise TypeError('pvals must be cudandarray', ten4)
        #if not isinstance(neib_shape.type, CudaNdarrayType):
        #    raise TypeError('unis must be cudandarray', neib_shape)
        #print 'neib_shape type and dtype', type(neib_shape), neib_shape.dtype

        return Apply(self, [ten4, neib_shape], [CudaNdarrayType(broadcastable=(False,False),
                                                                dtype=ten4.type.dtype)()])

    def c_code_cache_version(self):
        return (2,)

    def c_support_code_apply(self, node, nodename):
        return """
        static __global__ void k_multi_warp_%(nodename)s(
            const int nb_batch,
            const int nb_stack,
            const int height,
            const int width,
            const int c,
            const int d,
            const int grid_c,
            const int grid_d,
            const int stride0, const int stride1, const int stride2, const int stride3,
            float * global_ten4,
            float * global_out
        )
        {
        
            for(int tblock = blockIdx.x;tblock<nb_batch*nb_stack*grid_c*grid_d;tblock+=gridDim.x){
                const int b = tblock%%grid_d;
                int left = tblock/grid_d;
                const int a = left%%grid_c;
                left = left/grid_c;
                const int s = left%%nb_stack;
                left = left/nb_stack;
                const int n = left;

                if(n>nb_batch)continue;
                if(s>nb_stack)continue;
                if(a>grid_c)continue;
                if(b>grid_d)continue;
                            int z_row = b + grid_d*(a + grid_c*(s + nb_stack*n));
                            for (int i = 0; i < c; i++)     // loop over c
                            {
                                int ten4_2 = i + a * c;
                                for (int j = threadIdx.x; j < d; j+=blockDim.x)  // loop over d
                                {
                                    int ten4_3 = j + b * d;
                                    //int ten4_idx = ten4_3 + width*(ten4_2 + height*(s +nb_stack*n));
                                    //int ten4_idx = stride3*ten4_3 + stride2*(ten4_2 + stride1*(s + stride0*n)); 
                                    int ten4_idx = stride3*ten4_3 + stride2*ten4_2 + stride1*s + stride0*n; 

                                    int z_col = j + d * i;
                                    int z_idx = z_col + c*d*z_row;
                                    global_out[z_idx] = global_ten4[ten4_idx];
                                }
                            }
            }
        }

        """ % locals()


    def c_code(self, node, name, (ten4, neib_shape), (z,), sub):
        fail = sub['fail']
        return """
        {
            if (%(ten4)s->nd != 4)
            {
                PyErr_Format(PyExc_TypeError, "pvals wrong rank");
                %(fail)s;
            }
            if (%(neib_shape)s->nd != 1)
            {
                PyErr_Format(PyExc_TypeError, "unis wrong rank");
                %(fail)s;
            }
            
            //if (CudaNdarray_HOST_DIMS(%(neib_shape)s)[0] != 2)
            if (%(neib_shape)s->dimensions[0] != 2)
            {
                PyErr_Format(PyExc_ValueError, "neib_shape has to contain two elements");
                %(fail)s;
            }

            /*if (!CudaNdarray_is_c_contiguous(%(neib_shape)s))
            {
                PyErr_Format(PyExc_NotImplementedError, "require unis to be contiguous");
                %(fail)s;
            }*/
            /*if (!CudaNdarray_is_c_contiguous(%(ten4)s))
            {
                PyErr_Format(PyExc_NotImplementedError, "require ten4 to be contiguous");
                %(fail)s;
            }*/

            const int c = *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
            const int d = *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);
            
            //const float * cd = CudaNdarray_DEV_DATA(%(neib_shape)s);
            //const int c = (int) cd[0];
            //const int d = (int) cd[1];
            
            if ( CudaNdarray_HOST_DIMS(%(ten4)s)[2] %% c != 0)
            {
                PyErr_Format(PyExc_TypeError, "neib_shape[0] must divide ten4.shape[2]");
                %(fail)s;
            }
            if ( CudaNdarray_HOST_DIMS(%(ten4)s)[3] %% d != 0)
            {
                PyErr_Format(PyExc_TypeError, "neib_shape[1] must divide ten4.shape[3]");
                %(fail)s;
            }
            
            // new dimensions for z
            const int z_dim1 = c * d;
            const int z_dim0 =  CudaNdarray_HOST_DIMS(%(ten4)s)[2] / c
                                * CudaNdarray_HOST_DIMS(%(ten4)s)[3] / d
                                * CudaNdarray_HOST_DIMS(%(ten4)s)[1]
                                * CudaNdarray_HOST_DIMS(%(ten4)s)[0];
            
            if ((NULL == %(z)s)
                || (CudaNdarray_HOST_DIMS(%(z)s)[0] != z_dim0)
                || (CudaNdarray_HOST_DIMS(%(z)s)[1] != z_dim1))
            {
                Py_XDECREF(%(z)s);
                npy_intp dims[2];
                dims[0] = z_dim0;
                dims[1] = z_dim1;
                %(z)s = (CudaNdarray*)CudaNdarray_NewDims(2, dims);
                if (!%(z)s)
                {
                    PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                    %(fail)s;
                }
            }
        
        }

        { // NESTED SCOPE
        
            const int nb_batch = CudaNdarray_HOST_DIMS(%(ten4)s)[0];
            const int nb_stack = CudaNdarray_HOST_DIMS(%(ten4)s)[1];
            const int height = CudaNdarray_HOST_DIMS(%(ten4)s)[2];
            const int width = CudaNdarray_HOST_DIMS(%(ten4)s)[3];

            /*for (int i=0; i<4; i++)
            {
                 printf("\\ndim%%i %%i",i, CudaNdarray_HOST_DIMS(%(ten4)s)[i]);
                 printf("\\nstride%%i %%i",i, CudaNdarray_HOST_STRIDES(%(ten4)s)[i]);
            }*/
            // (c,d) = neib_shape
            //const float * cd = CudaNdarray_DEV_DATA(%(neib_shape)s);
            //const int c = (int) cd[0];
            //const int d = (int) cd[1];

            const int c = *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
            const int d = *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);
            
            const int grid_c = height/c;
            const int grid_d = width/d;

            int nb_block;
            if (nb_batch %% 32 == 0)
                nb_block = nb_batch/32;
            else
                nb_block = (int)((float)nb_batch/32. + 1.); 
                
            dim3 n_blocks(std::min(32*1024,CudaNdarray_HOST_DIMS(%(z)s)[0]),1,1);
            dim3 n_threads(32,1,1);
            int n_shared = 0;

            k_multi_warp_%(name)s<<<n_blocks, n_threads, n_shared>>>(                
                nb_batch,
                nb_stack,
                height, width,
                c, d,
                grid_c, grid_d,
                CudaNdarray_HOST_STRIDES(%(ten4)s)[0],
                CudaNdarray_HOST_STRIDES(%(ten4)s)[1],
                CudaNdarray_HOST_STRIDES(%(ten4)s)[2],
                CudaNdarray_HOST_STRIDES(%(ten4)s)[3],
                CudaNdarray_DEV_DATA(%(ten4)s),
                CudaNdarray_DEV_DATA(%(z)s)
            );
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i; shared: %%i)\\n",
                    "k_multi_warp_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z,
                    n_shared);
                %(fail)s;
            }

        } // END NESTED SCOPE
        """ % locals()
gpu_images2neibs = GpuImages2Neibs()

@local_optimizer()
def use_gpu_images2neibs(node):
    if node.op == images2neibs:
        return [host_from_gpu(gpu_images2neibs(*[gpu_from_host(node.inputs[0]),node.inputs[1]]))]

if cuda_available:
    register_gpu_opt()(use_gpu_images2neibs)
    
