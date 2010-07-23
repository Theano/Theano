import theano
from theano import Op, Apply
import theano.tensor as T
from theano.tensor.opt import register_specialize
from theano.gof import local_optimizer

from theano.sandbox.cuda import cuda_available, cuda_enabled
if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host

class Multinomial(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return self.__class__.__name__
    def make_node(self, pvals, unis):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        #assert pvals.dtype == 'float32'
        #assert unis.dtype == 'float32'
        return Apply(self, [pvals, unis], [pvals.type()])

    def grad(self, (pvals, unis), (gz,)):
        return [None, None]

    def c_code_cache_version(self):
        return (3,)
                
    def c_code(self, node, name, (pvals, unis), (z,), sub):

        fail = sub['fail']
        return """
        if (%(pvals)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals wrong rank");
            %(fail)s;
        }
        if (%(unis)s->nd != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis wrong rank");
            %(fail)s;
        }

        if (%(unis)s->dimensions[0] != %(pvals)s->dimensions[1])
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[1]");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || ((%(z)s->dimensions)[0] != (%(pvals)s->dimensions)[0])
            || ((%(z)s->dimensions)[1] != (%(pvals)s->dimensions)[1])
        )
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = (%(pvals)s->dimensions)[0];
            dims[1] = (%(pvals)s->dimensions)[1];
            
            %(z)s = (PyArrayObject*) PyArray_ZEROS(2,
                dims,
                type_num_%(pvals)s,
                0);
                       
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE

        const int nb_outcomes = %(pvals)s->dimensions[0];
        const int nb_multi = %(pvals)s->dimensions[1];

        //
        // For each multinomials, loop over each possible outcome
        //
        for (int n = 0; n < nb_multi; ++n)
        {
            dtype_%(pvals)s cummul = 0.;
            const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, n);
            for (int m = 0; m < nb_outcomes; ++m)
            {
                dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, m,n);
                const dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(%(pvals)s, m,n);
                cummul += *pvals_nm;
                if (*unis_n < cummul)
                {
                    *z_nm = 1.;
                    break;
                }
            }
        }
        
        } // END NESTED SCOPE
        """ % locals()
multinomial = Multinomial()


class GpuMultinomial(Multinomial):

    def make_node(self, pvals, unis):
        assert pvals.dtype == 'float32'
        assert unis.dtype == 'float32'
        if not isinstance(pvals.type, CudaNdarrayType):
            raise TypeError('pvals must be cudandarray', pvals)
        if not isinstance(unis.type, CudaNdarrayType):
            raise TypeError('unis must be cudandarray', unis)
        return Apply(self, [pvals, unis], [pvals.type()])

    def c_code_cache_version(self):
        #return ()
        return (super(GpuMultinomial,self).c_code_cache_version(),2)

    def c_support_code_apply(self, node, nodename):
        return """
        static __global__ void k_multi_warp_%(nodename)s(
            const int nb_multi,
            const int nb_outcomes,
            const int pvals_row_strides,
            const int pvals_col_strides,
            float * global_pvals,
            float * global_unis,
            float * global_outs
        )
        {            
            int n = blockDim.x*blockIdx.x + threadIdx.x;
            if (n < nb_multi)
            {    
            
            float cummul = 0.;
            bool done = false;
            for (int m = 0; m < nb_outcomes; ++m)
            {
                cummul += global_pvals[n * pvals_col_strides + m * pvals_row_strides];
                
                float current_out = 0.;

                if (!done && global_unis[n] < cummul)
                {
                    current_out = 1.;
                    done = true;
                }  
                global_outs[n + m * nb_multi] = current_out;
            }
            }
        }

        """ % locals()


    def c_code(self, node, name, (pvals, unis), (z,), sub):
        fail = sub['fail']
        return """
        
        if (%(pvals)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals wrong rank");
            %(fail)s;
        }
        if (%(unis)s->nd != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis wrong rank");
            %(fail)s;
        }
        
        if (CudaNdarray_HOST_DIMS(%(unis)s)[0] != CudaNdarray_HOST_DIMS(%(pvals)s)[1])
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[1]");
            %(fail)s;
        }
        if (!CudaNdarray_is_c_contiguous(%(unis)s))
        {
            PyErr_Format(PyExc_NotImplementedError, "require unis to be contiguous");
            %(fail)s;
        }
        // Would be more efficient if pvals were also contiguous but practically I think it is not often the cas,
        // since we are working on pvals.T here

        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] != CudaNdarray_HOST_DIMS(%(pvals)s)[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] != CudaNdarray_HOST_DIMS(%(pvals)s)[1]))
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = (CudaNdarray_HOST_DIMS(%(pvals)s)[0]);
            dims[1] = (CudaNdarray_HOST_DIMS(%(pvals)s)[1]);
            %(z)s = (CudaNdarray*)CudaNdarray_NewDims(2, dims);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE
            int nb_outcomes = CudaNdarray_HOST_DIMS(%(z)s)[0];
            int nb_multi = CudaNdarray_HOST_DIMS(%(z)s)[1];
            
            //TODO : change this for a beautiful constant
            int max_nb_blocks = 2<<15 - 1;
            int nb_blocks = max_nb_blocks + 1;
            int nb_threads=16; // so it really starts at 32, because of the *2
            do
            {
                nb_threads*=2;
                if (nb_multi %% nb_threads == 0)
                    nb_blocks = nb_multi/nb_threads;
                else
                    nb_blocks = (int)((float)nb_multi/(float)nb_threads + 1.); 
            } while (nb_blocks > max_nb_blocks);

            //printf("\\nN=%%i b=%%i t=%%i t*b=%%i", nb_multi, nb_blocks, nb_threads, nb_blocks*nb_threads);

            // TODO : next line is a bit hardcoded...
            if (nb_threads > 512)
            {
                PyErr_Format(PyExc_ValueError, "Mutinomial is not implemented for as many rows in the matrix (%%i)", nb_multi);
                %(fail)s;
            }

                
            dim3 n_blocks(nb_blocks,1,1);
            dim3 n_threads(nb_threads,1,1);
            int n_shared = 0;

            k_multi_warp_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                CudaNdarray_HOST_DIMS(%(z)s)[1],
                CudaNdarray_HOST_DIMS(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(pvals)s)[0],
                CudaNdarray_HOST_STRIDES(%(pvals)s)[1],
                CudaNdarray_DEV_DATA(%(pvals)s),
                CudaNdarray_DEV_DATA(%(unis)s),
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
gpu_multinomial = GpuMultinomial()

@local_optimizer()
def use_gpu_multinomial(node):
    if node.op == multinomial:
        return [host_from_gpu(gpu_multinomial(*[gpu_from_host(i) for i in node.inputs]))]
if cuda_enabled:#theano.config.device.startswith('gpu'):
    register_specialize(use_gpu_multinomial)
    
