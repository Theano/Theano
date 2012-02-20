import theano
from theano import Op, Apply
import theano.tensor as T
from theano.gof import local_optimizer

from theano.sandbox.cuda import cuda_available, GpuOp
if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host
    from theano.sandbox.cuda.opt import register_opt

class MultinomialFromUniform(Op):
    '''Converts samples from a uniform into sample from a multinomial.'''
    def __init__(self, odtype):
        self.odtype=odtype
    def __eq__(self, other):
        return type(self) == type(other) and self.odtype==other.odtype
    def __hash__(self):
        return hash((type(self), self.odtype))
    def __str__(self):
        return '%s{%s}'%(self.__class__.__name__, self.odtype)
    def __setstate__(self, dct):
        self.__dict__.update(dct)
        try:
            self.odtype
        except AttributeError:
            self.odtype='auto'
    def make_node(self, pvals, unis):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype=='auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        return Apply(self, [pvals, unis], [T.matrix(dtype=odtype)])

    def grad(self, ins, outgrads):
        pvals, unis = ins
        (gz,) = outgrads
        return [None, None]

    def c_code_cache_version(self):
        return (5,)

    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis) = ins
        (z,) = outs

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

        if (%(unis)s->dimensions[0] != %(pvals)s->dimensions[0])
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0]");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || ((%(z)s->dimensions)[0] != (%(pvals)s->dimensions)[0])
            || ((%(z)s->dimensions)[1] != (%(pvals)s->dimensions)[1])
        )
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_ZEROS(2,
                %(pvals)s->dimensions,
                type_num_%(z)s,
                0);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE

        const int nb_multi = %(pvals)s->dimensions[0];
        const int nb_outcomes = %(pvals)s->dimensions[1];

        //
        // For each multinomial, loop over each possible outcome
        //
        for (int n = 0; n < nb_multi; ++n)
        {
            int waiting = 1;
            dtype_%(pvals)s cummul = 0.;
            const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, n);
            for (int m = 0; m < nb_outcomes; ++m)
            {
                dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n,m);
                const dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(%(pvals)s, n,m);
                cummul += *pvals_nm;
                if (waiting && (cummul > *unis_n))
                {
                    *z_nm = 1.;
                    waiting = 0;
                }
                else
                {
                    // if we re-used old z pointer, we have to clear it out.
                    *z_nm = 0.;
                }
            }
        }
        } // END NESTED SCOPE
        """ % locals()


class GpuMultinomialFromUniform(MultinomialFromUniform, GpuOp):
    """
    The output is transposed compared to MultinomialFromUniform.
    We must insert a Transpose op after it.

    The optimization that move it to the gpu do it.
    """

    def make_node(self, pvals, unis):
        assert pvals.dtype == 'float32'
        assert unis.dtype == 'float32'
        if not isinstance(pvals.type, CudaNdarrayType):
            raise TypeError('pvals must be cudandarray', pvals)
        if not isinstance(unis.type, CudaNdarrayType):
            raise TypeError('unis must be cudandarray', unis)
        if self.odtype=='auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        if odtype != pvals.dtype:
            raise NotImplementedError(
                    'GpuMultinomialFromUniform works only if '
                    'self.odtype == pvals.dtype', odtype, pvals.dtype)
        return Apply(self, [pvals, unis], [pvals.type()])

    def c_code_cache_version(self):
        return (7,)

    def c_support_code_apply(self, node, nodename):
        return """
        static __global__ void k_multi_warp_%(nodename)s(
            const int nb_multi,
            const int nb_outcomes,
            const int pvals_row_strides,
            const int pvals_col_strides,
            const int unis_stride,
            float * global_pvals,
            float * global_unis,
            float * global_outs
        )
        {
            // each thread takes care of one multinomial draw
            int n = blockDim.x*blockIdx.x + threadIdx.x;
            if (n < nb_multi)
            {
            float cummul = 0.;
            bool done = false;
            const float unis_n = global_unis[n*unis_stride];
            for (int m = 0; m < nb_outcomes; ++m)
            {
                float current_out = 0.;
                if (!done)
                {
                    cummul += global_pvals[m * pvals_col_strides + n * pvals_row_strides];
                    if (unis_n < cummul)
                    {
                        current_out = 1.;
                        done = true;
                    }
                }
                //write out transposed for speed.
                global_outs[n + m * nb_multi] = current_out;
            }
            }
        }

        """ % locals()


    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis) = ins
        (z,) = outs

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
        if (CudaNdarray_HOST_DIMS(%(unis)s)[0] != CudaNdarray_HOST_DIMS(%(pvals)s)[0])
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0]");
            %(fail)s;
        }

        //N.B. that the output is TRANSPOSED compared with pvals
        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] != CudaNdarray_HOST_DIMS(%(pvals)s)[1])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] != CudaNdarray_HOST_DIMS(%(pvals)s)[0]))
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = (CudaNdarray_HOST_DIMS(%(pvals)s)[1]);
            dims[1] = (CudaNdarray_HOST_DIMS(%(pvals)s)[0]);
            %(z)s = (CudaNdarray*)CudaNdarray_NewDims(2, dims);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE
            int nb_multi = CudaNdarray_HOST_DIMS(%(pvals)s)[0];
            int nb_outcomes = CudaNdarray_HOST_DIMS(%(pvals)s)[1];
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
                PyErr_Format(PyExc_ValueError, "Mutinomial is not implemented for so many rows in the matrix (%%i)", nb_multi);
                %(fail)s;
            }
            dim3 n_blocks(nb_blocks,1,1);
            dim3 n_threads(nb_threads,1,1);
            int n_shared = 0;

            assert(nb_blocks*nb_threads >= nb_multi);

            k_multi_warp_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                CudaNdarray_HOST_DIMS(%(z)s)[1],
                CudaNdarray_HOST_DIMS(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(pvals)s)[0],
                CudaNdarray_HOST_STRIDES(%(pvals)s)[1],
                CudaNdarray_HOST_STRIDES(%(unis)s)[0],
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

@local_optimizer()
def local_gpu_multinomial(node):
    if type(node.op) is MultinomialFromUniform:
        p, u = node.inputs
        m, = node.outputs
        if (p.dtype == u.dtype == m.dtype == 'float32' and
            any([i.owner and isinstance(i.owner.op, theano.sandbox.cuda.HostFromGpu)
                 for i in node.inputs])):
            gpu_op = GpuMultinomialFromUniform(node.op.odtype)
            return [host_from_gpu(gpu_op(*[gpu_from_host(i) for i in node.inputs])).T]
    if (isinstance(node.op, theano.sandbox.cuda.GpuFromHost) and
        node.inputs[0].owner and type(node.inputs[0].owner.op) is MultinomialFromUniform):
        multi = node.inputs[0].owner
        p, u = multi.inputs
        m, = multi.outputs
        if (p.dtype == u.dtype == m.dtype == 'float32'):
            gpu_op = GpuMultinomialFromUniform(multi.op.odtype)
            ret = gpu_op(*[gpu_from_host(i) for i in multi.inputs]).T
            # The dimshuffle is on the cpu, but will be moved to the gpu by an opt.
            return [gpu_from_host(ret)]

if cuda_available:
    register_opt()(local_gpu_multinomial)
    pass
