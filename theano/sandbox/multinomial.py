from __future__ import absolute_import, print_function, division
import numpy

import theano
from theano import Op, Apply
import theano.tensor as T
from theano.gof import local_optimizer
from theano.tensor import NotScalarConstantError, get_scalar_constant_value
from theano.scalar import as_scalar
import copy

from theano.sandbox.cuda import cuda_available, GpuOp
if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host
    from theano.sandbox.cuda.opt import register_opt


class MultinomialFromUniform(Op):
    # TODO : need description for parameter 'odtype'
    """
    Converts samples from a uniform into sample from a multinomial.

    """

    __props__ = ("odtype",)

    def __init__(self, odtype):
        self.odtype = odtype

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        try:
            self.odtype
        except AttributeError:
            self.odtype = 'auto'

    def make_node(self, pvals, unis, n=1):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        out = T.tensor(dtype=odtype, broadcastable=pvals.type.broadcastable)
        return Apply(self, [pvals, unis, as_scalar(n)], [out])

    def grad(self, ins, outgrads):
        pvals, unis, n = ins
        (gz,) = outgrads
        return [T.zeros_like(x, dtype=theano.config.floatX) if x.dtype in
                T.discrete_dtypes else T.zeros_like(x) for x in ins]

    def c_code_cache_version(self):
        return (8,)

    def c_code(self, node, name, ins, outs, sub):
        # support old pickled graphs
        if len(ins) == 2:
            (pvals, unis) = ins
            n = 1
        else:
            (pvals, unis, n) = ins
        (z,) = outs
        if self.odtype == 'auto':
            t = "PyArray_TYPE(%(pvals)s)" % locals()
        else:
            t = theano.scalar.Scalar(self.odtype).dtype_specs()[1]
            if t.startswith('theano_complex'):
                t = t.replace('theano_complex', 'NPY_COMPLEX')
            else:
                t = t.upper()
        fail = sub['fail']
        return """
        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals ndim should be 2");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis ndim should be 2");
            %(fail)s;
        }

        if (PyArray_DIMS(%(unis)s)[0] != (PyArray_DIMS(%(pvals)s)[0] * %(n)s))
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0] * n");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(pvals)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != (PyArray_DIMS(%(pvals)s))[1])
        )
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                PyArray_DIMS(%(pvals)s),
                %(t)s,
                0);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE

        const int nb_multi = PyArray_DIMS(%(pvals)s)[0];
        const int nb_outcomes = PyArray_DIMS(%(pvals)s)[1];
        const int n_samples = %(n)s;

        //
        // For each multinomial, loop over each possible outcome
        //
        for (int c = 0; c < n_samples; ++c){
            for (int n = 0; n < nb_multi; ++n)
            {
                int waiting = 1;
                double cummul = 0.;
                const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, c*nb_multi + n);
                for (int m = 0; m < nb_outcomes; ++m)
                {
                    dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n,m);
                    const dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(%(pvals)s, n,m);
                    cummul += *pvals_nm;
                    if (c == 0)
                    {
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
                    else {
                        if (cummul > *unis_n)
                        {
                            *z_nm = *z_nm + 1.;
                            break;
                        }
                    }
                }
            }
        }
        } // END NESTED SCOPE
        """ % locals()

    def perform(self, node, ins, outs):
        # support old pickled graphs
        if len(ins) == 2:
            (pvals, unis) = ins
            n_samples = 1
        else:
            (pvals, unis, n_samples) = ins
        (z,) = outs

        if unis.shape[0] != pvals.shape[0] * n_samples:
            raise ValueError("unis.shape[0] != pvals.shape[0] * n_samples",
                             unis.shape[0], pvals.shape[0], n_samples)
        if z[0] is None or z[0].shape != pvals.shape:
            z[0] = numpy.zeros(pvals.shape, dtype=node.outputs[0].dtype)

        nb_multi = pvals.shape[0]
        nb_outcomes = pvals.shape[1]
        # For each multinomial, loop over each possible outcome
        for c in range(n_samples):
            for n in range(nb_multi):
                waiting = True
                cummul = 0
                unis_n = unis[c * nb_multi + n]

                for m in range(nb_outcomes):
                    cummul += pvals[n, m]
                    if c == 0:
                        if (waiting and (cummul > unis_n)):
                            z[0][n, m] = 1
                            waiting = False
                        else:
                            z[0][n, m] = 0
                    else:
                        if (cummul > unis_n):
                            z[0][n, m] += 1
                            break


class MultinomialWOReplacementFromUniform(MultinomialFromUniform):
    """
    Converts samples from a uniform into sample (without replacement) from a
    multinomial.

    """

    def make_node(self, pvals, unis, n=1):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = 'int64'
        else:
            odtype = self.odtype
        out = T.tensor(dtype=odtype, broadcastable=pvals.type.broadcastable)
        return Apply(self, [pvals, unis, as_scalar(n)], [out])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis, n) = ins
        (z,) = outs
        if self.odtype == 'auto':
            t = "NPY_INT64"
        else:
            t = theano.scalar.Scalar(self.odtype).dtype_specs()[1]
            if t.startswith('theano_complex'):
                t = t.replace('theano_complex', 'NPY_COMPLEX')
            else:
                t = t.upper()
        fail = sub['fail']
        return """
        // create a copy of pvals matrix
        PyArrayObject* pvals_copy = NULL;

        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals ndim should be 2");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis ndim should be 2");
            %(fail)s;
        }

        if ( %(n)s > (PyArray_DIMS(%(pvals)s)[1]) )
        {
            PyErr_Format(PyExc_ValueError, "Cannot sample without replacement n samples bigger than the size of the distribution.");
            %(fail)s;
        }

        if (PyArray_DIMS(%(unis)s)[0] != (PyArray_DIMS(%(pvals)s)[0] * %(n)s))
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0] * n");
            %(fail)s;
        }

        pvals_copy = (PyArrayObject*) PyArray_EMPTY(2,
            PyArray_DIMS(%(pvals)s),
            PyArray_TYPE(%(pvals)s),
            0);

        if (!pvals_copy)
        {
            PyErr_SetString(PyExc_MemoryError, "failed to alloc pvals_copy");
            %(fail)s;
        }
        PyArray_CopyInto(pvals_copy, %(pvals)s);

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(pvals)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != %(n)s)
        )
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = PyArray_DIMS(%(pvals)s)[0];
            dims[1] = %(n)s;
            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                dims,
                %(t)s,
                -1);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE

        const int nb_multi = PyArray_DIMS(%(pvals)s)[0];
        const int nb_outcomes = PyArray_DIMS(%(pvals)s)[1];
        const int n_samples = %(n)s;

        //
        // For each multinomial, loop over each possible outcome,
        // and set selected pval to 0 after being selected
        //
        for (int c = 0; c < n_samples; ++c){
            for (int n = 0; n < nb_multi; ++n)
            {
                double cummul = 0.;
                const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, c*nb_multi + n);
                dtype_%(z)s* z_nc = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n, c);
                for (int m = 0; m < nb_outcomes; ++m)
                {
                    dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(pvals_copy, n, m);
                    cummul += *pvals_nm;
                    if (cummul > *unis_n)
                    {
                        *z_nc = m;
                        // renormalize the nth row of pvals, reuse (cummul-*pvals_nm) to initialize the sum
                        dtype_%(pvals)s sum = cummul - *pvals_nm;
                        dtype_%(pvals)s* pvals_n = (dtype_%(pvals)s*)PyArray_GETPTR2(pvals_copy, n, m);
                        *pvals_nm = 0.;
                        for (int k = m; k < nb_outcomes; ++k)
                        {
                            sum = sum + *pvals_n;
                            pvals_n++;
                        }
                        pvals_n = (dtype_%(pvals)s*)PyArray_GETPTR2(pvals_copy, n, 0);
                        for (int k = 0; k < nb_outcomes; ++k)
                        {
                            *pvals_n = *pvals_n / sum;
                            pvals_n++;
                        }
                        break;
                    }
                }
            }
        }

        // delete pvals_copy
        {
            Py_XDECREF(pvals_copy);
        }
        } // END NESTED SCOPE
        """ % locals()

    def perform(self, node, ins, outs):
        (pvals, unis, n_samples) = ins
        # make a copy so we do not overwrite the input
        pvals = copy.copy(pvals)
        (z,) = outs

        if n_samples > pvals.shape[1]:
            raise ValueError("Cannot sample without replacement n samples "
                             "bigger than the size of the distribution.")

        if unis.shape[0] != pvals.shape[0] * n_samples:
            raise ValueError("unis.shape[0] != pvals.shape[0] * n_samples",
                             unis.shape[0], pvals.shape[0], n_samples)

        if self.odtype == 'auto':
            odtype = 'int64'
        else:
            odtype = self.odtype
        if (z[0] is None or
                not numpy.all(z[0].shape == [pvals.shape[0], n_samples])):
            z[0] = -1 * numpy.ones((pvals.shape[0], n_samples), dtype=odtype)

        nb_multi = pvals.shape[0]
        nb_outcomes = pvals.shape[1]

        # For each multinomial, loop over each possible outcome,
        # and set selected pval to 0 after being selected
        for c in range(n_samples):
            for n in range(nb_multi):
                cummul = 0
                unis_n = unis[c * nb_multi + n]
                for m in range(nb_outcomes):
                    cummul += pvals[n, m]
                    if (cummul > unis_n):
                        z[0][n, c] = m
                        # set to zero and re-normalize so that it's not
                        # selected again
                        pvals[n, m] = 0.
                        pvals[n] /= pvals[n].sum()
                        break


class GpuMultinomialFromUniform(MultinomialFromUniform, GpuOp):
    """
    The output is transposed compared to MultinomialFromUniform.
    We must insert a Transpose op after it.

    The optimization that moves it to the gpu does it.

    """

    def make_node(self, pvals, unis):
        assert pvals.dtype == 'float32'
        assert unis.dtype == 'float32'
        if not isinstance(pvals.type, CudaNdarrayType):
            raise TypeError('pvals must be cudandarray', pvals)
        if not isinstance(unis.type, CudaNdarrayType):
            raise TypeError('unis must be cudandarray', unis)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        if odtype != pvals.dtype:
            raise NotImplementedError(
                'GpuMultinomialFromUniform works only if '
                'self.odtype == pvals.dtype', odtype, pvals.dtype)
        br = (pvals.broadcastable[1], pvals.broadcastable[0])
        out = CudaNdarrayType(broadcastable=br)()
        return Apply(self, [pvals, unis], [out])

    def perform(self, node, ins, outs):
        # The perform from parent don't work with CudaNdarray.  We
        # don't need it as DebugMode will test again it as an
        # optimization insert the GPU op.
        return Op.perform(self, node, ins, outs)

    def c_code_cache_version(self):
        return (8,)

    def c_support_code_apply(self, node, nodename):
        return """
        static __global__ void k_multi_warp_%(nodename)s(
            const int nb_multi,
            const int nb_outcomes,
            float * global_pvals,
            const int pvals_row_stride,
            const int pvals_col_stride,
            float * global_unis,
            const int unis_stride,
            float * global_outs,
            const int outs_row_stride,
            const int outs_col_stride
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
                    cummul += global_pvals[m * pvals_col_stride + n * pvals_row_stride];
                    if (unis_n < cummul)
                    {
                        current_out = 1.;
                        done = true;
                    }
                }
                //write out transposed for speed.
                global_outs[n * outs_col_stride + m * outs_row_stride] = current_out;
            }
            }
        }

        """ % locals()

    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis) = ins
        (z,) = outs

        fail = sub['fail']
        return """
        if (CudaNdarray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals wrong rank");
            %(fail)s;
        }
        if (CudaNdarray_NDIM(%(unis)s) != 1)
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
                CudaNdarray_DEV_DATA(%(pvals)s),
                CudaNdarray_HOST_STRIDES(%(pvals)s)[0],
                CudaNdarray_HOST_STRIDES(%(pvals)s)[1],
                CudaNdarray_DEV_DATA(%(unis)s),
                CudaNdarray_HOST_STRIDES(%(unis)s)[0],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1]
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


@local_optimizer([MultinomialFromUniform])
def local_gpu_multinomial(node):
    # TODO : need description for function
    if type(node.op) is MultinomialFromUniform:
        if len(node.inputs) == 2:
            p, u = node.inputs
            n_samples = 1
        else:
            p, u, n_samples = node.inputs
        try:
            if get_scalar_constant_value(n_samples) != 1:
                return None
        except NotScalarConstantError:
            return None
        m, = node.outputs
        if (p.dtype == u.dtype == m.dtype == 'float32' and
            any([i.owner and isinstance(i.owner.op,
                                        theano.sandbox.cuda.HostFromGpu)
                 for i in node.inputs])):
            gpu_op = GpuMultinomialFromUniform(node.op.odtype)
            return [host_from_gpu(gpu_op(*[gpu_from_host(i)
                                           for i in [p, u]])).T]
    if (isinstance(node.op, theano.sandbox.cuda.GpuFromHost) and
            node.inputs[0].owner and
            type(node.inputs[0].owner.op) is MultinomialFromUniform):
        multi = node.inputs[0].owner
        if len(node.inputs) == 2:
            p, u = node.inputs
            n_samples = 1
        else:
            p, u, n_samples = node.inputs
        try:
            if get_scalar_constant_value(n_samples) != 1:
                return None
        except NotScalarConstantError:
            return None
        m, = multi.outputs
        if (p.dtype == u.dtype == m.dtype == 'float32'):
            gpu_op = GpuMultinomialFromUniform(multi.op.odtype)
            ret = gpu_op(*[gpu_from_host(i) for i in [p, u]]).T
            # The dimshuffle is on the cpu, but will be moved to the
            # gpu by an opt.
            return [gpu_from_host(ret)]

if cuda_available:
    register_opt()(local_gpu_multinomial)
    pass
