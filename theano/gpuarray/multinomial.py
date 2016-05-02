# TODO test dtype != float32
from __future__ import absolute_import, print_function, division
import os

try:
    import pygpu
except ImportError:
    pass

import theano
import theano.sandbox.multinomial
from theano import Apply, config
from theano.gof import Op
from theano.tensor import NotScalarConstantError, get_scalar_constant_value
from theano import gpuarray
from .basic_ops import as_gpuarray_variable, infer_context_name
from .opt import register_opt, op_lifter
from .type import GpuArrayType


class GPUAMultinomialFromUniform(gpuarray.basic_ops.GpuKernelBase, Op):
    __props__ = ("odtype",)

    def __init__(self, odtype):
        Op.__init__(self)
        self.odtype = odtype

    def get_params(self, node):
        return node.outputs[0].type.context

    def c_headers(self):
        return ['<numpy_compat.h>', 'gpuarray_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def make_node(self, pvals, unis):
        assert pvals.dtype == 'float32'
        assert unis.dtype == 'float32'
        ctx_name = infer_context_name(pvals, unis)

        pvals = as_gpuarray_variable(pvals, ctx_name)
        unis = as_gpuarray_variable(unis, ctx_name)

        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        assert odtype == 'float32', odtype
        if odtype != pvals.dtype:
            raise NotImplementedError(
                'GpuMultinomialFromUniform works only if '
                'self.odtype == pvals.dtype', odtype, pvals.dtype)
        br = (pvals.broadcastable[1], pvals.broadcastable[0])
        out = GpuArrayType(broadcastable=br,
                           dtype=odtype,
                           context_name=ctx_name)()

        return Apply(self, [pvals, unis], [out])

    def gpu_kernels(self, node, name):
        code = """
KERNEL void k_multi_warp_multinomial(
    const ga_size nb_multi,
    const ga_size nb_outcomes,
    GLOBAL_MEM float * global_pvals,
    const ga_ssize pvals_row_stride,
    const ga_ssize pvals_col_stride,
    GLOBAL_MEM float * global_unis,
    const ga_ssize unis_stride,
    GLOBAL_MEM float * global_outs,
    const ga_ssize outs_row_stride,
    const ga_ssize outs_col_stride
)
{
    // each thread takes care of one multinomial draw
    int n = LDIM_0*GID_0 + LID_0;
    if (n < nb_multi)
    {
        float cummul = 0.;
        bool done = false;
        const float unis_n = global_unis[n*unis_stride];
        for (ga_size m = 0; m < nb_outcomes; ++m)
        {
            float current_out = 0.;
            if (!done)
            {
                cummul += global_pvals[m * pvals_col_stride +
                                       n * pvals_row_stride];
                if (unis_n < cummul)
                {
                    current_out = 1.;
                    done = true;
                }
            }
            //write out transposed for speed.
            global_outs[n * outs_col_stride +
                        m * outs_row_stride] = current_out;
        }
    }
}
"""
        return [gpuarray.basic_ops.Kernel(
            code=code, name="k_multi_warp_multinomial",
            params=[pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.GpuArray,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.GpuArray,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.GpuArray,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.SSIZE],
            flags=gpuarray.basic_ops.Kernel.get_flags(node.outputs[0].dtype),
            objvar='k_multi_warp_multinomial_' + name)]

    def c_code(self, node, name, inp, outputs, sub):
        pvals, unis = inp
        out, = outputs
        fail = sub['fail']
        ctx = sub['params']
        sync = bool(config.gpuarray.sync)
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
        PyGpuArrayObject * pvals = %(pvals)s;
        PyGpuArrayObject * unis = %(unis)s;
        PyGpuArrayObject * out = %(out)s;

    size_t dims[2];
    if (PyGpuArray_NDIM(pvals) != 2)
    {
        PyErr_Format(PyExc_TypeError, "pvals wrong rank");
        %(fail)s
    }
    if (PyGpuArray_NDIM(unis) != 1)
    {
        PyErr_Format(PyExc_TypeError, "unis wrong rank");
        %(fail)s
    }
    if (PyGpuArray_DIMS(unis)[0] != PyGpuArray_DIMS(pvals)[0])
    {
        PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0]");
        %(fail)s
    }

    dims[0] = PyGpuArray_DIMS(pvals)[1];
    dims[1] = PyGpuArray_DIMS(pvals)[0];
    if (theano_prep_output(&out, 2, dims, unis->ga.typecode,
                           GA_C_ORDER, %(ctx)s) != 0){
      %(fail)s
    }
    %(out)s = out;
    GpuArray_memset(&(out->ga), 0);
    { // NESTED SCOPE
        int nb_multi = PyGpuArray_DIMS(pvals)[0];
        int nb_outcomes = PyGpuArray_DIMS(pvals)[1];
        //TODO : change this for a beautiful constant
        int max_nb_blocks = 2<<15 - 1;
        size_t nb_blocks = max_nb_blocks + 1;
        size_t nb_threads=16; // so it really starts at 32, because of the *2
        do
        {
            nb_threads*=2;
            if (nb_multi % %nb_threads == 0)
                nb_blocks = nb_multi/nb_threads;
            else
                nb_blocks = (int)((float)nb_multi/(float)nb_threads + 1.);
        } while (nb_blocks > max_nb_blocks);

        //printf("\\nN=%%i b=%%i t=%%i t*b=%%i",
        //         nb_multi, nb_blocks, nb_threads, nb_blocks*nb_threads);

        // TODO : next line is a bit hardcoded...
        if (nb_threads > 512)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Multinomial is not implemented for so many rows in the matrix (%%i)",
                nb_multi);
            %(fail)s
        }

        assert(nb_blocks*nb_threads >= nb_multi);

        void *args[10];
        ssize_t strides[5] = {
            PyGpuArray_STRIDES(pvals)[0]/sizeof(float),
            PyGpuArray_STRIDES(pvals)[1]/sizeof(float),
            PyGpuArray_STRIDES(unis)[0]/sizeof(float),
            PyGpuArray_STRIDES(out)[0]/sizeof(float),
            PyGpuArray_STRIDES(out)[1]/sizeof(float)
        };
        int err;
        args[0] = (void*)&PyGpuArray_DIMS(out)[1];
        args[1] = (void*)&PyGpuArray_DIMS(out)[0];
        args[2] = pvals->ga.data; //PyGpuArray_DEV_DATA(pvals);
        args[3] = (void*)&strides[0];
        args[4] = (void*)&strides[1];
        args[5] = unis->ga.data; //PyGpuArray_DEV_DATA(unis);
        args[6] = (void*)&strides[2];
        args[7] = out->ga.data; //PyGpuArray_DEV_DATA(out);
        args[8] = (void*)&strides[3];
        args[9] = (void*)&strides[4];

        err = GpuKernel_call(&%(kname)s, 1, &nb_threads, &nb_blocks, 0, args);
        if (err != GA_NO_ERROR) {
           PyErr_Format(
                PyExc_RuntimeError,
                "gpuarray error: %%s: %%s.\\n",
                "k_multi_warp_%(name)s",
                GpuKernel_error(&%(kname)s, err));
            %(fail)s;
        }
        if(%(sync)d)
            GpuArray_sync(&(out->ga));
    } // END NESTED SCOPE
        """ % locals()

        return s

    def c_code_cache_version(self):
        return (1,)


@register_opt()
@op_lifter([theano.sandbox.multinomial.MultinomialFromUniform])
def local_gpua_multinomial(node, context_name):
    # TODO : need description for function

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
    if (p.dtype == u.dtype == m.dtype == 'float32'):
        gpu_op = GPUAMultinomialFromUniform(node.op.odtype)
        return gpuarray.elemwise.GpuDimShuffle([False, False], [1, 0])(
            gpu_op(p, u))
