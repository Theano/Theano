# TODO test dtype != float32
from __future__ import absolute_import, print_function, division
import warnings

try:
    import pygpu
except ImportError:
    pass

import theano
import theano.sandbox.multinomial
from theano import Apply
from theano.gof import Op

from theano.tensor import NotScalarConstantError, get_scalar_constant_value
from .basic_ops import as_gpuarray_variable, infer_context_name, GpuKernelBase, Kernel, gpuarray_helper_inc_dir
from .opt import register_opt, op_lifter, register_opt2
from .type import GpuArrayType
from .elemwise import GpuDimShuffle
from theano.scalar import as_scalar
from .fp16_help import write_w, load_w, work_dtype


class GPUAMultinomialFromUniform(GpuKernelBase, Op):
    __props__ = ("odtype",)
    _f16_ok = True

    def __init__(self, odtype):
        Op.__init__(self)
        self.odtype = odtype

    def get_params(self, node):
        return node.outputs[0].type.context

    def c_headers(self):
        return ['<numpy_compat.h>', 'gpuarray_helper.h']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def make_node(self, pvals, unis):
        ctx_name = infer_context_name(pvals, unis)
        pvals = as_gpuarray_variable(pvals, ctx_name)
        unis = as_gpuarray_variable(unis, ctx_name)
        assert pvals.dtype in ['float32', 'float16', 'float64']
        assert unis.dtype in ['float32', 'float16', 'float64']

        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        br = (pvals.broadcastable[1], pvals.broadcastable[0])
        out = GpuArrayType(broadcastable=br,
                           dtype=odtype,
                           context_name=ctx_name)()

        return Apply(self, [pvals, unis], [out])

    def gpu_kernels(self, node, name):
        out_ctype = pygpu.gpuarray.dtype_to_ctype(node.outputs[0].dtype)
        pvals_ctype = pygpu.gpuarray.dtype_to_ctype(node.inputs[0].dtype)
        unis_ctype = pygpu.gpuarray.dtype_to_ctype(node.inputs[1].dtype)
        work_ctype = pygpu.gpuarray.dtype_to_ctype(work_dtype(node.inputs[0].dtype))
        write_out_ctype = write_w(node.outputs[0].dtype)
        load_in_ctype = load_w(node.inputs[0].dtype)
        code = """#include "cluda.h"

KERNEL void k_multi_warp_multinomial(
    const ga_size nb_multi,
    const ga_size nb_outcomes,
    GLOBAL_MEM %(pvals_ctype)s *global_pvals,
    const ga_size global_pvals_offset,
    const ga_ssize pvals_row_stride,
    const ga_ssize pvals_col_stride,
    GLOBAL_MEM %(unis_ctype)s *global_unis,
    const ga_size global_unis_offset,
    const ga_ssize unis_stride,
    GLOBAL_MEM %(out_ctype)s *global_outs,
    const ga_size global_outs_offset,
    const ga_ssize outs_row_stride,
    const ga_ssize outs_col_stride
)
{
    global_pvals = (GLOBAL_MEM %(pvals_ctype)s *)(((GLOBAL_MEM char *)global_pvals) + global_pvals_offset);
    global_unis = (GLOBAL_MEM %(unis_ctype)s *)(((GLOBAL_MEM char *)global_unis) + global_unis_offset);
    global_outs = (GLOBAL_MEM %(out_ctype)s *)(((GLOBAL_MEM char *)global_outs) + global_outs_offset);
    // each thread takes care of one multinomial draw
    int n = LDIM_0*GID_0 + LID_0;
    if (n < nb_multi)
    {
        %(work_ctype)s cummul = 0.;
        bool done = false;
        const %(work_ctype)s unis_n = %(load_in_ctype)s(global_unis[n*unis_stride]);
        for (ga_size m = 0; m < nb_outcomes; ++m)
        {
            %(work_ctype)s current_out = 0;
            if (!done)
            {
                cummul += %(load_in_ctype)s(global_pvals[m * pvals_col_stride + n * pvals_row_stride]);
                if (unis_n < cummul)
                {
                    current_out = 1;
                    done = true;
                }
            }
            //write out transposed for speed.
            global_outs[n * outs_col_stride +
                        m * outs_row_stride] = %(write_out_ctype)s(current_out);
        }
    }
}
""" % dict(out_ctype=out_ctype, write_out_ctype=write_out_ctype,
           work_ctype=work_ctype, pvals_ctype=pvals_ctype,
           unis_ctype=unis_ctype, load_in_ctype=load_in_ctype)
        return [Kernel(
            code=code, name="k_multi_warp_multinomial",
            params=[pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.GpuArray,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.GpuArray,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.GpuArray,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.SSIZE],
            flags=Kernel.get_flags(node.outputs[0].dtype),
            objvar='k_multi_warp_multinomial_' + name)]

    def c_code(self, node, name, inp, outputs, sub):
        pvals, unis = inp
        out, = outputs
        fail = sub['fail']
        ctx = sub['params']
        kname = self.gpu_kernels(node, name)[0].objvar
        out_typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        pvals_typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[0].dtype)
        unis_typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[1].dtype)
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
    if (theano_prep_output(&out, 2, dims, %(out_typecode)s,
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
            if (nb_multi %% nb_threads == 0)
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

        int err = k_multi_warp_multinomial_call(
          1, &nb_blocks, &nb_threads, 0,
          PyGpuArray_DIMS(out)[1], PyGpuArray_DIMS(out)[0], pvals->ga.data, pvals->ga.offset,
          PyGpuArray_STRIDES(pvals)[0]/gpuarray_get_elsize(%(pvals_typecode)s),
          PyGpuArray_STRIDES(pvals)[1]/gpuarray_get_elsize(%(pvals_typecode)s),
          unis->ga.data, unis->ga.offset,
          PyGpuArray_STRIDES(unis)[0]/gpuarray_get_elsize(%(unis_typecode)s), out->ga.data,
          out->ga.offset, PyGpuArray_STRIDES(out)[0]/gpuarray_get_elsize(%(out_typecode)s),
          PyGpuArray_STRIDES(out)[1]/gpuarray_get_elsize(%(out_typecode)s));

        if (err != GA_NO_ERROR) {
           PyErr_Format(
                PyExc_RuntimeError,
                "gpuarray error: %%s: %%s.\\n",
                "k_multi_warp_%(name)s",
                GpuKernel_error(&%(kname)s, err));
            %(fail)s;
        }

    } // END NESTED SCOPE
        """ % locals()

        return s

    def c_code_cache_version(self):
        return (7,)


class GPUAChoiceFromUniform(GpuKernelBase, Op):
    """
    The output is transposed compared to MultinomialWOReplacementFromUniform.
    We must insert a Transpose op after it.

    The optimization that moves it to the gpu does it.

    """

    __props__ = ("odtype", "replace")

    def __init__(self, odtype, replace=False):
        Op.__init__(self)
        self.odtype = odtype
        self.replace = replace

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "replace" not in state:
            self.replace = False

    def get_params(self, node):
        return node.outputs[0].type.context

    def c_headers(self):
        return ['<numpy_compat.h>', 'gpuarray_helper.h']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def make_node(self, pvals, unis, n):
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
            odtype = 'int64'
        else:
            odtype = self.odtype
        assert odtype == 'int64', odtype
        br = (pvals.broadcastable[1], pvals.broadcastable[0])
        out = GpuArrayType(broadcastable=br,
                           dtype=odtype,
                           context_name=ctx_name)()

        return Apply(self, [pvals, unis, as_scalar(n)], [out])

    def gpu_kernels(self, node, name):
        replace = int(self.replace)
        code = """#include "cluda.h"

KERNEL void k_multi_warp_multinomial_wor(
    const ga_size nb_multi,
    const ga_size nb_outcomes,
    const ga_size n_samples,
    GLOBAL_MEM float * global_pvals_copy,
    const ga_size global_pvals_offset,
    const ga_ssize pvals_row_stride,
    const ga_ssize pvals_col_stride,
    GLOBAL_MEM float * global_unis,
    const ga_size global_unis_offset,
    const ga_ssize unis_stride,
    GLOBAL_MEM ga_long * global_outs,
    const ga_size global_outs_offset,
    const ga_ssize outs_row_stride,
    const ga_ssize outs_col_stride
)
{
    global_pvals_copy = (GLOBAL_MEM float *)(((GLOBAL_MEM char *)global_pvals_copy) + global_pvals_offset);
    global_unis = (GLOBAL_MEM float *)(((GLOBAL_MEM char *)global_unis) + global_unis_offset);
    global_outs = (GLOBAL_MEM ga_long *)(((GLOBAL_MEM char *)global_outs) + global_outs_offset);
    // each thread takes care of one multinomial-wor n_samples-draw
    int n = LDIM_0*GID_0 + LID_0;

    if (n < nb_multi)
    {
        // Sum of the remaining p_vals in global_pvals_copy[n]
        float pvals_sum = 1.;
        for (int c = 0; c < n_samples; ++c)
        {
            float cummul = 0.;
            const float unis_n = global_unis[(c * nb_multi + n)*unis_stride] * pvals_sum;
            for (ga_size m = 0; m < nb_outcomes; ++m)
            {
                float pvals_nm = global_pvals_copy[m * pvals_col_stride + n * pvals_row_stride];
                cummul += pvals_nm;

                if (unis_n < cummul)
                {
                    // write out transposed for speed.
                    global_outs[n * outs_col_stride +
                                c * outs_row_stride] = m;

                    if (! %(replace)s )
                    {
                        global_pvals_copy[m * pvals_col_stride + n * pvals_row_stride] = 0.0;
                        pvals_sum -= pvals_nm;
                    }
                    break;
                }
            }
        }
    }
}
""" % {"replace": replace}
        return [Kernel(
            code=code, name="k_multi_warp_multinomial_wor",
            params=[pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.GpuArray,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.GpuArray,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.GpuArray,
                    pygpu.gpuarray.SIZE,
                    pygpu.gpuarray.SSIZE,
                    pygpu.gpuarray.SSIZE
                    ],
            flags=Kernel.get_flags(node.outputs[0].dtype),
            objvar='k_multi_warp_multinomial_wor_' + name)]

    def c_code(self, node, name, inp, outputs, sub):
        pvals, unis, n = inp
        out, = outputs
        replace = int(self.replace)
        fail = sub['fail']
        ctx = sub['params']
        kname = self.gpu_kernels(node, name)[0].objvar
        s = """
    PyGpuArrayObject * pvals = %(pvals)s;
    PyGpuArrayObject * unis = %(unis)s;
    const size_t n_samples = %(n)s;
    PyGpuArrayObject * out = %(out)s;
    // create a copy of pvals matrix
    PyGpuArrayObject * pvals_copy = NULL;

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
    if ( n_samples > (PyGpuArray_DIMS(pvals)[1]) )
    {
        PyErr_Format(PyExc_ValueError, "Cannot sample without replacement n samples bigger than the size of the distribution.");
        %(fail)s;
    }
    if (PyGpuArray_DIMS(unis)[0] != PyGpuArray_DIMS(pvals)[0] * n_samples)
    {
        PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0] * n");
        %(fail)s
    }
    if (! %(replace)s) {
        pvals_copy = pygpu_copy(pvals, GA_C_ORDER);
    } else {
        pvals_copy = pvals;
        Py_INCREF(pvals_copy);
    }
    dims[0] = n_samples;
    dims[1] = PyGpuArray_DIMS(pvals)[0];

    if (theano_prep_output(&out, 2, dims, GA_LONG,
                           GA_C_ORDER, %(ctx)s) != 0){
        Py_DECREF(pvals_copy);
        %(fail)s
    }

    %(out)s = out;

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
            if (nb_multi %% nb_threads == 0)
                nb_blocks = nb_multi/nb_threads;
            else
                nb_blocks = (int)((float)nb_multi/(float)nb_threads + 1.);
        } while (nb_blocks > max_nb_blocks);

        // TODO : next line is a bit hardcoded...
        if (nb_threads > 512)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Multinomial is not implemented for so many rows in the matrix (%%i)",
                nb_multi);
            Py_DECREF(pvals_copy);
            %(fail)s
        }

        assert(nb_blocks*nb_threads >= nb_multi);

        int err = k_multi_warp_multinomial_wor_call(1, &nb_blocks, &nb_threads, 0, PyGpuArray_DIMS(pvals)[0], PyGpuArray_DIMS(pvals)[1], n_samples, pvals_copy->ga.data, pvals_copy->ga.offset, PyGpuArray_STRIDES(pvals)[0]/sizeof(float), PyGpuArray_STRIDES(pvals)[1]/sizeof(float), unis->ga.data, unis->ga.offset, PyGpuArray_STRIDES(unis)[0]/sizeof(float), out->ga.data, out->ga.offset, PyGpuArray_STRIDES(out)[0]/8, PyGpuArray_STRIDES(out)[1]/8);
        if (err != GA_NO_ERROR) {
           PyErr_Format(
                PyExc_RuntimeError,
                "gpuarray error: %%s: %%s.\\n",
                "k_multi_warp_%(name)s",
                GpuKernel_error(&%(kname)s, err));
           Py_DECREF(pvals_copy);
           %(fail)s;
        }

        Py_DECREF(pvals_copy);
    } // END NESTED SCOPE
        """ % locals()
        return s

    def c_code_cache_version(self):
        return (10,)


@register_opt('fast_compile')
@op_lifter([theano.sandbox.multinomial.MultinomialFromUniform])
@register_opt2([theano.sandbox.multinomial.MultinomialFromUniform], 'fast_compile')
def local_gpua_multinomial(op, context_name, inputs, outputs):
    # TODO : need description for function

    if len(inputs) == 2:
        p, u = inputs
        n_samples = 1
    else:
        p, u, n_samples = inputs
    try:
        if get_scalar_constant_value(n_samples) != 1:
            return None
    except NotScalarConstantError:
        return None
    m, = outputs
    gpu_op = GPUAMultinomialFromUniform(op.odtype)
    return GpuDimShuffle([False, False], [1, 0])(
        gpu_op(p, u))


@register_opt('fast_compile')
@op_lifter([theano.sandbox.multinomial.ChoiceFromUniform])
@register_opt2([theano.sandbox.multinomial.ChoiceFromUniform], 'fast_compile')
def local_gpua_multinomial_wor(op, context_name, inputs, outputs):
    # TODO : need description for function
    p, u, n = inputs
    m, = outputs
    if ((p.dtype == u.dtype == 'float32') and (m.dtype == 'int64')):
        gpu_op = GPUAChoiceFromUniform(**op._props_dict())
        return GpuDimShuffle([False, False], [1, 0])(
            gpu_op(p, u, n))


class GPUAMultinomialWOReplacementFromUniform(GPUAChoiceFromUniform):
    def __init__(self, *args, **kwargs):
        warnings.warn("GPUAMultinomialWOReplacementFromUniform is deprecated, "
                      "use GPUAChoiceFromUniform instead.",
                      DeprecationWarning,
                      stacklevel=2)
        super(GPUAMultinomialWOReplacementFromUniform, self).__init__(*args, **kwargs)
