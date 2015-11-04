import copy
import os

from theano import gof

try:
    from pygpu import gpuarray
except ImportError:
    pass

from .type import GpuArrayType
from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel,
                        infer_context_name)
from theano.gof import utils


class GpuConv(GpuKernelBase, gof.Op):
    """
    Implement the batched and stacked 2d convolution on the gpu.

    Parameters
    ----------
    version
        Each version of c_code implements many kernels for the convolution.
        By default we try to guess the best one. You can force one version with
        this parameter. This parameter is used by the tests.
    direction_hint
        'forward', 'bprop weights' or 'bprop inputs'. Serves as a hint for graph
        optimizers replacing GpuConv by other implementations. If the GpuConv is
        inserted automatically, we take its value from ConvOp.
    verbose
        For value of 1,2 and 3. Print more information during the execution of
        the convolution. Mostly used for optimization or debugging.
    kshp
        The size of the kernel. If provided, can generate faster code. If the
        GpuConv op is automatically inserted, we take its value automatically
        from the Conv op.
    imshp
        The size of the image. Not used for code generation but allows to select
        an experimental new version in another repo.
    max_threads_dim0
        The maximum number of threads for the block size dimensions 0
        (blockDim.x) used by the GPU function.
    nkern
        The number of kernels. Not used for this op, but can be used by graph
        optimizers to select a more optimal convolution implementation. If the
        GpuConv op is inserted automatically, we take its value from the Conv
        op.
    bsize
        The batch size. Not used for this op, but can be used by graph
        optimizers to select a more optimal convolution implementation. If the
        GpuConv op is inserted automatically, we take its value from the Conv
        op.
    fft_opt
        Deactivate fft_opt optimization at the op level when set to False. Note
        that by default fft optimization aren't enabled.
        See :ref:`convolution documentation <libdoc_tensor_nnet_conv>` to enable
        them.

    """
    __props__ = ('border_mode', 'subsample', 'logical_img_hw',
                 'logical_kern_hw', 'logical_kern_align_top', 'version',
                 'verbose', 'kshp', 'imshp', 'max_threads_dim0')

    @staticmethod
    def logical_output_shape_2d(imshp, kshp, mode):
        if mode == 'valid':
            return imshp[0] - kshp[0] + 1, imshp[1] - kshp[1] + 1
        if mode == 'full':
            return imshp[0] + kshp[0] - 1, imshp[1] + kshp[1] - 1
        raise ValueError(mode)

    def __init__(self, border_mode, subsample=(1, 1),
                 logical_img_hw=None, logical_kern_hw=None,
                 logical_kern_align_top=True,
                 version=-1, direction_hint=None,
                 verbose=0, kshp=None, imshp=None,
                 max_threads_dim0=None,
                 nkern=None, bsize=None, fft_opt=True):
        self.border_mode = border_mode
        self.subsample = subsample
        if logical_img_hw is not None:
            h, w = logical_img_hw
            # TODO: reconsider this... since shapes are not given in
            # constructor, maybe a multiplier + offset is a more
            # appropriate way of passing this logical grid
            logical_img_hw = tuple(logical_img_hw)
        self.logical_img_hw = logical_img_hw
        if logical_kern_hw is not None:
            h, w = logical_kern_hw
            # TODO: reconsider this... since shapes are not given in
            # constructor, maybe a multiplier + offset is a more
            # appropriate way of passing this logical grid
            logical_kern_hw = tuple(logical_kern_hw)
        self.logical_kern_hw = logical_kern_hw
        self.logical_kern_align_top = logical_kern_align_top
        self.version = version
        self.direction_hint = direction_hint
        self.verbose = verbose
        self.kshp = kshp
        self.imshp = imshp
        self.max_threads_dim0 = max_threads_dim0
        self.nkern = nkern
        self.bsize = bsize
        self.fft_opt = fft_opt

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "imshp"):
            self.imshp = None
        if not hasattr(self, "max_threads_dim0"):
            self.max_threads_dim0 = None
        if not hasattr(self, "direction_hint"):
            self.direction_hint = None
        if not hasattr(self, "nkern"):
            self.nkern = None
        if not hasattr(self, "bsize"):
            self.bsize = None
        if not hasattr(self, "fft_opt"):
            self.fft_opt = True

    def make_node(self, img, kern):
        if img.dtype != "float32" or kern.dtype != "float32":
            raise NotImplementedError("GpuConv currently only work"
                                      " with float32 dtype")
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        ctx_name = infer_context_name(img, kern)
        img = as_gpuarray_variable(img, ctx_name)
        kern = as_gpuarray_variable(kern, ctx_name)
        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        out = GpuArrayType(img.dtype, broadcastable, context_name=ctx_name)()
        return gof.Apply(self, [img, kern], [out])

    def get_params(self, node):
        return node.inputs[0].type.context

    def flops(self, inputs, outputs):
        """
        Useful with the hack in profilemode to print the MFlops.

        """
        images, kerns = inputs
        out, = outputs
        assert images[1] == kerns[1]
        flops = 0
        if self.border_mode == "valid":
            # nb mul and add by output pixel
            flops = kerns[2] * kerns[3] * 2
            # nb flops by output image
            flops *= out[2] * out[3]
            # nb patch multiplied
            flops *= images[1] * kerns[0] * images[0]
        else:
            flops = (images[0] * kerns[0] * images[1] *
                     kerns[2] * kerns[3] *
                     images[2] * images[3] * 2)
        return flops

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_ = copy.copy(node)
        assert node.op is node_.op
        if node_.op.max_threads_dim0 is None:
            node_.op.max_threads_dim0 = node_.inputs[0].type.context.maxlsize
        return super(GpuConv, node_.op).make_thunk(node_, storage_map,
                                                   compute_map, no_recycling)

    def c_compile_args(self):
        nb = 0
        if self.kshp is not None:
            nb = self.kshp[1]
        return ['-DTHEANO_KERN_WID=' + str(nb)]

    def c_headers(self):
        return ['<stdio.h>', '<numpy_compat.h>', '<gpuarray/types.h>']

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 23)

    def c_code(self, node, nodename, inp, out_, sub):
        if node.inputs[0].type.context.kind != "cuda":
            raise NotImplementedError("GpuConv only works for cuda devices")
        img, kern = inp
        out, = out_
        dx = self.subsample[0]
        dy = self.subsample[1]
        border_mode = self.border_mode
        version = self.version
        verbose = self.verbose
        sub = sub.copy()
        max_threads_dim0 = self.max_threads_dim0
        if max_threads_dim0 is None:
            raise NotImplementedError("GpuConv.c_code should not be called "
                                      "directly. It should be called by "
                                      "make_thunk() that add some information "
                                      "related to the selected GPU.")
        sub.update(locals())
        return """
    //Mandatory args
    const char *mode_str = "%(border_mode)s";

    //Optional args
    int version = %(version)s;
    int verbose = %(verbose)s;
    size_t dx = %(dx)s;
    size_t dy = %(dy)s;

    int mode;
    if (strcmp(mode_str, "full") == 0)
    {
        mode = ConvMode_FULL;
    }
    else if (strcmp(mode_str, "valid") == 0)
    {
        mode = ConvMode_VALID;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError,
                        "mode must be one of 'full' or 'valid'");
        return 0;
    }

    // TODO, make out be decref before we alloc out2!
    PyGpuArrayObject * out2 = (PyGpuArrayObject *)PyGpuArray_Conv(
                                                         %(img)s, %(kern)s,
                                                         %(out)s, mode,
                                                         dx, dy,
                                                         version, verbose,
                                                         %(max_threads_dim0)s);
    Py_XDECREF(%(out)s);
    %(out)s = out2;

    if (%(out)s==NULL){
        %(fail)s
    }
""" % sub

    def c_support_code_apply(self, node, name):
        nb = 0
        if self.kshp is not None:
            nb = self.kshp[1]
        kernels = self.gpu_kernels(node, name)
        k = kernels[0]
        code = """
        #define THEANO_KERN_WID %(nb)d
        """ % locals()
        code += "\n".join([open(os.path.join(os.path.split(__file__)[0], f)).read()
                           for f in ["conv_kernel.cu", "conv_full_kernel.cu"]])
        gk = gpuarray.GpuKernel(code, k.name, k.params, **k.flags)
        bin = gk._binary
        bcode = ','.join(hex(ord(c)) for c in bin)
        code = code.replace('\\', '\\\\')
        code = code.replace('"', '\\"')
        code = code.replace('\n', '\\n')
        mod = """
        static const char conv_bcode[] = {%(bcode)s};
        static const char *conv_code = "%(code)s";
        """ % locals()
        return mod

    def c_support_code_struct(self, node, name):
        mod = GpuKernelBase.c_support_code_struct(self, node, name)
        with open(os.path.join(os.path.split(__file__)[0], "conv.cu")) as f:
            mod += f.read()
        return mod

    @utils.memoize
    def gpu_kernels(self, node, name):
        dtypes = [i.dtype for i in node.inputs]
        dtypes.extend([o.dtype for o in node.outputs])
        flags = Kernel.get_flags(*dtypes)
        kernels = self.conv_patch_kernels(name, flags)
        kernels.extend(self.conv_patch_stack_kernels(name, flags))
        kernels.extend(self.conv_patch_stack_reduce_kernels(name, flags))
        kernels.extend(self.conv_rows_kernels(name, flags))
        kernels.extend(self.conv_rows_stack_kernels(name, flags))
        kernels.extend(self.conv_rows_stack2_kernels(name, flags))
        kernels.extend(self.conv_valid_row_reduce_kernels(name, flags))
        kernels.extend(self.conv_reference_valid_kernels(name, flags))
        kernels.extend(self.conv_reference_full_kernels(name, flags))
        kernels.extend(self.conv_full_patch_kernels(name, flags))
        kernels.extend(self.conv_full_patch_stack_kernels(name, flags))
        kernels.extend(self.conv_full_patch_stack_padded_kernels(name, flags))
        kernels.extend(self.conv_full_load_everything_kernels(name, flags))
        return kernels

    def conv_patch_kernels(self, name, flags):
        kname = "conv_patch_%d"
        k_var = "conv_patch_%d_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname % i, flags,
                   'conv_code', 'conv_bcode', k_var % i)
            for i in [2, 3]
            ]

    def conv_patch_stack_kernels(self, name, flags):
        kname = "conv_patch_stack_%d"
        k_var = "conv_patch_stack_%d_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname % i, flags,
                   'conv_code', 'conv_bcode', k_var % i)
            for i in range(64, 96)
            ]

    def conv_patch_stack_reduce_kernels(self, name, flags):
        kname = "conv_patch_stack_reduce_%d"
        k_var = "conv_patch_stack_reduce_%d_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname % i, flags,
                   'conv_code', 'conv_bcode', k_var % i)
            for i in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
            ]

    def conv_rows_kernels(self, name, flags):
        kname = "conv_rows_%d"
        k_var = "conv_rows_%d_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname % i, flags,
                   'conv_code', 'conv_bcode', k_var % i)
            for i in [0, 1]
            ]

    def conv_rows_stack_kernels(self, name, flags):
        kname = "conv_rows_stack_%d"
        k_var = "conv_rows_stack_%d_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname % i, flags,
                   'conv_code', 'conv_bcode', k_var % i)
            for i in [0, 1]
            ]

    def conv_rows_stack2_kernels(self, name, flags):
        kname = "conv_rows_stack2_%d"
        k_var = "conv_rows_stack2_%d_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname % i, flags,
                   'conv_code', 'conv_bcode', k_var % i)
            for i in [0, 1, 2, 3]
            ]

    def conv_valid_row_reduce_kernels(self, name, flags):
        kname = "conv_valid_row_reduce_%d"
        k_var = "conv_valid_row_reduce_%d_" + name
        params = [
            'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname % i, flags,
                   'conv_code', 'conv_bcode', k_var % i)
            for i in [0, 1]
            ]

    def conv_reference_valid_kernels(self, name, flags):
        kname = "conv_reference_valid"
        k_var = "conv_reference_valid_" + name
        params = [
            'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc',
            'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname, flags,
                   'conv_code', 'conv_bcode', k_var)
            ]

    def conv_reference_full_kernels(self, name, flags):
        kname = "conv_reference_full"
        k_var = "conv_reference_full_" + name
        params = [
            'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc',
            'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname, flags,
                   'conv_code', 'conv_bcode', k_var)
            ]

    def conv_full_patch_kernels(self, name, flags):
        kname = "conv_full_patch"
        k_var = "conv_full_patch_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname, flags,
                   'conv_code', 'conv_bcode', k_var)
            ]

    def conv_full_patch_stack_kernels(self, name, flags):
        kname = "conv_full_patch_stack_%d"
        k_var = "conv_full_patch_stack_%d_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname % i, flags,
                   'conv_code', 'conv_bcode', k_var % i)
            for i in [0, 1, 2, 3]
            ]

    def conv_full_patch_stack_padded_kernels(self, name, flags):
        kname = "conv_full_patch_stack_padded_%d"
        k_var = "conv_full_patch_stack_padded_%d_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname % i, flags,
                   'conv_code', 'conv_bcode', k_var % i)
            for i in [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
            ]

    def conv_full_load_everything_kernels(self, name, flags):
        kname = "conv_full_load_everything"
        k_var = "conv_full_load_everything_" + name
        params = [
            gpuarray.GpuArray, 'uintp', gpuarray.GpuArray, 'uintp',
            gpuarray.GpuArray, 'uintp',
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc'
            ]
        return [
            Kernel(None, params, kname, flags,
                   'conv_code', 'conv_bcode', k_var)
            ]
