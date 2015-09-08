import copy
import os

import theano
from theano import config, gof

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from six.moves import reduce
from .comp import NVCC_compiler
from .type import GpuArrayType
from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel)
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

    @staticmethod
    def logical_output_shape_2d(imshp, kshp, mode):
        if mode == 'valid':
            return imshp[0] - kshp[0] + 1, imshp[1] - kshp[1] + 1
        if mode == 'full':
            return imshp[0] + kshp[0] - 1, imshp[1] + kshp[1] - 1
        raise ValueError(mode)

    def __init__(self, border_mode,
            subsample=(1, 1),
            logical_img_hw=None,
            logical_kern_hw=None,
            logical_kern_align_top=True,
            version=-1,
            direction_hint=None,
            verbose=0,
            kshp=None,
            imshp=None,
            max_threads_dim0=None,
            nkern=None,
            bsize=None,
            fft_opt=True):
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

    def __eq__(self, other):
        return type(self) == type(other) \
            and self.border_mode == other.border_mode \
            and self.subsample == other.subsample \
            and self.logical_img_hw == other.logical_img_hw \
            and self.logical_kern_hw == other.logical_kern_hw \
            and self.logical_kern_align_top == other.logical_kern_align_top \
            and self.version == other.version \
            and self.verbose == other.verbose \
            and self.kshp == other.kshp\
            and self.imshp == other.imshp\
            and self.max_threads_dim0 == other.max_threads_dim0

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

    def __hash__(self):
        # don't use hash(self.version) as hash(-1)==-2 and
        # hash(-2)==-2 in python!
        return hash(type(self)) \
            ^ hash(self.border_mode) \
            ^ hash(self.subsample) \
            ^ hash(self.logical_img_hw) \
            ^ hash(self.logical_kern_hw) \
            ^ hash(self.logical_kern_align_top) \
            ^ self.version \
            ^ hash(self.verbose) \
            ^ hash(self.kshp)\
            ^ hash(self.imshp)\
            ^ hash(self.max_threads_dim0)

    def __str__(self):
        return '%s{%s, %s, %s, %s, %s, %s, %s}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample),
            str(self.logical_img_hw),
            str(self.logical_kern_hw),
            str(self.logical_kern_align_top),
            str(self.imshp),
            str(self.kshp))

    def make_node(self, img, kern):
        if img.dtype != "float32" or kern.dtype != "float32":
            raise NotImplementedError("GpuConv currently only work"
                                      " with float32 dtype")
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        img = as_gpuarray_variable(img)
        kern = as_gpuarray_variable(kern)
        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        out = GpuArrayType(img.dtype, broadcastable)()
        return gof.Apply(self, [img, kern], [out])

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
        if config.gpuarray.sync:
            raise NotImplementedError("GpuConv do not implement gpuarray.sync Theano flag")
        if node_.op.max_threads_dim0 is None:
            cuda = theano.sandbox.cuda
            device_id = cuda.use.device_number
            if device_id is None:
                cuda.use("gpu",
                         force=False,
                         default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False,
                         test_driver=True)
                device_id = cuda.use.device_number
            cuda_ndarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
            prop = cuda_ndarray.device_properties(device_id)
            node_.op.max_threads_dim0 = prop['maxThreadsDim0']
        return super(GpuConv, node_.op).make_thunk(node_, storage_map,
                                                   compute_map, no_recycling)

    def c_compile_args(self):
        nb = 0
        if self.kshp is not None:
            nb = self.kshp[1]
        return ['-DTHEANO_KERN_WID=' + str(nb)]  # ,'-g','-G']

    def c_headers(self):
        if pygpu.get_default_context().kind == 'opencl':
            raise MethodNotDefined('cuda only')
        return ['<stdint.h>', '<stdio.h>', 'cuda.h',
                '<gpuarray/extension.h>', '<numpy_compat.h>',
                '<gpuarray/ext_cuda.h>', '<gpuarray/types.h>']

    def c_header_dirs(self):
        if pygpu.get_default_context().kind == 'opencl':
            raise MethodNotDefined('cuda only')
        cuda_root = config.cuda.root
        if cuda_root:
            return [os.path.join(cuda_root, 'include')]
        else:
            return []

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 21)

    def c_init_code(self):
        if pygpu.get_default_context().kind == 'opencl':
            raise MethodNotDefined('cuda only')
        return ['setup_ext_cuda();']

    def c_code(self, node, nodename, inp, out_, sub):
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
        kname = "conv_full_load_everything"
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
        for k in kernels:
            mod += "static GpuKernel " + k.name + '_' + name + ";\n"
        mod += open(os.path.join(os.path.split(__file__)[0], "conv.cu")).read()
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
