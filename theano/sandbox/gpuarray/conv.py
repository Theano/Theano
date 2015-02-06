import copy
import os

import theano
from theano import config, gof
from theano.sandbox.gpuarray.comp import NVCC_compiler
from theano.sandbox.gpuarray.type import GpuArrayType, gpu_context_type
from theano.sandbox.gpuarray.basic_ops import as_gpuarray_variable


class GpuConv(gof.Op):
    """
    Implement the batched and stacked 2d convolution on the gpu.
    """
    context_type = gpu_context_type

    @staticmethod
    def logical_output_shape_2d(imshp, kshp, mode):
        if mode == 'valid':
            return imshp[0] - kshp[0] + 1, imshp[1] - kshp[1] + 1
        if mode == 'full':
            return imshp[0] + kshp[0] - 1, imshp[1] + kshp[1] - 1
        raise ValueError(mode)

    __props__ = ('border_mode', 'subsample', 'logical_img_hw',
                 'logical_kern_hw', 'logical_kern_align_top',
                 'version', 'verbose', 'kshp', 'imshp', 'max_threads_dim0',
                 'context')

    def __init__(self, border_mode,
                 subsample=(1, 1),
                 logical_img_hw=None,
                 logical_kern_hw=None,
                 logical_kern_align_top=True,
                 version=-1,
                 verbose=0,
                 kshp=None,
                 imshp=None,
                 max_threads_dim0=None,
                 context=None):
        """
        :param verbose: for value of 1,2 and 3. Print more information during
                        the execution of the convolution. Mostly used for
                        optimization or debugging.
        :param kshp:    The size of the kernel. If provided, can generate
                        faster code. If the GpuConv op is automatically
                        inserted,
                        we take its value automatically from the Conv op.
        :param imshp:   The size of the image. Not used for code generation but
                        allows to select an experimental new version in another
                        repo.
        :param max_threads_dim0: The maximum number of threads for the
                        block size dimensions 0 (blockDim.x) used by the
                        GPU function.

        """
        if version != -1:
            raise Exception(
                """We don't test versions other than the auto-selected one
                anymore.  If you want to use a specific version comment this
                error.""")
        self.border_mode = border_mode
        self.subsample = subsample
        if logical_img_hw is not None:
            h, w = logical_img_hw
            #TODO: reconsider this... since shapes are not given in
            # constructor, maybe a multiplier + offset is a more
            # appropriate way of passing this logical grid
            logical_img_hw = tuple(logical_img_hw)
        self.logical_img_hw = logical_img_hw
        if logical_kern_hw is not None:
            h, w = logical_kern_hw
            #TODO: reconsider this... since shapes are not given in
            # constructor, maybe a multiplier + offset is a more
            # appropriate way of passing this logical grid
            logical_kern_hw = tuple(logical_kern_hw)
        self.logical_kern_hw = logical_kern_hw
        self.logical_kern_align_top = logical_kern_align_top
        self.version = version
        self.verbose = verbose
        self.kshp = kshp
        self.imshp = imshp
        self.max_threads_dim0 = max_threads_dim0
        self.context = context

    def get_context(self, node):
        return self.context

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "imshp"):
            self.imshp = None
        if not hasattr(self, "max_threads_dim0"):
            self.max_threads_dim0 = None

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
        img = as_gpuarray_variable(img, self.context)
        kern = as_gpuarray_variable(kern, self.context)
        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
        out = GpuArrayType(img.dtype, broadcastable,
                           context=self.context)()
        return gof.Apply(self, [img, kern], [out])

    def flops(self, inputs, outputs):
        """ Useful with the hack in profilemode to print the MFlops"""
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
            node_.op.max_threads_dim0 = node_.inputs[0].type.real_context.maxlsize
        return super(GpuConv, node_.op).make_thunk(node_, storage_map,
                                                   compute_map, no_recycling)

    def c_compile_args(self):
        nb = 0
        if self.kshp is not None:
            nb = self.kshp[1]
        return ['-DTHEANO_KERN_WID=' + str(nb)]  # ,'-g','-G']

    def c_headers(self):
        return ['<stdio.h>', 'cuda.h',
                '<gpuarray/ext_cuda.h>', '<numpy_compat.h>']

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (0, 22)

    def c_init_code(self):
        return ['setup_ext_cuda();']

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['conv_kernel.cu', 'conv_full_kernel.cu', 'conv.cu']
        codes = ["float* cuda_get_ptrf(PyGpuArrayObject * o){return (float*) (cuda_get_ptr(o->ga.data) + o->ga.offset);}",
                 "const float* cuda_get_ptrf(const PyGpuArrayObject * o){return (float*) (cuda_get_ptr(o->ga.data) + o->ga.offset);}"]
        codes += [open(os.path.join(os.path.split(__file__)[0], f)).read()
                  for f in files]
        return reduce(str.__add__, codes)

    def c_compiler(self):
        return NVCC_compiler(self.context)

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
    int dx = %(dx)s;
    int dy = %(dy)s;

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
        return NULL;
    }

    cuCtxPushCurrent(cuda_get_ctx(%(context)s->ctx));
    // TODO, make out be decref before we alloc out2!
    PyGpuArrayObject * out2 = (PyGpuArrayObject *)PyGpuArray_Conv(
                                                         %(img)s, %(kern)s,
                                                         %(out)s, mode,
                                                         dx, dy,
                                                         version, verbose,
                                                         %(max_threads_dim0)s);
    cuCtxPopCurrent(NULL);
    Py_XDECREF(%(out)s);
    %(out)s = out2;

    if (%(out)s==NULL){
        %(fail)s
    }
""" % sub
