from __future__ import absolute_import, print_function, division
import sys
import logging
import sys
import warnings

import theano
from theano import config
from theano.compile import optdb

from theano.tensor.basic import register_transfer

_logger_name = 'theano.gpuarray'
_logger = logging.getLogger(_logger_name)

error = _logger.error
info = _logger.info

pygpu_activated = False
try:
    import pygpu
    import pygpu.gpuarray
    import pygpu.version
except ImportError:
    pygpu = None

# This is for documentation not to depend on the availability of pygpu
from .type import (GpuArrayType, GpuArrayVariable, GpuArrayConstant,
                   GpuArraySharedVariable, gpuarray_shared_constructor,
                   reg_context, get_context, ContextNotDefined)
from .basic_ops import as_gpuarray_variable
from . import fft, dnn, opt, nerv, extra_ops, multinomial, reduction

def transfer(x, target):
    try:
        get_context(target)
        return as_gpuarray_variable(x, target)
    except ContextNotDefined:
        pass

register_transfer(transfer)


def init_dev(dev, name=None):
    global pygpu_activated
    if not config.cxx:
        raise RuntimeError("The new gpu-backend need a c++ compiler.")
    if (pygpu.version.major, pygpu.version.minor, pygpu.version.patch) < (0, 6, 1):
        raise ValueError(
            "Your installed version of pygpu is too old, please upgrade to 0.6.1 or later")
    # This is for the C headers API, we need to match the exact version.
    if pygpu.gpuarray.api_version()[0] != 1:
        raise ValueError(
            "Your installed libgpuarray is not in sync, please make sure to have the appropriate version")
    if dev not in init_dev.devmap:
        context = pygpu.init(
            dev,
            disable_alloc_cache=config.gpuarray.preallocate < 0,
            single_stream=config.gpuarray.single_stream,
            sched=config.gpuarray.sched)
        context.dev = dev
        init_dev.devmap[dev] = context
        reg_context(name, context)

        if dev.startswith('cuda'):
            avail = dnn.dnn_available(name)
            if avail:
                context.cudnn_handle = dnn._make_handle(context)
            if config.print_active_device:
                if avail:
                    print("Using cuDNN version %d on context %s" % (dnn.version(), name),
                          file=sys.stderr)
                else:
                    print("Can not use cuDNN on context %s: %s" % (name, dnn.dnn_available.msg),
                          file=sys.stderr)
        if config.gpuarray.preallocate < 0:
            print("Disabling allocation cache on %s" % (dev,))
        elif config.gpuarray.preallocate > 0:
            MB = (1024 * 1024)
            if config.gpuarray.preallocate <= 1:
                gmem = min(config.gpuarray.preallocate, 0.95) * context.total_gmem
            else:
                gmem = config.gpuarray.preallocate * MB
            if gmem > context.free_gmem - 50 * MB:
                print(
                    "WARNING: Preallocating too much memory can prevent cudnn and cublas from working properly")

            # This will allocate and immediatly free an object of size gmem
            # which will reserve that amount of memory on the GPU.
            pygpu.empty((gmem,), dtype='int8', context=context)
            if config.print_active_device:
                print("Preallocating %d/%d Mb (%f) on %s" %
                      (gmem//MB, context.total_gmem//MB,
                       gmem/context.total_gmem, dev),
                      file=sys.stderr)

        # Initialise the blas kernels.  We do this after the
        # preallocation to not fragment the heap accidentally.
        tmp = pygpu.empty((2, 2), dtype='float32', context=context)
        pygpu.blas.gemm(0, tmp, tmp, 0, tmp, overwrite_c=True)
        del tmp
    else:
        context = init_dev.devmap[dev]
    # This will map the context name to the real context object.
    if config.print_active_device:
        try:
            pcibusid = '(' + context.pcibusid + ')'
        except pygpu.gpuarray.UnsupportedException:
            pcibusid = ''

        print("Mapped name %s to device %s: %s %s" %
              (name, dev, context.devname, pcibusid),
              file=sys.stderr)
    pygpu_activated = True

# This maps things like 'cuda0' to the context object on that device.
init_dev.devmap = {}


def use(device,
        force=False,
        default_to_move_computation_to_gpu=True,
        move_shared_to_gpu=True):
    """
    Error and warning about CUDA should be displayed only when this
    function is called. We need to be able to load this module only
    to check if it is available!

    Parameters
    ----------
    device : string
        "cuda", "cuda0", "cudaN", "" (N is the device number to use).
        "" mean do all the rest and don't init a device.
    force
        Will always raise an exception if we can't use the gpu.
    default_to_move_computation_to_gpu
        If gpu init succeeded, enable by default optimizations to move
        computations to the gpu.
    move_shared_to_gpu
        If gpu init succeeded, put new shared variables on the gpu.

    """
    if force:
        if not device.startswith('cuda'):
            raise Exception("forced the init and bad device provided: " +
                            device)
        else:
            # If we force, the device should not already be initialized.
            assert device not in init_dev.devmap
    if device:
        init_dev(device)
    if default_to_move_computation_to_gpu:
        optdb.add_tags('gpuarray_opt', 'fast_run', 'fast_compile')
        optdb.add_tags('gpua_scanOp_make_inplace', 'fast_run')
    if move_shared_to_gpu:
        import theano.compile
        theano.compile.shared_constructor(gpuarray_shared_constructor)


if pygpu:
    try:
        if (config.device.startswith('cuda') or
            config.device.startswith('opencl')):
            use(config.device)
        elif (config.init_gpu_device.startswith('cuda') or
              config.init_gpu_device.startswith('opencl')):
            if config.device != 'cpu':
                raise ValueError(
                    'you must set device=cpu to use init_gpu_device.')
            if config.contexts != '':
                print("Using contexts will make init_gpu_device act like device and move all computations by default, which might not be what you want.")
            init_dev(config.init_gpu_device)
        if config.contexts != '':
            for n, d in (c.split('->') for c in config.contexts.split(';')):
                init_dev(d.strip(), n.strip())
            # To have shared var default on the GPU and opt to move to the GPU.
            use("")

    except Exception:
        error("Could not initialize pygpu, support disabled", exc_info=True)

    from .basic_ops import (GpuAlloc, GpuAllocEmpty, GpuContiguous, GpuEye,
                            GpuFromHost, GpuJoin, GpuReshape, GpuSplit,
                            HostFromGpu)
    from .basic_ops import host_from_gpu, GpuFromHost
    from .elemwise import GpuElemwise
    from .subtensor import (GpuSubtensor, GpuIncSubtensor,
                            GpuAdvancedIncSubtensor1)

else:
    if (config.init_gpu_device.startswith('cuda') or
            config.init_gpu_device.startswith('opencl') or
            config.device.startswith('opencl') or
            config.device.startswith('cuda') or
            config.contexts != ''):
        error("pygpu was configured but could not be imported or is too old (version 0.6 or higher required)",
              exc_info=True)
