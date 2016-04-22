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
except ImportError:
    pygpu = None

# This is for documentation not to depend on the availability of pygpu
from .type import (GpuArrayType, GpuArrayVariable, GpuArrayConstant,
                   GpuArraySharedVariable, gpuarray_shared_constructor,
                   reg_context, get_context, ContextNotDefined)
from .basic_ops import as_gpuarray_variable
from . import dnn, opt, nerv, extra_ops

def transfer(x, target):
    try:
        get_context(target)
        return as_gpuarray_variable(x, target)
    except ContextNotDefined:
        pass

register_transfer(transfer)


def init_dev(dev, name=None):
    v = pygpu.gpuarray.api_version()
    expected = -9998
    if v[0] != expected:
        raise RuntimeError("Wrong major API version for gpuarray:", v[0],
                           "Make sure Theano and libgpuarray/pygpu "
                           "are in sync. Expected", expected)
    if v[1] < 0:
        raise RuntimeError("Wrong minor API version for gpuarray:", v[1],
                           "Please update libgpuarray/pygpu.")
    global pygpu_activated
    if dev not in init_dev.devmap:
        ctx = pygpu.init(dev,
                         disable_alloc_cache=config.gpuarray.preallocate < 0)
        init_dev.devmap[dev] = ctx
        if config.gpuarray.preallocate > 0:
            MB = (1024 * 1024)
            if config.gpuarray.preallocate <= 1:
                gmem = min(config.gpuarray.preallocate, 0.95) * ctx.total_gmem
            else:
                gmem = config.gpuarray.preallocate * MB
            # This will allocate and immediatly free an object of size gmem
            # which will reserve that amount of memory on the GPU.
            pygpu.empty((gmem,), dtype='int8', context=ctx)
            if config.print_active_device:
                print("Preallocating %d/%d Mb (%f) on %s" %
                      (gmem//MB, ctx.total_gmem//MB, gmem/ctx.total_gmem, dev),
                      file=sys.stderr)
    context = init_dev.devmap[dev]
    # This will map the context name to the real context object.
    reg_context(name, context)
    if config.print_active_device:
        print("Mapped name %s to device %s: %s" %
              (name, dev, context.devname),
              file=sys.stderr)
    pygpu_activated = True
    if dev.startswith('cuda'):
        try:
            cudnn_version = dnn.version()
            # 5100 should not print warning with cudnn 5 final.
            if cudnn_version > 5100:
                warnings.warn("Your cuDNN version is more recent than Theano."
                              " If you see problems, try updating Theano or"
                              " downgrading cuDNN to version 5.")
            if config.print_active_device:
                print("Using cuDNN version %d on context %s" %
                      (cudnn_version, name), file=sys.stderr)
        except Exception:
            pass

# This maps things like 'cuda0' to the context object on that device.
init_dev.devmap = {}

if pygpu:
    try:
        if (config.device.startswith('cuda') or
            config.device.startswith('opencl')):
            init_dev(config.device)
            import theano.compile
            theano.compile.shared_constructor(gpuarray_shared_constructor)
            optdb.add_tags('gpuarray_opt', 'fast_run', 'fast_compile')
            optdb.add_tags('gpua_scanOp_make_inplace', 'fast_run')
        elif (config.init_gpu_device.startswith('cuda') or
              config.init_gpu_device.startswith('opencl')):
            if config.device != 'cpu':
                raise ValueError('you must set device=cpu to use init_gpu_device.')
            if config.contexts != '':
                print("Using contexts will make init_gpu_device act like device and move all computations by default, which might not be what you want.")
            init_dev(config.init_gpu_device)
        if config.contexts != '':
            for n, d in (c.split('->') for c in config.contexts.split(';')):
                init_dev(d.strip(), n.strip())
            import theano.compile
            theano.compile.shared_constructor(gpuarray_shared_constructor)
            optdb.add_tags('gpuarray_opt', 'fast_run', 'fast_compile')
            optdb.add_tags('gpua_scanOp_make_inplace', 'fast_run')

        from .basic_ops import (GpuAlloc, GpuAllocEmpty, GpuContiguous, GpuEye,
                                GpuFromHost, GpuJoin, GpuReshape, GpuSplit,
                                HostFromGpu)
        from .basic_ops import host_from_gpu, GpuFromHost
        from .elemwise import GpuElemwise
        from .subtensor import (GpuSubtensor, GpuIncSubtensor,
                                GpuAdvancedIncSubtensor1)

    except Exception:
        error("Could not initialize pygpu, support disabled", exc_info=True)
else:
    if (config.init_gpu_device.startswith('cuda') or
            config.init_gpu_device.startswith('opencl') or
            config.device.startswith('opencl') or
            config.device.startswith('cuda') or
            config.contexts != ''):
        error("pygpu was configured but could not be imported", exc_info=True)
