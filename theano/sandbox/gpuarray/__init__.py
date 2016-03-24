from __future__ import absolute_import, print_function, division
import sys
import logging
import sys
import warnings

import theano
from theano import config
from theano.compile import optdb

from theano.tensor.basic import register_transfer

_logger_name = 'theano.sandbox.gpuarray'
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
from . import dnn, opt, nerv

def transfer(x, target):
    try:
        get_context(target)
        return as_gpuarray_variable(x, target)
    except ContextNotDefined:
        pass

register_transfer(transfer)


def init_dev(dev, name=None):
    v = pygpu.gpuarray.api_version()
    if v[0] != -10000:
        raise RuntimeError("Wrong major API version for gpuarray:", v[0],
                           "Make sure Theano and libgpuarray/pygpu "
                           "are in sync.")
    if v[1] < 0:
        raise RuntimeError("Wrong minor API version for gpuarray:", v[1],
                           "Please update libgpuarray/pygpu.")
    global pygpu_activated
    if dev not in init_dev.devmap:
        ctx = pygpu.init(dev)
        init_dev.devmap[dev] = ctx
        if config.gpuarray.preallocate != 0:
            if config.gpuarray.preallocate < 1:
                gmem = min(config.gpuarray.preallocate, 0.98) * ctx.total_gmem
            else:
                gmem = config.gpuarray.preallocate * (1024*1024)
            # This will allocate and immediatly free an object of size gmem
            # which will reserve that amount of memory on the GPU.
            pygpu.empty((gmem,), dtype='int8', context=ctx)
    context = init_dev.devmap[dev]
    # This will map the context name to the real context object.
    reg_context(name, context)
    pygpu_activated = True
    if config.print_active_device:
        warn = None
        cudnn_version = ""
        if dev.startswith('cuda'):
            cudnn_version = " (CuDNN not available)"
            try:
                cudnn_version = dnn.version()
                # 4100 should not print warning with cudnn 4 final.
                if cudnn_version > 4100:
                    warn = ("Your CuDNN version is more recent than Theano."
                            " If you see problems, try updating Theano or"
                            " downgrading CuDNN to version 4.")
                cudnn_version = " (CuDNN version %s)" % cudnn_version
            except Exception:
                pass
        print("Mapped name %s to device %s: %s%s" % (
            name, dev, context.devname, cudnn_version),
              file=sys.stderr)
        if warn:
            warnings.warn(warn)

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
