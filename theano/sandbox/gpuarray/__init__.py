from __future__ import print_function
import sys
import logging

import theano
from theano.configparser import config, AddConfigVar, BoolParam
from theano.compile import optdb

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

AddConfigVar('gpuarray.sync',
             """If True, every op will make sure its work is done before
                returning.  Setting this to True will slow down execution,
                but give much more accurate results in profiling.""",
             BoolParam(False),
             in_c_key=True)

# This is for documentation not to depend on the availability of pygpu
from .type import (GpuArrayType, GpuArrayVariable, GpuArrayConstant,
                  GpuArraySharedVariable, gpuarray_shared_constructor)
from . import opt, nerv


def init_dev(dev):
    if pygpu.gpuarray.api_version() != (-10000, 0):
        raise RuntimeError("Wrong API version for gpuarray:",
                           pygpu.gpuarray.api_version(),
                           "Make sure Theano and libgpuarray/pygpu "
                           "are in sync.")
    global pygpu_activated
    context = pygpu.init(dev)
    pygpu.set_default_context(context)
    pygpu_activated = True
    if config.print_active_device:
        print("Using device %s: %s" % (dev, context.devname), file=sys.stderr)
    # remember the active device
    init_dev.device = dev

init_dev.device = None

if pygpu:
    try:
        if (config.device.startswith('cuda') or
            config.device.startswith('opencl')):
            init_dev(config.device)
            import theano.compile
            theano.compile.shared_constructor(gpuarray_shared_constructor)
            optdb.add_tags('gpuarray_opt', 'fast_run', 'fast_compile', 'inplace')
        elif config.gpuarray.init_device != '':
            init_dev(config.gpuarray.init_device)

        from .basic_ops import (GpuAlloc, GpuContiguous, GpuEye, GpuFromHost,
                                GpuJoin, GpuReshape, GpuSplit, HostFromGpu)
        from .basic_ops import host_from_gpu, gpu_from_host
        from .elemwise import GpuElemwise
        from .subtensor import (GpuSubtensor, GpuIncSubtensor,
                                GpuAdvancedIncSubtensor1)

    except Exception:
        error("Could not initialize pygpu, support disabled", exc_info=True)
else:
    if (config.gpuarray.init_device != '' or
        config.device.startswith('opencl') or
        config.device.startswith('cuda')):
        error("pygpu was configured but could not be imported", exc_info=True)
