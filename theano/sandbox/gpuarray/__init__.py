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

# This is for documentation not to depend on the availability of pygpu
from .type import (GpuArrayType, GpuArrayVariable, GpuArrayConstant,
                   GpuArraySharedVariable, gpuarray_shared_constructor,
                   reg_context)
from . import opt, nerv


def init_dev(dev, name=None):
    if pygpu.gpuarray.api_version() != (-10000, 0):
        raise RuntimeError("Wrong API version for gpuarray:",
                           pygpu.gpuarray.api_version(),
                           "Make sure Theano and libgpuarray/pygpu "
                           "are in sync.")
    global pygpu_activated
    if dev not in init_dev.devmap:
        init_dev.devmap[dev] = pygpu.init(dev)
    context = init_dev.devmap[dev]
    reg_context(name, context)
    pygpu_activated = True
    if config.print_active_device:
        print("Mapped name %s to device %s: %s" % (name, dev, context.devname),
              file=sys.stderr)

init_dev.devmap = {}

if pygpu:
    try:
        if (config.device.startswith('cuda') or
            config.device.startswith('opencl')):
            init_dev(config.device)
            import theano.compile
            theano.compile.shared_constructor(gpuarray_shared_constructor)
            optdb.add_tags('gpuarray_opt', 'fast_run', 'fast_compile')
        elif (config.init_gpu_device.startswith('cuda') or
              config.init_gpu_device.startswith('opencl')):
            init_dev(config.init_gpu_device)
        if config.contexts != '':
            for n, d in (c.split('->') for c in config.contexts.split(';')):
                init_dev(d, n)
            import theano.compile
            theano.compile.shared_constructor(gpuarray_shared_constructor)
            optdb.add_tags('gpuarray_opt', 'fast_run', 'fast_compile')

        from .basic_ops import (GpuAlloc, GpuContiguous, GpuEye, GpuFromHost,
                                GpuJoin, GpuReshape, GpuSplit, HostFromGpu)
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
