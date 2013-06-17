import logging

import theano
from theano.configparser import config, AddConfigVar, StrParam, \
    BoolParam, IntParam
from theano.compile import optdb

_logger_name = 'theano.sandbox.gpuarray'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.WARNING)

error = _logger.error
info = _logger.info

try:
    import pygpu
    import pygpu.gpuarray
except ImportError:
    pygpu = None


AddConfigVar('gpuarray.init_device',
             """
             Device to initialize for gpuarray use without moving
             computations automatically.
             """,
             StrParam(''))


# This is for documentation not to depend on the availability of pygpu
from type import (GpuArrayType, GpuArrayVariable, GpuArrayConstant,
                  GpuArraySharedVariable, gpuarray_shared_constructor)
import opt


def init_dev(dev):
    context = pygpu.init(dev)
    pygpu.set_default_context(context)

if pygpu:
    try:
        if (config.device.startswith('cuda') or
            config.device.startswith('opencl')):
            init_dev(config.device)
            # TODO add optimization tags here (when we will have some)
            import theano.compile
            theano.compile.shared_constructor(gpuarray_shared_constructor)
            optdb.add_tags('gpuarray_opt', 'fast_run', 'inplace')
            optdb.add_tags('gpuarray_after_fusion', 'fast_run', 'inplace')
        elif config.gpuarray.init_device != '':
            init_dev(config.gpuarray.init_device)
        else:
            info("pygpu support not configured, disabling")
            pygpu = None
    except Exception:
        error("Could not initialize pygpu, support disabled", exc_info=True)
        pygpu = None
else:
    if (config.gpuarray.init_device != '' or
        config.device.startswith('opencl') or
        config.device.startswith('cuda')):
        error("pygpu was configured but could not be imported", exc_info=True)
