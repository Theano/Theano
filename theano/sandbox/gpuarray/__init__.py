import logging

import theano
from theano.configparser import config, AddConfigVar, StrParam, \
    BoolParam, IntParam

_logger_name = 'theano.sandbox.gpuarray'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.WARNING)

error = _logger.error

try:
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
from type import GpuArrayType
from var import (GpuArrayVariable, GpuArrayConstant, GpuArraySharedVariable,
                 gpuarray_shared_constructor)


def init_dev(dev):
    import globals
    if dev.startswith('cuda'):
        globals.kind = 'cuda'
        devnum = int(dev[4:])
    elif dev.startswith('opencl'):
        globals.kind = 'opencl'
        devspec = dev[7:]
        plat, dev = devspec.split(':')
        devnum = int(dev)|(int(plat)>>16)
    else:
        globals.kind = None
    if globals.kind:
        globals.context = pygpu.gpuarray.init(globals.kind, devnum)

if pygpu:
    try:
        if (config.device.startswith('cuda') or
            config.device.startswith('opencl')):
            init_dev(config.device)
            # XXX add optimization tags here (when we will have some)
            import theano.compile
            theano.compile.shared_constructor(gpuarray_shared_constructor)
        elif config.gpuarray.init_device != '':
            init_dev(config.gpuarray.init_device)
    except Exception:
        error("Could not initialize pygpu, support disabled", exc_info=True)
        pygpu = None
