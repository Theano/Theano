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
from type import (GpuArrayType, GpuArrayVariable, GpuArrayConstant,
                  GpuArraySharedVariable, gpuarray_shared_constructor)
import opt


def init_dev(dev):
    global pygpu_activated
    context = pygpu.init(dev)
    pygpu.set_default_context(context)
    pygpu_activated = True
    if config.print_active_device:
        print >> sys.stderr, "Using device %s: %s" % (dev, context.devname)
    # remember the active device
    init_dev.device = dev

init_dev.device = None


def use(device,
        force=False,
        default_to_move_computation_to_gpu=True,
        move_shared_to_gpu=True, # cuda name: move_shared_float32_to_gpu
#        enable_cuda=True,
#        test_driver=True
):
    init_dev(device)
    if default_to_move_computation_to_gpu:
        optdb.add_tags('gpuarray_opt', 'fast_run', 'fast_compile', 'inplace')
    if move_shared_to_gpu:
        theano.compile.shared_constructor(gpuarray_shared_constructor)

    if force:
        try:
            import pdb;pdb.set_trace()
            cuda_ndarray.cuda_ndarray.CudaNdarray.zeros((5, 5))
        except (Exception, NameError), e:
            # NameError when no gpu present as cuda_ndarray is not loaded.
            e.args += ("ERROR: GPU forced but failed. ",)
            raise


if pygpu:
    try:
        if (config.device.startswith('cuda') or
            config.device.startswith('opencl')):
            use(device=config.device, force=config.force_device)
#            init_dev(config.device)
#            theano.compile.shared_constructor(gpuarray_shared_constructor)
#            optdb.add_tags('gpuarray_opt', 'fast_run', 'fast_compile', 'inplace')
        elif config.gpuarray.init_device != '':
            assert config.device == "cpu", (
                "We can use the Theano flag init_gpu_device"
                " only when the Theano flag device=='cpu'")
            _logger.warning(
        ("GPU device %s will be initialized, and used if a GPU is "
         "needed. "
         "However, no computation, nor shared variables, will be implicitly "
         "moved to that device. If you want that behavior, use the 'device' "
         "flag instead."),
          config.init_gpu_device)
            use(device=config.init_gpu_device,
                force=config.force_device,
                default_to_move_computation_to_gpu=False,
                move_shared_to_gpu=False,
                #enable_cuda=False
            )

            init_dev(config.gpuarray.init_device)
    except Exception:
        error("Could not initialize pygpu, support disabled", exc_info=True)
else:
    if (config.gpuarray.init_device != '' or
        config.device.startswith('opencl') or
        config.device.startswith('cuda')):
        error("pygpu was configured but could not be imported", exc_info=True)
