from __future__ import absolute_import, print_function, division
import os
import warnings

import theano
import theano.sandbox.cuda
from theano import config


def set_gpu_from_theano():
    """
    This set the GPU used by PyCUDA to the same as the one used by Theano.
    """
    # Transfer the theano gpu binding to pycuda, for consistency
    if config.device.startswith("gpu") and len(config.device) > 3:
        os.environ["CUDA_DEVICE"] = theano.config.device[3:]
    elif (config.init_gpu_device.startswith("gpu") and
          len(config.init_gpu_device) > 3):
        os.environ["CUDA_DEVICE"] = theano.config.init_gpu_device[3:]


set_gpu_from_theano()
pycuda_available = False
# If theano.sandbox.cuda don't exist, it is because we are importing
# it and it try to import this file! This mean we must init the device.
if (not hasattr(theano.sandbox, 'cuda') or
        theano.sandbox.cuda.use.device_number is None):
    try:
        import pycuda
        import pycuda.autoinit
        pycuda_available = True
    except (ImportError, RuntimeError):
        # presumably, the user wanted to use pycuda, else they wouldn't have
        # imported this module, so issue a warning that the import failed.
        warnings.warn("PyCUDA import failed in theano.misc.pycuda_init")
    except pycuda._driver.LogicError:
        if theano.config.force_device:
            raise
        else:
            if "CUDA_DEVICE" in os.environ:
                del os.environ["CUDA_DEVICE"]
            import pycuda.autoinit
            pycuda_available = True
else:
    try:
        import pycuda.driver
        pycuda_available = True
    except ImportError:
        pass
    if pycuda_available:
        if hasattr(pycuda.driver.Context, "attach"):
            pycuda.driver.Context.attach()
            import atexit
            atexit.register(pycuda.driver.Context.pop)
        else:
            # Now we always import this file when we call
            # theano.sandbox.cuda.use. So this should not happen
            # normally.
            # TODO: make this an error.
            warnings.warn("For some unknow reason, theano.misc.pycuda_init was"
                          " not imported before Theano initialized the GPU and"
                          " your PyCUDA version is 2011.2.2 or earlier."
                          " To fix the problem, import theano.misc.pycuda_init"
                          " manually before using/initializing the GPU, use the"
                          " Theano flag pycuda.init=True or use a"
                          " more recent version of PyCUDA.")
