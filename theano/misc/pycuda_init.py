import os

import theano
import theano.sandbox.cuda as cuda

def select_gpu_from_theano():
    # Transfer the theano gpu binding to pycuda, for consistency
    theano_to_pycuda_device_map = {"cpu": "0",
                                   "gpu0": "0",
                                   "gpu1": "1",
                                   "gpu2": "2",
                                   "gpu3": "3"}
    dev = theano_to_pycuda_device_map.get(theano.config.device, "0")
    if theano.config.device == 'gpu':
        dev = str(cuda.cuda_ndarray.cuda_ndarray.active_device_number())
    os.environ["CUDA_DEVICE"] = dev

select_gpu_from_theano()
pycuda_available = False
try:
    import pycuda
    import pycuda.autoinit
    pycuda_available = True
except ImportError:
    # presumably, the user wanted to use pycuda, else they wouldn't have
    # imported this module, so issue a warning that the import failed.
    import warnings
    warnings.warn("PyCUDA import failed in theano.misc.pycuda_init")
