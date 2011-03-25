import os

import theano

def select_gpu_from_theano():
    # Transfer the theano gpu binding to pycuda, for consistency
    theano_to_pycuda_device_map = {"cpu": "0",
                                   "gpu0": "0",
                                   "gpu1": "1",
                                   "gpu2": "2",
                                   "gpu3": "3"}
    os.environ["CUDA_DEVICE"] = theano_to_pycuda_device_map.get(theano.config.device, "0")

select_gpu_from_theano()
pycuda_available = False
try:
    import pycuda
    import pycuda.autoinit
    pycuda_available = True
except ImportError:
    pass
