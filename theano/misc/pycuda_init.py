import os
import warnings

import theano
import theano.sandbox.cuda as cuda
cuda_ndarray = cuda.cuda_ndarray.cuda_ndarray

def set_gpu_from_theano():
    """
    This set the GPU used by PyCUDA to the same as the one used by Theano.
    """
    #import pdb;pdb.set_trace()
    if cuda.use.device_number is None:
        cuda.use("gpu",
                 force=False,
                 default_to_move_computation_to_gpu=False,
                 move_shared_float32_to_gpu=False,
                 enable_cuda=True,
                 test_driver=True)

    assert cuda.use.device_number == cuda_ndarray.active_device_number()

#    os.environ["CUDA_DEVICE"] = str(cuda.use.device_number)

set_gpu_from_theano()
pycuda_available = False
if True:  # theano.sandbox.cuda.use.device_number is None:
    try:
        import pycuda
        import pycuda.autoinit
        pycuda_available = True
    except ImportError:
        # presumably, the user wanted to use pycuda, else they wouldn't have
        # imported this module, so issue a warning that the import failed.
        warnings.warn("PyCUDA import failed in theano.misc.pycuda_init")
else:
    warnings.warn("theano.misc.pycuda_init must be imported before theano"
                  " init its GPU")
