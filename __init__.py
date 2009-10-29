from .type import CudaNdarrayType

from .var import (CudaNdarrayVariable,
    CudaNdarrayConstant,
    CudaNdarraySharedVariable,
    shared_constructor)

import basic_ops
import opt
import cuda_ndarray

import theano.compile.sandbox

import logging, os

def use(device=None):
    if use.device_number is None:
        # No successful call to use() has been made yet
        if device is None:
            device = os.getenv("THEANO_GPU",0)
        import pdb
        pdb.set_trace()
        if device=="-1" or device=="CPU":
            return
        device=int(device)
        try:
            cuda_ndarray.gpu_init(device)
            handle_shared_float32(True)
            use.device_number = device
        except RuntimeError, e:
            logging.getLogger('theano_cuda_ndarray').warning("WARNING: Won't use the GPU as the initialisation of device %i failed. %s" %(device, e))
            raise
    elif use.device_number != device:
        logging.getLogger('theano_cuda_ndarray').warning("WARNING: ignoring call to use(%s), GPU number %i is already in use." %(str(device), use.device_number))

use.device_number = None

def handle_shared_float32(tf):
    """Set the CudaNdarrayType as the default handler for shared float32 arrays.

    This function is intended to be called from use(gpu_index), not directly.
    """
    if tf:
        theano.compile.sandbox.shared_constructor(shared_constructor)

    else:
        raise NotImplementedError('removing our handler')

