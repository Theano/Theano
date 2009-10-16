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

def use():
    handle_shared_float32(True)

def handle_shared_float32(tf):
    """Set the CudaNdarrayType as the default handler for shared float32 arrays
    
    Use use(tf) instead as this is a bad name.
    """
    if tf:
        try:
            v=os.getenv("THEANO_GPU",0)
            cuda_ndarray.gpu_init(int(v))
            theano.compile.sandbox.shared_constructor(shared_constructor)
        except RuntimeError, e:
            logging.getLogger('theano_cuda_ndarray').warning("WARNING: Won't use the GPU as the initialisation failed."+str(e))

    else:
        raise NotImplementedError('removing our handler')

