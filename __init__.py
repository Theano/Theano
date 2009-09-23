from .type import CudaNdarrayType

from .var import (CudaNdarrayVariable,
    CudaNdarrayConstant,
    CudaNdarraySharedVariable,
    shared_constructor)

import basic_ops
import opt

import theano.compile.sandbox

def handle_shared_float32(tf):
    """Set the CudaNdarrayType as the default handler for shared float32 arrays
    """
    if tf:
        theano.compile.sandbox.shared_constructor(shared_constructor)
    else:
        raise NotImplementedError('removing our handler')

