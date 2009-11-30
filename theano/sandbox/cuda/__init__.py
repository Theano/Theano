from theano.sandbox.cuda.type import CudaNdarrayType

from theano.sandbox.cuda.var import (CudaNdarrayVariable,
    CudaNdarrayConstant,
    CudaNdarraySharedVariable,
    shared_constructor)

import basic_ops
import opt
import cuda_ndarray

import theano.compile.sandbox

import os
import theano.config as config

import logging, copy
_logger_name = 'theano_cuda_ndarray'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.WARN)
_logger.addHandler(logging.StreamHandler())
def warning(*msg):
    _logger.warning(_logger_name+'WARNING: '+' '.join(str(m) for m in msg))
def info(*msg):
    _logger.info(_logger_name+'INFO: '+' '.join(str(m) for m in msg))
def debug(*msg):
    _logger.debug(_logger_name+'DEBUG: '+' '.join(str(m) for m in msg))


def use(device=config.THEANO_GPU):
    if use.device_number is None:
        # No successful call to use() has been made yet
        if device=="-1" or device=="CPU":
            return
        device=int(device)
        try:
            cuda_ndarray.gpu_init(device)
            handle_shared_float32(True)
            use.device_number = device
        except RuntimeError, e:
            logging.getLogger('theano_cuda_ndarray').warning("WARNING: Won't use the GPU as the initialisation of device %i failed. %s" %(device, e))
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

