import os, sys
from theano.gof.compiledir import get_compiledir
from theano.compile import optdb

import logging, copy
_logger_name = 'theano_cuda_ndarray'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())
def warning(*msg):
    _logger.warning(_logger_name+'WARNING: '+' '.join(str(m) for m in msg))
def info(*msg):
    _logger.info(_logger_name+'INFO: '+' '.join(str(m) for m in msg))
def debug(*msg):
    _logger.debug(_logger_name+'DEBUG: '+' '.join(str(m) for m in msg))


#compile type_support.cu
#this need that nvcc(part of cuda) is installed

old_file = os.path.join(os.path.split(__file__)[0],'type_support.so')
if os.path.exists(old_file):
    os.remove(old_file)

try:
    sys.path.append(get_compiledir())
    from type_support.type_support import *

except ImportError:

    import nvcc_compiler

    print __file__

#    if os.path.exists(os.path.join(get_compiledir(),))

    #cuda_path = os.path.split(globals()["__file__"])[:-1]
    cuda_path='/u/bastienf/repos/theano/sandbox/cuda'
    code = open(os.path.join(cuda_path, "type_support.cu")).read()

    loc = os.path.join(get_compiledir(),'type_support')
    if not os.path.exists(loc):
        os.makedirs(loc)
 
    CUDA_NDARRAY=os.getenv('CUDA_NDARRAY')
    include_dirs=[]
    lib_dirs=[]
    
    if CUDA_NDARRAY:
        include_dirs.append(CUDA_NDARRAY)
        lib_dirs.append(CUDA_NDARRAY)
    else:
        import theano.sandbox
        path = os.path.split(os.path.split(os.path.split(theano.sandbox.__file__)[0])[0])[0]
        path2 = os.path.join(path,'cuda_ndarray')
        if os.path.isdir(path2):
            include_dirs.append(path2)
            lib_dirs.append(path2)
        else:
            path = os.path.split()[:-1]
            path2 = os.path.join(path,'cuda_ndarray')
            include_dirs.append(path2)
            lib_dirs.append(path2)

    nvcc_compiler.nvcc_module_compile_str('type_support', code, location = loc, include_dirs=include_dirs, lib_dirs=lib_dirs, libs=['cuda_ndarray'])

    from type_support.type_support import *



from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.var import (CudaNdarrayVariable,
    CudaNdarrayConstant,
    CudaNdarraySharedVariable,
    shared_constructor)

import basic_ops
from basic_ops import (GpuFromHost, HostFromGpu, GpuElemwise, 
                       GpuDimShuffle, GpuSum, GpuReshape, 
                       GpuSubtensor, GpuIncSubtensor, GpuFlatten, GpuShape)
import opt
import cuda_ndarray

import theano.config as config

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
    optdb.add_tags('gpu',
                   'fast_run',
                   'inplace')

use.device_number = None

def handle_shared_float32(tf):
    """Set the CudaNdarrayType as the default handler for shared float32 arrays.

    This function is intended to be called from use(gpu_index), not directly.
    """
    if tf:
        import theano.compile
        theano.compile.shared_constructor(shared_constructor)

    else:
        raise NotImplementedError('removing our handler')

