import os, sys, stat
from theano.gof.compiledir import get_compiledir
from theano.compile import optdb
import theano.config as config

import logging, copy
_logger_name = 'theano.sandbox.cuda'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())
def warning(*msg):
    _logger.warning(_logger_name+'WARNING: '+' '.join(str(m) for m in msg))
def info(*msg):
    _logger.info(_logger_name+'INFO: '+' '.join(str(m) for m in msg))
def debug(*msg):
    _logger.debug(_logger_name+'DEBUG: '+' '.join(str(m) for m in msg))


# Compile cuda_ndarray.cu
# This need that nvcc (part of cuda) is installed. If it is not, a warning is
# printed and this module will not be working properly (we set `enable_cuda`
# to False).

# This variable is True by default, and set to False if something goes wrong
# when trying to initialize cuda.
enable_cuda = True

# Global variable to avoid displaying the same warning multiple times.
cuda_warning_is_displayed = False

# Code factorized within a function so that it may be called from multiple
# places (which is not currently the case, but may be useful in the future).
def set_cuda_disabled():
    """Function used to disable cuda.

    A warning is displayed, so that the user is aware that cuda-based code is
    not going to work.
    Note that there is no point calling this function from outside of
    `cuda.__init__`, since it has no effect once the module is loaded.
    """
    global enable_cuda, cuda_warning_is_displayed
    enable_cuda = False
    if not cuda_warning_is_displayed:
        cuda_warning_is_displayed = True
        warning('Cuda is disabled, cuda-based code will thus not be '
                'working properly')

#cuda_ndarray compile and import
sys.path.append(get_compiledir())
cuda_path = os.path.split(__file__)[0]
date = os.stat(os.path.join(cuda_path,'cuda_ndarray.cu'))[stat.ST_MTIME]
date = max(date,os.stat(os.path.join(cuda_path,'cuda_ndarray.cuh'))[stat.ST_MTIME])
date = max(date,os.stat(os.path.join(cuda_path,'conv_full_kernel.cu'))[stat.ST_MTIME])
date = max(date,os.stat(os.path.join(cuda_path,'conv_kernel.cu'))[stat.ST_MTIME])

cuda_ndarray_loc = os.path.join(get_compiledir(),'cuda_ndarray')
cuda_ndarray_so = os.path.join(cuda_ndarray_loc,'cuda_ndarray.so')
compile_cuda_ndarray = True

if os.path.exists(cuda_ndarray_so):
    compile_cuda_ndarray = date>=os.stat(cuda_ndarray_so)[stat.ST_MTIME]
if not compile_cuda_ndarray:
    try:
        from cuda_ndarray.cuda_ndarray import *
    except ImportError:
        compile_cuda_ndarray = True

if compile_cuda_ndarray:
    import nvcc_compiler
    if not nvcc_compiler.is_nvcc_available():
        set_cuda_disabled()

    if enable_cuda:
        code = open(os.path.join(cuda_path, "cuda_ndarray.cu")).read()

        if not os.path.exists(cuda_ndarray_loc):
            os.makedirs(cuda_ndarray_loc)
 
        nvcc_compiler.nvcc_module_compile_str('cuda_ndarray', code, location = cuda_ndarray_loc,
                                              include_dirs=[cuda_path], libs=['cublas'],
                                              preargs=['-DDONT_UNROLL', '-O3'])

        from cuda_ndarray.cuda_ndarray import *

if enable_cuda:
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


def use(device=config.THEANO_GPU):
    if use.device_number is None:
        # No successful call to use() has been made yet
        if device=="-1" or device=="CPU":
            return
        if device in [None,""]:
            device=0
        device=int(device)
        try:
            gpu_init(device)
            handle_shared_float32(True)
            use.device_number = device
        except RuntimeError, e:
            logging.getLogger('theano.sandbox.cuda').warning("WARNING: Won't use the GPU as the initialisation of device %i failed. %s" %(device, e))
    elif use.device_number != device:
        logging.getLogger('theano.sandbox.cuda').warning("WARNING: ignoring call to use(%s), GPU number %i is already in use." %(str(device), use.device_number))
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


if enable_cuda and config.THEANO_GPU not in [None, ""]:
    use()
