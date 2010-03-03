import os, sys, stat
from theano.compile import optdb
from theano import config

import logging, copy
_logger_name = 'theano.sandbox.cuda'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.WARNING)
def error(*msg):
    _logger.error('ERROR (%s): %s'% ( _logger_name, ' '.join(str(m) for m in msg)))
def warning(*msg):
    _logger.warning('WARNING (%s): %s'% ( _logger_name, ' '.join(str(m) for m in msg)))
def info(*msg):
    _logger.info('INFO (%s): %s'% ( _logger_name, ' '.join(str(m) for m in msg)))
def debug(*msg):
    _logger.debug('DEBUG (%s): %s'% ( _logger_name, ' '.join(str(m) for m in msg)))


# Compile cuda_ndarray.cu
# This need that nvcc (part of cuda) is installed. If it is not, a warning is
# printed and this module will not be working properly (we set `cuda_available`
# to False).

# This variable is True by default, and set to False if something goes wrong
# when trying to initialize cuda.
cuda_available = True

# Global variable to avoid displaying the same warning multiple times.
cuda_warning_is_displayed = False

#This variable is set to True when we enable the cuda.(i.e. when use() is called)
cuda_enabled = False

# Code factorized within a function so that it may be called from multiple
# places (which is not currently the case, but may be useful in the future).
def set_cuda_disabled():
    """Function used to disable cuda.

    A warning is displayed, so that the user is aware that cuda-based code is
    not going to work.
    Note that there is no point calling this function from outside of
    `cuda.__init__`, since it has no effect once the module is loaded.
    """
    global cuda_available, cuda_warning_is_displayed
    cuda_available = False
    if not cuda_warning_is_displayed:
        cuda_warning_is_displayed = True
        warning('Cuda is disabled, cuda-based code will thus not be '
                'working properly')

#cuda_ndarray compile and import
cuda_path = os.path.split(__file__)[0]
date = os.stat(os.path.join(cuda_path,'cuda_ndarray.cu'))[stat.ST_MTIME]
date = max(date,os.stat(os.path.join(cuda_path,'cuda_ndarray.cuh'))[stat.ST_MTIME])
date = max(date,os.stat(os.path.join(cuda_path,'conv_full_kernel.cu'))[stat.ST_MTIME])
date = max(date,os.stat(os.path.join(cuda_path,'conv_kernel.cu'))[stat.ST_MTIME])

cuda_ndarray_loc = os.path.join(config.compiledir,'cuda_ndarray')
cuda_ndarray_so = os.path.join(cuda_ndarray_loc,'cuda_ndarray.so')
compile_cuda_ndarray = True

if os.path.exists(cuda_ndarray_so):
    compile_cuda_ndarray = date>=os.stat(cuda_ndarray_so)[stat.ST_MTIME]
if not compile_cuda_ndarray:
    try:
        from cuda_ndarray.cuda_ndarray import *
    except ImportError:
        compile_cuda_ndarray = True

try:
    if compile_cuda_ndarray:
        import nvcc_compiler
        if not nvcc_compiler.is_nvcc_available():
            error('nvcc compiler not found on $PATH.'
                    '  Check your nvcc installation and try again')
            set_cuda_disabled()

        if cuda_available:
            code = open(os.path.join(cuda_path, "cuda_ndarray.cu")).read()

            if not os.path.exists(cuda_ndarray_loc):
                os.makedirs(cuda_ndarray_loc)

            nvcc_compiler.nvcc_module_compile_str('cuda_ndarray', code, location = cuda_ndarray_loc,
                                                  include_dirs=[cuda_path], libs=['cublas'])

            from cuda_ndarray.cuda_ndarray import *
except Exception, e:
    error( "Failed to compile cuda_ndarray.cu: %s" % str(e))
    set_cuda_disabled()

if cuda_available:
    #check if their is an old cuda_ndarray that was loading instead of the one we compiled!
    import cuda_ndarray.cuda_ndarray
    if os.path.join(config.compiledir,'cuda_ndarray','cuda_ndarray.so')!=cuda_ndarray.cuda_ndarray.__file__:
        _logger.warning("WARNING: cuda_ndarray was loaded from",cuda_ndarray.cuda_ndarray.__file__,"This is not expected as theano should compile it automatically for you. Do you have a directory called cuda_ndarray in your LD_LIBRARY_PATH environment variable? If so, please remove it as it is outdated!")

    from theano.sandbox.cuda.type import CudaNdarrayType
    from theano.sandbox.cuda.var import (CudaNdarrayVariable,
            CudaNdarrayConstant,
            CudaNdarraySharedVariable,
            float32_shared_constructor)
    shared_constructor = float32_shared_constructor

    import basic_ops
    from basic_ops import (GpuFromHost, HostFromGpu, GpuElemwise, 
            GpuDimShuffle, GpuSum, GpuReshape, 
            GpuSubtensor, GpuIncSubtensor, GpuFlatten, GpuShape)
    import opt
    import cuda_ndarray


def use(device):
    global cuda_enabled, enabled_cuda
    if device.startswith('gpu'):
        device = int(device[3:])
    elif device == 'cpu':
        device = -1
    else:
        raise ValueError("Invalid device identifier", device)
    if use.device_number is None:
        # No successful call to use() has been made yet
        if device<0:
            return
        if device in [None,""]:
            device=0
        device=int(device)
        try:
            gpu_init(device)
            handle_shared_float32(True)
            use.device_number = device
            cuda_enabled = True
        except RuntimeError, e:
            _logger.error("ERROR: Not using GPU. Initialisation of device %i failed. %s" %(device, e))
            enabled_cuda = False
    elif use.device_number != device:
        _logger.warning("WARNING: ignoring call to use(%s), GPU number %i is already in use." %(str(device), use.device_number))
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
        theano.compile.shared_constructor(float32_shared_constructor)

    else:
        raise NotImplementedError('removing our handler')

if cuda_available and config.device.startswith('gpu'):
    use(config.device)

