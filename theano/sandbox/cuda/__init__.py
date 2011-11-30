import atexit, logging, os, shutil, stat, sys
from theano.compile import optdb
from theano.gof.cmodule import get_lib_extension
from theano.configparser import config, AddConfigVar, StrParam
import nvcc_compiler

_logger_name = 'theano.sandbox.cuda'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.WARNING)

AddConfigVar('cuda.root',
        """directory with bin/, lib/, include/ for cuda utilities.
        This directory is included via -L and -rpath when linking
        dynamically compiled modules.  If AUTO and nvcc is in the
        path, it will use one of nvcc parent directory.  Otherwise
        /usr/local/cuda will be used.  Leave empty to prevent extra
        linker directives.  Default: environment variable "CUDA_ROOT"
        or else "AUTO".
        """,
        StrParam(os.getenv('CUDA_ROOT', "AUTO")))

if config.cuda.root == "AUTO":
    # set nvcc_path correctly and get the version
    nvcc_compiler.set_cuda_root()

#is_nvcc_available called here to initialize global vars in nvcc_compiler module
nvcc_compiler.is_nvcc_available()

# Compile cuda_ndarray.cu
# This need that nvcc (part of cuda) is installed. If it is not, a warning is
# printed and this module will not be working properly (we set `cuda_available`
# to False).

# This variable is True by default, and set to False if nvcc is not available or
# their is no cuda card or something goes wrong when trying to initialize cuda.
cuda_available = True

# Global variable to avoid displaying the same warning multiple times.
cuda_warning_is_displayed = False

#This variable is set to True when we enable cuda.(i.e. when use() is called)
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

#cuda_ndarray compile and import
cuda_path = os.path.abspath(os.path.split(__file__)[0])
cuda_files = (
        'cuda_ndarray.cu',
        'cuda_ndarray.cuh',
        'conv_full_kernel.cu',
        'conv_kernel.cu')
stat_times = [os.stat(os.path.join(cuda_path, cuda_file))[stat.ST_MTIME]
        for cuda_file in cuda_files]
date = max(stat_times)

cuda_ndarray_loc = os.path.join(config.compiledir, 'cuda_ndarray')
cuda_ndarray_so = os.path.join(cuda_ndarray_loc,
                               'cuda_ndarray.' + get_lib_extension())
libcuda_ndarray_so = os.path.join(cuda_ndarray_loc,
                               'libcuda_ndarray.' + get_lib_extension())


# Add the theano cache directory's cuda_ndarray subdirectory to the list of
# places that are hard-coded into compiled modules' runtime library search
# list.  This works in conjunction with nvcc_compiler.nvcc_module_compile_str
# which adds this folder during compilation with -L and also adds -lcuda_ndarray
# when compiling modules.
nvcc_compiler.add_standard_rpath(cuda_ndarray_loc)

compile_cuda_ndarray = True

if os.path.exists(cuda_ndarray_so):
    compile_cuda_ndarray = date>=os.stat(cuda_ndarray_so)[stat.ST_MTIME]
if not compile_cuda_ndarray:
    try:
        # If we load a previously-compiled version, config.compiledir should
        # be in sys.path.
        if config.compiledir not in sys.path:
            sys.path.append(config.compiledir)
        from cuda_ndarray.cuda_ndarray import *
    except ImportError:
        compile_cuda_ndarray = True

try:
    if compile_cuda_ndarray:
        if not nvcc_compiler.is_nvcc_available():
            set_cuda_disabled()

        if cuda_available:
            code = open(os.path.join(cuda_path, "cuda_ndarray.cu")).read()

            if not os.path.exists(cuda_ndarray_loc):
                os.makedirs(cuda_ndarray_loc)

            nvcc_compiler.nvcc_module_compile_str(
                    'cuda_ndarray',
                    code,
                    location=cuda_ndarray_loc,
                    include_dirs=[cuda_path], libs=['cublas'])
            from cuda_ndarray.cuda_ndarray import *
except Exception, e:
    _logger.error( "Failed to compile cuda_ndarray.cu: %s", str(e))
    set_cuda_disabled()

if cuda_available:
    # If necessary,
    # create a symlink called libcuda_ndarray.so
    # which nvcc_module_compile_str uses when linking
    # any module except "cuda_ndarray" itself.
    try:
        open(libcuda_ndarray_so).close()
    except IOError:
        if sys.platform == "win32":
            # The Python `os` module does not support symlinks on win32.
            shutil.copyfile(cuda_ndarray_so, libcuda_ndarray_so)
        else:
            os.symlink(cuda_ndarray_so, libcuda_ndarray_so)

    try:
        gpu_init()
        cuda_available = True
        cuda_initialization_error_message = ""
        # actively closing our gpu session presents segfault-on-exit on some systems
        atexit.register(gpu_shutdown)
    except EnvironmentError, e:
        cuda_available = False
        cuda_initialization_error_message = e.message

# We must do those import to be able to create the full doc when
# nvcc is not available
from theano.sandbox.cuda.var import (CudaNdarrayVariable,
                                     CudaNdarrayConstant,
                                     CudaNdarraySharedVariable,
                                     float32_shared_constructor)
from theano.sandbox.cuda.type import CudaNdarrayType

if cuda_available:
    # check if their is an old cuda_ndarray that was loading instead of the one
    # we compiled!
    import cuda_ndarray.cuda_ndarray
    if cuda_ndarray_so != cuda_ndarray.cuda_ndarray.__file__:
        _logger.warning("cuda_ndarray was loaded from %s, but Theano expected "
                "to load it from %s. This is not expected as theano should "
                "compile it automatically for you. Do you have a directory "
                "called cuda_ndarray in your LD_LIBRARY_PATH environment "
                "variable? If so, please remove it as it is outdated.",
                cuda_ndarray.cuda_ndarray.__file__,
                cuda_ndarray_so)

    shared_constructor = float32_shared_constructor

    import basic_ops
    from basic_ops import (GpuFromHost, HostFromGpu, GpuElemwise,
                           GpuDimShuffle, GpuSum, GpuReshape, GpuContiguous,
                           GpuSubtensor, GpuIncSubtensor,
                           GpuAdvancedSubtensor1, GpuAdvancedIncSubtensor1,
                           GpuFlatten, GpuShape, GpuAlloc,
                           GpuJoin, fscalar, fvector, fmatrix, frow, fcol,
                           ftensor3, ftensor4, scalar, vector, matrix, row, col,
                           tensor3, tensor4)
    from basic_ops import host_from_gpu, gpu_from_host, as_cuda_array
    import opt
    import cuda_ndarray
    from rng_curand import CURAND_RandomStreams


def use(device,
        force=False,
        default_to_move_computation_to_gpu=True,
        move_shared_float32_to_gpu=True,
        enable_cuda=True):
    """
    Error and warning about CUDA should be displayed only when this function is called.
    We need to be able to load this module only to check if it is available!
    """
    global cuda_enabled, cuda_initialization_error_message
    if force and not cuda_available and device.startswith('gpu'):
        if not nvcc_compiler.is_nvcc_available():
            raise EnvironmentError("You forced the use of gpu device '%s', but "
                                   "nvcc was not found. Set it in your PATH "
                                   "environment variable or set the Theano "
                                   "flags 'cuda.root' to its directory" % device)
        else:
            raise EnvironmentError("You forced the use of gpu device %s, "
                                   "but CUDA initialization failed "
                                   "with error:\n%s" % (
                device, cuda_initialization_error_message))
    elif not nvcc_compiler.is_nvcc_available():
        _logger.error('nvcc compiler not found on $PATH.'
              ' Check your nvcc installation and try again.')
        return
    elif not cuda_available:
        error_addendum = ""
        try:
            if cuda_initialization_error_message:
                error_addendum = " (error: %s)" % cuda_initialization_error_message
        except NameError: # cuda_initialization_error_message is not available b/c compilation failed
            pass
        _logger.warning('CUDA is installed, but device %s is not available %s',
                device, error_addendum)
        return

    if device == 'gpu':
        pass
    elif device.startswith('gpu'):
        device = int(device[3:])
    elif device == 'cpu':
        device = -1
    else:
        raise ValueError("Invalid device identifier", device)
    if use.device_number is None:
        # No successful call to use() has been made yet
        if device != 'gpu' and device<0:
            return
        if device in [None,""]:
            device=0
        try:
            if device !='gpu':
                gpu_init(device)

            if move_shared_float32_to_gpu:
                handle_shared_float32(True)
            use.device_number = device

            if enable_cuda:
                cuda_enabled = True
            print >> sys.stderr, "Using gpu device %d: %s" % (active_device_number(), active_device_name())
        except (EnvironmentError, ValueError), e:
            _logger.error(("ERROR: Not using GPU."
                " Initialisation of device %i failed:\n%s"),
                device, e)
            cuda_enabled = False
            if force:
                e.args+=(("You asked to force this device and it failed."
                        " No fallback to the cpu or other gpu device."),)
                raise

    elif use.device_number != device:
        _logger.warning(("Ignoring call to use(%s), GPU number %i "
            "is already in use."),
            str(device), use.device_number)

    if default_to_move_computation_to_gpu:
        optdb.add_tags('gpu_opt',
                       'fast_run',
                       'inplace')
        optdb.add_tags('gpu_after_fusion',
                       'fast_run',
                       'inplace')

    if force:
        try:
            #in case the device if just gpu,
            # we check that the driver init it correctly.
            cuda_ndarray.cuda_ndarray.CudaNdarray.zeros((5,5))
        except (Exception, NameError), e:
            # NameError when no gpu present as cuda_ndarray is not loaded.
            e.args+=("ERROR: GPU forced but failed. ",)
            raise


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

if config.device.startswith('gpu'):
    use(device=config.device, force=config.force_device)
elif config.init_gpu_device:
    assert config.device=="cpu", ("We can use the Theano flag init_gpu_device"
            " only when the Theano flag device=='cpu'")
    _logger.warning(("GPU device %s will be initialized, and used if a GPU is "
          "needed. "
          "However, no computation, nor shared variables, will be implicitly "
          "moved to that device. If you want that behavior, use the 'device' "
          "flag instead."),
          config.init_gpu_device)
    use(device=config.init_gpu_device,
        force=config.force_device,
        default_to_move_computation_to_gpu=False,
        move_shared_float32_to_gpu=False,
        enable_cuda=False)
