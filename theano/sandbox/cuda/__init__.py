import atexit
import errno
import logging
import os
import shutil
import stat
import sys

import numpy

import theano
from theano.compile import optdb
from theano.gof.cmodule import get_lib_extension
from theano.gof.compilelock import get_lock, release_lock
from theano.configparser import config, AddConfigVar, StrParam, BoolParam
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

AddConfigVar('pycuda.init',
        """If True, always initialize PyCUDA when Theano want to
           initilize the GPU.  Currently, we must always initialize
           PyCUDA before Theano do it.  Setting this flag to True,
           ensure that, but always import PyCUDA.  It can be done
           manually by importing theano.misc.pycuda_init before theano
           initialize the GPU device.
             """,
        BoolParam(False))

if config.cuda.root == "AUTO":
    # set nvcc_path correctly and get the version
    nvcc_compiler.set_cuda_root()

#is_nvcc_available called here to initialize global vars in
#nvcc_compiler module
nvcc_compiler.is_nvcc_available()

# Compile cuda_ndarray.cu
# This need that nvcc (part of cuda) is installed. If it is not, a warning is
# printed and this module will not be working properly (we set `cuda_available`
# to False).

# This variable is True by default, and set to False if nvcc is not
# available or their is no cuda card or something goes wrong when
# trying to initialize cuda.
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

cuda_ndarray_loc = os.path.join(config.compiledir, 'cuda_ndarray')
cuda_ndarray_so = os.path.join(cuda_ndarray_loc,
                               'cuda_ndarray.' + get_lib_extension())
libcuda_ndarray_so = os.path.join(cuda_ndarray_loc,
                               'libcuda_ndarray.' + get_lib_extension())


def try_import():
    """
    load the cuda_ndarray module if present and up to date
    return True if loaded correctly, otherwise return False
    """
    cuda_files = (
        'cuda_ndarray.cu',
        'cuda_ndarray.cuh',
        'conv_full_kernel.cu',
        'conv_kernel.cu')
    stat_times = [os.stat(os.path.join(cuda_path, cuda_file))[stat.ST_MTIME]
                  for cuda_file in cuda_files]
    date = max(stat_times)
    if os.path.exists(cuda_ndarray_so):
        if date >= os.stat(cuda_ndarray_so)[stat.ST_MTIME]:
            return False
    try:
        # If we load a previously-compiled version, config.compiledir should
        # be in sys.path.
        sys.path[0:0] = [config.compiledir]
        import cuda_ndarray.cuda_ndarray
        del sys.path[0]
    except ImportError:
        return False
    return True


# Add the theano cache directory's cuda_ndarray subdirectory to the
# list of places that are hard-coded into compiled modules' runtime
# library search list.  This works in conjunction with
# nvcc_compiler.NVCC_compiler.compile_str which adds this folder during
# compilation with -L and also adds -lcuda_ndarray when compiling
# modules.
nvcc_compiler.add_standard_rpath(cuda_ndarray_loc)

compile_cuda_ndarray = True

if not compile_cuda_ndarray:
    compile_cuda_ndarray = not try_import()

if not nvcc_compiler.is_nvcc_available() or not theano.config.cxx:
    # It can happen that the file cuda_ndarray.so is already compiled
    # but nvcc is not available. In that case we need to disable the CUDA
    # back-end as we won't be able to compile any new op and we can't only
    # use already compiled GPU op and not the others.
    # Also, if cxx is not available, we need to disable all GPU code.
    set_cuda_disabled()

if compile_cuda_ndarray and cuda_available:
    get_lock()
    try:
        # Retry to load again in case someone else compiled it
        # while we waited for the lock
        if not try_import():
            try:
                if not nvcc_compiler.is_nvcc_available():
                    set_cuda_disabled()

                if cuda_available:
                    code = open(os.path.join(cuda_path,
                                             "cuda_ndarray.cu")).read()

                    if not os.path.exists(cuda_ndarray_loc):
                        os.makedirs(cuda_ndarray_loc)

                    # If $TMPDIR is defined, nvopencc wants it to exist
                    if 'TMPDIR' in os.environ:
                        tmpdir = os.environ['TMPDIR']
                        if not os.path.exists(tmpdir):
                            os.makedirs(tmpdir)

                    compiler = nvcc_compiler.NVCC_compiler()
                    compiler.compile_str(
                            'cuda_ndarray',
                            code,
                            location=cuda_ndarray_loc,
                            include_dirs=[cuda_path], libs=['cublas'],
                            preargs=['-O3'] + compiler.compile_args())
                    from cuda_ndarray.cuda_ndarray import *
            except Exception, e:
                _logger.error("Failed to compile cuda_ndarray.cu: %s", str(e))
                set_cuda_disabled()
    finally:
        release_lock()

del compile_cuda_ndarray

if cuda_available:
    # The module should be compiled.
    from cuda_ndarray.cuda_ndarray import *

    # If necessary,
    # create a symlink called libcuda_ndarray.so
    # which nvcc_compiler.NVCC_compiler uses when linking
    # any module except "cuda_ndarray" itself.
    def ok():
        """
        Check if an existing library exists and can be read.
        """
        try:
            open(libcuda_ndarray_so).close()
            return True
        except IOError:
            return False
    if not ok():
        if sys.platform == "win32":
            # The Python `os` module does not support symlinks on win32.
            shutil.copyfile(cuda_ndarray_so, libcuda_ndarray_so)
        else:
            try:
                os.symlink(cuda_ndarray_so, libcuda_ndarray_so)
            except OSError, e:
                # This may happen for instance when running multiple
                # concurrent jobs, if two of them try to create the
                # symlink simultaneously.
                # If that happens, we verify that the existing symlink is
                # indeed working.
                if getattr(e, 'errno', None) != errno.EEXIST or not ok():
                    raise
    try:
        # This only test if the cuda driver is available and if there
        # is at least one GPU that support cuda. This do not select a
        # device.
        gpu_init()
        cuda_available = True
        cuda_initialization_error_message = ""
# actively closing our gpu session presents segfault-on-exit on some systems
        atexit.register(gpu_shutdown)
    except EnvironmentError, e:
        cuda_available = False
        cuda_initialization_error_message = e.message


class GpuOp(theano.gof.Op):

    """
    Parent class for all GPU Ops.

    This class ensures we verify the GPU is working properly when a GPU Op is
    used for the first time.

    It is defined in __init__.py so that it exists even when `cuda_available`
    is False (this is necessary to avoid breaking the test suite).
    """

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        if theano.sandbox.cuda.use.device_number is None:
            theano.sandbox.cuda.use("gpu",
                                    force=True,
                                    default_to_move_computation_to_gpu=False,
                                    move_shared_float32_to_gpu=False,
                                    enable_cuda=False)
        return super(GpuOp, self).make_thunk(node, storage_map,
                                             compute_map, no_recycling)

theano.compile.debugmode.default_make_thunk.append(GpuOp.make_thunk.im_func)

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
                           GpuDimShuffle, GpuCAReduce, GpuReshape, GpuContiguous,
                           GpuSubtensor, GpuIncSubtensor,
                           GpuAdvancedSubtensor1, GpuAdvancedIncSubtensor1,
                           GpuFlatten, GpuShape, GpuAlloc,
                           GpuJoin, fscalar, fvector, fmatrix, frow, fcol,
                           ftensor3, ftensor4,
                           scalar, vector, matrix, row, col,
                           tensor3, tensor4)
    from basic_ops import host_from_gpu, gpu_from_host, as_cuda_array
    import opt
    import cuda_ndarray
    from rng_curand import CURAND_RandomStreams


def use(device,
        force=False,
        default_to_move_computation_to_gpu=True,
        move_shared_float32_to_gpu=True,
        enable_cuda=True,
        test_driver=True):
    """
    Error and warning about CUDA should be displayed only when this
    function is called.  We need to be able to load this module only
    to check if it is available!

    :param device: string "cpu", "gpu", "gpuN" (N is the device number to use)
    :param force: Will always raise an exception if we can't use the gpu.
    :param default_to_move_computation_to_gpu: If gpu init succeeded, enable by
                                               default optimizations to move
                                               computations to the gpu
    :param move_shared_float32_to_gpu: If gpu init succeeded, put new shared
                                       variables in float32 on the gpu.
    :param enable_cuda: If the gpu is correctly enabled,
                        set the variable cuda_enabled to True.
    """
    global cuda_enabled, cuda_initialization_error_message
    if force and not cuda_available and device.startswith('gpu'):
        if not nvcc_compiler.is_nvcc_available():
            raise EnvironmentError("You forced the use of gpu device '%s', but"
                                   " nvcc was not found. Set it in your PATH "
                                   "environment variable or set the Theano "
                                   "flags 'cuda.root' to its directory"
                                   "" % device)
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
                error_addendum = (" (error: %s)" %
                                  cuda_initialization_error_message)
        except NameError:
# cuda_initialization_error_message is not available b/c compilation failed
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
        if device != 'gpu' and device < 0:
            return

        # Has PyCUDA already initialized the GPU context
        pycuda_init_dev = False
        if config.pycuda.init:
            import theano.misc.pycuda_init
            pycuda_init_dev = theano.misc.pycuda_init.pycuda_available

        try:
            if (device != 'gpu') and not pycuda_init_dev:
                assert isinstance(device, int)
                gpu_init(device)
                use.device_number = device
                assert active_device_number() == device
            else:
                # This mean the driver should select the GPU.  As we
                # need to get the device number now, we force the
                # selection of the GPU by the driver now and then we
                # query the active GPU. If we check the active GPU before
                # the device is initialized we will always receive 0
                # event if another device is selected later.
                cuda_ndarray.cuda_ndarray.CudaNdarray.zeros((2, 3))
                use.device_number = active_device_number()

            if test_driver:
                import theano.sandbox.cuda.tests.test_driver
                theano.sandbox.cuda.tests.test_driver.test_nvidia_driver1()
            if device_properties(use.device_number)["warpSize"] != 32:
                raise ValueError("Your GPU has a warpSize != 32. Currently"
                                 " we have code that depends on this. Email"
                                 " the Theano mailing list to tell us about"
                                 " this new GPU as we don't know any with"
                                 " this property")
            if move_shared_float32_to_gpu:
                handle_shared_float32(True)

            if enable_cuda:
                cuda_enabled = True

            if config.print_active_device:
                print >> sys.stderr, "Using gpu device %d: %s" %(
                        active_device_number(), active_device_name())
            if device_properties(use.device_number)['regsPerBlock'] < 16384:
                # We will try to use too much register per bloc at many places
                # when there is only 8k register per multi-processor.
                _logger.warning("You are probably using an old GPU."
                                " We didn't optimize nor we support those GPU."
                                " This mean GPU code will be slow AND will"
                                " crash when we try to use feature/properties"
                                " that your GPU don't support.")

        except (EnvironmentError, ValueError, RuntimeError), e:
            _logger.error(("ERROR: Not using GPU."
                           " Initialisation of device %s failed:\n%s"),
                          str(device), e)
            cuda_enabled = False
            if force:
                e.args += (("You asked to force this device and it failed."
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
            cuda_ndarray.cuda_ndarray.CudaNdarray.zeros((5, 5))
        except (Exception, NameError), e:
            # NameError when no gpu present as cuda_ndarray is not loaded.
            e.args += ("ERROR: GPU forced but failed. ",)
            raise
use.device_number = None


def handle_shared_float32(tf):
    """Set the default shared type for float32 tensor to CudaNdarrayType

    This function is intended to be called from use(gpu_index), not directly.
    """
    if tf:
        import theano.compile
        theano.compile.shared_constructor(float32_shared_constructor)

    else:
        theano.compile.shared_constructor(float32_shared_constructor, True)
        assert (float32_shared_constructor not in
                theano.compile.shared.constructors)

# We can't test the driver during import here as this cause circular
# import dependency. So we also test it in the file theano/__init__.py
if config.device.startswith('gpu'):
    use(device=config.device, force=config.force_device, test_driver=False)
elif config.init_gpu_device:
    assert config.device == "cpu", (
        "We can use the Theano flag init_gpu_device"
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
        enable_cuda=False, test_driver=False)
