from __future__ import absolute_import, print_function, division
import errno
import os
import sys
import logging
import numpy
import platform
import textwrap
import re
import socket
import struct
import warnings

from six import string_types

import theano
from theano.configparser import (AddConfigVar, BoolParam, ConfigParam, EnumStr,
                                 FloatParam, IntParam, StrParam,
                                 TheanoConfigParser, THEANO_FLAGS_DICT)
from theano.misc.cpucount import cpuCount
from theano.misc.windows import call_subprocess_Popen, output_subprocess_Popen
from theano.compat import maybe_add_to_os_environ_pathlist


_logger = logging.getLogger('theano.configdefaults')

config = TheanoConfigParser()


def floatX_convert(s):
    if s == "32":
        return "float32"
    elif s == "64":
        return "float64"
    elif s == "16":
        return "float16"
    else:
        return s

AddConfigVar('floatX',
             "Default floating-point precision for python casts.\n"
             "\n"
             "Note: float16 support is experimental, use at your own risk.",
             EnumStr('float64', 'float32', 'float16',
                     convert=floatX_convert,),
             # TODO: see gh-4466 for how to remove it.
             in_c_key=True
             )

AddConfigVar('warn_float64',
             "Do an action when a tensor variable with float64 dtype is"
             " created. They can't be run on the GPU with the current(old)"
             " gpu back-end and are slow with gamer GPUs.",
             EnumStr('ignore', 'warn', 'raise', 'pdb'),
             in_c_key=False,
             )

AddConfigVar('cast_policy',
             'Rules for implicit type casting',
             EnumStr('custom', 'numpy+floatX',
                     # The 'numpy' policy was originally planned to provide a
                     # smooth transition from numpy. It was meant to behave the
                     # same as numpy+floatX, but keeping float64 when numpy
                     # would. However the current implementation of some cast
                     # mechanisms makes it a bit more complex to add than what
                     # was expected, so it is currently not available.
                     # numpy,
                     ),
             )

# python 2.* define int / int to return int and int // int to return int.
# python 3* define int / int to return float and int // int to return int.
# numpy 1.6.1 behaves as python 2.*. I think we should not change it faster
# than numpy. When we will do the transition, we should create an int_warn
# and floatX_warn option.
AddConfigVar('int_division',
             "What to do when one computes x / y, where both x and y are of "
             "integer types",
             EnumStr('int', 'raise', 'floatX'),
             in_c_key=False)

# gpu means let the driver select the gpu. Needed in case of gpu in
# exclusive mode.
# gpuX mean use the gpu number X.


class DeviceParam(ConfigParam):
    def __init__(self, default, *options, **kwargs):
        self.default = default

        def filter(val):
            if val == self.default or val.startswith('gpu') \
                    or val.startswith('opencl') or val.startswith('cuda'):
                return val
            else:
                raise ValueError(('Invalid value ("%s") for configuration '
                                  'variable "%s". Valid options start with '
                                  'one of "%s", "gpu", "opencl", "cuda"'
                                  % (self.default, val, self.fullname)))
        over = kwargs.get("allow_override", True)
        super(DeviceParam, self).__init__(default, filter, over)

    def __str__(self):
        return '%s (%s, gpu*, opencl*, cuda*) ' % (self.fullname, self.default)

AddConfigVar(
    'device',
    ("Default device for computations. If cuda* or opencl*, change the"
     "default to try to move computation to the GPU. Do not use upper case"
     "letters, only lower case even if NVIDIA uses capital letters."),
    DeviceParam('cpu', allow_override=False),
    in_c_key=False)

AddConfigVar(
    'init_gpu_device',
    ("Initialize the gpu device to use, works only if device=cpu. "
     "Unlike 'device', setting this option will NOT move computations, "
     "nor shared variables, to the specified GPU. "
     "It can be used to run GPU-specific tests on a particular GPU."),
    DeviceParam('', allow_override=False),
    in_c_key=False)

AddConfigVar(
    'force_device',
    "Raise an error if we can't use the specified device",
    BoolParam(False, allow_override=False),
    in_c_key=False)

AddConfigVar(
    'print_global_stats',
    "Print some global statistics (time spent) at the end",
    BoolParam(False),
    in_c_key=False)


class ContextsParam(ConfigParam):
    def __init__(self):
        def filter(val):
            if val == '':
                return val
            for v in val.split(';'):
                s = v.split('->')
                if len(s) != 2:
                    raise ValueError("Malformed context map: %s" % (v,))
                if (s[0] == 'cpu' or s[0].startswith('cuda') or
                        s[0].startswith('opencl')):
                    raise ValueError("Cannot use %s as context name" % (s[0],))
            return val
        ConfigParam.__init__(self, '', filter, False)

AddConfigVar(
    'contexts',
    """
    Context map for multi-gpu operation. Format is a
    semicolon-separated list of names and device names in the
    'name->dev_name' format. An example that would map name 'test' to
    device 'cuda0' and name 'test2' to device 'opencl0:0' follows:
    "test->cuda0;test2->opencl0:0".

    Invalid context names are 'cpu', 'cuda*' and 'opencl*'
    """, ContextsParam(), in_c_key=False)

AddConfigVar(
    'print_active_device',
    "Print active device at when the GPU device is initialized.",
    BoolParam(True, allow_override=False),
    in_c_key=False)


AddConfigVar(
    'enable_initial_driver_test',
    "Tests the nvidia driver when a GPU device is initialized.",
    BoolParam(True, allow_override=False),
    in_c_key=False)


def default_cuda_root():
    v = os.getenv('CUDA_ROOT', "")
    if v:
        return v
    s = os.getenv("PATH")
    if not s:
        return ''
    for dir in s.split(os.path.pathsep):
        if os.path.exists(os.path.join(dir, "nvcc")):
            return os.path.dirname(os.path.abspath(dir))
    return ''

AddConfigVar(
    'cuda.root',
    """directory with bin/, lib/, include/ for cuda utilities.
       This directory is included via -L and -rpath when linking
       dynamically compiled modules.  If AUTO and nvcc is in the
       path, it will use one of nvcc parent directory.  Otherwise
       /usr/local/cuda will be used.  Leave empty to prevent extra
       linker directives.  Default: environment variable "CUDA_ROOT"
       or else "AUTO".
       """,
    StrParam(default_cuda_root),
    in_c_key=False)

AddConfigVar(
    'cuda.enabled',
    'If false, C code in old backend is not compiled.',
    BoolParam(True),
    in_c_key=False)


def filter_nvcc_flags(s):
    assert isinstance(s, str)
    flags = [flag for flag in s.split(' ') if flag]
    if any([f for f in flags if not f.startswith("-")]):
        raise ValueError(
            "Theano nvcc.flags support only parameter/value pairs without"
            " space between them. e.g.: '--machine 64' is not supported,"
            " but '--machine=64' is supported. Please add the '=' symbol."
            " nvcc.flags value is '%s'" % s)
    return ' '.join(flags)

AddConfigVar('nvcc.flags',
             "Extra compiler flags for nvcc",
             ConfigParam("", filter_nvcc_flags),
             # Not needed in c key as it is already added.
             # We remove it as we don't make the md5 of config to change
             # if theano.sandbox.cuda is loaded or not.
             in_c_key=False)

AddConfigVar('nvcc.compiler_bindir',
             "If defined, nvcc compiler driver will seek g++ and gcc"
             " in this directory",
             StrParam(""),
             in_c_key=False)

AddConfigVar('nvcc.fastmath',
             "",
             BoolParam(False),
             # Not needed in c key as it is already added.
             # We remove it as we don't make the md5 of config to change
             # if theano.sandbox.cuda is loaded or not.
             in_c_key=False)

AddConfigVar('gpuarray.sync',
             """If True, every op will make sure its work is done before
                returning.  Setting this to True will slow down execution,
                but give much more accurate results in profiling.""",
             BoolParam(False),
             in_c_key=True)

AddConfigVar('gpuarray.preallocate',
             """If negative it disables the allocation cache. If
             between 0 and 1 it enables the allocation cache and
             preallocates that fraction of the total GPU memory.  If 1
             or greater it will preallocate that amount of memory (in
             megabytes).""",
             FloatParam(0, allow_override=False),
             in_c_key=False)

AddConfigVar('gpuarray.sched',
             """The sched parameter passed for context creation to pygpu.
                With CUDA, using "multi" is equivalent to using the parameter
                cudaDeviceScheduleYield. This is useful to lower the
                CPU overhead when waiting for GPU. One user found that it
                speeds up his other processes that was doing data augmentation.
             """,
             EnumStr("default", "multi", "single"))

AddConfigVar('gpuarray.single_stream',
             """
             If your computations are mostly lots of small elements,
             using single-stream will avoid the synchronization
             overhead and usually be faster.  For larger elements it
             does not make a difference yet.  In the future when true
             multi-stream is enabled in libgpuarray, this may change.
             If you want to make sure to have optimal performance,
             check both options.
             """,
             BoolParam(True),
             in_c_key=False)


def safe_no_dnn_workmem(workmem):
    """
    Make sure the user is not attempting to use dnn.conv.workmem`.
    """
    if workmem:
        raise RuntimeError(
            'The option `dnn.conv.workmem` has been removed and should '
            'not be used anymore. Please use the option '
            '`dnn.conv.algo_fwd` instead.')
    return True

AddConfigVar('dnn.conv.workmem',
             "This flag is deprecated; use dnn.conv.algo_fwd.",
             ConfigParam('', allow_override=False, filter=safe_no_dnn_workmem),
             in_c_key=False)


def safe_no_dnn_workmem_bwd(workmem):
    """
    Make sure the user is not attempting to use dnn.conv.workmem_bwd`.
    """
    if workmem:
        raise RuntimeError(
            'The option `dnn.conv.workmem_bwd` has been removed and '
            'should not be used anymore. Please use the options '
            '`dnn.conv.algo_bwd_filter` and `dnn.conv.algo_bwd_data` instead.')
    return True

AddConfigVar('dnn.conv.workmem_bwd',
             "This flag is deprecated; use `dnn.conv.algo_bwd_filter` "
             "and `dnn.conv.algo_bwd_data` instead.",
             ConfigParam('', allow_override=False,
                         filter=safe_no_dnn_workmem_bwd),
             in_c_key=False)


def safe_no_dnn_algo_bwd(algo):
    """
    Make sure the user is not attempting to use dnn.conv.algo_bwd`.
    """
    if algo:
        raise RuntimeError(
            'The option `dnn.conv.algo_bwd` has been removed and '
            'should not be used anymore. Please use the options '
            '`dnn.conv.algo_bwd_filter` and `dnn.conv.algo_bwd_data` instead.')
    return True

# Those are the supported algorithm by Theano,
# The tests will reference those lists.
SUPPORTED_DNN_CONV_ALGO_FWD = ('small', 'none', 'large', 'fft', 'fft_tiling',
                               'winograd', 'guess_once', 'guess_on_shape_change',
                               'time_once', 'time_on_shape_change')

SUPPORTED_DNN_CONV_ALGO_BWD_DATA = ('none', 'deterministic', 'fft', 'fft_tiling',
                                    'winograd', 'guess_once', 'guess_on_shape_change',
                                    'time_once', 'time_on_shape_change')

SUPPORTED_DNN_CONV_ALGO_BWD_FILTER = ('none', 'deterministic', 'fft', 'small',
                                      'guess_once', 'guess_on_shape_change',
                                      'time_once', 'time_on_shape_change')

AddConfigVar('dnn.conv.algo_bwd',
             "This flag is deprecated; use dnn.conv.algo_bwd_data and "
             "dnn.conv.algo_bwd_filter.",
             ConfigParam('', allow_override=False,
                         filter=safe_no_dnn_algo_bwd),
             in_c_key=False)

AddConfigVar('dnn.conv.algo_fwd',
             "Default implementation to use for cuDNN forward convolution.",
             EnumStr(*SUPPORTED_DNN_CONV_ALGO_FWD),
             in_c_key=False)

AddConfigVar('dnn.conv.algo_bwd_data',
             "Default implementation to use for cuDNN backward convolution to "
             "get the gradients of the convolution with regard to the inputs.",
             EnumStr(*SUPPORTED_DNN_CONV_ALGO_BWD_DATA),
             in_c_key=False)

AddConfigVar('dnn.conv.algo_bwd_filter',
             "Default implementation to use for cuDNN backward convolution to "
             "get the gradients of the convolution with regard to the "
             "filters.",
             EnumStr(*SUPPORTED_DNN_CONV_ALGO_BWD_FILTER),
             in_c_key=False)

AddConfigVar('dnn.conv.precision',
             "Default data precision to use for the computation in cuDNN "
             "convolutions (defaults to the same dtype as the inputs of the "
             "convolutions, or float32 if inputs are float16).",
             EnumStr('as_input_f32', 'as_input', 'float16', 'float32',
                     'float64'),
             in_c_key=False)


def default_dnn_path(suffix):
    def f(suffix=suffix):
        if theano.config.cuda.root == '':
            return ''
        return os.path.join(theano.config.cuda.root, suffix)
    return f

AddConfigVar('dnn.include_path',
             "Location of the cudnn header (defaults to the cuda root)",
             StrParam(default_dnn_path('include')),
             # Added elsewhere in the c key only when needed.
             in_c_key=False)

AddConfigVar('dnn.library_path',
             "Location of the cudnn header (defaults to the cuda root)",
             StrParam(default_dnn_path('lib' if sys.platform == 'darwin' else 'lib64')),
             # Added elsewhere in the c key only when needed.
             in_c_key=False)

AddConfigVar('dnn.enabled',
             "'auto', use cuDNN if available, but silently fall back"
             " to not using it if not present."
             " If True and cuDNN can not be used, raise an error."
             " If False, disable cudnn",
             EnumStr("auto", "True", "False"),
             in_c_key=False)

# This flag determines whether or not to raise error/warning message if
# there is a CPU Op in the computational graph.
AddConfigVar(
    'assert_no_cpu_op',
    "Raise an error/warning if there is a CPU op in the computational graph.",
    EnumStr('ignore', 'warn', 'raise', 'pdb', allow_override=True),
    in_c_key=False)


# Do not add FAST_RUN_NOGC to this list (nor any other ALL CAPS shortcut).
# The way to get FAST_RUN_NOGC is with the flag 'linker=c|py_nogc'.
# The old all capital letter way of working is deprecated as it is not
# scalable.
# Also, please be careful not to modify the first item in the enum when adding
# new modes, since it is the default mode.
AddConfigVar(
    'mode',
    "Default compilation mode",
    EnumStr('Mode', 'DebugMode', 'FAST_RUN',
            'NanGuardMode',
            'FAST_COMPILE', 'DEBUG_MODE'),
    in_c_key=False)

param = "g++"

# Test whether or not g++ is present: disable C code if it is not.
try:
    rc = call_subprocess_Popen(['g++', '-v'])
except OSError:
    rc = 1

# Anaconda on Windows has mingw-w64 packages including GCC, but it may not be on PATH.
if rc != 0:
    if sys.platform == "win32":
        mingw_w64_gcc = os.path.join(os.path.dirname(sys.executable), "Library", "mingw-w64", "bin", "g++")
        try:
            rc = call_subprocess_Popen([mingw_w64_gcc, '-v'])
            if rc == 0:
                maybe_add_to_os_environ_pathlist('PATH', os.path.dirname(mingw_w64_gcc))
        except OSError:
            rc = 1
        if rc != 0:
            _logger.warning("g++ not available, if using conda: `conda install m2w64-toolchain`")

if rc != 0:
    param = ""

# On Mac we test for 'clang++' and use it by default
if sys.platform == 'darwin':
    try:
        rc = call_subprocess_Popen(['clang++', '-v'])
        if rc == 0:
            param = "clang++"
    except OSError:
        pass

# Try to find the full compiler path from the name
if param != "":
    import distutils.spawn
    newp = distutils.spawn.find_executable(param)
    if newp is not None:
        param = newp
    del newp
    del distutils

# to support path that includes spaces, we need to wrap it with double quotes on Windows
if param and os.name == 'nt':
    param = '"%s"' % param


def warn_cxx(val):
    """We only support clang++ as otherwise we hit strange g++/OSX bugs."""
    if sys.platform == 'darwin' and 'clang++' not in val:
        _logger.warning("Only clang++ is supported. With g++,"
                        " we end up with strange g++/OSX bugs.")
    return True

AddConfigVar('cxx',
             "The C++ compiler to use. Currently only g++ is"
             " supported, but supporting additional compilers should not be "
             "too difficult. "
             "If it is empty, no C++ code is compiled.",
             StrParam(param, is_valid=warn_cxx),
             in_c_key=False)
del param

if rc == 0 and config.cxx != "":
    # Keep the default linker the same as the one for the mode FAST_RUN
    AddConfigVar('linker',
                 "Default linker used if the theano flags mode is Mode",
                 EnumStr('cvm', 'c|py', 'py', 'c', 'c|py_nogc',
                         'vm', 'vm_nogc', 'cvm_nogc'),
                 in_c_key=False)
else:
    # g++ is not present or the user disabled it,
    # linker should default to python only.
    AddConfigVar('linker',
                 "Default linker used if the theano flags mode is Mode",
                 EnumStr('vm', 'py', 'vm_nogc'),
                 in_c_key=False)
    if type(config).cxx.is_default:
        # If the user provided an empty value for cxx, do not warn.
        _logger.warning(
            'g++ not detected ! Theano will be unable to execute '
            'optimized C-implementations (for both CPU and GPU) and will '
            'default to Python implementations. Performance will be severely '
            'degraded. To remove this warning, set Theano flags cxx to an '
            'empty string.')


# Keep the default value the same as the one for the mode FAST_RUN
AddConfigVar('allow_gc',
             "Do we default to delete intermediate results during Theano"
             " function calls? Doing so lowers the memory requirement, but"
             " asks that we reallocate memory at the next function call."
             " This is implemented for the default linker, but may not work"
             " for all linkers.",
             BoolParam(True),
             in_c_key=False)

# Keep the default optimizer the same as the one for the mode FAST_RUN
AddConfigVar(
    'optimizer',
    "Default optimizer. If not None, will use this optimizer with the Mode",
    EnumStr('fast_run', 'merge', 'fast_compile', 'None'),
    in_c_key=False)

AddConfigVar('optimizer_verbose',
             "If True, we print all optimization being applied",
             BoolParam(False),
             in_c_key=False)

AddConfigVar(
    'on_opt_error',
    ("What to do when an optimization crashes: warn and skip it, raise "
     "the exception, or fall into the pdb debugger."),
    EnumStr('warn', 'raise', 'pdb', 'ignore'),
    in_c_key=False)


def safe_no_home(home):
    """
    Make sure the user is not attempting to use `config.home`.

    This config option was removed in Thenao 0.5 since it was redundant with
    `config.base_compiledir`. This filter function ensures people who were
    setting the location of their compilation directory through `config.home`
    switch to `config.basecompiledir` instead, by raising an error when
    `config.home` is used.
    """
    if home:
        raise RuntimeError(
            'The `config.home` option has been removed and should not be '
            'used anymore. Please set the `config.base_compiledir` option '
            'instead (for instance to: %s)' %
            os.path.join(home, '.theano'))
    return True


AddConfigVar(
    'home',
    "This config option was removed in 0.5: do not use it!",
    ConfigParam('', allow_override=False, filter=safe_no_home),
    in_c_key=False)


AddConfigVar(
    'nocleanup',
    "Suppress the deletion of code files that did not compile cleanly",
    BoolParam(False),
    in_c_key=False)

AddConfigVar('on_unused_input',
             "What to do if a variable in the 'inputs' list of "
             " theano.function() is not used in the graph.",
             EnumStr('raise', 'warn', 'ignore'),
             in_c_key=False)

# This flag is used when we import Theano to initialize global variables.
# So changing it after import will not modify these global variables.
# This could be done differently... but for now we simply prevent it from being
# changed at runtime.
AddConfigVar(
    'tensor.cmp_sloppy',
    "Relax tensor._allclose (0) not at all, (1) a bit, (2) more",
    IntParam(0, lambda i: i in (0, 1, 2), allow_override=False),
    in_c_key=False)

AddConfigVar(
    'tensor.local_elemwise_fusion',
    ("Enable or not in fast_run mode(fast_run optimization) the elemwise "
     "fusion optimization"),
    BoolParam(True),
    in_c_key=False)

AddConfigVar(
    'gpu.local_elemwise_fusion',
    ("Enable or not in fast_run mode(fast_run optimization) the gpu "
     "elemwise fusion optimization"),
    BoolParam(True),
    in_c_key=False)

# http://developer.amd.com/CPU/LIBRARIES/LIBM/Pages/default.aspx
AddConfigVar(
    'lib.amdlibm',
    "Use amd's amdlibm numerical library",
    BoolParam(False),
    # Added elsewhere in the c key only when needed.
    in_c_key=False)

AddConfigVar(
    'gpuelemwise.sync',
    "when true, wait that the gpu fct finished and check it error code.",
    BoolParam(True),
    in_c_key=False)

AddConfigVar(
    'traceback.limit',
    "The number of stack to trace. -1 mean all.",
    # We default to a number to be able to know where v1 + v2 is created in the
    # user script. The bigger this number is, the more run time it takes.
    # We need to default to 8 to support theano.tensor.tensor(...).
    # import theano, numpy
    # X = theano.tensor.matrix()
    # y = X.reshape((5,3,1))
    # assert y.tag.trace
    IntParam(8),
    in_c_key=False)

AddConfigVar(
    'traceback.compile_limit',
    "The number of stack to trace to keep during compilation. -1 mean all."
    " If greater then 0, will also make us save Theano internal stack trace.",
    IntParam(0),
    in_c_key=False)

AddConfigVar('experimental.unpickle_gpu_on_cpu',
             "Allow unpickling of pickled CudaNdarrays as numpy.ndarrays."
             "This is useful, if you want to open a CudaNdarray without "
             "having cuda installed."
             "If you have cuda installed, this will force unpickling to"
             "be done on the cpu to numpy.ndarray."
             "Please be aware that this may get you access to the data,"
             "however, trying to unpicke gpu functions will not succeed."
             "This flag is experimental and may be removed any time, when"
             "gpu<>cpu transparency is solved.",
             BoolParam(default=False),
             in_c_key=False)

AddConfigVar('numpy.seterr_all',
             ("Sets numpy's behaviour for floating-point errors, ",
              "see numpy.seterr. "
              "'None' means not to change numpy's default, which can be "
              "different for different numpy releases. "
              "This flag sets the default behaviour for all kinds of floating-"
              "point errors, its effect can be overriden for specific errors "
              "by the following flags: seterr_divide, seterr_over, "
              "seterr_under and seterr_invalid."),
             EnumStr('ignore', 'warn', 'raise', 'call', 'print', 'log', 'None',
                     allow_override=False),
             in_c_key=False)

AddConfigVar('numpy.seterr_divide',
             ("Sets numpy's behavior for division by zero, see numpy.seterr. "
              "'None' means using the default, defined by numpy.seterr_all."),
             EnumStr('None', 'ignore', 'warn', 'raise', 'call', 'print', 'log',
                     allow_override=False),
             in_c_key=False)

AddConfigVar('numpy.seterr_over',
             ("Sets numpy's behavior for floating-point overflow, "
              "see numpy.seterr. "
              "'None' means using the default, defined by numpy.seterr_all."),
             EnumStr('None', 'ignore', 'warn', 'raise', 'call', 'print', 'log',
                     allow_override=False),
             in_c_key=False)

AddConfigVar('numpy.seterr_under',
             ("Sets numpy's behavior for floating-point underflow, "
              "see numpy.seterr. "
              "'None' means using the default, defined by numpy.seterr_all."),
             EnumStr('None', 'ignore', 'warn', 'raise', 'call', 'print', 'log',
                     allow_override=False),
             in_c_key=False)

AddConfigVar('numpy.seterr_invalid',
             ("Sets numpy's behavior for invalid floating-point operation, "
              "see numpy.seterr. "
              "'None' means using the default, defined by numpy.seterr_all."),
             EnumStr('None', 'ignore', 'warn', 'raise', 'call', 'print', 'log',
                     allow_override=False),
             in_c_key=False)

###
# To disable some warning about old bug that are fixed now.
###
AddConfigVar('warn.ignore_bug_before',
             ("If 'None', we warn about all Theano bugs found by default. "
              "If 'all', we don't warn about Theano bugs found by default. "
              "If a version, we print only the warnings relative to Theano "
              "bugs found after that version. "
              "Warning for specific bugs can be configured with specific "
              "[warn] flags."),
             EnumStr('0.7', 'None', 'all', '0.3', '0.4', '0.4.1', '0.5', '0.6',
                     '0.7', '0.8', '0.8.1', '0.8.2', '0.9',
                     allow_override=False),
             in_c_key=False)


def warn_default(version):
    """
    Return True iff we should warn about bugs fixed after a given version.
    """
    if config.warn.ignore_bug_before == 'None':
        return True
    if config.warn.ignore_bug_before == 'all':
        return False
    if config.warn.ignore_bug_before >= version:
        return False
    return True


AddConfigVar('warn.argmax_pushdown_bug',
             ("Warn if in past version of Theano we generated a bug with the "
              "theano.tensor.nnet.nnet.local_argmax_pushdown optimization. "
              "Was fixed 27 may 2010"),
             BoolParam(warn_default('0.3')),
             in_c_key=False)

AddConfigVar('warn.gpusum_01_011_0111_bug',
             ("Warn if we are in a case where old version of Theano had a "
              "silent bug with GpuSum pattern 01,011 and 0111 when the first "
              "dimensions was bigger then 4096. Was fixed 31 may 2010"),
             BoolParam(warn_default('0.3')),
             in_c_key=False)

AddConfigVar('warn.sum_sum_bug',
             ("Warn if we are in a case where Theano version between version "
              "9923a40c7b7a and the 2 august 2010 (fixed date), generated an "
              "error in that case. This happens when there are 2 consecutive "
              "sums in the graph, bad code was generated. "
              "Was fixed 2 August 2010"),
             BoolParam(warn_default('0.3')),
             in_c_key=False)

AddConfigVar('warn.sum_div_dimshuffle_bug',
             ("Warn if previous versions of Theano (between rev. "
              "3bd9b789f5e8, 2010-06-16, and cfc6322e5ad4, 2010-08-03) "
              "would have given incorrect result. This bug was triggered by "
              "sum of division of dimshuffled tensors."),
             BoolParam(warn_default('0.3')),
             in_c_key=False)

AddConfigVar(
    'warn.subtensor_merge_bug',
    "Warn if previous versions of Theano (before 0.5rc2) could have given "
    "incorrect results when indexing into a subtensor with negative "
    "stride (for instance, for instance, x[a:b:-1][c]).",
    BoolParam(warn_default('0.5')),
    in_c_key=False)

AddConfigVar(
    'warn.gpu_set_subtensor1',
    "Warn if previous versions of Theano (before 0.6) could have given "
    "incorrect results when moving to the gpu "
    "set_subtensor(x[int vector], new_value)",
    BoolParam(warn_default('0.6')),
    in_c_key=False)

AddConfigVar(
    'warn.vm_gc_bug',
    "There was a bug that existed in the default Theano configuration,"
    " only in the development version between July 5th 2012"
    " and July 30th 2012. This was not in a released version."
    " If your code was affected by this bug, a warning"
    " will be printed during the code execution if you use the"
    " `linker=vm,vm.lazy=True,warn.vm_gc_bug=True` Theano flags."
    " This warning is disabled by default as the bug was not released.",
    BoolParam(False),
    in_c_key=False)

AddConfigVar('warn.signal_conv2d_interface',
             ("Warn we use the new signal.conv2d() when its interface"
              " changed mid June 2014"),
             BoolParam(warn_default('0.7')),
             in_c_key=False)

AddConfigVar('warn.reduce_join',
             ('Your current code is fine, but Theano versions '
              'prior to 0.7 (or this development version) '
              'might have given an incorrect result. '
              'To disable this warning, set the Theano flag '
              'warn.reduce_join to False. The problem was an '
              'optimization, that modified the pattern '
              '"Reduce{scalar.op}(Join(axis=0, a, b), axis=0)", '
              'did not check the reduction axis. So if the '
              'reduction axis was not 0, you got a wrong answer.'),
             BoolParam(warn_default('0.7')),
             in_c_key=False)

AddConfigVar('warn.inc_set_subtensor1',
             ('Warn if previous versions of Theano (before 0.7) could have '
              'given incorrect results for inc_subtensor and set_subtensor '
              'when using some patterns of advanced indexing (indexing with '
              'one vector or matrix of ints).'),
             BoolParam(warn_default('0.7')),
             in_c_key=False)

AddConfigVar('warn.round',
             "Round changed its default from Seed to use for randomized unit tests. "
             "Special value 'random' means using a seed of None.",
             BoolParam(warn_default('0.9')),
             in_c_key=False)


AddConfigVar(
    'compute_test_value',
    ("If 'True', Theano will run each op at graph build time, using "
     "Constants, SharedVariables and the tag 'test_value' as inputs "
     "to the function. This helps the user track down problems in the "
     "graph before it gets optimized."),
    EnumStr('off', 'ignore', 'warn', 'raise', 'pdb'),
    in_c_key=False)


AddConfigVar(
    'print_test_value',
    ("If 'True', the __eval__ of a Theano variable will return its test_value "
     "when this is available. This has the practical conseguence that, e.g., "
     "in debugging `my_var` will print the same as `my_var.tag.test_value` "
     "when a test value is defined."),
    BoolParam(False),
    in_c_key=False)


AddConfigVar('compute_test_value_opt',
             ("For debugging Theano optimization only."
              " Same as compute_test_value, but is used"
              " during Theano optimization"),
             EnumStr('off', 'ignore', 'warn', 'raise', 'pdb'),
             in_c_key=False)

AddConfigVar('unpickle_function',
             ("Replace unpickled Theano functions with None. "
              "This is useful to unpickle old graphs that pickled"
              " them when it shouldn't"),
             BoolParam(True),
             in_c_key=False)

AddConfigVar(
    'reoptimize_unpickled_function',
    "Re-optimize the graph when a theano function is unpickled from the disk.",
    BoolParam(False, allow_override=True),
    in_c_key=False)

"""Note to developers:
    Generally your exceptions should use an apply node's __str__
    method when exception_verbosity == 'low'. When exception_verbosity
    == 'high', you should include a call to printing.min_informative_str
    on all important apply nodes.
"""
AddConfigVar(
    'exception_verbosity',
    "If 'low', the text of exceptions will generally refer "
    "to apply nodes with short names such as "
    "Elemwise{add_no_inplace}. If 'high', some exceptions "
    "will also refer to apply nodes with long descriptions "
    """ like:
    A. Elemwise{add_no_inplace}
            B. log_likelihood_v_given_h
            C. log_likelihood_h""",
    EnumStr('low', 'high'),
    in_c_key=False)

# Test if the env variable is set
var = os.getenv('OMP_NUM_THREADS', None)
if var:
    try:
        int(var)
    except ValueError:
        raise TypeError("The environment variable OMP_NUM_THREADS"
                        " should be a number, got '%s'." % var)
    else:
        default_openmp = not int(var) == 1
else:
    # Check the number of cores availables.
    count = cpuCount()
    if count == -1:
        _logger.warning("We are not able to detect the number of CPU cores."
                        " We disable openmp by default. To remove this"
                        " warning, set the environment variable"
                        " OMP_NUM_THREADS to the number of threads you"
                        " want theano to use.")
    default_openmp = count > 1

# Disable it by default for now as currently only the ConvOp supports
# it, and this causes slowdown by default as we do not disable it for
# too small convolution.
default_openmp = False

AddConfigVar('openmp',
             "Allow (or not) parallel computation on the CPU with OpenMP. "
             "This is the default value used when creating an Op that "
             "supports OpenMP parallelization. It is preferable to define it "
             "via the Theano configuration file ~/.theanorc or with the "
             "environment variable THEANO_FLAGS. Parallelization is only "
             "done for some operations that implement it, and even for "
             "operations that implement parallelism, each operation is free "
             "to respect this flag or not. You can control the number of "
             "threads used with the environment variable OMP_NUM_THREADS."
             " If it is set to 1, we disable openmp in Theano by default.",
             BoolParam(default_openmp),
             in_c_key=False,
             )

AddConfigVar('openmp_elemwise_minsize',
             "If OpenMP is enabled, this is the minimum size of vectors "
             "for which the openmp parallelization is enabled "
             "in element wise ops.",
             IntParam(200000),
             in_c_key=False,
             )

AddConfigVar(
    'check_input',
    "Specify if types should check their input in their C code. "
    "It can be used to speed up compilation, reduce overhead "
    "(particularly for scalars) and reduce the number of generated C "
    "files.",
    BoolParam(True),
    in_c_key=True)

AddConfigVar(
    'cache_optimizations',
    "WARNING: work in progress, does not work yet. "
    "Specify if the optimization cache should be used. This cache will "
    "any optimized graph and its optimization. Actually slow downs a lot "
    "the first optimization, and could possibly still contains some bugs. "
    "Use at your own risks.",
    BoolParam(False),
    in_c_key=False)


def good_seed_param(seed):
    if seed == "random":
        return True
    try:
        int(seed)
    except Exception:
        return False
    return True


AddConfigVar('unittests.rseed',
             "Seed to use for randomized unit tests. "
             "Special value 'random' means using a seed of None.",
             StrParam(666, is_valid=good_seed_param),
             in_c_key=False)

AddConfigVar('NanGuardMode.nan_is_error',
             "Default value for nan_is_error",
             BoolParam(True),
             in_c_key=False)

AddConfigVar('NanGuardMode.inf_is_error',
             "Default value for inf_is_error",
             BoolParam(True),
             in_c_key=False)

AddConfigVar('NanGuardMode.big_is_error',
             "Default value for big_is_error",
             BoolParam(True),
             in_c_key=False)

AddConfigVar('NanGuardMode.action',
             "What NanGuardMode does when it finds a problem",
             EnumStr('raise', 'warn', 'pdb'),
             in_c_key=False)

AddConfigVar('optimizer_excluding',
             ("When using the default mode, we will remove optimizer with "
              "these tags. Separate tags with ':'."),
             StrParam("", allow_override=False),
             in_c_key=False)

AddConfigVar('optimizer_including',
             ("When using the default mode, we will add optimizer with "
              "these tags. Separate tags with ':'."),
             StrParam("", allow_override=False),
             in_c_key=False)

AddConfigVar('optimizer_requiring',
             ("When using the default mode, we will require optimizer with "
              "these tags. Separate tags with ':'."),
             StrParam("", allow_override=False),
             in_c_key=False)

AddConfigVar('DebugMode.patience',
             "Optimize graph this many times to detect inconsistency",
             IntParam(10, lambda i: i > 0),
             in_c_key=False)

AddConfigVar('DebugMode.check_c',
             "Run C implementations where possible",
             BoolParam(
                 lambda: bool(theano.config.cxx)),
             in_c_key=False)

AddConfigVar('DebugMode.check_py',
             "Run Python implementations where possible",
             BoolParam(True),
             in_c_key=False)

AddConfigVar('DebugMode.check_finite',
             "True -> complain about NaN/Inf results",
             BoolParam(True),
             in_c_key=False)

AddConfigVar('DebugMode.check_strides',
             ("Check that Python- and C-produced ndarrays have same strides. "
              "On difference: (0) - ignore, (1) warn, or (2) raise error"),
             IntParam(0, lambda i: i in (0, 1, 2)),
             in_c_key=False)

AddConfigVar('DebugMode.warn_input_not_reused',
             ("Generate a warning when destroy_map or view_map says that an "
              "op works inplace, but the op did not reuse the input for its "
              "output."),
             BoolParam(True),
             in_c_key=False)


def is_valid_check_preallocated_output_param(param):
    if not isinstance(param, string_types):
        return False
    valid = ["initial", "previous", "c_contiguous", "f_contiguous",
             "strided", "wrong_size", "ALL", ""]
    for p in param.split(":"):
        if p not in valid:
            return False
    return True

AddConfigVar('DebugMode.check_preallocated_output',
             ('Test thunks with pre-allocated memory as output storage. '
              'This is a list of strings separated by ":". Valid values are: '
              '"initial" (initial storage in storage map, happens with Scan),'
              '"previous" (previously-returned memory), '
              '"c_contiguous", "f_contiguous", '
              '"strided" (positive and negative strides), '
              '"wrong_size" (larger and smaller dimensions), and '
              '"ALL" (all of the above).'),
             StrParam('', is_valid=is_valid_check_preallocated_output_param),
             in_c_key=False)

AddConfigVar('DebugMode.check_preallocated_output_ndim',
             ('When testing with "strided" preallocated output memory, '
              'test all combinations of strides over that number of '
              '(inner-most) dimensions. You may want to reduce that number '
              'to reduce memory or time usage, but it is advised to keep a '
              'minimum of 2.'),
             IntParam(4, lambda i: i > 0),
             in_c_key=False)

AddConfigVar('profiling.time_thunks',
             """Time individual thunks when profiling""",
             BoolParam(True),
             in_c_key=False)

AddConfigVar('profiling.n_apply',
             "Number of Apply instances to print by default",
             IntParam(20, lambda i: i > 0),
             in_c_key=False)

AddConfigVar('profiling.n_ops',
             "Number of Ops to print by default",
             IntParam(20, lambda i: i > 0),
             in_c_key=False)

AddConfigVar('profiling.output_line_width',
             "Max line width for the profiling output",
             IntParam(512, lambda i: i > 0),
             in_c_key=False)

AddConfigVar('profiling.min_memory_size',
             """For the memory profile, do not print Apply nodes if the size
             of their outputs (in bytes) is lower than this threshold""",
             IntParam(1024, lambda i: i >= 0),
             in_c_key=False)

AddConfigVar('profiling.min_peak_memory',
             """The min peak memory usage of the order""",
             BoolParam(False),
             in_c_key=False)

AddConfigVar('profiling.destination',
             """
             File destination of the profiling output
             """,
             StrParam('stderr'),
             in_c_key=False)

AddConfigVar('profiling.debugprint',
             """
             Do a debugprint of the profiled functions
             """,
             BoolParam(False),
             in_c_key=False)

AddConfigVar('profiling.ignore_first_call',
             """
             Do we ignore the first call of a Theano function.
             """,
             BoolParam(False),
             in_c_key=False)

AddConfigVar('optdb.position_cutoff',
             'Where to stop eariler during optimization. It represent the'
             ' position of the optimizer where to stop.',
             FloatParam(numpy.inf),
             in_c_key=False)

AddConfigVar('optdb.max_use_ratio',
             'A ratio that prevent infinite loop in EquilibriumOptimizer.',
             FloatParam(8),
             in_c_key=False)

AddConfigVar('gcc.cxxflags',
             "Extra compiler flags for gcc",
             StrParam(""),
             # Added elsewhere in the c key only when needed.
             in_c_key=False)

AddConfigVar('cmodule.warn_no_version',
             "If True, will print a warning when compiling one or more Op "
             "with C code that can't be cached because there is no "
             "c_code_cache_version() function associated to at least one of "
             "those Ops.",
             BoolParam(False),
             in_c_key=False)

AddConfigVar('cmodule.remove_gxx_opt',
             "If True, will remove the -O* parameter passed to g++."
             "This is useful to debug in gdb modules compiled by Theano."
             "The parameter -g is passed by default to g++",
             BoolParam(False),
             # TODO: change so that this isn't needed.
             # This can be done by handing this in compile_args()
             in_c_key=True)

AddConfigVar('cmodule.compilation_warning',
             "If True, will print compilation warnings.",
             BoolParam(False),
             in_c_key=False)


AddConfigVar('cmodule.preload_cache',
             "If set to True, will preload the C module cache at import time",
             BoolParam(False, allow_override=False),
             in_c_key=False)

AddConfigVar('cmodule.age_thresh_use',
             "In seconds. The time after which "
             "Theano won't reuse a compile c module.",
             # 24 days
             IntParam(60 * 60 * 24 * 24, allow_override=False),
             in_c_key=False)


def default_blas_ldflags():
    global numpy
    warn_record = []
    try:
        if (hasattr(numpy.distutils, '__config__') and
                numpy.distutils.__config__):
            # If the old private interface is available use it as it
            # don't print information to the user.
            blas_info = numpy.distutils.__config__.blas_opt_info
        else:
            # We do this import only here, as in some setup, if we
            # just import theano and exit, with the import at global
            # scope, we get this error at exit: "Exception TypeError:
            # "'NoneType' object is not callable" in <bound method
            # Popen.__del__ of <subprocess.Popen object at 0x21359d0>>
            # ignored"

            # This happen with Python 2.7.3 |EPD 7.3-1 and numpy 1.8.1
            import numpy.distutils.system_info  # noqa

            # We need to catch warnings as in some cases NumPy print
            # stuff that we don't want the user to see.
            # I'm not able to remove all printed stuff
            with warnings.catch_warnings(record=True):
                numpy.distutils.system_info.system_info.verbosity = 0
                blas_info = numpy.distutils.system_info.get_info("blas_opt")

        # If we are in a EPD installation, mkl is available
        if "EPD" in sys.version:
            use_unix_epd = True
            if sys.platform == 'win32':
                return ' '.join(
                    ['-L"%s"' % os.path.join(sys.prefix, "Scripts")] +
                    # Why on Windows, the library used are not the
                    # same as what is in
                    # blas_info['libraries']?
                    ['-l%s' % l for l in ["mk2_core", "mk2_intel_thread",
                                          "mk2_rt"]])
            elif sys.platform == 'darwin':
                # The env variable is needed to link with mkl
                new_path = os.path.join(sys.prefix, "lib")
                v = os.getenv("DYLD_FALLBACK_LIBRARY_PATH", None)
                if v is not None:
                    # Explicit version could be replaced by a symbolic
                    # link called 'Current' created by EPD installer
                    # This will resolve symbolic links
                    v = os.path.realpath(v)

                # The python __import__ don't seam to take into account
                # the new env variable "DYLD_FALLBACK_LIBRARY_PATH"
                # when we set with os.environ['...'] = X or os.putenv()
                # So we warn the user and tell him what todo.
                if v is None or new_path not in v.split(":"):
                    _logger.warning(
                        "The environment variable "
                        "'DYLD_FALLBACK_LIBRARY_PATH' does not contain "
                        "the '%s' path in its value. This will make "
                        "Theano use a slow version of BLAS. Update "
                        "'DYLD_FALLBACK_LIBRARY_PATH' to contain the "
                        "said value, this will disable this warning."
                        % new_path)

                    use_unix_epd = False
            if use_unix_epd:
                return ' '.join(
                    ['-L%s' % os.path.join(sys.prefix, "lib")] +
                    ['-l%s' % l for l in blas_info['libraries']])

                # Canopy
        if "Canopy" in sys.prefix:
            subsub = 'lib'
            if sys.platform == 'win32':
                subsub = 'Scripts'
            lib_path = os.path.join(sys.base_prefix, subsub)
            if not os.path.exists(lib_path):
                # Old logic to find the path. I don't think we still
                # need it, but I don't have the time to test all
                # installation configuration. So I keep this as a fall
                # back in case the current expectation don't work.

                # This old logic don't work when multiple version of
                # Canopy is installed.
                p = os.path.join(sys.base_prefix, "..", "..", "appdata")
                assert os.path.exists(p), "Canopy changed the location of MKL"
                lib_paths = os.listdir(p)
                # Try to remove subdir that can't contain MKL
                for sub in lib_paths:
                    if not os.path.exists(os.path.join(p, sub, subsub)):
                        lib_paths.remove(sub)
                assert len(lib_paths) == 1, (
                    "Unexpected case when looking for Canopy MKL libraries",
                    p, lib_paths, [os.listdir(os.path.join(p, sub))
                                   for sub in lib_paths])
                lib_path = os.path.join(p, lib_paths[0], subsub)
                assert os.path.exists(lib_path), "Canopy changed the location of MKL"

            if sys.platform == "linux2" or sys.platform == "darwin":
                return ' '.join(
                    ['-L%s' % lib_path] +
                    ['-l%s' % l for l in blas_info['libraries']])
            elif sys.platform == 'win32':
                return ' '.join(
                    ['-L"%s"' % lib_path] +
                    # Why on Windows, the library used are not the
                    # same as what is in blas_info['libraries']?
                    ['-l%s' % l for l in ["mk2_core", "mk2_intel_thread",
                                          "mk2_rt"]])

        # MKL
        # If mkl can be imported then use it. On conda:
        # "conda install mkl-service" installs the Python wrapper and
        # the low-level C libraries as well as optimised version of
        # numpy and scipy.
        try:
            import mkl  # noqa
        except ImportError as e:
            if any([m for m in ('conda', 'Continuum') if m in sys.version]):
                warn_record.append(('install mkl with `conda install mkl-service`: %s', e))
        else:
            # This branch is executed if no exception was raised
            if sys.platform == "win32":
                lib_path = [os.path.join(sys.prefix, 'Library', 'bin')]
                flags = ['-L"%s"' % lib_path]
            else:
                lib_path = blas_info.get('library_dirs', [])
                flags = []
                if lib_path:
                    flags = ['-L%s' % lib_path[0]]
            flags += ['-l%s' % l for l in ["mkl_core",
                                           "mkl_intel_thread",
                                           "mkl_rt"]]
            res = try_blas_flag(flags)
            if res:
                return res
            flags.extend(['-Wl,-rpath,' + l for l in
                          blas_info.get('library_dirs', [])])
            res = try_blas_flag(flags)
            if res:
                maybe_add_to_os_environ_pathlist('PATH', lib_path[0])
                return res

        # to support path that includes spaces, we need to wrap it with double quotes on Windows
        path_wrapper = "\"" if os.name == 'nt' else ""
        ret = (
            # TODO: the Gemm op below should separate the
            # -L and -l arguments into the two callbacks
            # that CLinker uses for that stuff.  for now,
            # we just pass the whole ldflags as the -l
            # options part.
            ['-L%s%s%s' % (path_wrapper, l, path_wrapper) for l in blas_info.get('library_dirs', [])] +
            ['-l%s' % l for l in blas_info.get('libraries', [])] +
            blas_info.get('extra_link_args', []))
        # For some very strange reason, we need to specify -lm twice
        # to get mkl to link correctly.  I have no idea why.
        if any('mkl' in fl for fl in ret):
            ret.extend(['-lm', '-lm'])
        res = try_blas_flag(ret)
        if res:
            return res

        # If we are using conda and can't reuse numpy blas, then doing
        # the fallback and test -lblas could give slow computation, so
        # warn about this.
        for warn in warn_record:
            _logger.warning(*warn)
        del warn_record

        # Some environment don't have the lib dir in LD_LIBRARY_PATH.
        # So add it.
        ret.extend(['-Wl,-rpath,' + l for l in
                    blas_info.get('library_dirs', [])])
        res = try_blas_flag(ret)
        if res:
            return res

        # Add sys.prefix/lib to the runtime search path. On
        # non-system installations of Python that use the
        # system linker, this is generally neccesary.
        if sys.platform in ("linux", "darwin"):
            lib_path = os.path.join(sys.prefix, 'lib')
            ret.append('-Wl,-rpath,' + lib_path)
            res = try_blas_flag(ret)
            if res:
                return res

    except KeyError:
        pass

    # Even if we could not detect what was used for numpy, or if these
    # libraries are not found, most Linux systems have a libblas.so
    # readily available. We try to see if that's the case, rather
    # than disable blas. To test it correctly, we must load a program.
    # Otherwise, there could be problem in the LD_LIBRARY_PATH.
    return try_blas_flag(['-lblas'])


def try_blas_flag(flags):
    from theano.gof.cmodule import GCC_compiler
    test_code = textwrap.dedent("""\
        extern "C" double ddot_(int*, double*, int*, double*, int*);
        int main(int argc, char** argv)
        {
            int Nx = 5;
            int Sx = 1;
            double x[5] = {0, 1, 2, 3, 4};
            double r = ddot_(&Nx, x, &Sx, x, &Sx);

            if ((r - 30.) > 1e-6 || (r - 30.) < -1e-6)
            {
                return -1;
            }
            return 0;
        }
        """)
    cflags = list(flags)
    # to support path that includes spaces, we need to wrap it with double quotes on Windows
    path_wrapper = "\"" if os.name == 'nt' else ""
    cflags.extend(['-L%s%s%s' % (path_wrapper, d, path_wrapper) for d in theano.gof.cmodule.std_lib_dirs()])

    res = GCC_compiler.try_compile_tmp(
        test_code, tmp_prefix='try_blas_',
        flags=cflags, try_run=True)
    # res[0]: shows successful compilation
    # res[1]: shows successful execution
    if res and res[0] and res[1]:
        return ' '.join(flags)
    else:
        return ""

AddConfigVar('blas.ldflags',
             "lib[s] to include for [Fortran] level-3 blas implementation",
             StrParam(default_blas_ldflags),
             # Added elsewhere in the c key only when needed.
             in_c_key=False)

AddConfigVar(
    'metaopt.verbose',
    "Enable verbose output for meta optimizers",
    theano.configparser.BoolParam(False),
    in_c_key=False)

AddConfigVar('profile',
             "If VM should collect profile information",
             BoolParam(False),
             in_c_key=False)

AddConfigVar('profile_optimizer',
             "If VM should collect optimizer profile information",
             BoolParam(False),
             in_c_key=False)

AddConfigVar('profile_memory',
             "If VM should collect memory profile information and print it",
             BoolParam(False),
             in_c_key=False)


def filter_vm_lazy(val):
    if val == 'False' or val is False:
        return False
    elif val == 'True' or val is True:
        return True
    elif val == 'None' or val is None:
        return None
    else:
        raise ValueError('Valid values for an vm.lazy parameter '
                         'should be None, False or True, not `%s`.' % val)

AddConfigVar('vm.lazy',
             "Useful only for the vm linkers. When lazy is None,"
             " auto detect if lazy evaluation is needed and use the apropriate"
             " version. If lazy is True/False, force the version used between"
             " Loop/LoopGC and Stack.",
             ConfigParam('None', filter_vm_lazy),
             in_c_key=False)

AddConfigVar(
    'warn.identify_1pexp_bug',
    'Warn if Theano versions prior to 7987b51 (2011-12-18) could have '
    'yielded a wrong result due to a bug in the is_1pexp function',
    BoolParam(warn_default('0.4.1')),
    in_c_key=False)

AddConfigVar('on_shape_error',
             "warn: print a warning and use the default"
             " value. raise: raise an error",
             theano.configparser.EnumStr("warn", "raise"),
             in_c_key=False)

AddConfigVar(
    'tensor.insert_inplace_optimizer_validate_nb',
    "-1: auto, if graph have less then 500 nodes 1, else 10",
    theano.configparser.IntParam(-1),
    in_c_key=False)

AddConfigVar('experimental.local_alloc_elemwise',
             "DEPRECATED: If True, enable the experimental"
             " optimization local_alloc_elemwise."
             " Generates error if not True. Use"
             " optimizer_excluding=local_alloc_elemwise"
             " to dsiable.",
             theano.configparser.BoolParam(
                 True,
                 is_valid=lambda x: x
             ),
             in_c_key=False)

# False could make the graph faster but not as safe.
AddConfigVar(
    'experimental.local_alloc_elemwise_assert',
    "When the local_alloc_elemwise is applied, add"
    " an assert to highlight shape errors.",
    theano.configparser.BoolParam(True),
    in_c_key=False)

AddConfigVar('scan.allow_gc',
             "Allow/disallow gc inside of Scan (default: False)",
             BoolParam(False),
             in_c_key=False)

AddConfigVar('scan.allow_output_prealloc',
             "Allow/disallow memory preallocation for outputs inside of scan "
             "(default: True)",
             BoolParam(True),
             in_c_key=False)

AddConfigVar('scan.debug',
             "If True, enable extra verbose output related to scan",
             BoolParam(False),
             in_c_key=False)

AddConfigVar('pycuda.init',
             """If True, always initialize PyCUDA when Theano want to
                initilize the GPU.  Currently, we must always initialize
                PyCUDA before Theano do it.  Setting this flag to True,
                ensure that, but always import PyCUDA.  It can be done
                manually by importing theano.misc.pycuda_init before theano
                initialize the GPU device.
                  """,
             BoolParam(False),
             in_c_key=False)

AddConfigVar('cublas.lib',
             """Name of the cuda blas library for the linker.""",
             StrParam('cublas'),
             # Added elsewhere in the c key only when needed.
             in_c_key=False)

AddConfigVar('lib.cnmem',
             """Do we enable CNMeM or not (a faster CUDA memory allocator).

             The parameter represent the start size (in MB or % of
             total GPU memory) of the memory pool.

             0: not enabled.
             0 < N <= 1: % of the total GPU memory (clipped to .985 for driver memory)
             > 0: use that number of MB of memory.

             """,
             # We should not mix both allocator, so we can't override
             FloatParam(0, lambda i: i >= 0, allow_override=False),
             in_c_key=False)

AddConfigVar('compile.wait',
             """Time to wait before retrying to aquire the compile lock.""",
             IntParam(5, lambda i: i > 0, allow_override=False),
             in_c_key=False)


def _timeout_default():
    return theano.config.compile.wait * 24

AddConfigVar('compile.timeout',
             """In seconds, time that a process will wait before deciding to
override an existing lock. An override only happens when the existing
lock is held by the same owner *and* has not been 'refreshed' by this
owner for more than this period. Refreshes are done every half timeout
period for running processes.""",
             IntParam(_timeout_default, lambda i: i >= 0,
                      allow_override=False),
             in_c_key=False)


try:
    p_out = output_subprocess_Popen([config.cxx, '-dumpversion'])
    gcc_version_str = p_out[0].strip().decode()
except OSError:
    # Typically means gcc cannot be found.
    gcc_version_str = 'GCC_NOT_FOUND'


def local_bitwidth():
    """
    Return 32 for 32bit arch, 64 for 64bit arch.

    By "architecture", we mean the size of memory pointers (size_t in C),
    *not* the size of long int, as it can be different.

    """
    # Note that according to Python documentation, `platform.architecture()` is
    # not reliable on OS X with universal binaries.
    # Also, sys.maxsize does not exist in Python < 2.6.
    # 'P' denotes a void*, and the size is expressed in bytes.
    return struct.calcsize('P') * 8


def python_int_bitwidth():
    """
    Return the bit width of Python int (C long int).

    Note that it can be different from the size of a memory pointer.

    """
    # 'l' denotes a C long int, and the size is expressed in bytes.
    return struct.calcsize('l') * 8


compiledir_format_dict = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python_version": platform.python_version(),
    "python_bitwidth": local_bitwidth(),
    "python_int_bitwidth": python_int_bitwidth(),
    "theano_version": theano.__version__,
    "numpy_version": numpy.__version__,
    "gxx_version": gcc_version_str.replace(" ", "_"),
    "hostname": socket.gethostname()}


def short_platform(r=None, p=None):
    """
    Return a safe shorter version of platform.platform().

    The old default Theano compiledir used platform.platform in
    it. This use the platform.version() as a substring. This is too
    specific as it contain the full kernel number and package
    version. This cause the compiledir to change each time there is a
    new linux kernel update. This function remove the part of platform
    that are too precise.

    If we have something else then expected, we do nothing. So this
    should be safe on other OS.

    Some example if we use platform.platform() direction. On the same
    OS, with just some kernel updates.

    compiledir_Linux-2.6.32-504.el6.x86_64-x86_64-with-redhat-6.6-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.29.2.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.23.3.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.20.3.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.17.1.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.11.2.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-431.el6.x86_64-x86_64-with-redhat-6.5-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.23.2.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.6.2.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.6.1.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.2.1.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6-64
    compiledir_Linux-2.6.32-358.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-279.14.1.el6.x86_64-x86_64-with-redhat-6.4-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-279.14.1.el6.x86_64-x86_64-with-redhat-6.3-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-279.5.2.el6.x86_64-x86_64-with-redhat-6.3-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-220.13.1.el6.x86_64-x86_64-with-redhat-6.3-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-220.13.1.el6.x86_64-x86_64-with-redhat-6.2-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-220.7.1.el6.x86_64-x86_64-with-redhat-6.2-Santiago-x86_64-2.6.6
    compiledir_Linux-2.6.32-220.4.1.el6.x86_64-x86_64-with-redhat-6.2-Santiago-x86_64-2.6.6

    We suppose the version are ``X.Y[.*]-(digit)*(anything)*``. We keep ``X.Y``
    and don't keep less important digit in the part before ``-`` and we remove
    the leading digit after the first ``-``.

    If the information don't fit that pattern, we do not modify platform.

    """
    if r is None:
        r = platform.release()
    if p is None:
        p = platform.platform()
    sp = r.split('-')
    if len(sp) < 2:
        return p

    # For the split before the first -, we remove all learning digit:
    kernel_version = sp[0].split('.')
    if len(kernel_version) <= 2:
        # kernel version should always have at least 3 number.
        # If not, it use another semantic, so don't change it.
        return p
    sp[0] = '.'.join(kernel_version[:2])

    # For the split after the first -, we remove leading non-digit value.
    rest = sp[1].split('.')
    while len(rest):
        if rest[0].isdigit():
            del rest[0]
        else:
            break
    sp[1] = '.'.join(rest)

    # For sp[2:], we don't change anything.
    sr = '-'.join(sp)
    p = p.replace(r, sr)

    return p
compiledir_format_dict['short_platform'] = short_platform()
# Allow to have easily one compiledir per device.
compiledir_format_dict['device'] = config.device
compiledir_format_keys = ", ".join(sorted(compiledir_format_dict.keys()))
default_compiledir_format = ("compiledir_%(short_platform)s-%(processor)s-"
                             "%(python_version)s-%(python_bitwidth)s")

AddConfigVar("compiledir_format",
             textwrap.fill(textwrap.dedent("""\
                 Format string for platform-dependent compiled
                 module subdirectory (relative to base_compiledir).
                 Available keys: %s. Defaults to %r.
             """ % (compiledir_format_keys, default_compiledir_format))),
             StrParam(default_compiledir_format, allow_override=False),
             in_c_key=False)


def default_compiledirname():
    formatted = theano.config.compiledir_format % compiledir_format_dict
    safe = re.sub("[\(\)\s,]+", "_", formatted)
    return safe


def filter_base_compiledir(path):
    # Expand '~' in path
    return os.path.expanduser(str(path))


def filter_compiledir(path):
    # Expand '~' in path
    path = os.path.expanduser(path)
    # Turn path into the 'real' path. This ensures that:
    #   1. There is no relative path, which would fail e.g. when trying to
    #      import modules from the compile dir.
    #   2. The path is stable w.r.t. e.g. symlinks (which makes it easier
    #      to re-use compiled modules).
    path = os.path.realpath(path)
    if os.access(path, os.F_OK):  # Do it exist?
        if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
            # If it exist we need read, write and listing access
            raise ValueError(
                "compiledir '%s' exists but you don't have read, write"
                " or listing permissions." % path)
    else:
        try:
            os.makedirs(path, 0o770)  # read-write-execute for user and group
        except OSError as e:
            # Maybe another parallel execution of theano was trying to create
            # the same directory at the same time.
            if e.errno != errno.EEXIST:
                raise ValueError(
                    "Unable to create the compiledir directory"
                    " '%s'. Check the permissions." % path)

    # PROBLEM: sometimes the initial approach based on
    # os.system('touch') returned -1 for an unknown reason; the
    # alternate approach here worked in all cases... it was weird.
    # No error should happen as we checked the permissions.
    init_file = os.path.join(path, '__init__.py')
    if not os.path.exists(init_file):
        try:
            open(init_file, 'w').close()
        except IOError as e:
            if os.path.exists(init_file):
                pass  # has already been created
            else:
                e.args += ('%s exist? %s' % (path, os.path.exists(path)),)
                raise
    return path


def get_home_dir():
    """
    Return location of the user's home directory.

    """
    home = os.getenv('HOME')
    if home is None:
        # This expanduser usually works on Windows (see discussion on
        # theano-users, July 13 2010).
        home = os.path.expanduser('~')
        if home == '~':
            # This might happen when expanduser fails. Although the cause of
            # failure is a mystery, it has been seen on some Windows system.
            home = os.getenv('USERPROFILE')
    assert home is not None
    return home


# On Windows we should avoid writing temporary files to a directory that is
# part of the roaming part of the user profile. Instead we use the local part
# of the user profile, when available.
if sys.platform == 'win32' and os.getenv('LOCALAPPDATA') is not None:
    default_base_compiledir = os.path.join(os.getenv('LOCALAPPDATA'), 'Theano')
else:
    default_base_compiledir = os.path.join(get_home_dir(), '.theano')


AddConfigVar(
    'base_compiledir',
    "platform-independent root directory for compiled modules",
    ConfigParam(
        default_base_compiledir,
        filter=filter_base_compiledir,
        allow_override=False),
    in_c_key=False)


def default_compiledir():
    return os.path.join(
        theano.config.base_compiledir,
        default_compiledirname())

AddConfigVar(
    'compiledir',
    "platform-dependent cache directory for compiled modules",

    ConfigParam(
        default_compiledir,
        filter=filter_compiledir,
        allow_override=False),
    in_c_key=False)

# Check if there are remaining flags provided by the user through THEANO_FLAGS.
for key in THEANO_FLAGS_DICT.keys():
    warnings.warn('Theano does not recognise this flag: {0}'.format(key))
