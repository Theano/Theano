import os
import sys
import logging

import theano
from theano.configparser import (AddConfigVar, BoolParam, ConfigParam, EnumStr,
                                 IntParam, StrParam, TheanoConfigParser)
from theano.misc.cpucount import cpuCount
from theano.misc.windows import call_subprocess_Popen

_logger = logging.getLogger('theano.configdefaults')

config = TheanoConfigParser()

def floatX_convert(s):
    if s == "32":
        return "float32"
    elif s == "64":
        return "float64"
    else:
        return s

AddConfigVar('floatX',
             "Default floating-point precision for python casts",
             EnumStr('float64', 'float32', convert=floatX_convert,),
)

AddConfigVar('warn_float64',
             "Do an action when a tensor variable with float64 dtype is"
             " created. They can't be run on the GPU with the current(old)"
             " gpu back-end and are slow with gamer GPUs.",
             EnumStr('ignore', 'warn', 'raise', 'pdb'),
             in_c_key=False,
)

AddConfigVar('cast_policy',
        "Rules for implicit type casting",
        EnumStr('custom', 'numpy+floatX',
                # The 'numpy' policy was originally planned to provide a smooth
                # transition from numpy. It was meant to behave the same as
                # numpy+floatX, but keeping float64 when numpy would. However
                # the current implementation of some cast mechanisms makes it
                # a bit more complex to add than what was expected, so it is
                # currently not available.
                #numpy,
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
            if val.startswith('cpu') or val.startswith('gpu') \
                    or val.startswith('opencl') or val.startswith('cuda'):
                return val
            else:
                raise ValueError(('Invalid value ("%s") for configuration '
                                  'variable "%s". Valid options start with '
                                  'one of "cpu", "gpu", "opencl", "cuda"'
                                  % (val, self.fullname)))
        over = kwargs.get("allow_override", True)
        super(DeviceParam, self).__init__(default, filter, over)

    def __str__(self):
        return '%s (cpu, gpu*, opencl*, cuda*) ' % (self.fullname,)

AddConfigVar('device',
        ("Default device for computations. If gpu*, change the default to try "
         "to move computation to it and to put shared variable of float32 "
         "on it. Do not use upper case letters, only lower case even if "
         "NVIDIA use capital letters."),
        DeviceParam('cpu', allow_override=False),
        in_c_key=False,
        )

AddConfigVar('gpuarray.init_device',
             """
             Device to initialize for gpuarray use without moving
             computations automatically.
             """,
             StrParam(''),
             in_c_key=False)

AddConfigVar('init_gpu_device',
        ("Initialize the gpu device to use, works only if device=cpu. "
         "Unlike 'device', setting this option will NOT move computations, "
         "nor shared variables, to the specified GPU. "
         "It can be used to run GPU-specific tests on a particular GPU."),
        EnumStr('', 'gpu',
            'gpu0', 'gpu1', 'gpu2', 'gpu3',
            'gpu4', 'gpu5', 'gpu6', 'gpu7',
            'gpu8', 'gpu9', 'gpu10', 'gpu11',
            'gpu12', 'gpu13', 'gpu14', 'gpu15',
                allow_override=False),
        in_c_key=False)

AddConfigVar('force_device',
        "Raise an error if we can't use the specified device",
        BoolParam(False, allow_override=False),
        in_c_key=False)

AddConfigVar('print_active_device',
        "Print active device at when the GPU device is initialized.",
        BoolParam(True, allow_override=False),
        in_c_key=False)


# Do not add FAST_RUN_NOGC to this list (nor any other ALL CAPS shortcut).
# The way to get FAST_RUN_NOGC is with the flag 'linker=c|py_nogc'.
# The old all capital letter way of working is deprecated as it is not
# scalable.
# Also, please be careful not to modify the first item in the enum when adding
# new modes, since it is the default mode.
AddConfigVar('mode',
        "Default compilation mode",
        EnumStr('Mode', 'ProfileMode', 'DebugMode', 'FAST_RUN',
                'FAST_COMPILE', 'PROFILE_MODE', 'DEBUG_MODE'),
        in_c_key=False)

param = "g++"

# Test whether or not g++ is present: disable C code if it is not.
try:
    rc = call_subprocess_Popen(['g++', '-v'])
except OSError:
    param = ""
    rc = 1

# On Mac we test for 'clang++' and use it by default
if sys.platform == 'darwin':
    try:
        rc = call_subprocess_Popen(['clang++', '-v'])
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

AddConfigVar('cxx',
             "The C++ compiler to use. Currently only g++ is"
             " supported, but supporting additional compilers should not be "
             "too difficult. "
             "If it is empty, no C++ code is compiled.",
             StrParam(param),
             in_c_key=False)
del param

if rc == 0 and config.cxx != "":
    # Keep the default linker the same as the one for the mode FAST_RUN
    AddConfigVar('linker',
                 ("Default linker used if the theano flags mode is Mode "
                  "or ProfileMode(deprecated)"),
                 EnumStr('cvm', 'c|py', 'py', 'c', 'c|py_nogc',
                         'vm', 'vm_nogc', 'cvm_nogc'),
                 in_c_key=False)
else:
    # g++ is not present or the user disabled it,
    # linker should default to python only.
    AddConfigVar('linker',
                 ("Default linker used if the theano flags mode is Mode "
                  "or ProfileMode(deprecated)"),
                 EnumStr('vm', 'py', 'vm_nogc'),
                 in_c_key=False)
    try:
        # If the user provided an empty value for cxx, do not warn.
        theano.configparser.fetch_val_for_key('cxx')
    except KeyError:
        _logger.warning(
            'g++ not detected ! Theano will be unable to execute '
            'optimized C-implementations (for both CPU and GPU) and will '
            'default to Python implementations. Performance will be severely '
            'degraded. To remove this warning, set Theano flags cxx to an '
            'empty string.')


#Keep the default value the same as the one for the mode FAST_RUN
AddConfigVar('allow_gc',
             "Do we default to delete intermediate results during Theano"
             " function calls? Doing so lowers the memory requirement, but"
             " asks that we reallocate memory at the next function call."
             " This is implemented for the default linker, but may not work"
             " for all linkers.",
             BoolParam(True),
             in_c_key=False)

#Keep the default optimizer the same as the one for the mode FAST_RUN
AddConfigVar('optimizer',
        ("Default optimizer. If not None, will use this linker with the Mode "
         "object (not ProfileMode(deprecated) or DebugMode)"),
        EnumStr('fast_run', 'merge', 'fast_compile', 'None'),
        in_c_key=False)

AddConfigVar('optimizer_verbose',
             "If True, we print all optimization being applied",
             BoolParam(False),
             in_c_key=False)

AddConfigVar('on_opt_error',
        ("What to do when an optimization crashes: warn and skip it, raise "
         "the exception, or fall into the pdb debugger."),
        EnumStr('warn', 'raise', 'pdb'),
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


AddConfigVar('home',
        "This config option was removed in 0.5: do not use it!",
        ConfigParam('', allow_override=False, filter=safe_no_home),
        in_c_key=False)


AddConfigVar('nocleanup',
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
AddConfigVar('tensor.cmp_sloppy',
        "Relax tensor._allclose (0) not at all, (1) a bit, (2) more",
        IntParam(0, lambda i: i in (0, 1, 2), allow_override=False),
        in_c_key=False)

AddConfigVar('tensor.local_elemwise_fusion',
        ("Enable or not in fast_run mode(fast_run optimization) the elemwise "
         "fusion optimization"),
        BoolParam(True),
        in_c_key=False)

AddConfigVar('gpu.local_elemwise_fusion',
        ("Enable or not in fast_run mode(fast_run optimization) the gpu "
         "elemwise fusion optimization"),
        BoolParam(True),
        in_c_key=False)

#http://developer.amd.com/CPU/LIBRARIES/LIBM/Pages/default.aspx
AddConfigVar('lib.amdlibm',
        "Use amd's amdlibm numerical library",
        BoolParam(False))

AddConfigVar('gpuelemwise.sync',
        "when true, wait that the gpu fct finished and check it error code.",
        BoolParam(True),
        in_c_key=False)

AddConfigVar('traceback.limit',
             "The number of stack to trace. -1 mean all.",
# We default to 6 to be able to know where v1 + v2 is created in the
# user script. The bigger this number is, the more run time it takes.
# We need to default to 7 to support theano.tensor.tensor(...).
             IntParam(7),
             in_c_key=False)

AddConfigVar('experimental.mrg',
             "Another random number generator that work on the gpu",
             BoolParam(False))

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
### To disable some warning about old bug that are fixed now.
###
AddConfigVar('warn.ignore_bug_before',
             ("If 'None', we warn about all Theano bugs found by default. "
              "If 'all', we don't warn about Theano bugs found by default. "
              "If a version, we print only the warnings relative to Theano "
              "bugs found after that version. "
              "Warning for specific bugs can be configured with specific "
              "[warn] flags."),
             EnumStr('0.6', 'None', 'all', '0.3', '0.4', '0.4.1', '0.5', '0.7',
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

AddConfigVar('warn.subtensor_merge_bug',
        "Warn if previous versions of Theano (before 0.5rc2) could have given "
        "incorrect results when indexing into a subtensor with negative "
        "stride (for instance, for instance, x[a:b:-1][c]).",
        BoolParam(warn_default('0.5')),
        in_c_key=False)

AddConfigVar('warn.gpu_set_subtensor1',
        "Warn if previous versions of Theano (before 0.6) could have given "
        "incorrect results when moving to the gpu "
        "set_subtensor(x[int vector], new_value)",
        BoolParam(warn_default('0.6')),
        in_c_key=False)

AddConfigVar('warn.vm_gc_bug',
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

AddConfigVar('compute_test_value',
        ("If 'True', Theano will run each op at graph build time, using "
         "Constants, SharedVariables and the tag 'test_value' as inputs "
         "to the function. This helps the user track down problems in the "
         "graph before it gets optimized."),
        EnumStr('off', 'ignore', 'warn', 'raise', 'pdb'),
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

AddConfigVar('reoptimize_unpickled_function',
        "Re-optimize the graph when a theano function is unpickled from the disk.",
        BoolParam(True, allow_override=True),
        in_c_key=False)


"""Note to developers:
    Generally your exceptions should use an apply node's __str__
    method when exception_verbosity == 'low'. When exception_verbosity
    == 'high', you should include a call to printing.min_informative_str
    on all important apply nodes.
"""
AddConfigVar('exception_verbosity',
        "If 'low', the text of exceptions will generally refer " \
        + "to apply nodes with short names such as " \
        + "Elemwise{add_no_inplace}. If 'high', some exceptions " \
        + "will also refer to apply nodes with long descriptions " \
        + """ like:
        A. Elemwise{add_no_inplace}
                B. log_likelihood_v_given_h
                C. log_likelihood_h""",
        EnumStr('low', 'high'),
        in_c_key=False)

#Test if the env variable is set
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
    #Check the number of cores availables.
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

AddConfigVar('check_input',
             "Specify if types should check their input in their C code. "
             "It can be used to speed up compilation, reduce overhead "
              "(particularly for scalars) and reduce the number of generated C "
              "files.",
             BoolParam(True))

AddConfigVar('cache_optimizations',
             "WARNING: work in progress, does not work yet. "
             "Specify if the optimization cache should be used. This cache will "
             "any optimized graph and its optimization. Actually slow downs a lot "
             "the first optimization, and could possibly still contains some bugs. "
             "Use at your own risks.",
             BoolParam(False))
