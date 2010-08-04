import os
from theano.configparser import TheanoConfigParser, AddConfigVar, EnumStr, StrParam, IntParam, FloatParam, BoolParam

config = TheanoConfigParser()

AddConfigVar('floatX',
        "Default floating-point precision for python casts",
        EnumStr('float64', 'float32'), 
        )

#gpu mean let the driver select the gpu. Needed in case of gpu in exclusive mode.
#gpuX mean use the gpu number X.
AddConfigVar('device',
        "Default device for computations",
        EnumStr('cpu', 'gpu',*['gpu%i'%i for i in range(4)])
        )

AddConfigVar('mode',
        "Default compilation mode",
        EnumStr('Mode', 'ProfileMode', 'DebugMode', 'FAST_RUN', 'FAST_COMPILE', 'PROFILE_MODE', 'DEBUG_MODE'))

#Keep the default linker the same as the one for the mode FAST_RUN
AddConfigVar('linker',
        "Default linker. If not None, will use this linker with the Mode object(not ProfileMode or DebugMode)",
        EnumStr('c|py', 'py', 'c', 'c|py_nogc', 'c&py'))

#Keep the default optimizer the same as the one for the mode FAST_RUN
AddConfigVar('optimizer',
        "Default optimizer. If not None, will use this linker with the Mode object(not ProfileMode or DebugMode)",
        EnumStr('fast_run', 'merge', 'fast_compile', 'None'))

AddConfigVar('home',
        "User home directory",
        StrParam(os.getenv("HOME", os.path.expanduser('~'))))
#This expanduser works on windows (see discussion on theano-users, July 13 2010)

AddConfigVar('nocleanup',
        "suppress the deletion of code files that did not compile cleanly",
        BoolParam(False))

AddConfigVar('tensor.cmp_sloppy',
        "Relax tensor._allclose (0) not at all, (1) a bit, (2) more",
        IntParam(0, lambda i: i in (0,1,2)))

AddConfigVar('tensor.local_elemwise_fusion',
        "Enable or not in fast_run mode(fast_run optimization) the elemwise fusion optimization",
        BoolParam(True))

AddConfigVar('gpu.local_elemwise_fusion',
        "Enable or not in fast_run mode(fast_run optimization) the gpu elemwise fusion optimization",
        BoolParam(True))

#http://developer.amd.com/CPU/LIBRARIES/LIBM/Pages/default.aspx
AddConfigVar('lib.amdlibm',
        "Use amd's amdlibm numerical library",
        BoolParam(False))

AddConfigVar('op.set_flops',
        "currently used only in ConvOp. The profile mode will print the flops/s for the op.",
        BoolParam(False))

AddConfigVar('nvcc.fastmath',
        "",
        BoolParam(False))

AddConfigVar('cuda.root',
        "directory with bin/, lib/, include/ for cuda utilities",
        StrParam(os.getenv('CUDA_ROOT', "/usr/local/cuda")))

AddConfigVar('gpuelemwise.sync',
        "when true, wait that the gpu fct finished and check it error code.",
        BoolParam(True))

AddConfigVar('traceback.limit',
             "The number of stack to trace. -1 mean all.",
             IntParam(5))


###
### To disable some warning about old bug that are fixed now.
###

AddConfigVar('warn.argmax_pushdown_bug',
             "Warn if in past version of Theano we generated a bug with the optimisation theano.tensor.nnet.nnet.local_argmax_pushdown optimization. Was fixed 27 may 2010",
             BoolParam(True))

AddConfigVar('warn.gpusum_01_011_0111_bug',
             "Warn if we are in a case where old version of Theano had a silent bug with GpuSum pattern 01,011 and 0111 when the first dimensions was bigger then 4096. Was fixed 31 may 2010",
             BoolParam(True))

AddConfigVar('warn.sum_sum_bug',
             "Warn if we are in a case where Theano version between version 9923a40c7b7a and the 2 august 2010(fixed date), generated an error in that case. This happen when their is 2 consecutive sum in the graph, bad code was generated. Was fixed 2 August 2010",
             BoolParam(True))

AddConfigVar('warn.sum_div_dimshuffle_bug',
             "Warn if previous versions of Theano (between rev. 3bd9b789f5e8, 2010-06-16, and cfc6322e5ad4, 2010-08-03) would have given incorrect result. This bug was triggered by sum of division of dimshuffled tensors.",
             BoolParam(True))
