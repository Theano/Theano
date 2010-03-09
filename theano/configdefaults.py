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

# keep the default mode.optimizer==config.optimizer and mode.linker==config.linker!
AddConfigVar('mode',
        "Default compilation mode",
        EnumStr('FAST_RUN', 'FAST_COMPILE', 'PROFILE_MODE', 'DEBUG_MODE', 'Mode', 'ProfileMode', 'DebugMode'))

#Keep the default linker the same as the one for the mode
AddConfigVar('linker',
        "Default linker. If not None, will use this linker with the Mode object(not ProfileMode or DebugMode)",
        EnumStr('c|py', 'py', 'c', 'c|py_nogc', 'c&py'))

#Keep the default optimizer the same as the one for the mode
AddConfigVar('optimizer',
        "Default optimizer. If not None, will use this linker with the Mode object(not ProfileMode or DebugMode)",
        EnumStr('fast_run', 'merge', 'fast_compile', 'None'))

AddConfigVar('home',
        "User home directory",
        StrParam(os.getenv("HOME")))

AddConfigVar('nocleanup',
        "suppress the deletion of code files that did not compile cleanly",
        BoolParam(False))

AddConfigVar('tensor.cmp_sloppy',
        "Relax tensor._allclose (0) not at all, (1) a bit, (2) more",
        IntParam(0, lambda i: i in (0,1,2)))

AddConfigVar('tensor.local_elemwise_fusion',
        "",
        BoolParam(False))

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

AddConfigVar('cmodule.mac_framework_link',
        "If set to true, breaks certain mac installations with the infamous Bus Error",
        BoolParam(False))
