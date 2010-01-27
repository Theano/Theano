import os
from theano.configparser import TheanoConfigParser, AddConfigVar, EnumStr, StrParam, IntParam, FloatParam, BoolParam

config = TheanoConfigParser()

AddConfigVar('floatX',
        "Default floating-point precision for python casts",
        EnumStr('float64', 'float32'), 
        )

AddConfigVar('device',
        "Default device for computations",
        EnumStr('cpu', *['gpu%i'%i for i in range(4)])
        )

AddConfigVar('mode',
        "Default compilation mode",
        EnumStr('FAST_RUN', 'FAST_COMPILE', 'PROFILE_MODE', 'DEBUG_MODE'))

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
