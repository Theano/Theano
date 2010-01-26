import os
from .configparser import TheanoConfigParser, AddConfigVar, EnumStr, StrParam, IntParam, FloatParam, BoolParam

config = TheanoConfigParser()

AddConfigVar('floatX',
        "Default floating-point precision for python casts",
        EnumStr('float64', 'float32'), 
        )

AddConfigVar('device',
        "Default device for computations",
        EnumStr('cpu', *['gpu%i'%i for i in range(16)])
        )
AddConfigVar('mode',
        "Default compilation mode",
        EnumStr('FAST_RUN', 'FAST_COMPILE', 'PROFILE_MODE', 'DEBUG_MODE'))

AddConfigVar('home',
        "User home directory",
        EnumStr(os.getenv("HOME")))

AddConfigVar('base_compiledir',
        "arch-independent cache directory for compiled modules",
        StrParam(os.path.join(config.home, '.theano')))

AddConfigVar('compiledir',
        "arch-dependent cache directory for compiled modules",
        StrParam("")) #NO DEFAULT??

AddConfigVar('nocleanup',
        "suppress the deletion of code files that did not compile cleanly",
        BoolParam(False))

AddConfigVar('blas.ldflags',
        "lib[s] to include for level-3 blas implementation",
        StrParam("-lblas"))

AddConfigVar('DebugMode.patience',
        "Optimize graph this many times",
        IntParam(10, lambda i: i > 0))

AddConfigVar('DebugMode.check_c',
        "Run C implementations where possible",
        BoolParam(True))

AddConfigVar('DebugMode.check_py',
        "Run Python implementations where possible",
        BoolParam(True))

AddConfigVar('DebugMode.check_finite',
        "True -> complain about NaN/Inf results",
        BoolParam(True))

AddConfigVar('DebugMode.check_strides',
        ("Check that Python- and C-produced ndarrays have same strides.  "
            "On difference: (0) - ignore, (1) warn, or (2) raise error"),
        IntParam(1, lambda i: i in (0,1,2)))

AddConfigVar('ProfileMode.n_apply_to_print',
        "",
        IntParam(15, lambda i: i > 0))

AddConfigVar('ProfileMode.n_ops_to_print',
        "",
        IntParam(20, lambda i: i > 0))

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
        StrParam("/usr/local/cuda"))

AddConfigVar('gpuelemwise.sync',
        "when true, wait that the gpu fct finished and check it error code.",
        BoolParam(True))
