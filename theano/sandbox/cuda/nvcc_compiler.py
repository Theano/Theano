from __future__ import absolute_import, print_function, division
import distutils
import logging
import os
import subprocess
import sys
import warnings
from locale import getpreferredencoding

import numpy

from theano import config
from theano.compat import decode, decode_with
from theano.configdefaults import local_bitwidth
from theano.gof.utils import hash_from_file
from theano.gof.cmodule import (std_libs, std_lib_dirs,
                                std_include_dirs, dlimport,
                                Compiler,
                                get_lib_extension)
from theano.misc.windows import output_subprocess_Popen

_logger = logging.getLogger("theano.sandbox.cuda.nvcc_compiler")

nvcc_path = 'nvcc'
nvcc_version = None


def is_nvcc_available():
    """
    Return True iff the nvcc compiler is found.

    """
    def set_version():
        p_out = output_subprocess_Popen([nvcc_path, '--version'])
        ver_line = decode(p_out[0]).strip().split('\n')[-1]
        build, version = ver_line.split(',')[1].strip().split()

        assert build == 'release'
        global nvcc_version
        nvcc_version = version
    try:
        set_version()
        return True
    except Exception:
        # try to find nvcc into cuda.root
        p = os.path.join(config.cuda.root, 'bin', 'nvcc')
        if os.path.exists(p):
            global nvcc_path
            nvcc_path = p
            try:
                set_version()
            except Exception:
                return False
            return True
        else:
            return False


rpath_defaults = []


def add_standard_rpath(rpath):
    rpath_defaults.append(rpath)


class NVCC_compiler(Compiler):
    supports_amdlibm = False

    @staticmethod
    def try_compile_tmp(src_code, tmp_prefix='', flags=(),
                        try_run=False, output=False):
        return Compiler._try_compile_tmp(src_code, tmp_prefix, flags,
                                         try_run, output,
                                         nvcc_path)

    @staticmethod
    def try_flags(flag_list, preambule="", body="",
                  try_run=False, output=False):
        return Compiler._try_flags(flag_list, preambule, body, try_run, output,
                                   nvcc_path)

    @staticmethod
    def version_str():
        return "nvcc " + nvcc_version

    @staticmethod
    def compile_args():
        """
        This args will be received by compile_str() in the preargs paramter.
        They will also be included in the "hard" part of the key module.

        """
        flags = [flag for flag in config.nvcc.flags.split(' ') if flag]
        if config.nvcc.fastmath:
            flags.append('-use_fast_math')
        cuda_ndarray_cuh_hash = hash_from_file(
            os.path.join(os.path.split(__file__)[0], 'cuda_ndarray.cuh'))
        flags.append('-DCUDA_NDARRAY_CUH=' + cuda_ndarray_cuh_hash)

        # NumPy 1.7 Deprecate the old API. I updated most of the places
        # to use the new API, but not everywhere. When finished, enable
        # the following macro to assert that we don't bring new code
        # that use the old API.
        flags.append("-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

        # numpy 1.7 deprecated the following macro but the didn't
        # existed in the past
        numpy_ver = [int(n) for n in numpy.__version__.split('.')[:2]]
        if bool(numpy_ver < [1, 7]):
            flags.append("-DNPY_ARRAY_ENSURECOPY=NPY_ENSURECOPY")
            flags.append("-DNPY_ARRAY_ALIGNED=NPY_ALIGNED")
            flags.append("-DNPY_ARRAY_WRITEABLE=NPY_WRITEABLE")
            flags.append("-DNPY_ARRAY_UPDATE_ALL=NPY_UPDATE_ALL")
            flags.append("-DNPY_ARRAY_C_CONTIGUOUS=NPY_C_CONTIGUOUS")
            flags.append("-DNPY_ARRAY_F_CONTIGUOUS=NPY_F_CONTIGUOUS")

        # If the user didn't specify architecture flags add them
        if not any(['-arch=sm_' in f for f in flags]):
            # We compile cuda_ndarray.cu during import.
            # We should not add device properties at that time.
            # As the device is not selected yet!
            # TODO: re-compile cuda_ndarray when we bind to a GPU?
            import theano.sandbox.cuda
            if hasattr(theano.sandbox, 'cuda'):
                n = theano.sandbox.cuda.use.device_number
                if n is None:
                    _logger.warn(
                        "We try to get compilation arguments for CUDA"
                        " code, but the GPU device is not initialized."
                        " This is probably caused by an Op that work on"
                        " the GPU that don't inherit from GpuOp."
                        " We Initialize the GPU now.")
                    theano.sandbox.cuda.use(
                        "gpu",
                        force=True,
                        default_to_move_computation_to_gpu=False,
                        move_shared_float32_to_gpu=False,
                        enable_cuda=False)
                    n = theano.sandbox.cuda.use.device_number
                p = theano.sandbox.cuda.device_properties(n)
                flags.append('-arch=sm_' + str(p['major']) +
                             str(p['minor']))

        return flags

    @staticmethod
    def compile_str(
            module_name, src_code,
            location=None, include_dirs=[], lib_dirs=[], libs=[], preargs=[],
            rpaths=rpath_defaults, py_module=True, hide_symbols=True):
        """

        Parameters
        ----------
        module_name: str
             This has been embedded in the src_code.
        src_code
            A complete c or c++ source listing for the module.
        location
            A pre-existing filesystem directory where the
            cpp file and .so will be written.
        include_dirs
            A list of include directory names (each gets prefixed with -I).
        lib_dirs
            A list of library search path directory names (each gets
            prefixed with -L).
        libs
            A list of libraries to link with (each gets prefixed with -l).
        preargs
            A list of extra compiler arguments.
        rpaths
            List of rpaths to use with Xlinker. Defaults to `rpath_defaults`.
        py_module
            If False, compile to a shared library, but
            do not import as a Python module.
        hide_symbols
            If True (the default), hide all symbols from the library symbol
            table unless explicitely exported.

        Returns
        -------
        module
            Dynamically-imported python module of the compiled code.
            (unless py_module is False, in that case returns None.)

        Notes
        -----
        On Windows 7 with nvcc 3.1 we need to compile in the real directory
        Otherwise nvcc never finish.

        """
        # Remove empty string directory
        include_dirs = [d for d in include_dirs if d]
        lib_dirs = [d for d in lib_dirs if d]

        rpaths = list(rpaths)

        if sys.platform == "win32":
            # Remove some compilation args that cl.exe does not understand.
            # cl.exe is the compiler used by nvcc on Windows.
            for a in ["-Wno-write-strings", "-Wno-unused-label",
                      "-Wno-unused-variable", "-fno-math-errno"]:
                if a in preargs:
                    preargs.remove(a)
        if preargs is None:
            preargs = []
        else:
            preargs = list(preargs)
        if sys.platform != 'win32':
            preargs.append('-fPIC')
        if config.cmodule.remove_gxx_opt:
            preargs = [p for p in preargs if not p.startswith('-O')]

        cuda_root = config.cuda.root

        # The include dirs gived by the user should have precedence over
        # the standards ones.
        include_dirs = include_dirs + std_include_dirs()
        if os.path.abspath(os.path.split(__file__)[0]) not in include_dirs:
            include_dirs.append(os.path.abspath(os.path.split(__file__)[0]))

        libs = libs + std_libs()
        if 'cudart' not in libs:
            libs.append('cudart')

        lib_dirs = lib_dirs + std_lib_dirs()

        if sys.platform != 'darwin':
            # config.dnn.include_path add this by default for cudnn in the
            # new back-end. This should not be used in this back-end. So
            # just remove them.
            lib_dirs = [ld for ld in lib_dirs if
                        not(ld == os.path.join(cuda_root, 'lib') or
                            ld == os.path.join(cuda_root, 'lib64'))]

        if sys.platform != 'darwin':
            # sometimes, the linker cannot find -lpython so we need to tell it
            # explicitly where it is located
            # this returns somepath/lib/python2.x
            python_lib = distutils.sysconfig.get_python_lib(plat_specific=1,
                                                            standard_lib=1)
            python_lib = os.path.dirname(python_lib)
            if python_lib not in lib_dirs:
                lib_dirs.append(python_lib)

        cppfilename = os.path.join(location, 'mod.cu')
        with open(cppfilename, 'w') as cppfile:

            _logger.debug('Writing module C++ code to %s', cppfilename)
            cppfile.write(src_code)

        lib_filename = os.path.join(location, '%s.%s' %
                (module_name, get_lib_extension()))

        _logger.debug('Generating shared lib %s', lib_filename)
        # TODO: Why do these args cause failure on gtx285 that has 1.3
        # compute capability? '--gpu-architecture=compute_13',
        # '--gpu-code=compute_13',
        # nvcc argument
        preargs1 = []
        preargs2 = []
        for pa in preargs:
            if pa.startswith('-Wl,'):
                # the -rpath option is not understood by the Microsoft linker
                if sys.platform != 'win32' or not pa.startswith('-Wl,-rpath'):
                    preargs1.append('-Xlinker')
                    preargs1.append(pa[4:])
                continue
            for pattern in ['-O', '-arch=', '-ccbin=', '-G', '-g', '-I',
                            '-L', '--fmad', '--ftz', '--maxrregcount',
                            '--prec-div', '--prec-sqrt',  '--use_fast_math',
                            '-fmad', '-ftz', '-maxrregcount',
                            '-prec-div', '-prec-sqrt', '-use_fast_math',
                            '--use-local-env', '--cl-version=']:

                if pa.startswith(pattern):
                    preargs1.append(pa)
                    break
            else:
                preargs2.append(pa)

        # Don't put -G by default, as it slow things down.
        # We aren't sure if -g slow things down, so we don't put it by default.
        cmd = [nvcc_path, '-shared'] + preargs1
        if config.nvcc.compiler_bindir:
            cmd.extend(['--compiler-bindir', config.nvcc.compiler_bindir])

        if sys.platform == 'win32':
            # add flags for Microsoft compiler to create .pdb files
            preargs2.extend(['/Zi', '/MD'])
            cmd.extend(['-Xlinker', '/DEBUG'])
            # remove the complaints for the duplication of `double round(double)`
            # in both math_functions.h and pymath.h,
            # by not including the one in pymath.h
            cmd.extend(['-D HAVE_ROUND'])
        else:
            if hide_symbols:
                preargs2.append('-fvisibility=hidden')

        if local_bitwidth() == 64:
            cmd.append('-m64')
        else:
            cmd.append('-m32')

        if len(preargs2) > 0:
            cmd.extend(['-Xcompiler', ','.join(preargs2)])

        # We should not use rpath if possible. If the user provided
        # provided an cuda.root flag, we need to add one, but
        # otherwise, we don't add it. See gh-1540 and
        # https://wiki.debian.org/RpathIssue for details.

        if (not type(config.cuda).root.is_default and
            os.path.exists(os.path.join(config.cuda.root, 'lib'))):

            rpaths.append(os.path.join(config.cuda.root, 'lib'))
            if sys.platform != 'darwin':
                # the CUDA libs are universal (contain both 32-bit and 64-bit)
                rpaths.append(os.path.join(config.cuda.root, 'lib64'))
        if sys.platform != 'win32':
            # the -rpath option is not understood by the Microsoft linker
            for rpath in rpaths:
                cmd.extend(['-Xlinker', ','.join(['-rpath', rpath])])
        cmd.extend('-I%s' % idir for idir in include_dirs)
        cmd.extend(['-o', lib_filename])
        cmd.append(os.path.split(cppfilename)[-1])
        cmd.extend(['-L%s' % ldir for ldir in lib_dirs])
        cmd.extend(['-l%s' % l for l in libs])
        if sys.platform == 'darwin':
            # This tells the compiler to use the already-loaded python
            # symbols (which should always be the right ones).
            cmd.extend(['-Xcompiler', '-undefined,dynamic_lookup'])

        # Remove "-u Symbol" arguments, since they are usually not
        # relevant for the new compilation, even if they were used for
        # compiling python.  If they are necessary, the nvcc syntax is
        # "-U Symbol" with a capital U.
        done = False
        while not done:
            try:
                indexof = cmd.index('-u')
                cmd.pop(indexof)  # Remove -u
                cmd.pop(indexof)  # Remove argument to -u
            except ValueError as e:
                done = True

        # CUDA Toolkit v4.1 Known Issues:
        # Host linker on Mac OS 10.7 (and 10.6 for me) passes -no_pie option
        # to nvcc this option is not recognized and generates an error
        # http://stackoverflow.com/questions/9327265/nvcc-unknown-option-no-pie
        # Passing -Xlinker -pie stops -no_pie from getting passed
        if sys.platform == 'darwin' and nvcc_version >= '4.1':
            cmd.extend(['-Xlinker', '-pie'])

        # cmd.append("--ptxas-options=-v") #uncomment this to see
        # register and shared-mem requirements
        _logger.debug('Running cmd %s', ' '.join(cmd))
        orig_dir = os.getcwd()
        try:
            os.chdir(location)
            p = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            nvcc_stdout_raw, nvcc_stderr_raw = p.communicate()[:2]
            console_encoding = getpreferredencoding()
            nvcc_stdout = decode_with(nvcc_stdout_raw, console_encoding)
            nvcc_stderr = decode_with(nvcc_stderr_raw, console_encoding)
        finally:
            os.chdir(orig_dir)

        for eline in nvcc_stderr.split('\n'):
            if not eline:
                continue
            if 'skipping incompatible' in eline:
                # ld is skipping an incompatible library
                continue
            if 'declared but never referenced' in eline:
                continue
            if 'statement is unreachable' in eline:
                continue
            _logger.info("NVCC: %s", eline)

        if p.returncode:
            for i, l in enumerate(src_code.split('\n')):
                print(i + 1, l, file=sys.stderr)
            print('===============================', file=sys.stderr)
            # filter the output from the compiler
            for l in nvcc_stderr.split('\n'):
                if not l:
                    continue
                # filter out the annoying declaration warnings

                try:
                    if l[l.index(':'):].startswith(': warning: variable'):
                        continue
                    if l[l.index(':'):].startswith(': warning: label'):
                        continue
                except Exception:
                    pass
                print(l, file=sys.stderr)
            print(nvcc_stdout)
            print(cmd)
            raise Exception('nvcc return status', p.returncode,
                            'for cmd', ' '.join(cmd))
        elif config.cmodule.compilation_warning and nvcc_stdout:
            print(nvcc_stdout)

        if nvcc_stdout:
            # this doesn't happen to my knowledge
            print("DEBUG: nvcc STDOUT", nvcc_stdout, file=sys.stderr)

        if py_module:
            # touch the __init__ file
            open(os.path.join(location, "__init__.py"), 'w').close()
            return dlimport(lib_filename)
