from __future__ import print_function
import distutils
import logging
import os
import tempfile
import sys
import warnings

import numpy

from theano.compat import decode
from six import b as byte_literal
from theano.gof import local_bitwidth
from theano.gof.utils import hash_from_file
from theano.gof.cmodule import (std_libs, std_lib_dirs,
                                std_include_dirs, dlimport,
                                Compiler,
                                get_lib_extension)
from theano.misc.windows import output_subprocess_Popen

from theano.configparser import (config, AddConfigVar, StrParam,
                                 BoolParam, ConfigParam)

# Initialize this module's logger
_logger = logging.getLogger("theano.sandbox.cuda.nvcc_compiler")

# Detect path to CUDA runtime
user_provided_cuda_root = True


def nvcc_path_autodetect():
    """Autodetect the root directory of the CUDA runtime."""
# If this function is called then the config defaulted to autodetect
    global user_provided_cuda_root
    user_provided_cuda_root = False
# Check is the user supplied the path via an environment variable
    env_cuda_root = os.getenv("CUDA_ROOT", "")
    if env_cuda_root and os.path.exists(env_cuda_root):
        return env_cuda_root
# Otherwise check the system path variable
    env_path = os.getenv("PATH", "")
    if not env_path:
        return None
# Try each directory in PATH until nvcc is found
    for path in env_path.split(os.path.pathsep):
        if os.path.exists(os.path.join(path, "nvcc")):
            return os.path.split(path)[0]

AddConfigVar('cuda.root',
             """directory with bin/, lib/, include/ for cuda utilities.
             This directory is included via -L and -rpath when linking
             dynamically compiled modules.  If AUTO and nvcc is in the
             path, it will use one of nvcc parent directory.  Otherwise
             /usr/local/cuda will be used.  Leave empty to prevent extra
             linker directives.  Default: environment variable "CUDA_ROOT"
             or else "AUTO".
             """,
             StrParam(nvcc_path_autodetect),
             in_c_key=False)


# Validate NVCC flags provided in the config
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
             " in this directory.",
             StrParam(""),
             in_c_key=False)


AddConfigVar('nvcc.fastmath',
             "",
             BoolParam(False),
             # Not needed in c key as it is already added.
             # We remove it as we don't make the md5 of config to change
             # if theano.sandbox.cuda is loaded or not.
             in_c_key=False)

# From here on cuda.root point exatcly to the CUDA runtime root that
#  theano autodetected or the user wanted to use.

nvcc_path = 'nvcc'
nvcc_version = None


def run_command(command, **params):
    # Define a primitive function that executes a given command in
    #  a controlled environment and collects all output.
    """A simple primitive for executing a given command and
    grabing its output.
    :param command: a list of string representing the command
    to be executed;
    :param **params: optional named parametrs passed to
    output_subprocess_Popen()."""
    _out, _err, _exit_code = '', '', -1
    try:
        _out, _err, _exit_code = output_subprocess_Popen(command, **params)
    except Exception as e:
        _err = str(e)
    return _exit_code, decode(_out), decode(_err)


# Detect the version of the nvcc compiler
def nvcc_get_version(path_to_nvcc):
    """Get the version of the CUDA Runtime compiler, by running 'nvcc'
    with --version flag. Raises an OSError exception if the path sepcified
    is invalid or non-executable."""
    _exit_code, _out, _err = run_command([path_to_nvcc, '--version'])
    ver_line = _out.strip().split('\n')[-1]
    return ver_line.split(',')[1].strip().split()


def nvcc_check_version(path='',
                       allowed_versions=[],
                       allowed_builds=['release']):
    """Check if the version and build of the CUDA compiler are acceptable."""
    path = os.path.join(path, 'nvcc')
    try:
        build, version = nvcc_get_version(path)
        assert not allowed_versions or version.lower() in allowed_versions
        assert not allowed_builds or build.lower() in allowed_builds
    except Exception:
        return path, None
    return path, version


# Detect CUDA runtime version
def is_nvcc_available():
    """Checks if the CUDA compiler is availabe on 'cuda_path'. Initializes
    global variables 'nvcc_version', 'nvcc_path' if it is."""
# Check nvcc on the specified CUDA directory
    path, version = nvcc_check_version(os.path.join(config.cuda.root, 'bin'))
    if version is None:
        return False
# Set the globabl parameters
    global nvcc_version, nvcc_path
    nvcc_version = version
    nvcc_path = path
    return True

# Initialize global variables.
cuda_available = is_nvcc_available()


rpath_defaults = []


def add_standard_rpath(rpath):
    rpath_defaults.append(rpath)


# A primitive for creating a source code file.
def __tmp_source(code, suffix='', prefix=''):
    """A primitive for writing the supplied source code into a temporary file.
    :param code: a complete source code listing to be written to a temporary
        file;
    :param suffix: a suffix appended to the name of the temporary file (empty
        by default);
    :param prefix: a string the name of the temporary file should start with
        (empty by default)."""
    path, error = None, ''
    try:
        handle = None
        # Python3 compatibility: try to cast Py3 strings as Py2 strings.
        #  Do nothing if failed.
        try:
            code = byte_literal(code)
        except Exception:
            pass
        try:
            # Create a temporary file
            handle, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.write(handle, code)
        finally:
            if handle is not None:
                os.close(handle)
    except Exception as e:
        # If for some reason the OS could n't write to the temporary file
        if path and os.path.exists(path):
            os.remove(path)
        path, error = None, str(e)
    return path, error


def test_build_and_run(compiler, source, suffix='', prefix='',
                       flags=[], run=False, output=False):
    """Attempt to compile the source code using the provided compiler and run
    the result if required.
    :param compiler: path to the compiler used for building the source;
    :param source: a complete source listing to be compiled;
    :param suffix: a the extension of the temporary file (empty by default);
    :param prefix: a prefix for the temporary file (no prefix by default);
    :param flags: a list of flags to be passed to the compiler;
    :param run: determines if the compiled code should be run
        (False by default);
    :param output: should the output of the compiled code be collected
        (defaults to False);
    """
    compilation_ok, run_ok, _out, _err = False, False, '', ''
    # Put the source code in a temporary file,
    input_path, _err = __tmp_source(source, suffix=suffix, prefix=prefix)
    if input_path:
        output_path = input_path[:-(len(suffix) + 1)]
        # ... then compile it,
        _exit_code, _out, _err = run_command(
            [compiler, input_path, '-o', output_path] + flags)
        compilation_ok = _exit_code == 0
        if compilation_ok and run:
            # ... and run, if necessary.
            _exit_code, _out, _err = run_command([output_path])
            run_ok = _exit_code == 0
        # Remove temporary files.
        for path in [input_path, output_path, output_path + ".exe"]:
            if os.path.exists(path):
                os.remove(path)
    if not run and not output:
        return compilation_ok
    elif not run and output:
        return (compilation_ok, _out, _err)
    elif not output:
        return (compilation_ok, run_ok)
    else:
        return (compilation_ok, run_ok, _out, _err)


# A class for the NVCC compiler
class NVCC_compiler(Compiler):
    @staticmethod
    def try_compile_tmp(src_code, tmp_prefix='', flags=(),
                        try_run=False, output=False):
        return test_build_and_run(nvcc_path, src_code, '.cu',
                                  tmp_prefix, list(flags),
                                  try_run, output)

    @staticmethod
    def try_flags(flag_list, preambule="", body="",
                  try_run=False, output=False):
        """Try to compile a dummy file with the given flags.
        Returns True if compilation was successful, False if there
        were errors."""
        return test_build_and_run(nvcc_path, """
                %(preambule)s
                int main(int argc, char* argv[])
                {
                    %(body)s
                    return 0;
                }
            """ % locals(), prefix='try_flags_', suffix='.cu',
            flags=list(flag_list), run=try_run,
            output=output)

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
        """:param module_name: string (this has been embedded in the src_code
        :param src_code: a complete c or c++ source listing for the module
        :param location: a pre-existing filesystem directory where the
                         cpp file and .so will be written
        :param include_dirs: a list of include directory names
                             (each gets prefixed with -I)
        :param lib_dirs: a list of library search path directory names
                         (each gets prefixed with -L)
        :param libs: a list of libraries to link with
                     (each gets prefixed with -l)
        :param preargs: a list of extra compiler arguments
        :param rpaths: list of rpaths to use with Xlinker.
                       Defaults to `rpath_defaults`.
        :param py_module: if False, compile to a shared library, but
            do not import as a Python module.

        :param hide_symbols: if True (the default), hide all symbols
        from the library symbol table unless explicitely exported.

        :returns: dynamically-imported python module of the compiled code.
            (unless py_module is False, in that case returns None.)

        :note 1: On Windows 7 with nvcc 3.1 we need to compile in the
                 real directory Otherwise nvcc never finish.

        """

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
        cuda_root = config.cuda.root

        # The include dirs gived by the user should have precedence over
        # the standards ones.
        include_dirs = include_dirs + std_include_dirs()
        if os.path.abspath(os.path.split(__file__)[0]) not in include_dirs:
            include_dirs.append(os.path.abspath(os.path.split(__file__)[0]))

        libs = std_libs() + libs
        if 'cudart' not in libs:
            libs.append('cudart')

        lib_dirs = std_lib_dirs() + lib_dirs
        if any(ld == os.path.join(cuda_root, 'lib') or
               ld == os.path.join(cuda_root, 'lib64') for ld in lib_dirs):
            warnings.warn("You have the cuda library directory in your "
                          "lib_dirs. This has been known to cause problems "
                          "and should not be done.")

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
        cppfile = open(cppfilename, 'w')

        _logger.debug('Writing module C++ code to %s', cppfilename)

        cppfile.write(src_code)
        cppfile.close()
        lib_filename = os.path.join(location, '%s.%s' %
                                    (module_name, get_lib_extension()))

        _logger.debug('Generating shared lib %s', lib_filename)
        # TODO: Why do these args cause failure on gtx285 that has 1.3
        # compute capability? '--gpu-architecture=compute_13',
        # '--gpu-code=compute_13',
        # nvcc argument
        preargs1 = []
        for pa in preargs:
            for pattern in ['-O', '-arch=', '-ccbin=', '-G', '-g', '-I',
                            '-L', '--fmad', '--ftz', '--maxrregcount',
                            '--prec-div', '--prec-sqrt',  '--use_fast_math',
                            '-fmad', '-ftz', '-maxrregcount',
                            '-prec-div', '-prec-sqrt', '-use_fast_math',
                            '--use-local-env', '--cl-version=']:

                if pa.startswith(pattern):
                    preargs1.append(pa)
        preargs2 = [pa for pa in preargs
                    if pa not in preargs1]  # other arguments

        # Don't put -G by default, as it slow things down.
        # We aren't sure if -g slow things down, so we don't put it by default.
        cmd = [nvcc_path, '-shared'] + preargs1
        if config.nvcc.compiler_bindir:
            cmd.extend(['--compiler-bindir', config.nvcc.compiler_bindir])

        if sys.platform == 'win32':
            # add flags for Microsoft compiler to create .pdb files
            preargs2.extend(['/Zi', '/MD'])
            cmd.extend(['-Xlinker', '/DEBUG'])
            # remove the complaints for duplication of 'double round(double)'
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
        if (user_provided_cuda_root and
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
        # If cwd is not None, the child's current directory will be changed to
        #  cwd before it is executed.
        nvcc_exit_code, nvcc_stdout, nvcc_stderr = run_command(cmd,
                                                               cwd=location)

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

        if nvcc_exit_code:
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
            raise Exception('nvcc return status', nvcc_exit_code,
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
