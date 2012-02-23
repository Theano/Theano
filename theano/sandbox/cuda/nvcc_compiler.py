import commands
import distutils
import logging
import os
import re
import subprocess
import sys
import warnings

from theano.gof.cc import hash_from_file
from theano.gof.cmodule import (std_libs, std_lib_dirs, std_include_dirs, dlimport,
    get_lib_extension, local_bitwidth)

_logger=logging.getLogger("theano.sandbox.cuda.nvcc_compiler")
_logger.setLevel(logging.WARN)

from theano.configparser import config, AddConfigVar, StrParam, BoolParam

AddConfigVar('nvcc.compiler_bindir',
        "If defined, nvcc compiler driver will seek g++ and gcc in this directory",
        StrParam(""))

AddConfigVar('cuda.nvccflags',
        "DEPRECATED, use nvcc.flags instead",
        StrParam("", allow_override=False),
        in_c_key=False)

if config.cuda.nvccflags != '':
    warnings.warn('Configuration variable cuda.nvccflags is deprecated. '
            'Please use nvcc.flags instead. You provided value: %s'
            % config.cuda.nvccflags)

AddConfigVar('nvcc.flags',
        "Extra compiler flags for nvcc",
        StrParam(config.cuda.nvccflags))

AddConfigVar('nvcc.fastmath',
        "",
        BoolParam(False))

nvcc_path = 'nvcc'
nvcc_version = None
def is_nvcc_available():
    """Return True iff the nvcc compiler is found."""
    try:
        p = subprocess.Popen(['nvcc', '--version'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        p.wait()
        s = p.stdout.readlines()[-1].split(',')[1].strip().split()
        assert s[0]=='release'
        global nvcc_version
        nvcc_version = s[1]
        return True
    except Exception:
        #try to find nvcc into cuda.root
        p = os.path.join(config.cuda.root,'bin','nvcc')
        if os.path.exists(p):
            global nvcc_path
            nvcc_path = p
            return True
        else: return False

def set_cuda_root():
    s = os.getenv("PATH")
    if not s:
        return
    for dir in s.split(os.path.pathsep):
        if os.path.exists(os.path.join(dir,"nvcc")):
            config.cuda.root = os.path.split(dir)[0]
            return

rpath_defaults = []
def add_standard_rpath(rpath):
    rpath_defaults.append(rpath)


class NVCC_compiler(object):
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
        return flags

    @staticmethod
    def compile_str(
            module_name, src_code,
            location=None, include_dirs=[], lib_dirs=[], libs=[], preargs=[],
            rpaths=rpath_defaults):
        """
        :param module_name: string (this has been embedded in the src_code
        :param src_code: a complete c or c++ source listing for the module
        :param location: a pre-existing filesystem directory where the cpp file and .so will be written
        :param include_dirs: a list of include directory names (each gets prefixed with -I)
        :param lib_dirs: a list of library search path directory names (each gets prefixed with -L)
        :param libs: a list of libraries to link with (each gets prefixed with -l)
        :param preargs: a list of extra compiler arguments
        :param rpaths: list of rpaths to use with Xlinker. Defaults to `rpath_defaults`.

        :returns: dynamically-imported python module of the compiled code.

        :note 1: On Windows 7 with nvcc 3.1 we need to compile in the real directory
                 Otherwise nvcc never finish.
        """

        rpaths = list(rpaths)

        if sys.platform=="win32":
            # Remove some compilation args that cl.exe does not understand.
            # cl.exe is the compiler used by nvcc on Windows.
            for a in ["-Wno-write-strings","-Wno-unused-label",
                      "-Wno-unused-variable", "-fno-math-errno"]:
                if a in preargs:
                    preargs.remove(a)
        if preargs is None:
            preargs= []
        else: preargs = list(preargs)
        if sys.platform!='win32':
            preargs.append('-fPIC')
        no_opt = False
        cuda_root = config.cuda.root

        #The include dirs gived by the user should have precedence over
        #the standards ones.
        include_dirs = include_dirs + std_include_dirs()
        if os.path.abspath(os.path.split(__file__)[0]) not in include_dirs:
            include_dirs.append(os.path.abspath(os.path.split(__file__)[0]))

        libs = std_libs() + libs
        if 'cudart' not in libs:
            libs.append('cudart')

        lib_dirs = std_lib_dirs() + lib_dirs
        if cuda_root:
            lib_dirs.append(os.path.join(cuda_root, 'lib'))

            # from Benjamin Schrauwen April 14 2010
            if sys.platform != 'darwin':
                # No 64 bit CUDA libraries available on the mac, yet..
                lib_dirs.append(os.path.join(cuda_root, 'lib64'))


        if sys.platform == 'darwin':
            # On the mac, nvcc is not able to link using -framework Python, so we have
            # manually add the correct library and paths
            darwin_python_lib = commands.getoutput('python-config --ldflags')
        else:
            # sometimes, the linker cannot find -lpython so we need to tell it
            # explicitly where it is located
            # this returns somepath/lib/python2.x
            python_lib = distutils.sysconfig.get_python_lib(plat_specific=1, \
                            standard_lib=1)
            python_lib = os.path.dirname(python_lib)
            if python_lib not in lib_dirs:
                lib_dirs.append(python_lib)

        cppfilename = os.path.join(location, 'mod.cu')
        cppfile = file(cppfilename, 'w')

        _logger.debug('Writing module C++ code to %s', cppfilename)
        ofiles = []
        rval = None

        cppfile.write(src_code)
        cppfile.close()
        lib_filename = os.path.join(location, '%s.%s' %
                (module_name, get_lib_extension()))

        _logger.debug('Generating shared lib %s', lib_filename)
        # TODO: Why do these args cause failure on gtx285 that has 1.3 compute capability? '--gpu-architecture=compute_13', '--gpu-code=compute_13',
        preargs1=[pa for pa in preargs if pa.startswith('-O') or pa.startswith('--maxrregcount=')]#nvcc argument
        preargs2=[pa for pa in preargs if pa not in preargs1]#other arguments

        cmd = [nvcc_path, '-shared', '-g'] + preargs1
        if config.nvcc.compiler_bindir:
            cmd.extend(['--compiler-bindir', config.nvcc.compiler_bindir])

        if sys.platform == 'win32':
            # add flags for Microsoft compiler to create .pdb files
            preargs2.append('/Zi')
            cmd.extend(['-Xlinker', '/DEBUG'])

        if sys.platform != 'win32':
            if local_bitwidth() == 64:
                cmd.append('-m64')
                preargs2.append('-m64')
            else:
                cmd.append('-m32')
                preargs2.append('-m32')

        if len(preargs2)>0:
            cmd.extend(['-Xcompiler', ','.join(preargs2)])

        if config.cuda.root and os.path.exists(os.path.join(config.cuda.root,'lib')):
            rpaths.append(os.path.join(config.cuda.root,'lib'))
            if sys.platform != 'darwin':
                # the 64bit CUDA libs are in the same files as are named by the function above
                rpaths.append(os.path.join(config.cuda.root,'lib64'))
        if sys.platform != 'win32':
            # the -rpath option is not understood by the Microsoft linker
            for rpath in rpaths:
                cmd.extend(['-Xlinker',','.join(['-rpath',rpath])])
        cmd.extend('-I%s'%idir for idir in include_dirs)
        cmd.extend(['-o',lib_filename])
        cmd.append(os.path.split(cppfilename)[-1])
        cmd.extend(['-L%s'%ldir for ldir in lib_dirs])
        cmd.extend(['-l%s'%l for l in libs])
        if module_name != 'cuda_ndarray':
            cmd.append("-lcuda_ndarray")
        if sys.platform == 'darwin':
            cmd.extend(darwin_python_lib.split())

        if sys.platform == 'darwin':
            done = False
            while not done:
                try:
                    indexof = cmd.index('-framework')
                    newarg = '-Xcompiler', ','.join(cmd[indexof:(indexof + 2)])
                    cmd.pop(indexof) # Remove -framework
                    cmd.pop(indexof) # Remove argument to -framework
                    cmd.extend(newarg)
                except ValueError, e:
                    done = True

        # Remove "-u Symbol" arguments, since they are usually not relevant
        # for the new compilation, even if they were used for compiling python.
        # If they are necessary, the nvcc syntax is "-U Symbol" with a capital U.
        done = False
        while not done:
            try:
                indexof = cmd.index('-u')
                cmd.pop(indexof) # Remove -u
                cmd.pop(indexof) # Remove argument to -u
            except ValueError, e:
                done = True

        # Fix for MacOS X.
        cmd = remove_python_framework_dir(cmd)

        #cmd.append("--ptxas-options=-v")  #uncomment this to see register and shared-mem requirements
        _logger.debug('Running cmd %s', ' '.join(cmd))
        orig_dir = os.getcwd()
        try:
            os.chdir(location)
            p = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            nvcc_stdout, nvcc_stderr = p.communicate()[:2]
        finally:
            os.chdir(orig_dir)

        if nvcc_stdout:
            # this doesn't happen to my knowledge
            print >> sys.stderr, "DEBUG: nvcc STDOUT", nvcc_stdout

        for eline in nvcc_stderr.split('\n'):
            if not eline:
                continue
            if 'skipping incompatible' in eline: #ld is skipping an incompatible library
                continue
            if 'declared but never referenced' in eline:
                continue
            if 'statement is unreachable' in eline:
                continue
            _logger.info("NVCC: %s", eline)

        if p.returncode:
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
                print >> sys.stderr, l
            print >> sys.stderr, '==============================='
            for i, l in enumerate(src_code.split('\n')):
                print >> sys.stderr,  i+1, l
            raise Exception('nvcc return status', p.returncode, 'for cmd', ' '.join(cmd))

        #touch the __init__ file
        file(os.path.join(location, "__init__.py"),'w').close()
        return dlimport(lib_filename)


def remove_python_framework_dir(cmd):
    """
    Search for Python framework directory and get rid of it.

    :param cmd: A list of strings corresponding to compilation arguments. On
    MacOS X, one of these strings may be of the form
    "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python"
    and it needs to be removed as otherwise compilation will fail.

    :return: The same list as `cmd`, but without the element of the form
    mentioned above, if one exists.
    """
    # The fix below was initially suggested by Nicolas Pinto:
    #   http://groups.google.com/group/theano-users/browse_thread/thread/c84bfe31bb411493
    # It was improved later following a bug report by Benjamin Hamner:
    #   https://groups.google.com/group/theano-users/browse_thread/thread/374ec2dadd3ac369/024e2be792f98d86
    # It was modified by Graham Taylor to support Enthought Python Distribution
    #     7.x (32 and 64 bit)
    # TODO It is a bit hack-ish, is it possible to find a more generic fix?
    fwk_pattern = '(Python|EPD64).framework/Versions/(2\.[0-9]|7\.[0-9])/Python$'
    rval = [element for element in cmd
            if (re.search(fwk_pattern, element) is None
                # Keep this element if it turns out to be part of an argument
                # like -L.
                or element.startswith('-'))]
    if len(rval) < len(cmd) - 1:
        warnings.warn("'remove_python_framework_dir' removed %s elements from "
                      "the command line, while it is expected to remove at "
                      "most one. If compilation fails, this would be a good "
                      "place to start looking for a problem." %
                      (len(cmd) - len(rval)))
    return rval
