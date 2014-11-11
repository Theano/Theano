import cPickle
import errno
import logging
import os
import platform
import re
import shutil
import struct
import socket
import subprocess
import sys
import textwrap

import numpy

import theano
from theano.configparser import config, AddConfigVar, ConfigParam, StrParam
from theano.gof.utils import flatten
from theano.misc.windows import output_subprocess_Popen


_logger = logging.getLogger("theano.gof.compiledir")

try:
    p_out = output_subprocess_Popen([theano.config.cxx, '-dumpversion'])
    gcc_version_str = p_out[0].strip().decode()
except OSError:
    # Typically means gcc cannot be found.
    gcc_version_str = 'GCC_NOT_FOUND'


def local_bitwidth():
    """
    Return 32 for 32bit arch, 64 for 64bit arch

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
        "hostname": socket.gethostname(),
        }
compiledir_format_keys = ", ".join(sorted(compiledir_format_dict.keys()))
default_compiledir_format = ("compiledir_%(platform)s-%(processor)s-"
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
    formatted = config.compiledir_format % compiledir_format_dict
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
            os.makedirs(path, 0770)  # read-write-execute for user and group
        except OSError, e:
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
    open(os.path.join(path, '__init__.py'), 'w').close()

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

AddConfigVar(
    'compiledir',
    "platform-dependent cache directory for compiled modules",
    ConfigParam(
        os.path.join(
            config.base_compiledir,
            default_compiledirname()),
        filter=filter_compiledir,
        allow_override=False),
    in_c_key=False)


def cleanup():
    """
    Delete keys in old format from the compiledir.

    Old clean up include key in old format or with old version of the c_code:
    1) keys that have an ndarray in them.
       Now we use a hash in the keys of the constant data.
    2) key that don't have the numpy ABI version in them
    3) They do not have a compile version string

    If there is no key left for a compiled module, we delete the module.
    """
    compiledir = theano.config.compiledir
    for directory in os.listdir(compiledir):
        file = None
        try:
            try:
                filename = os.path.join(compiledir, directory, "key.pkl")
                file = open(filename, 'rb')
                #print file
                try:
                    keydata = cPickle.load(file)
                    for key in list(keydata.keys):
                        have_npy_abi_version = False
                        have_c_compiler = False
                        for obj in flatten(key):
                            if isinstance(obj, numpy.ndarray):
                                #Reuse have_npy_abi_version to
                                #force the removing of key
                                have_npy_abi_version = False
                                break
                            elif isinstance(obj, basestring):
                                if obj.startswith('NPY_ABI_VERSION=0x'):
                                    have_npy_abi_version = True
                                elif obj.startswith('c_compiler_str='):
                                    have_c_compiler = True
                            elif (isinstance(obj, (theano.gof.Op, theano.gof.Type)) and
                                  hasattr(obj, 'c_code_cache_version')):
                                v = obj.c_code_cache_version()
                                if v not in [(), None] and v not in key[0]:
                                    #Reuse have_npy_abi_version to
                                    #force the removing of key
                                    have_npy_abi_version = False
                                    break

                        if not have_npy_abi_version or not have_c_compiler:
                            try:
                                #This can happen when we move the compiledir.
                                if keydata.key_pkl != filename:
                                    keydata.key_pkl = filename
                                keydata.remove_key(key)
                            except IOError, e:
                                _logger.error(
                                    "Could not remove file '%s'. To complete "
                                    "the clean-up, please remove manually "
                                    "the directory containing it.",
                                    filename)
                    if len(keydata.keys) == 0:
                        shutil.rmtree(os.path.join(compiledir, directory))

                except EOFError:
                    _logger.error(
                        "Could not read key file '%s'. To complete "
                        "the clean-up, please remove manually "
                        "the directory containing it.",
                        filename)
            except IOError:
                _logger.error(
                    "Could not clean up this directory: '%s'. To complete "
                    "the clean-up, please remove it manually.",
                    directory)
        finally:
            if file is not None:
                file.close()


def print_compiledir_content():
    max_key_file_size = 1 * 1024 * 1024  # 1M

    compiledir = theano.config.compiledir
    table = []
    more_than_one_ops = 0
    zeros_op = 0
    big_key_files = []
    total_key_sizes = 0
    nb_keys = {}
    for dir in os.listdir(compiledir):
        file = None
        try:
            try:
                filename = os.path.join(compiledir, dir, "key.pkl")
                file = open(filename, 'rb')
                keydata = cPickle.load(file)
                ops = list(set([x for x in flatten(keydata.keys)
                                if isinstance(x, theano.gof.Op)]))
                if len(ops) == 0:
                    zeros_op += 1
                elif len(ops) > 1:
                    more_than_one_ops += 1
                else:
                    types = list(set([x for x in flatten(keydata.keys)
                                      if isinstance(x, theano.gof.Type)]))
                    table.append((dir, ops[0], types))

                size = os.path.getsize(filename)
                total_key_sizes += size
                if size > max_key_file_size:
                    big_key_files.append((dir, size, ops))

                nb_keys.setdefault(len(keydata.keys), 0)
                nb_keys[len(keydata.keys)] += 1
            except IOError:
                pass
        finally:
            if file is not None:
                file.close()

    print "List of %d compiled individual ops in this theano cache %s:" % (
        len(table), compiledir)
    print "sub directory/Op/a set of the different associated Theano type"
    table = sorted(table, key=lambda t: str(t[1]))
    table_op_class = {}
    for dir, op, types in table:
        print dir, op, types
        table_op_class.setdefault(op.__class__, 0)
        table_op_class[op.__class__] += 1

    print
    print ("List of %d individual compiled Op classes and "
           "the number of times they got compiled" % len(table_op_class))
    table_op_class = sorted(table_op_class.iteritems(), key=lambda t: t[1])
    for op_class, nb in table_op_class:
        print op_class, nb

    if big_key_files:
        big_key_files = sorted(big_key_files, key=lambda t: str(t[1]))
        big_total_size = sum([size for dir, size, ops in big_key_files])
        print ("There are directories with key files bigger than %d bytes "
               "(they probably contain big tensor constants)" %
               max_key_file_size)
        print ("They use %d bytes out of %d (total size used by all key files)"
               "" % (big_total_size, total_key_sizes))

        for dir, size, ops in big_key_files:
            print dir, size, ops

    nb_keys = sorted(nb_keys.iteritems())
    print
    print "Number of keys for a compiled module"
    print "number of keys/number of modules with that number of keys"
    for n_k, n_m in nb_keys:
        print n_k, n_m

    print ("Skipped %d files that contained more than"
           " 1 op (was compiled with the C linker)" % more_than_one_ops)
    print ("Skipped %d files that contained 0 op "
           "(are they always theano.scalar ops?)" % zeros_op)


def compiledir_purge():
    shutil.rmtree(config.compiledir)


def basecompiledir_ls():
    subdirs = []
    others = []
    for f in os.listdir(config.base_compiledir):
        if os.path.isdir(os.path.join(config.base_compiledir, f)):
            subdirs.append(f)
        else:
            others.append(f)

    subdirs = sorted(subdirs)
    others = sorted(others)

    print 'Base compile dir is %s' % theano.config.base_compiledir
    print 'Sub-directories (possible compile caches):'
    for d in subdirs:
        print '    %s' % d
    if not subdirs:
        print '    (None)'

    if others:
        print
        print 'Other files in base_compiledir:'
        for f in others:
            print '    %s' % f


def basecompiledir_purge():
    shutil.rmtree(config.base_compiledir)
