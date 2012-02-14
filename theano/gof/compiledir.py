import cPickle
import errno
import os
import platform
import re
import shutil
import sys
import textwrap

import numpy

import theano
from theano.configparser import config, AddConfigVar, ConfigParam, StrParam

compiledir_format_dict = {"platform": platform.platform(),
                          "processor": platform.processor(),
                          "python_version": platform.python_version(),
                          "theano_version": theano.__version__,
                         }
compiledir_format_keys = ", ".join(compiledir_format_dict.keys())
default_compiledir_format =\
                    "compiledir_%(platform)s-%(processor)s-%(python_version)s"

AddConfigVar("compiledir_format",
             textwrap.fill(textwrap.dedent("""\
                 Format string for platform-dependent compiled
                 module subdirectory (relative to base_compiledir).
                 Available keys: %s. Defaults to %r.
             """ % (compiledir_format_keys, default_compiledir_format))),
             StrParam(default_compiledir_format, allow_override=False))


def default_compiledirname():
    formatted = config.compiledir_format % compiledir_format_dict
    safe = re.sub("[\(\)\s,]+", "_", formatted)
    return safe


def filter_compiledir(path):
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


AddConfigVar('base_compiledir',
        "platform-independent root directory for compiled modules",
        StrParam(default_base_compiledir, allow_override=False))

AddConfigVar('compiledir',
        "platform-dependent cache directory for compiled modules",
        ConfigParam(
            os.path.join(
                os.path.expanduser(config.base_compiledir),
                default_compiledirname()),
            filter=filter_compiledir,
            allow_override=False))


def flatten(a):
    if isinstance(a, (tuple, list, set)):
        l = []
        for item in a:
            l.extend(flatten(item))
        return l
    else:
        return [a]


def cleanup():
    """
    Delete keys in old format from the compiledir.

    We define keys in old format as keys that have an ndarray in them.
    Now we use a hash in the keys of the constant data.

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
                        for obj in flatten(key):
                            if isinstance(obj, numpy.ndarray):
                                keydata.remove_key(key)
                                break
                    if len(keydata.keys) == 0:
                        shutil.rmtree(os.path.join(compiledir, directory))

                except EOFError:
                    print ("ERROR while reading this key file '%s'."
                           " Delete its directory" % filename)
            except IOError:
                pass
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
