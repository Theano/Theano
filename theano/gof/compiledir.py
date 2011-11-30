import cPickle
import errno
import os
import platform
import re
import sys
import warnings

import theano
from theano.configparser import config, AddConfigVar, ConfigParam, StrParam


def default_compiledirname():
    platform_id = '-'.join([
        platform.platform(),
        platform.processor(),
        platform.python_version()])
    platform_id = re.sub("[\(\)\s,]+", "_", platform_id)
    return 'compiledir_' + platform_id


def filter_compiledir(path):
    # Turn path into the 'real' path. This ensures that:
    #   1. There is no relative path, which would fail e.g. when trying to
    #      import modules from the compile dir.
    #   2. The path is stable w.r.t. e.g. symlinks (which makes it easier
    #      to re-use compiled modules).
    path = os.path.realpath(path)
    valid = True
    if not os.access(path, os.R_OK | os.W_OK):
        try:
            os.makedirs(path, 0770)  # read-write-execute for user and group
        except OSError, e:
            # Maybe another parallel execution of theano was trying to create
            # the same directory at the same time.
            if e.errno != errno.EEXIST:
                valid = False

    if valid:
        try:
            # PROBLEM: sometimes the initial approach based on
            # os.system('touch') returned -1 for an unknown reason; the
            # alternate approach here worked in all cases... it was weird.
            open(os.path.join(path, '__init__.py'), 'w').close()
        except:
            valid = False

    if not valid:
        raise ValueError('Invalid value for compiledir: %s' % path)

    return path


AddConfigVar('base_compiledir',
        "arch-independent cache directory for compiled modules",
        StrParam(config.home, allow_override=False))

default_compiledir = os.path.join(
        os.path.expanduser(config.base_compiledir),
        default_compiledirname())

AddConfigVar('compiledir',
        "arch-dependent cache directory for compiled modules",
        ConfigParam(default_compiledir,
                    filter=filter_compiledir,
                    allow_override=False))


# The role of `config.home` was changed compared to Theano 0.4.1. It used to
# be the location of the user home directory. Now, it directly points to the
# directory where Theano should save its own data files. Typically, the
# difference is that it now includes the '.theano' folder, while it used to be
# the parent of that folder. In order for this change to be safe, we currently
# prevent people from changing `config.home` unless they also change
# the compilation directory.
if (config.home != theano.configdefaults.default_home and
    config.base_compiledir == config.home and
    # We need to compare to the `real` path because this is what is used in
    # `filter_compiledir`.
    config.compiledir == os.path.realpath(default_compiledir)):
    # The user changed `config.home` but is still using the default values for
    # `config.base_compiledir` and `config.compiledir`.
    raise RuntimeError(
            'You manually set your Theano home directory (to %s), but kept '
            'the default compilation directory. Please note that the meaning '
            'of the `home` directory was changed: it is now the base '
            'directory for all Theano files, instead of being its parent '
            'directory. To get rid of this error, please set the '
            '`base_compiledir` config option to the directory you want '
            'compiled files to be stored into (for instance: %s).' % (
                config.home,
                os.path.join(config.home, '.theano')))


def print_compiledir_content():

    def flatten(a):
        if isinstance(a, (tuple, list, set)):
            l = []
            for item in a:
                l.extend(flatten(item))
            return l
        else:
            return [a]

    compiledir = theano.config.compiledir
    print "List compiled ops in this theano cache:", compiledir
    print "sub directory/Op/Associated Type"
    print
    table = []

    for dir in os.listdir(compiledir):
        file = None
        try:
            try:
                file = open(os.path.join(compiledir, dir, "key.pkl"))
                keydata = cPickle.load(file)
                ops = list(set([x for x in flatten(keydata.keys)
                                if isinstance(x, theano.gof.Op)]))
                assert len(ops) == 1
                types = list(set([x for x in flatten(keydata.keys)
                                  if isinstance(x, theano.gof.Type)]))
                table.append((dir, ops[0], types))
            except IOError:
                pass
        finally:
            if file is not None:
                file.close()

    table = sorted(table, key=lambda t: str(t[1]))
    for dir, op, types in table:
        print dir, op, types
