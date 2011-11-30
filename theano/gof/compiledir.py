
import cPickle
import errno
import os
import platform
import re
import sys

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


# TODO Using the local user profile on Windows is currently disabled as it
# is not documented yet, and may break some existing code. It will be enabled
# in a future code update.
if False and sys.platform == 'win32':
    # On Windows we should not write temporary files to a directory 
    # that is part of the roaming part of the user profile. Instead
    # we use the local part of the user profile.
    basecompiledir = os.path.join(os.environ['LOCALAPPDATA'], 'theano')
else:
    basecompiledir = os.path.join(config.home, '.theano')
AddConfigVar('base_compiledir',
        "arch-independent cache directory for compiled modules",
        StrParam(basecompiledir, allow_override=False))

AddConfigVar('compiledir',
        "arch-dependent cache directory for compiled modules",
        ConfigParam(
            os.path.join(
                os.path.expanduser(config.base_compiledir),
                default_compiledirname()),
            filter=filter_compiledir,
            allow_override=False))


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
