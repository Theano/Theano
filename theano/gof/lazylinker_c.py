from __future__ import absolute_import, print_function, division
import errno
import logging
import os
from six.moves import reload_module as reload
import sys
import warnings


import theano
from theano import config
from theano.gof.compilelock import get_lock, release_lock
from theano.gof import cmodule

_logger = logging.getLogger('theano.gof.lazylinker_c')

force_compile = False
version = 0.211  # must match constant returned in function get_version()
lazylinker_ext = None


def try_import():
    global lazylinker_ext
    sys.path[0:0] = [config.compiledir]
    import lazylinker_ext  # noqa
    del sys.path[0]


def try_reload():
    sys.path[0:0] = [config.compiledir]
    reload(lazylinker_ext)
    del sys.path[0]

try:
    # See gh issue #728 for why these lines are here. Summary: compiledir must
    # be at the beginning of the path to avoid conflicts with any other
    # lazylinker_ext modules that might exist (this step handled in try_import
    # and try_reload). An __init__.py file must be created for the same reason.
    # Note that these lines may seem redundant (they are repeated in
    # compile_str()) but if another lazylinker_ext does exist then it will be
    # imported and compile_str won't get called at all.
    location = os.path.join(config.compiledir, 'lazylinker_ext')
    if not os.path.exists(location):
        try:
            # Try to make the location
            os.mkdir(location)
        except OSError as e:
            # If we get an error, verify that the error was # 17, the
            # path already exists, and that it is a directory Note: we
            # can't check if it exists before making it, because we
            # are not holding the lock right now, so we could race
            # another process and get error 17 if we lose the race
            assert e.errno == errno.EEXIST
            assert os.path.isdir(location)

    init_file = os.path.join(location, '__init__.py')
    if not os.path.exists(init_file):
        try:
            open(init_file, 'w').close()
        except IOError as e:
            if os.path.exists(init_file):
                pass  # has already been created
            else:
                e.args += ('%s exist? %s' % (location,
                                             os.path.exists(location)),)
                raise

    _need_reload = False
    if force_compile:
        raise ImportError()
    else:
        try_import()
        _need_reload = True
        if version != getattr(lazylinker_ext, '_version', None):
            raise ImportError()
except ImportError:
    get_lock()
    try:
        # Maybe someone else already finished compiling it while we were
        # waiting for the lock?
        try:
            if force_compile:
                raise ImportError()
            if _need_reload:
                # The module was successfully imported earlier: we need to
                # reload it to check if the version was updated.
                try_reload()
            else:
                try_import()
                _need_reload = True
            if version != getattr(lazylinker_ext, '_version', None):
                raise ImportError()
        except ImportError:
            # It is useless to try to compile if there isn't any
            # compiler!  But we still want to try to load it, in case
            # the cache was copied from another computer.
            if not theano.config.cxx:
                raise
            _logger.info("Compiling new CVM")
            dirname = 'lazylinker_ext'
            cfile = os.path.join(theano.__path__[0], 'gof', 'c_code', 'lazylinker_c.c')
            if not os.path.exists(cfile):
                # This can happen in not normal case. We just
                # disable the c clinker. If we are here the user
                # didn't disable the compiler, so print a warning.
                warnings.warn(
                    "The file lazylinker_c.c is not available. This do"
                    "not happen normally. You are probably in a strange"
                    "setup. This mean Theano can not use the cvm:"
                    "our c execution engine for Theano function. If you"
                    "want to remove this warning, use the Theano flag"
                    "'cxx=' (set to an empty string) to disable all c"
                    "code generation."
                )
                raise ImportError("The file lazylinker_c.c is not available.")
            code = open(cfile).read()
            loc = os.path.join(config.compiledir, dirname)
            if not os.path.exists(loc):
                try:
                    os.mkdir(loc)
                except OSError as e:
                    assert e.errno == errno.EEXIST
                    assert os.path.exists(loc)

            args = cmodule.GCC_compiler.compile_args()
            cmodule.GCC_compiler.compile_str(dirname, code, location=loc,
                                             preargs=args)
            # Save version into the __init__.py file.
            init_py = os.path.join(loc, '__init__.py')
            open(init_py, 'w').write('_version = %s\n' % version)
            # If we just compiled the module for the first time, then it was
            # imported at the same time: we need to make sure we do not
            # reload the now outdated __init__.pyc below.
            init_pyc = os.path.join(loc, '__init__.pyc')
            if os.path.isfile(init_pyc):
                os.remove(init_pyc)
            try_import()
            try_reload()
            from lazylinker_ext import lazylinker_ext as lazy_c
            assert (lazylinker_ext._version ==
                    lazy_c.get_version())
            _logger.info("New version %s", lazylinker_ext._version)
    finally:
        # Release lock on compilation directory.
        release_lock()

from lazylinker_ext.lazylinker_ext import *  # noqa
assert force_compile or (version == get_version())  # noqa
