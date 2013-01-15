import errno
import os, logging, sys

import theano
from theano import config
from theano.gof.compilelock import get_lock, release_lock
from theano.gof import cmodule

_logger = logging.getLogger('theano.gof.lazylinker_c')

force_compile = False
version = 0.20  # must match constant returned in function get_version()

def try_import():
    global lazylinker_ext
    sys.path[0:0] = [config.compiledir]
    import lazylinker_ext
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
        except OSError, e:
            # If we get an error, verify that the error was # 17, the path already exists,
            # and that it is a directory
            # Note: we can't check if it exists before making it, because we are not holding
            # the lock right now, so we could race another process and get error 17 if we lose
            # the race
            assert e.errno == errno.EEXIST
            assert os.path.isdir(location)

    if not os.path.exists(os.path.join(location, '__init__.py')):
        file(os.path.join(location, '__init__.py'), 'w').close()

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
            _logger.info("Compiling new CVM")
            dirname = 'lazylinker_ext'
            # We use a .txt extensions as otherwise it don't get
            # included when we create a package to send to pypi
            # This happen even if we tell to include *.c files
            cfile = os.path.join(theano.__path__[0], 'gof', 'lazylinker_c.c.txt')
            code = open(cfile).read()
            loc = os.path.join(config.compiledir, dirname)
            if not os.path.exists(loc):
                os.mkdir(loc)
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

from lazylinker_ext.lazylinker_ext import *
assert force_compile or (version == get_version())
