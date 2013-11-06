import os, logging, sys

import theano
from theano import config
from theano.compat import reload
from theano.gof.compilelock import get_lock, release_lock
from theano.gof import cmodule


_logger = logging.getLogger('theano.scan_module.scan_perform')
_logger.setLevel(logging.WARN)


version = 0.280  # must match constant returned in function get_version()

need_reload = False


def try_import():
    global scan_perform
    sys.path[0:0] = [config.compiledir]
    import scan_perform
    del sys.path[0]


def try_reload():
    sys.path[0:0] = [config.compiledir]
    reload(scan_perform)
    del sys.path[0]

try:
    try_import()
    need_reload = True
    if version != getattr(scan_perform, '_version', None):
        raise ImportError()
except ImportError:
    get_lock()
    try:
        # Maybe someone else already finished compiling it while we were
        # waiting for the lock?
        try:
            if need_reload:
                # The module was successfully imported earlier: we need to
                # reload it to check if the version was updated.
                try_reload()
            else:
                try_import()
                need_reload = True
            if version != getattr(scan_perform, '_version', None):
                raise ImportError()
        except ImportError:

            _logger.info("Compiling C code for scan")
            dirname = 'scan_perform'
            cfile = os.path.join(theano.__path__[0], 'scan_module',
                                 'scan_perform.c')
            code = open(cfile).read()
            loc = os.path.join(config.compiledir, dirname)
            if not os.path.exists(loc):
                os.mkdir(loc)
            preargs = ['-fwrapv', '-O2', '-fno-strict-aliasing']
            preargs += cmodule.GCC_compiler.compile_args()
            cmodule.GCC_compiler.compile_str(dirname, code, location=loc,
                                             preargs=preargs)
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
            from scan_perform import scan_perform as scan_c
            assert (scan_perform._version ==
                    scan_c.get_version())
            _logger.info("New version %s", scan_perform._version)
    finally:
        # Release lock on compilation directory.
        release_lock()

from scan_perform.scan_perform import *
assert version == get_version()
