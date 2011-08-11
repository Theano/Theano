import os, logging, sys

import theano
from theano import config
from theano.gof.compilelock import get_lock, release_lock
from theano.gof import cmodule

_logger = logging.getLogger('theano.scan_module.scan_perform')
logging.basicConfig(level=logging.DEBUG)


# Ensure the compiledir is in `sys.path` to be able to reload an existing
# precompiled library.
if config.compiledir not in sys.path:
    sys.path.append(config.compiledir)

version = 0.265 # must match constant returned in function get_version()

need_reload = False
try:
    import scan_perform
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
                reload(scan_perform)
            else:
                import scan_perform
                need_reload = True
            if version != getattr(scan_perform, '_version', None):
                raise ImportError()
        except ImportError:

            _logger.info("Compiling C code for scan")
            dirname = 'scan_perform'
            # We use a .txt extensions as otherwise it don't get
            # included when we create a package to send to pypi
            # This happen even if we tell to include *.c files
            cfile = os.path.join(theano.__path__[0], 'scan_module',
                                 'scan_perform.c.txt')
            code = open(cfile).read()
            loc = os.path.join(config.compiledir, dirname)
            if not os.path.exists(loc):
                os.mkdir(loc)
            cmodule.gcc_module_compile_str(dirname, code, location=loc,
                                           preargs = ['-pthread','-fwrapv',
                                                      '-O2',
                                                      '-fno-strict-aliasing'])
            # Save version into the __init__.py file.
            init_py = os.path.join(loc, '__init__.py')
            open(init_py, 'w').write('_version = %s\n' % version)
            # If we just compiled the module for the first time, then it was
            # imported at the same time: we need to make sure we do not
            # reload the now outdated __init__.pyc below.
            init_pyc = os.path.join(loc, '__init__.pyc')
            if os.path.isfile(init_pyc):
                os.remove(init_pyc)
            import scan_perform
            reload(scan_perform)
            from scan_perform import scan_perform as scan_c
            assert (scan_perform._version ==
                    scan_c.get_version())
            _logger.info("New version %s", scan_perform._version)
    finally:
        # Release lock on compilation directory.
        release_lock()

from scan_perform.scan_perform import *
assert version == get_version()
