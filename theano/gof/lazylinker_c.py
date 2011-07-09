import os, sys

import theano
from theano import config
from theano.gof.compilelock import get_lock, release_lock
from theano.gof import cmodule

# Ensure the compiledir is in `sys.path` to be able to reload an existing
# precompiled library.
if config.compiledir not in sys.path:
    sys.path.append(config.compiledir)

version = 0.1 # must match constant returned in function get_version()

need_reload = False
try:
    import lazylinker_ext
    need_reload = True
    if version != getattr(lazylinker_ext, '_version', None):
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
                reload(lazylinker_ext)
            else:
                import lazylinker_ext
                need_reload = True
            if version != getattr(lazylinker_ext, '_version', None):
                raise ImportError()
        except ImportError:
            print "COMPILING NEW CVM"
            dirname = 'lazylinker_ext'
            cfile = os.path.join(theano.__path__[0], 'gof', 'lazylinker_c.c')
            code = open(cfile).read()
            loc = os.path.join(config.compiledir, dirname)
            if not os.path.exists(loc):
                os.mkdir(loc)
            cmodule.gcc_module_compile_str(dirname, code, location=loc)
            # Save version into the __init__.py file.
            init_py = os.path.join(loc, '__init__.py')
            open(init_py, 'w').write('_version = %s\n' % version)
            # If we just compiled the module for the first time, then it was
            # imported at the same time: we need to make sure we do not
            # reload the now outdated __init__.pyc below.
            init_pyc = os.path.join(loc, '__init__.pyc')
            if os.path.isfile(init_pyc):
                os.remove(init_pyc)
            import lazylinker_ext
            reload(lazylinker_ext)
            from lazylinker_ext import lazylinker_ext as lazy_c
            assert (lazylinker_ext._version ==
                    lazy_c.get_version())
            print "NEW VERSION", lazylinker_ext._version
    finally:
        # Release lock on compilation directory.
        release_lock()

from lazylinker_ext.lazylinker_ext import *
assert version == get_version()
