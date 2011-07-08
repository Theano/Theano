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
try:
    import lazylinker_ext
    try:
        import lazylinker_ext.lazylinker_ext
        get_version = lazylinker_ext.lazylinker_ext.get_version
    except:
        get_version = lambda: None
    if version != get_version():
        raise ImportError()
except ImportError:
    get_lock()
    try:
        # Maybe someone else already finished compiling it while we were
        # waiting for the lock?
        try:
            import lazylinker_ext
            try:
                import lazylinker_ext.lazylinker_ext
                get_version = lazylinker_ext.lazylinker_ext.get_version
            except:
                get_version = lambda: None
            if version != get_version():
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
            print "NEW VERSION", lazylinker_ext.lazylinker_ext.get_version()
    finally:
        # Release lock on compilation directory.
        release_lock()

from lazylinker_ext.lazylinker_ext import *
assert version == get_version()
