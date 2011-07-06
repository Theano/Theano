import os
import theano
from theano import config
from theano.gof.compilelock import get_lock, release_lock
from theano.gof import cmodule

get_lock()
try:
    dirname = 'lazylinker_ext'
    cfile = os.path.join(theano.__path__[0], 'gof', 'lazylinker_c.c')
    code = open(cfile).read()
    loc = os.path.join(config.compiledir, dirname)
    if not os.path.exists(loc):
        os.mkdir(loc)
    cmodule.gcc_module_compile_str(dirname, code, location=loc)
    from lazylinker_ext.lazylinker_ext import *
finally:
    # Release lock on compilation directory.
    release_lock()

