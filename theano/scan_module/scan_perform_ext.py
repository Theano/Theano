"""
To update the scan Cython code in Theano you must
- update the version in this file and scan_perform.py
- call "cd theano/scan_module/; cython scan_perform.pyx; patch scan_perform.c numpy_api_changes.diff"

"""

from __future__ import absolute_import, print_function, division
import errno
import logging
import os
import sys
import warnings

import numpy as np

import theano
from theano import config
from theano.compat import reload
from theano.gof.compilelock import get_lock, release_lock
from theano.gof import cmodule


_logger = logging.getLogger('theano.scan_module.scan_perform')


version = 0.296  # must match constant returned in function get_version()

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
            if not theano.config.cxx:
                raise ImportError("no c compiler, can't compile cython code")
            _logger.info("Compiling C code for scan")
            dirname = 'scan_perform'
            cfile = os.path.join(theano.__path__[0], 'scan_module', 'c_code',
                                 'scan_perform.c')
            if not os.path.exists(cfile):
                # This can happen in not normal case. We just
                # disable the cython code. If we are here the user
                # didn't disable the compiler, so print a warning.
                warnings.warn(
                    "The file scan_perform.c is not available. This do"
                    "not happen normally. You are probably in a strange"
                    "setup. This mean Theano can not use the cython code for "
                    "scan. If you"
                    "want to remove this warning, use the Theano flag"
                    "'cxx=' (set to an empty string) to disable all c"
                    "code generation."
                )
                raise ImportError("The file lazylinker_c.c is not available.")

            with open(cfile) as f:
                code = f.read()
            loc = os.path.join(config.compiledir, dirname)
            if not os.path.exists(loc):
                try:
                    os.mkdir(loc)
                except OSError as e:
                    assert e.errno == errno.EEXIST
                    assert os.path.exists(loc)

            preargs = ['-fwrapv', '-O2', '-fno-strict-aliasing']
            preargs += cmodule.GCC_compiler.compile_args()
            # Cython 19.1 always use the old NumPy interface.  So we
            # need to manually modify the .c file to get it compiled
            # by Theano. As by default, we tell NumPy to don't import
            # the old interface.
            if False:
                # During scan cython development, it is helpful to keep the old interface, to don't manually edit the c file each time.
                preargs.remove('-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION')
            else:
                numpy_ver = [int(n) for n in np.__version__.split('.')[:2]]
                # Add add some macro to lower the number of edit
                # needed to the c file.
                if bool(numpy_ver >= [1, 7]):
                    # Needed when we disable the old API, as cython
                    # use the old interface
                    preargs.append("-DNPY_ENSUREARRAY=NPY_ARRAY_ENSUREARRAY")
                    preargs.append("-DNPY_ENSURECOPY=NPY_ARRAY_ENSURECOPY")
                    preargs.append("-DNPY_ALIGNED=NPY_ARRAY_ALIGNED")
                    preargs.append("-DNPY_WRITEABLE=NPY_ARRAY_WRITEABLE")
                    preargs.append("-DNPY_UPDATE_ALL=NPY_ARRAY_UPDATE_ALL")
                    preargs.append("-DNPY_C_CONTIGUOUS=NPY_ARRAY_C_CONTIGUOUS")
                    preargs.append("-DNPY_F_CONTIGUOUS=NPY_ARRAY_F_CONTIGUOUS")

            cmodule.GCC_compiler.compile_str(dirname, code, location=loc,
                                             preargs=preargs,
                                             hide_symbols=False)
            # Save version into the __init__.py file.
            init_py = os.path.join(loc, '__init__.py')
            with open(init_py, 'w') as f:
                f.write('_version = %s\n' % version)
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

# This is caused as cython use the old NumPy C-API but we use the new one.
# To fix it completly, we would need to modify Cython to use the new API.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",
                            message="numpy.ndarray size changed")
    from scan_perform.scan_perform import *
assert version == get_version()
