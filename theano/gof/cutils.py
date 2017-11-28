from __future__ import absolute_import, print_function, division
import errno
import os
import sys

from theano.compat import PY3
from theano.gof.compilelock import get_lock, release_lock
from theano import config
from . import cmodule

# TODO These two lines may be removed in the future, when we are 100% sure
# no one has an old cutils_ext.so lying around anymore.
if os.path.exists(os.path.join(config.compiledir, 'cutils_ext.so')):
    os.remove(os.path.join(config.compiledir, 'cutils_ext.so'))


def compile_cutils():
    """
    Do just the compilation of cutils_ext.

    """
    code = ("""
        #include <Python.h>
        #include "theano_mod_helper.h"

        extern "C"{
        static PyObject *
        run_cthunk(PyObject *self, PyObject *args)
        {
          PyObject *py_cthunk = NULL;
          if(!PyArg_ParseTuple(args,"O",&py_cthunk))
            return NULL;

          if (!PyCObject_Check(py_cthunk)) {
            PyErr_SetString(PyExc_ValueError,
                           "Argument to run_cthunk must be a PyCObject.");
            return NULL;
          }
          void * ptr_addr = PyCObject_AsVoidPtr(py_cthunk);
          int (*fn)(void*) = (int (*)(void*))(ptr_addr);
          void* it = PyCObject_GetDesc(py_cthunk);
          int failure = fn(it);

          return Py_BuildValue("i", failure);
         }
         static PyMethodDef CutilsExtMethods[] = {
            {"run_cthunk",  run_cthunk, METH_VARARGS|METH_KEYWORDS,
             "Run a theano cthunk."},
            {NULL, NULL, 0, NULL}        /* Sentinel */
        };""")
    if PY3:
        # This is not the most efficient code, but it is written this way to
        # highlight the changes needed to make 2.x code compile under python 3.
        code = code.replace("<Python.h>", '"numpy/npy_3kcompat.h"', 1)
        code = code.replace("PyCObject", "NpyCapsule")
        code += """
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "cutils_ext",
            NULL,
            -1,
            CutilsExtMethods,
        };

        PyMODINIT_FUNC
        PyInit_cutils_ext(void) {
            return PyModule_Create(&moduledef);
        }
        }
        """
    else:
        code += """
        PyMODINIT_FUNC
        initcutils_ext(void)
        {
          (void) Py_InitModule("cutils_ext", CutilsExtMethods);
        }
    } //extern C
        """

    loc = os.path.join(config.compiledir, 'cutils_ext')
    if not os.path.exists(loc):
        try:
            os.mkdir(loc)
        except OSError as e:
            assert e.errno == errno.EEXIST
            assert os.path.exists(loc), loc

    args = cmodule.GCC_compiler.compile_args(march_flags=False)
    cmodule.GCC_compiler.compile_str('cutils_ext', code, location=loc,
                                     preargs=args)

try:
    # See gh issue #728 for why these lines are here. Summary: compiledir
    # must be at the beginning of the path to avoid conflicts with any other
    # cutils_ext modules that might exist. An __init__.py file must be created
    # for the same reason. Note that these 5 lines may seem redundant (they are
    # repeated in compile_str()) but if another cutils_ext does exist then it
    # will be imported and compile_str won't get called at all.
    sys.path.insert(0, config.compiledir)
    location = os.path.join(config.compiledir, 'cutils_ext')
    if not os.path.exists(location):
        try:
            os.mkdir(location)
        except OSError as e:
            assert e.errno == errno.EEXIST
            assert os.path.exists(location), location
    if not os.path.exists(os.path.join(location, '__init__.py')):
        open(os.path.join(location, '__init__.py'), 'w').close()

    try:
        from cutils_ext.cutils_ext import *  # noqa
    except ImportError:
        get_lock()
    # Ensure no-one else is currently modifying the content of the compilation
    # directory. This is important to prevent multiple processes from trying to
    # compile the cutils_ext module simultaneously.
        try:
            try:
                # We must retry to import it as some other process could
                # have been compiling it between the first failed import
                # and when we receive the lock
                from cutils_ext.cutils_ext import *  # noqa
            except ImportError:

                compile_cutils()
                from cutils_ext.cutils_ext import *  # noqa

        finally:
            # Release lock on compilation directory.
            release_lock()
finally:
    if sys.path[0] == config.compiledir:
        del sys.path[0]
