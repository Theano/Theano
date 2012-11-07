import os
import sys
from compilelock import get_lock, release_lock
from theano import config

# TODO These two lines may be removed in the future, when we are 100% sure
# noone has an old cutils_ext.so lying around anymore.
if os.path.exists(os.path.join(config.compiledir, 'cutils_ext.so')):
    os.remove(os.path.join(config.compiledir, 'cutils_ext.so'))


def compile_cutils():
    """Do just the compilation of cutils_ext"""
    code = """
        #include <Python.h>
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
        };

        PyMODINIT_FUNC
        initcutils_ext(void)
        {
          (void) Py_InitModule("cutils_ext", CutilsExtMethods);
        }
        }
        """

    loc = os.path.join(config.compiledir, 'cutils_ext')
    if not os.path.exists(loc):
        os.mkdir(loc)

    args = cmodule.GCC_compiler.compile_args()
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
        os.mkdir(location)
    if not os.path.exists(os.path.join(location, '__init__.py')):
        file(os.path.join(location, '__init__.py'), 'w').close()

    try:
        from cutils_ext.cutils_ext import *
    except ImportError:
        import cmodule

        get_lock()
    # Ensure no-one else is currently modifying the content of the compilation
    # directory. This is important to prevent multiple processes from trying to
    # compile the cutils_ext module simultaneously.
        try:
            try:
                # We must retry to import it as some other process could
                # have been compiling it between the first failed import
                # and when we receive the lock
                from cutils_ext.cutils_ext import *
            except ImportError:
                import cmodule

                compile_cutils()
                from cutils_ext.cutils_ext import *

        finally:
            # Release lock on compilation directory.
            release_lock()
finally:
    if sys.path[0] == config.compiledir:
        del sys.path[0]
