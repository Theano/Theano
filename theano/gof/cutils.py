import os, sys
from compilelock import get_lock, release_lock
from theano import config

# TODO These two lines may be removed in the future, when we are 100% sure
# noone has an old cutils_ext.so lying around anymore.
if os.path.exists(os.path.join(config.compiledir,'cutils_ext.so')):
    os.remove(os.path.join(config.compiledir,'cutils_ext.so'))

# Ensure no-one else is currently modifying the content of the compilation
# directory. This is important to prevent multiple processes from trying to
# compile the cutils_ext module simultaneously.
try:
    # If we load a previously-compiled version, config.compiledir should
    # be in sys.path.
    if config.compiledir not in sys.path:
        sys.path.append(config.compiledir)
    from cutils_ext.cutils_ext import *
except ImportError:
    import cmodule

    get_lock()
    try:
        try:
            # We must retry to import it as some other processs could
            # have been compiling it between the first failed import
            # and when we receive the lock
            from cutils_ext.cutils_ext import *
        except ImportError:
            import cmodule

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
            from cutils_ext.cutils_ext import *

    finally:
        # Release lock on compilation directory.
        release_lock()
