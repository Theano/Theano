from compilelock import get_lock, release_lock
import sys, os
from theano import config

try:
    if os.path.exists(os.path.join(config.compiledir,'cutils_ext.so')):
        os.remove(os.path.join(config.compiledir,'cutils_ext.so'))

    from cutils_ext.cutils_ext import *
except ImportError:
    #try to compile it manually

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
  int (*fn)(void*) = reinterpret_cast<int (*)(void*)>(ptr_addr);
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

    import cmodule
    import os, time
    loc=os.path.join(config.compiledir,'cutils_ext')
    if not os.path.exists(loc):
        try:
            os.makedirs(loc)
        except OSError:
            # This may happen when running multiple jobs in parallel, if they
            # attempt to create the same directory simultaneously.
            time.sleep(5) # May not be needed, but who knows with NFS.
            if not os.path.exists(loc):
                # Looks like something else is not working.
                raise

    cmodule.gcc_module_compile_str('cutils_ext', code, location = loc)
    from cutils_ext.cutils_ext import *
