from __future__ import absolute_import, print_function, division
import errno
import os
import sys

from theano.compat import PY3
from theano.gof.compilelock import get_lock, release_lock
from theano import config
from . import cmodule

# TODO These two lines may be removed in the future, when we are 100% sure
# noone has an old cutils_ext.so lying around anymore.
if os.path.exists(os.path.join(config.compiledir, 'cutils_ext.so')):
    os.remove(os.path.join(config.compiledir, 'cutils_ext.so'))


def compile_cutils_code():
    types = ['npy_' + t for t in ['int8', 'int16', 'int32', 'int64', 'int128',
                                  'int256', 'uint8', 'uint16', 'uint32',
                                  'uint64', 'uint128', 'uint256',
                                  'float16', 'float32', 'float64',
                                  'float80', 'float96', 'float128',
                                  'float256']]

    complex_types = ['npy_' + t for t in ['complex32', 'complex64',
                                          'complex128', 'complex160',
                                          'complex192', 'complex512']]

    inplace_map_template = """
    #if defined(%(typen)s)
    static void %(type)s_inplace_add(PyArrayMapIterObject *mit,
                                     PyArrayIterObject *it, int inc_or_set)
    {
        int index = mit->size;
        while (index--) {
            %(op)s

            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
    }
    #endif
    """

    floatadd = ("((%(type)s*)mit->dataptr)[0] = "
                "(inc_or_set ? ((%(type)s*)mit->dataptr)[0] : 0)"
                " + ((%(type)s*)it->dataptr)[0];")
    complexadd = """
    ((%(type)s*)mit->dataptr)[0].real =
        (inc_or_set ? ((%(type)s*)mit->dataptr)[0].real : 0)
        + ((%(type)s*)it->dataptr)[0].real;
    ((%(type)s*)mit->dataptr)[0].imag =
        (inc_or_set ? ((%(type)s*)mit->dataptr)[0].imag : 0)
        + ((%(type)s*)it->dataptr)[0].imag;
    """

    fns = ''.join([inplace_map_template % {'type': t, 'typen': t.upper(),
                                           'op': floatadd % {'type': t}}
                   for t in types] +
                  [inplace_map_template % {'type': t, 'typen': t.upper(),
                                           'op': complexadd % {'type': t}}
                   for t in complex_types])

    def gen_binop(type, typen):
        return """
#if defined(%(typen)s)
%(type)s_inplace_add,
#endif
""" % dict(type=type, typen=typen)

    fn_array = ("static inplace_map_binop addition_funcs[] = {" +
                ''.join([gen_binop(type=t, typen=t.upper())
                         for t in types + complex_types]) + "NULL};\n")

    def gen_num(typen):
        return """
#if defined(%(typen)s)
%(typen)s,
#endif
""" % dict(type=type, typen=typen)

    type_number_array = ("static int type_numbers[] = {" +
                         ''.join([gen_num(typen=t.upper())
                                  for t in types + complex_types]) + "-1000};")

    code = ("""
        #if NPY_API_VERSION >= 0x00000008
        typedef void (*inplace_map_binop)(PyArrayMapIterObject *,
                                          PyArrayIterObject *, int inc_or_set);
        """ + fns + fn_array + type_number_array + """
static int
map_increment(PyArrayMapIterObject *mit, PyObject *op,
              inplace_map_binop add_inplace, int inc_or_set)
{
    PyArrayObject *arr = NULL;
    PyArrayIterObject *it;
    PyArray_Descr *descr;
    if (mit->ait == NULL) {
        return -1;
    }
    descr = PyArray_DESCR(mit->ait->ao);
    Py_INCREF(descr);
    arr = (PyArrayObject *)PyArray_FromAny(op, descr,
                                0, 0, NPY_ARRAY_FORCECAST, NULL);
    if (arr == NULL) {
        return -1;
    }
    if ((mit->subspace != NULL) && (mit->consec)) {
        PyArray_MapIterSwapAxes(mit, (PyArrayObject **)&arr, 0);
        if (arr == NULL) {
            return -1;
        }
    }
    it = (PyArrayIterObject*)
            PyArray_BroadcastToShape((PyObject*)arr, mit->dimensions, mit->nd);
    if (it  == NULL) {
        Py_DECREF(arr);
        return -1;
    }

    (*add_inplace)(mit, it, inc_or_set);

    Py_DECREF(arr);
    Py_DECREF(it);
    return 0;
}


static PyObject *
inplace_increment(PyObject *dummy, PyObject *args)
{
    PyObject *arg_a = NULL, *index=NULL, *inc=NULL;
    int inc_or_set = 1;
    PyArrayObject *a;
    inplace_map_binop add_inplace = NULL;
    int type_number = -1;
    int i = 0;
    PyArrayMapIterObject * mit;

    if (!PyArg_ParseTuple(args, "OOO|i", &arg_a, &index,
            &inc, &inc_or_set)) {
        return NULL;
    }
    if (!PyArray_Check(arg_a)) {
        PyErr_SetString(PyExc_ValueError,
                        "needs an ndarray as first argument");
        return NULL;
    }

    a = (PyArrayObject *) arg_a;

    if (PyArray_FailUnlessWriteable(a, "input/output array") < 0) {
        return NULL;
    }

    if (PyArray_NDIM(a) == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
        return NULL;
    }
    type_number = PyArray_TYPE(a);



    while (type_numbers[i] >= 0 && addition_funcs[i] != NULL){
        if (type_number == type_numbers[i]) {
            add_inplace = addition_funcs[i];
            break;
        }
        i++ ;
    }

    if (add_inplace == NULL) {
        PyErr_SetString(PyExc_TypeError, "unsupported type for a");
        return NULL;
    }
    mit = (PyArrayMapIterObject *) PyArray_MapIterArray(a, index);
    if (mit == NULL) {
        goto fail;
    }
    if (map_increment(mit, inc, add_inplace, inc_or_set) != 0) {
        goto fail;
    }

    Py_DECREF(mit);

    Py_INCREF(Py_None);
    return Py_None;

fail:
    Py_XDECREF(mit);

    return NULL;
}
        #endif
""")

    return code


def compile_cutils():
    """
    Do just the compilation of cutils_ext.

    """
    code = ("""
        #include <Python.h>
        #include "numpy/arrayobject.h"
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
         }""")

    code += compile_cutils_code()

    code += ("""static PyMethodDef CutilsExtMethods[] = {
            {"run_cthunk",  run_cthunk, METH_VARARGS|METH_KEYWORDS,
             "Run a theano cthunk."},
            #if NPY_API_VERSION >= 0x00000008
            {"inplace_increment",  inplace_increment,
              METH_VARARGS,
             "increments a numpy array inplace at the passed indexes."},
            #endif
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
            import_array();
            return PyModule_Create(&moduledef);
        }
        }
        """
    else:
        code += """
        PyMODINIT_FUNC
        initcutils_ext(void)
        {
          import_array();
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
