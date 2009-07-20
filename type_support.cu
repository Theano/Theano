#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>

#include "cuda_ndarray.cuh"

#define DECL(s) static PyObject * s(PyObject * self, PyObject *args)

static PyObject * 
filter(PyObject* self, PyObject *args) // args = (data, typenum, broadcastable, strict)
{
    PyObject *py_data=NULL;
    PyArrayObject * data = NULL;
    int dtype_typenum=-1;
    int strict = 0;
    PyObject * broadcastable=NULL;

    if (!PyArg_ParseTuple(args, "OiOi", &py_data, &dtype_typenum, &broadcastable, &strict)) return NULL;

    if (!PyTuple_Check(broadcastable)){
        PyErr_SetString(PyExc_TypeError, "broadcastable arg should be a tuple of int.");
        return NULL;
    }
    Py_XINCREF(py_data);
    Py_XINCREF(broadcastable);

    CudaNdarray * cnda = (CudaNdarray*)py_data;

    if (strict or CudaNdarray_Check(py_data))
    {
        //TODO: support non-strict "casting" from a vt to the broadcastable/type/size that we need.
        if (!CudaNdarray_Check(py_data)) 
        {
            Py_XDECREF(py_data);
            Py_XDECREF(broadcastable);
            PyErr_SetString(PyExc_TypeError, "strict mode requires CudaNdarray");
            return NULL;
        }
        if (cnda->nd != PyTuple_Size(broadcastable))
        {
            Py_XDECREF(py_data);
            Py_XDECREF(broadcastable);
            PyErr_Format(PyExc_TypeError, "Wrong rank: %i vs %li", cnda->nd, (long)PyTuple_Size(broadcastable));
            return NULL;
        }
        for (int i = 0; i < cnda->nd; ++i)
        {
            if ((cnda->dim[i] > 1) and PyInt_AsLong(PyTuple_GetItem(broadcastable, Py_ssize_t(i))))
            {
                PyErr_Format(PyExc_TypeError, "Non-unit size in broadcastable vt dimension %i", i);
                Py_XDECREF(py_data);
                Py_XDECREF(broadcastable);
                return NULL;
            }
        }
        Py_XDECREF(broadcastable);
        return py_data;
    }
    else
    {
        data = (PyArrayObject*)PyArray_FromObject(py_data, REAL_TYPENUM, PyTuple_Size(broadcastable), PyTuple_Size(broadcastable));
        if (!data)
        {
            //err message already defined
            Py_XDECREF(py_data);
            Py_XDECREF(broadcastable);
            return NULL;
        }
        for (int i = 0; i < data->nd; ++i)
        {
            if ((data->dimensions[i] > 1) and PyInt_AsLong(PyTuple_GetItem(broadcastable, Py_ssize_t(i))))
            {
                PyErr_Format(PyExc_TypeError, "Non-unit size in broadcastable dimension %i", i);
                Py_XDECREF(data);
                Py_XDECREF(py_data);
                Py_XDECREF(broadcastable);
                return NULL;
            }
        }

        CudaNdarray * rval = (CudaNdarray*) CudaNdarray_new_null();
        if (CudaNdarray_CopyFromArray(rval, data))
        {
            Py_XDECREF(rval);
            rval = NULL;
        }
        Py_XDECREF(data);
        Py_XDECREF(py_data);
        Py_XDECREF(broadcastable);
        return (PyObject*)rval;
    }
}

#define MDECL(s) {""#s, s, METH_VARARGS, "documentation of "#s"... nothing!"}
static PyMethodDef MyMethods[] = {
    MDECL(filter),
    {NULL, NULL, 0, NULL} /*end of list */
};


PyMODINIT_FUNC
init_theano_cuda_ndarray(void)
{
        (void) Py_InitModule("_theano_cuda_ndarray", MyMethods);
        import_array();
}

