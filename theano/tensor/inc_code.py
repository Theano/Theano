from __future__ import absolute_import, print_function, division


def inc_code():
    types = ['npy_' + t for t in ['int8', 'int16', 'int32', 'int64',
                                  'uint8', 'uint16', 'uint32', 'uint64',
                                  'float16', 'float32', 'float64']]

    complex_types = ['npy_' + t for t in ['complex32', 'complex64',
                                          'complex128']]

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
        typedef void (*inplace_map_binop)(PyArrayMapIterObject *,
                                          PyArrayIterObject *, int inc_or_set);
        """ + fns + fn_array + type_number_array + """
static int
map_increment(PyArrayMapIterObject *mit, PyArrayObject *op,
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
    arr = (PyArrayObject *)PyArray_FromAny((PyObject *)op, descr,
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


static int
inplace_increment(PyArrayObject *a, PyObject *index, PyArrayObject *inc,
                  int inc_or_set)
{
    inplace_map_binop add_inplace = NULL;
    int type_number = -1;
    int i = 0;
    PyArrayMapIterObject * mit;

    if (PyArray_FailUnlessWriteable(a, "input/output array") < 0) {
        return -1;
    }

    if (PyArray_NDIM(a) == 0) {
        PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed.");
        return -1;
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
        return -1;
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
    return 0;

fail:
    Py_XDECREF(mit);

    return -1;
}
""")

    return code
