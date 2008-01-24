
try:
    from cutils_ext import *

except ImportError:

    from scipy import weave

    single_runner = """
        if (!PyCObject_Check(py_cthunk)) {
            PyErr_SetString(PyExc_ValueError,
                            "Argument to run_cthunk must be a PyCObject returned by the c_thunk method of an omega_op.");
            return NULL;
        }
        int (*fn)(void*) = reinterpret_cast<int (*)(void*)>(PyCObject_AsVoidPtr(py_cthunk));
        void* it = PyCObject_GetDesc(py_cthunk);
        int failure = fn(it);
        if (failure) {
            return NULL;
        }
        """

    cthunk = object()
    mod = weave.ext_tools.ext_module('cutils_ext')
    mod.add_function(weave.ext_tools.ext_function('run_cthunk', single_runner, ['cthunk']))
    mod.compile()

    from cutils_ext import *
