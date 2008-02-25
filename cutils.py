
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
        void * ptr_addr = PyCObject_AsVoidPtr(py_cthunk);
        int (*fn)(void*) = reinterpret_cast<int (*)(void*)>(ptr_addr);
        //int (*fn)(void*) = static_cast<int (*)(void*)>(PyCObject_AsVoidPtr(py_cthunk));
        //int (*fn)(void*) = NULL;
        //fn += PyCObject_AsVoidPtr(py_cthunk);
        //int (*fn)(void*) = 

        void* it = PyCObject_GetDesc(py_cthunk);
        int failure = fn(it);
        if (failure) {
            return NULL;
        }
        """

    
    
    cthunk = object()
    mod = weave.ext_tools.ext_module('cutils_ext')
    fun =weave.ext_tools.ext_function('run_cthunk', single_runner, ['cthunk'])
    fun.customize.add_extra_compile_arg('--permissive')
    mod.add_function(fun)
    mod.compile()

    from cutils_ext import *
