
from compiledir import *
import sys



try:
    sys.path.append(get_compiledir())
    from cutils_ext import *

except ImportError:

    from scipy import weave

    # The following function takes a PyCObject instance that contains
    # a void*->int function in its VoidPtr field. It then calls that
    # function on the object's Desc field and returns the int result.
    single_runner = """
        if (!PyCObject_Check(py_cthunk)) {
            PyErr_SetString(PyExc_ValueError,
                            "Argument to run_cthunk must be a PyCObject.");
            return NULL;
        }
        void * ptr_addr = PyCObject_AsVoidPtr(py_cthunk);
        int (*fn)(void*) = reinterpret_cast<int (*)(void*)>(ptr_addr);
        void* it = PyCObject_GetDesc(py_cthunk);
        int failure = fn(it);
        return_val = failure;
        """
    
    cthunk = object()
    mod = weave.ext_tools.ext_module('cutils_ext')
    fun =weave.ext_tools.ext_function('run_cthunk', single_runner, ['cthunk'])
    fun.customize.add_extra_compile_arg('--permissive')
    mod.add_function(fun)
    mod.compile(location = get_compiledir())
    from cutils_ext import *
