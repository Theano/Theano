import os # for building the location of the .omega/omega_compiled cache directory
import sys # for adding the inline code cache to the include path
import platform #
import unittest
import weakref
import inspect
import md5
import copy

import numpy
from scipy import weave

import gof
from gof import current_mode, set_mode, build_mode, eval_mode, build_eval_mode
from gof import pop_mode, is_result, ResultBase

import type_spec
import cutils
import blas
import compile


# __all__ = ['set_mode', 'get_mode', 'NumpyR', 'NumpyOp']


def build(f, *args, **kwargs):
    build_mode()
    r = f(*args, **kwargs)
    pop_mode()
    return r

def as_string(*rs):
    s = gof.graph.as_string(gof.graph.inputs(rs), rs)
    if len(rs) == 1:
        return s[1:-1]
    else:
        return s

def print_graph(*rs):
    print as_string(*rs)

#useful mostly for unit tests
def _approx_eq(a,b,eps=1.0e-9):
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    if a.shape != b.shape:
        return False
    return numpy.max(numpy.abs(a-b)) < eps


# This function is only executed the first time it is called, subsequent calls 
# return immediately from a cache of the first return value
@blas._constant # TODO: move this decorator to a utility file
def _compile_dir():
    """Return the directory in which scipy.weave should store code objects.

    If the environment variable OMEGA_COMPILEDIR is set, its value is returned.
    If not, a directory of the form $HOME/.omega/compiledir_<platform Id>.

    As a test, this function touches the file __init__.py in the returned
    directory, and raises OSError if there's a problem.

    A directory coming from OMEGA_COMPILEDIR is not created automatically, but
    a directory in $HOME/.omega is created automatically.

    This directory is appended to the sys.path search path before being
    returned, if the touch was successful.
    """
    if os.getenv('OMEGA_COMPILEDIR'):
        cachedir = os.getenv('OMEGA_COMPILEDIR')
    else:
        # use (and possibly create) a default code cache location
        platform_id = platform.platform() + '-' + platform.processor()
        import re
        platform_id = re.sub("[\(\)\s]+", "_", platform_id)
        cachedir = os.path.join(os.getenv('HOME'), '.omega', 'compiledir_'+platform_id)
        if not os.access(cachedir, os.R_OK | os.W_OK):
            #this may raise a number of problems, I think all of which are serious.
            os.makedirs(cachedir, 7<<6)
    cachedir_init = cachedir+'/__init__.py'
    touch = os.system('touch '+cachedir_init)
    if touch:
        raise OSError('touch %s returned %i' % (cachedir_init, touch))

    if cachedir not in sys.path:
        sys.path.append(cachedir)
    return cachedir

class Numpy2(ResultBase):
    """Result storing a numpy ndarray"""
    __slots__ = ['_dtype', '_shape', '_order']

    class ShapeUnknown: pass # TODO: use this as the shape of uncomputed ndarrays of unknown shape
    class StateError(Exception): pass

    def __init__(self, role=None, data=None, constant=False):
        self._order = 'C'
        if isinstance(data, (tuple, list)): # unallocated setup
            shape, dtype = data
            ResultBase.__init__(self, role, data=None, constant=constant)
            self._shape = shape
            self._dtype = dtype
        else:                               # allocated setup
            ResultBase.__init__(self, role, data, constant)

    ################################
    # ResultBase
    # 
    def data_filter(self, data):
        return numpy.asarray(data)
        

    ################################
    # Numpy2 specific functionality
    #
    __array__ = property(lambda self: self.data.__array__)
    __array_struct__ = property(lambda self: self.data.__array_struct__)

    def data_alloc(self):
        return numpy.ndarray(shape=self.shape, dtype=self.dtype, order=self._order)

    # self._dtype is used when self.data hasn't been set yet
    def __dtype_get(self):
        if self.data is not None:
            self._dtype = self.data.dtype
        return self._dtype
    def __dtype_set(self, dtype):
        if self.data is None:
            self._dtype = dtype
        else:
            raise StateError('cannot set dtype after data has been set')
    dtype = property(__dtype_get, __dtype_set)

    # self._shape is used when self.data hasn't been set yet
    def __shape_get(self):
        if self.data is not None:
            self._shape = self.data.shape
        return self._shape
    def __shape_set(self, shape):
        if self.data is None:
            self._shape = shape
        else:
            raise StateError('cannot set shape after data has been set')
    shape = property(__shape_get, __shape_set)

    def  __add__(self, y): return add(self, y)
    def __radd__(self, x): return add(x, self)
    def __iadd__(self, y): return add_inplace(self, y)
    
    def  __sub__(self, y): return sub(self, y)
    def __rsub__(self, x): return sub(x, self)
    def __isub__(self, y): return sub_inplace(self, y)
    
    def  __mul__(self, y): return mul(self, y)
    def __rmul__(self, x): return mul(x, self)
    def __imul__(self, y): return mul_inplace(self, y)
 
    def  __div__(self, y): return div(self, y)
    def __rdiv__(self, x): return div(x, self)
    def __idiv__(self, y): return div_inplace(self, y)
        
    def  __pow__(self, y): return pow(self, y)
    def __rpow__(self, x): return pow(x, self)
    def __ipow__(self, y): return pow_inplace(self, y)

    def __neg__(self):     return neg(self)

    T  = property(lambda self: transpose(self))
    Tc = property(lambda self: transpose_copy(self))

    def __copy__(self):    return array_copy(self)

    def __getitem__(self, item): return get_slice(self, item)
    def __getslice__(self, *args): return get_slice(self, slice(*args))

    #################
    # NumpyR Compatibility
    #
    spec = property(lambda self: (numpy.ndarray, self.dtype, self.shape))
    def set_value_inplace(self, value):
        if 0 == len(self.shape):
            self.data.itemset(value) # for scalars
        else:
            self.data[:] = value     # for matrices
        self.state = gof.result.Computed

class _test_Numpy2(unittest.TestCase):
    def setUp(self):
        build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        pop_mode()
    def test_0(self):
        r = Numpy2()
    def test_1(self):
        o = numpy.ones((3,3))
        r = Numpy2(data=o)
        self.failUnless(r.data is o)
        self.failUnless(r.shape == (3,3))
        self.failUnless(str(r.dtype) == 'float64')

    def test_2(self):
        r = Numpy2(data=[(3,3),'int32'])
        self.failUnless(r.data is None)
        self.failUnless(r.shape == (3,3))
        self.failUnless(str(r.dtype) == 'int32')
        r.alloc()
        self.failUnless(isinstance(r.data, numpy.ndarray))
        self.failUnless(r.shape == (3,3))
        self.failUnless(str(r.dtype) == 'int32')

    def test_3(self):
        a = Numpy2(data=numpy.ones((2,2)))
        b = Numpy2(data=numpy.ones((2,2)))
        c = add(a,b)
        self.failUnless(_approx_eq(c, numpy.ones((2,2))*2))

    def test_4(self):
        ones = numpy.ones((2,2))
        a = Numpy2(data=ones)
        o = numpy.asarray(a)
        self.failUnless((ones == o).all())

    def test_5(self):
        ones = numpy.ones((2,2))
        self.failUnless(_approx_eq(Numpy2(data=ones), Numpy2(data=ones)))



def cgen(name, behavior, names, vals, converters = None):
    
    def cgetspecs(names, vals, converters):
        d = {}
        assert len(names) == len(vals)
        for name, value in zip(names, vals):
            d[name] = value.data
        specs = weave.ext_tools.assign_variable_types(names, d, type_converters = converters) #, auto_downcast = 0)
        return d, specs

    if not converters:
        converters = type_spec.default
    for converter in converters:
        assert isinstance(converter, type_spec.omega_type_converter_extension)


    d, specs = cgetspecs(names, vals, converters)
    
    template = {}
    template['name'] = name
    template['code'] = behavior
    template['members'] = "".join([spec.struct_members_code() for spec in specs])
    template['support'] = "".join([spec.struct_support_code() for spec in specs])
    template['typedefs'] = "".join([spec.struct_typedefs() for spec in specs])
    template['incref'] = "".join(["Py_INCREF(py_%s);\n" % spec.name for spec in specs if spec.use_ref_count])
    template['decref'] = "".join(["Py_DECREF(py_%s);\n" % spec.name for spec in specs if spec.use_ref_count])

    template['struct_contents'] = """
      %(typedefs)s

      %(members)s

      %(support)s

      void init(void) {
        %(incref)s
      }

      void cleanup(void) {
        %(decref)s
      }

      int execute(void) {
        %(code)s
        return 0;
      }
    """ % template

    template['md5'] = md5.md5(template['struct_contents']).hexdigest()
    template['struct_name'] = "_omega_%(name)s_%(md5)s" % template
    struct = "struct %(struct_name)s { %(struct_contents)s\n};" % template

    static = """
    int %(struct_name)s_executor(%(struct_name)s* self) {
        return self->execute();
    }

    void %(struct_name)s_destructor(void* executor, void* self) {
        ((%(struct_name)s*)self)->cleanup();
        free(self);
    }
    """ % template
    
    code = "%(struct_name)s* __STRUCT_P = new %(struct_name)s();\n" % template
    code += "".join([spec.struct_import_code() for spec in specs])
    code += "__STRUCT_P->init();\n"
    code += "return_val = PyCObject_FromVoidPtrAndDesc((void*)(&%(struct_name)s_executor), __STRUCT_P, %(struct_name)s_destructor);\n" % template

    return d, names, code, struct + static, converters    


class Numpy2Op(gof.lib.PythonOp):
    """What can we do given we are interacting with Numpy2 inputs and outputs"""
    def refresh(self, alloc = True):
        shape = self.refresh_shape()
        dtype = self.refresh_dtype()
        out = self.out

        if out.data is not None \
                and out.shape == shape \
                and out.dtype == dtype:
                    return

        alloc |= out.data is not None

        if alloc: out.data = None
        out.shape = shape
        out.dtype = dtype
        if alloc: out.alloc()

class omega_op(Numpy2Op):

    forbid_broadcast = False

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        for fname in ['grad', 'c_impl', 'impl']:
            if hasattr(cls, fname):
                gof.make_static(cls, fname)

    def __new__(cls, *inputs):
        inputs = [wrap(input) for input in inputs]
        return Numpy2Op.__new__(cls, *inputs)

    def gen_outputs(self):
        return [Numpy2() for i in xrange(self.nout)]
    
    #TODO: use the version of this code that is in grad.py
    #      requires: eliminating module dependency cycles
    def update_gradient(self, grad_d):
        """Call self.grad() and add the result to grad_d

        This function is called by grad.Grad.bprop() to construct a symbolic gradient graph.

        self.grad is called like this:

            self.grad(*(self.inputs + [grad_d[output] for output in self.outputs]))

        In general, grad() should return a list of ResultValue instances whose
        length matches that of self.inputs, and whose elements are the
        gradients of self.inputs.

        There is a (but often used) special feature in place to automatically
        wrap the return value of grad() in a list if it is a ResultValue instance
        and the op is unary.  This makes many grad implementations a little
        cuter.

        """
        inputgs = self.grad(*(self.inputs + [grad_d[output] for output in self.outputs]))
        if len(self.inputs) == 1 and gof.result.is_result(inputgs):
            inputgs = [inputgs]
        else:
            assert len(inputgs) == len(self.inputs)
        for input, inputg in zip(self.inputs, inputgs):
            grad_d.add(input, inputg)

    def c_code(self, converters = None):
        (inames, onames) = self.variable_names()
        behavior = self._c_impl()
        return cgen(self.__class__.__name__, behavior, inames + onames, self.inputs + self.outputs, converters)

    def c_headers(self):
        return []

    def c_libs(self):
        return []

    def c_support_code(self):
        return ""

    def variable_names(self):
        (inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
        return (inames, onames)

    def _c_impl(self):
        return self.c_impl(self.inputs, self.outputs)

    def c_impl(inputs, outputs):
        raise NotImplementedError()

    def c_compile_args(self):
        # I always used these, but they don't make much improvement
        #'-ffast-math', '-falign-loops=8'
        return ['-O2'] 

    def c_thunk_factory(self):
        self.refresh()
        d, names, code, struct, converters = self.c_code()

        cthunk = object()
        module_name = md5.md5(code).hexdigest()
        mod = weave.ext_tools.ext_module(module_name)
        instantiate = weave.ext_tools.ext_function('instantiate',
                                                   code,
                                                   names,
                                                   local_dict = d,
                                                   global_dict = {},
                                                   type_converters = converters)
        instantiate.customize.add_support_code(self.c_support_code() + struct)
        for arg in self.c_compile_args():
            instantiate.customize.add_extra_compile_arg(arg)
        for header in self.c_headers():
            instantiate.customize.add_header(header)
        for lib in self.c_libs():
            instantiate.customize.add_library(lib)
        #add_library_dir
        
        #print dir(instantiate.customize)
        #print instantiate.customize._library_dirs
        if os.getenv('OMEGA_BLAS_LD_LIBRARY_PATH'):
            instantiate.customize.add_library_dir(os.getenv('OMEGA_BLAS_LD_LIBRARY_PATH'))

        mod.add_function(instantiate)
        mod.compile(location = _compile_dir())
        module = __import__("%s" % (module_name), {}, {}, [module_name])

        def creator():
            return module.instantiate(*[x.data for x in self.inputs + self.outputs])
        return creator

    def c_thunk(self):
        return self.c_thunk_creator()

    def c_perform(self):
        thunk = self.c_thunk()
        cutils.run_cthunk(thunk)


def upcast(dtype, *dtypes):
    z = numpy.zeros((), dtype = dtype)
    for dtype in dtypes:
        z = z + numpy.zeros((), dtype = dtype)
    return z.dtype



from grad import Undefined

def wrap_producer(f):
    class producer(gof.lib.NewPythonOp):
        def __init__(self, shape, dtype, order):
            assert order == 'C' #TODO: let Numpy2 support this
            if current_mode() == 'build_eval':
                gof.lib.NewPythonOp.__init__(self, 
                        [input(shape), input(dtype), input(order)],
                        [Numpy2(data = f(shape, dtype))])
            elif current_mode() == 'build':
                gof.lib.NewPythonOp.__init__(self, 
                        [input(shape), input(dtype), input(order)],
                        [Numpy2(data = (shape, dtype))])
        def gen_outputs(self):
            return [Numpy2() for i in xrange(self.nout)]
        impl = f
        def grad(*args):
            return [Undefined] * (len(args) - 1)
    producer.__name__ = f.__name__
    def ret(shape, dtype = 'float64', order = 'C'):
        return producer(shape, dtype, order).out
    return ret

ndarray = wrap_producer(numpy.ndarray)
array = wrap_producer(numpy.array)
zeros = wrap_producer(numpy.zeros)
ones = wrap_producer(numpy.ones)

class _testCase_producer_build_mode(unittest.TestCase):
    def test_0(self):
        """producer in build mode"""
        build_mode()
        a = ones(4)
        self.failUnless(a.data is None, a.data)
        self.failUnless(a.state is gof.result.Empty, a.state)
        self.failUnless(a.shape == 4, a.shape)
        self.failUnless(str(a.dtype) == 'float64', a.dtype)
        pop_mode()
    def test_1(self):
        """producer in build_eval mode"""
        build_eval_mode()
        a = ones(4)
        self.failUnless((a.data == numpy.ones(4)).all(), a.data)
        self.failUnless(a.state is gof.result.Computed, a.state)
        self.failUnless(a.shape == (4,), a.shape)
        self.failUnless(str(a.dtype) == 'float64', a.dtype)
        pop_mode()





if __name__ == '__main__':
    unittest.main()

