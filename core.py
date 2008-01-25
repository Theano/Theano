
import gof
from gof import current_mode, set_mode, build_mode, eval_mode, build_eval_mode, pop_mode, UNCOMPUTED, UNDEFINED, PythonR

import type_spec
import cutils

import numpy
import weakref
import inspect
import md5
from scipy import weave

from copy import copy as pycopy

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


literals_db = {}
literals_id_db = weakref.WeakValueDictionary()

def input(x):
    if isinstance(x, numpy.ndarray):
        return NumpyR(x)
    elif isinstance(x, (int, float)):
        return NumpyR(numpy.array(x))
    elif isinstance(x, gof.Result):
        raise TypeError("%s is already a result." % x)
    else:
        return PythonR(x)

def wrap(x):
    if isinstance(x, NumpyR):
        return x
    elif isinstance(x, PythonR):
        return x
    elif isinstance(x, omega_op):
        return x.out
    else:
        return literal(x)

def _hashable(x):
    try:
        x in {}
        return True
    except TypeError: # x is unhashable
        return False

def _literal_hashable(x):
    if x in literals_db:
        return literals_db[x]
    else:
        r = input(x)
        r.constant = True
        literals_db[x] = r
        return r

def _literal_unhashable(x):
    idx = id(x)
    if idx in literals_id_db:
        return literals_id_db[idx]
    else:
        r = input(x)
        r.constant = True
        literals_id_db[idx] = r
        return r

def literal(x):
    if _hashable(x):
        return _literal_hashable(x)
    else:
        return _literal_unhashable(x)


inplace = gof.Destroyer
view = gof.Viewer


def cgetspecs(names, vals, converters):
    d = {}
    for name, value in zip(names, vals):
        d[name] = value.data
    specs = weave.ext_tools.assign_variable_types(names, d, type_converters = converters) #, auto_downcast = 0)
    return d, specs

def cgen(name, behavior, names, vals, converters = None):
    
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


class omega_op(gof.PythonOp):

    forbid_broadcast = False

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        for fname in ['grad', 'c_impl']:
            gof.make_static(cls, fname)

        # make impl a static method
        gof.PythonOp.__clsinit__(cls, name, bases, dct)
    
    def __new__(cls, *inputs):
        inputs = [wrap(input) for input in inputs]
        return gof.PythonOp.__new__(cls, *inputs)

    def gen_outputs(self):
        return [NumpyR() for i in xrange(self.nout)]
    
    def update_gradient(self, grad_d):
        inputgs = self.grad(*(self.inputs + [grad_d[output] for output in self.outputs]))
        if not isinstance(inputgs, (list, tuple)):
            inputgs = [inputgs] * len(self.inputs)
        for input, inputg in zip(self.inputs, inputgs):
            grad_d.add(input, inputg)

    def grad(*args):
        return UNDEFINED

    def c_code(self, converters = None):
        (inames, onames), behavior = self._c_impl()
        return cgen(self.__class__.__name__, behavior, inames + onames, self.inputs + self.outputs, converters)

#     def _alloc(self):
#         self.alloc(self.inputs, self.outputs)
    
#     def alloc(inputs, outputs):
#         raise NotImplementedError()

    def _c_impl(self):
        (inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
        return (inames, onames), self.c_impl(self.inputs, self.outputs)
    
    def c_impl(inputs, outputs):
        raise NotImplementedError()

    def c_thunk_creator(self):
        self.refresh()
        d, names, code, struct, converters = self.c_code()
        
        cthunk = object()
        module_name = md5.md5(code).hexdigest()
        mod = weave.ext_tools.ext_module(module_name)
        instantiate = weave.ext_tools.ext_function('instantiate',
                                                   code,
                                                   names,
                                                   local_dict = d, global_dict = {}, type_converters = converters)
        instantiate.customize.add_support_code(struct)
        mod.add_function(instantiate)
        mod.compile(location = 'compiled')

        module = __import__("compiled.%s" % module_name, fromlist = [module_name])
        
        def creator():
            return module.instantiate(*[x.data for x in self.inputs + self.outputs])
#         def creator():
#             return weave.inline(code, d.keys(), local_dict = d, global_dict = {}, support_code = struct, type_converters = converters)
        return creator
    
    def c_thunk(self):
        return self.c_thunk_creator()
    
    def c_perform(self):
        thunk = self.c_thunk()
        cutils.run_cthunk(thunk)
        

def elemwise_wrap_old(beforeloop, inloop, afterloop, loop_vars, writable_loop_vars):
    return """
    %(beforeloop)s
    for (int i = 0; i < N_%(v1)s[0]; i++) {
        for (int j = 0; j < N_%(v1)s[1]; j++) {
            %(idefs)s
            %(odefs)s
            %(inloop)s
        }
    }
    %(afterloop)s
    """ % dict(v1 = (loop_vars + writable_loop_vars)[0],
               idefs = "\n".join(["_%s_dtype %s = _%s2(i, j);" % (loop_var, loop_var, loop_var.upper())
                                  for loop_var in loop_vars]),
               odefs = "\n".join(["_%s_dtype& %s = _%s2(i, j);" % (writable_loop_var, writable_loop_var, writable_loop_var.upper())
                                  for writable_loop_var in writable_loop_vars]),
               beforeloop = beforeloop,
               inloop = inloop,
               afterloop = afterloop)

def elemwise_loopcode(loopcode, init_template, next_template, acquire_template, cleanup_template, loop_vars, writable_loop_vars, aliases):
    all_loop_vars = loop_vars + writable_loop_vars

    template = dict(
        init = "".join([init_template % dict(loop_var = loop_var) for loop_var in all_loop_vars]),
        next = "".join([next_template % dict(loop_var = loop_var) for loop_var in all_loop_vars]),
        cleanup = "".join([cleanup_template % dict(loop_var = loop_var) for loop_var in all_loop_vars]),
        idefs = "".join([("_%(loop_var)s_dtype %(loop_var)s = " + acquire_template + ";\n")
                         % dict(loop_var = loop_var) for loop_var in loop_vars]),
        odefs = "".join([("_%(loop_var)s_dtype& %(loop_var)s = " + acquire_template + ";\n")
                         % dict(loop_var = loop_var) for loop_var in writable_loop_vars]),
        aliasdefs = "".join(["_%(v1)s_dtype %(v1)s = %(v2)s;\n" % dict(v1=v1, v2=v2)
                             for v1, v2 in aliases.items()]),
        loopcode = loopcode
        )
    
    code = """
    %(init)s
    while (__elemwise_size--) {
        %(idefs)s
        %(odefs)s
        %(aliasdefs)s
        %(loopcode)s
        %(next)s
    }
    %(cleanup)s
    """ % template

    return code


def elemwise_wrap(beforeloop, inloop, afterloop, loop_vars, writable_loop_vars, aliases):
    general_init = "PyArrayIterObject* _%(loop_var)s_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)_%(loop_var)s_array);\n"
#         "if (_%(loop_var)s_iter == NULL) {\n" \
#         "    PyErr_SetString(PyExc_ValueError, \"Could not make an iterator over variable %(loop_var)s.\");\n" \
#         "    return 1;\n" \
#         "}\n"
    general_next = "PyArray_ITER_NEXT(_%(loop_var)s_iter);\n"
    general_acquire = "*((_%(loop_var)s_dtype*)_%(loop_var)s_iter->dataptr)";
    general_cleanup = "if (_%(loop_var)s_iter) Py_DECREF(_%(loop_var)s_iter);\n";

    contiguous_init = "_%(loop_var)s_dtype* _%(loop_var)s_iter = (_%(loop_var)s_dtype*)PyArray_DATA(_%(loop_var)s_array);\n"
    contiguous_next = "_%(loop_var)s_iter++;\n"
    contiguous_acquire = "*_%(loop_var)s_iter"
    contiguous_cleanup = ""
    
    all_loop_vars = loop_vars + writable_loop_vars
    template = dict(
        v1 = (loop_vars + writable_loop_vars)[0],
        beforeloop = beforeloop,
        general_loop = elemwise_loopcode(
            inloop,
            general_init, general_next, general_acquire, general_cleanup,
            loop_vars, writable_loop_vars, aliases),
        contiguous_loop = elemwise_loopcode(
            inloop,
            contiguous_init, contiguous_next, contiguous_acquire, contiguous_cleanup,
            loop_vars, writable_loop_vars, aliases),
        contiguity_check = "".join(["all_c_contiguous &= PyArray_ISCARRAY(_%(loop_var)s_array);\n" \
                                    "all_f_contiguous &= PyArray_ISFARRAY(_%(loop_var)s_array);\n" \
                                        % dict(loop_var = loop_var)
                                    for loop_var in all_loop_vars]),
        afterloop = afterloop)
    
    code = """
    npy_intp __elemwise_size = PyArray_SIZE(_%(v1)s_array);
    %(beforeloop)s
    bool all_c_contiguous = 1;
    bool all_f_contiguous = 1;
    %(contiguity_check)s
    if (all_c_contiguous || all_f_contiguous) {
        %(contiguous_loop)s
    }
    else {
        %(general_loop)s
    }
    %(afterloop)s
    """ % template

    return code


def upcast(dtype, *dtypes):
    z = numpy.zeros((), dtype = dtype)
    for dtype in dtypes:
        z = z + numpy.zeros((), dtype = dtype)
    return z.dtype



class elemwise(omega_op):

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        for fname in ['c_init', 'c_foreach', 'c_finalize']:
            gof.make_static(cls, fname)

        # make impl, grad, etc. static methods
        omega_op.__clsinit__(cls, name, bases, dct)

    def _specs(self):
        try:
            return self.specs(*[input.spec for input in self.inputs])
        except NotImplementedError:
            (inames, onames), code = self._c_foreach()
            for oname in onames:
                if oname.startswith("_"):
                    raise Exception("cannot infer a specification automatically for variable " \
                                    "%s because it is not part of the elementwise loop - "\
                                    "please override the specs method" % oname[1:])
            shape, dtype = None, None
            for iname, input in zip(inames, self.inputs):
                if not iname.startswith("_"):
                    if input.spec:
                        shape = input.spec[2]
            if shape is None:
                raise Exception("cannot infer a specification automatically for output variables " \
                                "because there is no input variable in the loop from which to get the shape, "\
                                "or their shape is unknown")

            try:
                dtype = upcast(*[input.spec[1]
                                 for iname, input in zip(inames, self.inputs)
                                 if isinstance(input, NumpyR)])
            except IndexError:
                raise Exception("not all numpy inputs are specified")

            if isinstance(self, inplace):
                dmap = self.destroy_map()
            else:
                dmap = {}

            res = []
            for output in self.outputs:
                inplace_inputs = dmap.get(output, [])
                if inplace_inputs:
                    assert len(inplace_inputs) == 1
                    res.append(inplace_inputs[0].spec)
                else:
                    res.append((numpy.ndarray, dtype, shape))
                    
            if self.nout == 1:
                return res[0]
            else:
                return res
        
    def alloc(self, except_list = []):
        if isinstance(self, inplace):
            dmap = self.destroy_map()
        else:
            dmap = {}

        gof.PythonOp.alloc(self, except_list = except_list + dmap.keys())
        for output, (input, ) in dmap.items():
            if output not in except_list:
                output.set_value(input.data)


#     def _alloc(self):
#         if isinstance(self, inplace):
#             dmap = self.destroy_map()
#         else:
#             dmap = {}
#         try:
#             return self.alloc(self.inputs, self.outputs)
#         except NotImplementedError:
#             (inames, onames), _1, _2, _3 = inspect.getargspec(self.c_foreach)
#             for oname in onames:
#                 if oname.startswith("_"):
#                     raise Exception("cannot infer an allocation policy automatically for variable " \
#                                     "%s because it is not part of the elementwise loop - "\
#                                     "please override the alloc method" % oname[1:])
#             shape, dtype = None, None
#             for iname, input in zip(inames, self.inputs):
#                 if not iname.startswith("_"):
#                     shape = input.data
#             if shape is None:
#                 raise Exception("cannot infer an allocation policy automatically for output variables " \
#                                 "because there is no input variable in the loop from which to get the shape")

#             dtype = upcast(*[input.data.dtype
#                              for iname, input in zip(inames, self.inputs)
#                              if isinstance(input.data, numpy.ndarray)])

#             for output in self.outputs:
#                 inplace_inputs = dmap.get(output, [])
#                 if inplace_inputs:
#                     assert len(inplace_inputs) == 1
#                     output.data = inplace_inputs[0].data
#                 else:
#                     output.data = numpy.ndarray(shape, dtype)

    def _c_init(self):
        (inames, onames), _1, _2, _3 = inspect.getargspec(self.c_init)
        return [list(inames), list(onames)], self.c_init(self.inputs, self.outputs)
        
    def c_init(inputs, outputs):
        return ""

    def _c_foreach(self):
        (inames, onames), _1, _2, _3 = inspect.getargspec(self.c_foreach)
        return [list(inames), list(onames)], self.c_foreach(self.inputs, self.outputs)
        
    def c_foreach(inputs, outputs):
        return ""

    def _c_finalize(self):
        (inames, onames), _1, _2, _3 = inspect.getargspec(self.c_finalize)
        return [list(inames), list(onames)], self.c_finalize(self.inputs, self.outputs)

    def c_finalize(inputs, outputs):
        return ""

    def c_code(self, converters = None, elemwise_wrap = elemwise_wrap):
        def mangle(name):
            if name.startswith("_"):
                return name
            else:
                return "_" + name

        try:
            self._c_impl()
            raise Exception("c_impl is not used by elemwise ops - define behavior in c_foreach instead")
        except NotImplementedError:
            pass

        spec_b, before = self._c_init()
        spec_d, during = self._c_foreach()
        spec_a, after  = self._c_finalize()
        
        # Sanity check - apart from loop vars, variables are shared in the before/during/after parts
        if before and spec_b != spec_d:
            raise Exception("The input signature of c_init differs from the input signature of c_foreach.")
        if after and spec_a != spec_d:
            raise Exception("The input signature of c_finalize differs from the input signature of c_foreach.")
        
        (inames, onames) = spec_d

        aliases = {}
        if isinstance(self, inplace):
            dmap = self.destroy_map()
            for oname, output in zip(onames, self.outputs):
                if not oname.startswith("_"):
                    for input in dmap.get(output, []):
                        aliases[inames[self.inputs.index(input)]] = oname
                        
        behavior = elemwise_wrap(before, during, after,
                                 [iname for iname in inames if not iname.startswith("_") and not iname in aliases],
                                 [oname for oname in onames if not oname.startswith("_")],
                                 aliases)

        inames = [mangle(name) for name in inames]
        onames = [mangle(name) for name in onames]
        
        return cgen(self.__class__.__name__, behavior, inames + onames, self.inputs + self.outputs, converters)

    @classmethod
    def inplace_version(cls, dmap = {0: 0}):
        (inames, onames), _1, _2, _3 = inspect.getargspec(cls.c_foreach)
        for i, oname in enumerate(onames):
            if i in dmap:
                assert not oname.startswith("_")
        
        class C(cls, inplace):
            def destroy_map(self):
                if issubclass(cls, inplace):
                    ret = cls.destroy_map(self)
                else:
                    ret = {}
                for output, input in dmap.items():
                    ret[self.outputs[output]] = [self.inputs[input]]
                return ret
            def _impl(self):
                if self.impl is not cls.impl:
                    # If the user sets his own inplace operation, we use it
                    return cls._impl(self)
                else:
                    res = cls._impl(self)
                    if isinstance(res, (list, tuple)):
                        res = pycopy(res)
                    else:
                        res = [res]
                    for output, input in dmap.items():
                        # The default implementation returned a copy, so we just
                        # overwrite the original input with the contents of that copy
                        # This is not meant to be efficient, only correct.
                        a = self.inputs[input].data
                        a[:] = res[output]
                        res[output] = a
                    if len(res) == 1:
                        return res[0]
                    else:
                        return res

        if dmap == {0:0}:
            C.__name__ = cls.__name__ + "_inplace" % dmap
        else:
            C.__name__ = cls.__name__ + "_inplace%s" % dmap
        return C


def scalar_switch(normal_f, scalar_f, scalar_f_reverse = None):
    def f(x, y):
        x, y = wrap(x), wrap(y)
        if y.constant and not y.data.shape:
            return scalar_f(x, y)
        if x.constant and not x.data.shape:
            if scalar_f_reverse:
                return scalar_f_reverse(y, x)
            else:
                raise TypeError("You cannot do this operation on a scalar.")
        return normal_f(x, y)
    return f


class NumpyR(gof.PythonR):

    def set_value(self, value):
        if value is None or value is UNCOMPUTED:
            self.data = UNCOMPUTED
        elif isinstance(value, numpy.ndarray):
            self.data = value
        elif isinstance(value, PythonR):
            self.set_value(value.data)
        else:
            self.data = numpy.array(value)
        self.refresh()
        self.up_to_date = True

    def refresh(self):
        if self.data is not UNCOMPUTED:
            self.spec = (numpy.ndarray, self.data.dtype, self.data.shape)
        
    def alloc(self):
        self.data = numpy.ndarray(self.spec[2], self.spec[1])

    def  __add__(self, y): return add(self, y)
    def __radd__(self, x): return add(x, self)
    def __iadd__(self, y): return iadd(self, y)
    
    def  __sub__(self, y): return sub(self, y)
    def __rsub__(self, x): return sub(x, self)
    def __isub__(self, y): return isub(self, y)
    
    def  __mul__(self, y): return mul(self, y)
    def __rmul__(self, x): return mul(x, self)
    def __imul__(self, y): return imul(self, y)
 
    def  __div__(self, y): return div(self, y)
    def __rdiv__(self, x): return div(x, self)
    def __idiv__(self, y): return idiv(self, y)
        
    def  __pow__(self, y): return pow(self, y)
    def __rpow__(self, x): return pow(x, self)
    def __ipow__(self, y): return ipow(self, y)

    def __neg__(self):     return neg(self)

    T  = property(lambda self: transpose(self))
    Tc = property(lambda self: transpose_copy(self))

    def __copy__(self):    return array_copy(self)

    
def wrap_producer(f):
    class producer(omega_op):
        impl = f
    producer.__name__ = f.__name__
    def ret(dim, dtype = 'float', order = 'C'):
        return producer(dim, dtype, order)
    return ret

ndarray = wrap_producer(numpy.ndarray)
array = wrap_producer(numpy.array)
zeros = wrap_producer(numpy.zeros)
ones = wrap_producer(numpy.ones)


# Wrapper to ensure that all inputs to the function impl have the same size (foils numpy's broadcasting)
def assert_same_shapes(impl):
    def ret(x, *rest):
        shape = x.shape
        for other in rest:
            if other.shape != shape:
                raise ValueError("The dimensions of the inputs do not match.")
        return impl(x, *rest)
    return ret

# Wrapper to ensure that the last input to impl is a scalar
def tensor_scalar_impl(impl):
    def ret(x, a):
        if a.shape:
            raise ValueError("The second argument to %s must be a scalar." % impl)
        return impl(x, a)
    return ret

class tensor_scalar_op(elemwise):
    def c_init((x, _a), (z, )):
        return "_a_dtype a = _a[0];"
    def _c_foreach(self):
        return [['x', '_a'], ['z', ]], "z = %s;" % self.c_expr



## Addition ##

class add_elemwise(elemwise):
    impl = assert_same_shapes(numpy.ndarray.__add__)
    def grad(x, y, gz):
        return gz
    def c_foreach((x, y), (z, )):
        return "z = x + y;"

iadd_elemwise = add_elemwise.inplace_version()
iadd_elemwise.set_impl(assert_same_shapes(numpy.ndarray.__iadd__))


class add_scalar(tensor_scalar_op):
    impl = tensor_scalar_impl(numpy.ndarray.__add__)
    def grad(x, a, gz):
        return gz, sum(gz)
    c_expr = "x + a"

iadd_scalar = add_scalar.inplace_version()
iadd_scalar.set_impl(tensor_scalar_impl(numpy.ndarray.__iadd__))

class twice(elemwise):
    def grad(x, gz):
        return scale(gz, 2.0)
    def impl(x):
        return x + x
    def c_foreach((x, ), (z, )):
        "z = x + x;"

itwice = twice.inplace_version()


## Subtraction ##

class sub_elemwise(elemwise):
    impl = assert_same_shapes(numpy.ndarray.__sub__)
    def grad(x, y, gz):
        return gz, -gz
    def c_foreach((x, y), (z, )):
        return "z = x - y;"

isub_elemwise = sub_elemwise.inplace_version()
isub_elemwise.set_impl(assert_same_shapes(numpy.ndarray.__isub__))

def sub_scalar_r(x, a):
    return add_scalar(x, -a)

def sub_scalar_l(x, a):
    return add_scalar(-x, a)

def isub_scalar_r(x, a):
    return iadd_scalar(x, -a)


## Element-wise multiplication ##

class mul_elemwise(elemwise):
    impl = assert_same_shapes(numpy.ndarray.__mul__)
    def grad(x, y, gz):
        return mul(y, gz), mul(x, gz)
    def c_foreach((x, y), (z, )):
        return "z = x * y;"

imul_elemwise = mul_elemwise.inplace_version()
imul_elemwise.set_impl(assert_same_shapes(numpy.ndarray.__imul__))


class scale(tensor_scalar_op):
    impl = tensor_scalar_impl(numpy.ndarray.__mul__)
    def grad(x, a, gz):
        return scale(a, gz), sum(mul_elemwise(x, gz))
    c_expr = "x * a"

iscale = scale.inplace_version()
iscale.set_impl(tensor_scalar_impl(numpy.ndarray.__imul__))


class sqr(elemwise):
    def impl(x):
        return x * x
    def grad(x, gz):
        return scale(mul_elemwise(x, gz), 2.0)
    def c_foreach((x, ), (z, )):
        return "z = x * x;"

isqr = sqr.inplace_version()
isqr.set_impl(lambda x: x.__imul__(x))



class sqrt(elemwise):
    impl = numpy.sqrt
    def grad(x, gz):
        return scale(div(gz, sqrt(x)), 0.5)
    def c_foreach((x, ), (z, )):
        "z = pow(x, 0.5);"

isqrt = sqrt.inplace_version()
isqrt.set_impl(lambda x: x.__ipow__(0.5))



## Exponentiation ##

class exp(elemwise):
    impl = numpy.exp
    def c_foreach((x, ), (z, )):
        return "z = exp(x);"
    

## Element-wise division ##

class div_elemwise(elemwise):
    impl = assert_same_shapes(numpy.ndarray.__div__)
    def grad(x, y, gz):
        return div(gz, y), -div(mul(x, gz), sqr(y))
    def c_foreach((x, y), (z, )):
        return "z = x / y;"

idiv_elemwise = div_elemwise.inplace_version()
idiv_elemwise.set_impl(assert_same_shapes(numpy.ndarray.__idiv__))

def div_scalar_r(x, a):
    return scale(x, inv_elemwise(a))

def div_scalar_l(x, a):
    return scale(inv_elemwise(x), a)

def idiv_scalar_r(x, a):
    return iscale(x, inv_elemwise(a))



## Scaling ##

class neg(elemwise):
    impl = numpy.ndarray.__neg__
    def grad(x, gz):
        return -gz
    def c_foreach((x, ), (z, )):
        return "z = -x;"

ineg = neg.inplace_version()
ineg.set_impl(lambda x: x.__imul__(-1))


class inv_elemwise(elemwise):
    impl = lambda x: 1 / x
    def grad(x, gz):
        return -gz
    def c_foreach((x, ), (z, )):
        return "z = 1 / x;"

iinv_elemwise = inv_elemwise.inplace_version()


## Dot product ##

class dot(omega_op):
    impl = numpy.dot
    def grad(x, y, gz):
        return dot(gz, transpose(y)), dot(transpose(x), gz)
    def specs(x, y):
        # todo: handle non-matrices!
        if len(x[2]) == 0:
            shape = y[2]
        elif len(y[2]) == 0:
            shape = x[2]
        elif len(x[2]) == 1:
            shape = (y[2][1], )
        elif len(y[2]) == 1:
            shape = (x[2][0], )
        else:
            shape = (x[2][0], y[2][1])
        return (numpy.ndarray, upcast(x[1], y[1]), shape)

## Transposition ##

class transpose(omega_op, view):
    impl = numpy.transpose
    def grad(x, gz):
        return transpose_copy(gz)
    def specs(x):
        # todo: handle non-matrices!
        return (numpy.ndarray, x[1], (x[2][1], x[2][0]))

def transpose_copy(x):
    return array_copy(transpose(x))


## Copy ##

class array_copy(elemwise):
    impl = numpy.array
    grad = lambda x, gz: gz
    def c_foreach((x, ), (z, )):
        return "z = x;"


## Power ##

class pow_elemwise(elemwise):
    impl = assert_same_shapes(numpy.ndarray.__pow__)
    def grad(x, s, gz):
        return gz * s * (pow_elemwise(x, s-1.0))
    def c_foreach((x, s), (z, )):
        return "z = pow(x, s)"

ipow_elemwise = pow_elemwise.inplace_version()
ipow_elemwise.set_impl(assert_same_shapes(numpy.ndarray.__ipow__))


class pow_scalar_l(tensor_scalar_op):
    impl = tensor_scalar_impl(lambda x, y: numpy.ndarray.__pow__(y, x))
    def grad(x, s, gz):
        return gz * x * (pow_scalar_l(s,x-1.0))
    c_expr = "pow(a, x)"

class pow_scalar_r(tensor_scalar_op):
    impl = tensor_scalar_impl(numpy.ndarray.__pow__)
    def grad(x, s, gz):
        return gz * s * (pow_scalar_r(x,s-1.0))
    c_expr = "pow(x, a)"

ipow_scalar_r = pow_scalar_r.inplace_version()
ipow_scalar_r.set_impl(tensor_scalar_impl(numpy.ndarray.__ipow__))



## Others ##

class minmax(elemwise):
    nout = 2
    def impl(x):
        return x.min, x.max
    def specs(x):
        return [(numpy.ndarray, x[1], ())] * 2
#     def alloc((x, ), (_min, _max)):
#         _min.data = numpy.ndarray((), x.dtype)
#         _max.data = numpy.ndarray((), x.dtype)
    def c_init((x, ), (_min, _max)):
        return """
        _x_dtype min = _x[0];
        _x_dtype max = _x[0];
        """
    def c_foreach((x, ), (_min, _max)):
        return """
        if (x < min) min = x;
        if (x > max) max = x;
        """
    def c_finalize((x, ), (_min, _max)):
        return """
        _min[0] = min;
        _max[0] = max;
        """


class fill(elemwise):
    impl = lambda model, value: (model * 0) + value
    def c_init((model, _value), (z, )):
        return "_z_dtype value = _value[0];"
    def c_foreach((model, _value), (z, )):
        return "z = value;"

ifill = fill.inplace_version()


class sum(elemwise):
    impl = numpy.sum
    def grad(x, gz):
        return fill(x, gz)
    def specs(x):
        return (numpy.ndarray, x[1], ())
#     def alloc((x, ), (_sum, )):
#         _sum.data = numpy.ndarray((), dtype = x.data.dtype)
    def c_init((x, ), (_sum, )):
        return "_sum[0] = 0;"
    def c_foreach((x, ), (_sum, )):
        return "_sum[0] += x;"

    
## Array slicing ##

class get_slice(omega_op, view):
    def grad(x, gz):
        raise NotImplementedError()
    def impl(x, dims):
        return x.__getitem__(*dims)

# class set_slice(omega_op, inplace):
#     def impl(x, dims):
#         x.__setitem__(*dims)
#         return x


add = scalar_switch(add_elemwise, add_scalar, add_scalar)
iadd = scalar_switch(iadd_elemwise, iadd_scalar)

sub = scalar_switch(sub_elemwise, sub_scalar_r, sub_scalar_l)
isub = scalar_switch(isub_elemwise, isub_scalar_r)

mul = scalar_switch(mul_elemwise, scale, scale)
imul = scalar_switch(imul_elemwise, iscale)

div = scalar_switch(div_elemwise, div_scalar_r, div_scalar_l)
idiv = scalar_switch(idiv_elemwise, idiv_scalar_r)

pow = scalar_switch(pow_elemwise, pow_scalar_r, pow_scalar_l)
ipow = scalar_switch(ipow_elemwise, ipow_scalar_r)



