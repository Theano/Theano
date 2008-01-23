
import gof
from gof import current_mode, set_mode, build_mode, eval_mode, build_eval_mode, pop_mode, UNCOMPUTED, UNDEFINED, PythonR

import type_spec

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


class Proxy(object):

    __slots__ = ['_obj']
    
    def __init__(self, obj = None):
        self._obj = obj

    def __getattribute__(self, attr):
        if attr in ['__class__', '_obj']:
            return object.__getattribute__(self, attr)
        else:
            return getattr(object.__getattribute__(self, '_obj'), attr)

    def __setattr__(self, attr, value):
        if attr in ['_obj']:
            object.__setattr__(self, attr, value)
        else:
            setattr(self._obj, attr, value)

    def __delattr__(self, attr):
        delattr(self._obj, attr)


def as_string(*rs):
    s = gof.graph.as_string(gof.graph.inputs(rs), rs)
    if len(rs) == 1:
        return s[1:-1]
    else:
        return s
#    return str(gof.Env(gof.graph.inputs([r]), [r]))[1:-1]

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
    elif isinstance(x, Proxy):
        return wrap(x._obj)
    else:
        return literal(x)

def _hashable(x):
    try:
        x in {}
        return True
    except TypeError: # x is unhashable
        return False

def _literal_hashable(x):
#     try:
#         present = x in literals_db
#         hashable = True
#     except TypeError: # x is unhashable
#         present = False
#         hashable = False

    if x in literals_db:
        return literals_db[x]
    else:
        r = input(x)
        r.constant = True
        literals_db[x] = r
        return r
    
#     elif isinstance(x, numpy.ndarray):
#         ret = NumpyR(x, constant = True)
#     elif isinstance(x, (int, float)):
#         ret = NumpyR(numpy.array(x), constant = True)
#     elif isinstance(x, gof.Result):
#         raise TypeError("%s is already a result." % x)
#     else:
#         return PythonR(x, constant = True)

#     if hashable:
#         literals_db[x] = ret

#     return ret

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


def cgen(name, behavior, inames, ivals, onames, ovals, converters = None):
    if not converters:
        converters = type_spec.default
    for converter in converters:
        assert isinstance(converter, type_spec.omega_type_converter_extension)

    d = {}
    for name, value in zip(inames + onames, ivals + ovals):
        d[name] = value.data
        #         print inames + onames
        #         print d
        #         print [x.__class__ for x in converters]
    specs = weave.ext_tools.assign_variable_types(inames + onames, d, type_converters = converters) #, auto_downcast = 0)

    template = {}
    template['name'] = name
    template['code'] = behavior
    template['members'] = "\n".join([spec.struct_members_code() for spec in specs])
    template['decl'] = "\n".join([spec.struct_declaration_code() for spec in specs])
    #        types = [spec.struct_template_types() for spec in specs]
    #        template['types'] = ", ".join([", ".join([c_type for c_type, name, init in spec.provides()]) for spec in specs])
    #        template['typenames'] = ", ".join([spec.struct_template_code() for spec in specs])
    template['support'] = "\n".join([spec.struct_support_code() for spec in specs])
    template['typedefs'] = "\n".join([spec.struct_typedefs() for spec in specs])

    template['struct_contents'] = """
      %(typedefs)s

      %(members)s

      %(support)s

      void execute(void) {
        %(decl)s
        %(code)s
      }
    """ % template

    template['md5'] = md5.md5(template['struct_contents']).hexdigest()
    template['struct_name'] = "_omega_%(name)s_%(md5)s" % template

    struct = "struct %(struct_name)s { %(struct_contents)s\n};" % template

    #        code = "_omega_%(name)s<%(types)s>* __STRUCT_P = new _omega_%(name)s();\n" % template
    #        code = "_omega_%(name)s<%(types)s>* __STRUCT_P = &_omega_%(name)s<%(types)s>();\n" % template
    code = "%(struct_name)s* __STRUCT_P = &%(struct_name)s();\n" % template
    code += "\n".join([spec.struct_import_code() for spec in specs])
    code += "\n__STRUCT_P->execute();\n"
    code += "return_val = 10;"
    code += "\n//%(md5)s" % template

    print struct
    print code

    for spec in specs:
        print spec.declaration_code()

    print d

    res = weave.inline(code, inames+onames, local_dict = d, global_dict = {}, support_code = struct, type_converters = converters)

    return res, None



class omega_op(gof.PythonOp):

    forbid_broadcast = False

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        # make grad a static method
        grad = cls.grad
        if hasattr(grad, 'im_func'):
            grad = grad.im_func
        cls.grad = staticmethod(grad)

        # make c_impl a static method
        c_impl = cls.c_impl
        if hasattr(c_impl, 'im_func'):
            c_impl = c_impl.im_func
        cls.c_impl = staticmethod(c_impl)

        # make c_alloc a static method
        c_alloc = cls.c_alloc
        if hasattr(c_alloc, 'im_func'):
            c_alloc = c_alloc.im_func
        cls.c_alloc = staticmethod(c_alloc)

#         # adjust impl
#         if cls.forbid_broadcast:
#             cls.impl = assert_same_shapes(cls.impl)

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

    def create_c_code(self, converters = None):
        behavior = self.c_impl(self.inputs, self.outputs)
        (inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
        return cgen(self.__class__.__name__, behavior, inames, self.inputs, onames, self.outputs, converters)


##        return code, struct


#         behavior = self.c_impl(self.inputs, self.outputs)
#         (inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
#         d = {}
#         for name, value in zip(inames + onames, self.inputs + self.outputs):
#             d[name] = value.data
#         converters = [omega_array_converter()] + weave.converters.default
#         specs = assign_variable_types(inames + onames, d, type_converters = converters) #, auto_downcast = 0)
        
# #        itypes = [isinstance(input.data, numpy.ndarray) and num_to_c_types[input.data.dtype.char] for input in self.inputs]
# #        otypes = [num_to_c_types[output.data.dtype.char] for output in self.outputs]

#         tvars_list = [spec.template_vars() for spec in specs]

#         template = {}
#         template['name'] = self.__class__.__name__
#         template['code'] = behavior
#         template['members'] = ";\n".join([" %(name)s"])
#         template['decl'] = ""
#         template['typespecs'] = ", ".join(["%(name)s_type" % spec.name for spec in specs] +
#                                           ["%(name)s_num_type" % tvars['name'] for tvars in tvars_list if 'num_type' in tvars])

#         struct = """
#         template<%(typespecs)s>
#         struct _omega_%(name)s {

#           %(members)s
        
#           _omega_%(name)s() {}
          
#           void execute(void) {
#             %(decl)s
#             %(code)s
#           }
#         };
#         """ % template

    def _c_alloc(self):
        self.c_alloc(self.inputs, self.outputs)
    
    def c_alloc(inputs, outputs):
        raise NotImplementedError()
    
    def c_impl(inputs, outputs):
        raise NotImplementedError()

    def c_thunk(self):
        self._c_alloc()
        if self.c_module:
            a
        else:
            aaaaaa
    
    def c_perform(self):
        self.c_thunk()()
        

def elemwise_wrap(beforeloop, inloop, afterloop, inames, onames):
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
    """ % dict(v1 = (inames + onames)[0],
               idefs = "\n".join(["_%s_dtype %s = _%s2(i, j);" % (iname, iname, iname.upper()) for iname in inames if not iname.startswith("_")]),
               odefs = "\n".join(["_%s_dtype& %s = _%s2(i, j);" % (oname, oname, oname.upper()) for oname in onames if not oname.startswith("_")]),
               beforeloop = beforeloop,
               inloop = inloop,
               afterloop = afterloop)


class elemwise(omega_op):

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        # make c_init a static method
        c_init = cls.c_init
        if hasattr(c_init, 'im_func'):
            c_init = c_init.im_func
        cls.c_init = staticmethod(c_init)

        # make c_foreach a static method
        c_foreach = cls.c_foreach
        if hasattr(c_foreach, 'im_func'):
            c_foreach = c_foreach.im_func
        cls.c_foreach = staticmethod(c_foreach)

        # make c_finalize a static method
        c_finalize = cls.c_finalize
        if hasattr(c_finalize, 'im_func'):
            c_finalize = c_finalize.im_func
        cls.c_finalize = staticmethod(c_finalize)

#         # adjust impl
#         if cls.forbid_broadcast:
#             cls.impl = assert_same_shapes(cls.impl)

        # make impl a static method
        omega_op.__clsinit__(cls, name, bases, dct)

    def _c_alloc(self):
        if isinstance(self, inplace):
            dmap = self.destroy_map()
        else:
            dmap = {}
        try:
            return self.c_alloc(self.inputs, self.outputs)
        except NotImplementedError:
            (inames, onames), _1, _2, _3 = inspect.getargspec(self.c_foreach)
            for oname in onames:
                if oname.startswith("_"):
                    raise Exception("cannot infer an allocation policy automatically for variable " \
                                    "%s because it is not part of the elementwise loop - "\
                                    "please override the c_alloc method" % oname[1:])
            model = None
            for iname, input in zip(inames, self.inputs):
                if not iname.startswith("_"):
                    model = input.data
            if model is None:
                raise Exception("cannot infer an allocation policy automatically for output variables " \
                                "because there is no input variable in the loop from which to get the shape")
            for output in self.outputs:
                inplace_inputs = dmap.get(output, [])
                if inplace_inputs:
                    assert len(inplace_inputs) == 1
                    output.data = inplace_inputs[0].data
                else:
                    output.data = numpy.ndarray(model.shape, model.dtype)

    def c_init(inputs, outputs):
        return ""

    def c_foreach(inputs, outputs):
        return ""

    def c_finalize(inputs, outputs):
        return ""

    def create_c_code(self, converters = None):
        def mangle(name):
            if name.startswith("_"):
                return name#[1:]
            else:
                return "_" + name

        try:
            self.c_impl(self.inputs, self.outputs)
            raise Exception("c_impl is not used by elemwise ops - define behavior in c_foreach instead")
        except NotImplementedError:
            pass

        before = self.c_init(self.inputs, self.outputs)
        during = self.c_foreach(self.inputs, self.outputs)
        after = self.c_finalize(self.inputs, self.outputs)

        spec_b = inspect.getargspec(self.c_init)
        spec_d = inspect.getargspec(self.c_foreach)
        spec_a = inspect.getargspec(self.c_finalize)

        if before and spec_b != spec_d:
            raise Exception("The input signature of c_init differs from the input signature of c_foreach.")
        if after and spec_a != spec_d:
            raise Exception("The input signature of c_finalize differs from the input signature of c_foreach.")
        
        (inames, onames), _1, _2, _3 = spec_d

        behavior = elemwise_wrap(before, during, after, inames, onames)

        inames = [mangle(name) for name in inames]
        onames = [mangle(name) for name in onames]
        
        return cgen(self.__class__.__name__, behavior, inames, self.inputs, onames, self.outputs, converters)



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
        self.up_to_date = True

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
def tensor_scalar_op(impl):
    def ret(x, a):
        if a.shape:
            raise ValueError("The second argument to %s must be a scalar." % impl)
        return impl(x, a)
    return ret


# @omega_op
# def add((x, y), (z, )):

#     def grad(gz):
#         return gz

#     def c_alloc():
#         return numpy.ndarray(x.shape, dtype = x.dtype)

#     c_impl = """
#              for (int i = 0; i < z.ncols; i++) {
#                  for (int j = 0; j < z.nrows; j++) {
#                      z(i, j) = x(i, j) + y(i, j);
#                  }
#              }
#              """

    


## Addition ##

class add(omega_op):
    impl = assert_same_shapes(numpy.ndarray.__add__)
    def grad(x, y, gz):
        return gz
    def alloc(x, y):
        return numpy.ndarray(x.shape, dtype = x.dtype)
    def c_impl(x, y, z):
        return """
        for (int i = 0; i < z.ncols; i++) {
            for (int j = 0; j < z.nrows; j++) {
                z(i, j) = x(i, j) + y(i, j);
            }
        }
        """

class proto_add_elemwise(omega_op):
    def grad(x, y, gz):
        return gz

class add_elemwise(proto_add_elemwise):
    impl = assert_same_shapes(numpy.ndarray.__add__)

class iadd_elemwise(proto_add_elemwise, inplace):
    impl = assert_same_shapes(numpy.ndarray.__iadd__)

class proto_add_scalar(omega_op):
    def grad(x, a, gz):
        return gz, sum(gz)

class add_scalar(proto_add_scalar):
    impl = tensor_scalar_op(numpy.ndarray.__add__)
    
#     def c_impl(x, s, z):
#         """
#         if (*__z == NULL) {
#             *__z = new ndarray
#         }
#         ndarray& z = **__z
#         """

#         return """
#         z.resize_like(x);
#         for (int i = 0; i < z.size(); i++) {
#             z[i] = x[i] * s;
#         }
#         return z;
#         """

class iadd_scalar(proto_add_scalar, inplace):
    impl = tensor_scalar_op(numpy.ndarray.__iadd__)


class proto_twice(omega_op):
    def grad(x, gz):
        return scale(gz, 2.0)

class twice(proto_twice):
    def impl(x):
        return x + x

class itwice(proto_twice, inplace):
    def impl(x):
        x += x
        return x


## Subtraction ##

class proto_sub_elemwise(omega_op):
    def grad(x, y, gz):
        return gz, -gz

class sub_elemwise(proto_sub_elemwise):
    impl = assert_same_shapes(numpy.ndarray.__sub__)

class isub_elemwise(proto_sub_elemwise, inplace):
    impl = assert_same_shapes(numpy.ndarray.__isub__)

def sub_scalar_r(x, a):
    return add_scalar(x, -a)

def sub_scalar_l(x, a):
    return add_scalar(-x, a)

def isub_scalar_r(x, a):
    return iadd_scalar(x, -a)


## Element-wise multiplication ##

class proto_mul_elemwise(omega_op):
    def grad(x, y, gz):
        return mul(y, gz), mul(x, gz)

class mul_elemwise(proto_mul_elemwise):
    impl = assert_same_shapes(numpy.ndarray.__mul__)

class imul_elemwise(proto_mul_elemwise, inplace):
    impl = assert_same_shapes(numpy.ndarray.__imul__)


class proto_scale(omega_op):
    def grad(x, a, gz):
        return scale(a, gz), sum(mul_elemwise(x, gz))

class scale(proto_scale):
    impl = tensor_scalar_op(numpy.ndarray.__mul__)

class iscale(proto_scale, inplace):
    impl = tensor_scalar_op(numpy.ndarray.__imul__)


class proto_sqr(omega_op):
    def grad(x, gz):
        return scale(mul_elemwise(x, gz), 2.0)

class sqr(proto_sqr):
    impl = lambda x: numpy.multiply(x, x)

class isqr(proto_sqr, inplace):
    impl = lambda x: x.__imul__(x)


class proto_sqrt(omega_op):
    def grad(x, gz):
        return scale(div(gz, sqrt(x)), 0.5)

class sqrt(proto_sqrt):
    impl = numpy.sqrt

class isqrt(proto_sqrt, inplace):
    impl = lambda x: x.__ipow__(0.5)


## Exponentiation ##

class exp(omega_op):
    impl = numpy.exp
    

## Element-wise division ##

class proto_div_elemwise(omega_op):
    def grad(x, y, gz):
        return div(gz, y), -div(mul(x, gz), sqr(y))

class div_elemwise(proto_div_elemwise):
    impl = assert_same_shapes(numpy.ndarray.__div__)

class idiv_elemwise(proto_div_elemwise, inplace):
    impl = assert_same_shapes(numpy.ndarray.__idiv__)

def div_scalar_r(x, a):
    return scale(x, inv_elemwise(a))

def div_scalar_l(x, a):
    return scale(inv_elemwise(x), a)

def idiv_scalar_r(x, a):
    return iscale(x, inv_elemwise(a))



## Scaling ##

    
class proto_neg(omega_op):
    def grad(x, gz):
        return -gz

class neg(proto_neg):
    impl = numpy.ndarray.__neg__

class ineg(proto_neg, inplace):
    impl = lambda x: x.__imul__(-1)


class proto_inv_elemwise(omega_op):
    def grad(x, gz):
        raise NotImplemented

class inv_elemwise(omega_op):
    impl = lambda x: 1 / x

class iinv_elemwise(omega_op, inplace):
    def impl(x):
        x[:] = 1 / x


## Dot product ##

class dot(omega_op):
    impl = numpy.dot
    def grad(x, y, gz):
        return dot(gz, transpose(y)), dot(transpose(x), gz)


## Transposition ##

class transpose(omega_op, view):
    impl = numpy.transpose
    def grad(x, gz):
        return transpose_copy(gz)

def transpose_copy(x):
    return array_copy(transpose(x))


## Copy ##

class array_copy(omega_op):
    impl = numpy.array
    grad = lambda x, gz: gz


## Power ##

class proto_pow(omega_op):
    def grad(x, s, gz):
        return gz * s * (pow_elemwise(x, s-1.0))

class pow_elemwise(proto_pow):
    impl = assert_same_shapes(numpy.ndarray.__pow__)

class ipow_elemwise(proto_pow, inplace):
    impl = assert_same_shapes(numpy.ndarray.__ipow__)


class pow_scalar_l(omega_op):
    impl = tensor_scalar_op(numpy.ndarray.__pow__)
    def grad(x, s, gz):
        return gz * x * (pow_scalar_l(s,x-1.0))

class pow_scalar_r(omega_op):
    impl = tensor_scalar_op(numpy.ndarray.__pow__)
    def grad(x, s, gz):
        return gz * s * (pow_scalar_r(x,s-1.0))

class ipow_scalar_r(omega_op, inplace):
    impl = tensor_scalar_op(numpy.ndarray.__ipow__)
    def grad(x, s, gz):
        return gz * s * (pow_scalar_r(x,s-1.0))

## Others ##

class minmax(omega_op):
    nout = 2
    def impl(x):
        return x.min, x.max

class fill(omega_op):
    impl = lambda model, value: (model * 0) + value

class sum(omega_op):
    impl = numpy.sum
    def grad(x, gz):
        return fill(x, gz)

    
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



