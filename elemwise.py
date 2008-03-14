
from copy import copy
from gof import Op
from gof.utils import AbstractFunctionError

from tensor import _Op

class Elemwise(_Op):
    
    def var_desc(self):
        raise AbstractFunctionError()

    def c_var_names(self):
        idesc, odesc = self.var_desc()
        return [[i[0] for i in idesc],
                [o[0] for o in odesc]]

    def loop_variables(self):
        idesc, odesc = self.var_desc()
        return [[i[0] for i in idesc if i[1]],
                [o[0] for o in odesc if o[1]]]

    def propagate_broadcastable(self, *inputs):
        idesc, odesc = self.var_desc()
        nonloop_o = [o[0] for o in odesc if not o[1]]
        if nonloop_o:
            raise Exception("Cannot infer broadcastable for non-loop variable(s) %s" % nonloop_o)
        all_bcast = [broadcastable for broadcastable, i in zip(inputs, idesc) if i[1]]
        if reduce(lambda x, y: x is not False and x == y and y, [len(x) for x in all_bcast]) is False:
            raise TypeError("Inputs that are loop variables do not all have the same number of dimensions.")
        ret = []
        for arr in zip(*all_bcast):
            if 0 in arr:
                ret.append(0)
            else:
                ret.append(1)
        return [ret] * self.nout

    def c_code_init(self):
        raise AbstractFunctionError()

    def c_code_foreach(self):
        raise AbstractFunctionError()

    def c_code_finalize(self):
        raise AbstractFunctionError()

    def __c_code(self):

        try: before = self.c_code_init()
        except AbstractFunctionError: before = ""

        during = self.c_code_foreach()

        try: after  = self.c_code_finalize()
        except AbstractFunctionError: after = ""

        idesc, odesc = self.var_desc()
        input_loop_vars = [x[0] for x in idesc if x[1]]
        output_loop_vars = [x[0] for x in odesc if x[1]]
        
        aliases = {}
        try:
            dmap = self.destroy_map()
        except:
            dmap = {}
        if dmap != {}:
            for (oname, isloopvar), output in zip(odesc, self.outputs):
                if isloopvar:
                    for input in dmap.get(output, []):
                        iname = idesc[self.inputs.index(input)][0]
                        aliases["%("+oname+")s"] = "%("+iname+")s"
                        output_loop_vars.remove(oname)

        ret = elemwise_wrap(before, during, after,
                            ["%("+v+")s" for v in input_loop_vars],
                            ["%("+v+")s" for v in output_loop_vars],
                            aliases)

        return ret

    def c_validate_update(self):
        (valupd, valupd_cleanup), (code, code_cleanup) = self.__c_code()
        return valupd

    def c_validate_update_cleanup(self):
        (valupd, valupd_cleanup), (code, code_cleanup) = self.__c_code()
        return valupd_cleanup

    def c_code(self):
        (valupd, valupd_cleanup), (code, code_cleanup) = self.__c_code()
        return code

    def c_code_cleanup(self):
        (valupd, valupd_cleanup), (code, code_cleanup) = self.__c_code()
        return code_cleanup
    
    @classmethod
    def inplace_version(cls):
        class Ret(cls, Destroyer):
            def destroy_map(self):
                return {self.outputs[0]: [self.inputs[0]]}
        return Ret







class Looper:

    def __init__(self, declare, init, acquire, next, cleanup):
        self.declare = declare
        self.init = init
        self.acquire = acquire
        self.next = next
        self.cleanup = cleanup

    def fill(self, **template):
        ret = copy(self)
        ret.declare %= template
        ret.init %= template
        ret.acquire %= template
        ret.next %= template
        ret.cleanup %= template
        return ret

general_looper = Looper(declare = """
                        PyArrayIterObject* %(iter)s_%(name)s = NULL;
                        //dtype_%(name)s& %(name)s_i;
                        """,
                        init = """
                        %(iter)s_%(name)s = (PyArrayIterObject*)PyArray_IterNew((PyObject*)%(name)s);
                        if (%(iter)s_%(name)s == NULL) {
                            PyErr_SetString(PyExc_ValueError, "Could not make an iterator over variable %(name)s.");
                            %%(fail)s;
                        }
                        """,
                        acquire = "dtype_%(name)s& %(name)s_i = *((dtype_%(name)s*)%(iter)s_%(name)s->dataptr);",
                        next = "PyArray_ITER_NEXT(%(iter)s_%(name)s);",
                        cleanup = "if (%(iter)s_%(name)s) Py_DECREF(%(iter)s_%(name)s);")

contiguous_looper = Looper(declare = """
                           dtype_%(name)s* %(iter)s_%(name)s = NULL;
                           //dtype_%(name)s %(name)s_i;
                           """,
                           init = "%(iter)s_%(name)s = (dtype_%(name)s*)PyArray_DATA(%(name)s);",
                           acquire = "dtype_%(name)s& %(name)s_i = *%(iter)s_%(name)s;",
                           next = "%(iter)s_%(name)s++;",
                           cleanup = "")

alias_looper = Looper(declare = "", #dtype_%(name)s%(type_suffix)s %(name)s_i;",
                      init = "",
                      acquire = "dtype_%(name)s& %(name)s_i = %(other)s_i;",
                      next = "",
                      cleanup = "")

def elemwise_loopcode(loopcode, loopers):

    def make_block(loopers, type):
        return "\n".join([getattr(looper, type) for looper in loopers])

    template = dict(loopcode = loopcode)
    for block_type in ['declare', 'init', 'next', 'acquire', 'cleanup']:
        template[block_type] = make_block(loopers, block_type)

    code = """
    %(init)s
    while (__SIZE--) {
        %(acquire)s
        %(loopcode)s
        %(next)s
    }
    """ % template

    cleanup = """
    %(cleanup)s
    """ % template

    return template['declare'], code, cleanup


def elemwise_wrap(beforeloop, inloop, afterloop, input_loop_vars, output_loop_vars, aliases):

    if len(input_loop_vars) == 0:
        raise Exception("You need at least one input loop variable.")
    
    validate = """
    npy_intp nd = %(first_loop_var)s->nd;
    npy_intp* dims = %(first_loop_var)s->dimensions;
    npy_intp* dims2;
    """ % dict(first_loop_var = input_loop_vars[0])

    for other_loop_var in input_loop_vars[1:]:
        validate += """
        if (%(other_loop_var)s->nd != nd) {
            PyErr_SetString(PyExc_ValueError, \"The number of dimensions of the inputs do not match.\");
            %%(fail)s
        }
        dims2 = %(other_loop_var)s->dimensions;
        for (int i = 0; i < nd; i++) {
            if (dims2[i] != dims[i]) {
                PyErr_SetString(PyExc_ValueError, \"The dimensions of the inputs do not match.\");
                %%(fail)s;
            }
        }
        """ % dict(other_loop_var = other_loop_var)

    update = ""
    for output_loop_var in output_loop_vars:
        update += """
        if (!%(output_loop_var)s) {
            %(output_loop_var)s = (PyArrayObject*)PyArray_EMPTY(nd, dims, type_num_%(output_loop_var)s, 0);
        }
        else {
            PyArray_Dims new_dims;
            new_dims.len = nd;
            new_dims.ptr = dims;
            PyObject* success = PyArray_Resize(%(output_loop_var)s, &new_dims, 0, PyArray_CORDER);
            if (!success) {
                // If we can't resize the ndarray we have we can allocate a new one.
                PyErr_Clear();
                Py_XDECREF(%(output_loop_var)s);
                %(output_loop_var)s = (PyArrayObject*)PyArray_EMPTY(nd, dims, type_num_%(output_loop_var)s, 0);
            }
        }
        if (!%(output_loop_var)s) {
            %%(fail)s
        }
        """ % dict(output_loop_var = output_loop_var)

    validate_update = validate + update

    validate_update_cleanup = ""
    
    
    general_loopers = []
    general_loopers += [general_looper.fill(name = loop_var, iter = "iter_general_") for loop_var in input_loop_vars + output_loop_vars]
    general_loopers += [alias_looper.fill(name = v1, other = v2, iter = "iter_general_") for v1, v2 in aliases.items()]

    contig_loopers = []
    contig_loopers += [contiguous_looper.fill(name = loop_var, iter = "iter_contig_") for loop_var in input_loop_vars + output_loop_vars]
    contig_loopers += [alias_looper.fill(name = v1, other = v2, iter = "iter_contig_") for v1, v2 in aliases.items()]

    general_decl, general_loop, general_loop_cleanup = elemwise_loopcode(inloop, general_loopers)
    contig_decl, contig_loop, contig_loop_cleanup = elemwise_loopcode(inloop, contig_loopers)
    
    contiguity_check = "".join(["all_c_contiguous &= PyArray_ISCARRAY(%(loop_var)s);\n" \
                                "all_f_contiguous &= PyArray_ISFARRAY(%(loop_var)s);\n" \
                                    % dict(loop_var = loop_var)
                                for loop_var in input_loop_vars + output_loop_vars])

    v1 = input_loop_vars[0]


    code = """
    %(general_decl)s
    %(contig_decl)s
    npy_intp __SIZE = PyArray_SIZE(%(v1)s);
    %(beforeloop)s
    bool all_c_contiguous = 1;
    bool all_f_contiguous = 1;
    %(contiguity_check)s
    if (all_c_contiguous || all_f_contiguous) {
        %(contig_loop)s
    }
    else {
        %(general_loop)s
    }
    %(afterloop)s
    """ % locals()

    cleanup = """
    if (all_c_contiguous || all_f_contiguous) {
        %(contig_loop_cleanup)s
    }
    else {
        %(general_loop_cleanup)s
    }
    """ % locals()

#     print """
#     VALIDATE_UPDATE
#     %(validate_update)s
#     CODE
#     %(code)s
#     CLEANUP
#     %(cleanup)s
#     VU_CLEANUP
#     %(validate_update_cleanup)s
#     """ % locals()
    
    return (validate_update, validate_update_cleanup), (code, cleanup)















































# import core



# def elemwise_loopcode(loopcode, init_template, next_template, acquire_template, cleanup_template, loop_vars, writable_loop_vars, aliases):
#     all_loop_vars = loop_vars + writable_loop_vars

#     template = dict(
#         init = "".join([init_template % dict(loop_var = loop_var) for loop_var in all_loop_vars]),
#         next = "".join([next_template % dict(loop_var = loop_var) for loop_var in all_loop_vars]),
#         cleanup = "".join([cleanup_template % dict(loop_var = loop_var) for loop_var in all_loop_vars]),
#         idefs = "".join([("%(loop_var)s_dtype %(loop_var)s_i = " + acquire_template + ";\n")
#                          % dict(loop_var = loop_var) for loop_var in loop_vars]),
#         odefs = "".join([("%(loop_var)s_dtype& %(loop_var)s_i = " + acquire_template + ";\n")
#                          % dict(loop_var = loop_var) for loop_var in writable_loop_vars]),
#         aliasdefs = "".join(["%(v1)s_dtype %(v1)s_i = %(v2)s_i;\n" % dict(v1=v1, v2=v2)
#                              for v1, v2 in aliases.items()]),
#         loopcode = loopcode
#         )

#     code = """
#     %(init)s
#     while (__elemwise_size--) {
#         %(idefs)s
#         %(odefs)s
#         %(aliasdefs)s
#         %(loopcode)s
#         %(next)s
#     }
#     %(cleanup)s
#     """ % template

#     return code


# def elemwise_wrap(beforeloop, inloop, afterloop, loop_vars, writable_loop_vars, aliases):

#     check_init = """
#     npy_intp nd = %(loop_var)s->nd;
#     npy_intp* dims = %(loop_var)s->dimensions;
#     npy_intp* dims2;
#     """

#     check = """
#     if (%(loop_var)s->nd != nd) {
#         PyErr_SetString(PyExc_ValueError, \"The number of dimensions of the inputs do not match.\");
#     }
#     dims2 = %(loop_var)s->dimensions;
#     for (int i = 0; i < nd; i++) {
#         if (dims2[i] != dims[i]) {
#             PyErr_SetString(PyExc_ValueError, \"The dimensions of the inputs do not match.\");
#             return 1;
#         }
#     }
#     """
    
#     general_init = "PyArrayIterObject* %(loop_var)s_iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)%(loop_var)s);\n"
# #         "if (%(loop_var)s_iter == NULL) {\n" \
# #         "    PyErr_SetString(PyExc_ValueError, \"Could not make an iterator over variable %(loop_var)s.\");\n" \
# #         "    return 1;\n" \
# #         "}\n"
#     general_next = "PyArray_ITER_NEXT(%(loop_var)s_iter);\n"
#     general_acquire = "*((%(loop_var)s_dtype*)%(loop_var)s_iter->dataptr)";
#     general_cleanup = "if (%(loop_var)s_iter) Py_DECREF(%(loop_var)s_iter);\n";

#     contiguous_init = "%(loop_var)s_dtype* __restrict__ %(loop_var)s_iter = (%(loop_var)s_dtype*)PyArray_DATA(%(loop_var)s);\n"
#     contiguous_next = "%(loop_var)s_iter++;\n"
#     contiguous_acquire = "*%(loop_var)s_iter"
#     contiguous_cleanup = ""
    
#     all_loop_vars = loop_vars + writable_loop_vars
#     v1 = (loop_vars + writable_loop_vars)[0]
#     template = dict(
#         v1 = v1,
#         check_init = check_init % dict(loop_var = v1),
#         check = "\n".join([check % dict(loop_var = loop_var) for loop_var in loop_vars + writable_loop_vars if loop_var is not v1]),
#         beforeloop = beforeloop,
#         general_loop = elemwise_loopcode(
#             inloop,
#             general_init, general_next, general_acquire, general_cleanup,
#             loop_vars, writable_loop_vars, aliases),
#         contiguous_loop = elemwise_loopcode(
#             inloop,
#             contiguous_init, contiguous_next, contiguous_acquire, contiguous_cleanup,
#             loop_vars, writable_loop_vars, aliases),
#         contiguity_check = "".join(["all_c_contiguous &= PyArray_ISCARRAY(%(loop_var)s);\n" \
#                                     "all_f_contiguous &= PyArray_ISFARRAY(%(loop_var)s);\n" \
#                                         % dict(loop_var = loop_var)
#                                     for loop_var in all_loop_vars]),
#         afterloop = afterloop)
    
#     code = """
#     {
#     %(check_init)s
#     %(check)s
#     }
#     npy_intp __elemwise_size = PyArray_SIZE(%(v1)s);
#     %(beforeloop)s
#     bool all_c_contiguous = 1;
#     bool all_f_contiguous = 1;
#     %(contiguity_check)s
#     if (all_c_contiguous || all_f_contiguous) {
#         %(contiguous_loop)s
#     }
#     else {
#         %(general_loop)s
#     }
#     %(afterloop)s
#     """ % template

#     return code




# class elemwise(omega_op):

#     @staticmethod
#     def __clsinit__(cls, name, bases, dct):
#         for fname in ['c_init', 'c_foreach', 'c_finalize']:
#             gof.make_static(cls, fname)

#         # make impl, grad, etc. static methods
#         omega_op.__clsinit__(cls, name, bases, dct)

#     def TOGO_specs(self):
#         try:
#             return self.specs(*[input.spec for input in self.inputs])
#         except NotImplementedError:
#             inames, onames = self.variable_names()
#             linames, lonames = self.loop_variables()
#             for oname in onames:
#                 if oname not in lonames:
#                     raise Exception("cannot infer a specification automatically for variable " \
#                                     "%s.%s because it is not part of the elementwise loop - "\
#                                     "please override the specs method" % (self.__class__.__name__, oname))
#             shape, dtype = None, None
#             for iname, input in zip(inames, self.inputs):
#                 if iname in linames:
#                     if input.spec:
#                         shape = input.spec[2]
#             if shape is None:
#                 raise Exception("cannot infer a specification automatically for output variables " \
#                                 "because there is no input variable in the loop from which to get the shape, "\
#                                 "or their shape is unknown")

#             try:
#                 dtype = core.upcast(*[input.spec[1]
#                                       for iname, input in zip(inames, self.inputs)
#                                       if input.spec[0] is numpy.ndarray])
#             except IndexError:
#                 raise Exception("not all numpy inputs are specified")

#             dmap = self.destroy_map()

#             res = []
#             for output in self.outputs:
#                 inplace_inputs = dmap.get(output, [])
#                 if inplace_inputs:
#                     assert len(inplace_inputs) == 1
#                     res.append(inplace_inputs[0].spec)
#                 else:
#                     res.append((numpy.ndarray, dtype, shape))
                    
#             if self.nout == 1:
#                 return res[0]
#             else:
#                 return res
        
#     def TOGO_alloc(self, except_list = []):
#         dmap = self.destroy_map()
#         vmap = self.view_map()

#         gof.PythonOp.alloc(self, except_list = except_list + dmap.keys())
#         for output, (input, ) in dmap.items():
#             if output not in except_list:
#                 output.set_value(input.data)

#     def refresh_shape(self):
#         """Make the output have the right stuff"""
#         if len(self.outputs) > 1:
#             raise NotImplementedError('multiple outputs')

#         dmap = self.destroy_map()
#         vmap = self.view_map()
#         if dmap != {} or vmap != {}:
#             raise NotImplementedError('destroys or views confuse things',
#                     self.__class__, dmap, vmap)

#         # take the shape of the leftmost loop_variable input
#         inames, onames = self.variable_names()
#         linames, lonames = self.loop_variables()

#         unknown_output_names = [n for n in onames if n not in lonames]
#         if len(unknown_output_names):
#             raise Exception("cannot infer a specification automatically for variables " \
#                             "%s.{%s} because it is not part of the elementwise loop - "\
#                             "please override the specs method" % 
#                             (self.__class__.__name__, str(unknown_output_names)))

#         # shape is leftmost loop-variable input
#         input_loop_shapes = [i.shape for n,i in zip(inames, self.inputs) if n in linames]
#         if len(input_loop_shapes) == 0:
#             raise Exception("cannot infer a specification automatically for output variables " \
#                             "because there is no input loop variable ")
#         for i in xrange(1,len(input_loop_shapes)):
#             if  input_loop_shapes[i] != input_loop_shapes[0]:
#                 raise Exception("Input loop variables have different shapes", self.__class__)

#         return input_loop_shapes[0]

#     def refresh_dtype(self):
#         return core.upcast(*[i.dtype for i in self.inputs if hasattr(i, 'dtype')])

#     @classmethod
#     def set_impl(cls, impl):
#         gof.lib.make_static(cls, 'impl')

#     @staticmethod
#     def is_loop_var(name):
#         return name.endswith("_i")

#     @staticmethod
#     def extract_name(name):
#         if name.endswith("_i"):
#             return name[:-2]
#         else:
#             return name

#     @classmethod
#     def variable_names(cls):
#         (inames, onames), _1, _2, _3 = inspect.getargspec(cls.c_foreach)
#         spec = ([cls.extract_name(name) for name in inames],
#                 [cls.extract_name(name) for name in onames])
#         if cls.c_init is not elemwise.c_init:
#             (inames, onames), _1, _2, _3 = inspect.getargspec(cls.c_init)
#             assert spec == (list(inames), list(onames))
#         if cls.c_finalize is not elemwise.c_finalize:
#             (inames, onames), _1, _2, _3 = inspect.getargspec(cls.c_finalize)
#             assert spec == (list(inames), list(onames))
#         return spec

#     @classmethod
#     def loop_variables(cls):
#         (inames, onames), _1, _2, _3 = inspect.getargspec(cls.c_foreach)
#         return ([cls.extract_name(name) for name in inames if cls.is_loop_var(name)],
#                 [cls.extract_name(name) for name in onames if cls.is_loop_var(name)])

#     def _c_init(self):
#         return self.c_init(self.inputs, self.outputs)
        
#     def c_init(inputs, outputs):
#         return ""

#     def _c_foreach(self):
#         return self.c_foreach(self.inputs, self.outputs)
        
#     def c_foreach(inputs, outputs):
#         raise NotImplementedError()

#     def _c_finalize(self):
#         return self.c_finalize(self.inputs, self.outputs)

#     def c_finalize(inputs, outputs):
#         return ""

#     def c_code(self, converters = None, elemwise_wrap = elemwise_wrap):
#         def mangle(name):
#             if name.endswith("_i"):
#                 return name[:-2]
#             else:
#                 return name

#         try:
#             self._c_impl()
#             raise Exception("c_impl is not used by elemwise ops - define behavior in c_foreach instead")
#         except NotImplementedError:
#             pass

#         before = self._c_init()
#         during = self._c_foreach()
#         after  = self._c_finalize()
        
#         (inames, onames) = self.variable_names()
#         (linames, lonames) = self.loop_variables()

#         aliases = {}
#         dmap = self.destroy_map()
#         if dmap != {}:
#             for oname, output in zip(onames, self.outputs):
#                 if oname in lonames:
#                     for input in dmap.get(output, []):
#                         aliases[inames[self.inputs.index(input)]] = oname
                        
#         behavior = elemwise_wrap(before, during, after,
#                                  [name for name in linames if name not in aliases],
#                                  lonames,
#                                  aliases)
        
#         return cgen(self.__class__.__name__, behavior, inames + onames, self.inputs + self.outputs, converters)

#     @classmethod
#     def inplace_version(cls, dmap = {0: 0}):
#         inames, onames = cls.variable_names()
#         linames, lonames = cls.loop_variables()
#         for i, oname in enumerate(onames):
#             if i in dmap:
#                 assert oname in lonames
        
#         class C(cls):
#             def destroy_map(self):
#                 assert cls.destroy_map(self) == {}
#                 ret = {}
#                 for output, input in dmap.items():
#                     ret[self.outputs[output]] = [self.inputs[input]]
#                 return ret
#             def _impl(self):
#                 if self.impl is not cls.impl:
#                     # If the user sets his own inplace operation, we use it
#                     return cls._impl(self)
#                 else:
#                     res = cls._impl(self)
#                     if isinstance(res, (list, tuple)):
#                         res = copy.copy(res)
#                     else:
#                         res = [res]
#                     for output, input in dmap.items():

#                         # The default implementation returned a copy, so we just
#                         # overwrite the original input with the contents of that copy
#                         # This is not meant to be efficient, only correct.
#                         #
#                         # TODO: change this to use set_value_inplace
#                         a = self.inputs[input].data
#                         a[:] = res[output]
#                         res[output] = a
#                     if len(res) == 1:
#                         return res[0]
#                     else:
#                         return res

#         if dmap == {0:0}:
#             C.__name__ = cls.__name__ + "_inplace" % dmap
#         else:
#             C.__name__ = cls.__name__ + "_inplace%s" % dmap
#         return C


