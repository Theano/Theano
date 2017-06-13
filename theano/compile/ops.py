"""
This file contains auxiliary Ops, used during the compilation phase and Ops
building class (:class:`FromFunctionOp`) and decorator (:func:`as_op`) that
help make new Ops more rapidly.

"""
from __future__ import absolute_import, print_function, division
from collections import OrderedDict

import copy
import six.moves.cPickle as pickle
import warnings

import theano
from theano import gof
from six import iteritems, integer_types
from six.moves import xrange


import numpy as np


def register_view_op_c_code(type, code, version=()):
    """
    Tell ViewOp how to generate C code for a Theano Type.

    Parameters
    ----------
    type : Theano type
        It must be the Theano class itself and not an instance of the class.
    code : C code
        Returns a view for the Theano type 'type'. Use %(iname)s and %(oname)s
        for the input and output C variable names respectively.
    version
        A number indicating the version of the code, for cache.

    """
    ViewOp.c_code_and_version[type] = (code, version)


class ViewOp(gof.Op):
    """
    Returns an inplace view of the input. Used internally by Theano.

    """

    view_map = {0: [0]}
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}
    __props__ = ()
    _f16_ok = True

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = x

    def __str__(self):
        return '%s' % self.__class__.__name__

    def c_code(self, node, nodename, inp, out, sub):
        iname, = inp
        oname, = out
        fail = sub['fail']

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            return code % locals()

        # Else, no C code
        return super(ViewOp, self).c_code(node, nodename, inp, out, sub)

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(iteritems(self.c_code_and_version),
                                key=lambda pair: str(pair[0])):
            if not v:
                warnings.warn("Type %s has C code for ViewOp, but it has no "
                              "version. You should add a 'version' keyword "
                              "arg when calling register_view_op_c_code." % t,
                              stacklevel=2)
                return ()
            version.append((str(t), v))

        return tuple(version)

    def infer_shape(self, node, input_shapes):
        return input_shapes

    def grad(self, args, g_outs):
        return g_outs

view_op = ViewOp()


class OutputGuard(ViewOp):
    """
    This op is used only internally by Theano.

    Only the AddDestroyHandler optimizer tries to insert them in the graph.

    This Op is declared as destructive while it is not destroying anything.
    It returns a view. This is used to prevent destruction of the output
    variables of a Theano function.

    There is a mechanism in Theano that should prevent this, but the use
    of OutputGuard adds a safeguard: it may be possible for some optimization
    run before the add_destroy_handler phase to bypass this mechanism, by
    making in-place optimizations.

    TODO: find a current full explanation.

    """
    destroy_map = {0: [0]}

    check_input = False

_output_guard = OutputGuard()


def register_deep_copy_op_c_code(typ, code, version=()):
    """
    Tell DeepCopyOp how to generate C code for a Theano Type.

    Parameters
    ----------
    typ : Theano type
        It must be the Theano class itself and not an instance of the class.
    code: C code
        Deep copies the Theano type 'typ'. Use %(iname)s and %(oname)s for the
        input and output C variable names respectively.
    version
        A number indicating the version of the code, for cache.

    """
    DeepCopyOp.c_code_and_version[typ] = (code, version)


class DeepCopyOp(gof.Op):
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}

    check_input = False
    __props__ = ()
    _f16_ok = True

    def __init__(self):
        pass

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, args, outs):
        if hasattr(args[0], 'copy'):
            # when args[0] is a an ndarray of 0 dimensions,
            # this return a numpy.dtype and not an ndarray
            # So when the args have a copy attribute we use it
            # as this don't have this problem
            outs[0][0] = args[0].copy()
        else:
            outs[0][0] = copy.deepcopy(args[0])

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(iteritems(self.c_code_and_version),
                                key=lambda pair: str(pair[0])):
            if not v:
                warnings.warn("Type %s has C code for DeepCopyOp, but it has "
                              "no version. You should add a 'version' keyword"
                              " arg when calling "
                              "register_deep_copy_op_c_code." % t,
                              stacklevel=2)
                return ()
            version.append((str(t), v))

        if version:
            version.append(1)
        return tuple(version)

    def c_code(self, node, name, inames, onames, sub):
        iname, = inames
        oname, = onames
        fail = sub['fail']

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            return code % locals()

        # Else, no C code
        return super(DeepCopyOp, self).c_code(node, name, inames, onames, sub)


deep_copy_op = DeepCopyOp()


def register_shape_c_code(type, code, version=()):
    """
    Tell Shape Op how to generate C code for a Theano Type.

    Parameters
    ----------
    typ : Theano type
        It must be the Theano class itself and not an instance of the class.
    code : C code
        Returns a vector representing the shape for the Theano type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively.
    version
        A number indicating the version of the code, for cache.

    """
    Shape.c_code_and_version[type] = (code, version)


class Shape(gof.Op):
    """
    L{Op} to return the shape of a matrix.

    Notes
    -----
    Non-differentiable.

    """

    _f16_ok = True

    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}

    check_input = False
    __props__ = ()

    def make_node(self, x):
        # Must work for all type that have a shape attribute.
        # This will fail at execution time.
        if not isinstance(x, theano.Variable):
            x = theano.tensor.as_tensor_variable(x)
        return gof.Apply(self, [x], [theano.tensor.lvector()])

    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        out[0] = theano._asarray(x.shape, dtype='int64')

    def infer_shape(self, node, in_shapes):
        return [[len(in_shapes[0])]]

    def connection_pattern(self, node):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [[False]]

    def grad(self, inp, grads):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [theano.gradient.DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        return [None]

    def c_code(self, node, name, inames, onames, sub):
        iname, = inames
        oname, = onames
        fail = sub['fail']

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            return code % locals()

        # Else, no C code
        return super(Shape, self).c_code(node, name, inames, onames, sub)

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(iteritems(self.c_code_and_version),
                                key=lambda pair: str(pair[0])):
            if not v:
                warnings.warn("Type %s has C code for Shape, but it has no "
                              "version. You should add a 'version' keyword "
                              "arg when calling register_shape_c_code." % t,
                              stacklevel=2)
                return ()
            version.append((str(t), v))

        if version:
            version.append(1)

        return tuple(version)


shape = Shape()
_shape = shape  # was used in the past, now use shape directly.


class Shape_i(gof.Op):
    """
    L{Op} to return the shape of a matrix.

    Notes
    -----
    Non-differentiable.

    """

    _f16_ok = True

    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}

    check_input = False

    __props__ = ("i",)

    def __init__(self, i):
        # As i will be used in the hash and that ndarray are not hashable,
        # we need to convert it to an int as it is hashable.
        if isinstance(i, np.ndarray):
            assert i.dtype in theano.tensor.integer_dtypes
        assert i == int(i)
        i = int(i)
        self.i = i

    # NB:
    # 1) params_type is defined as a property to avoid
    #    loop in Python import caused by importing theano.scalar below
    #    when params_type is defined directly in class code.
    # 2) We wrap scalar into ParamsType (instead of directly using scalar as op param)
    #    to avoid Theano converting scalar param to constant that would be later
    #    hardcoded as litteral in C code, making us loose all the advantages of
    #    using params.
    @property
    def params_type(self):
        return gof.ParamsType(i=theano.scalar.basic.int64)

    def __str__(self):
        return '%s{%i}' % (self.__class__.__name__, self.i)

    def make_node(self, x):
        # x could be one of a number of types
        # the only thing we require is that the variable have a .ndim,
        # and that the value have a .shape
        if not isinstance(x, theano.Variable):
            raise TypeError('x must be Variable with ndim attribute', x)
        if x.ndim <= self.i:
            raise TypeError('x has too few dimensions for Shape_i',
                            (x, self.i))
        return theano.Apply(self, [x], [theano.tensor.lscalar()])

    def perform(self, node, inp, out_, params):
        x, = inp
        out, = out_
        if out[0] is None:
            out[0] = theano._asarray(x.shape[self.i], dtype='int64')
        else:
            out[0][...] = x.shape[self.i]

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, ci, v) in sorted(iteritems(self.c_code_and_version),
                                    key=lambda pair: str(pair[0])):
            if not v:
                warnings.warn("Type %s has C code for Shape_i, but it has "
                              "no version. You should add a 'version' keyword "
                              "arg when calling register_shape_i_c_code." % t,
                              stacklevel=2)
                return ()
            version.append((str(t), v))

        if version:
            version.append(2)

        return tuple(version)

    def c_code(self, node, name, inames, onames, sub):
        iname, = inames
        oname, = onames
        fail = sub['fail']
        # i is then 'params->i', not just 'params'.
        i = sub['params'] + '->i'

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, check_input, version = self.c_code_and_version[itype]
            return (check_input + code) % locals()

        # Else, no C code
        return super(Shape_i, self).c_code(node, name, inames, onames, sub)

    def infer_shape(self, node, input_shapes):
        return [()]

    def connection_pattern(self, node):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [[False]]

    def grad(self, inp, grads):
        return [theano.gradient.grad_not_implemented(
                op=self, x_pos=0, x=inp[0],
                comment=("No gradient for the shape of a matrix "
                         "is implemented."))]


def shape_i(var, i, fgraph=None):
    """
    Equivalent of var.shape[i], but apply if possible the shape feature
    optimization.

    This is useful in optimization that need to get the shape. This
    remove the need of the following shape_feature optimization that
    convert it. So this speed up optimization and remove Equilibrium
    max iteration problems.

    Parameters
    ----------
    var
        The variable we want to take the shape of.
    i
        The shape dimensions we want
    fgraph : optional
        If var.fgraph do not exist, the fgraph that have the shape_feature to
        introduce var in to get the optimized shape.

    """
    if fgraph is None and hasattr(var, 'fgraph'):
        fgraph = var.fgraph
    if fgraph and hasattr(fgraph, 'shape_feature'):
        shape_feature = fgraph.shape_feature
        shape_of = shape_feature.shape_of

        def recur(node):
            if not node.outputs[0] in shape_of:
                for inp in node.inputs:
                    if inp.owner:
                        recur(inp.owner)
                # If the output var isn't marked as being in the graph,
                # we need to add it in the ShapeFeature.
                shape_feature.on_import(fgraph, node,
                                        'gof.ops.shape_i')
        if var not in shape_of:
            recur(var.owner)
        return shape_of[var][i]

    # If we are not able to use the shape feature, we should not put
    # Shape_i in the graph. Otherwise, the shape feature optimization
    # won't get applied.
    return var.shape[i]


def shape_i_op(i):
    key = i
    if key not in shape_i_op.cache:
        shape_i_op.cache[key] = Shape_i(i)
    return shape_i_op.cache[key]
shape_i_op.cache = {}


def register_shape_i_c_code(typ, code, check_input, version=()):
    """
    Tell Shape_i how to generate C code for a Theano Type.

    Parameters
    ----------
    typ : Theano type
        It must be the Theano class itself and not an instance of the class.
    code : C code
        Gets the shape of dimensions %(i)s for the Theano type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively.
    version
        A number indicating the version of the code, for cache.

    """
    Shape_i.c_code_and_version[typ] = (code, check_input, version)


# List of Theano Types that one can add an extra dimension and for which
# Scan can deal with.
expandable_types = ()


def load_back(mod, name):
    __import__(mod)
    import sys
    module = sys.modules[mod]
    obj = getattr(module, name)
    return obj


class FromFunctionOp(gof.Op):
    """
    Build a basic Theano Op around a function.

    Since the resulting Op is very basic and is missing most of the
    optional functionalities, some optimizations may not apply.  If you
    want to help, you can supply an infer_shape function that computes
    the shapes of the output given the shapes of the inputs.

    Also the gradient is undefined in the resulting op and Theano will
    raise an error if you attempt to get the gradient of a graph
    containing this op.

    """

    def __init__(self, fn, itypes, otypes, infer_shape):
        self.__fn = fn
        self.itypes = itypes
        self.otypes = otypes
        self.__infer_shape = infer_shape
        if self.__infer_shape is not None:
            self.infer_shape = self._infer_shape

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.__fn == other.__fn)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.__fn)

    def __str__(self):
        return 'FromFunctionOp{%s}' % self.__fn.__name__

    def perform(self, node, inputs, outputs):
        outs = self.__fn(*inputs)
        if not isinstance(outs, (list, tuple)):
            outs = (outs,)
        assert len(outs) == len(outputs)
        for i in range(len(outs)):
            outputs[i][0] = outs[i]

    def __reduce__(self):
        mod = self.__fn.__module__
        name = self.__fn.__name__
        try:
            obj = load_back(mod, name)
        except (ImportError, KeyError, AttributeError):
            raise pickle.PicklingError(
                "Can't pickle as_op(), not found as %s.%s" %
                (mod, name))
        else:
            if obj is not self:
                raise pickle.PicklingError(
                    "Can't pickle as_op(), not the object "
                    "at %s.%s" % (mod, name))
        return load_back, (mod, name)

    def _infer_shape(self, node, input_shapes):
        return self.__infer_shape(node, input_shapes)


def as_op(itypes, otypes, infer_shape=None):
    """
    Decorator that converts a function into a basic Theano op that will call
    the supplied function as its implementation.

    It takes an optional infer_shape parameter that should be a callable with
    this signature:

        def infer_shape(node, input_shapes):
            ...
            return output_shapes

    Here `input_shapes` and `output_shapes` are lists of tuples that represent
    the shape of the corresponding inputs/outputs.

    This should not be used when performance is a concern since the very basic
    nature of the resulting Op may interfere with certain graph optimizations.

    Examples
    --------
    @as_op(itypes=[theano.tensor.fmatrix, theano.tensor.fmatrix],
           otypes=[theano.tensor.fmatrix])
    def numpy_dot(a, b):
        return numpy.dot(a, b)

    """
    if not isinstance(itypes, (list, tuple)):
        itypes = [itypes]
    if any(not isinstance(t, theano.Type) for t in itypes):
        raise TypeError("itypes has to be a list of Theano types")
    if not isinstance(otypes, (list, tuple)):
        otypes = [otypes]
    if any(not isinstance(t, theano.Type) for t in otypes):
        raise TypeError("otypes has to be a list of Theano types")

    # make sure they are lists and not tuples
    itypes = list(itypes)
    otypes = list(otypes)

    if infer_shape is not None and not callable(infer_shape):
        raise TypeError("infer_shape needs to be a callable")

    def make_op(fn):
        return FromFunctionOp(fn, itypes, otypes, infer_shape)
    return make_op


def register_rebroadcast_c_code(typ, code, version=()):
    """
    Tell Rebroadcast how to generate C code for a Theano Type.

    typ : Theano type
        It must be the Theano class itself and not an instance of the class.
    code : C code
        That checks if the dimension %(axis)s is of shape 1 for the Theano type
        'typ'. Use %(iname)s and %(oname)s for the input and output C variable
        names respectively, and %(axis)s for the axis that we need to check.
        This code is put in a loop for all axes.
    version
        A number indicating the version of the code, for cache.

    """
    Rebroadcast.c_code_and_version[typ] = (code, version)


class Rebroadcast(gof.Op):
    """
    Change the input's broadcastable fields in some predetermined way.

    See Also
    --------
    unbroadcast <theano.tensor.unbroadcast>
    addbroadcast <theano.tensor.addbroadcast>
    patternbroadcast <theano.tensor.patternbroadcast>

    Notes
    -----
    Works inplace and works for CudaNdarrayType.

    Example
    -------
    `Rebroadcast((0, True), (1, False))(x)` would make `x` broadcastable in
    axis 0 and not broadcastable in axis 1.

    """

    view_map = {0: [0]}
    _f16_ok = True
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}

    check_input = False
    __props__ = ("axis",)
    _f16_ok = True

    def __init__(self, *axis):
        # Sort them to make sure we merge all possible case.
        items = sorted(axis)
        self.axis = OrderedDict(items)
        for axis, broad in iteritems(self.axis):
            if not isinstance(axis, (np.integer, integer_types)):
                raise TypeError("Rebroadcast needs integer axes. "
                                "Got {}".format(axis))

            if not isinstance(broad, (np.bool_, bool)):
                raise TypeError("Rebroadcast needs bool for new broadcast "
                                "pattern. Got {}".format(broad))

    def __hash__(self):
        # Need special __hash__ as dict aren't hashable.
        # no ambiguity because each item key is unique
        items = sorted(iteritems(self.axis))
        return hash((type(self), tuple(items)))

    def __str__(self):
        if len(self.axis) == 0:
            broadcast_pattern = []
        else:
            broadcast_pattern = ['?' for i
                                 in xrange(1 + max(self.axis.keys()))]
        for k, v in iteritems(self.axis):
            broadcast_pattern[k] = str(int(v))
        return '%s{%s}' % (self.__class__.__name__,
                           ','.join(broadcast_pattern))

    def make_node(self, x):
        if self.axis.keys() and (x.ndim <= max(self.axis.keys())):
            raise ValueError('Trying to rebroadcast non-existent dimension')
        t = x.type.clone(
            broadcastable=[self.axis.get(i, b)
                           for i, b in enumerate(x.type.broadcastable)])
        return gof.Apply(self, [x], [t()])

    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        for axis, value in iteritems(self.axis):
            if value and x.shape[axis] != 1:
                raise ValueError('Dimension %s in Rebroadcast\'s input was'
                                 ' supposed to be 1 (got %s instead)' %
                                 (axis, x.shape[axis]))
        out[0] = x

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        # restore the broadcasting pattern of the input
        return Rebroadcast(*[(axis, x.type.broadcastable[axis])
                             for axis, value in iteritems(self.axis)])(gz),

    def infer_shape(self, node, ishapes):
        assert len(ishapes) == 1
        l = []
        one = theano.tensor.basic.constant(1)
        for ax in xrange(len(ishapes[0])):
            if self.axis.get(ax, False):
                l.append(one)
            else:
                l.append(ishapes[0][ax])

        return [tuple(l)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(*eval_points, **dict(return_list=True))

    def c_code(self, node, nodename, inp, out, sub):
        iname, = inp
        oname, = out
        fail = sub['fail']

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            final_code = ""
            for axis, value in iteritems(self.axis):
                if value:
                    final_code += code % locals()
            return final_code + """
            Py_XDECREF(%(oname)s);
            %(oname)s = %(iname)s;
            Py_XINCREF(%(oname)s);
            """ % locals()
        return super(Rebroadcast, self).c_code(node, nodename, inp, out, sub)

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(iteritems(self.c_code_and_version),
                                key=lambda pair: str(pair[0])):
            if not v:
                warnings.warn("Type %s has C code for Rebroadcast, but it "
                              "has no version. You should add a 'version' "
                              "keyword arg when calling "
                              "register_rebroadcast_c_code." % t,
                              stacklevel=2)
                return ()
            version.append((str(t), v))

        if version:
            version.append(1)
        return tuple(version)


def register_specify_shape_c_code(typ, code, version=(),
                                  c_support_code_apply=None):
    """
    Tell SpecifyShape how to generate C code for a Theano Type.

    Parameters
    ----------
    typ : Theano type
        It must be the Theano class itself and not an instance of the class.
    code : C code
        Checks the shape and returns a view for the Theano type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively. %(shape)s is the vector of shape of %(iname)s.
        Check that its length is good.
    version
        A number indicating the version of the code, for cache.
    c_support_code_apply
        Extra code.

    """
    SpecifyShape.c_code_and_version[typ] = (code, version,
                                            c_support_code_apply)


class SpecifyShape(gof.Op):
    """
    L{Op} that puts into the graph the user-provided shape.

    In the case where this op stays in the final graph, we assert the shape.
    For this the output of this op must be used in the graph. This is not
    the case most of the time if we only take the shape of the output.
    Maybe there are other optimizations that will mess with this.

    Notes
    -----
    Maybe in the future we will never do the assert!

    We currently don't support specifying partial shape information.

    TODO : test this op with sparse. Do C code for them too.

    """

    view_map = {0: [0]}
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}
    __props__ = ()
    _f16_ok = True

    def make_node(self, x, shape):
        if not isinstance(x, gof.Variable):
            x = theano.tensor.as_tensor_variable(x)
        shape = theano.tensor.as_tensor_variable(shape)
        assert shape.ndim == 1
        assert shape.dtype in theano.tensor.integer_dtypes
        if isinstance(shape, theano.tensor.TensorConstant):
            assert shape.data.size == x.ndim
        return gof.Apply(self, [x, shape], [x.type()])

    def perform(self, node, inp, out_):
        x, shape = inp
        out, = out_
        assert x.ndim == shape.size
        assert np.all(x.shape == shape), ("got shape", x.shape,
                                          "expected", shape)
        out[0] = x

    def infer_shape(self, node, shapes):
        xshape, sshape = shapes
        new_shape = []
        for dim in xrange(node.inputs[0].ndim):
            try:
                s = theano.tensor.get_scalar_constant_value(
                    node.inputs[1][dim])
                s = theano.tensor.as_tensor_variable(s)
                new_shape.append(s)
            except theano.tensor.NotScalarConstantError:
                new_shape.append(node.inputs[1][dim])

        assert len(new_shape) == len(xshape)
        return [new_shape]

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inp, grads):
        x, s = inp
        gz, = grads
        # Should I set an SpecifyShape on gz? I think so
        # But I don't do it now as we need to make an optimization
        # to remove that op from the graph to don't block other optimization
        # Should I do an optimizer that will remove the SpecifyShape?
        # I think Yes
        return [gz, theano.gradient.DisconnectedType()()]
        return [specify_shape(gz, s), theano.gradient.DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            # It means that the this op sits on top of a non-differentiable
            # path
            return [None]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def c_support_code_apply(self, node, name):
        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            _, _, support_code = self.c_code_and_version[itype]
            if support_code:
                return support_code
        return super(SpecifyShape, self).c_support_code_apply(node, name)

    def c_code(self, node, name, inames, onames, sub):
        iname, shape = inames
        oname, = onames
        fail = sub['fail']

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version, _ = self.c_code_and_version[itype]
            return code % locals()

        return super(SpecifyShape, self).c_code(node, node, inames,
                                                onames, sub)

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v, _) in sorted(iteritems(self.c_code_and_version),
                                   key=lambda pair: str(pair[0])):
            if not v:
                warnings.warn("Type %s has C code for SpecifyShape, but it "
                              "has no version. You should add a 'version' "
                              "keyword arg when calling "
                              "register_specify_shape_c_code." % t,
                              stacklevel=2)
                return ()
            version.append((str(t), v))

        return tuple(version)


specify_shape = SpecifyShape()
