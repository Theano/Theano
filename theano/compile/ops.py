"""This file contain auxiliary Ops, used during the compilation phase."""
import copy
import warnings

#import theano
from theano import gof


def register_view_op_c_code(type, code, version=()):
    """ Tell ViewOp how to generate C code for a Theano Type

    :param typ: A Theano type. It must be the Theano class itself and not an
                instance of the class.
    :param code: C code that deep copies the Theano type 'typ'.
                 Use %(iname)s and %(oname)s for the input and output C
                 variable names respectively.
    :param version: A number indicating the version of the code, for cache.
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

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

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
        for t, (c, v) in sorted(self.c_code_and_version.items()):
            if not v:
                warnings.warn("Type %s has C code for ViewOp, but it has "
                        "no version. You should add a 'version' keyword arg "
                        "when calling register_deep_copy_op_c_code." % t,
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

    This Op is declared as destructive while it is not destroying
    anything. It returns a view. This is used to prevent destruction of
    the output variables of a Theano function.

    There is a mechanism in Theano that should prevent this, but the use
    of OutputGuard adds a safeguard: it may be possible for some optimization
    run before the add_destroy_handler phase to bypass this mechanism, by
    making in-place optimizations.

    TODO: find a current full explanation.
    """
    destroy_map = {0: [0]}

_output_guard = OutputGuard()


def register_deep_copy_op_c_code(typ, code, version=()):
    """ Tell DeepCopyOp how to generate C code for a Theano Type

    :param typ: A Theano type. It must be the Theano class itself and not an
                instance of the class.
    :param code: C code that deep copies the Theano type 'typ'.
                 Use %(iname)s and %(oname)s for the input and output C
                 variable names respectively.
    :param version: A number indicating the version of the code, for cache.
    """
    DeepCopyOp.c_code_and_version[typ] = (code, version)


class DeepCopyOp(gof.Op):
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version = {}

    def __init__(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, args, outs):
        if hasattr(args[0], 'copy'):
            #when args[0] is a an ndarray of 0 dimensions,
            #this return a numpy.dtype and not an ndarray
            #So when the args have a copy attribute we use it
            #as this don't have this problem
            outs[0][0] = args[0].copy()
        else:
            outs[0][0] = copy.deepcopy(args[0])

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(self.c_code_and_version.items(), key=lambda pair: str(pair[0])):
            if not v:
                warnings.warn("Type %s has C code for OutputGuard, but it has "
                        "no version. You should add a 'version' keyword arg "
                        "when calling register_OutputGuard_c_code." % t,
                        stacklevel=2)
                return ()
            version.append((str(t), v))

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

# List of Theano Types that one can add an extra dimension and for which
# Scan can deal with.
expandable_types = ()
