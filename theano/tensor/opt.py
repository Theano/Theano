"""
Tensor optimizations addressing the ops in basic.py
"""
# TODO: intelligent merge for mul/add
# TODO: 0*x -> 0

import logging
_logger = logging.getLogger('theano.tensor.opt')

import operator
import itertools
import sys
import traceback
from itertools import izip

import numpy
import numpy as N  # guys... please don't do this in the library :(

import theano
from theano import gof
from theano.gof import opt, InconsistencyError, TopoOptimizer, graph
from theano.gof import Variable, Constant
from theano.gof.utils import MethodNotDefined
from theano.configparser import config
from elemwise import Elemwise, DimShuffle
from theano import scalar
import basic as T
from theano import compile  # to register the optimizer built by this file

from theano.gof.python25 import any, all
from theano.gof.opt import (Optimizer, pre_constant_merge,
                            pre_greedy_local_optimizer)
from theano.gof import toolbox, DestroyHandler
from basic import get_constant_value, ShapeError

# Remove0 is lazily imported to avoid circular imports.
Remove0 = None


theano.configparser.AddConfigVar('on_shape_error',
                                 "warn: print a warning and use the default"
                                 " value. raise: raise an error",
                                 theano.configparser.EnumStr("warn", "raise"),
                                 in_c_key=False)

# Utilities


def out2in(*local_opts):
    """WRITEME """
    return opt.TopoOptimizer(opt.LocalOptGroup(*local_opts),
                             order='out_to_in',
                             failure_callback=TopoOptimizer.warn_inplace)


def in2out(*local_opts, **kwargs):
    """WRITEME """
    return opt.TopoOptimizer(opt.LocalOptGroup(*local_opts),
                             order='in_to_out',
                             failure_callback=TopoOptimizer.warn_inplace,
                             **kwargs)


def _fill_chain(new_out, orig_inputs):
    for i in orig_inputs:
        new_out = T.fill(i, new_out)
    return [new_out]


def encompasses_broadcastable(b1, b2):
    """
    Returns True if the broadcastable patterns b1 and b2 are such that b2 is
    broadcasted to b1's shape and not the opposite.

    :param b1: the broadcastable attribute of a tensor type
    :param b2: the broadcastable attribute of a tensor type
    """
    if len(b1) < len(b2):
        return False
    b1 = b1[-len(b2):]
    return not any(v1 and not v2 for v1, v2 in zip(b1, b2))


def merge_broadcastables(broadcastables):
    return [all(bcast) for bcast in zip(*broadcastables)]


def scalarconsts_rest(inputs):
    """Partition a list of variables into two kinds:
    scalar constants, and the rest."""
    consts = []
    origconsts = []
    nonconsts = []
    for i in inputs:
        try:
            v = get_constant_value(i)
            consts.append(v)
            origconsts.append(i)
        except Exception:
            nonconsts.append(i)
    return consts, origconsts, nonconsts


def broadcast_like(value, template, env, dtype=None):
    """Return a Variable with the same shape and dtype as the template,
    filled by broadcasting value through it. `value` will be cast as
    necessary.

    """
    value = T.as_tensor_variable(value)
    if value.type == template.type:
        return value
    if template not in env.variables:
        raise NotImplementedError('broadcast_like currently requires the '
                                  'template Variable to be in the env already')
    if hasattr(env, 'shape_feature'):
        new_shape = env.shape_feature.shape_of[template]
    else:
        new_shape = template.shape
    if dtype is None:
        dtype = template.dtype
    rval = T.alloc(T.cast(value, dtype), *new_shape)
    # the template may have 1s in its shape without being broadcastable
    if rval.broadcastable != template.broadcastable:
        rval = T.unbroadcast(rval, *[i for i in xrange(rval.ndim)
                                     if rval.broadcastable[i]
            and not template.broadcastable[i]])
    assert rval.type.dtype == dtype
    assert rval.type.broadcastable == template.broadcastable
    return rval


theano.configparser.AddConfigVar('tensor.insert_inplace_optimizer_validate_nb',
        "-1: auto, if graph have less then 500 nodes 1, else 10",
        theano.configparser.IntParam(-1),
        in_c_key=False)


def inplace_elemwise_optimizer_op(OP):
    """
    We parametrise it to make it work for Elemwise and GpuElemwise op.
    """
    @gof.optimizer
    def inplace_elemwise_optimizer(env):
        """
        Usage: inplace_elemwise_optimizer.optimize(env)

        Attempts to replace all Broadcast ops by versions of them
        that operate inplace. It operates greedily: for each Broadcast
        Op that is encountered, for each output, tries each input to
        see if it can operate inplace on that input. If so, makes the
        change and go to the next output or Broadcast Op.

        Examples:
          x + y + z -> x += y += z
          (x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)
        """
        # We should not validate too often as this takes too much time to
        # execute!
        # It is the _dfs_toposort() fct in theano/gof/destroyhandler.py
        # that takes so much time.
        # Should we try to use another lib that does toposort?
        #   igraph: http://igraph.sourceforge.net/
        #   networkx: https://networkx.lanl.gov/
        # Should we try to use cython?
        #   Compiling only that fct is not enough, should we try to add the
        #   deque class too?
        #   And init the deque and other list to an upper bound number of
        #   elements?
        # Maybe Theano should do online toposort as in
        #   http://code.google.com/p/acyclic
        #
        # The next longest optimizer is the canonizer phase.
        # Then I think it is the [io_?]toposort (need to validate) so check if
        # the solution is also applicable there.

        # We execute `validate` after this number of change.
        check_each_change = config.tensor.insert_inplace_optimizer_validate_nb
        if check_each_change == -1:
            if len(env.nodes) > 500:
                check_each_change = 10
            else:
                check_each_change = 1

        nb_change_no_validate = 0
        chk = env.checkpoint()

        for node in list(graph.io_toposort(env.inputs, env.outputs)):
            op = node.op
            if not isinstance(op, OP):
                continue
            baseline = op.inplace_pattern
            protected_inputs = [
                f.protected for f in node.env._features if
                isinstance(f, theano.compile.function_module.Supervisor)]
            protected_inputs = sum(protected_inputs, [])  # flatten the list
            protected_inputs.extend(env.outputs)
            candidate_outputs = [i for i in xrange(len(node.outputs))
                                 if i not in baseline]
            # node inputs that are Constant, already destroyed,
            # env protected inputs and env outputs can't be used as inplace
            # target.
            # Remove here as faster.
            candidate_inputs = [i for i in xrange(len(node.inputs))
                                if i not in baseline.values() \
                                    and not isinstance(node.inputs[i],
                                                       Constant)\
                                    and not env.destroyers(node.inputs[i])\
                                    and node.inputs[i] not in protected_inputs]

            verbose = False

            raised_warning = not verbose

            for candidate_output in candidate_outputs:
                for candidate_input in candidate_inputs:
                    #remove inputs that don't have the same dtype as the output
                    if node.inputs[candidate_input].type != node.outputs[
                        candidate_output].type:
                        continue

                    inplace_pattern = dict(baseline, **{candidate_output:
                                                            candidate_input})
                    try:
                        if hasattr(op.scalar_op, "make_new_inplace"):
                            new_scal = op.scalar_op.make_new_inplace(
                                scalar.transfer_type(
                                    *[inplace_pattern.get(i, None) \
                                          for i in xrange(len(node.outputs))]))
                        else:
                            new_scal = op.scalar_op.__class__(
                                scalar.transfer_type(
                                    *[inplace_pattern.get(i, None) \
                                          for i in xrange(len(node.outputs))]))
                        new = OP(new_scal, inplace_pattern).make_node(
                            *node.inputs)

                        for r, new_r in zip(node.outputs, new.outputs):
                            env.replace(r, new_r,
                                        reason="inplace_elemwise_optimizer")
                        nb_change_no_validate += 1
                        if nb_change_no_validate >= check_each_change:
                            env.validate()
                            chk = env.checkpoint()
                            nb_change_no_validate = 0
                    except (ValueError, TypeError, InconsistencyError), e:
                        if check_each_change != 1 and not raised_warning:
                            print >> sys.stderr, (
                                    "Some inplace optimization was not "
                                    "performed due to unexpected error:")
                            print >> sys.stderr, e
                            raised_warning = True
                        env.revert(chk)
                        continue
                    candidate_inputs.remove(candidate_input)
                    node = new
                    baseline = inplace_pattern
                    break

        if nb_change_no_validate > 0:
            try:
                env.validate()
            except Exception:
                if not raised_warning:
                    print >> sys.stderr, ("Some inplace optimization was not "
                                          "performed due to unexpected error")
                env.revert(chk)
    return inplace_elemwise_optimizer

inplace_elemwise_optimizer = inplace_elemwise_optimizer_op(T.Elemwise)

compile.optdb.register('inplace_opt', inplace_elemwise_optimizer, 75,
                       'fast_run', 'inplace')


def register_canonicalize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['canonicalize'].register(name, lopt, 'fast_run', *tags)
    return lopt


def register_stabilize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['stabilize'].register(name, lopt, 'fast_run', *tags)
    return lopt


def register_specialize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['specialize'].register(name, lopt, 'fast_run', *tags)
    return lopt


def register_uncanonicalize(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['uncanonicalize'].register(name, lopt, 'fast_run', *tags)
    return lopt


def register_specialize_device(lopt, *tags, **kwargs):
    name = (kwargs and kwargs.pop('name')) or lopt.__name__
    compile.optdb['specialize_device'].register(name, lopt, 'fast_run', *tags)
    return lopt


#####################
# Dot optimizations #
#####################

@register_canonicalize
@register_stabilize
@gof.local_optimizer([None])
def local_0_dot_x(node):
    if not isinstance(node.op, T.Dot):
        return False

    x = node.inputs[0]
    y = node.inputs[1]
    replace = False
    try:
        if get_constant_value(x) == 0:
            replace = True
    except TypeError:
        pass

    try:
        if get_constant_value(y) == 0:
            replace = True
    except TypeError:
        pass

    if replace:
        constant_zero = T.constant(0, dtype=node.outputs[0].type.dtype)
        if x.ndim == 2 and y.ndim == 2:
            return [T.alloc(constant_zero, x.shape[0], y.shape[1])]
        elif x.ndim == 1 and y.ndim == 2:
            return [T.alloc(constant_zero, y.shape[1])]
        elif x.ndim == 2 and y.ndim == 1:
            return [T.alloc(constant_zero, x.shape[0])]
        elif x.ndim == 1 and y.ndim == 1:
            return [constant_zero]
        else:
            _logger.warning("Optimization Warning: "
                            "Optimization theano/opt.py:local_0_dot_x Found "
                            "that it could apply, but was not implemented "
                            "for dot product with these input types:\n"
                            "(%s, %s)",
                            x.type, y.type)

######################
# DimShuffle lifters #
######################


@gof.local_optimizer([None, None])
def local_dimshuffle_lift(node):
    """
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.
    """
    op = node.op
    if not isinstance(op, DimShuffle):
        return False

    input = node.inputs[0]
    inode = input.owner
    if inode and isinstance(inode.op, Elemwise) and (len(input.clients) == 1):
        return inode.op.make_node(*[DimShuffle(input.type.broadcastable,
                                               op.new_order,
                                               op.inplace)(input) for input in
                                    inode.inputs]).outputs
    if inode and isinstance(inode.op, DimShuffle):
        new_order = [x == 'x' and 'x' or inode.op.new_order[x] for x in
                     op.new_order]
        inplace = op.inplace and inode.op.inplace
        iinput = inode.inputs[0]
        if new_order == range(len(new_order)) and (len(new_order) ==
                                                   iinput.type.ndim):
            return [iinput]
        else:
            return DimShuffle(iinput.type.broadcastable, new_order,
                              inplace).make_node(iinput).outputs


@register_canonicalize
@gof.local_optimizer([])
def local_lift_transpose_through_dot(node):
    """
    dot(x,y).T -> dot(y.T, x.T)

    These optimizations "lift" (propagate towards the inputs) DimShuffle
    through dot product.  It allows to put the graph in a more standard shape,
    and to later merge consecutive DimShuffles.

    The transformation should be apply whether or not the transpose is
    inplace.  The newly-introduced transpositions are not inplace, this will
    be taken care of in a later optimization phase.
    """
    if not (isinstance(node.op, T.DimShuffle)
            and node.op.new_order == (1, 0)):
        return False
    if not (node.inputs[0].owner and node.inputs[0].owner.op == T.dot):
        return False
    x, y = node.inputs[0].owner.inputs

    if x.ndim == y.ndim == 2:
        return [T.dot(y.T, x.T)]


@gof.local_optimizer([])
def dimshuffle_as_view(node):
    op = node.op
    if not isinstance(op, DimShuffle) or op.inplace:
        return False
    new_op = DimShuffle(op.input_broadcastable, op.new_order, inplace=True)
    return [new_op(*node.inputs)]


register_specialize(dimshuffle_as_view, 'inplace')
register_canonicalize(local_dimshuffle_lift)
register_specialize(local_dimshuffle_lift)


@register_canonicalize
@gof.local_optimizer([])
def local_dimshuffle_no_inplace_at_canonicalize(node):
    if isinstance(node.op, T.DimShuffle) and node.op.inplace:
        return [T.DimShuffle(node.op.input_broadcastable,
                             node.op.new_order, inplace=False)(node.inputs[0])]


######################
# Casting operations #
######################

@register_canonicalize
@register_specialize
@gof.local_optimizer([T.TensorFromScalar])
def local_tensor_scalar_tensor(node):
    '''tensor_from_scalar(scalar_from_tensor(x)) -> x'''
    if isinstance(node.op, T.TensorFromScalar):
        s = node.inputs[0]
        if s.owner and isinstance(s.owner.op, T.ScalarFromTensor):
            t = s.owner.inputs[0]
            return [t]


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.ScalarFromTensor])
def local_scalar_tensor_scalar(node):
    '''scalar_from_tensor(tensor_from_scalar(x)) -> x'''
    if isinstance(node.op, T.ScalarFromTensor):
        t = node.inputs[0]
        if t.owner and isinstance(t.owner.op, T.TensorFromScalar):
            s = t.owner.inputs[0]
            return [s]

#####################################
# ShapeFeature, Shape optimizations
#####################################


class MakeVector(T.Op):
    """Concatenate a number of scalars together into a vector

    This is a simple version of stack() that introduces far less cruft
    into the graph. Should work with 0 inputs. The constant_folding
    optimization will remove it.
    """
    def __init__(self, dtype='int64'):
        self.dtype = dtype

    def __eq__(self, other):
        return type(self) == type(other) and self.dtype == other.dtype

    def __hash__(self):
        return hash(type(self)) ^ hash(self.dtype)

    def make_node(self, *inputs):
        inputs = map(T.as_tensor_variable, inputs)
        if not all(a.type == inputs[0].type for a in inputs) or (
            len(inputs) > 0 and inputs[0].dtype != self.dtype):
            dtype = theano.scalar.upcast(self.dtype,
                                         *[i.dtype for i in inputs])
            #upcast the input to the determined dtype,
            #but don't downcast anything
            assert dtype == self.dtype, (
                    "The upcast of the inputs to MakeVector should match the "
                    "dtype given in __init__.")
            if not all(self.dtype == T.cast(i, dtype=dtype).dtype
                       for a in inputs):
                raise TypeError("MakeVector.make_node expected inputs"
                                " upcastable to %s. got %s" % (
                        self.dtype,
                        str([i.dtype for i in inputs])
                        ))
            inputs = [T.cast(i, dtype=dtype) for i in inputs]
        assert all(self.dtype == a.dtype for a in inputs)
        assert all(a.ndim == 0 for a in inputs)

        if inputs:
            dtype = inputs[0].type.dtype
        else:
            dtype = self.dtype
        #bcastable = (len(inputs) == 1)
        bcastable = False
        otype = T.TensorType(
                broadcastable=(bcastable,),
                dtype=dtype)
        return T.Apply(self, inputs, [otype()])

    def __str__(self):
        return self.__class__.__name__

    def perform(self, node, inputs, out_):
        out, = out_
        # not calling theano._asarray as optimization
        if out[0] is None:
            out[0] = theano._asarray(inputs, dtype=node.outputs[0].dtype)
        else:
            # assume that out has correct dtype. there is no cheap way to check
            out[0][...] = inputs

    def grad(self, inputs, output_gradients):
        # If the output is of an integer dtype, no gradient shall pass
        if 'int' in self.dtype:
            return [None] * len(inputs)

        grads = []
        for i, inp in enumerate(inputs):
            if 'int' in inp.dtype:
                # No gradient wrt integer inputs
                grads.append(None)
            else:
                grads.append(output_gradients[0][i])
        return grads

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs

make_vector = MakeVector()


class MakeVectorPrinter:
    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print make_vector.")
        elif isinstance(r.owner.op, MakeVector):
            return "[%s]" % ", ".join(pstate.pprinter.process(
                    input, pstate.clone(precedence=1000)) for input
                                      in r.owner.inputs)
        else:
            raise TypeError("Can only print make_vector.")
T.pprint.assign(lambda pstate, r: r.owner and isinstance(
        r.owner.op, MakeVector), MakeVectorPrinter())


class Shape_i(T.Op):
    """
    L{Op} to return the shape of a matrix.

    @note: Non-differentiable.
    """
    def __init__(self, i):
        self.i = i

    def __hash__(self):
        return hash(type(self)) ^ self.i

    def __eq__(self, other):
        return type(self) == type(other) and self.i == other.i

    def __str__(self):
        return '%s{%i}' % (self.__class__.__name__, self.i)

    def make_node(self, x):
        # x could be one of a number of types
        # the only thing we require is that the variable have a .ndim,
        # and that the value have a .shape
        if not isinstance(x, T.Variable):
            raise TypeError('x must be Variable with ndim attribute', x)
        if x.ndim <= self.i:
            raise TypeError('x has too few dimensions for Shape_i',
                            (x, self.i))
        return T.Apply(self, [x], [T.lscalar()])

    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        if out[0] is None:
            out[0] = theano._asarray(x.shape[self.i], dtype='int64')
        else:
            out[0][...] = x.shape[self.i]

    def c_code_cache_version(self):
        return (0, 1)

    def c_code(self, node, name, inp, out_, sub):
        x, = inp
        out, = out_
        i = self.i
        if isinstance(node.inputs[0].type, T.TensorType):
            return """
            if(!%(out)s)
            %(out)s=(PyArrayObject*)PyArray_ZEROS(0, NULL, PyArray_INT64, 0);
            ((npy_int64*)PyArray_DATA(%(out)s))[0]=%(x)s->dimensions[%(i)s];
            """ % locals()

        elif node.inputs[0].type.__class__.__name__ == "CudaNdarrayType":
            #Don't want to import cuda stuff here.
            return """
            if(!%(out)s)
            %(out)s=(PyArrayObject*)PyArray_ZEROS(0, NULL, PyArray_INT64, 0);
            ((npy_int64*)PyArray_DATA(%(out)s))[0]=
                            CudaNdarray_HOST_DIMS(%(x)s)[%(i)s];
            """ % locals()
        else:
            #TODO: if your type is not listed here, make a damn registry of
            #      shape_i ops for various types of variables.
            #      Do not continue this madness.
            return super(Shape_i, self).c_code(node, name, (x,), (out,), sub)

    def grad(self, inp, grads):
        return [None]


class ShapeFeature(object):
    """Graph optimizer for removing all calls to shape()

    This optimizer replaces all Shapes and Subtensors of Shapes with
    Shape_i and MakeVector Ops.

    This optimizer has several goals:
    1. to 'lift' Shapes to as close to the inputs as possible.
    2. to infer the shape of every node in the graph in terms of the
       input shapes.
    3. remove all fills (T.second, T.fill) from the graph

    Lifting shapes as close to the inputs as possible is important for
    canonicalization because it is very bad form to have to compute
    something just to know how big it will be.  Firstly, it is a waste
    of time to compute such outputs.  But it is important to get rid
    of these outputs as early as possible in the compilation process
    because the extra computations make it appear as if many internal
    graph nodes have multiple clients.  Many optimizations refuse to
    work on nodes with multiple clients.

    Lifting is done by using an `<Op>.infer_shape` function if one is
    present, or else using a conservative default.  An Op that
    supports shape-lifting should define a infer_shape(self, node,
    input_shapes) function.  The argument input_shapes is a tuple of
    tuples... there is an interior tuple for each input to the node.
    The tuple has as many elements as dimensions.  The element in
    position i of tuple j represents the i'th shape component of the
    j'th input.  The function should return a tuple of tuples.  One
    output tuple for each node.output.  Again, the i'th element of the
    j'th output tuple represents the output[j].shape[i] of the
    function.  If an output is not a TensorType, then None should be
    returned instead of a tuple for that output.

    For example the infer_shape for a matrix-matrix product would accept
    input_shapes=((x0,x1), (y0,y1)) and return ((x0, y1),).


    Inferring the shape of internal nodes in the graph is important
    for doing size-driven optimizations.  If we know how big various
    intermediate results will be, we can estimate the cost of many Ops
    accurately, and generate c-code that is specific [e.g. unrolled]
    to particular sizes.

    In cases where you cannot figure out the shape, raise a ShapeError.

    .. note::

        Right now there is only the ConvOp that could really take
        advantage of this shape inference, but it is worth it even
        just for the ConvOp.  All that's necessary to do shape
        inference is 1) to mark shared inputs as having a particular
        shape, either via a .tag or some similar hacking; and 2) to
        add an optional Param() argument to promise that inputs will
        have a certain shape (or even to have certain shapes in
        certain dimensions). We can't automatically infer the shape of
        shared variables as they can change of shape during the
        execution by default.  (NOT IMPLEMENTED YET, BUT IS IN TRAC)


    Using Shape information in Optimizations
    ========================================

    To use this shape information in OPTIMIZATIONS, use the
    ``shape_of`` dictionary.

    For example:

    .. code-block:: python

        try:
            shape_of = node.env.shape_feature.shape_of
        except AttributeError:
            # This can happen when the mode doesn't include the ShapeFeature.
            return

        shape_of_output_zero = shape_of[node.output[0]]

    The ``shape_of_output_zero'' symbol will contain a tuple, whose
    elements are either integers or symbolic integers.

    TODO: check to see if the symbols are necessarily
    non-constant... or are integer literals sometimes Theano
    constants?? That would be confusing.

    """

    def shape_ir(self, i, r):
        """Return symbolic r.shape[i] for tensor variable r, int i"""
        if hasattr(r.type,"broadcastable") and r.type.broadcastable[i]:
            return self.lscalar_one
        else:
            return Shape_i(i).make_node(r).outputs[0]

    def shape_tuple(self, r):
        """Return a tuple of symbolic shape vars for tensor variable r"""
        return tuple([self.shape_ir(i,r) for i in xrange(r.ndim)])

    def default_infer_shape(self, node, i_shapes):
        """Return a list of shape tuple or None for the outputs of node.

        This function is used for Ops that don't implement infer_shape.
        Ops that do implement infer_shape should use the i_shapes parameter,
        but this default implementation ignores it.
        """
        rval = []
        for r in node.outputs:
            try:
                rval.append(self.shape_tuple(r))
            except AttributeError:
                rval.append(None)
        return rval

    def unpack(self, s_i):
        """Return a symbolic integer scalar for the shape element s_i.

        The s_i argument was produced by the infer_shape() of an Op subclass.
        """
        # unpack the s_i that the Op returned
        assert s_i is not None
        if s_i == 1:
            # don't make the optimizer merge a zillion ones together
            # by always returning the same object to represent 1
            return self.lscalar_one
        if type(s_i) in (int, long) or isinstance(s_i, numpy.integer):
            # this shape is a constant
            assert s_i >= 0
            return T.constant(s_i, dtype='int64')
        if type(s_i) in (tuple, list):
            # this dimension is the same as many of the inputs
            # which tells us that if one of the inputs is known,
            # the others all become known.
            # TODO: should be implemented in Elemwise, and Dot
            #
            # worst case, we loop over shape_of and replace things
            raise NotImplementedError(s_i)
        elif s_i.type.dtype[:3] in ('int', 'uint'):
            if getattr(s_i.type, 'ndim', 0):
                raise TypeError('Shape element must be scalar', s_i)
            return s_i
        else:
            raise TypeError('Unsupported shape element',
                    s_i, type(s_i), getattr(s_i, 'type', None))

    def set_shape(self, r, s):
        """Assign the shape `s` to previously un-shaped variable `r`.

        :type r: a variable
        :type s: None or a tuple of symbolic integers
        """
        assert r not in self.shape_of, 'r already in shape_of'
        if s is None:
            self.shape_of[r] = s
        else:
            if not isinstance(s, (tuple, list)):
                raise TypeError('shapes must be tuple/list', (r, s))
            if r.ndim != len(s):
                raise AssertionError(
                        "Something inferred a shape with %d dimensions "
                        "for a variable with %d dimensions." % (
                        len(s), r.ndim))

            shape_vars = [self.unpack(s_i) for s_i in s]
            self.shape_of[r] = tuple(shape_vars)
            for sv in shape_vars:
                self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def update_shape(self, r, other_r):
        '''Replace shape of r by shape of other_r.

        If, on some dimensions, the shape of other_r is not informative,
        keep the shape of r on those dimensions.
        '''
        # other_r should already have a shape
        assert other_r in self.shape_of, ('other_r not in shape_of', other_r)
        other_shape = self.shape_of[other_r]

        # If other_shape has no information, call is pointless.
        if other_shape is None:
            return

        if r in self.shape_of:
            r_shape = self.shape_of[r]
        else:
            # If no info is known on r's shape, use other_shape
            self.shape_of[r] = other_shape
            for sv in other_shape:
                self.shape_of_reverse_index.setdefault(sv, set()).add(r)
            return

        # Merge other_shape with r_shape, giving the priority to other_shape
        merged_shape = []
        for i, ps in enumerate(other_shape):
            # If other_shape[i] is uninformative, use r_shape[i].
            # For now, we consider 2 cases of uninformative other_shape[i]:
            #  - Shape_i(i)(other_r);
            #  - Shape_i(i)(r).
            if (ps.owner
                    and isinstance(getattr(ps.owner, 'op', None), Shape_i)
                    and ps.owner.op.i == i
                    and ps.owner.inputs[0] in (r, other_r)):
                merged_shape.append(r_shape[i])
            else:
                merged_shape.append(other_shape[i])
        self.shape_of[r] = tuple(merged_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def set_shape_i(self, r, i, s_i):
        '''Replace element i of shape_of[r] by s_i'''
        assert r in self.shape_of
        prev_shape = self.shape_of[r]
        # prev_shape is a tuple, so we cannot change it inplace,
        # so we build another one.
        new_shape = []
        for j, s_j in enumerate(prev_shape):
            if j == i:
                new_shape.append(self.unpack(s_i))
            else:
                new_shape.append(s_j)
        self.shape_of[r] = tuple(new_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def init_r(self, r):
        '''Register r's shape in the shape_of dictionary.'''
        if r not in self.shape_of:
            try:
                self.set_shape(r, self.shape_tuple(r))
            except AttributeError: #XXX: where would this come from?
                self.set_shape(r, None)

    def make_vector_shape(self, r):
        return make_vector(*self.shape_of[r])

    #
    # Feature interface
    #
    #
    def on_attach(self, env):
        assert not hasattr(env, 'shape_feature')
        env.shape_feature = self
        # Must be local to the object as otherwise we reuse the same
        # variable for multiple env!
        self.lscalar_one = T.constant(1, dtype='int64')
        assert self.lscalar_one.type == T.lscalar

        self.shape_of = {}
        # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)

        self.scheduled = {}
        # Variable ->

        self.shape_of_reverse_index = {}
        # shape var -> graph v

        for node in env.toposort():
            self.on_import(env, node)

    def on_import(self, env, node):
        if node.outputs[0] in self.shape_of:
            # this is a revert, not really an import
            for r in node.outputs + node.inputs:
                assert r in self.shape_of
            return

        for i, r in enumerate(node.inputs):
            # make sure we have shapes for the inputs
            self.init_r(r)

        try:
            shape_infer = node.op.infer_shape
        except AttributeError:
            shape_infer = self.default_infer_shape

        try:
            o_shapes = shape_infer(node,
                                   [self.shape_of[r] for r in node.inputs])
        except ShapeError:
            o_shapes = self.default_infer_shape(node, [self.shape_of[r] for
                                                       r in node.inputs])
        except NotImplementedError, e:
            raise NotImplementedError(
                    'Code called by infer_shape failed raising a '
                    'NotImplementedError. Raising NotImplementedError to '
                    'indicate that a shape cannot be computed is no longer '
                    'supported, and one should now use tensor.ShapeError '
                    'instead. The original exception message is: %s' % e)
        except Exception, e:
            msg = ('Failed to infer_shape from Op %s.\nInput shapes: '
                   '%s\nException encountered during infer_shape: '
                   '%s\nException message: %s\nTraceback: %s') % (
                node.op, [self.shape_of[r] for r in node.inputs],
                type(e), str(e), traceback.format_exc())
            if config.on_shape_error == "raise":
                raise Exception(msg)
            else:
                _logger.warning(msg)
            o_shapes = self.default_infer_shape(
                node, [self.shape_of[r] for r in node.inputs])

        # this is packed information
        # an element of o_shapes is either None or a tuple
        #   elements of the tuple can be either strings, or ints
        if len(o_shapes) != len(node.outputs):
            raise Exception('len(o_shapes) = '
                    + str(len(o_shapes))
                    + ' != len(node.outputs) = '
                    + str(len(node.outputs)))

        # Ensure shapes are in 'int64'. This is to make sure the assert
        # found in the `local_useless_subtensor` optimization does not fail.
        new_shape = []
        for sh_idx, sh in enumerate(o_shapes):
            if sh is None:
                continue
            for i, d in enumerate(sh):
                # Note: we ignore any shape element that is not typed (i.e. does
                # not have a 'dtype' attribute). This means there may still
                # remain int elements that are int32 on 32-bit platforms, but
                # this works with `local_useless_subtensor`, so for now we
                # keep it this way. See #266 for a better long-term fix.
                if getattr(d, 'dtype', 'int64') != 'int64':
                    assert d.dtype in theano.tensor.int_dtypes
                    new_shape += sh[len(new_shape):i + 1]
                    new_shape[i] = theano.tensor.cast(d, 'int64')
            if new_shape:
                # We replace the shape with wrong dtype by the one with 'int64'.
                new_shape += sh[len(new_shape):]
                o_shapes[sh_idx] = tuple(new_shape)
                new_shape = []

        for r, s in izip(node.outputs, o_shapes):
            self.set_shape(r, s)

    def on_change_input(self, env, node, i, r, new_r):
        if new_r not in self.shape_of:
            # It happen that the env didn't called on_import for some
            # new_r.  This happen when new_r don't have an
            # owner(i.e. it is a constant or an input of the graph)
            # update_shape suppose that r and new_r are in shape_of.
            self.init_r(new_r)

        # This tells us that r and new_r must have the same shape if
        # we didn't know that the shapes are related, now we do.
        self.update_shape(new_r, r)

        # change_input happens in two cases:
        # 1) we are trying to get rid of r, or
        # 2) we are putting things back after a failed transaction.

        # In case 1, if r has a shape_i client, we will want to
        # replace the shape_i of r with the shape of new_r.  Say that
        # r is *scheduled*.
        # At that point, node is no longer a client of r, but of new_r
        for (shpnode, idx) in (r.clients + [(node, i)]):
            if isinstance(getattr(shpnode, 'op', None), Shape_i):
                self.scheduled[shpnode] = new_r
        # In case 2, if r is a variable that we've scheduled for shape update, then we
        # should cancel it.
        unscheduled = [k for k, v in self.scheduled.items() if v == r]
        for k in unscheduled:
            del self.scheduled[k]

        # In either case, r could be in shape_of.values(), that is, r itself
        # is the shape of  something. In that case, we want to update
        # the value in shape_of, to keep it up-to-date.
        for v in self.shape_of_reverse_index.get(r, []):
            # The reverse index is only approximate. It is not updated on
            # deletion of variables, or on change_input so it might be the
            # case that there are a few extra `v`'s in it that no longer have
            # a shape of r or possibly have been deleted from shape_of
            # entirely. The important thing is that it permits to recall
            # all variables with r in their shape.
            for ii, svi in enumerate(self.shape_of.get(v, [])):
                if svi == r:
                    self.set_shape_i(v, ii, new_r)
        self.shape_of_reverse_index[r] = set()


class ShapeOptimizer(Optimizer):
    """Optimizer that serves to add ShapeFeature as an env feature.
    """
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, env):
        env.extend(ShapeFeature())

    def apply(self, env):
        pass

# -1 should make it run right before the first merge
theano.compile.mode.optdb.register('ShapeOpt', ShapeOptimizer(),
                                   -1, 'fast_run', 'fast_compile')


@register_specialize
@register_stabilize
@register_canonicalize
@gof.local_optimizer([T.fill])
def local_fill_to_alloc(node):
    """fill(s,v) -> alloc(v, shape(s))

    This is an important optimization because with the shape_to_shape_i
    optimization, the dependency on 's' is often removed.

    """
    if node.op == T.fill:
        r, v = node.inputs
        if v.type == node.outputs[0].type:
            # this is a useless fill, erase it.
            rval = [v]
        elif v.type.broadcastable == node.outputs[0].type.broadcastable:
            # this is a cast
            rval = [T.cast(v, node.outputs[0].type.dtype)]
        elif r.type.broadcastable == node.outputs[0].type.broadcastable:
            # we are broadcasting v somehow, but not r
            rval = [broadcast_like(v, r, node.env, dtype=v.dtype)]
        else:
            # we are broadcasting both v and r,
            # the output shape must be computed
            #
            # TODO: implement this case (including a test!)
            #
            #  I think the strategy should be to extend the shorter
            #  shape vector with 1s (how?) and then take the
            #  elementwise max of the two.  - how to flag an error of
            #  shape mismatch where broadcasting should be illegal?
            return
            # TODO: cut out un-necessary dimshuffles of v

        assert rval[0].type == node.outputs[0].type, ('rval', rval[0].type,
                'orig', node.outputs[0].type,
                'node', node,
                )  # theano.printing.debugprint(node.outputs[0], file='str'))
        return rval


@register_specialize
@register_canonicalize
@gof.local_optimizer([T.alloc])
def local_useless_alloc(node):
    """
    If the input type is the same as the output type (dtype and broadcast)
    there is no change in the shape of the input. So this is just a simple copy
    of the input. This is not needed.
    """
    if node.op == T.alloc:
        if node.inputs[0].type == node.outputs[0].type:
            return [node.inputs[0]]


@register_specialize
@register_canonicalize
@gof.local_optimizer([T._shape])
def local_shape_to_shape_i(node):
    if node.op == T._shape:
        # This optimization needs ShapeOpt and env.shape_feature
        if not hasattr(node.env, 'shape_feature'):
            return
        shape_feature = node.env.shape_feature
        return [shape_feature.make_vector_shape(node.inputs[0])]


@register_specialize
@register_canonicalize
@gof.local_optimizer([T._shape])
def local_track_shape_i(node):
    try:
        shape_feature = node.env.shape_feature
    except AttributeError:
        return
    if node in shape_feature.scheduled:
        assert isinstance(node.op, Shape_i)
        replacement = shape_feature.scheduled[node]
        return [shape_feature.shape_of[replacement][node.op.i]]


@register_specialize
@register_canonicalize
@gof.local_optimizer([T.Subtensor])
def local_subtensor_make_vector(node):
    # replace all subtensor(make_vector) like:
    # [a,b,c][0] -> a
    # [a,b,c][0:2] -> [a,b]
    # we can do this for constant indexes
    if isinstance(node.op, T.Subtensor):
        # This optimization needs ShapeOpt and env.shape_feature
        x = node.inputs[0]
        if x.owner and x.owner.op == make_vector:
            try:
                idx, = node.op.idx_list
            except Exception:
                #'how can you have multiple indexes into a shape?'
                raise

            if isinstance(idx, (scalar.Scalar, T.TensorType)):
                # The idx is a Scalar, ie a Type. This means the actual index
                # is contained in node.inputs[1]
                old_idx, idx = idx, node.inputs[1]
                assert idx.type == old_idx

            if isinstance(idx, (int, numpy.integer)):
                return [x.owner.inputs[idx]]
            elif isinstance(idx, Variable):
                # if it is a constant we can do something with it
                try:
                    v = get_constant_value(idx)
                    return [x.owner.inputs[v]]
                except Exception:
                    pass
            else:
                # it is a slice of ints and/or Variables
                #TODO: check subtensor to see if it can contain
                #      constant variables, and if it can, then try to
                #      unpack them.
                try:
                    return [make_vector(*x.owner.inputs.__getitem__(idx))]
                except TypeError:
                    pass
                except Exception:
                    _logger.error('failed to index with "%s"' % str(idx))
                    raise

#TODO: the other optimization for and, or, xor, le and ge see ticket #496.


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_useless_elemwise(node):
    """

    eq(x,x) -> 1
    neq(x,x) -> 0
    mul(x) -> x
    add(x) -> x
    identity(x) -> x

    """
    if isinstance(node.op, T.Elemwise):
        if node.op.scalar_op == theano.scalar.eq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
            #it is the same var in the graph. That will always be true
                return [T.fill(node.inputs[0],
                               T.constant(1.0,
                                          dtype=node.outputs[0].type.dtype))]
        if node.op.scalar_op == theano.scalar.neq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
            #it is the same var in the graph. That will always be false
                return [T.fill(node.inputs[0],
                               T.constant(0.0,
                                          dtype=node.outputs[0].type.dtype))]
        if node.op.scalar_op == theano.scalar.mul and len(node.inputs) == 1:
            return [node.inputs[0]]
        if node.op.scalar_op == theano.scalar.add and len(node.inputs) == 1:
            return [node.inputs[0]]
        if (node.op.scalar_op == theano.scalar.identity
            and len(node.inputs) == 1):
            return [node.inputs[0]]


@register_specialize
@gof.local_optimizer([T.Elemwise])
def local_alloc_unary(node):
    """unary(alloc(x, shp)) -> alloc(unary(x), shp)
    """
    if isinstance(node.op, T.Elemwise) and len(node.inputs) == 1:
        a = node.inputs[0]
        if a.owner and isinstance(a.owner.op, T.Alloc):
            x = a.owner.inputs[0]
            shp = a.owner.inputs[1:]
            v = node.op(x)
            return [T.alloc(T.cast(v, node.outputs[0].dtype), *shp)]


class Assert(T.Op):
    """
    Implements assertion in a computational graph.
    Notes:
    This Op can be removed from the graph because of optimizations, and can hide
    some possible optimizations to the optimizer.
    Also, the output of the Op must be returned by the function computing the
    graph, otherwise it will not be used.
    """
    view_map = {0: [0]}

    def make_node(self, value, *conds):
        cond = [T.as_tensor_variable(c) for c in conds]
        assert numpy.all([c.type.ndim == 0 for c in cond])
        return gof.Apply(self, [value] + cond, [value.type()])

    def __str__(self):
        return self.__class__.__name__

    def perform(self, node, inputs, out_):
        out, = out_
        v = inputs[0]
        out[0] = v
        assert numpy.all(inputs[1:])

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def grad(self, input, output_gradients):
        return output_gradients

    def c_code(self, node, name, inames, onames, sub):
        value = inames[0]
        out = onames[0]
        check = []
        fail = sub['fail']
        for idx in xrange(len(inames) - 1):
            i = inames[idx + 1]
            dtype = node.inputs[idx + 1].dtype
            check.append('if(!((npy_%(dtype)s*)PyArray_DATA(%(i)s))[0])'
                         '{PyErr_SetString(PyExc_AssertionError,"Theano'
                         ' Assert failed!");%(fail)s}' % locals())
        check = "\n".join(check)
        return """
        %(check)s
        %(out)s = %(value)s;
        Py_INCREF(%(value)s);
        """ % locals()

    def c_code_cache_version(self):
        return (0, 1)

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

assert_ = Assert()


@register_specialize
@gof.local_optimizer([Assert])
def local_remove_useless_assert(node):
    if isinstance(node.op, Assert):
        cond = []
        for c in node.inputs[1:]:
            try:
                const = get_constant_value(c)

                if 0 != const.ndim or const == 0:
                    #Should we raise an error here? How to be sure it
                    #is not catched?
                    cond.append(c)
            except TypeError:
                cond.append(c)

        if len(cond) == 0:
            return [node.inputs[0]]
        if len(cond) != len(node.inputs) - 1:
            return [assert_(node.inputs[0], *cond)]


@gof.local_optimizer([T.Alloc])
def local_alloc_elemwise(node):
    """
    elemwise(alloc(x, shp), ..., y.TensorType(BROADCAST CONDITION))
      -> elemwise(x, y.TensorType(no broadcast flag))

    elemwise(dimshuffle(alloc(x, shp)),... ,y.TensorType(BROADCAST CONDITION))
      -> elemwise(x, y.TensorType(no broadcast flag))

    BROADCAST CONDITION: the condition is that the one input that are
    not to be optimized to have the same braodcast pattern as the
    output

         We can change the alloc by a dimshuffle as the elemwise
         already have the shape info.  The dimshuffle will be faster
         to exec
    """
    if not isinstance(node.op, T.Elemwise):
        return False
    if len(node.outputs) > 1:
        #This is a supposition this code make that I'm not sure is always true.
        assert all([list(o.type.broadcastable) == list(
                    node.outputs[0].type.broadcastable) for o in
                    node.outputs[1:]])

    if not any([list(i.type.broadcastable) == list(
                node.outputs[0].type.broadcastable) for i in node.inputs]):
        return False
    if not any([i.owner and (isinstance(i.owner.op, T.Alloc) or \
                             (isinstance(i.owner.op, T.DimShuffle) and
                              i.owner.inputs[0].owner and \
                              isinstance(i.owner.inputs[0].owner.op, T.Alloc)))
                for i in node.inputs]):
        return False
    no_broad_idx = -1
    for idx, i in enumerate(node.inputs):
        if not i.owner:
            if list(i.type.broadcastable) == [False, ] * i.type.ndim:
                no_broad_idx = idx
                break
            else:
                continue
        if not any(i.type.broadcastable) and not isinstance(i.owner.op,
                                                            T.Alloc):
            no_broad_idx = idx
            break
        elif list(i.type.broadcastable) == list(
            node.outputs[0].type.broadcastable) \
            and not isinstance(i.owner.op, T.Alloc) \
            and not (isinstance(i.owner.op, T.DimShuffle) and
                     i.owner.inputs[0].owner and \
                         isinstance(i.owner.inputs[0].owner.op, T.Alloc)):
            no_broad_idx = idx
            break

    assert no_broad_idx >= 0
    assert_op = node.inputs[no_broad_idx]
    cmp_op = assert_op
    new = []

    for i in node.inputs:
        if (i.owner and isinstance(i.owner.op, T.Alloc)
            and i.owner.inputs[0].type != i.owner.outputs[0].type):
            # when i.owner.inputs[0].type == i.owner.outputs[0].type we
            # will remove that alloc later

            assert i.type.ndim == cmp_op.ndim
            if theano.config.experimental.local_alloc_elemwise_assert:
                assert_op = assert_(assert_op,
                                    *[T.eq(i.shape[idx], cmp_op.shape[idx])\
                                          for idx in xrange(i.type.ndim) \
                                          if not i.type.broadcastable[idx]])
                new.append(i.owner.inputs[0])
        elif i.owner and isinstance(i.owner.op, T.DimShuffle) \
                and i.owner.inputs[0].owner \
                and isinstance(i.owner.inputs[0].owner.op, T.Alloc):
            assert i.type.ndim == cmp_op.type.ndim
            if theano.config.experimental.local_alloc_elemwise_assert:
                assert_op = assert_(assert_op,
                                    *[T.eq(i.shape[idx], cmp_op.shape[idx])
                                      for idx in xrange(i.type.ndim)
                                      if not i.type.broadcastable[idx]])
            new.append(i.owner.inputs[0].owner.inputs[0])
        else:
            new.append(i)
    new[no_broad_idx] = assert_op
    if theano.config.experimental.local_alloc_elemwise_assert:
        assert assert_op.owner.op is assert_
    return [node.op(*new)]

#TODO, global optimizer that lift the assert to the beginning of the graph.
#TODO, when all inputs can be optimized do all except one

theano.configparser.AddConfigVar('experimental.local_alloc_elemwise',
        "If True enable the experimental optimization local_alloc_elemwise",
        theano.configparser.BoolParam(False),
        in_c_key=False)
#This version if faster but not as save.
theano.configparser.AddConfigVar('experimental.local_alloc_elemwise_assert',
        "If False enable the experimental optimization local_alloc_elemwise"
                                 " but WITHOUT assert into the graph!",
        theano.configparser.BoolParam(True),
        in_c_key=False)
if theano.config.experimental.local_alloc_elemwise:
    #enabled by default when the lifter of assert is done.
    register_specialize(local_alloc_elemwise)
else:
    #don't register them in fast_run by default to have them disabled
    #by default disable them by default as we are not sure it is
    #always a good idea to replace an alloc with multiple op.
    compile.optdb['specialize'].register("local_alloc_elemwise",
                                         local_alloc_elemwise)

############################
# Constant Canonicalization
############################


@register_canonicalize
@gof.local_optimizer([])
def local_upcast_elemwise_constant_inputs(node):
    """This explicitly upcasts constant inputs to elemwise Ops, when
    those Ops do implicit upcasting anyway.

    Rationale: it helps merge things like (1-x) and (1.0 - x).
    """
    if len(node.outputs) > 1:
        return
    try:
        shape_i = node.env.shape_feature.shape_i
    except AttributeError:
        shape_i = None
    if isinstance(node.op, T.Elemwise):
        scalar_op = node.op.scalar_op
        #print "aa", scalar_op.output_types_preference
        if (getattr(scalar_op, 'output_types_preference', None)
            in (T.scal.upgrade_to_float, T.scal.upcast_out)):
            # this is the kind of op that we can screw with the input
            # dtypes by upcasting explicitly
            output_dtype = node.outputs[0].type.dtype
            new_inputs = []
            for i in node.inputs:
                if i.type.dtype == output_dtype:
                    new_inputs.append(i)
                else:
                    try:
                        # works only for scalars
                        cval_i = get_constant_value(i)
                        if all(i.broadcastable):
                            new_inputs.append(T.cast(cval_i, output_dtype))
                        else:
                            if shape_i is None:
                                return
                            new_inputs.append(T.alloc(T.cast(cval_i,
                                                             output_dtype),
                                *[shape_i(d)(i) for d in xrange(i.ndim)]))
                            #print >> sys.stderr, "AAA",
                            #*[Shape_i(d)(i) for d in xrange(i.ndim)]
                    except TypeError:
                        #for the case of a non-scalar
                        if isinstance(i, T.TensorConstant):
                            new_inputs.append(T.cast(i, output_dtype))
                        else:
                            new_inputs.append(i)

            if new_inputs != node.inputs:
                rval = [node.op(*new_inputs)]
                if rval[0].type != node.outputs[0].type:
                    print >> sys.stderr, "NODE:", node
                    print >> sys.stderr, "NODE INPUT TYPES:", [i.type for i
                                                               in node.inputs]
                    print >> sys.stderr, "NODE OUTPUT TYPES:", [
                        o.type for o in node.outputs]
                    print >> sys.stderr, "RVAL:", rval
                    print >> sys.stderr, "NEW INPUT TYPES:", [i.type for i
                                                              in new_inputs]
                    print >> sys.stderr, "RVAL INPUT TYPES:", [
                        i.type for i in rval[0].owner.inputs]
                    print >> sys.stderr, "RVAL TYPES:", [o.type for o in rval]
                assert rval[0].type == node.outputs[0].type, (node, rval[0])
                return rval

##################
# Subtensor opts #
##################


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Subtensor])
def local_useless_subtensor(node):
    """
    Remove Subtensor if it takes the full input
    """
    if isinstance(node.op, T.Subtensor):
        # This optimization needs ShapeOpt and env.shape_feature
        if not hasattr(node.env, 'shape_feature'):
            return
        shape_of = node.env.shape_feature.shape_of
        node_input_idx = 1
        for pos, idx in enumerate(node.op.idx_list):
            if not isinstance(idx, slice):
                # If idx is not a slice, this means we remove this dimension
                # from the output, so the subtensor is not useless
                return False
            if idx.start not in [0, None]:
                # If the start of the slice is different from 0, or is a
                # variable, then we assume the subtensor is not useless
                return False
            if idx.step not in [1, None]:
                # If we are going backwards, or skipping elements, then this
                # is not a useless subtensor
                return False

            length_pos_data = sys.maxint

            length_pos = shape_of[node.inputs[0]][pos]
            try:
                length_pos_data = get_constant_value(length_pos)
            except TypeError:
                pass

            if isinstance(idx.stop, int):
                if idx.stop < length_pos_data:
                    return False
            elif isinstance(idx.stop, theano.scalar.Scalar):
                length_pos_shape_i = node.inputs[node_input_idx]
                # length_pos is a tensor variable, but length_pos_shape_i
                # is a scalar variable. We try to see if they represent
                # the same underlying variable.
                if (length_pos_shape_i.owner and
                        isinstance(length_pos_shape_i.owner.op,
                            T.ScalarFromTensor)):
                    length_pos_shape_i = length_pos_shape_i.owner.inputs[0]
                elif (length_pos.owner and
                        isinstance(length_pos.owner.op,
                            T.TensorFromScalar)):
                    length_pos = length_pos.owner.inputs[0]
                else:
                    # We did not find underlying variables of the same type
                    return False

                # The type can be different: int32 vs int64. length_pos
                # should always be int64 as that is what the shape
                # tracker keep. Subtensor accept any scalar int{8,16,32,64}
                # as index type.
                assert str(length_pos.type.dtype) == "int64"
                assert str(length_pos_shape_i.type.dtype) in ["int8", "int16",
                                                              "int32", "int64"]
                # We already know that start and step are not variables
                # and so they don't appear in the input of the node
                node_input_idx += 1

                # length_pos_shape_i cannot be None
                if length_pos_shape_i != length_pos:
                    return False
            elif idx.stop is None:
                pass
            else:
                return False

        return [node.inputs[0]]


@register_canonicalize
@gof.local_optimizer([])
def local_subtensor_lift(node):
    """
    unary(x)[idx] -> unary(x[idx])#any broadcast pattern.

    Handles the following unary ops:
    elemwise(x,...)[idx] -> elemwise(x[idx],...)
      when x,... are broadcasted scalar or not broadcasted at all
    rebroadcast(x)[idx] => rebroadcast(x[idx])
    """
    if isinstance(node.op, T.Subtensor):
        u = node.inputs[0]
        if not u.owner or len(u.clients) > 1:
            return False

        if isinstance(u.owner.op, T.Elemwise) and len(u.owner.inputs) == 1:
            idx = node.inputs[1:]
            x_idx = node.op(u.owner.inputs[0], *idx)
            return [u.owner.op(x_idx)]

        if isinstance(u.owner.op, T.Elemwise):
            new_inputs = []
            if all([sum(i.type.broadcastable) == 0 for i in u.owner.inputs]):
                # There is no broadcastable in the inputs
                idx = node.inputs[1:]
                new_inputs = [node.op(i, *idx) for i in u.owner.inputs]
                return [u.owner.op(*new_inputs)]
            elif all([sum(i.type.broadcastable) in [i.ndim, 0]
                      for i in u.owner.inputs]):
                # There is no broadcastable in the inputs or it is scalar
                idx = node.inputs[1:]
                new_inputs = []
                for i in u.owner.inputs:
                    if sum(i.type.broadcastable) == 0:
                        new_inputs.append(node.op(i, *idx))
                    else:
                        # If the subtensor remove some dims, we must
                        # lower the number of dimensions of this scalar.
                        if node.outputs[0].ndim == i.ndim:
                            new_inputs.append(i)
                        else:
                            new_inputs.append(
                                i.dimshuffle(['x'] * node.outputs[0].ndim))
                return [u.owner.op(*new_inputs)]

        if isinstance(u.owner.op, T.Rebroadcast):
            # make sure that Subtensor and Rebroadcast only have 1 input/output
            assert len(node.inputs) == 1
            assert len(u.owner.inputs) == 1

            # Subtensor might reduce dim., adapt broadcast pattern accordingly
            new_axis = []

            # loop through indices being subtensor-ed
            # i indexes broadcastable pattern before subtensor
            # j indexes broadcastable pattern after subtensor
            j = 0
            for (i, x) in enumerate(node.op.idx_list):
                # if its not a slice, it will reduce the dimension, should
                # not appear in the broascastable dimensions
                if isinstance(x, slice):
                    new_axis += [(j, u.broadcastable[i])]
                    j += 1
            # now keep the broadcastable pattern of all
            # items not appearing in subtensor list
            for i in xrange(len(node.op.idx_list), len(u.broadcastable)):
                new_axis += [(j, u.broadcastable[i])]
                j += 1

            subt_x = T.Subtensor(node.op.idx_list)(u.owner.inputs[0])
            rbcast_subt_x = T.Rebroadcast(*new_axis)(subt_x)

            return [rbcast_subt_x]


def merge_two_slices(slice1, len1, slice2, len2):
    '''
     This function merges two slices into a single slice. The code works on
     the assumption that:
          a) slice1 is actually a slice and not an index, while slice2
          can be just an index.
          b) the two slices **have been applied consecutively** on the same
          tensor

    The output slice is **not** in canonical form, but actually just a slice
    that can be applied to a tensor to produce the same output as applying
    the two consecutive slices.
    ``len1`` is the length of the tensor **before** applying the first slice,
    while ``len2`` is the length **after** applying the first slice.
    '''
    list_opt = [local_abs_merge, local_mul_switch_sink,
                local_upcast_elemwise_constant_inputs,
                local_remove_switch_const_cond, constant_folding]

    if type(slice1) is not slice:
        raise ValueError(('First provided slice should actually be of type'
                         'slice and not an index !'), slice1)
    sl1, reverse1 = T.get_canonical_form_slice(slice1, len1)
    sl2, reverse2 = T.get_canonical_form_slice(slice2, len2)

    if type(sl2) is not slice:
        if reverse1 is None:
            # The first slice is not in reverse, which makes things a lot
            # more clear.
            # In this case we need to take care only of the special cases:
            # len2 <=0    -> throw index error regardless of sl2
            # sl2 > len2  -> throw index error
            # sl2 < -len2 -> throw index error
            # To get a index error we simply use len1+1 to indicate we are
            # out of bounds, because passing this index through the formula
            # of getting the mixed slice is not guaranteed to result in an
            # index error. The **issue though** if that the error will
            # complain about accessing element len1+1 which is probably not
            # too intuitive for the user
            val = sl1.start + sl2 * sl1.step
            val = T.switch(T.le(len2, 0), len1 + 1, val)
            val = T.switch(T.ge(sl2, len2), len1 + 1, val)
            val = T.switch(T.lt(sl2, 0), - len1 - 1, val)
            if sl1.step:
                val = T.switch(T.eq(sl1.step, 0), len1 + 1, val)
            return val
        else:
            # We are in the more complex case when we do not actually know
            # if the first slice was in reverse or not.
            # in case it was not in reverse:
            p_val = sl1.start + sl2 * sl1.step
            # case it was in reverse we need to realize that we do not want
            # the k-th element from sl.start but the k-th element from
            # sl.stop backwards
            n_val = sl1.stop - 1 - sl2 * sl1.step
            if config.warn.subtensor_merge_bug:
                _logger.warn((
                    'Your current code is fine, but Theano versions '
                    'prior to 0.5rc2 might have given an incorrect result. '
                    'To disable this warning, set the Theano flag '
                    'warn.subtensor_merge_bug to False.'))
            # we need to pick either n_val or p_val and then follow same
            # steps as above for covering the index error cases
            val = T.switch(T.lt(reverse1, 0), n_val, p_val)
            val = T.switch(T.le(len2, 0), len1 + 1, val)
            val = T.switch(T.ge(sl2, len2), len1 + 1, val)
            val = T.switch(T.lt(sl2, 0), - len1 - 1, val)
            if sl1.step:
                val = T.switch(T.eq(sl1.step, 0), len1 + 1, val)
            return val
    else:
        # We are deleaing with two slices that need to be put together
        # according to the two steps we have 4 different combinations of
        # positive/negative. I will denote the case I'm looking at by
        # suffixes to the variables (nn,np,pn,pp):
        flen = sl2.stop - sl2.start
        p_step = sl1.step * sl2.step
        n_step = sl1.step * sl2.step * -1

        pp_start = T.minimum(sl1.start + sl2.start * sl1.step, sl1.stop)
        pp_stop = T.minimum(sl1.start + sl2.stop * sl1.step, sl1.stop)

        pn_stop = sl1.start + (sl2.start - 1) * sl1.step
        pn_stop = T.switch(T.and_(T.lt(pn_stop, 0),
                                  T.gt(flen, 0)),
                            -len1 - 1,
                            T.minimum(pn_stop, sl1.stop))
        pn_start = sl1.start + (sl2.stop - 1) * sl1.step
        pn_start = T.minimum(pn_start, sl1.stop)
        pn_start = T.maximum(pn_start, 0)

        np_stop = sl1.stop - sl2.stop * sl1.step - 1
        np_stop = T.switch(T.and_(T.lt(np_stop, 0),
                                  T.gt(flen, 0)),
                           -len1 - 1,
                           T.maximum(sl1.start - 1, np_stop))
        np_start = T.maximum(sl1.start, sl1.stop - sl2.start * sl1.step - 1)

        nn_start = T.maximum(sl1.start,
                             (sl1.stop - 1) - (sl2.stop - 1) * sl1.step)
        nn_stop = T.maximum(sl1.start, sl1.stop - sl2.start * sl1.step)

        start = T.switch(T.lt(reverse2 * reverse1, 0),
                         T.switch(T.lt(reverse1, 0), np_start, pn_start),
                         T.switch(T.lt(reverse1, 0), nn_start,
                                  pp_start))

        stop = T.switch(T.lt(reverse2 * reverse1, 0),
                         T.switch(T.lt(reverse1, 0), np_stop, pn_stop),
                         T.switch(T.lt(reverse1, 0), nn_stop, pp_stop
                                 ))

        step = T.switch(T.lt(reverse2 * reverse1, 0), n_step, p_step)
        start = T.switch(T.le(flen, 0), 0, start)
        stop = T.switch(T.le(flen, 0), 0, stop)

        # The canonical form of the slice is pretty complicated
        # and is not simplified. We simplify it in advance here
        # as otherwise this create too many useless optimization that
        # DebugMode must check.
        start = pre_greedy_local_optimizer(list_opt, start)
        stop = pre_greedy_local_optimizer(list_opt, stop)
        step = pre_greedy_local_optimizer(list_opt, step)
        start = pre_greedy_local_optimizer(list_opt, start)
        stop = pre_greedy_local_optimizer(list_opt, stop)
        step = pre_greedy_local_optimizer(list_opt, step)

        #Pre merge constant for the same reason.
        start, stop, step = pre_constant_merge([start, stop, step])

        return slice(start, stop, step)


@register_canonicalize
@register_specialize
@gof.local_optimizer([])
def local_subtensor_merge(node):
    """
    Refactored optimization to deal with all cases of tensor merging.
    Given a subgraph of the form Subtensor(Subtensor(u)), the optimization
    expresses all slices in a canonical form, and then merges them together.
    """

    if isinstance(node.op, T.Subtensor):
        u = node.inputs[0]
        if u.owner and isinstance(u.owner.op, T.Subtensor):
            # We can merge :)
            # x actual tensor on which we are picking slices
            x = u.owner.inputs[0]
            # slices of the first applied subtensor
            slices1 = T.get_idx_list(u.owner.inputs, u.owner.op.idx_list)
            slices2 = T.get_idx_list(node.inputs, node.op.idx_list)
            # Get the shapes of the vectors !
            try:
                # try not to introduce new shape into the graph
                xshape = node.env.shape_feature.shape_of[x]
                ushape = node.env.shape_feature.shape_of[u]
            except AttributeError:
                # Following the suggested use of shape_feature which should
                # consider the case when the compilation mode doesn't
                # include the ShapeFeature
                xshape = x.shape
                ushape = u.shape

            merged_slices = []
            pos_2 = 0
            for pos_1, slice1 in enumerate(slices1):
                if type(slice1) is slice:
                    merged_slices.append(
                        merge_two_slices(slice1,
                                         xshape[pos_1],
                                         slices2[pos_2],
                                         ushape[pos_2]))
                    pos_2 += 1
                else:
                    merged_slices.append(slice1)

            merged_slices += slices2[pos_2:]
            subtens = T.Subtensor(merged_slices)
            sl_ins = T.Subtensor.collapse(
                merged_slices,
                lambda x: isinstance(x, T.Variable))
            out = subtens.make_node(x, *sl_ins).outputs[0]

            return [out]


@register_canonicalize
@register_specialize
@gof.local_optimizer([])
def local_subtensor_of_alloc(node):
    """alloc[x:y] -> alloc"""
    if not isinstance(node.op, T.Subtensor):
        return False
    u = node.inputs[0]
    if u.owner is None:
        return False
    if not isinstance(u.owner.op, T.Alloc):
        return False
    slices = T.get_idx_list(node.inputs, node.op.idx_list)
    val = u.owner.inputs[0]
    dims = u.owner.inputs[1:]
    assert len(slices) <= len(dims)

    # Number of dimensions added to val
    n_added_dims = u.ndim - val.ndim
    # Dimensions of the returned alloc
    nw_dims = []
    # Slices to take from val
    val_slices = []

    for i, (sl, dim) in enumerate(zip(slices, dims)):
        # If val was not copied over that dim,
        # we need to take the appropriate subtensor on it.
        if i >= n_added_dims:
            # We check that the corresponding val dimensions was
            # not a broadcasted dimensions.
            if (val.type.ndim > (i - n_added_dims) and
                val.type.broadcastable[i - n_added_dims]):
                val_slices.append(slice(None))
            else:
                val_slices.append(sl)

        csl, _ = T.get_canonical_form_slice(sl, dim)
        if type(csl) is not slice:
            # That dimension is removed.
            pass
        else:
            nw_dims += [T.ceil_intdiv((csl.stop - csl.start), csl.step)]

    nw_val = val[tuple(val_slices)]
    nw_dims += dims[len(slices):]
    rval = T.alloc(nw_val, *nw_dims)
    if type(rval) not in (list, tuple):
        rval = [rval]

    return rval


@register_canonicalize
@gof.local_optimizer([None])
def local_IncSubtensor_serialize(node):
    """
    When using Subtensor, gradient graphs can be ugly.

    If we ask for grad(f(a[0]), a), we are going to get something like

        IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])

    This might be ugly, but at least it's as fast as you could want.
    If we ask for grad(f(a[0], a[1], a[2]), a), it's much worse...

        Elemwise{Add}
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[1])), [1])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[2])), [2])

    This is much worse because this time we have to produce 3 matrices
    the size of 'a', just so we can add them together.

    This Op rearranges IncSubtensor's that all work on the same
    initial argument (here, Elemwise{second}(a,0)) into a chain.  The
    advantage of the chain structure is that each one can be optimized
    later in the pipeline to operate inplace.

    Ideally, the op will do something like this:

    #
    #  add(x, incsubtensor(b, c), incsubtensor(b, d))
    #  -> incsubtensor(incsubtensor(add(x,b,b), c), d)

    """
    def movable(i):
        # Return True iff this is a incsubtensor that we can move
        return i.owner \
                and isinstance(i.owner.op, T.IncSubtensor) \
                and i.type == o_type \
                and len(i.clients) == 1 \
                and not i.owner.op.set_instead_of_inc

    if node.op == T.add:
        o_type = node.outputs[0].type

        movable_inputs = [i for i in node.inputs if movable(i)]

        if movable_inputs:
            new_inputs = [i for i in node.inputs if not movable(i)] \
                    + [mi.owner.inputs[0] for mi in movable_inputs]
            new_add = T.add(*new_inputs)

            # stack up the new incsubtensors
            tip = new_add
            for mi in movable_inputs:
                assert tip.type == o_type
                assert tip.type == mi.owner.inputs[0].type
                tip = mi.owner.op(tip, *mi.owner.inputs[1:])
            return [tip]

        #print incsub_inputs, [id(i.owner.inputs[0]) for i in incsub_inputs]


#after priority 50 Destructive inplace operations
#gemm is the first one now, at priority 70

@gof.local_optimizer([None])
def local_inplace_setsubtensor(node):
    """
    Also work for GpuIncSubtensor
    """
    if isinstance(node.op, T.IncSubtensor) and not node.op.inplace:
        new_op = node.op.__class__(
       node.op.idx_list, inplace=True,
       set_instead_of_inc=node.op.set_instead_of_inc,
       destroyhandler_tolerate_aliased=node.op.destroyhandler_tolerate_aliased)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('inplace_setsubtensor',
                       TopoOptimizer(local_inplace_setsubtensor,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')  # DEBUG


@gof.local_optimizer([None])
def local_inplace_incsubtensor1(node):
    """ also work for GpuAdvancedIncSubtensor1 """
    if isinstance(node.op, T.AdvancedIncSubtensor1) and not node.op.inplace:
        new_op = node.op.__class__(
            inplace=True, set_instead_of_inc=node.op.set_instead_of_inc)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_incsubtensor1',
                       TopoOptimizer(
        local_inplace_incsubtensor1,
        failure_callback=TopoOptimizer.warn_inplace),
                       60, 'fast_run', 'inplace')  # DEBUG

@gof.local_optimizer([None])
def local_inplace_remove0(node):
    """
    Optimization to insert inplace versions of Remove0.
    """
    global Remove0
    if Remove0 is None:
        from theano.sparse.sandbox.sp import Remove0
    if isinstance(node.op, Remove0) and not node.op.inplace:
        new_op = node.op.__class__(inplace=True)
        new_node = new_op(*node.inputs)
        return [new_node]
    return False
compile.optdb.register('local_inplace_remove0',
                       TopoOptimizer(local_inplace_remove0,
    failure_callback=TopoOptimizer.warn_inplace), 60,
                       'fast_run', 'inplace')


@register_canonicalize
@register_stabilize
@gof.local_optimizer([None])
def local_incsubtensor_of_allocs(node):
    """
    IncSubtensor(x, zeros, idx) -> x
    """
    if isinstance(node.op, T.IncSubtensor) and not node.op.set_instead_of_inc:
        x = node.inputs[0]
        y = node.inputs[1]
        replace = False
        try:
            if get_constant_value(y) == 0:
                replace = True
        except TypeError:
            pass

        if replace:
            return [x]
        else:
            return False


@register_canonicalize
@register_stabilize
@gof.local_optimizer([None])
def local_setsubtensor_of_allocs(node):
    """
    SetSubtensor(x, x[idx], idx) -> x

    when x is constant or alloc.
    """
    if isinstance(node.op, T.IncSubtensor) and node.op.set_instead_of_inc:
        x = node.inputs[0]
        y = node.inputs[1]
        replace_x = None
        replace_y = None

        try:
            replace_x = get_constant_value(x)
        except TypeError:
            pass

        try:
            replace_y = get_constant_value(y)
        except TypeError:
            pass

        if (replace_x == replace_y and
            replace_x is not None and
            replace_y is not None):
            return [x]
        else:
            return False


####################
# Rebroadcast opts #
####################

@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Rebroadcast])
def local_useless_rebroadcast(node):
    """
    Remove Rebroadcast if id does not actually change the broadcasting pattern
    """
    if isinstance(node.op, T.Rebroadcast):
        x = node.inputs[0]
        if numpy.all(x.broadcastable == node.outputs[0].broadcastable):
            # No broadcastable flag was modified
            return [x]
        else:
            # Keep the flags that modify something
            new_axis = {}
            for dim, bc in node.op.axis.items():
                if x.broadcastable[dim] != bc:
                    new_axis[dim] = bc
            if new_axis == node.op.axis:
                # All flags are useful
                return
            else:
                return [T.Rebroadcast(*new_axis.items())(x)]


@register_canonicalize
@register_specialize
@gof.local_optimizer([T.Rebroadcast])
def local_rebroadcast_lift(node):
    """
    Lifts Rebroadcast through unary Elemwise operations,
    and merges consecutive Rebroadcasts.

    Rebroadcast(Elemwise(x)) => Elemwise(Rebroadcast(x))
    Rebroadcast(Rebroadcast(x)) => Rebroadcast(x)
    """
    op = node.op
    if not isinstance(op, T.Rebroadcast):
        return False

    input = node.inputs[0]
    inode = input.owner
    if inode and isinstance(inode.op, Elemwise) and len(inode.inputs) == 1:
        if len(input.clients) == 1:
            rval = inode.op.make_node(T.Rebroadcast(*op.axis.items())(
                    inode.inputs[0])).outputs
            return rval
    if inode and isinstance(inode.op, T.Rebroadcast):
        # the "axis" specification in the outer Rebroadcast overrides
        # the axis of the inner one
        axis = inode.op.axis.copy()
        axis.update(op.axis)
        iinput = inode.inputs[0]
        rval = [T.Rebroadcast(*axis.items())(iinput)]
        return rval


def apply_rebroadcast_opt(rval):
    """
    Apply as many times as required the optimization local_useless_rebroadcast
    and local_rebroadcast_lift.

    :param rval: a Variable
    :retrun: a Variable. The same if not optimisation can be applied.
    """

    changed = True
    while changed and rval.owner:
        changed = False
        rval2 = theano.tensor.opt.local_useless_rebroadcast.transform(
            rval.owner)
        if rval2:
            assert len(rval2) == 1
            rval = rval2[0]
            changed = True
        if rval.owner:
            rval2 = theano.tensor.opt.local_rebroadcast_lift.transform(
                rval.owner)
            if rval2:
                assert len(rval2) == 1
                rval = rval2[0]
                changed = True
    return rval


#############
# Join opts #
#############
@register_specialize
@register_canonicalize
@gof.local_optimizer([T.Join])
def local_join_1(node):
    """Join(i, x) => x

    Remove Join() when only one element is joined.
    """
    if not isinstance(node.op, T.Join):
        return
    axis = node.inputs[0]
    tensors = node.inputs[1:]
    if len(tensors) == 1:
        return [tensors[0]]


###############
# Switch opts #
###############

@register_canonicalize
@gof.local_optimizer([])
def local_remove_switch_const_cond(node):
    """
    This optimization makes the following changes in the graph:
        T.switch(cond,left,right) -->
               if cond is constant and cond == 0: right
               if cond is constant and cond != 0: left
    """
    if (isinstance(node.op, T.Elemwise) and
        isinstance(node.op.scalar_op, scalar.basic.Switch)):
        cond = T.extract_constant(node.inputs[0])
        if type(cond) is numpy.ndarray and cond.ndim == 0:
            if cond == 0:
                out = node.inputs[2]
            else:
                out = node.inputs[1]

            if out.ndim != node.outputs[0].ndim:
                #TODO: broadcast?
                return False
            if out.dtype != node.outputs[0].dtype:
                out = T.cast(out, node.outputs[0].dtype)
            if out.type.broadcastable != node.outputs[0].type.broadcastable:
                # We need to copy data to the new dimensions during execution
                out = T.alloc(out, *[node.outputs[0].shape[i] for i
                                     in xrange(out.ndim)])
            return [out]

        return False
    return False


@register_canonicalize
@gof.local_optimizer([T.mul])
def local_mul_switch_sink(node):
    """
    This optimization makes the folowing changes in the graph:
    T.mul(A,T.switch(cond,0,iff),B) -->  T.switch(cond,0,T.mul(A,B,iff))
    T.mul(A,T.switch(cond,ift,0),B) -->  T.switch(cond,T.mul(A,B,ift),0)
    A and B being several (or none) symbolic variables.
    This is useful because A and B may not be numerically stable and give
    NaN or inf values for cases where the switch returns 0.
    With this optimization T.grad(T.switch(...)) has the right behavior.
    Exemple:
      x -> f(x)
      x -> g(x)
      y = T.switch(cond,f(x),g(x))
      **without the optimization
      T.grad(y,x) -> grad(f(x),x) * grad(y,f(x)) +  grad(g(x),x) * grad(y,g(x))
      **with the optimization
      T.grad(y,x) -> switch(cond,grad(f(x),x), 0) + switch(cond,0,grad(g(x),x))
    This will be particularly useful for the lazyif because we skip
    an entire part of the graph.
    """
    if node.op != T.mul:
        return False
    for idx, i in enumerate(node.inputs):
        if i.owner and i.owner.op == T.switch:
            switch = i.owner
            try:
                if get_constant_value(switch.inputs[1]) == 0.:
                    listmul = node.inputs[:idx] + node.inputs[idx + 1:]
                    fct = [T.switch(switch.inputs[0], 0,
                                    T.mul(*(listmul + [switch.inputs[2]])))]
                    fct[0].values_eq_approx = fct[
                        0].type.values_eq_approx_remove_nan
                    return fct
            except TypeError:
                pass
            try:
                if get_constant_value(switch.inputs[2]) == 0.:
                    listmul = node.inputs[:idx] + node.inputs[idx + 1:]
                    fct = [T.switch(switch.inputs[0],
                                    T.mul(*(listmul + [switch.inputs[1]])), 0)]
                    fct[0].values_eq_approx = fct[
                        0].type.values_eq_approx_remove_nan
                    return fct
            except TypeError:
                pass
    return False


@register_canonicalize
@gof.local_optimizer([T.true_div])
def local_div_switch_sink(node):
    """
    This optimization makes the folowing changes in the graph:
    T.div(T.switch(cond,0,iff),A) -->  T.switch(cond,0,T.div(iff,A))
    T.div(T.switch(cond,ift,0),A) -->  T.switch(cond,T.div(ift,A),0)

    A being a symbolic variable.
    This is useful because A may not be numerically stable and give
    NaN or inf values for cases where the switch returns 0.
    See local_mul_switch_sink for more details.
    """
    if (node.op != T.true_div and node.op != T.int_div
        and node.op != T.floor_div):
        return False
    op = node.op
    if node.inputs[0].owner and node.inputs[0].owner.op == T.switch:
        switch = node.inputs[0].owner
        try:
            if get_constant_value(switch.inputs[1]) == 0.:
                fct = [T.switch(switch.inputs[0], 0,
                                op(switch.inputs[2], node.inputs[1]))]
                fct[0].values_eq_approx = fct[
                    0].type.values_eq_approx_remove_nan
                return fct
        except TypeError:
            pass
        try:
            if get_constant_value(switch.inputs[2]) == 0.:
                fct = [T.switch(switch.inputs[0],
                                op(switch.inputs[1], node.inputs[1]), 0)]
                fct[0].values_eq_approx = fct[
                    0].type.values_eq_approx_remove_nan
                return fct
        except TypeError:
            pass
    return False


##################
# Reshape opts   #
##################


@gof.local_optimizer([None, None])
def local_reshape_chain(node):
    """
    Reshape(Reshape(shape1),shape2) -> Reshape(shape2)
    """
    if not opt.check_chain(node, T.Reshape, T.Reshape):
        return False

    # TODO: this can permit a failing program to run by eliminating
    #       the lower reshape
    rval = node.op(node.inputs[0].owner.inputs[0], node.inputs[1])
    # It might happen that the desired output of this node has a broadcastable
    # pattern that does not match that of 'rval'. This is when originally, we
    # were able to figure out that one of the dimensions of the reshape is one,
    # but some other transformation replaced the shape by one for which this
    # cannot be guessed.
    # We should try to figure out why we lost the information about this
    # constant value... but in the meantime, better not apply this
    # optimization.
    if rval.broadcastable == node.outputs[0].broadcastable:
        return [rval]
    else:
        return False
register_canonicalize(local_reshape_chain)

if 0:
    # TODO: Test that this optimziation works.
    @register_canonicalize
    @gof.local_optimizer([])
    def local_scalar_reshape(node):
        """Eliminate reshape Ops whose inputs and outputs are scalars """
        if isinstance(node.op, T.Reshape):
            x, shp = node.inputs
            if x.ndim == 0 and T.get_vector_length(shp) == 0:
                return [x]

if 0:
    # TODO: Finish writing and testing this optimization.  The idea is
    #       that if we can prove the output to this sum has a
    #       zero-size dimension, then it can be replaced by an
    #       appropriately typed and broadcasted zero.
    # TODO: Remember to take into account the new sum dtype argument if this
    #       optimization is enabled.
    @register_canonicalize
    @gof.local_optimizer([])
    def local_sum_over_empty(node):
        if isinstance(node.op, T.Sum):
            # This optimization needs ShapeOpt and env.shape_feature
            if not hasattr(node.env, 'shape_feature'):
                return
            y, = node.outputs
            y_shape = node.env.shape_feature.shape_of[y]

            def tmp(thing):
                try:
                    return T.get_constant_value(thing)
                except (TypeError, ValueError), e:
                    print e, thing.owner.inputs[0]
                    return None
            print 'LOCAL SUM EMPTY', [tmp(s) for s in y_shape]

##################
# Middleman cuts #
##################


@gof.local_optimizer([None, T.fill])
def local_fill_cut(node):
    """
    f(fill(a,b), c) -> f(b, c)
    If c.type == a.type.
    """

    # this optimization is essentially for getting broadcasting to
    # replace fill.  This is always possible when using a Compound
    # Elemwise operation, but it is not always possible without one
    # (consider filling a large matrix with a scalar, and then adding
    # another scalar.  The only numbers that count are the two
    # scalars, but we can't ignore the large matrix because it gives
    # the shape of the result.

    if not opt.check_chain(node, T.Elemwise):
        return False

    output = node.outputs[0]
    try:
        #reference is some input with the same type as the input but
        #that is not produced by a fill
        reference = [input
                     for input in node.inputs
                     if input.type == output.type and
                     (not input.owner or input.owner.op != T.fill)][0]
    except IndexError:
        return False

    new_inputs = []
    for input in node.inputs:
        if opt.check_chain(input, T.fill):
            model, filling = input.owner.inputs
            if encompasses_broadcastable(reference.type.broadcastable,
                                         filling.type.broadcastable):
                new_inputs.append(filling)
                continue
        new_inputs.append(input)

    if new_inputs == node.inputs:
        return False

    rval = node.op(*new_inputs)
    if isinstance(rval, gof.Variable):
        return rval.owner.outputs
    else:
        return rval[0].owner.outputs

register_canonicalize(local_fill_cut)


register_canonicalize(gof.OpRemove(T.tensor_copy), name='remove_tensor_copy')


@gof.local_optimizer([None, T.fill])
def local_fill_sink(node):
    """
    f(fill(a, b), fill(c, d), e) -> fill(a, fill(c, f(b, d, e)))
    """
    if not (node.op and isinstance(node.op, T.Elemwise) and node.op != T.fill):
        return False
    models = []
    inputs = []
    for input in node.inputs:
        if input.owner and input.owner.op == T.fill:
            models.append(input.owner.inputs[0])
            inputs.append(input.owner.inputs[1])
        else:
            inputs.append(input)
    if inputs == node.inputs:
        return False
    c = node.op(*inputs)
    for model in models:
        c = T.fill(model, c)
    return [c]

register_canonicalize(local_fill_sink)


################
# Canonization #
################

class Canonizer(gof.LocalOptimizer):
    """
    Simplification tool.

    Usage: Canonizer(main, inverse, reciprocal, calculate)

    * main: a suitable Op class that is commutative, associative and
            takes one to an arbitrary number of inputs, e.g. add or
            mul
    * inverse: an Op class such that inverse(main(x, y), y) == x
               e.g. sub or true_div
    * reciprocal: a function such that main(x, reciprocal(y)) ==
                  inverse(x, y) e.g. neg or inv

    * calculate: function that takes a list of numpy.ndarray instances
                 for the numerator, another list for the denumerator,
                 and calculates inverse(main(*num), main(*denum)). It
                 takes a keyword argument, aslist. If True, the value
                 should be returned as a list of one element, unless
                 the value is such that value = main(). In that case,
                 the return value should be an empty list.

    The variable is a local_optimizer. It is best used with a TopoOptimizer in
    in_to_out order.

    Examples:
      T = theano.tensor
      add_canonizer = Canonizer(T.add, T.sub, T.neg,
                                lambda n, d: sum(n) - sum(d))
      mul_canonizer = Canonizer(T.mul, T.true_div, T.inv,
                                lambda n, d: prod(n) / prod(d))

    Examples of optimizations mul_canonizer can perform:
      x / x -> 1
      (x * y) / x -> y
      x / y / x -> 1 / y
      x / y / z -> x / (y * z)
      x / (y / z) -> (x * z) / y
      (a / b) * (b / c) * (c / d) -> a / d
      (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
      2 * x / 2 -> x
      x * y * z -> Elemwise(T.mul){x,y,z} #only one pass over the memory.
                !-> Elemwise(T.mul){x,Elemwise(T.mul){y,z}}
    """

    def __init__(self, main, inverse, reciprocal, calculate,
                 use_reciprocal=True):
        self.main = main
        self.inverse = inverse
        self.reciprocal = reciprocal
        self.calculate = calculate
        self.use_reciprocal = use_reciprocal

        self.external_simplifiers = []

    def add_simplifier(self, simplifier, reason):
        self.external_simplifiers.append((reason, simplifier))

    def tracks(self):
        return [[self.main, None], [self.inverse, None],
                [self.reciprocal, None]]

    def get_num_denum(self, input):
        """
        This extract two lists, num and denum, such that the input is:
        self.inverse(self.main(*num), self.main(*denum)). It returns
        the two lists in a (num, denum) pair.

        For example, for main, inverse and reciprocal = *, / and inv(),

        input -> returned value (num, denum)

        x*y -> ([x, y], [])
        inv(x) -> ([], [x])
        inv(x) * inv(y) -> ([], [x, y])
        x*y/z -> ([x, y], [z])
        log(x) / y * (z + x) / y -> ([log(x), z + x], [y, y])
        (((a / b) * c) / d) -> ([a, c], [b, d])
        a / (b / c) -> ([a, c], [b])
        log(x) -> ([log(x)], [])
        x**y -> ([x**y], [])
        x * y * z -> ([x, y, z], [])

        """

        # This function is recursive.  The idea is that there is a
        # get_num_denum recursion in which the internal ops are all
        # one of (main, inverse, reciprocal, DimShuffle) and the
        # internal data nodes all have the dtype of the 'input'
        # argument. The leaf-Variables of the graph covered by the
        # recursion may be of any Variable type.

        if 0:
            # UPDATE: This logic makes it impossible to recognize some
            # important patterns (e.g. variants on the x/x) and it is
            # screwing up the RBM free energy gradient.
            #TODO: review this
            if len(input.clients) > 1:
                # this logic is too conservative, but doing it is
                # better than not doing it.
                #
                # we don't want to canonize a subgraph that we will
                # need to compute anyway for the other clients.

                # This check is too conservative because if the other
                # clients are also in the subgraph we are canonizing,
                # then we should [probably?] recurse anyway.
                return [input], []

        if input.owner is None or input.owner.op not in [
            self.main, self.inverse, self.reciprocal]:
            if input.owner and isinstance(input.owner.op, T.DimShuffle):
                # If input is a DimShuffle of some input which does
                # something like this:

                # * change a vector of length N into a 1xN row matrix
                # * change a scalar into a 1x1x1 tensor
                # * in general, complete the shape of a tensor
                #   with broadcastable 1s to the *left*
                # Then we will simply discard the DimShuffle and return
                # the num/denum of its input
                dsn = input.owner    # dimshuffle node
                dsop = dsn.op        # dimshuffle op
                dsi0 = dsn.inputs[0]  # the first input of the
                                      # dimshuffle i.e. the ndarray to
                                      # redim

                # The compatible order is a DimShuffle "new_order" of the form:
                # ('x', ..., 'x', 0, 1, 2, ..., dimshuffle_input.type.ndim)

                # That kind of DimShuffle only adds broadcastable
                # dimensions on the left, without discarding any
                # existing broadcastable dimension and is inserted
                # automatically by Elemwise when the inputs have
                # different numbers of dimensions (hence why we can
                # discard its information - we know we can retrieve it
                # later on).
                compatible_order = ('x',) * (input.type.ndim
                                             - dsi0.type.ndim) + tuple(
                    range(dsi0.type.ndim))
                if dsop.new_order == compatible_order:
                    # If the "new_order" is the one we recognize,
                    # we return the num_denum of the dimshuffled input.
                    return self.get_num_denum(input.owner.inputs[0])
                else:
                    # This is when the input isn't produced by main,
                    # inverse or reciprocal.
                    return [input], []
            else:
                return [input], []
        num = []
        denum = []
        parent = input.owner

        # We get the (num, denum) pairs for each input
        #pairs = [self.get_num_denum(input2) if input2.type.dtype ==
        #input.type.dtype else ([input2], []) for input2 in
        #parent.inputs]
        pairs = [self.get_num_denum(input2) for input2 in parent.inputs]

        if parent.op == self.main:
            # If we have main(x, y), numx, denumx, numy and denumy
            # then num is concat(numx, numy) and denum is
            # concat(denumx, denumy) note that main() can have any
            # number of arguments >= 0 concat is list concatenation
            num = reduce(list.__iadd__, map(operator.itemgetter(0), pairs))
            denum = reduce(list.__iadd__, map(operator.itemgetter(1), pairs))
        elif parent.op == self.inverse:
            # If we have inverse(x, y), numx, denumx, numy and denumy
            # then num is concat(numx, denumy) and denum is
            # concat(denumx, numy) note that inverse() is binary
            num = pairs[0][0] + pairs[1][1]
            denum = pairs[0][1] + pairs[1][0]
        elif parent.op == self.reciprocal:
            # If we have reciprocal(x), numx, denumx
            # then num is denumx and denum is numx
            # note that reciprocal() is unary
            num = pairs[0][1]
            denum = pairs[0][0]
        return num, denum

    def merge_num_denum(self, num, denum):
        """
        Utility function which takes two lists, num and denum, and
        returns something which is equivalent to inverse(main(*num),
        main(*denum)), but depends on the length of num and the length
        of denum (in order to minimize the number of operations).

        Let n = len(num) and d = len(denum):

        n=0, d=0: neutral element (given by self.calculate([], []))
                  (for example, this would be 0 if main is addition
                  and 1 if main is multiplication)
        n=1, d=0: num[0]
        n=0, d=1: reciprocal(denum[0])
        n=1, d=1: inverse(num[0], denum[0])
        n=0, d>1: reciprocal(main(*denum))
        n>1, d=0: main(*num)
        n=1, d>1: inverse(num[0], main(*denum))
        n>1, d=1: inverse(main(*num), denum[0])
        n>1, d>1: inverse(main(*num), main(*denum))

        Given the values of n and d to which they are associated, all
        of the above are equivalent to:
        inverse(main(*num), main(*denum))
        """

        ln, ld = len(num), len(denum)
        if not ln and not ld:
            return T.as_tensor_variable(self.calculate([], []))
        if not ln:
            if self.use_reciprocal:
                return self.reciprocal(self.merge_num_denum(denum, []))
            else:
                ln = [self.calculate([], [], aslist=False)]
        if not ld:
            if ln == 1:
                # num[0] should always be a variable
                assert isinstance(num[0], gof.Variable)
                return num[0]
            else:
                return self.main(*num)
        return self.inverse(self.merge_num_denum(num, []),
                            self.merge_num_denum(denum, []))

    @staticmethod
    def get_constant(v):
        """
        Returns a numeric constant if v is a Constant or, well, a
        numeric constant. If v is a plain Variable, returns None.
        """
        if isinstance(v, Variable):
            try:
                return get_constant_value(v)
            except TypeError:
                return None
        else:
            return v

    def simplify(self, num, denum):
        """
        Shorthand for:
        self.simplify_constants(*self.simplify_factors(num, denum))
        """
        rval = self.simplify_constants(*self.simplify_factors(num, denum))
        for reason, simplifier in self.external_simplifiers:
            # TODO: document that 'reason' is associated with this
            #       simplification to help auditing when things go
            #       wrong
            rval = simplifier(*rval)
        return rval

    def simplify_factors(self, num, denum):
        """
        For any Variable r which is both in num and denum, removes it
        from both lists. Modifies the lists inplace. Returns the
        modified lists. For example:

        [x], [x] -> [], []
        [x, y], [x] -> [y], []
        [a, b], [c, d] -> [a, b], [c, d]
        """
        for v in list(num):
            if v in denum:
                num.remove(v)
                denum.remove(v)
        return num, denum

    def simplify_constants(self, orig_num, orig_denum):
        """

        Finds all constants in orig_num and orig_denum (using
        get_constant) and puts them together into a single
        constant. The constant is inserted as the first element of the
        numerator. If the constant is the neutral element, it is
        removed from the numerator. Examples:

        Let main be multiplication:

        [2, 3, x], [] -> [6, x], []
        [x, y, 2], [4, z] -> [0.5, x, y], [z]
        [x, 2, y], [z, 2] -> [x, y], [z]
        """

        # Lists representing the numerator and denumerator
        num, denum = list(orig_num), list(orig_denum)
        out_type = self.merge_num_denum(orig_num, orig_denum).type

        # Lists representing the *constant* elements of num and denum
        numct, denumct = [], []

        for v in orig_num:
            ct = self.get_constant(v)
            if ct is not None:
                # We found a constant in the numerator!
                # We remove it from num
                num.remove(v)
                # We add it to numct
                numct.append(ct)
        for v in orig_denum:
            ct = self.get_constant(v)
            if ct is not None:
                denum.remove(v)
                denumct.append(ct)

        if self.use_reciprocal or num:
            # This will calculate either:
            # [inverse(main(*numct), main(*denumct))]
            # [] - if inverse(main(*numct), main(*denumct)) is the
            # neutral element
            ct = self.calculate(numct, denumct, aslist=True,
                                out_type=out_type)
        else:
            # This happens if we don't allow the reciprocal and the
            # numerator is empty. That means we will need to represent
            # reciprocal(x) like inverse(neutral_element, x) so
            # we can't allow ct == []
            # TODO: why is this branch needed when merge_num_denum
            # does it for us?
            ct = [self.calculate(numct, denumct, aslist=False,
                                 out_type=out_type)]

        # Wrapping ct in a Constant with the right dtype
        ct = [T.constant(c, dtype=out_type.dtype) for c in ct]

        if orig_num and len(numct) == 1 and len(denumct) == 0 and ct and\
                N.all([c.data for c in ct] == self.get_constant(orig_num[0])):
            # this is an important trick :( if it so happens that:
            # * there's exactly one constant on the numerator and none on
            #   the denominator
            # * it's not the neutral element (ct is an empty list in that case)
            # * the constant is the same as the first argument in the numerator
            # Then we return very exactly the original num/denum
            # If we don't do that the optimizer will just loop
            # infinitely because it will not catch on that there are
            # no changes to be made and everytime it will want to
            # replace something by the same thing...
            return orig_num, orig_denum
        return ct + num, denum

    def transform(self, node):
        op = node.op
        if op not in [self.main, self.inverse, self.reciprocal]:
            return False

        inputs = node.inputs
        out = node.outputs[0]
        assert len(node.outputs) == 1

        # check if any of the clients of this node would be part of
        # this canonized graph...  if so, we do nothing and wait for
        # them to be transformed.
        def _bypass_dimshuffle(n):
            if isinstance(n.op, DimShuffle) and len(n.outputs[0].clients) <= 1:
                return _bypass_dimshuffle(n.outputs[0].clients.__iter__(
                        ).next()[0])
            else:
                return n
        for c, c_idx in out.clients:
            if c == 'output':
                continue
            if _bypass_dimshuffle(c).op in [self.main, self.inverse,
                                            self.reciprocal]:
                return False

        # Here we make the canonical version of the graph around this node
        # See the documentation of get_num_denum and simplify
        orig_num, orig_denum = self.get_num_denum(node.outputs[0])
        num, denum = self.simplify(list(orig_num), list(orig_denum))

        def same(x, y):
            return len(x) == len(y) and all(N.all(xe == ye) for xe, ye in
                                            zip(x, y))

        if same(orig_num, num) and same(orig_denum, denum):
            # We return False if there are no changes
            return False

        new = self.merge_num_denum(num, denum)
        if new.type.dtype != out.type.dtype:
            #new = T.fill(out, new)
            elem_op = T.Elemwise(scalar.Identity(scalar.specific_out(
                        getattr(scalar, out.type.dtype))))
            new = elem_op(new)

        assert (new.type == out.type) == (not (new.type != out.type))

        if not (new.type == out.type):
            new = _fill_chain(new, node.inputs)[0]

        if new.type == out.type:
            return [new]
        else:
            _logger.warning(' '.join(('CANONIZE FAILED: new, out = ',
                                      new, ',', out, 'types',
                new.type, ',', out.type)))
            return False

    def __str__(self):
        return getattr(self, 'name', 'Canonizer(%s, %s, %s)' % (
                self.main, self.inverse, self.reciprocal))


def mul_calculate(num, denum, aslist=False, out_type=None):
    if not num and not denum:
        # Smallest 1 possible.
        if aslist:
            return []
        else:
            return N.int8(1)

    # Make sure we do not accidently upcast data types.
    if out_type is None:
        out_dtype = scalar.upcast(*[v.dtype for v in (num + denum)])
    else:
        out_dtype = out_type.dtype
    one = theano._asarray(1, dtype=out_dtype)

    v = reduce(N.multiply, num, one) / reduce(N.multiply, denum, one)
    if aslist:
        if N.all(v == 1):
            return []
        else:
            return [v]
    return v

local_mul_canonizer = Canonizer(T.mul, T.true_div, T.inv, mul_calculate, False)
register_canonicalize(local_mul_canonizer, name='local_mul_canonizer')


@gof.local_optimizer([T.neg])
def local_neg_to_mul(node):
    if node.op == T.neg:
        return [T.mul(numpy.array(-1, dtype=node.inputs[0].dtype),
            node.inputs[0])]
register_canonicalize(local_neg_to_mul)


@register_specialize
@gof.local_optimizer([])
def local_sum_mul_by_scalar(node):
    """sum(scalar * smth) -> scalar * sum(smth)
       sum(-smth) -> -sum(smth)
    """
    # TODO: if the the thing inside the Sum is a division,
    # we should get at the numerator....
    if isinstance(node.op, T.Sum):
        thing_summed, = node.inputs
        if thing_summed.owner and thing_summed.owner.op == T.mul:
            terms = thing_summed.owner.inputs
            scalars = [t.dimshuffle() for t in terms if
                       numpy.all(t.type.broadcastable)]
            non_scalars = [t for t in terms if not numpy.all(t.broadcastable)]
            if scalars:
                if len(scalars) > 1:
                    if len(non_scalars) > 1:
                        return [T.mul(T.mul(*scalars),
                                      node.op(T.mul(*non_scalars)))]
                    elif len(non_scalars) == 1:
                        return [T.mul(T.mul(*scalars),
                                      node.op(non_scalars[0]))]
                    else:
                        return [T.mul(*scalars)]
                else:
                    if len(non_scalars) > 1:
                        return [T.mul(scalars[0],
                                      node.op(T.mul(*non_scalars)))]
                    elif len(non_scalars) == 1:
                        return [T.mul(scalars[0], node.op(non_scalars[0]))]
                    else:
                        return [scalars[0]]
        if thing_summed.owner and thing_summed.owner.op == T.neg:
            return [T.neg(node.op(thing_summed.owner.inputs[0]))]


@register_specialize
@gof.local_optimizer([])
def local_elemwise_sub_zeros(node):
    """
    Elemwise{sub}(X,X) -> zeros_like(X)
    """
    if (isinstance(node.op, T.Elemwise)
        and node.op.scalar_op.nin == 2
        and node.op.scalar_op == scalar.sub
        and node.inputs[0] == node.inputs[1]):
        return [T.zeros_like(node.inputs[0])]


@register_canonicalize
@register_specialize
@gof.local_optimizer([])
def local_sum_div_dimshuffle(node):
    '''sum(a / dimshuffle{...}(b), axis=l) -> sum(a, axis={...}) / b,
    if dimension l of the DimShuffle is 'x'.'''
    # TODO: extend it to product, and quotient of products

    # It does not make much sense now to extend it to the case where the
    # dimshuffle is in the numerator, since elemwise inversion of the
    # denominator would still be needed before the summation.

    if isinstance(node.op, T.Sum):
        axis = node.op.axis
        if axis is None:
            axis = range(node.inputs[0].ndim)
        #print 'axis =', axis
        thing_summed = node.inputs[0]
        dimshuffled = None
        if thing_summed.owner and thing_summed.owner.op == T.true_div:
            numerator, denominator = thing_summed.owner.inputs

            ## Old, bugged logic, reproduced here only to warn users
            if config.warn.sum_div_dimshuffle_bug:
                if numerator.owner and isinstance(numerator.owner.op,
                                                  T.DimShuffle):
                    new_order = numerator.owner.op.new_order
                    compatible_dims = True
                    for ax in axis:
                        if len(new_order) <= ax or new_order[ax] != 'x':
                            compatible_dims = False
                            break
                    if compatible_dims:
                        _logger.warn('WARNING: Your current code is fine, but'
                                     ' Theano versions between '
                                     'rev. 3bd9b789f5e8 (2010-06-16) and'
                                     ' cfc6322e5ad4 (2010-08-03) would '
                                     'have given an incorrect result. '
                                     'To disable this warning, set the Theano'
                                     ' flag warn.sum_div_dimshuffle_bug to'
                                     ' False.')

            if denominator.owner and isinstance(denominator.owner.op,
                                                T.DimShuffle):
                thing_dimshuffled = denominator.owner.inputs[0]
                new_order = denominator.owner.op.new_order
                #print 'new_order =', new_order
                # check compatibility
                compatible_dims = True
                for ax in axis:
                    #print 'ax =', ax
                    #print 'len(new_order) =', len(new_order)
                    #print 'new_order[ax] =', new_order[ax]
                    if len(new_order) <= ax or new_order[ax] != 'x':
                        compatible_dims = False
                        break

                if compatible_dims:
                    #print 'getting denom out'
                    # Keep needed dimensions for new dimshuffle
                    new_new_order = list(ax for i, ax in enumerate(new_order)
                                         if i not in axis or ax != 'x')
                    #print 'new_new_order =', new_new_order
                    # Remove useless rebroadcast axes
                    while len(new_new_order) > 0 and new_new_order[0] == 'x':
                        del new_new_order[0]
                    #print 'new_new_order =', new_new_order

                    if all(i == e for i, e in enumerate(new_new_order)):
                        new_denom = thing_dimshuffled
                    else:
                        if config.warn.sum_div_dimshuffle_bug:
                            _logger.warn('WARNING: Your current code is fine,'
                                         ' but Theano versions between '
                                         'rev. 3bd9b789f5e8 (2010-06-16) and'
                                         ' cfc6322e5ad4 (2010-08-03) would '
                                         'have given an incorrect result. '
                                         'To disable this warning, set the'
                                         ' Theano flag '
                                         'warn.sum_div_dimshuffle_bug'
                                         ' to False.')

                        new_denom = T.DimShuffle(
                                    thing_dimshuffled.type.broadcastable,
                                    new_new_order
                                    )(thing_dimshuffled)
                    return [T.true_div(node.op(numerator), new_denom)]
                #else:
                #    print 'incompatible dims:', axis, new_order


@register_canonicalize
@gof.local_optimizer([])
def local_sum_all_to_none(node):
    """Sum{0,1,...N} -> Sum{}"""
    if isinstance(node.op, T.Sum):
        # if all the axes are named, then use None as a shorthand
        # this permits more merging
        if node.op.axis is None:
            return
        if set(node.op.axis) == set(range(node.inputs[0].type.ndim)):
            return [T.Sum(axis=None, dtype=node.op.dtype)(node.inputs[0])]


@register_canonicalize
@gof.local_optimizer([])
def local_sum_sum(node):
    """
    Sum(Sum()) -> Sum
    """
    if isinstance(node.op, T.Sum):
        summed, = node.inputs
        out_dtype = node.op.dtype
        if len(summed.clients) == 1:
            if (summed.owner and
                    isinstance(summed.owner.op, T.Sum)):

                if summed.owner.op.axis is None:
                    # special case of local_cut_useless_reduce
                    return [T.Sum(None, dtype=out_dtype)(summed.owner.inputs[0])]
                if node.op.axis is None:
                    # we're summing up everything anyway so lets
                    # do it all at once
                    return [T.Sum(None, dtype=out_dtype)(summed.owner.inputs[0])]

                newaxis = list(tuple(summed.owner.op.axis))
                # figure out which dimensions of the original input
                # are preserved
                for i in node.op.axis:
                    new_i = i
                    for ii in summed.owner.op.axis:
                        if i >= ii:
                            new_i += 1
                    assert new_i not in newaxis
                    newaxis.append(new_i)

                assert len(newaxis) == len(list(summed.owner.op.axis) +
                                           list(node.op.axis))

                # The old bugged logic. We keep it there to generate a warning
                # when we generated bad code.
                alldims = range(summed.owner.inputs[0].type.ndim)
                alldims = [d for i, d in enumerate(alldims) if i
                           in summed.owner.op.axis]
                alldims = [d for i, d in enumerate(alldims)
                           if i in node.op.axis]
                newaxis_old = [i for i in
                               xrange(summed.owner.inputs[0].type.ndim)
                               if i not in alldims]

                if (theano.config.warn.sum_sum_bug and
                    newaxis != newaxis_old and
                    len(newaxis) == len(newaxis_old)):
                    _logger.warn(
                            "WARNING (YOUR CURRENT CODE IS FINE): Theano "
                            "versions between version 9923a40c7b7a and August "
                            "2nd, 2010 generated bugged code in this case. "
                            "This happens when there are two consecutive sums "
                            "in the graph and the intermediate sum is not "
                            "used elsewhere in the code. Some safeguard "
                            "removed some bad code, but not in all cases. You "
                            "are in one such case. To disable this warning "
                            "(that you can safely ignore since this bug has "
                            "been fixed) set the theano flag "
                            "`warn.sum_sum_bug` to False.")

                combined_sum = T.Sum(newaxis, dtype=out_dtype)
                return [combined_sum(summed.owner.inputs[0])]


@register_canonicalize
@gof.local_optimizer([])
def local_cut_useless_reduce(node):
    """Sum(a, axis=[]) -> a  """
    if isinstance(node.op, T.CAReduce):
        summed, = node.inputs
        # if reduce were doing anything, the output ndim would be reduced
        if summed.type == node.outputs[0].type:
            return [summed]


@register_specialize
@gof.local_optimizer([])
def local_sum_alloc(node):
    """ sum(alloc(constant,shapes...)) => constant*prod(shapes)"""
    if isinstance(node.op, T.Sum):
        summed, = node.inputs
        if summed.owner and isinstance(summed.owner.op, T.Alloc):
            input = summed.owner.inputs[0]
            shapes = summed.owner.inputs[1:]
            if (node.op.axis is None or
                node.op.axis == tuple(range(input.ndim))):
                try:
                    val = get_constant_value(input)
                    assert val.size == 1
                    val = val.reshape(1)[0] * T.mul(*shapes)
                    return [T.cast(val, dtype=node.outputs[0].dtype)]
                except TypeError, e:
                    pass
            else:
                try:
                    val = get_constant_value(input)
                    assert val.size == 1
                    val = val.reshape(1)[0]
                    to_prod = [shapes[i] for i in xrange(len(shapes))
                               if i in node.op.axis]
                    if to_prod:
                        val *= T.mul(*to_prod)
                    return [T.alloc(T.cast(val, dtype=node.outputs[0].dtype),
                                    *[shapes[i] for i in xrange(len(shapes))
                                      if i not in node.op.axis])]
                except TypeError, e:
                    pass


@gof.local_optimizer([T.mul])
def local_mul_to_neg(node):
    if node.op == T.mul and N.all(
        local_mul_canonizer.get_constant(node.inputs[0]) == -1.0):
        other_prod = local_mul_canonizer.merge_num_denum(node.inputs[1:], [])
        if other_prod.type == node.outputs[0].type:
            return [-other_prod]
        # else the multiplication is also acting as a cast, so we
        # might as well leave it alone.  I don't think it's better to
        # turn this into a negation in the wrong type, followed by an
        # explicit cast.
register_specialize(local_mul_to_neg)


@register_specialize
@gof.local_optimizer([T.neg])
def local_neg_neg(node):
    # other specializations shouldn't put this in,
    # but sometimes they do
    if node.op == T.neg:
        if node.inputs[0].owner and node.inputs[0].owner.op == T.neg:
            return [node.inputs[0].owner.inputs[0]]


@register_specialize
@gof.local_optimizer([T.neg])
def local_neg_div_neg(node):
    """- (-a / b) -> a / b

    Also performs - (c / b) -> ((-c) / b) when c is a scalar constant.
    """
    if node.op == T.neg:
        if node.inputs[0].owner and node.inputs[0].owner.op == T.true_div:
            frac = node.inputs[0]
            num, denom = frac.owner.inputs
            if num.owner and num.owner.op == T.neg:
                if len(frac.clients) == 1:
                    # No other clients of the original division
                    new_num = num.owner.inputs[0]
                    return [T.true_div(new_num, denom)]
            elif numpy.all(num.broadcastable) and isinstance(num, Constant):
                if len(frac.clients) == 1:
                    new_num = -num.data
                    return [T.true_div(new_num, denom)]


@gof.local_optimizer([T.mul])
def local_mul_zero(node):
    """As part of canonicalization, we replace multiplication by zero
    with zero.
    """
    if node.op == T.mul:
        otype = node.outputs[0].type

        for i in node.inputs:
            try:
                value = get_constant_value(i)
            except TypeError:
                continue
            #print 'MUL by value', value, node.inputs
            if N.all(value == 0):
                #print '... returning zeros'
                return _fill_chain(theano._asarray(0, dtype=otype.dtype),
                                   node.inputs)
register_canonicalize(local_mul_zero)


@gof.local_optimizer([T.true_div])
def local_div_to_inv(node):
    if node.op == T.true_div and N.all(
        local_mul_canonizer.get_constant(node.inputs[0]) == 1.0):
        out = node.outputs[0]
        new_out = T.inv(local_mul_canonizer.merge_num_denum(node.inputs[1:],
                                                            []))
        # The ones could have forced upcasting
        if new_out.dtype != out.dtype:
            new_out = T.cast(new_out, dtype=out.dtype)
        # The ones could have forced a specific length
        if new_out.type != out.type:
            new_out = broadcast_like(new_out, out, node.env)
        return [new_out]
    else:
        return False
register_specialize(local_div_to_inv)


@gof.local_optimizer([T.inv])
def local_inv_canon(node):
    if node.op == T.inv:
        return [T.pow(node.inputs[0], -1.0)]
    else:
        return False
register_canonicalize(local_inv_canon)


@gof.local_optimizer([T.pow])
def local_pow_canonicalize(node):
    if node.op == T.pow:
        if N.all(local_mul_canonizer.get_constant(node.inputs[1]) == 0):
            return [broadcast_like(1, node.outputs[0], node.env)]
        if N.all(local_mul_canonizer.get_constant(node.inputs[1]) == 1):
            return [broadcast_like(node.inputs[0], node.outputs[0], node.env)]
    else:
        return False
register_canonicalize(local_pow_canonicalize)


@register_specialize
@gof.local_optimizer([T.mul])
def local_mul_to_sqr(node):
    """x*x -> sqr(x)

    This is faster on the GPU when memory fetching is a big part of
    the computation time.
    """
    if node.op == T.mul:
        if len(node.inputs) == 2:
            if node.inputs[0] is node.inputs[1]:
                return [T.sqr(node.inputs[0])]


@gof.local_optimizer([T.pow])
def local_pow_specialize(node):
    #here, we are past the point of canonicalization, so we don't want
    #to put in un-necessary fills.
    if node.op == T.pow:
        #the idea here is that we have pow(x, y)
        odtype = node.outputs[0].dtype
        xsym = node.inputs[0]
        ysym = node.inputs[1]
        y = local_mul_canonizer.get_constant(ysym)
        if (y is not None) \
                and encompasses_broadcastable(xsym.type.broadcastable,
                                              ysym.type.broadcastable):
            rval = None

            if N.all(y == 2):
                rval = [T.sqr(xsym)]
            if N.all(y == 1):
                rval = [xsym]
            if N.all(y == 0):
                rval = [T.fill(xsym, numpy.asarray(1, dtype=odtype))]
            if N.all(y == 0.5):
                rval = [T.sqrt(xsym)]
            if N.all(y == -0.5):
                rval = [T.inv(T.sqrt(xsym))]
            if N.all(y == -1):
                rval = [T.inv(xsym)]
            if N.all(y == -2):
                rval = [T.inv(T.sqr(xsym))]
            if rval:
                rval[0] = T.cast(rval[0], odtype)
                assert rval[0].type == node.outputs[0].type, (
                    rval, node.outputs)
                return rval
    else:
        return False
register_specialize(local_pow_specialize)


@register_specialize_device
@gof.local_optimizer([T.pow])
def local_pow_specialize_device(node):
    """
    This optimization is not the same on all device. We do it only on cpu here.
    """
    if node.op == T.pow:
        #the idea here is that we have pow(x, y)
        odtype = node.outputs[0].dtype
        xsym = node.inputs[0]
        ysym = node.inputs[1]
        y = local_mul_canonizer.get_constant(ysym)

        #the next line is needed to fix a strange case that I don't
        #know how to make a separate test.
        #That happen in the test_opt.py:test_log_erfc test.
        #y is a ndarray with dtype int8 and value 2,4 or 6. This make
        #the abs(y) <= 512 fail!
        #taking the value outside ndarray solve the problem.
        #it could be that in that case, numpy make the comparaison
        #into the wrong type(do in int8 that overflow.)
        if isinstance(y, numpy.ndarray):
            assert y.size == 1
            try:
                y = y[0]
            except IndexError:
                pass
        if (y is not None) \
                and encompasses_broadcastable(xsym.type.broadcastable,
                                              ysym.type.broadcastable):
            rval = None
            # 512 is too small for the cpu and too big for some gpu!
            if abs(y) == int(abs(y)) and abs(y) <= 512:
                pow2 = [xsym]
                pow2_scal = [theano.scalar.Scalar(xsym.dtype)()]
                y_to_do = abs(y)
                for i in xrange(int(numpy.log2(y_to_do))):
                    pow2.append(T.sqr(pow2[i]))
                    pow2_scal.append(theano.scalar.sqr(pow2_scal[i]))
                rval1 = None
                rval1_scal = None
                while y_to_do > 0:
                    log_to_do = int(numpy.log2(y_to_do))
                    if rval1:
                        rval1 *= pow2[log_to_do]
                        rval1_scal *= pow2_scal[log_to_do]
                    else:
                        rval1 = pow2[log_to_do]
                        rval1_scal = pow2_scal[log_to_do]
                    y_to_do -= 2 ** log_to_do

                if abs(y) > 2:
                    #We fuse all the pow together here to make
                    #compilation faster
                    rval1 = Elemwise(theano.scalar.Composite(
                            [pow2_scal[0]], [rval1_scal])).make_node(xsym)
                if y < 0:
                    rval = [T.inv(rval1)]
                else:
                    rval = [rval1]
            if rval:
                rval[0] = T.cast(rval[0], odtype)
                assert rval[0].type == node.outputs[0].type, (
                    rval, node.outputs)
                return rval


@gof.local_optimizer([T.mul])
def local_mul_specialize(node):
    """Remove special-case constants from mul arguments
    """
    # here, we are past the point of canonicalization, so we don't
    # want to put in un-necessary fills.
    #
    # at this point [post canonicalize], mul() may have many inputs.
    if node.op == T.mul:
        #the idea here is that we have pow(x, y)
        neg = False
        new_inputs = []
        for input in node.inputs:
            # remove any neg arguments
            while input.owner and input.owner.op == T.neg:
                neg ^= True
                input = input.owner.inputs[0]

            # remove special case arguments of 1, -1 or 0
            y = local_mul_canonizer.get_constant(input)
            if N.all(y == 1.0):
                continue
            elif N.all(y == -1.0):
                neg ^= True  # toggles
            elif N.all(y == 0.0):
                # if we find any zero, we just return right away
                return [broadcast_like(0, node.outputs[0], node.env)]
            else:
                new_inputs.append(input)

        if new_inputs != node.inputs:
            if new_inputs:
                if len(new_inputs) == 1:
                    if neg:
                        rval = -new_inputs[0]
                    else:
                        rval = new_inputs[0]
                else:
                    if neg:
                        rval = -T.mul(*new_inputs)
                    else:
                        rval = T.mul(*new_inputs)

                return [broadcast_like(rval, node.outputs[0], node.env)]
            else:
                # there are no variable inputs to mul
                # N.B. this could have been constant-folded...
                if neg:
                    return [broadcast_like(-1, node.outputs[0], node.env)]
                else:
                    return [broadcast_like(1, node.outputs[0], node.env)]

register_specialize(local_mul_specialize)


@gof.local_optimizer([T.add])
def local_add_specialize(node):
    def fill_chain(v):
        out = _fill_chain(v, node.inputs)
        return out

    #here, we are past the point of canonicalization, so we don't want
    #to put in un-necessary fills.
    if node.op == T.add:
        new_inputs = []
        for input in node.inputs:
            try:
                y = get_constant_value(input)
            except TypeError:
                y = input
            if numpy.all(y == 0.0):
                continue
            new_inputs.append(input)

        if len(new_inputs) < len(node.inputs):
            dtype = node.outputs[0].type.dtype
            if len(new_inputs) == 0:
                #we got rid of the entire expression!
                ndim = node.outputs[0].type.ndim
                return fill_chain(
                        T.TensorConstant(
                            T.TensorType(
                                dtype=dtype,
                                broadcastable=[True] * ndim),
                            numpy.zeros((1,) * ndim, dtype=dtype)))

            if len(new_inputs) == 1:
                ret = fill_chain(new_inputs[0])
            else:
                ret = fill_chain(T.add(*new_inputs))
            # The dtype should not be changed. It can happen if the input
            # that was forcing upcasting was equal to 0.
            if ret[0].dtype != dtype:
                ret = [T.cast(ret[0], dtype)]
            return ret
    else:
        return False
register_specialize(local_add_specialize)

# neg_to_mul = out2in(gof.LocalOptGroup(local_neg_to_mul))
# mul_to_neg = out2in(gof.LocalOptGroup(local_mul_to_neg))

mul_canonizer = in2out(gof.LocalOptGroup(local_mul_canonizer, local_fill_cut,
                                         local_fill_sink))


def check_for_x_over_absX(numerators, denominators):
    """Convert x/abs(x) into sign(x). """
    # TODO: this function should dig/search through dimshuffles
    # This won't catch a dimshuffled absolute value
    for den in list(denominators):
        if (den.owner and den.owner.op == T.abs_
            and den.owner.inputs[0] in numerators):
            if den.owner.inputs[0].type.dtype.startswith('complex'):
                #TODO: Make an Op that projects a complex number to
                #      have unit length but projects 0 to 0.  That
                #      would be a weird Op, but consistent with the
                #      special case below.  I heard there's some
                #      convention in Matlab that is similar to
                #      this... but not sure.
                pass
            else:
                denominators.remove(den)
                numerators.remove(den.owner.inputs[0])
                numerators.append(T.sgn(den.owner.inputs[0]))
    return numerators, denominators
local_mul_canonizer.add_simplifier(check_for_x_over_absX, 'X_over_absX')


@register_canonicalize
@gof.local_optimizer([T.abs_])
def local_abs_lift(node):
    """
    move the abs toward the input. This is needed for
    check_for_x_over_absX to apply in more case.

    """
    if node.op == T.abs_ and node.inputs[0].owner:
        assert node.nin == 1
        if node.inputs[0].owner.op == T.mul:
            return [T.mul(*[T.abs_(i) for i in node.inputs[0].owner.inputs])]
        if node.inputs[0].owner.op == T.true_div:
            i = node.inputs[0].owner.inputs
            return [T.true_div(T.abs_(i[0]), T.abs_(i[1]))]


@register_specialize
@gof.local_optimizer([])
def local_abs_merge(node):
    """
    merge abs generated by local_abs_lift when the canonizer don't
    need it anymore

    """
    if node.op == T.mul and sum([i.owner.op == T.abs_ for i in node.inputs
                                 if i.owner]) > 1:
        inputs = []
        for i in node.inputs:
            if i.owner and i.owner.op == T.abs_:
                inputs.append(i.owner.inputs[0])
            else:
                const = get_constant_value(i)
                if not (const >= 0).all():
                    return False
                inputs.append(i)
        return [T.abs_(T.mul(*inputs))]
    if node.op == T.true_div and sum([i.owner.op == T.abs_ for i in
                                      node.inputs if i.owner]) == 2:
        return [T.abs_(T.true_div(node.inputs[0].owner.inputs[0],
                                  node.inputs[1].owner.inputs[0]))]


@register_stabilize
@register_specialize
@gof.local_optimizer([T.log])
def local_log1p(node):
    # log(1+x) -> log1p(x)

    if node.op == T.log:
        log_arg, = node.inputs
        if log_arg.owner and log_arg.owner.op == T.add:
            scalars, scalar_inputs, nonconsts = \
                    scalarconsts_rest(log_arg.owner.inputs)
            # scalar_inputs are potentially dimshuffled and fill'd scalars
            if scalars and numpy.allclose(numpy.sum(scalars), 1):
                if not nonconsts:
                    pass  # leave for constant-merge
                if len(nonconsts) == 1:
                    return _fill_chain(T.log1p(nonconsts[0]), scalar_inputs)
                else:
                    return _fill_chain(T.log1p(T.add(*nonconsts)),
                                       scalar_inputs)


#TODO: in canonicalize, change log10 and log2 -> log
@register_stabilize
@register_specialize
@gof.local_optimizer([T.log])
def local_log_add(node):
    # log(exp(x)+exp(y))
    #
    # Suppose x >= y
    # log(exp(x) + exp(y))
    # log(exp(x) * (1 + exp(y)/exp(x)))
    # x + log(1 + exp(y)/exp(x))
    # x + log1p(exp(y)/exp(x))
    # x + log1p(exp(y-x))
    if node.op == T.log:
        z = node.inputs[0]
        if z.owner and z.owner.op == T.add:
            zi = z.owner.inputs
            pre_exp = [x.owner.inputs[0] for x in zi
                       if x.owner and x.owner.op == T.exp]
            if len(pre_exp) == len(zi):
                # all arguments to add are exp(<something>)
                max_pre = T.maximum(*pre_exp)

                ret = max_pre + T.log1p(T.exp(T.add(*[p - max_pre
                                                      for p in pre_exp])))
                ret.values_eq_approx = ret.type.values_eq_approx_remove_inf
                return [ret]


def add_calculate(num, denum, aslist=False, out_type=None):
    #TODO: make sure that this function and mul_calculate are similar
    if out_type is None:
        zero = 0.0
    else:
        zero = theano._asarray(0, dtype=out_type.dtype)
    #zero = 0.0 if out_type is None else theano._asarray(0,
    #dtype=out_type.dtype)
    v = reduce(N.add, num, zero) - reduce(N.add, denum, zero)
    if aslist:
        if N.all(v == 0):
            return []
        else:
            return [v]
    return v


local_add_canonizer = Canonizer(T.add, T.sub, T.neg, add_calculate)
add_canonizer = in2out(gof.LocalOptGroup(local_add_canonizer, local_fill_cut,
                                         local_fill_sink))


register_canonicalize(local_add_canonizer, name='local_add_canonizer')


##################
# Distributivity #
##################


def distribute_greedy(pos_pairs, neg_pairs, num, denum, minscore=0):
    # each pair in pos_pairs and neg_pairs is a num/denum pair. this
    # function attempts to add num and denum to the corresponding parts
    # of each pair, and counts how many multiplications/divisions can
    # be saved in that way.

    # each division is counted like div_cost multiplications
    # (typically, division costs more so we are willing to multiply more
    # in order to divide less)
    # 1.5 was obtained through an informal test and may very well be
    # platform dependent
    div_cost = 1.5

    # score is number of operations saved, higher is better
    score = len(num) + div_cost * len(denum)
    new_pos_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n + num, d + denum) for (n, d)
                                            in pos_pairs]))
    new_neg_pairs = list(itertools.starmap(local_mul_canonizer.simplify,
                                           [(n + num, d + denum) for (n, d)
                                            in neg_pairs]))
    for (n, d), (nn, dd) in zip(pos_pairs + neg_pairs, new_pos_pairs +
                                new_neg_pairs):
        # We calculate how many operations we are saving with the new
        # num and denum
        score += len(n) + div_cost * len(d) - len(nn) - div_cost * len(dd)
    if score <= minscore:
        # the change is not applied because it adds too many operations
        return False, pos_pairs, neg_pairs
    return True, new_pos_pairs, new_neg_pairs


def attempt_distribution(factor, num, denum):
    # we try to insert each num and each denum in the factor
    # returns: changes?, new_factor, new_num, new_denum
    # if there are changes, new_num and new_denum contain all the numerators
    # and denumerators that could not be distributed in the factor
    pos, neg = local_add_canonizer.get_num_denum(factor)
    if len(pos) == 1 and not neg:
        return False, factor, num, denum
    pos_pairs = map(local_mul_canonizer.get_num_denum, pos)
    neg_pairs = map(local_mul_canonizer.get_num_denum, neg)
    change = False
    for n in list(num):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs,
                                                          neg_pairs, [n], [])
        if success:
            change = True
            num.remove(n)
    for d in list(denum):
        success, pos_pairs, neg_pairs = distribute_greedy(pos_pairs,
                                                          neg_pairs, [], [d])
        if success:
            change = True
            denum.remove(d)
    if not change:
        return change, factor, num, denum
    else:
        return change, local_add_canonizer.merge_num_denum(
            list(itertools.starmap(local_mul_canonizer.merge_num_denum,
                                   pos_pairs)),
            list(itertools.starmap(local_mul_canonizer.merge_num_denum,
                                   neg_pairs))), num, denum


@gof.local_optimizer([T.mul, T.add, T.mul], [T.mul, T.sub, T.mul],
                     [T.mul, T.add, T.true_div], [T.mul, T.sub, T.true_div])
def local_greedy_distributor(node):
    """
    This optimization tries to apply distributivity of multiplication
    to addition in order to reduce the number of multiplications
    and/or divisions that must be done. The algorithm weighs division
    more than multiplication to account for the former's slightly
    greater computational cost.

    The following expressions are simplified:
    1. ((a/x + b/y) * x * y) --> a*y + b*x
    2. ((a/x + b) * x) --> a + b*x

    The following expressions are not simplified:
    3. ((a + b) * x) -/-> a*x + b*x

    This optimization aims to reduce computational cost. It may also
    increase numerical stability, e.g. when x and/or y tend to 0 in
    example 1.
    """

    out = node.outputs[0]
    num, denum = local_mul_canonizer.get_num_denum(out)
    if len(num) == 1 and not denum:
        return False

    new_num, new_denum = [], []

    change = False

    for candidate in list(num):
        if candidate not in num:
            continue
        num.remove(candidate)
        _change, candidate, num, denum = attempt_distribution(candidate,
                                                              num, denum)
        change |= _change
        new_num.append(candidate)

    for candidate in list(denum):
        if candidate not in denum:
            continue
        denum.remove(candidate)
        _change, candidate, denum, num = attempt_distribution(candidate,
                                                              denum, num)
        change |= _change
        new_denum.append(candidate)

    if not change:
        return False

    new_num += num
    new_denum += denum

    rval = local_mul_canonizer.merge_num_denum(new_num, new_denum)

    if not (rval.type == out.type):
        #WHY DOES THIS HAPPEN?
        return False

    return [rval]

register_canonicalize(local_greedy_distributor)
register_stabilize(local_greedy_distributor)


@gof.local_optimizer([None])
def constant_folding(node):
    for input in node.inputs:
        if not isinstance(input, Constant):
            return False
    #condition:  all inputs are constant

    storage_map = dict([(i, [i.data]) for i in node.inputs])
    compute_map = dict([(i, [True]) for i in node.inputs])
    for o in node.outputs:
        storage_map[o] = [None]
        compute_map[o] = [False]

    thunk = node.op.make_thunk(node, storage_map, compute_map,
            no_recycling=[])

    required = thunk()
    assert not required  # a node whose inputs are all provided should always
    # return successfully

    rval = []
    for output in node.outputs:
        assert compute_map[output][0], (output, storage_map[output][0])
        try:
            constant = output.type.Constant
        except AttributeError:
            constant = Constant
        rval.append(constant(output.type, storage_map[output][0]))
    return rval

register_canonicalize(constant_folding, 'fast_compile')
register_stabilize(constant_folding)
register_specialize(constant_folding)


def _is_1(expr):
    """rtype bool. True iff expr is a constant close to 1
    """
    try:
        v = get_constant_value(expr)
        return numpy.allclose(v, 1)
    except TypeError:
        return False


def _is_minus1(expr):
    """rtype bool. True iff expr is a constant close to -1
    """
    try:
        v = get_constant_value(expr)
        return numpy.allclose(v, -1)
    except TypeError:
        return False

#1+erf(x)=>erfc(-x)
local_one_plus_erf = gof.PatternSub((T.add,
                                     dict(pattern='y', constraint=_is_1),
                                     (T.erf, 'x')),
                                    (T.erfc, (T.neg, 'x')),
                                    allow_multiple_clients=True,)
register_canonicalize(local_one_plus_erf, name='local_one_plus_erf')
register_stabilize(local_one_plus_erf, name='local_one_plus_erf')
register_specialize(local_one_plus_erf, name='local_one_plus_erf')

#1-erf(x)=>erfc(x)
local_one_minus_erf = gof.PatternSub((T.sub,
                                     dict(pattern='y', constraint=_is_1),
                                     (T.erf, 'x')),
                                    (T.erfc, 'x'),
                                    allow_multiple_clients=True,)
register_canonicalize(local_one_minus_erf, name='local_one_minus_erf')
register_stabilize(local_one_minus_erf, name='local_one_minus_erf')
register_specialize(local_one_minus_erf, name='local_one_minus_erf')

local_one_minus_erf2 = gof.PatternSub((T.add,
                                      1,
                                      (T.mul, -1, (T.erf, 'x'))),
                                     (T.erfc, 'x'),
                                     allow_multiple_clients=True,
                                     name='local_one_minus_erf2')
register_canonicalize(local_one_minus_erf2)
register_stabilize(local_one_minus_erf2)
register_specialize(local_one_minus_erf2)

#1+(-erf(x))=>erfc(x) This is a different graph then the previous as
#the canonicalize don't work completly
local_one_plus_neg_erf = gof.PatternSub((T.add,
                                     dict(pattern='y', constraint=_is_1),
                                     (T.neg, (T.erf, 'x'))),
                                    (T.erfc, 'x'),
                                    allow_multiple_clients=True,)
register_canonicalize(local_one_plus_neg_erf, name='local_one_plus_neg_erf')
register_stabilize(local_one_plus_neg_erf, name='local_one_plus_neg_erf')
register_specialize(local_one_plus_neg_erf, name='local_one_plus_neg_erf')

#(-1)+erf(x) => -erfc(x) don't need erf(x)+(-1) as the canonicalize
#will put the -1 as the first argument.
local_erf_minus_one = gof.PatternSub((T.add,
                                     dict(pattern='y', constraint=_is_minus1),
                                     (T.erf, 'x')),
                                    (T.neg, (T.erfc, 'x')),
                                    allow_multiple_clients=True,)
register_canonicalize(local_erf_minus_one, name='local_erf_minus_one')
register_stabilize(local_erf_minus_one, name='local_erf_minus_one')
register_specialize(local_erf_minus_one, name='local_erf_minus_one')

#1-erfc(x) => erf(x)
local_one_minus_erfc = gof.PatternSub((T.sub,
                                     dict(pattern='y', constraint=_is_1),
                                     (T.erfc, 'x')),
                                    (T.erf, 'x'),
                                    allow_multiple_clients=True,)
register_canonicalize(local_one_minus_erfc, name='local_one_minus_erfc')
register_stabilize(local_one_minus_erfc, name='local_one_minus_erfc')
register_specialize(local_one_minus_erfc, name='local_one_minus_erfc')

local_one_minus_erfc2 = gof.PatternSub((T.add,
                                        1,
                                        (T.neg, (T.erfc, 'x'))),
                                       (T.erf, 'x'),
                                       allow_multiple_clients=True,
                                       name='local_one_minus_erfc2')
register_canonicalize(local_one_minus_erfc2)
register_stabilize(local_one_minus_erfc2)
register_specialize(local_one_minus_erfc2)

local_one_minus_erfc3 = gof.PatternSub((T.add,
                                        1,
                                        (T.mul, -1, (T.erfc, 'x'))),
                                       (T.erf, 'x'),
                                       allow_multiple_clients=True,
                                       name='local_one_minus_erfc3')
register_canonicalize(local_one_minus_erfc3)
register_stabilize(local_one_minus_erfc3)
register_specialize(local_one_minus_erfc3)

#1+(-erfc(x)) => erf(x) This is a different graph then the previous as
#the canonicalize don't work completly
local_one_add_neg_erfc = gof.PatternSub((T.add,
                                     dict(pattern='y', constraint=_is_1),
                                     (T.neg, (T.erfc, 'x'))),
                                    (T.erf, 'x'),
                                    allow_multiple_clients=True,)
register_canonicalize(local_one_add_neg_erfc, name='local_one_add_neg_erfc')
register_stabilize(local_one_add_neg_erfc, name='local_one_add_neg_erfc')
register_specialize(local_one_add_neg_erfc, name='local_one_add_neg_erfc')

#(-1)+erfc(-x)=>erf(x)
local_erf_neg_minus_one = gof.PatternSub((T.add,
                                     dict(pattern='y', constraint=_is_minus1),
                                     (T.erfc, (T.neg, 'x'))),
                                    (T.erf, 'x'),
                                    allow_multiple_clients=True,)
register_canonicalize(local_erf_neg_minus_one, name='local_erf_neg_minus_one')
register_stabilize(local_erf_neg_minus_one, name='local_erf_neg_minus_one')
register_specialize(local_erf_neg_minus_one, name='local_erf_neg_minus_one')

#(-1)+erfc(-1*x)=>erf(x)
local_erf_neg_minus_one2 = gof.PatternSub((T.add,
                                     dict(pattern='y', constraint=_is_minus1),
                                     (T.erfc, (T.mul, -1, 'x'))),
                                    (T.erf, 'x'),
                                    allow_multiple_clients=True,
                                          name='local_erf_neg_minus_one2')
register_canonicalize(local_erf_neg_minus_one2)
register_stabilize(local_erf_neg_minus_one2)
register_specialize(local_erf_neg_minus_one2)


#Stability optimization
#log(erfc(x)) => when x>threashold,
#              -x**2-log(x)-.5*log(pi)+log(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))
#for float64: threshold=26.641747557 was choosed with:
#  [(i,numpy.log(scipy.special.erfc(numpy.asarray([i],dtype='float64'))))
#   for i in numpy.arange(26.641747557,26.6417475571,.00000000001)]
#for float32: threshold=10.0541949, [(i,numpy.log(scipy.special.erfc(
#        numpy.asarray([i],dtype='float32')))) for i in numpy.arange(
#        10.0541948,10.0541951,.0000001)]
@register_stabilize
@register_specialize
@gof.local_optimizer([T.log])
def local_log_erfc(node):
    if node.op != T.log:
        return False
    if not node.inputs[0].owner or node.inputs[0].owner.op != T.erfc:
        return False

    if hasattr(node.tag, 'local_log_erfc_applied'):
        #We use that flag to don't apply the optimization recursively
        return False
    node.tag.local_log_erfc_applied = True

    x = node.inputs[0].owner.inputs[0]
    stab_value = (-x ** 2 - T.log(x) - .5 * T.log(numpy.pi) +
                   T.log(1 - 1 / (2 * x ** 2) + 3 / (4 * x ** 4)
                         - 15 / (8 * x ** 6)))

    if node.outputs[0].dtype == 'float32':
        threshold = 10.0541949
    elif node.outputs[0].dtype == 'float64':
        threshold = 26.641747557

    ret = T.switch(x < threshold, node.outputs[0], stab_value)
    ret.values_eq_approx = ret.type.values_eq_approx_remove_inf
    return [ret]


#Stability optimization of the grad of log(erfc(x))
#([y*]exp(-(x**2)))/erfc(x) # The y* is optional
#([y*]exp(x**2))/erfc(-x) => [y*](when x>threashold,
#                            sqrt(pi)*-x/(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6)))
#for float64: threshold=26.63 see at the end of the fct for the explaination
#for float32: threshold=9.3 see at the end of the fct for the explaination
#TODO: remove the contraint that there are only 2 inputs to mul and exp(x**2)
#      is the second.
#TODO: at the test point 10 in float32, there is instability in the original
#      value. The original gives -30.0, the stab -20.1 and in float64 -18.1.
#      Make it so that the test does not generate an error in that case!
@register_stabilize
@register_specialize
@gof.local_optimizer([T.true_div])
def local_grad_log_erfc_neg(node):
    if node.op != T.true_div:
        return False
    if not node.inputs[1].owner or node.inputs[1].owner.op != T.erfc:
        return False
    erfc = node.inputs[1]
    erfc_x = erfc.owner.inputs[0]
    if not node.inputs[0].owner:
        return False

    #The mul is optional.
    if node.inputs[0].owner.op != T.mul:
        mul = None
        y = 1
        if not node.inputs[0].owner or node.inputs[0].owner.op != T.exp:
            return False
        exp = node.inputs[0]
    else:
        mul = node.inputs[0]
        if mul.owner.inputs[0].owner or len(mul.owner.inputs) != 2:
            return False
        y = mul.owner.inputs[0]
        if (not mul.owner.inputs[1].owner
            or mul.owner.inputs[1].owner.op != T.exp):
            return False
        exp = mul.owner.inputs[1]

    if not exp.owner.inputs[0].owner:
        return False

    if exp.owner.inputs[0].owner.op == T.neg:
        neg = exp.owner.inputs[0]
        if (not neg.owner.inputs[0].owner
            or neg.owner.inputs[0].owner.op != T.sqr):
            return False
        sqr = neg.owner.inputs[0]
        x = sqr.owner.inputs[0]
    elif exp.owner.inputs[0].owner.op == T.mul:
        # We should compare that -(erfc_x**2) is equivalent to mul_neg.
        # There is currently no easy way to do this in the general case,
        # so we implement some common case for now.

        # In many cases the neg are replaced by mul in the graph.
        # This also allows to stabilize log(erfc(cst*x)).
        mul_neg = exp.owner.inputs[0]

        # In case that multiple mul are not fused together, we do it here.
        def check_input(inputs):
            new_inputs = []
            for i in inputs:
                if i.owner and i.owner.op == T.mul:
                    new_inputs.extend(check_input(i.owner.inputs))
                else:
                    new_inputs.append(i)
            return new_inputs
        mul_inputs = check_input(mul_neg.owner.inputs)

        # Put the constant first.
        for i in xrange(len(mul_inputs)):
            if isinstance(i, Constant):
                if i == 0:
                    break
                else:
                    tmp = mul_inputs[0]
                    mul_inputs[0] = mul_inputs[i]
                    mul_inputs[i] = tmp
                    break
        mul_neg = T.mul(*mul_inputs)

        try:
            cst2 = get_constant_value(mul_neg.owner.inputs[0])
        except TypeError:
            return False

        if len(mul_neg.owner.inputs) == 2:
            if (not mul_neg.owner.inputs[1].owner
                or mul_neg.owner.inputs[1].owner.op != T.sqr):
                return False
            sqr = mul_neg.owner.inputs[1]
            x = sqr.owner.inputs[0]
        elif len(mul_neg.owner.inputs) == 3:
            if mul_neg.owner.inputs[1] is not mul_neg.owner.inputs[2]:
                return False
            x = mul_neg.owner.inputs[1]
        else:
            return False

        if cst2 != -1:
            if (not erfc_x.owner or erfc_x.owner.op != T.mul
                or len(erfc_x.owner.inputs) != 2):
                #todo implement that case
                return False
            if erfc_x.owner.inputs[1] is not mul_neg.owner.inputs[1]:
                return False

            x = erfc_x
            try:
                cst = get_constant_value(erfc_x.owner.inputs[0])
            except TypeError:
                return False
            if cst2 != -cst * 2:
                return False

            #The constant is valid. Must check that the
        elif erfc_x is not x:
            return False

    else:
        return False

    if hasattr(node.tag, 'local_grad_log_erfc_neg'):
        #We use that flag to don't apply the optimization recursively
        return False

    #we move the y outside the div.
    true_div_no_mul = T.true_div(exp, erfc)
    true_div_no_mul.owner.tag.local_grad_log_erfc_neg = True

    #aaron value
    stab_value = (x * T.pow(1 - 1 / (2 * (x ** 2)) +
                            3 / (4 * (x ** 4)) - 15 / (8 * (x ** 6)), -1)
                  * T.cast(T.sqrt(numpy.pi), dtype=x.dtype))

    if x.dtype == 'float32':
        threshold = 9.3
        #threshold = 10.1
    elif x.dtype == 'float64':
        threshold = 26.641747557
    ret = T.switch(x < threshold, true_div_no_mul, stab_value) * y
    ret.values_eq_approx = ret.type.values_eq_approx_remove_inf_nan

    return [ret]
    """
The libm used for the test is amdlibm
    #([y*]exp(-(x**2)))/erfc(x) # The mul is optional
#exp(x**2)/erfc(-x) => when x>threashold,
#-x*(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))*sqrt(pi) for float64:
#threshold=26.63 see below for float32: threshold=9.3 see below TODO
#remove the contraint that there are only 2 inputs to mul TODO: should
#we cast numpy.pi to x.dtype?

#float32 threshold 9.3 as the approximation is more precise at that
#point and more stable.
import numpy, scipy.special
r = numpy.arange(9,10.06,.01)

p64=[(numpy.exp(-(x**2)))/scipy.special.erfc(x) for x in r]
p32=[(numpy.exp(-(x**2)))/scipy.special.erfc(x) for x in
numpy.asarray(r,dtype='float32')]
a64=[x*((1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))**(-1))*numpy.sqrt(numpy.pi)
for x in r]
a32=[x*((1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))**(-1))
     * numpy.float32(numpy.sqrt(numpy.pi))
for x in numpy.asarray(r,dtype='float32')] for idx,(a,b,c,d,e) in
enumerate(zip(r,p64,p32,a64,a32)):print
a,b,c,d,e,c-b,e-b,numpy.absolute(c-b)<numpy.absolute(e-b)

#, show that the value don't look stable at some point before inf.
for i in xrange(1,len(p32)): print r[i], p32[i]-p32[i-1]

#float64 threshold is 26.63 the approx seam more precise at that
point.  r = numpy.arange(26.2,26.7,.001)
#scipy.special.erfc(numpy.float128(x)) don't work
#p128=[(numpy.exp(-(x**2)))/scipy.special.erfc(x)for x in
numpy.float128(r)] #those value have been computed with g++
theano/misc/erfc_stability_threshold.c && ./a.out
p128=numpy.float128(['46.47206725', '46.47383842', '46.47560959',
'46.47738076', '46.47915193', '46.48092309', '46.48269426',
'46.48446543', '46.48623660', '46.48800777', '46.48977894',
'46.49155011', '46.49332128', '46.49509245', '46.49686362',
'46.49863479', '46.50040596', '46.50217713', '46.50394830',
'46.50571947', '46.50749064', '46.50926181', '46.51103298',
'46.51280415', '46.51457532', '46.51634649', '46.51811766',
'46.51988883', '46.52166000', '46.52343118', '46.52520235',
'46.52697352', '46.52874469', '46.53051586', '46.53228703',
'46.53405820', '46.53582938', '46.53760055', '46.53937172',
'46.54114289', '46.54291407', '46.54468524', '46.54645641',
'46.54822758', '46.54999876', '46.55176993', '46.55354110',
'46.55531227', '46.55708345', '46.55885462', '46.56062579',
'46.56239697', '46.56416814', '46.56593931', '46.56771049',
'46.56948166', '46.57125283', '46.57302401', '46.57479518',
'46.57656636', '46.57833753', '46.58010871', '46.58187988',
'46.58365105', '46.58542223', '46.58719340', '46.58896458',
'46.59073575', '46.59250693', '46.59427810', '46.59604928',
'46.59782045', '46.59959163', '46.60136280', '46.60313398',
'46.60490516', '46.60667633', '46.60844751', '46.61021868',
'46.61198986', '46.61376104', '46.61553221', '46.61730339',
'46.61907456', '46.62084574', '46.62261692', '46.62438809',
'46.62615927', '46.62793045', '46.62970163', '46.63147280',
'46.63324398', '46.63501516', '46.63678633', '46.63855751',
'46.64032869', '46.64209987', '46.64387104', '46.64564222',
'46.64741340', '46.64918458', '46.65095576', '46.65272693',
'46.65449811', '46.65626929', '46.65804047', '46.65981165',
'46.66158283', '46.66335401', '46.66512519', '46.66689636',
'46.66866754', '46.67043872', '46.67220990', '46.67398108',
'46.67575226', '46.67752344', '46.67929462', '46.68106580',
'46.68283698', '46.68460816', '46.68637934', '46.68815052',
'46.68992170', '46.69169288', '46.69346406', '46.69523524',
'46.69700642', '46.69877760', '46.70054878', '46.70231997',
'46.70409115', '46.70586233', '46.70763351', '46.70940469',
'46.71117587', '46.71294705', '46.71471824', '46.71648942',
'46.71826060', '46.72003178', '46.72180296', '46.72357414',
'46.72534533', '46.72711651', '46.72888769', '46.73065887',
'46.73243006', '46.73420124', '46.73597242', '46.73774361',
'46.73951479', '46.74128597', '46.74305715', '46.74482834',
'46.74659952', '46.74837070', '46.75014189', '46.75191307',
'46.75368426', '46.75545544', '46.75722662', '46.75899781',
'46.76076899', '46.76254018', '46.76431136', '46.76608254',
'46.76785373', '46.76962491', '46.77139610', '46.77316728',
'46.77493847', '46.77670965', '46.77848084', '46.78025202',
'46.78202321', '46.78379439', '46.78556558', '46.78733677',
'46.78910795', '46.79087914', '46.79265032', '46.79442151',
'46.79619269', '46.79796388', '46.79973507', '46.80150625',
'46.80327744', '46.80504863', '46.80681981', '46.80859100',
'46.81036219', '46.81213337', '46.81390456', '46.81567575',
'46.81744693', '46.81921812', '46.82098931', '46.82276050',
'46.82453168', '46.82630287', '46.82807406', '46.82984525',
'46.83161644', '46.83338762', '46.83515881', '46.83693000',
'46.83870119', '46.84047238', '46.84224357', '46.84401475',
'46.84578594', '46.84755713', '46.84932832', '46.85109951',
'46.85287070', '46.85464189', '46.85641308', '46.85818427',
'46.85995546', '46.86172665', '46.86349784', '46.86526903',
'46.86704022', '46.86881141', '46.87058260', '46.87235379',
'46.87412498', '46.87589617', '46.87766736', '46.87943855',
'46.88120974', '46.88298093', '46.88475212', '46.88652331',
'46.88829450', '46.89006569', '46.89183688', '46.89360807',
'46.89537927', '46.89715046', '46.89892165', '46.90069284',
'46.90246403', '46.90423522', '46.90600642', '46.90777761',
'46.90954880', '46.91131999', '46.91309119', '46.91486238',
'46.91663357', '46.91840476', '46.92017596', '46.92194715',
'46.92371834', '46.92548953', '46.92726073', '46.92903192',
'46.93080311', '46.93257431', '46.93434550', '46.93611669',
'46.93788789', '46.93965908', '46.94143028', '46.94320147',
'46.94497266', '46.94674386', '46.94851505', '46.95028625',
'46.95205744', '46.95382864', '46.95559983', '46.95737103',
'46.95914222', '46.96091341', '46.96268461', '46.96445581',
'46.96622700', '46.96799820', '46.96976939', '46.97154059',
'46.97331178', '46.97508298', '46.97685417', '46.97862537',
'46.98039657', '46.98216776', '46.98393896', '46.98571015',
'46.98748135', '46.98925255', '46.99102374', '46.99279494',
'46.99456614', '46.99633733', '46.99810853', '46.99987973',
'47.00165092', '47.00342212', '47.00519332', '47.00696452',
'47.00873571', '47.01050691', '47.01227811', '47.01404931',
'47.01582050', '47.01759170', '47.01936290', '47.02113410',
'47.02290530', '47.02467649', '47.02644769', '47.02821889',
'47.02999009', '47.03176129', '47.03353249', '47.03530369',
'47.03707489', '47.03884608', '47.04061728', '47.04238848',
'47.04415968', '47.04593088', '47.04770208', '47.04947328',
'47.05124448', '47.05301568', '47.05478688', '47.05655808',
'47.05832928', '47.06010048', '47.06187168', '47.06364288',
'47.06541408', '47.06718528', '47.06895648', '47.07072768',
'47.07249888', '47.07427009', '47.07604129', '47.', '47.07958369',
'47.08135489', '47.08312609', '47.08489729', '47.08666850',
'47.08843970', '47.09021090', '47.09198210', '47.09375330',
'47.09552450', '47.09729571', '47.09906691', '47.10083811',
'47.10260931', '47.10438052', '47.10615172', '47.10792292',
'47.10969412', '47.11146533', '47.11323653', '47.11500773',
'47.11677894', '47.11855014', '47.12032134', '47.12209255',
'47.12386375', '47.12563495', '47.12740616', '47.12917736',
'47.13094857', '47.13271977', '47.13449097', '47.13626218',
'47.13803338', '47.13980459', '47.14157579', '47.14334700',
'47.14511820', '47.14688941', '47.14866061', '47.15043182',
'47.15220302', '47.15397423', '47.15574543', '47.15751664',
'47.15928784', '47.16105905', '47.16283025', '47.16460146',
'47.16637266', '47.16814387', '47.16991508', '47.17168628',
'47.17345749', '47.17522869', '47.17699990', '47.17877111',
'47.18054231', '47.18231352', '47.18408473', '47.18585593',
'47.18762714', '47.18939835', '47.19116956', '47.19294076',
'47.19471197', '47.19648318', '47.19825439', '47.20002559',
'47.20179680', '47.20356801', '47.20533922', '47.20711042',
'47.20888163', '47.21065284', '47.21242405', '47.21419526',
'47.21596647', '47.21773767', '47.21950888', '47.22128009',
'47.22305130', '47.22482251', '47.22659372', '47.22836493',
'47.23013614', '47.23190735', '47.23367855', '47.23544976',
'47.23722097', '47.23899218', '47.24076339', '47.24253460',
'47.24430581', '47.24607702', '47.24784823', '47.24961944',
'47.25139065', '47.25316186', '47.25493307', '47.25670429',
'47.25847550', '47.26024671', '47.26201792', '47.26378913',
'47.26556034', '47.26733155', '47.26910276', '47.27087397',
'47.27264518', '47.27441640', '47.27618761', '47.27795882',
'47.27973003', '47.28150124', '47.28327246', '47.28504367',
'47.28681488', '47.28858609', '47.29035730', '47.29212852',
'47.29389973', '47.29567094', '47.29744215', '47.29921337',
'47.30098458', '47.30275579', '47.30452701', '47.30629822',
'47.30806943', '47.30984065', '47.31161186', '47.31338307',
'47.31515429', '47.31692550', '47.31869671', '47.32046793',
'47.32223914', '47.32401036', '47.32578157', '47.32755278',
'47.32932400', '47.33109521', '47.33286643', '47.33463764',
'47.33640886', '47.33818007', '47.33995129', '47.34172250',
'47.34349372', '47.34526493', '47.34703615', '47.34880736',
'47.35057858', '47.35234979', '47.35412101', '47.35589223'])
p64=[(numpy.exp(-(x**2)))/scipy.special.erfc(x)for x in r]
a128=[x*((1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))**(-1))
      *numpy.float128(numpy.sqrt(numpy.pi))
      for x in numpy.asarray(r,dtype='float128')]
a64=[x*((1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6)+63/(7*x**8))**(-1))
     *numpy.sqrt(numpy.pi)
     for x in r] for a,b,c,d in zip(r,p128,p64,a64):print a,b,c,d,c-b,d-b

for i in xrange(1,len(p64)): print i, 64[i]-p64[i-1]
   """


# ###############
# # Loop fusion #
# ###############
def local_elemwise_fusion_op(OP, max_input_fct=lambda node: 1024):
    """
    We parametrize it to make it work for Elemwise and GpuElemwise op.

    :param OP: GpuElemwise or Elemwise class (the one that we want to fuse)

    :param max_input_fct: a function that returns the maximum number of inputs
                          that this elemwise can take (useful for GpuElemwise).
                          GPU kernel currently has a limit of 256 bytes for
                          the size of all parameters passed to it. As currently
                          we pass many information only by parameter, we must
                          limit how many ops we fuse together to avoid busting
                          that 256 limit.

                          On the CPU we limit to 1024 input variable
                          to the resulting fused op. This is big
                          enough that if we hit it, I'm not sure it
                          will affect performance.
    """
    def local_fuse(node):
        """
        As part of specialization, we fuse two consecutive elemwise Ops of the
        same shape.

        For mixed dtype, we let the Composite op do the cast. It lets the C
        compiler do the cast.
        The number of dimensions is validated at call time by theano itself.
        """
        # META TODO:  PUT THESE THINGS IN TRAC, NOT TODO NOTES!!
        # TODO: use broadcast flag?

        # TODO: don't do this optimization as a localOptimizer.
        # Analyze the graph in terms of elemwise subgraphs, and then
        # replace each subgraph with a Composite version.

        # TODO: use malloc and copy to transfer arguments that don't
        # fit within the parameter space of 256 bytes
        #
        # TODO: Merge with multiple output to merge when an inputs
        # have multiple clients. This can't be done with a local
        # optimiser.

        # TODO: Related: Support composites with multiple outputs

        # TODO: Use Composite to combine Elemwise and Reduce
        # operations.  We have to loop over the data anyway... might
        # as well sum it up while we're at it (this can be trickier
        # than i'm making it seound here. The data-traversal should be
        # done contiguously, and the summing-up might not be easy or
        # worthwhile if the summation axis doesn't line up with a
        # contiguous dimension)

        if not isinstance(node.op, OP):
            return False
        inputs = []  # inputs of the new Elemwise op.
        s_inputs = []  # inputs of the new scalar op used by the Composite.
        # Inputs of the new scalar op that represents the current node.
        s_g = []

        # There is a hard limit of 256 bytes for the formal argument list to a
        # GPU kernel function.
        max_nb_input = max_input_fct(node)
        # The number of inputs to the new fused op if we do not fuse more
        # inputs.
        new_nb_input = len(node.inputs)
        # Did we fuse something?
        # Needed as we can fuse unary op that don't change the number of
        # inputs.
        # And there is a case where the inputs are the same as the current
        # node. That won't change the number of inputs of the new op.
        fused = False

        for i in node.inputs:
            do_fusion = False
            catch = False
            # Will store inputs of the fused node that are not currently inputs
            # of the node we want to create (to avoid duplicating inputs).
            tmp_input = []
            # Same as tmp_input, but for scalars.
            tmp_scalar = []

            # We should not check the number of inputs here
            # As fusing op don't always change the number of input.
            if (i.owner and
                isinstance(i.owner.op, OP) and
                len(i.clients) == 1):

                do_fusion = True
                try:
                    tmp_s_input = []
                    #we should not put duplicate input into s_inputs and inputs
                    for ii in i.owner.inputs:
                        if ii in inputs:
                            tmp_s_input.append(s_inputs[inputs.index(ii)])
                        elif ii in tmp_input:
                            tmp_s_input.append(tmp_scalar[tmp_input.index(ii)])
                        else:
                            tmp_s_input.append(scalar.Scalar(
                                    ii.dtype).make_variable())
                            tmp_input.append(ii)
                            tmp_scalar.append(tmp_s_input[-1])
                    s_op = i.owner.op.scalar_op(*tmp_s_input)

                    #if the scalar_op don't have a c implementation,
                    #we skip its fusion to allow the fusion of the
                    #other ops.
                    i.owner.op.scalar_op.c_code(s_op.owner,
                                                "test_presence_of_c_code",
                                                ["x" for x in i.owner.inputs],
                                                "z", {})
                except MethodNotDefined:
                    catch = True
                except NotImplementedError:
                    catch = True
                if catch:
                    _logger.info(("%s does not implement the c_code function."
                                  " As well as being potentially slow, this"
                                  " disables loop fusion of this op.") %
                                 str(i.owner.op.scalar_op))
                    do_fusion = False

            # Compute the number of inputs in case we fuse this input.
            # We subtract 1 because we replace the existing input with the new
            # inputs from `tmp_input`.
            new_nb_input_ = new_nb_input + len(tmp_input) - 1

            # If the new input is already an input of the current node, it was
            # already counted when `new_nb_input` was initialized to
            # len(node.inputs).
            # This can happen when a variable is used both by the Elemwise to
            # fuse and the current node.
            for x in tmp_input:
                if x in node.inputs:
                    new_nb_input_ -= 1

            if do_fusion and (new_nb_input_ <= max_nb_input):
                fused = True
                new_nb_input = new_nb_input_
                inputs.extend(tmp_input)
                s_inputs.extend(tmp_scalar)
                s_g.append(s_op)
            else:
                # We must support the case where the same variable appear many
                # time in the inputs
                if inputs.count(i) == node.inputs.count(i):
                    s = s_inputs[inputs.index(i)]
                else:
                    s = scalar.Scalar(i.dtype).make_variable()
                    inputs.append(i)
                    s_inputs.append(s)
                s_g.append(s)

        if not fused:
            return False

        if new_nb_input != len(inputs) or len(s_inputs) != len(inputs):
            raise Exception("""Something has gone wrong with the elemwise
fusion optimization. We skip this optimization. You can ignore this message,
your code will run correctly, but may be slower.""")

        otype = node.outputs[0].type
        s_new_out = node.op.scalar_op(*s_g)
        try:
            s_new_out.owner.op.c_code(s_new_out.owner,
                                      "test_presence_of_c_code",
                                      ["x" for x in s_g],
                                      "z", {})
        except MethodNotDefined:
            _logger.info(("%s does not implement the c_code function."
                          " As well as being potentially slow, this disables "
                          "loop fusion of this op.") % str(s_new_out.owner.op))
            return False
        except NotImplementedError:
            _logger.info(("%s does not implement the c_code function. As well"
                          " as being potentially slow, this disables loop"
                          " fusion of this op.") % str(s_new_out.owner.op))
            return False

        #create the composite op.
        C = scalar.Composite(s_inputs, [s_new_out])

        #create the new node.
        n = OP(C).make_node(*inputs)
        assert len(n.outputs) == 1
        assert node.outputs[0].dtype == n.outputs[0].dtype

        if len(n.inputs) > max_nb_input:
            _logger.info('loop fusion failed because Op would exceed'
                         ' kernel argument limit.')
            return False

        #we fuse as many that we can at the same time to make debug mode faster
        #debug mode will be faster as it won't test all intermediate step.
        while True:
            ret = local_fuse(n)
            if ret is not False and ret is not None:
                #print n,ret
                assert len(ret) == len(n.outputs)
                assert len(ret) == 1
                n = ret[0].owner
            else:
                break

        return n.outputs
    return local_fuse

local_elemwise_fusion = local_elemwise_fusion_op(T.Elemwise)


class FusionOptimizer(Optimizer):
    """Graph optimizer for Fusion of elemwise operations"""
    def __init__(self, local_optimizer):
        Optimizer.__init__(self)
        self.optimizer = local_optimizer

    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())
        env.extend(DestroyHandler())

    def apply(self, env):
        did_something = True
        while did_something:
            nodelist = list(env.toposort())
            nodelist.reverse()
            did_something = False
            for node in nodelist:
                # Don't try to fuse node that have already been fused.
                if node in env.nodes:
                    new_outputs = self.optimizer(node)
                    if new_outputs:
                        assert len(new_outputs) == len(node.outputs)
                        try:
                            env.replace_all_validate(
                                zip(node.outputs, new_outputs),
                                reason=self.__class__.__name__)
                            did_something = True
                        except InconsistencyError, e:
                            pass

if config.tensor.local_elemwise_fusion:
    _logger.debug("enabling optimization fusion elemwise in fast_run")
    compile.optdb.register('elemwise_fusion',
                           FusionOptimizer(local_elemwise_fusion), 71.00,
                           'fast_run', 'fusion', 'local_elemwise_fusion')
else:
    _logger.debug("not enabling optimization fusion elemwise in fast_run")
    compile.optdb.register('elemwise_fusion',
                           FusionOptimizer(local_elemwise_fusion), 71.00,
                           'fusion', 'local_elemwise_fusion')
