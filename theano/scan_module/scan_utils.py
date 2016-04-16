"""
This module provides utility functions for the Scan Op.

See scan.py for details on scan.

"""
from __future__ import absolute_import, print_function, division
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Frederic Bastien "
               "James Bergstra "
               "Pascal Lamblin "
               "Arnaud Bergeron")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import copy
import logging
import warnings

import numpy

import theano
from theano.compat import izip
from six import string_types, iteritems
from six.moves import xrange
from theano.compile.pfunc import rebuild_collect_shared
from theano import gof, compat
from theano import tensor, scalar
from theano.compat import OrderedDict
from theano.tensor.basic import get_scalar_constant_value


################ Utility Functions and Classes #######################

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_utils')


def safe_new(x, tag='', dtype=None):
    """
    Internal function that constructs a new variable from x with the same
    type, but with a different name (old name + tag). This function is used
    by gradient, or the R-op to construct new variables for the inputs of
    the inner graph such that there is no interference between the original
    graph and the newly constructed graph.

    """
    if hasattr(x, 'name') and x.name is not None:
        nw_name = x.name + tag
    else:
        nw_name = None

    if isinstance(x, theano.Constant):
        if dtype and x.dtype != dtype:
            casted_x = x.astype(dtype)
            nwx = x.__class__(casted_x.type, x.data, x.name)
            nwx.tag = copy(x.tag)
            return nwx
        else:
            return x.clone()
    # Note, as_tensor_variable will convert the Scalar into a
    # TensorScalar that will require a ScalarFromTensor op,
    # making the pushout optimization fail
    elif isinstance(x, scalar.ScalarVariable):
        if dtype:
            nw_x = scalar.get_scalar_type(dtype=dtype)()
        else:
            nw_x = x.type()
        nw_x.name = nw_name
        if theano.config.compute_test_value != 'off':
            # Copy test value, cast it if necessary
            try:
                x_test_value = gof.op.get_test_value(x)
            except AttributeError:
                # There is no test value
                pass
            else:
                # This clause is executed if no exception was raised
                nw_x.tag.test_value = nw_x.type.filter(x_test_value)
        return nw_x
    else:
        try:
            x = tensor.as_tensor_variable(x)
        except TypeError:
            # This could happen for example for random states, and I really
            # want to avoid the convoluted logic that checks for cuda
            # ndarrays
            pass

    # Cast x if needed. If x has a test value, this will also cast it.
    if dtype and x.dtype != dtype:
        x = x.astype(dtype)

    nw_x = x.type()
    nw_x.name = nw_name
    # Preserve test values so that the 'compute_test_value' option can be used.
    # The test value is deep-copied to ensure there can be no interactions
    # between test values, due to inplace operations for instance. This may
    # not be the most efficient memory-wise, though.
    if theano.config.compute_test_value != 'off':
        try:
            nw_x.tag.test_value = copy.deepcopy(gof.op.get_test_value(x))
        except AttributeError:
            # This means `x` has no test value.
            pass

    return nw_x


class until(object):
    """
    Class used to encode the different things the inner function of scan can
    (or needs) to return.

    This class has to be used when scan needs to halt when a condition is
    met, otherwise the list of outputs and dictionary can directly be return
    as a tuple. The reason is that otherwise scan has no way to distinguish
    between the condition and the list of outputs ( unless we enforce and
    order, but since this was not impose up to know it can make quite a bit
    of code to fail).

    """

    def __init__(self, condition):
        self.condition = tensor.as_tensor_variable(condition)
        assert self.condition.ndim == 0


def traverse(out, x, x_copy, d, visited=None):
    """
    Function used by scan to parse the tree and figure out which nodes
    it needs to replace.

    There are two options :
        1) x and x_copy or on host, then you would replace x with x_copy
        2) x is on gpu, x_copy on host, then you need to replace
        host_from_gpu(x) with x_copy
    This happens because initially shared variables are on GPU... which is
    fine for the main computational graph but confuses things a bit for the
    inner graph of scan.

    """
    # ``visited`` is a set of nodes that are already known and don't need to be
    # checked again, speeding up the traversal of multiply-connected graphs.
    # if a ``visited`` set is given, it will be updated in-place so the callee
    # knows which nodes we have seen.
    if visited is None:
        visited = set()
    if out in visited:
        return d
    visited.add(out)
    from theano.sandbox import cuda, gpuarray
    if out == x:
        if isinstance(x.type, cuda.CudaNdarrayType):
            d[out] = cuda.gpu_from_host(x_copy)
        else:
            assert isinstance(x.type, gpuarray.GpuArrayType)
            d[out] = gpuarray.GpuFromHost(x.type.context_name)(x_copy)
        return d
    elif out.owner is None:
        return d
    elif (cuda.cuda_available and
          out.owner.op == cuda.host_from_gpu and
          out.owner.inputs == [x]):
        d[out] = tensor.as_tensor_variable(x_copy)
        return d
    elif (gpuarray.pygpu_activated and
          out.owner.op == gpuarray.host_from_gpu and
          out.owner.inputs == [x]):
        d[out] = tensor.as_tensor_variable(x_copy)
        return d
    else:
        for inp in out.owner.inputs:
            d = traverse(inp, x, x_copy, d, visited)
        return d


# Hashing a dictionary/list/tuple by xoring the hash of each element
def hash_listsDictsTuples(x):
    hash_value = 0
    if isinstance(x, dict):
        for k, v in iteritems(x):
            hash_value ^= hash_listsDictsTuples(k)
            hash_value ^= hash_listsDictsTuples(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            hash_value ^= hash_listsDictsTuples(v)
    else:
        hash_value ^= hash(x)
    return hash_value


DEPRECATED_ARG = object()


def clone(output,
          replace=None,
          strict=True,
          share_inputs=True,
          copy_inputs=DEPRECATED_ARG):
    """
    Function that allows replacing subgraphs of a computational graph.

    It returns a copy of the initial subgraph with the corresponding
    substitutions.

    Parameters
    ----------
    output : Theano Variables (or Theano expressions)
        Theano expression that represents the computational graph.
    replace : dict
        Dictionary describing which subgraphs should be replaced by what.
    share_inputs : bool
        If True, use the same inputs (and shared variables) as the original
        graph. If False, clone them. Note that cloned shared variables still
        use the same underlying storage, so they will always have the same
        value.
    copy_inputs
        Deprecated, use share_inputs.

    """
    if copy_inputs is not DEPRECATED_ARG:
        warnings.warn('In `clone()` function, the argument `copy_inputs` has been deprecated and renamed into `share_inputs`')
        assert share_inputs  # since we used `copy_inputs` we should have default value for `share_inputs`
        share_inputs = copy_inputs

    if isinstance(replace, dict):
        items = list(replace.items())
    elif isinstance(replace, (list, tuple)):
        items = replace
    elif replace is None:
        items = []
    else:
        raise ValueError(("replace is neither a dictionary, list, "
                          "tuple or None ! The value provided is %s,"
                          "of type %s")%(str(replace), str(type(replace))))
    tmp_replace = [(x, x.type()) for x, y in items]
    new_replace = [(x, y) for ((_, x), (_, y)) in zip(tmp_replace,
                                                           items)]
    _, _outs, _ = rebuild_collect_shared(output,
                                         [],
                                         tmp_replace,
                                         [],
                                         strict,
                                         share_inputs)

    # TODO Explain why we call it twice ?!
    _, outs, _ = rebuild_collect_shared(_outs,
                                        [],
                                        new_replace,
                                        [],
                                        strict,
                                        share_inputs)

    return outs


def map_variables(replacer, graphs, additional_inputs=[]):
    """Construct new graphs based on 'graphs' with some variables replaced
    according to 'replacer'.

    :param replacer: function that takes a variable and returns its
         replacement.
    :param graphs: an iterable of graphs in which to replace variables
    :param additional_inputs: an iterable of graph inputs not used in any
         of 'graphs' but possibly used in the graphs returned by `replacer`
    :return: the new graphs, in the same order as 'graphs'

    Example:

    .. code-block:: python

        tag = "replaceme"

        a = tensor.scalar("a")
        b = tensor.scalar("b")
        c = tensor.scalar("c")

        ab = a + b
        ab.tag.replacement = a * b

        u = ab + c
        v, = map_variables(lambda graph:
            return getattr(graph.tag, "replacement", graph),
            [u])

        # v is now equal to a * b + c
    """

    # wrap replacer to avoid replacing things we just put there.
    graphs_seen = set()
    def wrapped_replacer(graph):
        if graph in graphs_seen:
            return graph
        else:
            new_graph = replacer(graph)
            graphs_seen.add(new_graph)
            return new_graph

    graphs = list(graphs)
    inputs_ = list(set(gof.graph.inputs(graphs) + list(additional_inputs)))

    # perform any desired replacement of input variables.  these
    # aren't replaced by the local optimizer approach because they are
    # not outputs of any Apply node.
    new_inputs = list(map(wrapped_replacer, inputs_))
    replacements = [(input_, new_input)
                    for input_, new_input
                    in zip(inputs_, new_inputs)
                    if new_input is not input_]
    graphs = clone(graphs, share_inputs=True, replace=replacements)
    inputs_ = list(set(gof.graph.inputs(graphs) + list(additional_inputs)))

    # clone cached constants or FunctionGraph will complain.  this has
    # to occur in a separate pass from the replacement above because
    # both may suggest different replacements for the same variables.
    # since the replacements introduced above may involve cached
    # constants, the replacement of said constants has to come after.
    cached_constants = [x for x in inputs_ if getattr(x, "cached", False)]
    copied_constants = clone(cached_constants, share_inputs=False)
    replacements = list(zip(cached_constants, copied_constants))
    inputs_ = list(set(inputs_) - set(cached_constants)) + list(copied_constants)
    graphs = clone(graphs, share_inputs=True, replace=replacements)

    fg = gof.fg.FunctionGraph(inputs_, graphs, clone=False)

    nodes_seen = set()

    @gof.opt.local_optimizer(None)
    def local_transform(node):
        if node in nodes_seen:
            return False

        # importing Scan into module scope would be circular
        from theano.scan_module.scan_op import Scan
        from theano.compile import OpFromGraph

        if isinstance(node.op, (Scan, OpFromGraph)):
            # recurse on the inner graph
            (new_inner_inputs,
             new_outer_inputs,
             new_inner_outputs) = _map_variables_inner(
                 wrapped_replacer,
                 inner_inputs=node.op.inputs,
                 outer_inputs=node.inputs,
                 inner_outputs=node.op.outputs,
                 containing_op=node.op)
            # reinstantiate the op
            if isinstance(node.op, Scan):
                new_op = Scan(new_inner_inputs,
                              new_inner_outputs,
                              node.op.info,
                              # FIXME: infer this someday?
                              typeConstructor=None)
            elif isinstance(node.op, OpFromGraph):
                new_op = OpFromGraph(new_inner_inputs,
                                     new_inner_outputs,
                                     **node.op.kwargs)
            # make a new node to replace the old one
            new_node = new_op.make_node(*new_outer_inputs)
            nodes_seen.add(new_node)
            return new_node.outputs
        else:
            nodes_seen.add(node)
            return list(map(wrapped_replacer, node.outputs))

    topo_transform = gof.opt.TopoOptimizer(local_transform, 'out_to_in')
    topo_transform.optimize(fg)

    new_graphs = fg.outputs
    fg.disown()
    return new_graphs


def _map_variables_inner(replacer, inner_inputs, outer_inputs,
                         inner_outputs, containing_op):
    # the replacements returned by the replacer may involve variables
    # that are already owned by the outer fgraph (`fg` in the caller)
    # and so cannot be added to the inner fgraph (`fg` in the
    # recursive call).  wrap the replacer to catch these before they
    # are added.

    # additionally, some of these may be fgraph inputs or shared
    # variables, which we cannot directly use inside the inner graph.
    # we need to create inner inputs to access them through.

    outer_to_inner = dict(zip(outer_inputs, inner_inputs))
    extra_inner_inputs = []
    extra_outer_inputs = []

    from theano.scan_module import scan_utils
    from itertools import chain
    from theano import gof

    def inner_replacer(graph):
        new_graph = replacer(graph)

        other_inputs = []
        constants = []
        for input_ in gof.graph.inputs([new_graph]):
            if isinstance(input_, gof.Variable):
                if isinstance(input_, gof.Constant):
                    constants.append(input_)
                else:
                    other_inputs.append(input_)

        # foreign inputs are fgraph inputs and shared variables that we need
        # to access through inner inputs
        foreign_inputs = list(set(other_inputs) - set(outer_to_inner.values()))

        # skip further processing if there is nothing to do
        if not constants and not foreign_inputs:
            return new_graph

        replacements = []

        # constants just need to be replaced by copies that the inner
        # `fg` can take ownership of
        for input_ in constants:
            new_input = input_.clone()
            new_input.name = "%s_copied" % new_input.name
            replacements.append((input_, new_input))

        for outer_input in foreign_inputs:
            if getattr(outer_input, "update", False):
                # when theano.scan() constructs a scan node, it detects
                # shared variables with updates and returns these updates
                # to the user.  we need to do the same thing for every new
                # use of such a variable that is introduced.  it's hard to
                # do that at this point.
                # shared variables with updates inside the inner graph of
                # OpFromGraph are not supported at all, so we don't support
                # introducing those either.
                raise NotImplementedError(
                    "Replacement introduces shared variable %s "
                    "which has an update associated with it into "
                    "the inner graph of %s. This is not currently "
                    "supported." % (outer_input, containing_op))
            # if this foreign input is not already available
            # as an inner input, connect it through a new
            # inner input
            if outer_input not in outer_to_inner.keys():
                inner_input = scan_utils.safe_new(outer_input, tag="_copy")
                outer_to_inner[outer_input] = inner_input
                extra_inner_inputs.append(inner_input)
                extra_outer_inputs.append(outer_input)
                # the inner FunctionGraph wants to know its inputs
                # beforehand, but we don't always know.  so add them
                # as we discover them.
                graph.owner.fgraph.add_input(inner_input)

        replacements.extend(outer_to_inner.items())

        new_graph, = theano.clone([new_graph],
                                  share_inputs=True,
                                  replace=replacements)
        return new_graph

    new_inner_outputs = map_variables(inner_replacer, inner_outputs)
    new_inner_inputs = list(chain(inner_inputs, extra_inner_inputs))
    new_outer_inputs = list(chain(outer_inputs, extra_outer_inputs))

    return new_inner_inputs, new_outer_inputs, new_inner_outputs


def get_updates_and_outputs(ls):
    """
    This function tries to recognize the updates OrderedDict, the
    list of outputs and the stopping condition returned by the
    lambda expression and arrange them in a predefined order.

    WRITEME: what is the type of ls? how is it formatted?
            if it's not in the predefined order already, how does
            this function know how to put it in that order?

    """
    def is_outputs(elem):
        if (isinstance(elem, (list, tuple)) and
            all([isinstance(x, theano.Variable) for x in elem])):
            return True
        if isinstance(elem, theano.Variable):
            return True
        return False

    def is_updates(elem):
        if isinstance(elem, dict):
            # Make sure the updates will be applied in a deterministic order
            if (not isinstance(elem, compat.OrderedDict) and
                len(elem) > 1):
                warnings.warn("Expected OrderedDict or OrderedUpdates, got "\
                        + str(type(elem)) + ". This can make your script non-"
                        "deterministic.")
            return True
        # Dictionaries can be given as lists of tuples
        if (isinstance(elem, (list, tuple)) and
            all([isinstance(x, (list, tuple)) and len(x) == 2
                 for x in elem])):
            return True
        return False

    def is_condition(elem):
        return isinstance(elem, theano.scan_module.until)

    def _list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        else:
            return [x]

    def _filter(x):
        """
        Ensure `x` is made only of allowed data types.

        Return True iff `x` is made only of lists, tuples, dictionaries, Theano
        variables or `theano.scan_module.until` objects.

        """
        # Is `x` a container we can iterate on?
        iter_on = None
        if isinstance(x, list) or isinstance(x, tuple):
            iter_on = x
        elif isinstance(x, dict):
            iter_on = iteritems(x)
        if iter_on is not None:
            return all(_filter(y) for y in iter_on)
        else:
            return (isinstance(x, theano.Variable) or
                    isinstance(x, theano.scan_module.until))

    if not _filter(ls):
        raise ValueError(
                'The return value of your scan lambda expression may only be '
                'made of lists, tuples, or dictionaries containing Theano '
                'variables (or `theano.scan_module.until` objects for '
                'conditions). In particular if you need to use constant '
                'values, you can use `tensor.constant` to turn them into '
                 'Theano variables.')

    if is_outputs(ls):
        return None, _list(ls), OrderedDict()
    if is_updates(ls):
        return None, [], OrderedDict(ls)
    error_msg = ('Scan cannot parse the return value of your lambda '
                 'expression, which is: %s' % (ls,))
    if not isinstance(ls, (list, tuple)):
        raise ValueError(error_msg)
    ls = list(ls)
    deprecation_msg = ('The return value of the lambda function'
                    ' has been restricted. you have to always return first the'
                    ' outputs (if any), afterwards the updates (if any) and'
                    ' at the end the conclusion')
    if len(ls) == 2:
        if is_outputs(ls[0]):
            if is_updates(ls[1]):
                return (None, _list(ls[0]), OrderedDict(ls[1]))
            elif is_condition(ls[1]):
                return (ls[1].condition, _list(ls[0]), OrderedDict())
            else:
                raise ValueError(error_msg)
        elif is_updates(ls[0]):
            if is_outputs(ls[1]):
                raise ValueError(deprecation_msg)
            elif is_condition(ls[1]):
                return (ls[1].condition, [], OrderedDict(ls[0]))
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)
    elif len(ls) == 3:
        if is_outputs(ls[0]):
            if is_updates(ls[1]):
                if is_condition(ls[2]):
                    return (ls[2].condition, _list(ls[0]), OrderedDict(ls[1]))
                else:
                    raise ValueError(error_msg)
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)
    else:
        raise ValueError(error_msg)


def isNaN_or_Inf_or_None(x):
    isNone = x is None
    try:
        isNaN = numpy.isnan(x)
        isInf = numpy.isinf(x)
        isStr = isinstance(x, string_types)
    except Exception:
        isNaN = False
        isInf = False
        isStr = False
    if not isNaN and not isInf:
        try:
            val = get_scalar_constant_value(x)
            isInf = numpy.isinf(val)
            isNaN = numpy.isnan(val)
        except Exception:
            isNaN = False
            isInf = False
    if isinstance(x, gof.Constant) and isinstance(x.data, string_types):
        isStr = True
    else:
        isStr = False
    return isNone or isNaN or isInf or isStr


def expand_empty(tensor_var, size):
    """
    Transforms the shape of a tensor from (d1, d2 ... ) to ( d1+size, d2, ..)
    by adding uninitialized memory at the end of the tensor.

    """

    if size == 0:
        return tensor_var
    shapes = [tensor_var.shape[x] for x in xrange(tensor_var.ndim)]
    new_shape = [size + shapes[0]] + shapes[1:]
    empty = tensor.AllocEmpty(tensor_var.dtype)(*new_shape)

    ret = tensor.set_subtensor(empty[:shapes[0]], tensor_var)
    ret.tag.nan_guard_mode_check = False
    return ret


def equal_computations(xs, ys, in_xs=None, in_ys=None):
    """Checks if Theano graphs represent the same computations.

    The two lists `xs`, `ys` should have the same number of entries. The
    function checks if for any corresponding pair `(x,y)` from `zip(xs,ys)`
    `x` and `y` represent the same computations on the same variables
    (unless equivalences are provided using `in_xs`, `in_ys`).

    If `in_xs` and `in_ys` are provided, then when comparing a node `x` with
    a node `y` they are automatically considered as equal if there is some
    index `i` such that `x == in_xs[i]` and `y == in_ys[i]`(and they both
    have the same type). Note that `x` and `y` can be in the list `xs` and
    `ys`, but also represent subgraphs of a computational graph in `xs`
    or `ys`.

    """
    assert len(xs) == len(ys)
    if in_xs is None:
        in_xs = []
    if in_ys is None:
        in_ys = []

    for x, y in izip(xs, ys):
        if x.owner and not y.owner:
            return False
        if y.owner and not x.owner:
            return False
        if x.owner:  # Check above tell that y.owner eval to True too.
            if x.owner.outputs.index(x) != y.owner.outputs.index(y):
                return False
        if x not in in_xs and x.type != y.type:
            return False
    if len(in_xs) != len(in_ys):
        return False
    for _x, _y in izip(in_xs, in_ys):
        if _x.type != _y.type:
            return False

    common = set(zip(in_xs, in_ys))
    different = set()
    for dx, dy in izip(xs, ys):
        # We checked above that both dx and dy have an owner or not
        if not dx.owner:
            if (isinstance(dx, tensor.Constant) and
                  isinstance(dy, tensor.Constant)):
                if not dx.equals(dy):
                    return False
                else:
                    pass
            elif (dx, dy) not in common and dx != dy:
                return False

    # Explore the two graphs, in parallel, depth first, comparing the nodes
    # along the way for equality.
    def compare_nodes(nd_x, nd_y, common, different):
        """
        Compare two nodes to determine if they perform equal computation.
        This is done by comparing the ops, the number of inputs, outputs and
        by ensuring that the inputs themselves are the result of equal
        computation.

        NOTE : This function relies on the variable common to cache
        results to be more efficient.

        """

        if nd_x.op != nd_y.op:
            return False
        elif len(nd_x.inputs) != len(nd_y.inputs):
            return False
        elif len(nd_x.outputs) != len(nd_y.outputs):
            return False
        else:
            all_in_common=True
            for dx, dy in izip(nd_x.outputs, nd_y.outputs):
                if (dx, dy) in different:
                    return False
                if (dx, dy) not in common:
                    all_in_common = False

            if all_in_common:
                return True

            # Compare the individual inputs for equality
            for dx, dy in izip(nd_x.inputs, nd_y.inputs):
                if (dx, dy) not in common:

                    # Equality between the variables is unknown, compare
                    # their respective owners, if they have some
                    if (dx.owner and dy.owner and
                        dx.owner.outputs.index(dx) ==
                        dy.owner.outputs.index(dy)):

                        nodes_equal = compare_nodes(dx.owner, dy.owner, common, different)
                        if not nodes_equal:
                            different.add((dx, dy))
                            return False

                    # If both variables don't have an owner, then they are
                    # inputs and can be directly compared
                    elif dx.owner is None and dy.owner is None:

                        if dx != dy:
                            if (isinstance(dx, tensor.Constant) and
                                isinstance(dy, tensor.Constant)):
                                if not dx.equals(dy):
                                    return False
                            else:
                                return False

                    else:
                        return False

            # If the code reaches this statement then the inputs are pair-wise
            # equivalent so the outputs of the current nodes are also
            # pair-wise equivalents
            for dx, dy in izip(nd_x.outputs, nd_y.outputs):
                common.add((dx, dy))

            return True

    # Validate that each xs[i], ys[i] pair represents the same computation
    for i in range(len(xs)):
        if xs[i].owner:
            # The case where pairs of x[i]s and y[i]s don't both have an owner
            # have already been adressed.
            is_equal = compare_nodes(xs[i].owner, ys[i].owner, common, different)
            if not is_equal:
                return False

    return True


def infer_shape(outs, inputs, input_shapes):
    """
    Compute the shape of the outputs given the shape of the inputs of a theano
    graph.

    We do it this way to avoid compiling the inner function just to get
    the shape. Changes to ShapeFeature could require changes in this function.

    """
    # We use a ShapeFeature because it has all the necessary logic
    # inside.  We don't use the full ShapeFeature interface, but we
    # let it initialize itself with an empty fgraph, otherwise we will
    # need to do it manually
    for inp, inp_shp in izip(inputs, input_shapes):
        if inp_shp is not None and len(inp_shp) != inp.ndim:
            assert len(inp_shp) == inp.ndim

    shape_feature = tensor.opt.ShapeFeature()
    shape_feature.on_attach(theano.gof.FunctionGraph([], []))

    # Initialize shape_of with the input shapes
    for inp, inp_shp in izip(inputs, input_shapes):
        shape_feature.set_shape(inp, inp_shp)

    def local_traverse(out):
        """
        Go back in the graph, from out, adding computable shapes to shape_of.

        """
        if out in shape_feature.shape_of:
            # Its shape is already known
            return
        elif out.owner is None:
            # This is an input of the graph
            shape_feature.init_r(out)
        else:
            # Recurse over inputs
            for inp in out.owner.inputs:
                if not inp in shape_feature.shape_of:
                    local_traverse(inp)

            # shape_feature.on_import does not actually use an fgraph
            # It will call infer_shape and set_shape appropriately
            dummy_fgraph = None
            shape_feature.on_import(dummy_fgraph, out.owner, reason="dummy")

    ret = []
    for o in outs:
        local_traverse(o)
        ret.append(shape_feature.shape_of[o])
    return ret


class Validator(object):
    """
    Check if variables can be expressed without using variables in invalid.

    Parameters
    ----------
    valid_equivalent
        Provides a dictionary mapping some invalid variables to valid ones that
        can be used instead.

    """

    def __init__(self, valid=None, invalid=None, valid_equivalent=None):
        if valid is None:
            valid = []
        if invalid is None:
            invalid = []
        if valid_equivalent is None:
            valid_equivalent = OrderedDict()

        # Nodes that are valid to have in the graph computing outputs
        self.valid = set(valid)

        # Nodes that are NOT valid to have in the graph computing outputs
        self.invalid = set(invalid)

        # Mapping from invalid variables to equivalent valid ones.
        self.valid_equivalent = valid_equivalent.copy()
        self.valid.update(list(valid_equivalent.values()))
        self.invalid.update(list(valid_equivalent.keys()))

    def check(self, out):
        """
        Go backwards in the graph, from out, and check if out is valid.

        If out is a valid node, (out, True) is returned.
        If out is not valid, but has an equivalent e, (e, False) is returned.
        If out is not valid and has no equivalent, None is returned.

        """
        if out in self.valid:
            return out, True
        elif out in self.valid_equivalent:
            return self.valid_equivalent[out], False
        elif out in self.invalid:
            return None

        if out.owner is None:
            if isinstance(out, tensor.TensorConstant):
                # This might be a constant from the outer graph or a constant
                # from the inner graph. In all cases, we can clone it to be
                # certain we have a valid constant
                cloned_out = out.clone()
                self.valid.add(cloned_out)
                self.invalid.add(out)
                self.valid_equivalent[out] = cloned_out
                return cloned_out, False
            else:
                # This is an input node and it has not been explicitly marked
                # as invalid so we can use it
                return out, True

        # Recurse over inputs
        inputs = [self.check(i) for i in out.owner.inputs]

        # If some inputs are invalid without equivalent, so is out
        if None in inputs:
            self.invalid.add(out)
            return None

        # If some inputs are invalid with equivalent,
        # an equivalent out should be built and returned
        all_inputs = [inp for (inp, is_valid) in inputs]
        equiv_inputs = [inp for (inp, is_valid) in inputs if not is_valid]
        if equiv_inputs:
            cloned_node = out.owner.clone_with_new_inputs(all_inputs)
            cloned_out = cloned_node.outputs[out.index]
            self.invalid.add(out)
            self.valid.add(cloned_out)
            self.valid_equivalent[out] = cloned_out
            return cloned_out, False

        # All inputs are valid, so is out
        return out, True


def scan_can_remove_outs(op, out_idxs):
    """
    Looks at all outputs defined by indices ``out_idxs`` and see whom can be
    removed from the scan op without affecting the rest. Return two lists,
    the first one with the indices of outs that can be removed, the second
    with the outputs that can not be removed.

    """
    non_removable = [o for i, o in enumerate(op.outputs) if i not in
                     out_idxs]
    required_inputs = gof.graph.inputs(non_removable)

    out_ins = []
    offset = op.n_seqs
    lim = op.n_mit_mot + op.n_mit_sot + op.n_sit_sot
    for idx in range(lim):
        n_ins = len(op.info['tap_array'][idx])
        out_ins += [op.inputs[offset:offset + n_ins]]
        offset += n_ins
    out_ins += [[] for k in xrange(op.n_nit_sot)]
    out_ins += [[op.inputs[offset + k]] for k in xrange(op.n_shared_outs)]

    added = True
    out_idxs_mask = [1 for idx in out_idxs]
    while added:
        added = False
        for pos, idx in enumerate(out_idxs):
            if (out_idxs_mask[pos] and
                 numpy.any([x in required_inputs for x in out_ins[idx]])):
                # This output is required ..
                out_idxs_mask[pos] = 0
                required_inputs += gof.graph.inputs([op.outputs[idx]])
                added = True

    required_outs = [x for i, x in enumerate(out_idxs)
                        if out_idxs_mask[i] == 0]
    not_required = [x for i, x in enumerate(out_idxs) if out_idxs_mask[i] == 1]
    return (required_outs, not_required)


def compress_outs(op, not_required, inputs):
    """
    Helpful function that gets a Scan op, a list of indices indicating
    which outputs are not required anymore and should be removed, and
    a list of inputs to the apply node corresponding to the scan op and
    produces the list of inputs and outputs and the info dictionary where
    the indicated outputs are eliminated. Note that eliminating an output
    means removing its inputs from the inner funciton and from the
    node inputs, and changing the dictionary.

    """
    info = OrderedDict()
    info['tap_array'] = []
    info['n_seqs'] = op.info['n_seqs']
    info['n_mit_mot'] = 0
    info['n_mit_mot_outs'] = 0
    info['mit_mot_out_slices'] = []
    info['n_mit_sot'] = 0
    info['n_sit_sot'] = 0
    info['n_shared_outs'] = 0
    info['n_nit_sot'] = 0
    info['truncate_gradient'] = op.info['truncate_gradient']
    info['name'] = op.info['name']
    info['gpu'] = op.info['gpu']
    info['gpua'] = op.info['gpua']
    info['mode'] = op.info['mode']
    info['as_while'] = op.info['as_while']
    info['profile'] = op.info['profile']
    info['allow_gc'] = op.info['allow_gc']

    op_inputs = op.inputs[:op.n_seqs]
    op_outputs = []
    node_inputs = inputs[:op.n_seqs + 1]
    map_old_new = OrderedDict()

    offset = 0
    ni_offset = op.n_seqs + 1
    i_offset = op.n_seqs
    o_offset = 0
    curr_pos = 0
    for idx in xrange(op.info['n_mit_mot']):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info['n_mit_mot'] += 1
            info['tap_array'] += [op.tap_array[offset + idx]]
            info['mit_mot_out_slices'] += [op.mit_mot_out_slices[offset + idx]]
            # input taps
            for jdx in op.tap_array[offset + idx]:
                op_inputs += [op.inputs[i_offset]]
                i_offset += 1
            # output taps
            for jdx in op.mit_mot_out_slices[offset + idx]:
                op_outputs += [op.outputs[o_offset]]
                o_offset += 1
            # node inputs
            node_inputs += [inputs[ni_offset + idx]]
        else:
            o_offset += len(op.mit_mot_out_slices[offset + idx])
            i_offset += len(op.tap_array[offset + idx])
    info['n_mit_mot_outs'] = len(op_outputs)
    offset += op.n_mit_mot
    ni_offset += op.n_mit_mot

    for idx in xrange(op.info['n_mit_sot']):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info['n_mit_sot'] += 1
            info['tap_array'] += [op.tap_array[offset + idx]]
            # input taps
            for jdx in op.tap_array[offset + idx]:
                op_inputs += [op.inputs[i_offset]]
                i_offset += 1
            # output taps
            op_outputs += [op.outputs[o_offset]]
            o_offset += 1
            # node inputs
            node_inputs += [inputs[ni_offset + idx]]
        else:
            o_offset += 1
            i_offset += len(op.tap_array[offset + idx])

    offset += op.n_mit_sot
    ni_offset += op.n_mit_sot
    for idx in xrange(op.info['n_sit_sot']):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info['n_sit_sot'] += 1
            info['tap_array'] += [op.tap_array[offset + idx]]
            # input taps
            op_inputs += [op.inputs[i_offset]]
            i_offset += 1
            # output taps
            op_outputs += [op.outputs[o_offset]]
            o_offset += 1
            # node inputs
            node_inputs += [inputs[ni_offset + idx]]
        else:
            o_offset += 1
            i_offset += 1

    offset += op.n_sit_sot
    ni_offset += op.n_sit_sot
    nit_sot_ins = []
    for idx in xrange(op.info['n_nit_sot']):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info['n_nit_sot'] += 1
            op_outputs += [op.outputs[o_offset]]
            o_offset += 1
            nit_sot_ins += [inputs[ni_offset + idx + op.n_shared_outs]]
        else:
            o_offset += 1

    offset += op.n_nit_sot
    shared_ins = []
    for idx in xrange(op.info['n_shared_outs']):
        if offset + idx not in not_required:
            map_old_new[offset + idx] = curr_pos
            curr_pos += 1
            info['n_shared_outs'] += 1
            op_outputs += [op.outputs[o_offset]]
            o_offset += 1
            op_inputs += [op.inputs[i_offset]]
            i_offset += 1
            shared_ins += [inputs[ni_offset + idx]]
        else:
            o_offset += 1
            i_offset += 1
    node_inputs += shared_ins
    node_inputs += nit_sot_ins
    # other stuff
    op_inputs += op.inputs[i_offset:]
    node_inputs += inputs[ni_offset + op.n_shared_outs + op.n_nit_sot:]
    if op.as_while:
        op_outputs += [op.outputs[o_offset]]
        map_old_new[o_offset] = len(op_outputs) - 1
        #map_old_new[len(op_outputs)-1] = o_offset

    return (op_inputs, op_outputs, info, node_inputs, map_old_new)


def find_up(l_node, f_node):
    r"""
    Goes up in the graph and returns True if a node in nodes is found.

    """
    if isinstance(l_node, gof.Apply):
        l_outs = l_node.outputs
    else:
        l_outs = l_node
    l_ins = gof.graph.inputs(l_outs)
    nodes = gof.graph.io_toposort(l_ins, l_outs)
    return f_node in nodes


def reconstruct_graph(inputs, outputs, tag=None):
    """
    Different interface to clone, that allows you to pass inputs.
    Compared to clone, this method always replaces the inputs with
    new variables of the same type, and returns those (in the same
    order as the original inputs).

    """
    if tag is None:
        tag = ''
    nw_inputs = [safe_new(x, tag) for x in inputs]
    givens = OrderedDict()
    for nw_x, x in izip(nw_inputs, inputs):
        givens[x] = nw_x
    allinputs = theano.gof.graph.inputs(outputs)
    for inp in allinputs:
        if isinstance(inp, theano.Constant):
            givens[inp] = inp.clone()

    nw_outputs = clone(outputs, replace=givens)
    return (nw_inputs, nw_outputs)


class scan_args(object):
    """
    Parses the inputs and outputs of scan in an easy to manipulate format.

    """

    def __init__(self, outer_inputs, outer_outputs,
                 _inner_inputs, _inner_outputs, info):
        self.n_steps = outer_inputs[0]
        rval = reconstruct_graph(_inner_inputs, _inner_outputs, '')
        if info['as_while']:
            self.cond = [rval[1][-1]]
            inner_outputs = rval[1][:-1]
        else:
            self.cond = []
            inner_outputs = rval[1]
        inner_inputs = rval[0]

        p = 1
        q = 0

        n_seqs = info['n_seqs']
        self.outer_in_seqs = outer_inputs[p:p + n_seqs]
        self.inner_in_seqs = inner_inputs[q:q + n_seqs]
        p += n_seqs
        q += n_seqs

        n_mit_mot = info['n_mit_mot']
        n_mit_sot = info['n_mit_sot']

        self.mit_mot_in_slices = info['tap_array'][:n_mit_mot]
        self.mit_sot_in_slices = info['tap_array'][
            n_mit_mot:n_mit_mot + n_mit_sot]

        n_mit_mot_ins = sum(len(s) for s in self.mit_mot_in_slices)
        n_mit_sot_ins = sum(len(s) for s in self.mit_sot_in_slices)

        iimm = inner_inputs[q:q + n_mit_mot_ins]
        self.inner_in_mit_mot = []
        qq = 0
        for sl in self.mit_mot_in_slices:
            self.inner_in_mit_mot.append(iimm[qq:qq + len(sl)])
            qq += len(sl)
        q += n_mit_mot_ins

        iims = inner_inputs[q:q + n_mit_sot_ins]
        self.inner_in_mit_sot = []
        qq = 0
        for sl in self.mit_sot_in_slices:
            self.inner_in_mit_sot.append(iims[qq:qq + len(sl)])
            qq += len(sl)
        q += n_mit_sot_ins

        self.outer_in_mit_mot = outer_inputs[p:p + n_mit_mot]
        p += n_mit_mot
        self.outer_in_mit_sot = outer_inputs[p:p + n_mit_sot]
        p += n_mit_sot

        n_sit_sot = info['n_sit_sot']
        self.outer_in_sit_sot = outer_inputs[p:p + n_sit_sot]
        self.inner_in_sit_sot = inner_inputs[q:q + n_sit_sot]
        p += n_sit_sot
        q += n_sit_sot

        n_shared_outs = info['n_shared_outs']
        self.outer_in_shared = outer_inputs[p:p + n_shared_outs]
        self.inner_in_shared = inner_inputs[q:q + n_shared_outs]
        p += n_shared_outs
        q += n_shared_outs

        n_nit_sot = info['n_nit_sot']
        self.outer_in_nit_sot = outer_inputs[p:p + n_nit_sot]
        p += n_nit_sot

        self.outer_in_non_seqs = outer_inputs[p:]
        self.inner_in_non_seqs = inner_inputs[q:]

        # now for the outputs
        p = 0
        q = 0

        self.mit_mot_out_slices = info['mit_mot_out_slices']
        n_mit_mot_outs = info['n_mit_mot_outs']
        self.outer_out_mit_mot = outer_outputs[p:p + n_mit_mot]
        iomm = inner_outputs[q:q + n_mit_mot_outs]
        self.inner_out_mit_mot = []
        qq = 0
        for sl in self.mit_mot_out_slices:
            self.inner_out_mit_mot.append(iomm[qq:qq + len(sl)])
            qq += len(sl)
        p += n_mit_mot
        q += n_mit_mot_outs

        self.outer_out_mit_sot = outer_outputs[p:p + n_mit_sot]
        self.inner_out_mit_sot = inner_outputs[q:q + n_mit_sot]
        p += n_mit_sot
        q += n_mit_sot

        self.outer_out_sit_sot = outer_outputs[p:p + n_sit_sot]
        self.inner_out_sit_sot = inner_outputs[q:q + n_sit_sot]
        p += n_sit_sot
        q += n_sit_sot

        self.outer_out_nit_sot = outer_outputs[p:p + n_nit_sot]
        self.inner_out_nit_sot = inner_outputs[q:q + n_nit_sot]
        p += n_nit_sot
        q += n_nit_sot

        self.outer_out_shared = outer_outputs[p:p + n_shared_outs]
        self.inner_out_shared = inner_outputs[q:q + n_shared_outs]
        p += n_shared_outs
        q += n_shared_outs

        assert p == len(outer_outputs)
        assert q == len(inner_outputs)

        self.other_info = OrderedDict()
        for k in ('truncate_gradient', 'name', 'mode', 'destroy_map',
                  'gpu', 'gpua', 'as_while', 'profile', 'allow_gc'):
            if k in info:
                self.other_info[k] = info[k]

    inner_inputs = property(lambda self: (self.inner_in_seqs +
                                          sum(self.inner_in_mit_mot, []) +
                                          sum(self.inner_in_mit_sot, []) +
                                          self.inner_in_sit_sot +
                                          self.inner_in_shared +
                                          self.inner_in_non_seqs))

    outer_inputs = property(lambda self: ([self.n_steps] +
                                          self.outer_in_seqs +
                                          self.outer_in_mit_mot +
                                          self.outer_in_mit_sot +
                                          self.outer_in_sit_sot +
                                          self.outer_in_shared +
                                          self.outer_in_nit_sot +
                                          self.outer_in_non_seqs))

    inner_outputs = property(lambda self: (sum(self.inner_out_mit_mot, []) +
                                           self.inner_out_mit_sot +
                                           self.inner_out_sit_sot +
                                           self.inner_out_nit_sot +
                                           self.inner_out_shared +
                                           self.cond))

    outer_outputs = property(lambda self: (self.outer_out_mit_mot +
                                           self.outer_out_mit_sot +
                                           self.outer_out_sit_sot +
                                           self.outer_out_nit_sot +
                                           self.outer_out_shared))

    info = property(lambda self: OrderedDict(
            n_seqs=len(self.outer_in_seqs),
            n_mit_mot=len(self.outer_in_mit_mot),
            n_mit_sot=len(self.outer_in_mit_sot),
            tap_array=(self.mit_mot_in_slices +
                       self.mit_sot_in_slices +
                       [[-1]] * len(self.inner_in_sit_sot)),
            n_sit_sot=len(self.outer_in_sit_sot),
            n_nit_sot=len(self.outer_in_nit_sot),
            n_shared_outs=len(self.outer_in_shared),
            n_mit_mot_outs=sum(len(s) for s in self.mit_mot_out_slices),
            mit_mot_out_slices=self.mit_mot_out_slices,
            **self.other_info))

    def __copy__(self):
        res = object.__new__(type(self))
        res.__dict__.update(self.__dict__)
        # also copy mutable attrs
        for attr in self.__dict__:
            if (attr.startswith('inner_in') or attr.startswith('inner_out') or
                attr.startswith('outer_in') or attr.startswith('outer_out') or
                attr in ('mit_mot_out_slices', 'mit_mot_in_slices',
                         'mit_sot_in_slices', 'other_info')):
                setattr(res, attr, copy.copy(getattr(self, attr)))
        return res

    def merge(self, other):
        res = copy.copy(self)
        for attr in self.__dict__:
            if (attr.startswith('inner_in') or attr.startswith('inner_out') or
                attr.startswith('outer_in') or attr.startswith('outer_out') or
                attr in ('mit_mot_out_slices', 'mit_mot_in_slices',
                         'mit_sot_in_slices')):
                getattr(res, attr).extend(getattr(other, attr))
        return res


def forced_replace(out, x, y):
    """
    Check all internal values of the graph that compute the variable ``out``
    for occurrences of values identical with ``x``. If such occurrences are
    encountered then they are replaced with variable ``y``.

    Parameters
    ----------
    out : Theano Variable
    x : Theano Variable
    y : Theano Variable

    Examples
    --------
    out := sigmoid(wu)*(1-sigmoid(wu))
    x := sigmoid(wu)
    forced_replace(out, x, y) := y*(1-y)

    """
    if out is None:
        return None

    # ``visited`` is a set of nodes that are already known and don't need to be
    # checked again, speeding up the traversal of multiply-connected graphs.
    visited = set()
    def local_traverse(graph, x):
        if graph in visited:
            return []
        visited.add(graph)
        if equal_computations([graph], [x]):
            return [graph]
        elif not graph.owner:
            return []
        else:
            rval = []
            for inp in graph.owner.inputs:
                rval += local_traverse(inp, x)
            return rval
    to_replace = local_traverse(out, x)
    return clone(out, replace=OrderedDict((v, y) for v in to_replace))
