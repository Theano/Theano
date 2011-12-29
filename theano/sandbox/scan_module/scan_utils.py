"""
This module provides utility functions for the Scan Op

See scan.py for details on scan
"""
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
from itertools import izip

import numpy

import theano
from theano.compile.pfunc import rebuild_collect_shared
from theano import gof
from theano import tensor, scalar
from theano.gof.python25 import all
from theano.tensor.basic import get_constant_value


# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_utils')


def expand(tensor_var, size):
    """
    Given ``tensor_var``, a Theano tensor of shape (d1, d2, ..), this
    function constructs a rval Theano tensor of shape (d1 + size, d2, ..)
    filled with 0s, except the first d1 entries which are taken from
    ``tensor_var``, namely:
        rval[:d1] = tensor_var

    :param tensor_var: Theano tensor variable
    :param size: int
    """
    # Corner case that I might use in an optimization
    if size == 0:
        return tensor_var
    shapes = [tensor_var.shape[x] for x in xrange(tensor_var.ndim)]
    zeros_shape = [size + shapes[0]] + shapes[1:]
    empty = tensor.zeros(zeros_shape,
                               dtype=tensor_var.dtype)
    return tensor.set_subtensor(empty[:shapes[0]], tensor_var)


def to_list(ls):
    """
    Converts ``ls`` to list if it is a tuple, or wraps ``ls`` into a list if
    it is not a list already
    """
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x]


class until(object):
    """
    Theano can end on a condition. In order to differentiate this condition
    from the other outputs of scan, this class is used to wrap the condition
    around it.
    """
    def __init__(self, condition):
        self.condition = tensor.as_tensor_variable(condition)
        assert self.condition.ndim == 0


def get_updates_and_outputs(ls):
    """
    Parses the list ``ls`` into outputs and updates. The semantics
    of ``ls`` is defined by the constructive function of scan.
    The elemets of ``ls`` are either a list of expressions representing the
    outputs/states, a dictionary of updates or a condition.
    """
    def is_list_outputs(elem):
        if (isinstance(elem, (list, tuple)) and
            all([isinstance(x, theano.Variable) for x in elem])):
            return True
        if isinstance(elem, theano.Variable):
            return True
        return False

    def is_updates(elem):
        if isinstance(elem, dict):
            return True
        # Dictionaries can be given as lists of tuples
        if (isinstance(elem, (list, tuple)) and
            all([isinstance(x, (list, tuple)) and len(x) == 2
                 for x in elem])):
            return True
        return False

    def is_condition(elem):
        return isinstance(elem, until)

    if is_list_outputs(ls):
        return None, _list(ls), {}
    if is_updates(ls):
        return None, [], dict(ls)
    if not isinstance(ls, (list, tuple)):
        raise ValueError(('Scan can not parse the return value'
                          ' of your constructive function given to scan'))
    ls = list(ls)
    deprication_msg = ('The return value of the lambda function'
                    ' has been restricted. you have to always return first the'
                    ' outputs (if any), afterwards the updates (if any) and'
                    ' at the end the condition')
    error_msg = ('Scan can not parse the return value of your constructive '
                 'funtion given to scan')
    if len(ls) == 2:
        if is_list_outputs(ls[0]):
            if is_updates(ls[1]):
                return (None, _list(ls[0]), dict(ls[1]))
            elif is_condition(ls[1]):
                return (ls[1].condition, _list(ls[0]), {})
            else:
                raise ValueError(error_msg)
        elif is_updates(ls[0]):
            if is_outputs(ls[1]):
                raise ValueError(deprication_msg)
            elif is_condition(ls[1]):
                return (ls[1].condition, [], dict(ls[0]))
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)
    elif len(ls) == 3:
        if is_outputs(ls[0]):
            if is_updates(ls[1]):
                if is_condition(ls[2]):
                    return (ls[2].condition, _list(ls[0]), dict(ls[1]))
                else:
                    raise ValueError(error_msg)
            else:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)


def clone(output, replace=None, strict=True, copy_inputs=True):
    """
    Function that allows replacing subgraphs of a computational
    graph. It returns a copy of the initial subgraph with the corresponding
    substitutions.

    :type output: Theano Variables (or Theano expressions)
    :param outputs: Theano expression that represents the computational
                    graph

    :type replace: dict
    :param replace: dictionary describing which subgraphs should be
                    replaced by what
    """

    inps, outs, other_stuff = rebuild_collect_shared(output,
                                                     [],
                                                     replace,
                                                     [],
                                                     strict,
                                                     copy_inputs)
    return outs


def canonical_arguments(sequences,
                        outputs_info,
                        non_sequences,
                        go_backwards,
                        n_steps):
    """
    This re-writes the arguments obtained from scan into a more friendly
    form for the scan_op.

    Mainly it makes sure that arguments are given as lists of dictionaries,
    and that the different fields of of a dictionary are set to default
    value if the user has not provided any.
    """
    states_info = to_list(outputs_info)
    parameters = [tensor.as_tensor_variable(x) for x in to_list(non_sequences)]

    inputs = []
    for input in to_list(sequences):
        if not isinstance(u, dict):
            inputs.append(input)
        elif input.get('taps', True) is None:
            inputs.append(input)
        elif input.get('taps', None):
            mintap = numpy.min(input['taps'])
            maxtap = numpy.max(input['taps'])
            for k in input['taps']:
                # We cut the sequence such that seq[i] to correspond to
                # seq[i-k]
                if maxtap < 0:
                    offset = abs(maxtap)
                else:
                    offset = 0
                if maxtap == mintap and maxtap != 0:
                    nw_input = input['input'][:abs(maxtap)]
                elif maxtap - k != 0:
                    nw_input = input['input'][offset + k - mintap:\
                                              -(maxtap - k)]
                else:
                    nw_input = input['input'][offset + k - mintap:]
                if go_backwards:
                    nw_input = nw_input[::-1]
                inputs.append(nw_input)
        else:
            raise ValueError('Provided sequence makes no sense', str(input))

    # Since we've added all sequences now we need to level them up based on
    # n_steps or their different shapes
    if n_steps is None:
        if len(inputs) == 0:
            # No information about the number of steps
            raise ValueError('You need to provide either at least '
                             'one sequence over which scan should loop '
                             'or a number of steps for scan to loop. '
                             'Neither of the two had been provided !')
        T = inputs[0].shape[0]
        for input in inputs[1:]:
            T = tensor.minimum(T, input.shape[0])
    else:
        T = tensor.as_tensor(n_steps)
    # Level up sequences
    inputs = [input[:T] for input in inputs]

    # wrap outputs info in a dictionary if they are not already in one
    for i, state in enumerate(states_info):
        if state is not None and not isinstance(state, dict):
            states_info[i] = dict(initial=state, taps=[-1])
        elif isinstance(state, dict):
            if not state.get('initial', None) and state.get('taps', None):
                raise ValueError(('If you are using slices of an output '
                                  'you need to provide a initial state '
                                  'for it'), state)
            elif state.get('initial', None) and not state.get('taps', None):
                # ^ initial state but taps not provided
                if 'taps' in state:
                    # ^ explicitly provided a None for taps
                    _logger.warning('Output %s ( index %d) has a initial '
                            'state but taps is explicitly set to None ',
                             getattr(states_info[i]['initial'], 'name',
                                     'None'), i)
                states_info[i]['taps'] = [-1]
        else:
            # if a None is provided as the output info we replace it
            # with an empty dict() to simplify handling
            states_info[i] = dict()
    return inputs, staess_info, parameters, T


def infer_shape(outs, inputs, input_shapes):
    '''
    Compute the shape of the outputs given the shape of the inputs
    of a theano graph.

    We do it this way to avoid compiling the inner function just to get
    the shape. Changes to ShapeFeature could require changes in this function.
    '''
    # We use a ShapeFeature because it has all the necessary logic
    # inside.  We don't use the full ShapeFeature interface, but we
    # let it initialize itself with an empty env, otherwise we will
    # need to do it manually
    for inp, inp_shp in izip(inputs, input_shapes):
        if inp_shp is not None and len(inp_shp) != inp.ndim:
            assert len(inp_shp) == inp.ndim

    shape_feature = tensor.opt.ShapeFeature()
    shape_feature.on_attach(theano.gof.Env([], []))

    # Initialize shape_of with the input shapes
    for inp, inp_shp in izip(inputs, input_shapes):
        shape_feature.set_shape(inp, inp_shp)

    def local_traverse(out):
        '''
        Go back in the graph, from out, adding computable shapes to shape_of.
        '''

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

            # shape_feature.on_import does not actually use an env
            # It will call infer_shape and set_shape appropriately
            dummy_env = None
            shape_feature.on_import(dummy_env, out.owner)

    ret = []
    for o in outs:
        local_traverse(o)
        ret.append(shape_feature.shape_of[o])
    return ret


class Validator(object):
    def __init__(self, valid=[], invalid=[], valid_equivalent={}):
        '''
        Check if variables can be expressed without using variables in invalid.

        init_valid_equivalent provides a dictionary mapping some invalid
        variables to valid ones that can be used instead.
        '''

        # Nodes that are valid to have in the graph computing outputs
        self.valid = set(valid)

        # Nodes that are NOT valid to have in the graph computing outputs
        self.invalid = set(invalid)

        # Mapping from invalid variables to equivalent valid ones.
        self.valid_equivalent = valid_equivalent.copy()
        self.valid.update(valid_equivalent.values())
        self.invalid.update(valid_equivalent.keys())

    def check(self, out):
        '''
        Go backwards in the graph, from out, and check if out is valid.

        If out is a valid node, (out, True) is returned.
        If out is not valid, but has an equivalent e, (e, False) is returned.
        If out is not valid and has no equivalent, None is returned.
        '''
        if out in self.valid:
            return out, True
        elif out in self.valid_equivalent:
            return self.valid_equivalent[out], False
        elif out in self.invalid:
            return None

        if out.owner is None:
            # This is an unknown input node, so it is invalid.
            self.invalid.add(out)
            if isinstance(out, tensor.TensorConstant):
                # We can clone it to get a valid constant
                cloned_out = out.clone()
                self.valid.add(cloned_out)
                self.valid_equivalent[out] = cloned_out
                return cloned_out, False

            return None

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


def allocate_memory(T, y_info, y):
    """
    Allocates memory for an output of scan.

    :param T: scalar
        Variable representing the number of steps scan will run
    :param y_info: dict
        Dictionary describing the output (more specifically describing shape
        information for the output
    :param y: Tensor variable
        Expression describing the computation resulting in out entry of y.
        It can be used to infer the shape of y
    """
    if 'shape' in y_info:
        return tensor.zeros([T, ] + list(y_info['shape']),
                            dtype=y.dtype)
    else:
        inputs = gof.graph.inputs([y])
        ins_shapes = []
        for inp in inputs:
            in_shape = [inp.shape[k] for k in xrange(inp.ndim)]
            ins_shapes.append(in_shape)
        shape = infer_shape([y], inputs, ins_shapes)[0]
        return tensor.zeros([T, ] + shape, dtype=y.dtype)
