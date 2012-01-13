"""
This module provides the Scan Op

Scanning is a general form of recurrence, which can be used for looping.
The idea is that you *scan* a function along some input sequence, producing
an output at each time-step that can be seen (but not modified) by the
function at the next time-step. (Technically, the function can see the
previous K  time-steps of your outputs and L time steps (from past and
future) of your inputs.

So for example, ``sum()`` could be computed by scanning the ``z+x_i``
function over a list, given an initial state of ``z=0``.

Special cases:

* A *reduce* operation can be performed by using only the last
  output of a ``scan``.
* A *map* operation can be performed by applying a function that
  ignores previous steps of the outputs.

Often a for-loop or while-loop can be expressed as a ``scan()`` operation,
and ``scan`` is the closest that theano comes to looping. The advantages
of using ``scan`` over `for` loops in python (amongs other) are:

* it allows the number of iterations to be part of the symbolic graph
* it allows computing gradients through the for loop
* there exist a bunch of optimizations that help re-write your loop
such that less memory is used and that it runs faster
* it ensures that data is not copied from host to gpu and gpu to
host at each step

The Scan Op should typically be used by calling any of the following
functions: ``scan()``, ``map()``, ``reduce()``, ``foldl()``,
``foldr()``.
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Frederic Bastien "
               "James Bergstra "
               "Pascal Lamblin ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


from itertools import izip
import logging
import numpy

from theano.compile import SharedVariable, function
from theano import compile
from theano import gof
from theano.tensor import opt, TensorVariable
from theano.tensor.sharedvar import TensorSharedVariable
from theano import tensor
from theano import config
from theano.updates import Updates
from theano.scalar.sharedvar import shared as scalar_shared
from theano.compile.pfunc import rebuild_collect_shared
import theano

import scan_op
import scan_utils

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan')


def scan(fn,
         sequences=None,
         outputs_info=None,
         non_sequences=None,
         n_steps=None,
         truncate_gradient=-1,
         go_backwards=False,
         mode=None,
         name=None,
         options=None,
         profile=False):
    """
    This function constructs and applies a Scan op to the provided
    arguments.

    :param fn:
        ``fn`` is a function that describes the operations involved in one
        step of ``scan``. ``fn`` should construct variables describing the
        output of one iteration step. It should expect as input theano
        variables representing all the slices of the input sequences
        and previous values of the outputs, as well as all other arguments
        given to scan as ``non_sequences``. The order in which scan passes
        these variables to ``fn``  is the following :

        * all time slices of the first sequence
        * all time slices of the second sequence
        * ...
        * all time slices of the last sequence
        * all past slices of the first output
        * all past slices of the second otuput
        * ...
        * all past slices of the last output
        * all other arguments (the list given as `non_sequences` to
            scan)

        The order of the sequences is the same as the one in the list
        `sequences` given to scan. The order of the outputs is the same
        as the order of ``output_info``. For any sequence or output the
        order of the time slices is the same as the one in which they have
        been given as taps. For example if one writes the following :

        .. code-block:: python

            scan(fn, sequences = [ dict(input= Sequence1, taps = [-3,2,-1])
                                 , Sequence2
                                 , dict(input =  Sequence3, taps = 3) ]
                   , outputs_info = [ dict(initial =  Output1, taps = [-3,-5])
                                    , dict(initial = Output2, taps = None)
                                    , Output3 ]
                   , non_sequences = [ Argument1, Argument 2])

        ``fn`` should expect the following arguments in this given order:

        #. ``Sequence1[t-3]``
        #. ``Sequence1[t+2]``
        #. ``Sequence1[t-1]``
        #. ``Sequence2[t]``
        #. ``Sequence3[t+3]``
        #. ``Output1[t-3]``
        #. ``Output1[t-5]``
        #. ``Output3[t-1]``
        #. ``Argument1``
        #. ``Argument2``

        The list of ``non_sequences`` can also contain shared variables
        used in the function, though ``scan`` is able to figure those
        out on its own so they can be skipped. For the clarity of the
        code we recommand though to provide them to scan. To some extend
        ``scan`` can also figure out other ``non sequences`` (not shared)
        even if not passed to scan (but used by `fn`). A simple example of
        this would be :

        .. code-block:: python

            import theano.tensor as TT
            W   = TT.matrix()
            W_2 = W**2
            def f(x):
                return TT.dot(x,W_2)

        The function is expected to return two things. One is a list of
        outputs ordered in the same order as ``outputs_info``, with the
        difference that there should be only one output variable per
        output initial state (even if no tap value is used). Secondly
        `fn` should return an update dictionary (that tells how to
        update any shared variable after each iteration step). The
        dictionary can optionally be given as a list of tuples. There is
        no constraint on the order of these two list, ``fn`` can return
        either ``(outputs_list, update_dictionary)`` or
        ``(update_dictionary, outputs_list)`` or just one of the two (in
        case the other is empty).

        To use ``scan`` as a while loop, the user needs to change the
        function ``fn`` such that also a stopping condition is returned.
        To do so, he/she needs to wrap the condition in an ``until`` class.
        The condition should be returned as a third element, for example:

        .. code-block:: python

            ...
            return [y1_t, y2_t], {x:x+1}, theano.scan_module.until(x < 50)

        Note that a number of steps (considered in here as the maximum
        number of steps ) is still required even though a condition is
        passed (and it is used to allocate memory if needed). = {}):

    :param sequences:
        ``sequences`` is the list of Theano variables or dictionaries
        describing the sequences ``scan`` has to iterate over. If a
        sequence is given as wrapped in a dictionary, then a set of optional
        information can be provided about the sequence. The dictionary
        should have the following keys:

        * ``input`` (*mandatory*) -- Theano variable representing the
          sequence.

        * ``taps`` -- Temporal taps of the sequence required by ``fn``.
          They are provided as a list of integers, where a value ``k``
          impiles that at iteration step ``t`` scan will pass to ``fn``
          the slice ``t+k``. Default value is ``[0]``

        Any Theano variable in the list ``sequences`` is automatically
        wrapped into a dictionary where ``taps`` is set to ``[0]``


    :param outputs_info:
        ``outputs_info`` is the list of Theano variables or dictionaries
        describing the initial state of the outputs computed
        recurrently. When this initial states are given as dictionary
        optional information can be provided about the output corresponding
        to these initial states. The dictionary should have the following
        keys:

        * ``initial`` -- Theano variable that represents the initial
          state of a given output. In case the output is not computed
          recursively (think of a map) and does not require a initial
          state this field can be skiped. Given that only the previous
          time step of the output is used by ``fn`` the initial state
          should have the same shape as the output. If multiple time
          taps are used, the initial state should have one extra
          dimension that should cover all the possible taps. For example
          if we use ``-5``, ``-2`` and ``-1`` as past taps, at step 0,
          ``fn`` will require (by an abuse of notation) ``output[-5]``,
          ``output[-2]`` and ``output[-1]``. This will be given by
          the initial state, which in this case should have the shape
          (5,)+output.shape. If this variable containing the initial
          state is called ``init_y`` then ``init_y[0]`` *corresponds to*
          ``output[-5]``. ``init_y[1]`` *correponds to* ``output[-4]``,
          ``init_y[2]`` corresponds to ``output[-3]``, ``init_y[3]``
          coresponds to ``output[-2]``, ``init_y[4]`` corresponds to
          ``output[-1]``. While this order might seem strange, it comes
          natural from splitting an array at a given point. Assume that
          we have a array ``x``, and we choose ``k`` to be time step
          ``0``. Then our initial state would be ``x[:k]``, while the
          output will be ``x[k:]``. Looking at this split, elements in
          ``x[:k]`` are ordered exactly like those in ``init_y``.
        * ``taps`` -- Temporal taps of the output that will be pass to
          ``fn``. They are provided as a list of *negative* integers,
          where a value ``k`` implies that at iteration step ``t`` scan
          will pass to ``fn`` the slice ``t+k``.

        ``scan`` will follow this logic if partial information is given:

        * If an output is not wrapped in a dictionary, ``scan`` will wrap
          it in one assuming that you use only the last step of the output
          (i.e. it makes your tap value list equal to [-1]).
        * If you wrap an output in a dictionary and you do not provide any
          taps but you provide an initial state it will assume that you are
          using only a tap value of -1.
        * If you wrap an output in a dictionary but you do not provide any
          initial state, it assumes that you are not using any form of
          taps.
        * If you provide a ``None`` instead of a variable or a empty
          dictionary ``scan`` assumes that you will not use any taps for
          this output (like for example in case of a map)

        If ``outputs_info`` is an empty list or None, ``scan`` assumes
        that no tap is used for any of the outputs. If information is
        provided just for a subset of the outputs an exception is
        raised (because there is no convention on how scan should map
        the provided information to the outputs of ``fn``)


    :param non_sequences:
        ``non_sequences`` is the list of arguments that are passed to
        ``fn`` at each steps. One can opt to exclude variable
        used in ``fn`` from this list as long as they are part of the
        computational graph, though for clarity we encourage not to do so.


    :param n_steps:
        ``n_steps`` is the number of steps to iterate given as an int
        or Theano scalar. If any of the input sequences do not have
        enough elements, scan will raise an error. If the *value is 0* the
        outputs will have *0 rows*. If the value is negative, ``scan``
        will run backwards in time. If the ``go_backwards`` flag is already
        set and also ``n_steps`` is negative, ``scan`` will run forward
        in time. If n stpes is not provided, ``scan`` will figure
        out the amount of steps it should run given its input sequences.


    :param truncate_gradient:
        ``truncate_gradient`` is the number of steps to use in truncated
        BPTT.  If you compute gradients through a scan op, they are
        computed using backpropagation through time. By providing a
        different value then -1, you choose to use truncated BPTT instead
        of classical BPTT, where you go for only ``truncate_gradient``
        number of steps back in time.


    :param go_backwards:
        ``go_backwards`` is a flag indicating if ``scan`` should go
        backwards through the sequences. If you think of each sequence
        as indexed by time, making this flag True would mean that
        ``scan`` goes back in time, namely that for any sequence it
        starts from the end and goes towards 0.


    :param name:
        When profiling ``scan``, it is crucial to provide a name for any
        instance of ``scan``. The profiler will produce an overall
        profile of your code as well as profiles for the computation of
        one step of each instance of ``scan``. The ``name`` of the instance
        appears in those profiles and can greatly help to disambiguate
        information.

    :param mode:
        It is recommended to leave this argument to None, especially
        when profiling ``scan`` (otherwise the results are not going to
        be accurate). If you prefer the computations of one step of
        ``scan`` to be done differently then the entire function, you
        can use this parameter to describe how the computations in this
        loop are done (see ``theano.function`` for details about
        possible values and their meaning).

    :param profile:
        Flag or string. If true, or different from the empty string, a
        profile object will be created and attached to the inner graph of
        scan. In case ``profile`` is True, the profile object will have the
        name of the scan instance, otherwise it will have the passed string.
        Profile object collect (and print) information only when running the
        inner graph with the new cvm linker ( with default modes,
        other linkers this argument is useless)

    :rtype: tuple
    :return: tuple of the form (outputs, updates); ``outputs`` is either a
             Theano variable or a list of Theano variables representing the
             outputs of ``scan`` (in the same order as in
             ``outputs_info``). ``updates`` is a subclass of dictionary
             specifying the
             update rules for all shared variables used in scan
             This dictionary should be passed to ``theano.function`` when
             you compile your function. The change compared to a normal
             dictionary is that we validate that keys are SharedVariable
             and addition of those dictionary are validated to be consistent.
    """
    # Note : see the internal documentation of the scan op for naming
    # conventions and all other details
    if options is None:
        options = {}
    rvals = scan_utils.canonical_arguments(sequences,
                                           outputs_info,
                                           non_sequences,
                                           go_backwards,
                                           n_steps)
    inputs, states_and_outputs_info, parameters, T = rvals
    # If we provided a known number of steps ( before compilation)
    # and if that number is 1 or -1, then we can skip the Scan Op,
    # and just apply the inner function once
    # To do that we check here to see the nature of n_steps
    T_value = None
    if isinstance(n_steps, (float, int)):
        T_value = int(n_steps)
    else:
        try:
            T_value = opt.get_constant_value(n_steps)
        except (TypeError, AttributeError):
            T_value = None

    if T_value in (1, -1):
        return one_step_scan(fn,
                             inputs,
                             states_and_outputs_info,
                             parameters,
                             truncate_gradient)

    # 1. Variable representing the current time step
    t = scalar_shared(numpy.int64(0), name='t')

    # 2. Allocate memory for the states of scan.
    mintaps = []
    lengths = []
    for pos, arg_info in enumerate(states_and_outputs_info):
        if arg_info.get('taps', None) == [-1]:
            mintaps.append(1)
            lengths.append(scalar_shared(numpy.int64(0),
                                         name='l%d' % pos))
            arg_info['initial'] = scan_utils.expand(tensor.unbroadcast(
                    tensor.shape_padleft(arg_info['initial']), 0), T)
        elif arg_info.get('taps', None):
            if numpy.any(numpy.array(arg_info.get('taps', [])) > 0):
                # Make sure we do not have requests for future values of a
                # sequence we can not provide such values
                raise ValueError('Can not use future taps of outputs',
                                 arg_info)
            mintap = abs(numpy.min(arg_info['taps']))
            lengths.append(scalar_shared(numpy.int64(0),
                                         name='l%d' % pos))
            mintaps.append(mintap)
            arg_info['initial'] = scan_utils.expand(
                arg_info['initial'][:mintap], T)
        else:
            mintaps.append(0)
            lengths.append(scalar_shared(numpy.int64(0),
                                         name='l%d' % pos))

    # 3. Generate arguments for the function passed to scan. This will
    # function will return the outputs that need to be computed at every
    # timesteps
    inputs_slices = [input[t] for input in inputs]
    states_slices = []
    for n, state in enumerate(states_and_outputs_info):
        # Check if it is actually a state and not an output
        if mintaps[n] != 0:
            for k in state['taps']:
                states_slices.append(
                    state['initial'][(t + mintaps[n] + k) % lengths[n]])

    # 4. Construct outputs that are to be computed by the inner
    # function of scan
    args = inputs_slices + states_slices + parameters
    cond, states_and_outputs, updates = \
            scan_utils.get_updates_and_outputs(fn(*args))

    # User is allowed to provide no information if it only behaves like a
    # map
    if (len(states_and_outputs) != len(states_and_outputs_info) and
        len(states_and_outputs_info) == 0):
        mintaps = [0] * len(states_and_outputs)

    # 5. Construct the scan op
    # 5.1 Construct list of shared variables with updates (those that
    # can be treated as states (i.e. of TensorType) and those that can not
    # (like Random States)

    if cond is not None:
        _cond = [cond]
    else:
        _cond = []
    rvals = rebuild_collect_shared(
        states_and_outputs + _cond,
        updates=updates,
        rebuild_strict=True,
        copy_inputs_over=True,
        no_default_updates=False)

    # extracting the arguments
    input_variables, cloned_outputs, other_rval = rvals
    clone_d, update_d, update_expr, shared_inputs = other_rval
    additional_input_states = []
    additional_output_states = []
    additional_lengths = []
    additional_mintaps = []
    original_numeric_shared_variables = []

    non_numeric_input_states = []
    non_numeric_output_states = []
    original_non_numeric_shared_variables = []
    pos = len(lengths)
    for sv in shared_inputs:
        if sv in update_d:
            if isinstance(sv, (TensorVariable, TensorSharedVariable)):
                # We can treat it as a sit sot
                nw_state = scan_utils.expand(
                    tensor.unbroadcast(tensor.shape_padleft(sv), 0), T)
                additional_lengths.append(scalar_shared(numpy.int64(0),
                                                       name='l%d' % pos))
                pos = pos + 1
                additional_mintaps.append(1)
                additional_input_states.append(nw_state)
                additional_output_states.append(
                    scan_utils.clone(tensor.set_subtensor(
                        nw_state[(t + 1) % additional_lengths[-1]],
                        update_d[sv])))
                original_numeric_shared_variables.append(sv)
            else:
                non_numeric_input_states.append(sv)
                non_numeric_output_states.append(update_d[sv])
                original_non_numeric_shared_variables.append(sv)

    # Replace shared variables in the update
    _additional_output_states = []
    replace = {}
    for sv, buf in zip(original_numeric_shared_variables,
                       additional_input_states):
        replace[sv] = buf[t]
    for out in additional_output_states:
        _additional_output_states.append(
            scan_utils.clone(out, replace=replace))
    additional_output_states = _additional_output_states

    # 5.2 Collect inputs/outputs of the inner function
    inputs = []
    outputs = []
    for n, mintap in enumerate(mintaps):
        if mintap != 0:
            input_state = states_and_outputs_info[n]['initial']
            inputs.append(input_state)
            outputs.append(
                tensor.set_subtensor(
                    input_state[(t + mintap) % lengths[n]],
                    states_and_outputs[n]))
        else:
            mem_buffer = scan_utils.allocate_memory(
                T, states_and_outputs_info[n], states_and_outputs[n])
            inputs.append(output)
            outputs.append(
                tensor.set_subtensor(output[t % lengths[n]],
                                     states_and_outputs[n]))
    inputs.extend(additional_input_states)
    outputs.extend(additional_output_states)
    lengths.extend(additional_lengths)
    mintaps.extend(additional_mintaps)
    inputs.extend(non_numeric_input_states)
    outputs.extend(non_numeric_output_states)
    all_other_inputs = gof.graph.inputs(outputs)
    parameters = [x for x in all_other_inputs
                  if (x not in inputs and x not in lengths and x is not t
                      and isinstance(x, gof.Variable) and
                      not isinstance(x, gof.Constant))]
    inputs.extend(parameters)
    # 5.3 Construct the the options dictionary
    options['name'] = name
    options['profile'] = profile
    options['mode'] = mode
    options['inplace'] = False
    options['gpu'] = False
    options['truncate_gradient'] = truncate_gradient
    options['hash_inner_graph'] = 0
    # 5.4 Construct the ScanOp instance
    local_op = scan_op.ScanOp(inputs=inputs,
                              outputs=outputs,
                              lengths=lengths,
                              switches=[],
                              mintaps=mintaps,
                              index=t,
                              options=options,
                              as_repeatUntil=cond)
    # Note that we get here all the outputs followed by the update rules to
    # the shared variables we had in our scan
    # we know that we have (in this given order):
    #   * len(states_and_outputs) real outputs
    #   * len(additional_input_states) updates for numeric shared variable
    #   * len(non_numeric_input_states) updates for non numeric shared
    #   variables
    scan_inputs = [T] + inputs
    scan_outputs_update_rules = scan_utils.to_list(local_op(*scan_inputs))
    # 5.5 Collect outputs and add permutation object
    scan_outputs = []
    for pos in xrange(len(states_and_outputs)):
        out = scan_utils.ScanPermutation(mintaps[pos])(
            scan_outputs_update_rules[pos], t)
        scan_outputs.append(out[mintaps[pos]:])
    # 5.6 Construct updates dictionary
    update_rules = scan_outputs_update_rules[len(states_and_outputs):]
    updates = {}
    for v, u in izip(original_numeric_shared_variables,
                     update_rules[:len(additional_input_states)]):
        updates[v] = u[-1]
    for v, u in izip(original_non_numeric_shared_variables,
                     update_rules[len(additional_input_states):]):
        updates[v] = u
    # Step 5.7 We are done and can return everything back to the user
    return scan_outputs, updates


def one_step_scan(fn,
                  inputs,
                  states_and_outputs_info,
                  parameters,
                  truncate_gradient):
    """
    This function is evaluated if `n_steps` evaluates to either 1 or -1.
    """
    # 1. Grab slices of sequences
    inputs_slices = [input[0] for input in inputs]

    # 2. Grab slices of states
    states_slices = []
    for n, arg_info in enumerate(states_and_outputs_info):
        if arg_info.get('taps', None) == [-1]:
            states_slices.append(arg_info['initial'])
        elif arg_info.get('taps', None):
            if numpy.any(numpy.array(arg_info.get('taps', [])) > 0):
                # Make sure we do not have requests for future values of a
                # sequence we can not provide such values
                raise ValueError('Can not use future taps of outputs',
                                    arg_info)
            # go through the taps
            mintap = abs(numpy.min(arg_info['taps']))
            states_slices.extend(
              [arg_info['initial'][k + mintap] for k in arg_info['taps']])

    # Re-order args
    args = (inputs_slices + states_slices + parameters)
    cond, states_and_outputs, updates = \
                scan_utils.get_updates_and_outputs(fn(*args))

    # We do not need to use the scan op anymore, so we can just return
    # the outputs and updates we have
    if cond is not None:
        _logger.warning(('When the number of steps is fixed and equal '
                'to 1, the provided stopping condition, ',
                str(cond), ' is ignored'))
    states_and_outputs = [tensor.unbroadcast(
        tensor.shape_padleft(arg), 0) for arg in states_and_outputs]
    if len(states_and_outputs) == 1:
        states_and_outputs = states_and_outputs[0]

    return (states_and_outputs, updates)
