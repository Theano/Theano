"""
This module provides a different interface for the Scan Op.
This is a sligthly more advanced interface that helps avoiding certain
issues that scan can cause.

"""
__docformat__ = 'restructedtext en'
__authors__ = "Razvan Pascanu "
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import itertools
import logging
import numpy
import warnings

from theano.compile import SharedVariable, function
from theano import compile
from theano import gof
from theano.gof.python25 import OrderedDict
from theano.tensor import opt
from theano import tensor
from theano import config
from theano.updates import OrderedUpdates


from theano.scan_module import scan_op
from theano.scan_module import  scan_utils
from theano.scan_module.scan_utils import safe_new

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan')


def scan(fn,
         sequences=None,
         states=None,
         params=None,
         n_steps=None,
         mode=None,
         name=None,
         profile=False):
    """
    Similar to Theano's official scan, this function gives the user more
    control over the scan op, avoiding certain difficulties that arose from
    missing optimizations.

    :param fn: lambda function that describes one step of scan (see the
        official Theano scan function)
    :param sequences: similar to the official Theano's scan. This version
        of scan does not support taps for the sequences (it can only be a
        list of tensor). Scan assumes that sequences have the right length
        and it does not check for this.
    :param states: similar to outputs_info of the official scan function.
        There is one crucial difference though, namely that the `initial`
        key in the dictionary has been replace by 'membuf' key. This
        reflects the change of meaning. Instead of passing to scan just
        the initial steps misisng, one has now to pass a memory buffer in
        which scan will try to store its output. In this memory buffer the
        first entries should be set to the initial states of the
        corresponding states.
        Providing a memory buffer that has less entries then the number of
        steps, mneans scan will only use that amount of memory. The user has
        to match the memory buffer size with the number of steps, otherwise
        scan will produce wrong results. Also if gradients are to be
        computed through the scan, the memory buffer should have the same
        length as the number of steps.
        For states that do not require a initial state, one has to provide a
        dictionary with a single key 'steps' that says how many intermediate
        results to store. See examples below for more insight.
    :param n_steps: This parameter is mandatory and it will represent the
        number of steps scan will do (scan will not check sequences or any
        other source of information to figure out how many steps it needs
        to do).
    :param mode: Same as for the official scan
    :param name: Same as for the official scan
    :param profile: Same as for the official scan

    Note:
     - there is no truncate / go_backwards anymore !
     - the outputs returned by scan contain the initial states as well (i.e.
     if I loop over k steps, with my smallest tap for an output -3 and keep
     al intermediate results, my output will be of length k+3

     Examples:
         (a) if you do not want to store any intermediate results (just the
         last one)

         # The memory buffer can be the initial state, just that we need to
         # add one extra dimension in front of it
         state = TT.unbroadcast(TT.shape_padleft(x0),0)
         out,_ = scan(lambda x:x+1, states = state, n_steps = 5)
         # Once we got our result we need to remove the extra dimension
         out = out[0]

        (b) if you want to keep every intermediate results

        state = TT.alloc(TT.constant(0), 6, x0.shape[0])
        state = TT.set_subtensor(state[0], x0)
        out,_ = scan(lambda x:x+1, states = state, n_steps = 5)
        out = out[1:]

    """
    def wrap_into_list(x):
        '''
        Wrap the input into a list if it is not already a list
        '''
        if x is None:
            return []
        elif not isinstance(x, (list, tuple)):
            return [x]
        else:
            return list(x)

    seqs = wrap_into_list(sequences)
    outs_info = wrap_into_list(states)

    # Make sure we get rid of numpy arrays or ints or anything like that
    # passed as inputs to scan
    non_seqs = []
    for elem in wrap_into_list(params):
        if not isinstance(elem, gof.Variable):
            non_seqs.append(tensor.as_tensor_variable(elem))
        else:
            non_seqs.append(elem)

    # If we provided a known number of steps ( before compilation)
    # and if that number is 1 or -1, then we can skip the Scan Op,
    # and just apply the inner function once
    # To do that we check here to see the nature of n_steps
    n_fixed_steps = None

    if isinstance(n_steps, (float, int)):
        n_fixed_steps = int(n_steps)
    else:
        try:
            n_fixed_steps = opt.get_scalar_constant_value(n_steps)
        except tensor.basic.NotScalarConstantError:
            n_fixed_steps = None

    # Check n_steps is an int
    if (hasattr(n_steps, 'dtype') and
        str(n_steps.dtype)[:3] not in ('uin', 'int')):
        raise ValueError(' n_steps must be an int. dtype provided '
                         'is %s' % n_steps.dtype)

    # compute number of sequences and number of outputs
    n_seqs = len(seqs)
    n_outs = len(outs_info)

    return_steps = OrderedDict()
    # wrap outputs info in a dictionary if they are not already in one
    for i in xrange(n_outs):
        if outs_info[i] is not None:
            if not isinstance(outs_info[i], dict):
                # by default any output has a tap value of -1
                outs_info[i] = dict(membuf=outs_info[i], taps=[-1])
            elif (not outs_info[i].get('membuf', None) and
                    outs_info[i].get('taps', None)):
                # ^ no initial state but taps provided
                raise ValueError(('If you are using slices of an output '
                                  'you need to provide a memory buffer for '
                                  'the state '), outs_info[i])
            elif (outs_info[i].get('membuf', None) and
                  not outs_info[i].get('taps', None)):
                # ^ initial state but taps not provided
                if 'taps' in outs_info[i]:
                    # ^ explicitly provided a None for taps
                    _logger.warning(
                            'Output %s (index %d) has a memory '
                            'buffer but taps is explicitly set to None ',
                            getattr(outs_info[i]['membuf'], 'name', 'None'),
                            i)
                outs_info[i]['taps'] = [-1]
        else:
            # if a None is provided as the output info we replace it
            # with an dict(steps=n_steps) to simplify handling
            outs_info[i] = dict(steps=n_steps)

    ##
    ###   Step 2. Generate inputs and outputs of the inner functions
    ###           for compiling a dummy function (Iteration #1)
    ##

    # create theano inputs for the recursive function
    # note : this is a first batch of possible inputs that will
    #        be compiled in a dummy function; we used this dummy
    #        function to detect shared variables and their updates
    #        and to construct a new and complete list of inputs and
    #        outputs

    n_seqs = 0
    scan_seqs = []     # Variables passed as inputs to the scan op
    inner_seqs = []    # Variables passed as inputs to the inner function
    inner_slices = []  # Actual slices if scan is removed from the picture
    # go through sequences picking up time slices as needed
    for i, seq in enumerate(seqs):
        if isinstance(seq, dict):
            seq = seq['input']
        actual_slice = seq[0]
        _seq_val = tensor.as_tensor_variable(seq)
        _seq_val_slice = _seq_val[0]

        nw_slice = _seq_val_slice.type()
        # Try to transfer test_value to the new variable
        if config.compute_test_value != 'off':
            try:
                nw_slice.tag.test_value = gof.Op._get_test_value(
                    _seq_val_slice)
            except AttributeError, e:
                if config.compute_test_value != 'ignore':
                    # No need to print a warning or raise an error now,
                    # it will be done when fn will be called.
                    _logger.info(('Cannot compute test value for '
                        'the inner function of scan, input value '
                        'missing %s'), e)

        if seq.name:
            nw_slice.name = seq.name + '[t]'
        scan_seqs.append(_seq_val)
        inner_seqs.append(nw_slice)
        inner_slices.append(actual_slice)

        n_seqs += 1

    actual_n_steps = tensor.as_tensor(n_steps)

    # Conventions :
    #   mit_mot = multiple input taps, multiple output taps ( only provided
    #             by the gradient function )
    #   mit_sot = multiple input taps, single output tap (t + 0)
    #   sit_sot = single input tap, single output tap (t + 0)
    #   nit_sot = no input tap, single output tap (t + 0)

    # MIT_MOT -- not provided by the user only by the grad function
    n_mit_mot = 0
    n_mit_mot_outs = 0
    mit_mot_scan_inputs = []
    mit_mot_inner_inputs = []
    mit_mot_inner_outputs = []
    mit_mot_out_slices = []
    mit_mot_rightOrder = []

    # SIT_SOT -- provided by the user
    n_mit_sot = 0
    mit_sot_scan_inputs = []
    mit_sot_inner_inputs = []
    mit_sot_inner_slices = []
    mit_sot_inner_outputs = []
    mit_sot_return_steps = OrderedDict()
    mit_sot_tap_array = []
    mit_sot_rightOrder = []

    n_sit_sot = 0
    sit_sot_scan_inputs = []
    sit_sot_inner_inputs = []
    sit_sot_inner_slices = []
    sit_sot_inner_outputs = []
    sit_sot_return_steps = OrderedDict()
    sit_sot_rightOrder = []
    nit_sot_steps = []
    # go through outputs picking up time slices as needed
    for i, init_out in enumerate(outs_info):
        # Note that our convention dictates that if an output uses
        # just the previous time step, as a initial state we will only
        # provide a tensor of the same dimension as one time step; This
        # makes code much cleaner for those who do not use taps. Otherwise
        # they would always had to shape_padleft the initial state ..
        # which is ugly

        # Note, 'taps' might not be in the dictionary
        if 'taps' in init_out and init_out['taps'] == [-1]:

            actual_arg = init_out['membuf']
            arg = safe_new(init_out['membuf'][0])
            if isinstance(arg, tensor.Constant):
                # safe new returns a clone of the constants, but that is not
                # what we need for initial states
                arg = arg.type()

            # Try to transfer test_value to the new variable
            if config.compute_test_value != 'off':
                try:
                    arg.tag.test_value = gof.Op._get_test_value(actual_arg)
                except AttributeError, e:
                    if config.compute_test_value != 'ignore':
                        # No need to print a warning or raise an error now,
                        # it will be done when fn will be called.
                        _logger.info(('Cannot compute test value for the '
                            'inner function of scan, input value missing %s'),
                                     e)

            if getattr(init_out['membuf'], 'name', None) is not None:
                arg.name = init_out['membuf'].name + '[t-1]'

            # We need now to allocate space for storing the output and copy
            # the initial state over. We do this using the expand function
            # defined in scan utils
            sit_sot_scan_inputs.append(actual_arg)
            sit_sot_inner_slices.append(actual_arg[0])
            if i in return_steps:
                sit_sot_return_steps[n_sit_sot] = return_steps[i]
            sit_sot_inner_inputs.append(arg)
            sit_sot_rightOrder.append(i)
            n_sit_sot += 1

        elif init_out.get('taps', None):

            if numpy.any(numpy.array(init_out.get('taps', [])) > 0):
                # Make sure we do not have requests for future values of a
                # sequence we can not provide such values
                raise ValueError('Can not use future taps of outputs',
                                    init_out)
            # go through the taps
            mintap = abs(numpy.min(init_out['taps']))
            mit_sot_tap_array.append(init_out['taps'])
            idx_offset = abs(numpy.min(init_out['taps']))
            # Sequence
            mit_sot_scan_inputs.append(init_out['membuf'])

            if i in return_steps:
                mit_sot_return_steps[n_mit_sot] = return_steps[i]
            mit_sot_rightOrder.append(i)
            n_mit_sot += 1
            for k in init_out['taps']:
                # create a new slice
                actual_nw_slice = init_out['membuf'][k + mintap]
                _init_out_var = tensor.as_tensor_variable(init_out['membuf'])
                _init_out_var_slice = _init_out_var[k + mintap]
                nw_slice = _init_out_var_slice.type()

                # Try to transfer test_value to the new variable
                if config.compute_test_value != 'off':
                    try:
                        nw_slice.tag.test_value = gof.Op._get_test_value(
                            _init_out_var_slice)
                    except AttributeError, e:
                        if config.compute_test_value != 'ignore':
                            # No need to print a warning or raise an error now,
                            # it will be done when fn will be called.
                            _logger.info(('Cannot compute test value for '
                                'the inner function of scan, input value '
                                'missing. %s'), e)

                # give it a name or debugging and pretty printing
                if getattr(init_out['membuf'], 'name', None) is not None:
                    if k > 0:
                        nw_slice.name = (init_out['membuf'].name +
                                            '[t+%d]' % k)
                    elif k == 0:
                        nw_slice.name = init_out['membuf'].name + '[t]'
                    else:
                        nw_slice.name = (init_out['membuf'].name +
                                            '[t%d]' % k)
                mit_sot_inner_inputs.append(nw_slice)
                mit_sot_inner_slices.append(actual_nw_slice)
        else:
            pass

    # Re-order args
    max_mit_sot = numpy.max([-1] + mit_sot_rightOrder) + 1
    max_sit_sot = numpy.max([-1] + sit_sot_rightOrder) + 1
    n_elems = numpy.max([max_mit_sot, max_sit_sot])
    _ordered_args = [[] for x in xrange(n_elems)]
    offset = 0
    for idx in xrange(n_mit_sot):
        n_inputs = len(mit_sot_tap_array[idx])
        if n_fixed_steps == 1:
            _ordered_args[mit_sot_rightOrder[idx]] = \
                            mit_sot_inner_slices[offset:offset + n_inputs]
        else:
            _ordered_args[mit_sot_rightOrder[idx]] = \
                            mit_sot_inner_inputs[offset:offset + n_inputs]
        offset += n_inputs

    for idx in xrange(n_sit_sot):
        if n_fixed_steps == 1:
            _ordered_args[sit_sot_rightOrder[idx]] = \
                                        [sit_sot_inner_slices[idx]]
        else:
            _ordered_args[sit_sot_rightOrder[idx]] = \
                                        [sit_sot_inner_inputs[idx]]

    ordered_args = []
    for ls in _ordered_args:
        ordered_args += ls
    if n_fixed_steps == 1:
        args = (inner_slices +
                ordered_args +
                non_seqs)

    else:
        args = (inner_seqs +
                ordered_args +
                non_seqs)

    # add only the non-shared variables and non-constants to the arguments of
    # the dummy function [ a function should not get shared variables or
    # constants as input ]
    dummy_args = [arg for arg in args
                  if (not isinstance(arg, SharedVariable) and
                      not isinstance(arg, tensor.Constant))]
    # when we apply the lambda expression we get a mixture of update rules
    # and outputs that needs to be separated
    lambda_result = fn(*args)
    condition, outputs, updates = scan_utils.get_updates_and_outputs(
                                                                lambda_result)
    if condition is not None:
        as_while = True
    else:
        as_while = False
    ##
    ###   Step 3. Check if we actually need scan and remove it if we don't
    ##

    if n_fixed_steps == 1:
        # We do not need to use the scan op anymore, so we can just return
        # the outputs and updates we have
        if condition is not None:
            _logger.warning(('When the number of steps is fixed and equal '
                    'to 1, the provided stopping condition, ',
                    str(condition), ' is ignored'))

        for pos, inner_out in enumerate(outputs):
            # we need to see if we need to pad our sequences with an
            # unbroadcastable dimension; case example : we return an
            # output for which we want all intermediate. If n_steps is 1
            # then, if we return the output as given by the innner function
            # this will represent only a slice and it will have one
            # dimension less.
            if (isinstance(inner_out.type, tensor.TensorType) and
                return_steps.get(pos, 0) != 1):
                outputs[pos] = tensor.unbroadcast(
                    tensor.shape_padleft(inner_out), 0)
        if len(outputs) == 1:
            outputs = outputs[0]

        return (outputs, updates)

    ##
    ###   Step 4. Compile the dummy function
    ##

    # We can now compile a dummy function just to see what shared variable
    # we have and what are their update rules (note that the user has
    # the option not to pass the shared variable to scan, so we need to
    # pick them manually and add them to scan)
    # make the compilation as fast as possible by not applying any
    # optimization or conversion to C [ note this region is not important
    # for performance so we can do stuff as unoptimal as we wish ]

    # extract still missing inputs (there still might be so) and add them
    # as non sequences at the end of our args
    fake_nonseqs = [x.type() for x in non_seqs]
    fake_outputs = scan_utils.clone(outputs + updates.values(),
                                    replace=dict(zip(non_seqs,
                                                     fake_nonseqs)))
    all_inputs = itertools.ifilter(
        lambda x: (isinstance(x, gof.Variable) and
                   not isinstance(x, SharedVariable) and
                   not isinstance(x, gof.Constant)),
        gof.graph.inputs(fake_outputs))
    extra_inputs = filter(lambda x: x not in args + fake_nonseqs,
                                    all_inputs)
    non_seqs += extra_inputs
    ## Note we do not use all_inputs directly since the order of variables
    ## in args is quite important
    dummy_args += extra_inputs

    dummy_outs = outputs
    if condition is not None:
        dummy_outs.append(condition)

    # If we use a regular dict here, the results are non-deterministic
    if not isinstance(updates, (list, tuple)):
        if isinstance(updates, dict) and \
            not isinstance(updates, gof.python25.OrderedDict):
                warnings.warn("Using non-deterministic dictionary.")

    dummy_f = function(dummy_args,
                       dummy_outs,
                       updates=updates,
                       mode=compile.mode.Mode(linker='py',
                                              optimizer=None),
                      on_unused_input='ignore')

    ##
    ### Step 5. Re-arange inputs of scan into a more strict order
    ##

    ## Step 5.0 Check the outputs of the dummy function to see if they
    ##          match with user provided data

    # if the number of outputs to the function does not match the number of
    # assumed outputs until now (provided by the user) there can be
    # only one explanation: No information is provided for any of the
    # outputs (i.e. we are dealing with a map)
    tmp_dummy_f_outs = len(dummy_f.maker.outputs)
    if as_while:
        tmp_dummy_f_outs -= 1
    if not (tmp_dummy_f_outs == n_outs or outs_info == []):
        raise ValueError('Please provide None as output_info for '
                         'any output that does not feed back into '
                         'scan (i.e. it behaves like a map) ')

    if outs_info == []:
        n_outs = len(dummy_f.maker.outputs)
        if as_while:
            n_outs = n_outs - 1
        outs_info = [dict(steps=n_steps) for x in xrange(n_outs)]

    ## Step 5.1 Outputs with taps different then -1

    for i, out in enumerate(outs_info):
        if 'taps' in out and out['taps'] != [-1]:
            mit_sot_inner_outputs.append(outputs[i])

    ## Step 5.2 Outputs with tap equal to -1
    for i, out in enumerate(outs_info):
        if 'taps' in out and out['taps'] == [-1]:
            sit_sot_inner_outputs.append(outputs[i])

    ## Step 5.3 Outputs that correspond to update rules of shared variables
    givens = OrderedDict()
    n_shared_outs = 0
    shared_scan_inputs = []
    shared_inner_inputs = []
    shared_inner_outputs = []
    for input in dummy_f.maker.expanded_inputs:
        if isinstance(input.variable, SharedVariable) and input.update:
            new_var = safe_new(input.variable)
            if getattr(input.variable, 'name', None) is not None:
                new_var.name = input.variable.name + '_copy'
            shared_inner_inputs.append(new_var)
            shared_scan_inputs.append(input.variable)
            shared_inner_outputs.append(input.update)
            givens[input.variable] = new_var
            n_shared_outs += 1

    ## Step 5.4 Outputs with no taps used in the input
    n_nit_sot = 0
    nit_sot_inner_outputs = []
    nit_sot_return_steps = OrderedDict()
    nit_sot_rightOrder = []
    for i, out in enumerate(outs_info):
        if not 'taps' in out:
            nit_sot_inner_outputs.append(outputs[i])
            if i in return_steps:
                nit_sot_return_steps[n_nit_sot] = return_steps[i]
            nit_sot_rightOrder.append(i)
            nit_sot_steps.append(out['steps'])
            n_nit_sot += 1

    ## Step 5.5 all other arguments including extra inputs
    other_scan_args = []
    other_inner_args = []

    other_scan_args += [arg for arg in non_seqs
                        if (not isinstance(arg, SharedVariable) and
                            not isinstance(arg, tensor.Constant))]

    ## Step 5.6 all shared variables with no update rules
    other_inner_args += [safe_new(arg, '_copy') for arg in non_seqs
                         if (not isinstance(arg, SharedVariable) and
                             not isinstance(arg, tensor.Constant))]

    givens.update(dict(zip(other_scan_args, other_inner_args)))
    other_shared_scan_args = [arg.variable for arg
                        in dummy_f.maker.expanded_inputs
                        if (isinstance(arg.variable, SharedVariable) and
                            not arg.update)]
    other_shared_inner_args = [safe_new(arg.variable, '_copy') for arg
                        in dummy_f.maker.expanded_inputs
                        if (isinstance(arg.variable, SharedVariable) and
                            not arg.update)]
    givens.update(dict(zip(other_shared_scan_args,
                           other_shared_inner_args)))

    ##
    ### Step 6. Re-order the outputs and clone them replacing things
    ###         using the givens
    ##
    inner_inputs = (inner_seqs +
                    mit_mot_inner_inputs +
                    mit_sot_inner_inputs +
                    sit_sot_inner_inputs +
                    shared_inner_inputs +
                    other_shared_inner_args +
                    other_inner_args)

    inner_outs = (mit_mot_inner_outputs +
                  mit_sot_inner_outputs +
                  sit_sot_inner_outputs +
                  nit_sot_inner_outputs +
                  shared_inner_outputs)
    if condition is not None:
        inner_outs.append(condition)
    new_givens = OrderedDict()
    for w, w_copy in givens.iteritems():
        new_givens[w] = w.type.filter_variable(w_copy)

    new_outs = scan_utils.clone(inner_outs, replace=new_givens)

    ##
    ### Step 7. Create the Scan Op
    ##

    tap_array = mit_sot_tap_array + [[-1] for x in xrange(n_sit_sot)]
    info = OrderedDict()

    info['tap_array'] = tap_array
    info['n_seqs'] = n_seqs
    info['n_mit_mot'] = n_mit_mot
    info['n_mit_mot_outs'] = n_mit_mot_outs
    info['mit_mot_out_slices'] = mit_mot_out_slices
    info['n_mit_sot'] = n_mit_sot
    info['n_sit_sot'] = n_sit_sot
    info['n_shared_outs'] = n_shared_outs
    info['n_nit_sot'] = n_nit_sot
    info['truncate_gradient'] = -1
    info['name'] = name
    info['mode'] = mode
    info['destroy_map'] = OrderedDict()
    info['inplace'] = False
    info['gpu'] = False
    info['as_while'] = as_while
    info['profile'] = profile
    info['_scan_savemem_visited'] = True

    local_op = scan_op.Scan(inner_inputs, new_outs, info)

    ##
    ### Step 8. Compute the outputs using the scan op
    ##
    _scan_inputs = (scan_seqs +
                    mit_mot_scan_inputs +
                    mit_sot_scan_inputs +
                    sit_sot_scan_inputs +
                    shared_scan_inputs +
                    nit_sot_steps +
                    other_shared_scan_args +
                    other_scan_args)

    scan_inputs = []
    for arg in [actual_n_steps] + _scan_inputs:
        if not isinstance(arg, gof.Variable):
            arg = tensor.as_tensor_variable(arg)
        scan_inputs += [arg]
    scan_outs = local_op(*scan_inputs)
    if type(scan_outs) not in (list, tuple):
        scan_outs = [scan_outs]
    ##
    ### Step 9. Figure out which outs are update rules for shared variables
    ###         and so on ...
    ##

    update_map = OrderedUpdates()

    offset = n_mit_mot
    offsets = [abs(numpy.min(x)) for x in mit_sot_tap_array]
    mit_sot_outs = scan_outs[offset:offset + n_mit_sot]

    offset += n_mit_sot
    offsets = [1 for x in xrange(n_sit_sot)]
    sit_sot_outs = scan_outs[offset:offset + n_sit_sot]

    offset += n_sit_sot
    nit_sot_outs = scan_outs[offset:offset + n_nit_sot]

    offset += n_nit_sot
    for idx, update_rule in enumerate(
                scan_outs[offset:offset + n_shared_outs]):
        update_map[shared_scan_inputs[idx]] = update_rule

    _scan_out_list = (mit_sot_outs +
                      sit_sot_outs +
                      nit_sot_outs)
    # Step 10. I need to reorder the outputs to be in the order expected by
    # the user
    rightOrder = (mit_sot_rightOrder +
                  sit_sot_rightOrder +
                  nit_sot_rightOrder)
    scan_out_list = [None] * len(rightOrder)
    for idx, pos in enumerate(rightOrder):
        scan_out_list[pos] = _scan_out_list[idx]
    if len(scan_out_list) == 1:
        scan_out_list = scan_out_list[0]
    elif len(scan_out_list) == 0:
        scan_out_list = None

    assert isinstance(update_map, OrderedDict)
    return (scan_out_list, update_map)
