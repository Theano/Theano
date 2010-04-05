"""This module provides the Scan Op

Scanning is a general form of recurrence, which can be used for looping.
The idea is that you *scan* a function along some input sequence, producing
an output at each time-step that can be seen (but not modified) by the
function at the next time-step. (Technically, the function can see the
previous K  time-steps of your outputs and L time steps (from the past and
future of the sequence) of your inputs.

So for example, ``sum()`` could be computed by scanning the ``z+x_i``
function over a list, given an initial state of ``z=0``.

Special cases:

* A *reduce* operation can be performed by returning only the last
  output of a ``scan``.
* A *map* operation can be performed by applying a function that
  ignores each previous output.

Often a for-loop can be expressed as a ``scan()`` operation, and ``scan`` is
the closest that theano comes to looping. The advantage of using ``scan``
over for loops is that it allows the number of iterations to be a part of the symbolic graph.

The Scan Op should typically be used by calling the ``scan()`` function.
"""
__docformat__ = 'restructedtext en'

import tensor
import misc.safe_asarray as safe_asarray
from tensor import opt, TensorType
import gof
from gof import Optimizer, toolbox, Op, Apply
from compile import optdb, SharedVariable, function, Param
import compile
import tensor.shared_randomstreams as shared_random
import gradient
from gof.python25 import all
import copy
import tensor.elemwise as elemwise

import numpy

# Logging function for sending warning or info
import logging
_logger = logging.getLogger('theano.scan')
def warning(*msg):
    _logger.warning('WARNING theano.scan: '+' '.join(msg))
def info(*msg):
    _logger.info('INFO theano.scan: '+' '.join(msg))


# Hashing a dictionary or a list or a tuple or any type that is hashable with
# the hash() function
def hash_listsDictsTuples(x):
    hash_value = 0
    if type(x) == dict :
        for k,v in x.iteritems():
            hash_value ^= hash_listsDictsTuples(k)
            hash_value ^= hash_listsDictsTuples(v)
    elif type(x) in (list,tuple):
        for v in x:
            hash_value ^= hash_listsDictsTuples(v)
    else:
        try:
            hash_value ^= hash(x)
        except:
            pass
    return hash_value


## TODO
###################################
## Implement specific function calls : map, reduce, generate

def map(fn, sequences, non_sequences = [],
        truncate_gradient = -1, go_backwards = False,
        mode = 'FAST_RUN'):
    ''' Similar behaviour as python map 
    
    :param fn: the function to be applied over the elements in 
               sequences ( see scan `fn` for more info)

    :param sequences: list of arrays over which map should 
                      iterate (see scan for more info)

    :param non_sequences: list of other arguments of `fn` over which 
                          map shouldn't iterate (see scan for more info)

    :param truncate_gradient: see scan for more info

    :param go_backwards: if map should also inverse the order in the arrays
                         see scan for more info

    :param mode: see scan 

    '''
    return scan(fn, sequences= sequences, outputs_info = [],non_sequences= non_sequences,
                truncate_gradient= truncate_gradient,
                go_backwards= go_backwards, mode = mode)


def reduce(fn, sequences, outputs_info, non_sequences = [], go_backwards = False, mode = 'FAST_RUN'):
    ''' Similar behaviour as python reduce 
    
    :param fn: the function to be applied over the elements in 
               sequences ( see scan `fn` for more info)

    :param outputs_info: information about outputs (mainly the initial state
                        of each )
    :param sequences: list of arrays over which reduce should 
                      iterate (see scan for more info)

    :param non_sequences: list of other arguments of `fn` over which 
                          reduce shouldn't iterate (see scan for more info)

    :param go_backwards: if reduce should also inverse the order in the arrays
                         see scan for more info

    :param mode: see scan 
    '''
    # Specify that you only want the last value of the output
    if type(outputs_info) not in (list,tuple):
        outs_info = [outputs_info]
    else:
        outs_info = outputs_info

    for i,out_info in enumerate(outs_info):
        if out_info:
            if not type(out_info) == dict:
                outs_info[i] = dict(initial = out_info,  return_steps = 1)
            else:
                # we tell scan to store only the last step
                outs_info[i]['store_steps'] = 1
                # NOTE : Maybe some errors can be detected here and 
                # we could give more meaningfull error messages then in scan ?
    return scan(fn, sequences = sequences, outputs_info = outs_info, 
            non_sequences = non_sequences, go_backwards = go_backwards,
            truncate_gradient = 1, mode = mode)

def foldl(fn, sequences, outputs_info, non_sequences = [], mode = 'FAST_RUN'):
    ''' Similar behaviour as haskell foldl 

    :param fn: the function to be applied over the elements in 
               sequences ( see scan `fn` for more info)

    :param sequences: list of arrays over which foldl should 
                      iterate (see scan for more info)

    :param outputs_info: information about outputs (mainly the initial state
                        of each )

    :param non_sequences: list of other arguments of `fn` over which 
                          foldl shouldn't iterate (see scan for more info)

    :param mode: see scan 
    '''
    return reduce(fn = fn, sequences = sequences, outputs_info = outputs_info,
            non_sequences= non_sequences, go_backwards = False, mode = mode)
 
def foldr(fn, sequences, outputs_info, non_sequences = [], mode = 'FAST_RUN'):
    ''' Similar behaviour as haskell foldr 

    :param fn: the function to be applied over the elements in 
               sequences ( see scan `fn` for more info)

    :param sequences: list of arrays over which foldr should 
                      iterate (see scan for more info)

    :param outputs_info: information about outputs (mainly the initial state
                        of each )

    :param non_sequences: list of other arguments of `fn` over which 
                          foldr shouldn't iterate (see scan for more info)

    :param truncate_gradient: see scan for more info

    :param mode: see scan 
    '''
    return reduce(fn = fn,sequences = sequences, outputs_info = outputs_info, 
            non_sequences = non_sequences, go_backwards = True, mode = mode)
 
# CONSIDER ALTERNATE CALLING CONVENTIONS:
# simple:
#    scan(fn, [a,b], [c])
# complex:
#    scan(fn, [dict(input=a, taps=[0,-1,-2]), b], [dict(initial=c, taps=[-1,-3], inplace=a)])
#
#
# So for example, if we wanted a scan that took a window of 3 inputs, and produced
# x - a sequence that we need one previous value of, and only need to return the last value;
# y - a sequence that we need no previous values of;
# z - a sequence that we need two previous values of
#     and we want z to be computed inplace using the storage of 'a'.
#
# scan(fn, [dict(input=a, taps=[-1,0,1])],
#     [dict(initial=x_init, taps=[-1], ????????),
#      None
#      dict(initial=z_init, taps=[-2,-1], inplace=a,)])


#
# QUESTION:
#   If the larger (in absolute values) the sequence_taps, the shorter the output
#   right?  If the sequence_taps = {0: [-10, 10]}, and I pass an input with 22
#   rows, then the scan will output something of length <=2 right?
#
# ANSWER:
#   Yes, actually it will be exactly 2 ( if there are no other constraints)


def scan(fn, sequences=[], outputs_info=[], non_sequences=[],
         n_steps = None, truncate_gradient = -1, go_backwards = False,
         mode = None):
    '''Function that constructs and applies a Scan op

    :param fn:
        Function that describes the operations involved in one step of scan
        Given variables representing all the slices of input and past values of
        outputs and other non sequences parameters, ``fn`` should produce
        variables describing the output of one time step of scan. The order in
        which the argument to this function are given is very important. You
        should have the following order:

        * all time slices of the first sequence (as given in the
          ``sequences`` list) ordered in the same fashion as the time taps provided
        * all time slices of the second sequence (as given in the
          ``sequences`` list) ordered in the same fashion as the time taps provided
        * ...
        * all time slices of the first output (as given in the
          ``initial_state`` list) ordered in the same fashion as the time taps provided
        * all time slices of the second otuput (as given in the
          ``initial_state`` list) ordered in the same fashion as the time taps provided
        * ...
        * all other parameters over which scan doesn't iterate given

        in the same order as in ``non_sequences`` If you are using shared
        variables over which you do not want to iterate, you do not need to
        provide them as arguments to ``fn``, though you can if you wish so. The
        function should return the outputs after each step plus the updates for
        any of the shared variables. You can either return only outputs or only
        updates. If you have both outputs and updates the function should return
        them as a tuple : (outputs, updates) or (updates, outputs).

        Outputs can be just a theano expression if you have only one outputs or
        a list of theano expressions. Updates can be given either as a list of tuples or
        as a dictionary. If you have a list of outputs, the order of these
        should match that of their ``initial_states``.

    :param sequences:
        list of Theano variables or dictionaries containing Theano variables over which
        scan needs to iterate. The reason you might want to wrap a certain Theano
        variable in a dictionary is to provide auxiliary information about how to iterate
        over that variable. For example this is how you specify that you want to use
        several time slices of this sequence at each iteration step. The dictionary
        should have the following keys :

        * ``input`` -- Theano variable representing the sequence
        * ``taps`` -- temporal taps to use for this sequence. They are given as a list
          of ints, where a value ``k`` means that at iteration step ``t`` scan needs to
          provide also the slice ``t+k`` The order in which you provide these int values
          here is the same order in which the slices will be provided to ``fn``.

        If you do not wrap a variable around a dictionary, scan will do it for you, under
        the assumption that you use only one slice, defined as a tap of offset 0. This
        means that at step ``t`` scan will provide the slice at position ``t``.

    :param outputs_info:
        list of Theano variables or dictionaries containing Theano variables used
        to initialize the outputs of scan. As before (for ``sequences``) the reason
        you would wrap a Theano variable in a dictionary is to provide additional
        information about how scan should deal with that specific output. The dictionary
        should contain the following keys:

        * ``initial`` -- Theano variable containing the initial state of the output
        * ``taps`` -- temporal taps to use for this output. The taps are given as a
          list of ints (only negative .. since you can not use future values of outputs),
          with the same meaning as for ``sequences`` (see above).
        * ``inplace`` -- theano variable pointing to one of the input sequences; this
          flag tells scan that the output should be computed in the memory spaced occupied
          by that input sequence. Note that scan will only do this if allowed by the
          rest of your computational graph.
        * ``return_steps`` how many steps to return from your output. If not given, or 
          0 scan will return all steps, otherwise it will return the last ``return_steps``.
          Note that if you set this to something else then 0, scan will always be smart 
          about the amount of memory it allocates for a given input.

        If the function applied recursively uses only the
        previous value of the output, the initial state should have
        same shape as one time step of the output; otherwise, the initial state
        should have the same number of dimension as output. This is easily
        understood through an example. For computing ``y[t]`` let us assume that we
        need ``y[t-1]``, ``y[t-2]`` and ``y[t-4]``. Through an abuse of
        notation, when ``t = 0``, we would need values for ``y[-1]``, ``y[-2]``
        and ``y[-4]``. These values are provided by the initial state of ``y``,
        which should have same number  of dimension as ``y``, where the first
        dimension should be large enough to cover all past values, which in this
        case is 4.  If ``init_y`` is the variable containing the initial state
        of ``y``, then ``init_y[0]`` corresponds to ``y[-4]``, ``init_y[1]``
        corresponds to ``y[-3]``, ``init_y[2]`` corresponds to ``y[-2]``,
        ``init_y[3]`` corresponds to ``y[-1]``. The default behaviour of scan is
        the following :

        * if you do not wrap an output in a dictionary, scan will wrap it for you
          assuming that you use only the last step of the output ( i.e. it makes your tap
          value list equal to [-1]) and that it is not computed inplace
        * if you wrap an output in a dictionary but you do not provide any taps, but
          you provide an initial state it will assume that you are using only a tap value
          of -1
        * if you wrap an output in a dictionary but you do not provide any initial state,
          it assumes that you are not using any form of taps
        * if you provide a ``None`` scan assumes that you will not use any taps for this 
          output

        If you did not provide any information for your outputs, scan will assume by default
        that you are not using any taps for any of the input. If you provide information for
        just a subset of outputs, scan will not know to which outputs these information 
        corresponds and will raise an error.

    :param non_sequences:
        Parameters over which scan should not iterate.  These parameters are
        given at each time step to the function applied recursively.


    :param n_steps:
        Number of steps to iterate. If the input sequences are not long enough, scan 
        will produce a warning and run only for the maximal amount of steps allowed by 
        the input sequences. If the value is 0, the outputs will have 0 rows. If the 
        value is negative, scan will run backwards (or if the flag go_backwards is 
        already set to true it will run forward in time). If n_steps is not provided, 
        or evaluetes not None, scan will figure out the maximal amount of steps it can 
        take and do that. 

    :param truncate_gradient:
        Number of steps to use in truncated BPTT.  If you compute gradients
        through a scan op, they are computed using backpropagation through time.
        By providing a different value then -1, you choose to use truncated BPTT
        instead of classical BPTT, where you only do ``truncate_gradient``
        number of steps. (NOT YET IMPLEMENTED)

    :param go_backwards:
        Flag indicating if you should go backwards through the sequences

    :rtype: tuple
    :return: tuple of the form (outputs, updates); ``outputs`` is either a
             Theano variable or a list of Theano variables representing the
             outputs of scan. ``updates`` is a dictionary specifying the
             updates rules for all shared variables used in the scan
             operation; this dictionary should be pass to ``theano.function``
    '''

    # check if inputs are just single variables instead of lists
    if not (type(sequences) in (list, tuple)):
        seqs = [sequences]
    else:
        seqs = sequences

    if not (type(outputs_info) in (list,tuple)):
        outs_info = [outputs_info]
    else:
        outs_info = outputs_info

    if not (type(non_sequences) in (list,tuple)):
        non_seqs = [non_sequences]
    else:
        non_seqs = non_sequences


    # compute number of sequences and number of outputs
    n_seqs = len(seqs)
    n_outs = len(outs_info)

    inplace_map    = {}
    sequences_taps = {}
    outputs_taps   = {}
    # wrap sequences in a dictionary if they are not already
    # in the same pass create a sequences_taps dictionary
    for i in xrange(n_seqs):
        if not type(seqs[i]) == dict :
            seqs[i] = dict(input=seqs[i], taps=[0])
        # see if taps values are provided as a list
        elif seqs[i].get('taps',None):
            if not type(seqs[i]['taps']) in (tuple,list):
                seqs[i]['taps'] = [seqs[i]['taps']]
        else:
            seqs[i][taps] = [0]

        if seqs[i].get('taps',None):
            sequences_taps[i] = seqs[i]['taps']

    # wrap outputs info in a dictionary if they are not already
    # in the same pass create a init_outs_taps dictionary and a inplace map
    for i in xrange(n_outs):
        if outs_info[i]:
            if not type(outs_info[i]) == dict:
                outs_info[i] = dict(initial=outs_info[i], taps = [-1])
                # if there is no initial state but there are taps
            elif (not outs_info[i].get('initial',None)) and(outs_info[i].get('taps',None)):
                raise ValueError('If you are using slices of an output you need to '\
                        'provide a initial state for it', outs_info[i])
            elif outs_info[i].get('initial',None) and (not outs_info[i].get('taps',None)):
                outs_info[i]['taps'] = [-1]
        else:
            outs_info[i] = dict()

        if outs_info[i].get('taps', None):
            outputs_taps[i] = outs_info[i]['taps']
        if outs_info[i].get('inplace', None):
            # look for that variable to get the index
            found = None
            for k in xrange(n_seqs):
                if seqs[k].get('input', None) == outs_info[i].get('inplace',None):
                    found = k
            if found != None:
                # NOTE : inplace_map is identical to destroy_map, i.e. it tells what output
                #     is computed inplace of what input !!
                inplace_map[i] = found
            else:
                raise ValueError('Asked to compute in place of a non-input variable',\
                          outs_info[i].get('inplace', None))

    # create theano inputs for the recursive function
    # note : this is a first batch of possible inputs that will
    #        be compiled in a dummy function; we used this dummy
    #        function to detect shared variables and their updates
    #        and to construct a new list of possible inputs
    args = []
    dummy_notshared_ins = 0
    dummy_notshared_init_outs = 0
    slice_to_seqs = []
    # go through sequences picking up time slices as needed
    for i,seq in enumerate(seqs):
        if seq.get('taps', None):
            for k in seq['taps']:
                nw_slice = seq['input'][0].type()
                # Add names to slices for debugging and pretty printing ..
                if seq['input'].name:
                    nw_slice.name = seq['input'].name + '[%d]'%seq['taps'][k]
                args.append(nw_slice)
                slice_to_seqs.append(i)
            dummy_notshared_ins += len(seq['taps'])
    # go through outputs picking up time slices as needed
    for i,init_out in enumerate(outs_info):
        if init_out.get('taps', None) == [-1]:
            args += [init_out['initial'].type()]
            # Added name to slices for debugging and pretty printing
            if init_out['initial'].name:
                args[-1].name = init_out['initial'].name+'[-1]'
            if slice_to_seqs:
                val = slice_to_seqs[-1]
            else:
                val = -1
            slice_to_seqs += [ val+1 ]
            dummy_notshared_init_outs += 1
        elif init_out.get('taps',None):
            if numpy.any(numpy.array(init_out.get('taps',[])) > 0):
                raise ValueError('Can not use future taps of outputs', init_out)
            if slice_to_seqs:
                val = slice_to_seqs[-1]
            else:
                val = -1
            for k in init_out['taps']:
                nw_slice = init_out['initial'][0].type()
                if init_out['initial'].name:
                    nw_slice.name = init_out['initial'].name + '[%d]'%init_out['taps'][k]
                args.append(nw_slice)
                slice_to_seqs.append(val+1)

            dummy_notshared_init_outs += len(init_out['taps'])

    # remove shared variables from the non sequences list
    notshared_other_args = []
    for non_seq in non_seqs:
        if not isinstance(non_seq, SharedVariable):
            notshared_other_args += [non_seq]

    # add only the not shared variables to the arguments of the dummy
    # function [ a function should not get shared variables as input ]
    dummy_args = args + notshared_other_args
    # arguments for the lambda expression that gives us the output
    # of the inner function
    args += non_seqs

    outputs_updates  = fn(*args)
    outputs = []
    updates = {}
    # we will try now to separate the outputs from the updates
    if not type(outputs_updates) in (list,tuple):
        if type(outputs_updates) == dict :
            # we have just an update dictionary
            updates = outputs_updates
        else:
            outputs = [outputs_updates]
    else:
        elem0 = outputs_updates[0]
        elem1 = outputs_updates[1]
        t_el0 = type(elem0)
        t_el1 = type(elem1)
        if t_el0 == dict or ( t_el0 in (list,tuple) and type(elem0[0]) in (list,tuple)):
            # elem0 is the updates dictionary / list
            updates = elem0
            outputs = elem1
            if not type(outputs) in (list,tuple):
                outputs = [outputs]
        elif ( type(elem1) == dict) or \
             ( type(elem1) in (list,tuple) and type(elem1[0]) in (list,tuple)):
            # elem1 is the updates dictionary / list
            updates = elem1
            outputs = elem0
            if not type(outputs) in (list,tuple):
                outputs = [outputs]
        else :
            if type(outputs_updates) in (list,tuple) and \
                    (type(outputs_updates[0]) in (list,tuple)):
                outputs = []
                updates = outputs_updates
            else:
                outputs = outputs_updates
                updates = {}
    # in case you return a tuple .. convert it to a list (there are certain 
    # operation that are not permited on tuples, like element assignment)
    outputs = list(outputs)
    for i,out in enumerate(outputs):
        outputs[i] = tensor.as_tensor(out)
    # Wo compile a dummy function just to see what shared variable
    # we have and what are their update rules

    dummy_f = function(dummy_args, outputs, updates = updates, mode = \
                 compile.mode.Mode(linker = 'py', optimizer = None) )

    inner_fn_out_states = [ out.variable for out in dummy_f.maker.outputs]
    update_map       = {}
    shared_outs      = []
    shared_non_seqs  = []
    givens           = {}

    # if the number of outputs to the function does not match the number of
    # assumed outputs

    if len(inner_fn_out_states) != n_outs:
        if outs_info == []:
            # We know how to deal with this case, assume that none of the outputs
            # are required to have any sort of time taps
            # we just need to update the number of actual outputs
            n_outs = len(inner_fn_out_states)
            # other updates :
            for i in xrange(n_outs):
                outs_info += [ dict() ]
        else:
            raise ValueError('There has been a terrible mistake in our input arguments'
                    ' and scan is totally lost. Make sure that you indicate for every '
                    ' output what taps you want to use, or None, if you do not want to '
                    ' use any !')
    inner_fn_inputs=[input.variable for input in \
        dummy_f.maker.expanded_inputs[:dummy_notshared_ins+dummy_notshared_init_outs]]
    fromIdx = dummy_notshared_ins + dummy_notshared_init_outs

    store_steps = [ 0 for i in xrange(n_outs)]

    for i in xrange(n_outs):
        if outs_info[i].get('return_steps', None):
            store_steps[i] = outs_info[i]['return_steps']

    # add shared variable that act as outputs
    #
    n_extended_outs = n_outs
    for input in dummy_f.maker.expanded_inputs[fromIdx:] :
        if isinstance(input.variable, SharedVariable) and input.update:
            new_var = input.variable.type()
            inner_fn_inputs.append(new_var)
            if slice_to_seqs:
                val = slice_to_seqs[-1]
            else: val = -1
            slice_to_seqs += [ val+1 ]
            inner_fn_out_states += [input.update]
            update_map[ input.variable ] = n_extended_outs
            outputs_taps[ n_extended_outs ] = [-1]
            n_extended_outs += 1
            store_steps += [1]
            shared_outs += [input.variable]
            givens[input.variable] = inner_fn_inputs[-1]

    # add the rest:
    for input in dummy_f.maker.expanded_inputs[fromIdx:] :
        if isinstance(input.variable, SharedVariable) and not input.update:
           shared_non_seqs += [input.variable]
           inner_fn_inputs += [input.variable.type() ]
           if slice_to_seqs:
               val = slice_to_seqs[-1]
           else: val = -1
           slice_to_seqs += [val +1]
           givens[input.variable] = inner_fn_inputs[-1]
        elif not isinstance(input.variable, SharedVariable):
            inner_fn_inputs.append(input.variable)

    if type(n_steps) in (float,int):
        n_fixed_steps = int(n_steps) 
    else:
        # check if it is actually a Theano constant
        try :
            n_fixed_steps = opt.get_constant_value(n_steps)
        except:
            n_fixed_steps = None
    
    if (n_steps == None or n_steps == numpy.inf or n_steps == numpy.nan) and n_seqs == 0 : 
        raise ValueError('Scan does not know for how many steps to iterate. '
                'You need to provide the number of steps through the '
                ' ``n_steps`` argument if you do not iterate over any sequence')
    # Create the Scan op object
    local_op = Scan( (inner_fn_inputs,inner_fn_out_states, givens, slice_to_seqs ), n_seqs,
            n_extended_outs, inplace_map, sequences_taps,  outputs_taps, truncate_gradient,
            go_backwards, store_steps, mode, n_fixed_steps = n_fixed_steps)

    # Call the object on the input sequences, initial values for outs,
    # and non sequences
    for seq in seqs :
        if not seq.get('input', None):
            raiseValue('All input sequences should provide')
    unwrapped_seqs = [ seq.get('input',tensor.as_tensor(0.)) for seq in seqs ]
    unwrapped_outs = [ out.get('initial',tensor.as_tensor(0.)) for out in outs_info ]

    if n_steps != None:
        n_steps = tensor.as_tensor(n_steps)
    else:
        #n_steps = tensor.constant(numpy.inf,'?_steps')
        n_steps = gof.Constant(gof.generic, 'unknown', '?_steps')

    values =  local_op( *(    [n_steps]
                         + unwrapped_seqs
                         + unwrapped_outs
                         + shared_outs
                         + notshared_other_args
                         + shared_non_seqs))

    if not type(values) in (tuple, list):
        values = [values]
    for val in update_map.keys():
        update_map[val] = values [ update_map[val] ]

    if n_outs == 1:
        values = values[0]
    else:
        values = values[:n_outs]

    return (values, update_map)


class Scan(Op):
    #
    # OLD DOCUMENTATION CAN BE FOUND NEAR REVISION 2581
    #

    def __init__(self,(inputs, outputs, givens, slice_to_seqs),n_seqs,  n_outs,
                 inplace_map={}, seqs_taps={}, outs_taps={},
                 truncate_gradient = -1,
                 go_backwards = False, store_steps = {},
                 mode = 'FAST_RUN', n_fixed_steps = None, inplace=False):
        '''
        :param (inputs,outputs, givens,slice_to_seqs):
            inputs and outputs Theano variables that describe the function that is
            applied recursively; givens list is used to replace shared
            variables with not shared ones; slice_to_seqs is a convinience list that
            tells which of the inputs is slice to which of the sequences
        :param n_seqs: number of sequences over which scan will have to
                       iterate
        :param n_outs: number of outputs of the scan op
        :param inplace_map: see scan function above
        :param seqs_taps: see scan function above
        :param outs_taps: see scan function above
        :param truncate_gradient: number of steps after which scan should
                                  truncate -1 implies no truncation
        :param go_bacwards: see scan funcion above
        :param store_steps:
            a list of booleans of same size as the number of outputs; the value at position
            ``i`` in the list corresponds to the ``i-th`` output, and it tells how many
            steps (from the end towards the begining) of the outputs you really need and should
            return; given this information, scan can know (if possible) to allocate only
            the amount of memory needed to compute that many entries
        :param n_fixed_steps: this is a number if n_steps in the scan function
            received a number or None otherwise. The value is used to optimize
            the graph, since a scan that has n_steps fixed to 1 or 0 is not
            really needed in the graph. (? could we use tag hints ?)
        '''
        #check sequences past taps
        for k,v in seqs_taps.iteritems():
          if k > n_seqs:
            raise ValueError(('Sequences past taps dictionary reffers to '
                    'an unexisting sequence %d')%k)
        #check outputs past taps
        for k,v in outs_taps.iteritems():
          if k > n_outs:
            raise ValueError(('Output past taps dictionary reffers to '
                    'an unexisting sequence %d')%k)
          if v and (max(v) > -1):
            raise ValueError(('Can not require future value %d of output' \
                    ' %d')%(k,max(v)))

        # build a list of output types for any Apply node using this op.
        self.apply_output_types = []
        for i, o in enumerate(outputs):
            if 1 == store_steps[i]:
                self.apply_output_types.append(o.type)
            else:
                expanded_otype = TensorType(
                        broadcastable=(False,)+o.type.broadcastable,
                        dtype=o.type.dtype)
                self.apply_output_types.append(expanded_otype)


        self.destroy_map = {}
        if inplace:
            for i in inplace_map.keys():
                self.destroy_map.update({i: [inplace_map[i]+1] } )
            # make all inplace inputs mutable for the inner function for extra efficency
            for idx in xrange(len(inputs)):
                # get seq number
                n_seq = slice_to_seqs[idx]
                if n_seq in inplace_map.keys():
                    if type(inputs[n_seq]) is Param:
                        inputs[n_seq].mutable = True
                    else:
                        inputs[n_seq] = Param( inputs[n_seq], mutable = True)

        self.seqs_taps      = seqs_taps
        self.outs_taps      = outs_taps
        self.n_seqs         = n_seqs
        self.n_outs         = n_outs
        self.n_args         = n_seqs+n_outs+1
        self.inplace_map    = inplace_map
        self.store_steps    = store_steps
        self.inplace        = inplace
        self.inputs         = inputs
        self.givens         = givens
        self.outputs        = outputs
        # This is here just for an optimization to be able to pick up if 
        # scan is really needed in the graph; if the number of steps 
        # scan does is a constant of 1, -1 or 0 then we can remove scan 
        # from the graph
        self.n_fixed_steps  = n_fixed_steps
        self.mode           = mode
        self.truncate_gradient = truncate_gradient
        self.go_backwards   = go_backwards
        self.slice_to_seqs  = slice_to_seqs

        self.fn = function(inputs,outputs, mode = mode, givens = givens)
        assert not numpy.any([isinstance(x.variable,SharedVariable) for x in
            self.fn.maker.inputs])



    def make_node(self,*inputs):
        assert all(isinstance(i, gof.Variable) for i in inputs)
        return Apply(self, inputs, [t() for t in self.apply_output_types])


    def __eq__(self,other):
        # the self.apply_output_types are a function of all these things
        # no need to compare it as well
        rval = type(self) == type(other)
        if rval:
            rval = (self.inputs == other.inputs) and \
            (self.outputs == other.outputs) and \
            (self.givens  == other.givens) and \
            (self.store_steps == other.store_steps) and \
            (self.seqs_taps == other.seqs_taps) and \
            (self.outs_taps == other.outs_taps) and \
            (self.inplace_map == other.inplace_map) and \
            (self.mode == other.mode) and \
            (self.n_seqs == other.n_seqs) and\
            (self.inplace == other.inplace) and\
            (self.go_backwards == other.go_backwards) and\
            (self.truncate_gradient == other.truncate_gradient) and\
            (self.n_outs == other.n_outs) and\
            (self.n_fixed_steps == other.n_fixed_steps) and\
            (self.n_args == other.n_args)
        return rval


    def __hash__(self):
        # the self.apply_output_types are a function of all these things
        # no need to compare it as well
        return hash(type(self)) ^ \
            hash(self.n_seqs) ^ \
            hash(self.n_outs) ^ \
            hash(self.inplace) ^\
            hash(self.go_backwards) ^\
            hash(self.truncate_gradient) ^\
            hash(self.n_args) ^ \
            hash(self.mode) ^\
            hash(self.n_fixed_steps) ^\
            hash_listsDictsTuples(self.outputs) ^ \
            hash_listsDictsTuples(self.inputs) ^ \
            hash_listsDictsTuples(self.givens) ^ \
            hash_listsDictsTuples(self.seqs_taps) ^\
            hash_listsDictsTuples(self.outs_taps) ^\
            hash_listsDictsTuples(self.store_steps)


    def perform(self,node,args, outs):
        """
        The args are packed like this:

            n_steps

            X sequence inputs x_1, x_2, ... x_<self.n_seqs>

            Y initial states (u_1, u_2, ... u_<self.n_outs>) for our outputs. Each must have appropriate length (T_1, T_2, ..., T_Y).

            W other inputs w_1, w_2, ... w_W

        There are at least 1 + self.n_seqs + self.n_outs inputs, and the ones above this number
        are passed to the scanned function as non-sequential inputs.


        The outputs are more straightforward:

            Y sequence outputs y_1, y_2, ... y_<self.n_outs>

        """
        n_steps = args[0]
        if n_steps != 'unknown':
            n_steps = int(n_steps)
            if n_steps < 0:
                n_steps = abs(n_steps)
                go_backwards = not self.go_backwards
            else:
                go_backwards = self.go_backwards
        else:
            n_steps = None
            go_backwards = self.go_backwards

        if (self.n_seqs == 0 ) and (not numpy.isfinite(n_steps) ):
            raise ValueError('Scan does not know how many steps it '
                'should iterate! Either provide some input sequences from '
                'which scan could find out the number of steps, or directly'
                'the number of steps you want through the n_steps argument.')

        for i in xrange(self.n_seqs):
            if self.seqs_taps.has_key(i):
                # compute actual length of the sequence ( we need to see what
                # past taps this sequence has, and leave room for them
                seq_len = args[i+1].shape[0] + min(self.seqs_taps[i])
                if  max( self.seqs_taps[i]) > 0:
                    # using future values, so need to end the sequence earlier
                    seq_len -= max(self.seqs_taps[i])
                if n_steps == None :
                    # length of the sequences, leaving room for the largest
                    n_steps = seq_len
                if seq_len != n_steps :
                    if seq_len > n_steps:
                        warning('Input sequence is longer then required. '
                                'Extra values will be ignored')
                    else:
                        warning(' Input sequence is shorter then the number '
                               'of steps scan was suppose to do. Readjusting'
                               'the number of steps scan will iterate ... ')
                    n_steps = min(seq_len,n_steps)



        # check if we deal with an inplace operation
        inplace_map  = self.inplace_map
        if not self.inplace: #if it was not optimized to work inplace
            inplace_map = {}


        # check lengths of init_outs
        for i in xrange(self.n_seqs+1, self.n_seqs+self.n_outs+1):
            if self.outs_taps.has_key(i-self.n_seqs-1):
                if self.outs_taps[i-self.n_seqs-1] != [-1]:
                    req_size = abs(min(self.outs_taps[i-self.n_seqs-1]))-1
                    if args[i].shape[0] < req_size:
                        warning(('Initial state for output %d has fewer values then '
                            'required by the maximal past value %d. Scan will use 0s'
                            ' for missing values')%(i-self.n_iterable-1,req_size))

        self.n_steps = n_steps
        y = self.scan(self.fn, args[1:],self.n_seqs, self.n_outs,
                 self.seqs_taps, self.outs_taps, n_steps, go_backwards,
                 inplace_map)

        for i in xrange(self.n_outs):
            if self.store_steps[i] > 1 :
                # we need to reorder the steps .. to have them in the correct order
                # we use numpy advanced indexing for this
                # index order :
                index_order = range(self.idx_store_steps[i],self.store_steps[i]) + \
                              range(self.idx_store_steps[i])
                outs[i][0] = y[i][index_order]
            else:
                outs[i][0] = y[i]



    def scan(self, fn, args, n_seqs, n_outs, seqs_taps, outs_taps,  n_steps, go_backwards, inplace_map):
        ''' Actual loop of the scap op perform function '''
        # Note that we removed the n_steps from the args for this function, so the
        # order of arguments is slightly different compared to perform
        y = []
        # When you have taps, you need to leave borders in your sequences, initial outputs
        # for those taps; here we compute what are those borders for sequences
        seqs_mins = {}
        for j in xrange(n_seqs):
            if seqs_taps.has_key(j):
                seqs_mins.update({j:  min(seqs_taps[j])})

        # create storage space for the outputs ( using corresponding inputs if we are
        # dealing with inplace operations
        # `idx_store_steps` is a dictionary telling us the current position in y of an
        # output where we want to store only the last k steps


        self.idx_store_steps = {}
        for i in xrange(n_outs):

            if inplace_map.has_key(i) and seqs_taps.has_key(inplace_map[i]) and\
                    seqs_taps[inplace_map[i]] >=0:
                y += [args[inplace_map[i]][:n_steps]]
            else:
                # check if you are using past value .. through in a warning and do not
                # work inplace
                if inplace_map.has_key(i) and seqs_taps.has_key(inplace_map[i]) and seqs_taps[inplace_map[i]] < 0:
                    warning('Can not work inplace because of past values')
                if self.store_steps[i] == 1 :
                    y+= [ None ]
                else:
                    arg_shape = args[i+n_seqs].shape[1:]
                    if (not self.outs_taps.has_key(i)) or self.outs_taps[i] == [-1]:
                        arg_shape = args[i+n_seqs].shape
                    if self.store_steps[i] < 1 :
                        y_shape = (n_steps,)+arg_shape
                    else:
                        # we need to store only a fixed number of steps of our output
                        self.idx_store_steps[i] = 0
                        y_shape = (self.store_steps[i],)+arg_shape
                    y += [numpy.empty(y_shape, dtype=args[i+n_seqs].dtype)]

        # and here we compute the borders for initial states of outputs
        outs_mins = {}
        initOuts_size = {}
        for j in xrange(n_outs):
            if outs_taps.has_key(j):
                outs_mins.update({j: min(outs_taps[j])})
                if self.outs_taps[j] != [-1]:
                    initOuts_size.update({j: args[n_seqs+j].shape[0]})
                else:
                    initOuts_size.update({j: 0})

        ############## THE MAIN LOOP ############################
        for i in xrange(n_steps):
            fn_args = []
            # sequences over which scan iterates
            # check to see if we are scaning them backwards or no
            # and get a new index ``_i`` accordingly
            _i = i
            if go_backwards:
                _i = n_steps-1-i
            # collect data from sequences
            for j in xrange(n_seqs):
                # get borders
                if seqs_taps.has_key(j):
                    ls_taps = seqs_taps[j]
                    min_tap = seqs_mins[j]
                    for tap_value in ls_taps:
                        # use the borders to figure out what value you actually need
                        k = _i - min_tap + tap_value
                        fn_args += [args[j][k]]

            # past values of outputs
            for j in xrange(n_outs):
                if outs_taps.has_key(j):
                    ls_taps = outs_taps[j]
                    min_tap = outs_mins[j]
                    sz = initOuts_size[j]
                    for tap_value in ls_taps:
                        if i + tap_value < 0:
                            if sz < 1:
                                # this is a special case, when our initial state has no
                                # temporal dimension
                                fn_args += [args[j+n_seqs] ]
                            else:
                                k = i + sz + tap_value
                                if k < 0:
                                    # past value not provided.. issue a warning and use 0s of the
                                    # correct dtype
                                    fn_args += [numpy.zeros(args[j+n_seqs][0].shape, dtype =
                                        args[j+n_sqs][0].dtype)]
                                    warning(('Past value %d for output %d not given in '
                                        'inital out') % (j,tap_value))
                                else:
                                    fn_args += [args[j+n_seqs][k]]
                        else:
                            if self.store_steps[j] < 1:
                                # no limit on how many steps to store from our output
                                fn_args += [y[j][i + tap_value]]
                            elif self.store_steps[j] == 1:
                                # just the last one
                                fn_args += [y[j] ]
                            else:
                                # storing only the last k
                                # get what idx we want
                                req_idx = (self.idx_store_steps[j] + tap_value + self.store_steps[j])
                                # we need this modula self.store_steps[j]
                                req_idx = req_idx % self.store_steps[j]
                                fn_args += [y[j][req_idx] ]

            # get the non-iterable sequences
            fn_args += list(args[(n_seqs+n_outs):])
            # compute output
            something = fn(*fn_args)
            #update outputs
            for j in xrange(n_outs):
                if self.store_steps[j] <1:
                    # if you have provided no size for the missing output you might find yourself
                    # here with a incorect array .. if that happens realocate memory for the
                    # needed array
                    try :
                        if hasattr(something[j],'dtype') and (y[j].dtype != something[j].dtype) :
                            raise ValueError('wrong dtype')

                        y[j][i] = something[j]
                    except :

                        y[j]= numpy.empty((n_steps,)+something[j].shape, dtype= something[j].dtype)
                        y[j][i] = something[j]

                elif self.store_steps[j] == 1:
                    try:
                        if hasattr(something[j],'dtype') and y[j].dtype != something[j].dtpye:
                            raise ValueError('wrong dtype')
                        y[j] = something[j]
                    except:
                        y[j] = numpy.empty( something[j].shape, dtype = something[j].dtype)
                        y[j] = something[j]
                else:
                    try:
                        if hasattr(something[j],'dtype') and y[j].dtype != something[j].dtype:
                            raise ValueError('worng dtype')
                        y[j][self.idx_store_steps[j]] = something[j]
                        self.idx_store_steps[j] = (self.idx_store_steps[j] + 1) % self.store_steps[j]
                    except:
                        y[j] = numpy.empty( (self.store_steps[j],)+something[j].shape, \
                                dtype = something[j].dtype)
                        y[j][idx_sotre_steps[j]] = something[j]
                        self.idx_store_steps[j] = (self.idx_store_steps[j] + 1) % self.store_steps[j]
        return y

    def grad(self, args, g_outs):

        raise NotImplementedError('This will be implemented in the near future');
        '''
        if True:
           #((self.updates.keys() != []) or (self.inplace_map.keys() != [])\
           # or numpy.any(self.store_steps)):
           # warning('Can not compute gradients if inplace or updates ' \
           #         'are used or if you do not keep past value of outputs.'\
           #         'Use force_gradient if you know for sure '\
           #         'that the gradient can be computed automatically.')
           warning('Gradient not fully tested yet !')
           return [None for i in args]
        else:
            # forward pass
            y = self(*args)
            if not( type(y) in (list,tuple)):
                y = [y]

        g_y = [outputs[0].type()]

        def compute_gradient(y, g_y):
            gmap = gradient.grad_sources_inputs( \
                        [(y,g_y)], gof.graph.inputs([y]), False)
            def zero(p):
              return tensor.TensorConstant(tensor.TensorType(\
                      dtype=p.type.dtype, broadcastable=[]),
                      safe_asarray._asarray(0,dtype = p.type.dtype))

            return [gmap.get(p, zero(p)) for p in inputs]


        i = 0
        while
        g_args = compute_gradient( outputs[0], g_y[-1])
        # for all outputs compute gradients and then sum them up
        for y in outputs[1:]:
            g_y += [y.type()]
            g_args_y = compute_gradient( y,g_y[-1])
            for i in xrange(len(g_args)):
                g_args[i] += g_args_y[i]


        self.g_ins = g_y+inputs
        self.g_outs = g_args


            # backwards pass
            for i in xrange(len(y)):
               if g_outs[i] == None:
                  g_outs[i] = tensor.zeros_like(y[i])

            g_args = [self.n_steps]+g_outs + y
            # check if go_backwards is true
            if self.go_backwards:
               for seq in args[1:self.n_seqs]:
                 g_args += [seq[::-1]]
            else:
               g_args += args[1:self.n_seqs]

            g_args += args[1+self.n_seqs: ]


            g_scan = ScanGrad((self.g_ins,self.g_outs), self.n_seqs, \
                              self.n_outs,self.seqs_taps, self.outs_taps,
                              self.truncate_gradient)

            return g_scan(g_args)
            '''


'''
class ScanGrad(Op):
    """Gradient Op for Scan"""
    def __init__(self,(g_ins, g_outs) , n_seqs, n_outs,
                 seqs_taps = {}, outs_taps= {}, truncate_gradient = -1):
        self.grad_fn = function(g_ins, g_outs)
        self.inputs = g_ins
        self.outputs = g_outs
        self.n_seqs = n_seqs
        self.truncate_gradient = truncate_gradient
        self.n_outs = n_outs
        self.seqs_taps = seqs_taps
        self.outs_taps = outs_taps
        self.destroy_map = {}


    def __eq__(self,other):
        rval = type(self) == type(other)
        if rval:
           rval = (self.inputs == other.inputs) and \
                  (self.outputs == other.outputs) and \
                  (self.n_seqs == other.n_seqs) and \
                  (self.n_outs == other.n_outs) and \
                  (self.truncate_gradient == other.truncate_gradient) and\
                  (self.seqs_taps == other.seqs_taps) and \
                  (self.outs_taps == other.outs_taps)
        return rval

    def __hash__(self):
        return hash(type(self)) ^ \
               hash(self.n_seqs) ^ \
               hash(self.n_outs) ^ \
               hash(self.truncate_gradient) ^\
               hash_list(self.inputs) ^ \
               hash_list(self.outputs) ^ \
               hash_dict(self.seqs_taps) ^ \
               hash_dict(self.outs_taps)

    def make_node(self, *args):
        # input of the gradient op :
        # | g_outs | y      | seqs   | outs    | non_seqs   |
        # | n_outs | n_outs | n_seqs | n_outs  | unknown    |
        # return
        # | grad of seqs | grad of outs | grad of non_seqs  |
        # |   n_seqs     |  n_outs      |  unknown          |
        return Apply(self, list(args),
                    [i.type() for i in args[1+2*self.n_outs:] ])

    def perform(self, node, args, storage):
            # get scan inputs
            n_steps = args[0]
            inputs = args[2*self.n_outs+1:]
            seqs = inputs[:self.n_seqs]
            seeds = inputs[self.n_seqs:self.n_seqs+self.n_outs]
            non_seqs = inputs[self.n_outs+self.n_seqs:]

            # generate space for gradient
            g_seqs     = [numpy.zeros_like(k) for k in seqs]
            g_seeds    = [numpy.zeros_like(k) for k in seeds]
            g_non_seqs = [numpy.zeros_like(k) for k in non_seqs]
            # get gradient from above
            g_outs = args[:self.n_outs]

            # get the output of the scan operation
            outs = args[self.n_outs:2*self.n_outs]


            # go back through time to 0 or n_steps - truncate_gradient
            lower_limit = n_steps - self.truncate_gradient
            if lower_limit > n_steps-1:
                the_range = xrange(n_steps-1,-1,-1)
            elif lower_limit < -1:
                the_range = xrange(n_steps-1,-1,-1)
            else:
                the_range = xrange(n_steps-1, lower_limit,-1)



            seqs_mins = {}
            for j in xrange(self.n_seqs):
              if self.seqs_taps.has_key(j):
                seqs_mins.update({j: min(self.seqs_taps[j])})

            outs_mins = {}
            seed_size = {}
            for j in xrange(self.n_outs):
              if self.outs_taps.has_key(j):
                outs_mins.update({j: min(self.outs_taps[j])})
                seed_size.update({j: g_seeds[j].shape[0]})

            for i in the_range:
              # time slice of inputs
              _ins = []
              for j in xrange(self.n_seqs):
                if self.seqs_taps.has_key(j):
                  ls_taps = self.seqs_taps[j]
                  min_tap =      seqs_mins[j]
                  for tap_value in ls_taps:
                    k = i - min_tap + tap_value
                    _ins += [ins[j][k]]
              # time slice of outputs + taps
              _outs = []
              for j in xrange(self.n_outs):
                if self.outs_taps.has_key(j):
                  ls_taps = self.outs_taps[j]
                  min_tap =      outs_mins[j]
                  seed_sz =      seed_size[j]
                  for tap_value in ls_taps:
                    if i + tap_value < 0:
                      k = i + seed_sz  + tap_value
                      if k < 0 :
                        #past value not provided .. issue a warning and use 0
                        _outs += [numpy.zeros(seeds[j][0].shape)]
                        warning('Past value %d for output $d not given' \
                              %(j,tap_value))
                      else:
                        _outs += [seeds[j][k]]
                    else:
                      _outs += [outs[j][i + tap_value]]

              g_out = [arg[i] for arg in g_outs]
              grad_args = g_out + _ins + _outs + non_seqs
              grads=self.grad_fn(*grad_args)

              # get gradient for inputs
              pos = 0
              for j in xrange(self.n_seqs):
                if self.seqs_taps.has_key(j):
                  ls_taps = self.seqs_taps[j]
                  min_tap =      seqs_mins[j]
                  for tap_value in ls_taps :
                    k = i - min_tap + tap_value
                    g_ins[j][k] += grads[pos]
                    pos += 1


              # get gradient for outputs
              for j in xrange(self.n_outs):
                if self.outs_taps.has_key(j):
                  ls_taps = self.outs_taps[j]
                  min_tap =      outs_mins[j]
                  seed_sz =      seed_size[j]
                  for tap_value in ls_taps:
                    if i+tap_value < 0 :
                     k = i + seed_sz + tap_value
                     if  k > 0 :
                        g_seeds[j][k] += grads[pos]
                        pos += 1
              for j in xrange(len(g_non_seqs)):
                g_non_seqs[j] += grads[j+pos]


            # return the gradient

            for i,v in enumerate(g_ins + g_seeds+ g_non_seqs):
                storage[i][0] = v
    '''



class ScanSpaceOptimizer(Optimizer):
    """ Graph Optimizer that reduces scan memory consumption """
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self,env):
        env.extend(toolbox.ReplaceValidate())

    def apply(self, env):
        nodelist = list(env.toposort())
        for node in nodelist:
            op = node.op
            # If it is a scan Op
            if isinstance(op, Scan):
                outputs = node.outputs
                store_steps = [0 for x in outputs]
                # check the otuputs
                for i,out in enumerate(node.outputs):
                    if op.store_steps[i] == 0 :
                        # if we do not have a range for this output
                        req_steps = numpy.max(numpy.abs(op.outs_taps.get(i,1)))
                        # look at all its clients
                        for cl,_dx in out.clients:
                            if type(cl) == str:
                                # if the node is actually an output, then 
                                # we need to store the entire thing 
                                req_steps = 0
                                break
                            else:
                                if not isinstance(cl.op,
                                        tensor.basic.Subtensor):
                                    # if any of the clients is not a subtensor
                                    # we also need to store the enitre thing
                                    req_steps = 0
                                    break
                                else:
                                    # if it is a tensor, and the first 
                                    # dimension is just -1 
                                    if cl.op.idx_list[0] == -1 :
                                                req_steps = numpy.max([1, req_steps])
                                    else:
                                        # or a constant that evaluates to 
                                        # -1
                                        try:
                                            idx = opt.get_constant_value(cl.op.idx_list[0])
                                            if idx== -1:
                                                req_steps = numpy.max([1, req_steps])
                                            else:
                                                req_steps = 0
                                                break
                                        except:
                                            req_steps = 0
                                            break
                        store_steps[i] = req_steps
                    else:
                        store_steps[i] = op.store_steps[i]
                if numpy.any(store_steps!= op.store_steps):
                    new_scan = Scan((op.inputs, op.outputs, op.givens, 
                        op.slice_to_seqs),op.n_seqs, op.n_outs,
                        op.inplace_map, op.seqs_taps, op.outs_taps, 
                        op.truncate_gradient, op.go_backwards,
                        store_steps, op.mode,op.n_fixed_steps, 
                        op.inplace).make_node(*node.inputs)
                    # we not need to replace the outputs of scan
                    for i,out in enumerate(node.outputs):
                        # if we are dealing with an output for which 
                        # we changed the number of stored steps we 
                        # also need to get rid off the subtensor
                        if op.store_steps[i] == 0 and store_steps[i] == 1:
                            # get the output of the subtensor variables 
                            outSubTens = [ x[0].outputs[0] for x in out.clients ]
                            new_old = [(x,new_scan.outputs[i]) for x in outSubTens]
                            env.replace_all_validate(new_old,reason = 
                            'scan_space_optimizer')
                        else:
                            env.replace_all_validate([(out,
                                new_scan.outputs[i])], reason =
                                'scan_space_optimizer')


optdb.register('scanOp_space_optimization', ScanSpaceOptimizer(), 74, 'fast_run')

@gof.local_optimizer([None])
def scan_make_inplace(node):
    op = node.op
    if isinstance(op, Scan) and (not op.inplace) and (op.inplace_map.keys() != []):
        return Scan((op.inputs, op.outputs, op.givens, op.slice_to_seqs ) , op.n_seqs,
            op.n_outs, op.inplace_map, op.seqs_taps, op.outs_taps,
            op.truncate_gradient, op.go_backwards, op.store_steps, op.mode,
            op.n_fixed_steps, inplace=True ).make_node(*node.inputs).outputs
    return False


optdb.register('scanOp_make_inplace', opt.in2out(scan_make_inplace,
    ignore_newtrees=True), 75, 'fast_run', 'inplace')


class ScanRemoveFromGraph(Optimizer):
    ''' Graph Optmizer that removes scan if you just do a loop of 1 '''
    def __init__(self):
        Optimizer.__init__(self)
    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())
    def apply(self,env):
        nodelist = list(env.toposort())
        for node in nodelist:
            op = node.op
            # If it is a scan Op
            if isinstance(op, Scan) and op.n_fixed_steps != None:
                if abs(op.n_fixed_steps) < 2:
                    # Step 1 replace the inputs of the inner function 
                    #        with the inputs of scan

                    # Start replacing
                    # idx_curr_inp -> index that goes through the extended
                    # inputs of the op (includes shared variables) that are 
                    # not provided to the node as inputs !!
                    idx_curr_inp = -1
                    # keeps track of what slice of the current input we are
                    # currently dealing with
                    slice = -1
                    # keeps track of the index that goes through the actual
                    # inputs of the node
                    idx_node_inp = 0
                    # pairs of variables that we need to replace in the end
                    replace_pairs = {}
                    # go through the inputs of the inner function
                    for i,inp in enumerate(op.inputs):
                        # figure what what slice of what node input this represents
                        if i < len(op.slice_to_seqs):
                            # slice_to_seqs is an array of the form [1 1 2 3 3 3 ], 
                            # meaning that the 1st input of the inner function is a 
                            # slice of the 1st input of scan, 2nd input of the inner 
                            # function is a slice of the 1st input of scan and so on..
                            arg = op.slice_to_seqs[i]
                            # check if this is a slice of the current input
                            if arg == idx_curr_inp:
                                # if so increase the number of the current slice
                                slice+= 1
                            else:
                                # if not reset slice, make this the new current 
                                # input
                                slice = 0
                                idx_curr_inp = arg
                                # and check if it is a shared variables
                                # scan deals with shared variables by replacing them 
                                # with copies using the given argument of theano.function
                                # so if we have a shared variable it should appear in 
                                # op.givens !!
                                if inp not in op.givens:
                                    # if it is not a shared variable increase the index
                                    # of the current input 
                                    # note that we will jump to 1; this is fine since 
                                    # node.inputs[0] is the number of steps, which we 
                                    # should not consider here .. we care of what follows
                                    # namely the sequences, initial states, non sequences...
                                    idx_node_inp += 1
                            if inp not in op.givens:
                                # This is not a shared variable so we can replace it 
                                # ( we should not replace the shared variables, theano.function
                                # will take care of shared variables here ..)
                                if idx_curr_inp >= op.n_seqs:
                                    # we are dealing with a initial state of some output
                                    # check if we are dealing with a 1 past tap output
                                    one_step = False
                                    if not op.outs_taps.has_key(idx_curr_inp-op.n_seqs):
                                        one_step = True
                                    else:
                                        if op.outs_taps[idx_curr_inp - op.n_seqs] == [-1]:
                                            one_step = True

                                    if one_step:
                                        node_input = node.inputs[idx_node_inp]
                                    else:
                                        tap = op.outs_taps[idx_curr_inp-op.n_seqs][slice]
                                        min_tap = min(op.outs_taps[idx_curr_inp-op.n_seqs])
                                        node_input = node.inputs[idx_node_inp][tap-min_tap]
                                else:
                                    # we are dealing with a slice of a sequence
                                    tap = op.seqs_taps[idx_curr_inp][slice]
                                    min_tap = min(op.seqs_taps[idx_curr_inp])
                                    node_input = node.inputs[idx_node_inp][tap-min_tap]
                                # add to our replace_pairs list
                                replace_pairs[inp] = node_input
                        else:
                            # if we got here this means we are dealing with non_sequences, 
                            # which do not have slices !
                            # check to see if we are dealing with a shared variable
                            if inp not in op.givens:
                                idx_node_inp += 1
                                replace_pairs[inp] = node.inputs[idx_node_inp]



                    def my_replace( node, replace_pairs):
                        # Turns out that using env replace (while safe) is 
                        # a real pain because of many condition that have to
                        # be met which I can not met while doing the 
                        # replacement, so I did my little hack that does 
                        # something like a replacement 
                        # ASSUMPTIONS:
                        #   we do not do anything crazy like replacing x 
                        #   with something in terms of x ! 
                        #
                        #   we do not have envs or anything, just a simple
                        #   computational graph that has not been compiled 
                        #   yet
                        if node:
                            for i,inp in enumerate(node.inputs):
                                if inp in replace_pairs:
                                    node.inputs[i] = replace_pairs[inp]
                                else:
                                    inp.owner = my_replace(inp.owner, replace_pairs)
                            return node
                        else:
                            return node
                    my_outs = op.outputs
                    for i, out in enumerate(my_outs):
                        my_outs[i].owner = my_replace(out.owner, replace_pairs)

                    for idx in xrange(len(my_outs)):
                        t = my_outs[idx]
                        nwout = tensor.Rebroadcast((0,False))(tensor.shape_padleft(t))
                        env.replace(node.outputs[idx],nwout)
                    # we are done ...



# is 30 soon enough !? I want to do it as early as possible .. such that 
# the new graph gets optimized
optdb.register('scanOp_remove_from_graph', ScanRemoveFromGraph() , 30, 'fast_run')
