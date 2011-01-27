"""
This module provides the Scan Op

Scanning is a general form of recurrence, which can be used for looping.
The idea is that you *scan* a function along some input sequence, producing
an output at each time-step that can be seen (but not modified) by the
function at the next time-step. (Technically, the function can see the
previous K  time-steps of your outputs and L time steps (from the past and
future) of your inputs.

So for example, ``sum()`` could be computed by scanning the ``z+x_i``
function over a list, given an initial state of ``z=0``.

Special cases:

* A *reduce* operation can be performed by returning only the last
  output of a ``scan``.
* A *map* operation can be performed by applying a function that
  ignores previous steps of the outputs.

Often a for-loop can be expressed as a ``scan()`` operation, and ``scan`` is
the closest that theano comes to looping. The advantage of using ``scan``
over for loops is that it allows the number of iterations to be a part of
the symbolic graph.

The Scan Op should typically be used by calling any of the following
functions: ``scan()``, ``map()``, ``reduce()``, ``foldl()``,
``foldr()``.
"""
__docformat__ = 'restructedtext en'
__authors__ = ( "Razvan Pascanu "
                "Frederic Bastien "
                "James Bergstra "
                "Pascal Lamblin "  )
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import logging
import numpy

import theano
import tensor
import misc.safe_asarray as safe_asarray
from tensor import opt, TensorType
import gof
from gof import Optimizer, toolbox, Op, Apply, Variable
from compile import optdb, SharedVariable, function, Param
import compile
import gradient
from gof.python25 import all


# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan')


def warning(*msg):
    _logger.warning('WARNING theano.scan: '+' '.join(msg))


def info(*msg):
    _logger.info('INFO theano.scan: '+' '.join(msg))


# Hashing a dictionary/list/tuple by xoring the hash of each element
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


# The ``map`` view of Scan Op.
def map( fn
        , sequences
        , non_sequences     = None
        , truncate_gradient = -1
        , go_backwards      = False
        , mode              = None
        , name              = None  ):
    """
    Similar behaviour as python's map.

    :param fn: The function that ``map`` applies at each iteration step
               (see ``scan`` for more info).

    :param sequences: List of sequences over which ``map`` iterates
                      (see ``scan`` for more info).

    :param non_sequences: List of arguments passed to ``fn``. ``map`` will
                          not iterate over these arguments (see ``scan`` for
                          more info).

    :param truncate_gradient: See ``scan``.

    :param go_backwards: Boolean value that decides the direction of
                         iteration. True means that sequences are parsed
                         from the end towards the begining, while False
                         is the other way around.

    :param mode: See ``scan``.

    :param name: See ``scan``.
    """
    return scan( fn                 = fn
                , sequences         = sequences
                , outputs_info      = []
                , non_sequences     = non_sequences
                , truncate_gradient = truncate_gradient
                , go_backwards      = go_backwards
                , mode              = mode
                , name              = name )


# The ``reduce`` view of Scan Op.
def reduce( fn
           , sequences
           , outputs_info
           , non_sequences = None
           , go_backwards  = False
           , mode          = None
           , name          = None ):
    """
    Similar behaviour as python's reduce

    :param fn: The function that ``reduce`` applies at each iteration step
               (see ``scan``  for more info).

    :param sequences: List of sequences over which ``reduce`` iterates
                      (see ``scan`` for more info)

    :param outputs_info: List of dictionaries describing the outputs of
                        reduce (see ``scan`` for more info).

    :param non_sequences: List of arguments passed to ``fn``. ``reduce`` will
                          not iterate over these arguments (see ``scan`` for
                          more info).

    :param go_backwards: Boolean value that decides the direction of
                         iteration. True means that sequences are parsed
                         from the end towards the begining, while False
                         is the other way around.

    :param mode: See ``scan``.

    :param name: See ``scan``.
    """
    # Makes sure the outputs_info is a list.
    if type(outputs_info) not in (list,tuple):
        outs_info = [outputs_info]
    else:
        outs_info = list(outputs_info)

    for i,out_info in enumerate(outs_info):
        if out_info:
            if not type(out_info) == dict:
                # Specifies that it should return only the last step.
                outs_info[i] = dict(
                    initial = out_info,  return_steps = 1, store_steps = 1)
            else:
                # Specifies that it should return only the last step.
                outs_info[i]['store_steps']  = 1
                outs_info[i]['return_steps'] = 1
                # NOTE : If the user asks for more then the last step,
                # it means he does not understand ``reduce``. We could
                # issue a warning in that case
    return scan( fn                 = fn
                , sequences         = sequences
                , outputs_info      = outs_info
                , non_sequences     = non_sequences
                , go_backwards      = go_backwards
                , truncate_gradient = 1
                , mode              = mode
                , name              = name )


# The ``foldl`` view of Scan Op.
def foldl( fn
          , sequences
          , outputs_info
          , non_sequences = None
          , mode          = None
          , name          = None  ):
    """
    Similar behaviour as haskell's foldl

    :param fn: The function that ``foldl`` applies at each iteration step
               (see ``scan`` for more info).


    :param sequences: List of sequences over which ``foldl`` iterates
                      (see ``scan`` for more info)

    :param outputs_info: List of dictionaries describing the outputs of
                        reduce (see ``scan`` for more info).

    :param non_sequences: List of arguments passed to `fn`. ``foldl`` will
                          not iterate over these arguments (see ``scan`` for
                          more info).

    :param mode: See ``scan``.

    :param name: See ``scan``.
    """
    return reduce( fn             = fn
                  , sequences     = sequences
                  , outputs_info  = outputs_info
                  , non_sequences = non_sequences
                  , go_backwards  = False
                  , mode          = mode
                  , name          = name )


# The ``foldl`` view of Scan Op.
def foldr( fn
          , sequences
          , outputs_info
          , non_sequences = None
          , mode          = None
          , name          = None ):
    """
    Similar behaviour as haskell' foldr

    :param fn: The function that ``foldr`` applies at each iteration step
               (see ``scan`` for more info).


    :param sequences: List of sequences over which ``foldr`` iterates
                      (see ``scan`` for more info)

    :param outputs_info: List of dictionaries describing the outputs of
                        reduce (see ``scan`` for more info).

    :param non_sequences: List of arguments passed to `fn`. ``foldr`` will
                          not iterate over these arguments (see ``scan`` for
                          more info).

    :param mode: See ``scan``.

    :param name: See ``scan``.
    """
    return reduce( fn             = fn
                  , sequences     = sequences
                  , outputs_info  = outputs_info
                  , non_sequences = non_sequences
                  , go_backwards  = True
                  , mode          = mode
                  , name          = name )

#
# QUESTION:
#   If the larger (in absolute values) the sequence_taps, the shorter the output
#   right?  If the sequence_taps = {0: [-10, 10]}, and I pass an input with 22
#   rows, then the scan will output something of length <=2 right?
#
# ANSWER:
#   Yes, actually it will be exactly 2 ( if there are no other constraints)


def scan( fn
         , sequences         = None
         , outputs_info      = None
         , non_sequences     = None
         , n_steps           = None
         , truncate_gradient = -1
         , go_backwards      = False
         , mode              = None
         , name              = None ):
    """
    This function constructs and applies a Scan op to the provided
    arguments.

    :param fn:
        ``fn`` is a function that describes the operations involved in one step
        of ``scan``. ``fn`` should construct variables describing the output of
        one iteration step. It should expect as input theano variables
        representing all the time slices of the input sequences and outputs,
        and all other arguments given to scan as ``non_sequences``. The order
        in which scan passes this variables to ``fn``  is the following :

        * all time slices of the first sequence
        * all time slices of the second sequence
        * ...
        * all time slices of the last sequence
        * all time slices of the first output
        * all time slices of the second otuput
        * ...
        * all time slices of the last output
        * all other arguments (the list given as `non_sequences` to
            scan)

        The order of the sequences is the same as the one in the list
        `sequences` given to scan. The order of the outputs is the sane
        as the order of ``output_info``. For any sequence or output the
        order of the time slices is the same as the order of the time
        taps provided. For example if one writes the following :

        .. code-block:: python

            scan(fn, sequences = [ dict( Sequence1, taps = [-3,2,-1])
                                 , Sequence2
                                 , dict( Sequence3, taps = 3) ]
                   , outputs_info = [ dict( Output1, taps = [-3,-5])
                                    , dict( Output2, taps = None)
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
        code we recommend though to provide them to scan.

        The function is expected to return two things. One is a list of
        outputs ordered in the same order as ``outputs_info``, with the
        difference that there should be only one output variable per
        output initial state (even if no tap value is used). Secondly
        `fn` should return an update dictionary ( that tells how to
        update any shared variable after each iteration ste). The
        dictionary can optionally be given as a list of tuples. There is
        no constraint on the order of these two list, ``fn`` can return
        either ``(outputs_list, update_dictionary)`` or ``(update_dictionary,
        outputs_list)`` or just one of the two (in case the other is
        empty).


    :param sequences:
        ``sequences`` is the list of Theano variables or dictionaries
        describing the sequences ``scan`` has to iterate over. If a
        sequence is given as wrapped in a dictionary a set of optional
        information can be provided about the sequence. The dictionary
        should have the following keys:

        * ``input`` (*mandatory*) -- Theano variable representing the
          sequence.

        * ``taps`` -- Temporal taps of the sequence required by ``fn``.
          They are provided as a list of integers, where a value ``k`` implies
          that at iteration step ``t`` scan will pass to ``fn`` the slice
          ``t+k``. Default value is ``[0]``

        Any Theano variable in the list ``sequences`` is automatically
        wrapped into a dictionary where ``taps`` is set to ``[0]``.


    :param outputs_info:
        ``outputs_info`` is the list of Theano variables or dictionaries
        describing the initial state of the outputs computed
        recurrently. When this initial states are given as dictionary
        optional information can be provided about the output corresponding
        to these initial states. The dictionary should have the following
        keys:

        * ``initial`` -- Theano variable that represents the initial
          state of a given output. In case the output is not computed
          recursively (think of a map) and does not require an initial
          state this field can be skipped. Given that only the previous
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
          ``output[-5]``; ``init_y[1]`` *correponds to* ``output[-4]``;
          ``init_y[2]`` corresponds to ``output[-3]``; ``init_y[3]``
          coresponds to ``output[-2]``; ``init_y[4]`` corresponds to
          ``output[-1]``. While this order might seem strange, it comes
          natural from splitting an array at a given point. Assume that
          we have a array ``x``, and we choose ``k`` to be time step
          ``0``. Then our initial state would be ``x[:k]``, while the
          output will be ``x[k:]``. Looking at this split, elements in
          ``x[:k]`` are ordered exactly like those in ``init_y``.
        * ``taps`` -- Temporal taps of the output that will be passed to
          ``fn``. They are provided as a list of *negative* integers,
          where a value ``k`` implies that at iteration step ``t`` scan will
          pass to ``fn`` the slice ``t+k``.
        * ``inplace`` -- One of the Theano variables provided as
          ``sequences``. ``scan`` will try to compute this output *in
          place* of the provided input *iff* it respects the following
          constraints:

            * There is no other output that is denied to be computed in
              place for whatever reason.

            * ``fn`` is not using past taps of the input sequence that
              will get overwritten by the output

        * ``return_steps`` -- Integer representing the number of steps
          to return for the current steps. For example, if ``k`` is
          provided, ``scan`` will return ``output[-k:]``. This is meant as a
          hint, based on ``k`` and the past taps of the outputs used, scan
          can be smart about the amount of memory it requires to store
          intermediate results. If not given, or ``0``, ``scan`` will return
          all computed steps.
        * ``store_steps`` -- Integer representing the number of
          intermediate steps ``scan`` should use for a given output. Use
          this key only if you really know what you are doing. In general
          it is recommended to let scan decide for you the ammount of memory
          it should use.

        ``scan`` will follow this logic if partial information is given:

        * If an output is not wrapped in a dictionary, ``scan`` will wrap
          it in one assuming that you use only the last step of the output
          (i.e. it makes your tap value list equal to [-1]) and that it is
          not computed inplace.
        * If you wrap an output in a dictionary and you do not provide any
          taps but you provide an initial state it will assume that you are
          using only a tap value of -1.
        * If you wrap an output in a dictionary but you do not provide any
          initial state, it assumes that you are not using any form of
          taps.
        * If you provide a ``None`` instead of a variable or a dictionary
          ``scan`` assumes that you will not use any taps for this output
          (like for example in case of a map)

        If ``outputs_info`` is an empty list or None, ``scan`` assumes
        that no tap is used for any of the otuputs. If information is
        provided just for a subset of the outputs an exception is
        raised (because there is no convention on how scan should map
        the provided information to the outputs of ``fn``)


    :param non_sequences:
        ``non_sequences`` is the list of arguments that are passed to
        ``fn`` at each steps. One can opt to exclude shared variables
        used in ``fn`` from this list.


    :param n_steps:
        ``n_steps`` is the number of steps to iterate given as an int
        or Theano scalar. If any of the input sequences do not have
        enough elements, scan will produce a warning and run only for
        the maximal amount of steps it can. If the *value is 0* the
        outputs will have *0 rows*. If the value is negative, ``scan``
        run backwards in time. If the ``go_backwards`` flag is already
        set and also ``n_steps`` is negative, ``scan`` will run forward
        in time. If ``n_steps`` is not provided, or evaluates to ``None``,
        ``inf`` or ``NaN``, then ``scan`` will figure out the amount of
        steps it should run given its input sequences.


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
        When profiling ``scan`` it is crucial to provide a name for any
        instance of ``scan``. The profiler will produce an overall
        profile of your code as well as profiles for doing one iteration
        step for each instance of ``scan``. The ``name`` of the instance is
        how you differentiate between all these profiles.


    :param mode:
        It is recommended to leave this argument to None, especially
        when profiling ``scan`` (otherwise the results are not going to
        be accurate). If you prefer the computations of one step of
        ``scan`` to be done differently compared to the entire function, set
        this parameters (see ``theano.function`` for details about
        possible values and their meaning).


    :rtype: tuple
    :return: tuple of the form (outputs, updates); ``outputs`` is either a
             Theano variable or a list of Theano variables representing the
             outputs of ``scan`` (in the same order as in
             ``outputs_info``). ``updates`` is a dictionary specifying the
             update rules for all shared variables used in the scan
             operation. This dictionary should be passed to ``theano.function``
             when you compile your function.
    """
    # General observation : this code is executed only once, at creation
    # of the computational graph, so we don't yet need to be smart about
    # anything ( to speed things up)

    # check if inputs are just single variables instead of lists
    if not (type(sequences) in (list, tuple)) and sequences != None:
        seqs = [sequences]
    elif sequences == None:
        seqs = []
    else:
        seqs = sequences

    if not (type(outputs_info) in (list,tuple)) and outputs_info != None:
        outs_info = [outputs_info]
    elif outputs_info == None:
        outs_info = []
    else:
        outs_info = outputs_info

    if ( not (type(non_sequences) in (list,tuple))
        and non_sequences != None):
        non_seqs = [non_sequences]
    elif non_sequences == None:
        non_seqs = []
    else:
        non_seqs = non_sequences


    # If we provided a known number of steps ( before compilation)
    # and if that number is 1 or -1, then we can skip the Scan Op,
    # and just apply the inner function once
    # To do that we check here to see the nature of n_steps
    if type(n_steps) in (float,int):
        n_fixed_steps = int(n_steps)
    else:
        # also check if this value happens to be a constant,
        # then we could do the same
        try :
            n_fixed_steps = opt.get_constant_value(n_steps)
        except:
            n_fixed_steps = None
    # compute number of sequences and number of outputs
    n_seqs = len(seqs)
    n_outs = len(outs_info)
    # initialize the inplace map, sequences map and
    # outputs map
    ''' Details:
            The scan op identifies different properties attached
            to input tensors by their order in the input list.
            These maps ( inplace, sequence_taps, output_taps,
            store_steps, return_steps) go from the index of an input to
            its properties. Note that inputs are always first, followed
            by outputs. Since we always know the number of inputs we
            index the outputs from 0 ( so sometimes you will need to
            do something like outputs_taps[i-n_ins]
    '''
    inplace_map    = {}
    sequences_taps = {}
    outputs_taps   = {}

    # Assume that for any output we want to store everythin that it produces
    store_steps = []
    return_steps = {}

    # wrap sequences in a dictionary if they are not already dictionaries
    # in the same pass create a sequences_taps dictionary
    for i in xrange(n_seqs):
        if not type(seqs[i]) == dict :
            # if it is not a dictionary make it into one
            seqs[i] = dict(input=seqs[i], taps=[0])
        # see if taps values are provided as a list
        elif seqs[i].get('taps',None):
            # users can optionally provide the past value (if is just
            # one) as a number instead of a list. Wrap it in a list
            # to have a uniform way of dealing with inputs later on
            if not type(seqs[i]['taps']) in (tuple,list):
                seqs[i]['taps'] = [seqs[i]['taps']]
        else:
            # See if the user actually provided the None value to taps,
            # which would indicate that the sequence was provided but
            # not used by the internal function; Only if the user has
            # not provided anything add the defaul [0]
            # Possible reason to provide a squence and not use it  is
            # if you want to compute the output
            # inplace of this input; it is a very unlikely behaviour but
            # we do want to cover it for completeness
            if not seqs[i].has_key('taps'):
                seqs[i][taps] = [0]
        # Now that our input is well behaved, collect the taps in the
        # sequences_taps map that we will use later in the body of scan
        # since inputs will be just tensors there
        if seqs[i].get('taps',None):
            sequences_taps[i] = seqs[i]['taps']

    # wrap outputs info in a dictionary if they are not already
    # in one and in the same pass create a init_outs_taps dictionary and a inplace map
    for i in xrange(n_outs):
        if outs_info[i]:
            # If output is a dictionary, collect the number of steps the
            # user would like scan to return
            if type(outs_info[i]) == dict:
                if outs_info[i].get('return_steps', None):
                    return_steps[i] = outs_info[i]['return_steps']
                # If you provide the number of steps to store internally,
                # (not advocated in the user documentation), then also
                # make sure you are returning only those number of steps
                if outs_info[i].get('store_steps', None):
                    store_steps += [outs_info[i].get('store_steps',None)]
                    return_steps[i] = outs_info[i].get('store_steps',None)
                else:
                    store_steps += [0]
            else:
                store_steps += [0]

            # trying to collect taps of the output
            if not type(outs_info[i]) == dict:
                # by default any output has a tap value of -1
                outs_info[i] = dict(initial=outs_info[i], taps = [-1])
                # if there is no initial state but there are taps
                # then return an error because it makes no sense
            elif (not outs_info[i].get('initial',None)) and \
                    (outs_info[i].get('taps',None)):
                raise ValueError('If you are using slices of an output you need to '\
                        'provide an initial state for it', outs_info[i])
                # if there is an intial state but no tap, we will add the default value
                # for taps, namely [-1] ( previous value); not that this will happen
                # even though you have provided for taps the value None, which is a bit
                # strange (why would one provide an initial state but tell scan not to
                # use it ? ), just that in that case we will throw in a warning message
                # pointing out this inconsistency
            elif outs_info[i].get('initial',None) and \
                    ( not outs_info[i].get('taps',None)):
                if outs_info[i].has_key('taps'):
                    warning('You are providing an initial state for an output and then '
                        'tell scan not to use it. Why? Scan will overwrite this setting'
                        ' and use the previous value of the provided initial state. If'
                        ' this is not what you wanted, check your code and do not '
                        'provide the initial state')
                outs_info[i]['taps'] = [-1]
        else:
            # if the output is a None then replace it with an empty dictionary for
            # easing up dealing with this case later one ( we can directly call .has_key
            # and things like this
            outs_info[i] = dict()
            store_steps += [0]

        if outs_info[i].get('taps', None):
            # Create a separate outputs_taps dictionary with all the outputs taps; This
            # is how the Scan Op expects this information, separeted from the variables
            outputs_taps[i] = outs_info[i]['taps']
        if outs_info[i].get('inplace', None):
            # The same is true for the inplace info; it has to go into a separate
            # dictionary based on index; Note that the input we're replacing should also
            # come as an index, therefore we have to look for it at this point
            found = None
            for k in xrange(n_seqs):
                if seqs[k].get('input', None) == outs_info[i].get('inplace',None):
                    found = k
            if found != None:
                # NOTE : inplace_map is identical to destroy_map, i.e. it tells what
                #        output is computed inplace of what input !!
                inplace_map[i] = found
            else:
                raise ValueError('Asked to compute in place of a non-input variable',\
                          outs_info[i].get('inplace', None))

    # create theano inputs for the recursive function
    # note : this is a first batch of possible inputs that will
    #        be compiled in a dummy function; we used this dummy
    #        function to detect shared variables and their updates
    #        and to construct a new and complete list of inputs and outputs
    args = []                       # list of arguments
    dummy_notshared_ins = 0         # number of arguments corresponding to input seqs
    dummy_notshared_init_outs = 0   # number of arguments corresponding to output seqs
    slice_to_seqs = []              # for each slice index of the corresponding input
    # go through sequences picking up time slices as needed
    for i,seq in enumerate(seqs):
        # Note that you can have something like no taps for
        # a sequence, though is highly unlikely in practice
        if seq.get('taps', None):
            # go through the indicated slice
            mintap = numpy.min(seq['taps'])
            for k in seq['taps']:
                # create one slice of the input
                '''
                    Later on, if we decide not to use scan because we are going
                    for just one step, it makes things easier if we compute the
                    correct outputs here. This way we can use the output of the
                    lambda expression directly to replace the output of scan.

                    If not we need to use copies, that will be replaced at each
                    frame by the corresponding slice
                '''
                if n_fixed_steps not in [1,-1]:
                    nw_slice = seq['input'][0].type()
                elif n_fixed_steps == 1:
                    nw_slice = seq['input'][k-mintap]
                else:
                    nw_slice = seq['input'][-1+mintap-k]
                # Add names to slices for debugging and pretty printing ..
                # that is if the input already has a name
                if seq['input'].name:
                    if seq['taps'][k] > 0:
                        nw_slice.name = seq['input'].name + '[t+%d]'%seq['taps'][k]
                    elif seq['taps'][k] == 0:
                        nw_slice.name = seq['input'].name + '[t]'
                    else:
                        nw_slice.name = seq['input'].name + '[t%d]'%seq['taps'][k]
                args.append(nw_slice)
                # Specify to whom this slice belongs
                slice_to_seqs.append(i)
            # Any slice is not a shared variable, even though the sequence
            # from where we pick the slices is shared, therefore we should
            # increase the number of notshared inputs to the dummy function
            # by the number of slices
            dummy_notshared_ins += len(seq['taps'])
    # go through outputs picking up time slices as needed
    for i,init_out in enumerate(outs_info):
        # Note that our convention dictates that if an output uses
        # just the previous time step, as an initial state we will only provide
        # a tensor of the same dimension as one time step; This makes code
        # much cleaner for those who do not use taps. Otherwise they would
        # always had to shape_pad_left the initial state .. which is ugly
        if init_out.get('taps', None) == [-1]:
            if n_fixed_steps in [-1,1]:
                args += [init_out['initial']]
            else:
                args += [init_out['initial'].type()]
            # Added name to slices for debugging and pretty printing
            if init_out['initial'].name:
                args[-1].name = init_out['initial'].name+'[t-1]'
            # we need to specify in slice_seqs to which output this
            # slice belongs; Because we might get confused afterwards
            # if a number is an index of a sequence or an output, and
            # because we do not want to create yet another list, we will
            # add the number of sequences + the current output. This makes
            # decoding easy and spares us from writing a lot of lines
            slice_to_seqs += [ i+n_seqs ]
            dummy_notshared_init_outs += 1
        elif init_out.get('taps',None):
            if numpy.any(numpy.array(init_out.get('taps',[])) > 0):
                # Make sure we do not have requests for future values of a sequence
                # we can not provide such values
                raise ValueError('Can not use future taps of outputs', init_out)
            # go through the taps
            minstep = abs(numpy.min(init_out['taps']))
            for k in init_out['taps']:
                # create a new slice
                if n_fixed_steps in [1,-1]:
                    nw_slice = init_out['initial'][k+minstep]
                else:
                    nw_slice = init_out['initial'][0].type()
                # give it a name or debugging and pretty printing
                if init_out['initial'].name:
                    if k > 0:
                        nw_slice.name = init_out['initial'].name + '[t+%d]'%k
                    elif k == 0:
                        nw_slice.name = init_out['initial'].name + '[t]'
                    else:
                        nw_slice.name = init_out['initial'].name + '[t%d]'%k
                args.append(nw_slice)
                # indicate the output index + n_seqs ( see above why)
                slice_to_seqs.append(i + n_seqs)
            # add as many slices as there are taps
            dummy_notshared_init_outs += len(init_out['taps'])
        #NOTE: there is another case, in which we do not want to provide any previous
        # value of the output to the inner case; in this case we do not have to do
        # anything ..

    # remove shared variables from the non sequences list
    # such that we can compile the function ( the user has the option to add them when
    # writing scan, because in some situations this might make the code more readable)
    notshared_other_args = []
    for non_seq in non_seqs:
        if not isinstance(non_seq, SharedVariable):
            notshared_other_args += [non_seq]

    # add only the not shared variables to the arguments of the dummy
    # function [ a function should not get shared variables as input ]
    dummy_args = []
    for arg in args:
        if not isinstance(arg, SharedVariable):
            dummy_args += [arg]
    dummy_args += notshared_other_args
    # arguments for the lambda expression that gives us the output
    # of the inner function
    args += non_seqs

    # when we apply the lambda expression we get a mixture of update rules
    # and outputs that needs to be separated
    outputs_updates  = fn(*args)
    # The code that follows tries to be as flexible as possible allowing the
    # user to return the output and updates in any order, and giving the updates
    # however he wants ( as a dictionary or a list o pairs ..)
    # Is there a way to compress all this by writing it in a more python/functional way?
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

    # If you return numbers (highly unlikely) this will not go well for theano
    # We need to convert them to Theano constants
    for i,out in enumerate(outputs):
        outputs[i] = tensor.as_tensor(out)

    # We can now compile a dummy function just to see what shared variable
    # we have and what are their update rules (note that the user has
    # the option not to pass the shared variable to scan, so we need to
    # pick them manually and add them to scan)
    # make the compilation as fast as possible by not applying any optimization
    # or conversion to C [ note this region is not important for performance
    # so we can do stuff as unoptimal as we wish ]
    if n_fixed_steps in [-1,1]:
        ''' We do have a special case here, namely is so might happen that
        whatever we have in dummy_args is not sufficient to compile the
        function( i.e. missing inputs). Furthermore we might not even need
        to compile the function here for this special case. But due to the
        way I wrote the code is easier to have a compiled function here
        that I can ignore later. Plus it is easier this way to take care
        of shared variables with non-default updates. Therefore only for
        this case I need to use gof.graph.inputs to look for the real inputs
        so that I can compile the function. RP '''
        dummy_f = function(filter(lambda x: isinstance(x, gof.Variable) and \
            not isinstance(x,SharedVariable) and not isinstance(x,gof.Constant), \
            gof.graph.inputs(dummy_args)), outputs, updates = updates, mode = compile.mode.Mode(linker='py',optimizer=None))
    else:

        dummy_f = function(filter(lambda x: isinstance(x, gof.Variable) and \
            not isinstance(x,SharedVariable) and not isinstance(x,gof.Constant), \
            dummy_args), outputs, updates = updates, mode = compile.mode.Mode(linker='py',optimizer=None))
    # We now look at what outputs our function returns
    inner_fn_outs = [ out.variable for out in dummy_f.maker.outputs]
    update_map       = {}
    shared_outs      = []
    shared_non_seqs  = []
    givens           = {}

    # if the number of outputs to the function does not match the number of
    # assumed outputs until now (provided by the initial case) there can be
    # only one explanation that we now how to deal with. Namely no information
    # is provided for any outputs which will indicate that we deal with a map,
    # i.e. we never use previous values of outputs
    if len(inner_fn_outs) != n_outs:
        if outs_info == []:
            # We know how to deal with this case, assume that none of the outputs
            # are required to have any sort of time taps
            # we just need to update the number of actual outputs
            n_outs = len(inner_fn_outs)
            # other updates :
            for i in xrange(n_outs):
                outs_info += [ dict() ]
            # we also need to re-initialize the store_steps list to match the
            # number of outputs
            store_steps = [ 0 for i in xrange(n_outs)]

        else:
            # Otherwise there is a bit of confusion, since Scan works on the index of
            # a sequence /output. There are maybe corner cases that could be added here
            # or defult behaviour ( like always add the extra outputs at the end !?)
            # But I did not bother implementing this, I leave it to the user to clearly
            # express what he/she wants to do
            raise ValueError('Scan is totally lost. Make sure that you indicate for each'
                    ' output what taps you want to use, or None, if you do not want to'
                    ' use any !')
    inner_fn_inputs=[input.variable for input in \
        dummy_f.maker.expanded_inputs[:dummy_notshared_ins+dummy_notshared_init_outs]]

    # Keep track of the range (place) where you insert shared variables with updates
    # Because we will not be able to compute the gradient with respect to those variables
    # inner_fn_notshared_ins_idx is from where these shared variables with updates start
    inner_fn_notshared_ins_idx = dummy_notshared_ins + dummy_notshared_init_outs

    # Because scan is particularly sensitive at the order in which it gets its
    # arguments, we need to separete the shared variables that act as outputs
    # from those that are not outputs of the network as well
    n_extended_outs = n_outs
    # Skip the slices that we've added to the inner_fn which will be the first elements
    # of f.maker.epanded_inputs and which we know that are not shared
    fromIdx = dummy_notshared_ins + dummy_notshared_init_outs
    copy_map = {}
    for input in dummy_f.maker.expanded_inputs[fromIdx:] :
        # If input is a shared variable that gets updated, then
        # this shared variable will be an output of our inner function
        if isinstance(input.variable, SharedVariable) and input.update:
            # Create a copy of it
            new_var = input.variable.type()
            if input.variable.name:
                new_var.name = input.variable.name + '_copy'
            copy_map[new_var] = input.variable
            inner_fn_inputs.append(new_var)
            # add it to the slices at the end
            slice_to_seqs += [ n_extended_outs ]
            inner_fn_outs += [input.update]
            update_map[ input.variable ] = n_extended_outs
            # We know that we only have access to the last step
            outputs_taps[ n_extended_outs ] = [-1]
            n_extended_outs += 1
            # we shouldn't try to store more then the last step
            # this might not even be a tensor ! ( RandomState )
            store_steps += [1]
            return_steps[n_extended_outs -1] = 1
            shared_outs += [input.variable]
            givens[input.variable] = inner_fn_inputs[-1]
    # inner_fn_shared_ins_idx stores where we stop having shared variables with updates
    inner_fn_shared_ins_idx = len(inner_fn_inputs) - inner_fn_notshared_ins_idx

    # Now that we took out the shared variables that have an update rule
    # we need to take care of all the other shared variables
    for input in dummy_f.maker.expanded_inputs[fromIdx:] :
        # make sure that we do not add the same shared variable twice
        if isinstance(input.variable, SharedVariable) and not input.update:
           shared_non_seqs += [input.variable]
           new_var = input.variable.type()
           if input.variable.name:
               new_var.name = input.variable.name + '_copy'
           inner_fn_inputs += [new_var]
           slice_to_seqs += [ n_extended_outs]
           givens[input.variable] = inner_fn_inputs[-1]
           copy_map[inner_fn_inputs[-1]] = input.variable
        elif not isinstance(input.variable, SharedVariable):
            # also add the normal tensor that are non sequences at the
            # end of the inputs intertwingled with the shared variables
            inner_fn_inputs.append(input.variable)


    # If we haven't provided a number of steps nor did we provide a sequence
    # scan will not know how long to iterate
    if (n_steps == None or n_steps == numpy.inf or n_steps == numpy.nan) and n_seqs == 0 :
        raise ValueError('Scan does not know for how many steps to iterate. '
                'You need to provide the number of steps through the '
                ' ``n_steps`` argument if you do not iterate over any sequence')
    # We can now create the Scan Op Object

    if n_fixed_steps not in [1,-1]:

        if n_steps != None:
            n_steps = tensor.as_tensor(n_steps)
        else:
            n_steps = gof.Constant(gof.generic, 'unknown', '?_steps')

        local_op = Scan( (inner_fn_inputs,inner_fn_outs, givens, slice_to_seqs ), n_seqs,
                n_extended_outs, inplace_map, sequences_taps,  outputs_taps, n_steps,truncate_gradient,
                # n_outs, inner_fn_notshared_ins_idx and inner_fn_shared_ins_idx are used by the gradient
                # to figure out where in the input are shared variables with updates, for whom I can't compute
                # a gradient
                n_outs, inner_fn_notshared_ins_idx, inner_fn_shared_ins_idx,
                go_backwards, store_steps, return_steps, mode, name = name )
        # Shortcut for attaching this property to the Scan op
        local_op.copy_map = copy_map
        # Call the object on the input sequences, initial values for outs,
        # and non sequences
        for seq in seqs :
            if not seq.get('input', None):
                    raiseValue('All input sequences should provide')
        unwrapped_seqs = [ seq.get('input',tensor.as_tensor(0.)) for seq in seqs ]
        unwrapped_outs = [ out.get('initial',tensor.as_tensor(0.)) for out in outs_info ]

        values =  local_op( *(    [n_steps]
                            + unwrapped_seqs
                            + unwrapped_outs
                            + shared_outs
                            + notshared_other_args
                            + shared_non_seqs))

    else:
        # If we do not actually need scan
        for pos, inner_out in enumerate(inner_fn_outs):
            if isinstance(inner_out.type, tensor.TensorType) and store_steps[pos] != 1:
                inner_fn_outs[pos] = tensor.unbroadcast( tensor.shape_padleft(inner_out),0)
        values = inner_fn_outs


    if not type(values) in (tuple, list):
        values = [values]
    # take out the updates of shared variable and build the dictionary
    # that tells what to update and with what value
    for val in update_map.keys():
        update_map[val] = values [ update_map[val] ]

    # Now we need to check the values returned
    # if it just one strip the list around it
    if n_outs == 1:
        # if we need to return just one step or several steps
        # note that when we return one step we have two cases, in
        # the first one store_steps is set to 1, case in which we don't
        # need to take a slice of the output (is already of the right
        # dimension) and case 2 when we store more then one step,
        # and we actually need to take  a slice
        if return_steps.has_key(0):
            if return_steps[0] > 1:
                values = values[0][-return_steps[0]:]
            else:
                if store_steps[0] == 1:
                    values = values[0]
                else:
                    values = values[0][-1]
        else:
            values = values[0]
    else:
        values = values[:n_outs]
        for idx,val in enumerate(values):
            if return_steps.has_key(idx):
                if return_steps[idx] > 1:
                    values[idx] = val[-return_steps[idx]:]
                else:
                    if store_steps[idx] == 1:
                        values[idx] = val
                    else:
                        values[idx] = val[-1]
    return (values, update_map)


class Scan(Op):
    #
    # OLD DOCUMENTATION CAN BE FOUND NEAR REVISION 2581
    #

    def __init__(self,(inputs, outputs, givens, slice_to_seqs),n_seqs,  n_outs,
                 inplace_map={}, seqs_taps={}, outs_taps={},
                 n_steps = gof.Constant(gof.generic, 'unknown', '?_steps'),
                 truncate_gradient = -1, n_outs_not_shared =0,
                 inner_fn_start_shared = 0, inner_fn_end_shared = 0,
                 go_backwards = False, store_steps = {},
                 return_steps={}, mode = None, inplace=False, name = None):
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
        :param name: see scan fct
        :param mode: see scan fct
        '''
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
                # the n_steps is always the first argument of scan's perform,
                # so we need to shift everything by 1
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

        self.seqs_taps             = seqs_taps
        self.outs_taps             = outs_taps
        self.n_seqs                = n_seqs
        self.n_outs                = n_outs
        self.n_args                = n_seqs+n_outs+1
        self.inplace_map           = inplace_map
        self.store_steps           = store_steps
        self.inplace               = inplace
        self.inputs                = inputs
        self.return_steps          = return_steps
        self.givens                = givens
        self.n_outs_not_shared     = n_outs_not_shared
        self.inner_fn_start_shared = inner_fn_start_shared
        self.inner_fn_end_shared   = inner_fn_end_shared
        self.outputs               = outputs
        self.n_steps               = n_steps # It will be computed at runtime
        # This is here just for an optimization to be able to pick up if
        # scan is really needed in the graph; if the number of steps
        # scan does is a constant of 1, -1 or 0 then we can remove scan
        # from the graph
        self.mode           = mode
        self.name           = name
        self.truncate_gradient = truncate_gradient
        self.go_backwards   = go_backwards
        self.slice_to_seqs  = slice_to_seqs

        mode_instance = compile.mode.get_mode(mode)
        #if we use the default mode and it is a ProfileMode
        #we must make a copy otherwise in the profile their will time counted many times
        #1) The scan op and its time will include all time spend into the inner node.
        #2) The inner scan op with their real time.
        #This is done for the Scan and ScanGred op
        if mode is None and isinstance(mode_instance, compile.profilemode.ProfileMode):
            mode_instance = compile.profilemode.ProfileMode(
                optimizer=mode_instance.provided_optimizer,
                linker=mode_instance.provided_linker)
            compile.profilemode.prof_mode_instance_to_print.append(mode_instance)
            self.mode_instance = mode_instance
            if self.name:
                self.mode_instance.message=self.name+" sub profile"
            else:
                self.mode_instance.message="Scan sub profile"

        if name is None: name = 'scan_fn'
        self.fn = function(inputs,outputs, mode = mode_instance, givens = givens,
                           name = name)
        # asert that we don't have shasred variables anymore ( we replaced them
        # with non shared versions)
        assert not numpy.any([isinstance(x.variable,SharedVariable) for x in
            self.fn.maker.inputs])

    def __str__(self):
        if self.name:
            return self.name
        else:
            return 'scan'

    def make_node(self,*inputs):
        assert all(isinstance(i, gof.Variable) for i in inputs)

        self.n_steps = inputs[0]
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
            (self.return_steps == other.return_steps) and \
            (self.n_outs_not_shared == other.n_outs_not_shared) and \
            (self.inner_fn_start_shared == other.inner_fn_start_shared) and\
            (self.inner_fn_end_shared == other.inner_fn_end_shared) and \
            (self.mode == other.mode) and \
            (self.n_seqs == other.n_seqs) and\
            (self.inplace == other.inplace) and\
            (self.go_backwards == other.go_backwards) and\
            (self.truncate_gradient == other.truncate_gradient) and\
            (self.n_outs == other.n_outs) and\
            (self.n_args == other.n_args)
        return rval


    def __hash__(self):
        # the self.apply_output_types are a function of all these things
        # no need to compare it as well
        return hash(type(self)) ^ \
            hash(self.n_seqs) ^ \
            hash(self.n_outs) ^ \
            hash(self.n_outs_not_shared) ^ \
            hash(self.inner_fn_start_shared) ^\
            hash(self.inner_fn_end_shared) ^\
            hash(self.inplace) ^\
            hash(self.go_backwards) ^\
            hash(self.truncate_gradient) ^\
            hash(self.n_args) ^ \
            hash(self.mode) ^\
            hash_listsDictsTuples(self.outputs) ^ \
            hash_listsDictsTuples(self.inputs) ^ \
            hash_listsDictsTuples(self.givens) ^ \
            hash_listsDictsTuples(self.seqs_taps) ^\
            hash_listsDictsTuples(self.outs_taps) ^\
            hash_listsDictsTuples(self.return_steps) ^\
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
                if inplace_map.has_key(i) and seqs_taps.has_key(inplace_map[i]) and\
                        seqs_taps[inplace_map[i]] < 0:
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
                                    # past value not provided.. issue a warning and use
                                    # 0s of the correct dtype
                                    fn_args += [numpy.zeros(args[j+n_seqs][0].shape, \
                                            dtype = args[j+n_sqs][0].dtype)]
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
                                req_idx = (self.idx_store_steps[j] + tap_value + \
                                        self.store_steps[j])
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
                    # if you have provided no size for the missing output you might
                    # find yourself here with a incorect array .. if that happens
                    # realocate memory for the needed array
                    try :
                        if hasattr(something[j],'dtype') and (y[j].dtype != \
                                something[j].dtype) :
                            raise ValueError('wrong dtype')

                        y[j][i] = something[j]
                    except :

                        y[j]= numpy.empty((n_steps,)+something[j].shape, dtype= \
                                something[j].dtype)
                        y[j][i] = something[j]

                elif self.store_steps[j] == 1:
                    try:
                        if hasattr(something[j],'dtype') and y[j].dtype != \
                                something[j].dtype:
                            raise ValueError('wrong dtype')
                        y[j] = something[j]
                    except:
                        y[j] = numpy.empty( something[j].shape, dtype = \
                                something[j].dtype)
                        y[j] = something[j]
                else:
                    try:
                        if hasattr(something[j],'dtype') and y[j].dtype != \
                                something[j].dtype:
                            raise ValueError('worng dtype')
                        y[j][self.idx_store_steps[j]] = something[j]
                        self.idx_store_steps[j] = (self.idx_store_steps[j] + 1) %\
                                self.store_steps[j]
                    except:
                        y[j] = numpy.empty( (self.store_steps[j],)+something[j].shape, \
                                dtype = something[j].dtype)
                        y[j][idx_store_steps[j]] = something[j]
                        self.idx_store_steps[j] = (self.idx_store_steps[j] + 1) %\
                                self.store_steps[j]
        return y

    def grad(self, args, g_outs):
        # forward pass - get the outputs after applying scan
        scan_outputs = self(*args)
        # make sure they are given as a list
        if not( type(scan_outputs) in (list,tuple)):
            scan_outputs = [scan_outputs]
        # get a list of clean inputs ( against which one can compute
        # gradients ) [ everything except shared variables with updates ]
        clean_inputs = self.inputs[:self.inner_fn_start_shared] + \
                       self.inputs[self.inner_fn_start_shared + \
                                   self.inner_fn_end_shared:]


        clean_inputs = [ self.copy_map.get(x,x) for x in clean_inputs]
        s_inputs = [self.copy_map.get(x,x) for x in self.inputs ]
        # function that computes the gradient (we sum over the gradients
        # with respect to all outputs
        def compute_gradient(y, g_y):
            gmp = gradient.grad_sources_inputs( \
                        [(y,g_y)], clean_inputs, False)
            def zero(p):
                try:
                    use_dtype = p.type.dtype
                except:
                    use_dtype = theano.config.floatX
                return tensor.TensorConstant(tensor.TensorType(\
                      dtype=use_dtype, broadcastable=[]),
                      safe_asarray._asarray(0,dtype = use_dtype))
            return [gmp.get(p, zero(p)) for p in s_inputs]


        # this are g_outs for the inner function (that computes the gradients)
        inner_g_outs = []
        # the outs of the gradient computting inner function
        inner_gfn_outs = []
        inner_gfn_ins = []
        # Go through the outputs that don't represent update rules
        for out in self.outputs[:self.n_outs_not_shared]:
            inner_g_out = out.type()
            if out.name:
                # for debugging add names to all variables I'm creating
                g_y.name = 'g_'+out.name

            inner_g_outs.append(inner_g_out)
            _grad_outs = compute_gradient(out, inner_g_out)
            grad_outs = _grad_outs[:self.n_seqs+self.n_outs_not_shared] + \
                        _grad_outs[self.n_seqs+self.n_outs:]
            if not inner_gfn_outs :
                inner_gfn_outs = grad_outs
            else:
                # safety check, some of this inputs might still not be differentiable,
                # for those we don't add them to the mix (assume their gradient is 0)
                for i,(x,y) in enumerate(zip(grad_outs, inner_gfn_outs)):
                    if x and y:
                        inner_gfn_outs[i] = x+y
                    elif y:
                        inner_gfn_outs[i] = y
                    else:
                        inner_gfn_outs[i] = x


        # backwards pass
        for i in xrange(len(inner_gfn_outs)):
            if inner_gfn_outs[i] == None:
                inner_gfn_outs[i] = tensor.zeros_like(clean_inputs[i])
        for i in xrange(self.n_outs_not_shared):
            # Safety check
            if g_outs[i] == None:
                try:
                    # this try is for catching non ndarray inputs (random states)
                    # it is more of a safety check ( all random states should be
                    # after n_outs_not_shared ...
                    g_outs[i] = tensor.zeros_like(scan_outputs[i])
                except:
                    g_outs[i] = theano.tensor.constant(numpy.array(0,dtype=\
                            theano.config.floatX))
        inner_gfn_ins = inner_g_outs + self.inputs


        # Make sure you don't have numbers in here
        if not isinstance(self.n_steps, Variable):
            n_steps = tensor.as_tensor(self.n_steps)
        else:
            n_steps = self.n_steps
        g_args = [n_steps] + g_outs[:self.n_outs_not_shared] \
                            + scan_outputs + args[1:]

        truncate_gradient = self.truncate_gradient
        for x in self.store_steps[:self.n_outs_not_shared]:
            if x>0 :
                raise ValueError('Can not compute gradients if one does not ',
                        'store all intermediate results (remove store_steps'
                        'from the dictionaries describing your outputs)')
        g_scan = ScanGrad((inner_gfn_ins, inner_gfn_outs),
                self.n_seqs, self.n_outs, self.n_outs_not_shared,
                self.go_backwards, self.seqs_taps, self.outs_taps,
                truncate_gradient)

        g_scan_outs = g_scan(g_args)
        if not type(g_scan_outs) in (list, tuple):
            g_scan_outs = [ g_scan_outs ]
        # We need to add several None's for shared vars with updates
        gradients = [None] + g_scan_outs[:self.n_seqs+self.n_outs_not_shared]
        gradients += [None for i in xrange(self.n_outs-self.n_outs_not_shared)]
        gradients += g_scan_outs[self.n_seqs+self.n_outs_not_shared:]
        return gradients


class ScanGrad(Op):
    """Gradient Op for Scan"""
    def __init__(self,(g_ins, g_outs) , n_seqs, n_outs,
            n_outs_not_shared,
            go_backwards = False, seqs_taps = {}, outs_taps= {},
            truncate_gradient = -1, mode = None, name = None):
        """
        :param mode: see scan fct
        :param name: see scan fct
        """


        self.inputs = g_ins
        self.outputs = g_outs
        self.n_outs_not_shared = n_outs_not_shared
        self.n_seqs = n_seqs
        self.go_backwards = go_backwards
        self.truncate_gradient = truncate_gradient
        self.n_outs = n_outs
        self.seqs_taps = seqs_taps
        self.outs_taps = outs_taps
        self.destroy_map = {}
        self.mode = mode

        mode_instance = compile.mode.get_mode(mode)
        #if we use the default mode and it is a ProfileMode
        #we must make a copy otherwise in the profile their will time counted many times
        #1) The scan op and its time will include all time spend into the inner node.
        #2) The inner scan op with their real time.
        #This is done for the Scan and ScanGred op
        if mode is None and isinstance(mode_instance, compile.profilemode.ProfileMode):
            mode_instance = compile.profilemode.ProfileMode(
                optimizer=mode_instance.provided_optimizer,
                linker=mode_instance.provided_linker)
            compile.profilemode.prof_mode_instance_to_print.append(mode_instance)
            self.mode_instance = mode_instance
            self.mode_instance.message="ScanGrad sub profile"
        if name is None: name = 'scan_grad_fn'

        self.grad_fn = function(g_ins, g_outs, mode = mode_instance, name = name)

    def __eq__(self,other):
        rval = type(self) == type(other)
        if rval:
           rval = (self.inputs == other.inputs) and \
                  (self.outputs == other.outputs) and \
                  (self.n_seqs == other.n_seqs) and \
                  (self.n_outs == other.n_outs) and \
                  (self.go_backwards == other.go_backwards) and \
                  (self.n_outs_not_shared == other.n_outs_not_shared) and\
                  (self.truncate_gradient == other.truncate_gradient) and\
                  (self.mode == other.mode) and \
                  (self.seqs_taps == other.seqs_taps) and \
                  (self.outs_taps == other.outs_taps)
        return rval

    def __hash__(self):
        return hash(type(self)) ^ \
               hash(self.n_seqs) ^ \
               hash(self.n_outs) ^ \
               hash(self.go_backwards) ^\
               hash(self.truncate_gradient) ^\
               hash(self.mode) ^\
               hash_listsDictsTuples(self.inputs) ^ \
               hash_listsDictsTuples(self.outputs) ^ \
               hash_listsDictsTuples(self.seqs_taps) ^ \
               hash_listsDictsTuples(self.outs_taps)

    def make_node(self, *args):
        # input of the gradient op :
        # | g_outs | y      | seqs   | outs    | non_seqs   |
        # | n_outs | n_outs | n_seqs | n_outs  | unknown    |
        # return
        # | grad of seqs | grad of outs | grad of non_seqs  |
        # |   n_seqs     |  n_outs      |  unknown          |

        scan_inputs = args[0][1+self.n_outs_not_shared+self.n_outs:]

        outputs_grad = scan_inputs[:self.n_seqs+self.n_outs_not_shared]
        outputs_grad += scan_inputs[self.n_seqs+self.n_outs:]
        return Apply(self, list(args[0]),
                    [i.type() for i in outputs_grad ])

    def perform(self, node, args, storage):
        # get scan inputs
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

        inputs = args[self.n_outs_not_shared+self.n_outs+1:]
        seqs = inputs[:self.n_seqs]
        outInfo = inputs[self.n_seqs:self.n_seqs+self.n_outs]
        non_seqs = inputs[self.n_outs+self.n_seqs:]
        if (self.n_seqs == 0 ) and (not numpy.isfinite(n_steps) ):
            raise ValueError('Scan does not know how many steps it '
                'should iterate! Either provide some input sequences from '
                'which scan could find out the number of steps, or directly'
                'the number of steps you want through the n_steps argument.')

        for i in xrange(self.n_seqs):
            if self.seqs_taps.has_key(i):
                # compute actual length of the sequence ( we need to see what
                # past taps this sequence has, and leave room for them
                seq_len = seqs[i].shape[0] + min(self.seqs_taps[i])
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


        # go back through time to 0 or n_steps - truncate_gradient
        lower_limit = n_steps - self.truncate_gradient
        length = n_steps
        if lower_limit > n_steps-1:
            the_range = xrange(n_steps-1,-1,-1)
            lower_limit = 0
        elif lower_limit < -1:
            the_range = xrange(n_steps-1,-1,-1)
            lower_limit = 0
        else:
            the_range = xrange(n_steps-1, lower_limit-1,-1)
            lower_limit = lower_limit + 1
        # generate space for gradient
        if lower_limit != 0 :
            length = len(the_range)
            g_seqs = []
            # Check for taps ==> you need to enlarge the sequence length
            for j in xrange(self.n_seqs):
                if self.seqs_taps.has_key(j):
                    length = length - min(self.seqs_taps[j])
                    length = length + max(self.seqs_taps[j])
                g_seqs += [ numpy.zeros_like(seqs[j][:length]) ]
        else:
            g_seqs = [numpy.zeros_like(k) for k in seqs]
        g_outInfo  = [numpy.zeros_like(k) \
                                for k in outInfo[:self.n_outs_not_shared]]

        g_non_seqs = [numpy.zeros_like(k) for k in non_seqs]
        # get gradient on the outputs
        g_outs = [arg.copy() for arg in args[1:self.n_outs_not_shared+1]]

        # get the output of the scan operation
        outs = args[1+self.n_outs_not_shared:self.n_outs_not_shared+self.n_outs+1]



        seqs_mins = {}
        for j in xrange(self.n_seqs):
            if self.seqs_taps.has_key(j):
                seqs_mins.update({j: min(self.seqs_taps[j])})

        outs_mins = {}
        initOuts_size = {}
        for j in xrange(self.n_outs):
            if j >= self.n_outs_not_shared:
                outs_mins.update({j:-1})
                initOuts_size.update({j:0})

            elif self.outs_taps.has_key(j):
                outs_mins.update({j: min(self.outs_taps[j])})
                if self.outs_taps[j] != [-1]:
                    initOuts_size.update({j:g_outInfo[j].shape[0]})
                else:
                    initOuts_size.update({j:0})

        for i in the_range:
          # time slice of inputs
          _ins = []
          _i = i
          if go_backwards:
              _i = n_steps -1 -i

          for j in xrange(self.n_seqs):
            if self.seqs_taps.has_key(j):
              ls_taps = self.seqs_taps[j]
              min_tap =      seqs_mins[j]
              for tap_value in ls_taps:
                k = _i - min_tap + tap_value
                _ins += [seqs[j][k]]
          # time slice of outputs + taps
          _outs = []
          for j in xrange(self.n_outs):
            if self.outs_taps.has_key(j):
              ls_taps = self.outs_taps[j]
              min_tap =      outs_mins[j]
              seed_sz =      initOuts_size[j]
              for tap_value in ls_taps:
                if i + tap_value < 0:
                  if seed_sz < 1:
                      _outs += [outInfo[j]]
                  else:
                      k = i + seed_sz  + tap_value
                      if k < 0 :
                        #past value not provided .. issue a warning and use 0
                        _outs += [numpy.zeros(outInfo[j][0].shape)]
                        warning('Past value %d for output $d not given' \
                              %(j,tap_value))
                      else:
                        _outs += [outInfo[j][k]]
                else:
                    if j>= self.n_outs_not_shared:
                        _outs += [outs[j] ]
                    else:
                        _outs += [outs[j][i + tap_value]]
          g_out = []
          g_out = [ arg[i] for arg in g_outs]
          grad_args = g_out + _ins + _outs + non_seqs
          grads=self.grad_fn(*grad_args)
          # get gradient for inputs
          pos = 0
          for j in xrange(self.n_seqs):
              if self.seqs_taps.has_key(j):
                  ls_taps = self.seqs_taps[j]
                  min_tap =      seqs_mins[j]
                  for tap_value in ls_taps :
                      k = _i - min_tap + tap_value
                      g_seqs[j][k-lower_limit] += grads[pos]
                      pos += 1


          # get gradient for outputs
          for j in xrange(self.n_outs_not_shared):
            if self.outs_taps.has_key(j):
              ls_taps = self.outs_taps[j]
              min_tap =      outs_mins[j]
              seed_sz =      initOuts_size[j]
              for tap_value in ls_taps:
                if i+tap_value < 0 :
                    k = i + seed_sz + tap_value
                    if  k >= 0 :
                        g_outInfo[j][k] += grads[pos]
                    else:
                        g_outInfo[j] += grads[pos]

                else:
                    g_outs[j][i+tap_value] += grads[pos]
                pos += 1
          for j in xrange(len(g_non_seqs)):
            g_non_seqs[j] += grads[j+pos]


        # return the gradient
        for i,v in enumerate(g_seqs + g_outInfo+ g_non_seqs):
            storage[i][0] = v



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
                # check the outputs
                for i,out in enumerate(node.outputs):
                    if op.store_steps[i] == 0 :
                        # if we do not have a range for this output
                        req_steps = numpy.max(numpy.abs(op.outs_taps.get(i,1)))
                        # look at all its clients
                        for cl,_dx in out.clients:
                            if type(cl) == str:
                                # if the node is actually an output, then
                                # we need to store the entire thing
                                req_steps = None
                                break
                            else:
                                if not isinstance(cl.op,
                                        tensor.basic.Subtensor):
                                    # if any of the clients is not a subtensor
                                    # we also need to store the enitre thing
                                    req_steps = None
                                    break
                                else:
                                    # if it is a tensor, and the first
                                    # dimension is just -1
                                    if cl.op.idx_list[0] == -1 and req_steps != None:
                                                req_steps = numpy.max([1, req_steps])
                                    else:
                                        # or a constant that evaluates to
                                        # -1
                                        try:
                                            idx = opt.get_constant_value(\
                                                    cl.op.idx_list[0])
                                            if idx== -1:
                                                req_steps = numpy.max([1, req_steps])
                                            else:
                                                req_steps = None
                                                break
                                        except:
                                            req_steps = None
                                            break
                        if req_steps != None:
                            store_steps[i] = req_steps
                        else:
                            store_steps[i] = 0
                    else:
                        store_steps[i] = op.store_steps[i]
                if numpy.any(store_steps!= op.store_steps):
                    new_scan = Scan((op.inputs, op.outputs, op.givens,
                        op.slice_to_seqs),op.n_seqs, op.n_outs,
                        op.inplace_map, op.seqs_taps, op.outs_taps, op.n_steps,
                        op.truncate_gradient, op.n_outs_not_shared,
                        op.inner_fn_start_shared, op.inner_fn_end_shared,
                        op.go_backwards, store_steps, op.return_steps, op.mode,
                        op.inplace, name = op.fn.name).make_node(*node.inputs)
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
            op.n_outs, op.inplace_map, op.seqs_taps, op.outs_taps, op.n_steps,
            op.truncate_gradient, op.n_outs_not_shared, op.inner_fn_start_shared,
            op.inner_fn_end_shared, op.go_backwards, op.store_steps, op.return_steps,
            op.mode, inplace=True, name = op.fn.name).make_node(*node.inputs).outputs
    return False


optdb.register('scanOp_make_inplace', opt.in2out(scan_make_inplace,
    ignore_newtrees=True), 75, 'fast_run', 'inplace')



