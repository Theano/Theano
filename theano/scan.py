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

import theano
from theano.tensor import opt
from theano import gof
from theano.compile import optdb
import theano.tensor.shared_randomstreams as shared_random

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

def scan(fn, sequences, initial_states, non_sequences, inplace_map={}, \
         sequences_taps={}, outputs_taps = {}, n_steps = 0, \
         truncate_gradient = -1, go_backwards = False, 
         mode = None):
    '''Function that constructs and applies a Scan op

    :param fn: Function that describes the operations involved in one step 
               of scan Given variables representing all the slices of input 
               and past values of outputs and other non sequences parameters, 
               ``fn`` should produce variables describing the output of one 
               time step of scan. The order in which the argument to this 
               function are given is very important. You should have the 
               following order:

               * all time slices of the first sequence (as given in the 
                 ``sequences`` list) ordered cronologically
               * all time slices of the second sequence (as given in the 
                 ``sequences`` list) ordered cronologically
               * ...
               * all time slices of the first output (as given in the  
                 ``initial_state`` list) ordered cronologically 
               * all time slices of the second otuput (as given in the 
                 ``initial_state`` list) ordered cronologically
               * ...
               * all other parameters over which scan doesn't iterate given 

               in the same order as in ``non_sequences``
               If you are using shared variables over which you do not want to
               iterate, you do not need to provide them as arguments to 
               ``fn``, though you can if you wish so. The function should 
               return the outputs after each step plus the updates for any of
               the shared variables. You can either return only outputs or 
               only updates. If you have both outputs and updates the 
               function should return them as a tuple : (outputs, updates) 
               or (updates, outputs). 

               Outputs can be just a theano expression if you have only one 
               outputs or a list of theano expressions. Updates can be given 
               either as a list of as a dictionary. If you have a list of 
               outputs, the order of these should match that of their 
               ``initial_states``. 
    :param sequences: list of Theano variables over which scan needs to 
                      iterate.
    :param initial_states: list of Theano variables containing the initial 
                           state used for the output. Note that if the 
                           function applied recursively uses only the 
                           previous value of the output or none, this 
                           initial state should have same shape 
                           as one time step of the output; otherwise, the 
                           initial state should have the same number of 
                           dimension as output. This can easily be understand
                           through an example. For computing ``y[t]`` let 
                           assume that we need ``y[t-1]``, ``y[t-2]`` and 
                           ``y(t-4)``. Through an abuse of notation, 
                           when ``t = 0``, we would need values for 
                           ``y[-1]``, ``y[-2]`` and ``y[-4]``. These values 
                           are provided by the initial state of ``y``, which 
                           should have same number  of dimension as ``y``, 
                           where the first dimension should be large enough 
                           to cover all past values, which in this case is 4.
                           If ``init_y`` is the variable containing the 
                           initial state of ``y``, then ``init_y[0]`` 
                           corresponds to ``y[-4]``, ``init_y[1]`` 
                           corresponds to ``y[-3]``, ``init_y[2]`` 
                           corresponds to ``y[-2]``, ``init_y[3]`` 
                           corresponds to ``y[-1]``. By default, scan is set 
                           to use the last time step for each output. 
    :param non_sequences: Parameters over which scan should not iterate.
                          These parameters are given at each time step to 
                          the function applied recursively.
    :param inplace_map: Dictionary describing outputs computed *inplace*.
                        ``inplace_map`` is a dictionary where keys are 
                        output indexes, and values are sequence indexes. 
                        Assigning a value ``j`` to a key ``i`` means that 
                        output number ``j`` will be computed inplace (in the
                        same memory buffer) as the input number ``i``.
    :param sequences_taps: Dictionary describing what slices of the input 
                           sequences scan should use. At each step of the 
                           iteration you can use different slices of your 
                           input sequences(called here taps), and this 
                           dictionary lets you define exactly that. The 
                           keys of the dictionary are sequence indexes, 
                           the values are list of numbers. Having the 
                           following entry ``i : [k_1,k_2,k_3]``, means that 
                           at step ``t``, for sequence ``x``, that has the 
                           index ``i`` in the list of sequences, you would 
                           use the values  ``x[t+k_1]``, ``x[t+k_2]`` and  
                           ``x[t+k_3]``. ``k_1``, ``k_2``, ``k_3`` values 
                           can be positive or negative and the sequence for 
                           you request this taps should be large enough to 
                           accomodate them. If in the cronological order, 
                           ``k`` is the first past value of sequence ``x``, 
                           then index 0 of ``x`` will correspond to step ``k``
                           (if ``k`` is -3, then, abusing notation ``x[0]`` 
                           will be seen by scan as ``x[-3]``). If you do not 
                           want to use any taps for a given sequence you need
                           to set the corresponding entry in the dictionary 
                           to the empy list. By default, for each sequence 
                           that is not represented in the dictionary scan 
                           will assume that the at every step it needs to 
                           provide the current value of that sequence.
    :param outputs_taps: Dictionary describing what slices of the input 
                         sequences scan should use. The ``outputs_taps`` are 
                         defined in an analogouws way to ``sequences_taps``,
                         just that the taps are for the outputs generated by 
                         scan. As such they can only be negative, i.e. refer 
                         to past value of outputs. By default scan will 
                         expect to use for any outpu the last time step, if 
                         nothing else is specified.
    :param n_steps: Number of steps to iterate. Sometimes you want to either 
                    enforce a fixed number of steps, or you might not even 
                    have any sequences you want to iterate over, but rather
                    just to repeat some computation for a fixed number of 
                    steps. ``n_steps`` gives you this possibility. It can be 
                    a theano scalar or a number. 
    :param truncate_gradient: Number of steps to use in truncated BPTT.
                              If you compute gradients through a scan op,
                              they are computed using backpropagation through
                              time. By providing a different value then -1, 
                              you choose to use truncated BPTT instead of 
                              classical BPTT, where you only do 
                              ``truncate_gradient`` number of steps.
    :param go_backwards: Flag indicating if you should go bacwards through 
                         the sequences
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
        
    if not (type(initial_states) in (list,tuple)):
        init_outs = [initial_states]
    else: 
        init_outs = initial_states
        
    if not (type(non_sequences) in (list,tuple)):
        non_seqs = [non_sequences]
    else:
        non_seqs = non_sequences



    # compute number of sequences and number of seqs   
    n_seqs     = len(seqs)
    n_outs   = len(init_outs)

    # update sequences_taps[idx] to contain 0 if it is not defined
    for i in xrange(n_seqs):
        if not sequences_taps.has_key(i):
            sequences_taps.update({i:[0]})
        # if input sequence is not actually used by the recursive function
        elif sequences_taps[i] == []:
            sequences_taps.__delitem__(i)
        elif not (type(sequences_taps[i]) in (list,tuple)):
            sequences_taps[i] = [sequences_taps[i]]
    # update outputs_taps[idx] to contain -1 if it is not defined
    for i in xrange(n_outs):
        if not outputs_taps.has_key(i):
            outputs_taps.update({i:[-1]})
        elif outputs_taps[i] == []:
            outputs_taps.__delitem__(i)
        elif not(type(outputs_taps[i]) in (list,tuple)):
            outputs_taps[i] = [outputs_taps[i]]
    stored_steps_output = [ 0 for i in xrange(n_outs)]
                      



    # create theano inputs for the recursive function  
    args = []
    _ins = 0 
    _outs = 0
    for (i,seq) in enumerate(seqs):
      if sequences_taps.has_key(i):
        for k in xrange(len(sequences_taps[i])):
            args += [seq[0].type() ]
            _ins += 1
    for (i,init_out) in enumerate(init_outs):
      if outputs_taps.has_key(i):
        for k in xrange(len(outputs_taps[i])):
            if outputs_taps[i] == [-1]:
                args += [init_out.type() ]
                _outs += 1
            else:
                args += [init_out[0].type() ]
                _outs += 1
    noshared = []
    for non_seq in non_seqs:
        if not isinstance(non_seq, theano.compile.SharedVariable):
            noshared += [non_seq]


    dummy_args = args + noshared
    args += non_seqs

    outputs_updates  = fn(*args)
    outputs = []
    updates = {}
    # we try now to separate the outputs from the updates
    if not type(outputs_updates) in (list,tuple):
        if type(outputs_updates) == dict :
            # we have just an update dictionary
            updates = outputs_updates
        else:
            outputs = [outputs_updates]
    else:
        elem0 = outputs_updates[0]
        elem1 = outputs_updates[1]
        if ( type(elem0) == dict ) or \
           ( type(elem0) in (list,tuple) and type(elem0[0]) in (list,tuple)):
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
               ( type(outputs_updates[0]) in (list,tuple)):
                 outputs = []
                 updates = outputs_updates
            else:
                outputs = outputs_updates
                updates = {}


    # Wo compile a dummy function just to see what shared variable
    # we have and what are their update rules

    dummy_f = theano.function(dummy_args, outputs, updates = updates, mode = \
                 theano.compile.mode.Mode(linker = 'py', optimizer = None) )
    


    ls_outputs      = [ sout.variable for sout in dummy_f.maker.outputs]
    update_map      = {}
    n_actual_outs   = n_outs
    shared_outs     = []
    shared_non_seqs = []
    givens          = {}

    ls_inputs=[inp.variable for inp in \
                    dummy_f.maker.expanded_inputs[:_ins+_outs]]
    fromIdx = _ins + _outs
    # add shared variable that act as outputs
    for inp in dummy_f.maker.expanded_inputs[fromIdx:] :
        if isinstance(inp.variable, theano.compile.SharedVariable) and inp.update:
            ls_inputs.append(inp.variable.type())
            ls_outputs += [inp.update]
            update_map[ inp.variable ] = n_outs 
            outputs_taps[ n_outs ] = [-1]
            n_outs += 1
            stored_steps_output += [1] 
            shared_outs += [inp.variable]
            givens[inp.variable] = ls_inputs[-1]

    # add the rest:
    for inp in dummy_f.maker.expanded_inputs[fromIdx:] :
        if isinstance(inp.variable, theano.compile.SharedVariable) and not inp.update:
           shared_non_seqs += [inp.variable]
           ls_inputs += [inp.variable.type() ]
           givens[inp.variable] = ls_inputs[-1]
        elif not isinstance(inp.variable, theano.compile.SharedVariable):
            ls_inputs.append(inp.variable)
    
    # Create the Scan op object
    local_op = Scan( (ls_inputs,ls_outputs, givens ), n_seqs, n_outs, inplace_map,
            sequences_taps, outputs_taps, truncate_gradient,
            go_backwards, stored_steps_output, mode)

    # Call the object on the input sequences, initial values for outs, 
    # and non sequences
    values =  local_op( *(    [theano.tensor.as_tensor(n_steps)]  \
                         + seqs \
                         + init_outs \
                         + shared_outs \
                         + noshared
                         + shared_non_seqs))

    if not type(values) in (tuple, list):
        values = [values]
    for k in update_map.keys():
        update_map[k] = values [ update_map[k] ] 

    if n_actual_outs != n_outs : 
        if n_actual_outs == 1:
            values = values[0]
        else:
            values = values[:n_actual_outs]


    return (values, update_map)




class Scan(theano.Op):
    def __init__(self,(inputs, outputs, givens),n_seqs,  n_outs,
                 inplace_map={}, seqs_taps={}, outs_taps={},
                 truncate_gradient = -1,
                 go_backwards = False, stored_steps_output = {},
                 mode = 'FAST_RUN', inplace=False):
        '''
        :param (inputs,outputs, givens): inputs and outputs Theano variables 
                                         that describe the function that is 
                                         applied recursively; givens
                                         list is used to replace shared
                                         variables with not shared ones
        :param n_seqs: number of sequences over which scan will have to 
                       iterate
        :param n_outs: number of outputs of the scan op
        :param inplace_map: see scan function above
        :param seqs_taps: see scan function above
        :param outs_taps: see scan function above
        :param truncate_gradient: number of steps after which scan should 
                                  truncate -1 implies no truncation 
        :param go_bacwards: see scan funcion above
        :param stored_steps_output: a list of booleans of same size as the 
                                    number of outputs; the value at position 
                                    ``i`` in the list corresponds to the 
                                    ``i-th`` output, and it tells how many 
                                    steps (from the end towards the begining)
                                    of the outputs you really need and should
                                    return; given this information, scan can 
                                    know (if possible) to allocate only
                                    the amount of memory needed to compute 
                                    that many entries
        '''
        

        # check inplace map
        for _out,_in in inplace_map.iteritems():
            if _out > n_outs:
                raise ValueError(('Inplace map reffers to an unexisting'\
                          'output %d')% _out)
            if _in > n_seqs:
                raise ValueError(('Inplace map reffers to an unexisting'\
                          'input sequence %d')%_in)
            if (_in >= 0) and (min(seqs_taps[_in]) < 0):
                raise ValueError(('Input sequence %d uses past values that '\
                         'will be overwritten by inplace operation')%_in)


        #check sequences past taps
        for k,v in seqs_taps.iteritems():
          if k > n_seqs:
            raise ValueError(('Sequences past taps dictionary reffers to '
                    'an unexisting sequence %d')%k)

        #check outputs past taps
        for k,v in outs_taps.iteritems():
          if k > n_outs:
            raise ValueError(('Sequences past taps dictionary reffers to '
                    'an unexisting sequence %d')%k)
          if max(v) > -1:
            raise ValueError(('Can not require future value %d of output' \
                    ' %d')%(k,max(v)))



        self.destroy_map = {}
        if inplace:
            for i in inplace_map.keys():
                self.destroy_map.update({i: [inplace_map[i]+1] } )

        self.seqs_taps      = seqs_taps
        self.outs_taps      = outs_taps
        self.n_seqs         = n_seqs
        self.n_outs         = n_outs
        self.n_args         = n_seqs+n_outs+1
        self.inplace_map    = inplace_map
        self.stored_steps_output   = stored_steps_output
        self.inplace        = inplace
        self.inputs         = inputs
        self.givens         = givens
        self.outputs        = outputs
        self.truncate_gradient = truncate_gradient
        self.go_backwards   = go_backwards

        self.fn = theano.function(inputs,outputs, mode = mode, givens = givens)




    def make_node(self,*inputs):
      n_args = len(inputs)
      if n_args < self.n_args :
         err = 'There should be at least '+str(self.n_args)+ 'arguments'
         raise ValueError(err)

      # return a new variable of same type and same shape 
      def new_same_dim(var): 
        try:
            nw_var = theano.tensor.as_tensor_variable(var)
            return nw_var.type()
        except TypeError:
          if isinstance(var, shared_random.RandomStateSharedVariable):
            return var.type()
          else:
            raise TypeError("Could not convert %s to suitable type"%var, 
                                                                 type(var))

      # return a new variable of same type but with an extra dimension
      def new_add_one_dim(var):
        nw_var = theano.tensor.as_tensor_variable(var)
        return theano.tensor.Tensor( dtype = nw_var.dtype, \
                       broadcastable = (False,)+nw_var.broadcastable)()

      def new_replace_one_dim(var):
        nw_var = theano.tensor.as_tensor_variable(var)
        return theano.tensor.Tensor( dtype = nw_var.dtype, \
                       broadcastable = (False,)+nw_var.broadcastable[1:])()

      def new_remove_one_dim(var):
        nw_var = theano.tensor.as_tensor_variable(var)
        return theano.tensor.Tensor( dtype = nw_var.dtype, \
                       broadcastable = nw_var.broadcastable[1:])()


      # Create list of output datatypes
      out_types = []
      for i in xrange(self.n_seqs+1, self.n_seqs+self.n_outs+1):
         out_idx = i - 1 - self.n_seqs
         if not (inputs[i] == []):
            ## CASES :
            #    outs_taps[i] == [-1] or == [] => inputs[i] no extra dim
            #    outs_taps anything else  => inputs[i] remove one dim

            #
            #     stored_steps_outputs = 1 ==> outs no extra dim
            #     anything else --> needs extra dim
            sw_inputs  = self.outs_taps.get(out_idx, [-1]) == [-1]
            sw_outputs = self.stored_steps_output[out_idx] == 1

            if sw_inputs:
                if sw_outputs:
                    # You need to output something identical to the 
                    # input.. which can even be a non tensor
                    out_types += [ new_same_dim(inputs[i]) ] 
                else:
                    # You need to output a list of things identical to 
                    # the input .. (here we force it to be a tensor )
                    out_types += [ new_add_one_dim(inputs[i]) ]
            else:
                if sw_outputs:
                    # your input has one dimension more, so you need 
                    # to strip it by its first dimension
                    out_types += [new_remove_one_dim(inputs[i])]
                else:
                    # input and output have the same # of dimensions, 
                    # just that you need to "refresh" the first one 
                    # this is important only in the corner case that 
                    # the first dimension of the input is 1, in which 
                    # case the output broadcastable pattern does not 
                    # match the input broadcastable pattern 
                    #
                    # Note that this should in practice never happen !!
                    # I add it here just for safety 
                    out_types += [new_replace_one_dim(inputs[i])]

    
         else:
            raise ValueError(('You need to provide initial state for outputs'
                      ' such that scan can infer what dataype they are'))
      return theano.Apply(self,inputs, out_types)


    def __eq__(self,other):
      rval = type(self) == type(other)
      if rval:
        rval = (self.inputs == other.inputs) and \
               (self.outputs == other.outputs) and \
               (self.givens  == other.givens) and \
               (self.stored_steps_output == other.stored_steps_output) and \
               (self.seqs_taps == other.seqs_taps) and \
               (self.outs_taps == other.outs_taps) and \
               (self.inplace_map == other.inplace_map) and \
               (self.n_seqs == other.n_seqs) and\
               (self.inplace == other.inplace) and\
               (self.go_backwards == other.go_backwards) and\
               (self.truncate_gradient == other.truncate_gradient) and\
               (self.n_outs == other.n_outs) and\
               (self.n_args == other.n_args)
      return rval
      

    def __hash__(self):
      return hash(type(self)) ^ \
             hash(self.n_seqs) ^ \
             hash(self.n_outs) ^ \
             hash(self.inplace) ^\
             hash(self.go_backwards) ^\
             hash(self.truncate_gradient) ^\
             hash(self.n_args) ^ \
             hash_listsDictsTuples(self.outputs) ^ \
             hash_listsDictsTuples(self.inputs) ^ \
             hash_listsDictsTuples(self.givens) ^ \
             hash_listsDictsTuples(self.seqs_taps) ^\
             hash_listsDictsTuples(self.outs_taps) ^\
             hash_listsDictsTuples(self.stored_steps_output)




    def perform(self,node,args, outs):
        n_steps = 0 
        if (self.n_seqs ==0 ) and (args[0] == 0):
            raise ValueError('Scan does not know over how many steps it '
                'should iterate! No input sequence or number of steps to '
                'iterate given !')

        if (args[0] != 0):
            n_steps = args[0]
        
        for i in xrange(self.n_seqs):
          if self.seqs_taps.has_key(i):
              # compute actual length of the sequence ( we need to see what
              # past taps this sequence has, and leave room for them 
              seq_len = args[i+1].shape[0] + min(self.seqs_taps[i])
              if  max( self.seqs_taps[i]) > 0: 
                  # using future values, so need to end the sequence earlier
                  seq_len -= max(self.seqs_taps[i])
              if n_steps == 0 :
                  # length of the sequences, leaving room for the largest
                  n_steps = seq_len
              if seq_len != n_steps : 
                  warning(('Input sequence %d has a shorter length then the '
                          'expected number of steps %d')%(i,n_steps))
                  n_steps = min(seq_len,n_steps)



        # check if we deal with an inplace operation 
        inplace_map  = self.inplace_map
        if not self.inplace: #if it was not optimized to work inplace
            inplace_map = {}

 
        # check lengths of init_outs
        for i in xrange(self.n_seqs+1, \
                        self.n_seqs+self.n_outs+1):
          if self.outs_taps.has_key(i-self.n_seqs-1):
            if self.outs_taps[i-self.n_seqs-1] != [-1]:
              req_size = abs(min(self.outs_taps[i-self.n_seqs-1]))-1
              if args[i].shape[0] < req_size:
                warning(('Initial state for output %d has fewer values then '
                    'required by the maximal past value %d. Scan will use 0s'
                    ' for missing values')%(i-self.n_iterable-1,req_size))
            
        self.n_steps = n_steps
        y = self.scan(self.fn, args[1:],self.n_seqs, self.n_outs, 
                 self.seqs_taps, self.outs_taps, n_steps, self.go_backwards, 
                 inplace_map)


        # write to storage
        for i in xrange(self.n_outs):
            outs[i][0]=y[i]



    def scan(self,fn, args, n_seqs, n_outs, seqs_taps, outs_taps,  n_steps, 
             go_backwards, inplace_map):

      y = []
      for i in xrange(n_outs):
        if inplace_map.has_key(i) and (inplace_map[i] >= 0):
          y += [args[inplace_map[i]]]
        else:
          if self.stored_steps_output[i] == 1 :
            y+= [ None ]
          else:
            arg_shape = args[i+n_seqs].shape[1:]
            if (not self.outs_taps.has_key(i)) or \
                    self.outs_taps[i] == [-1]:
                arg_shape = args[i+n_seqs].shape
            if self.stored_steps_output[i] < 1 :
                y_shape = (n_steps,)+arg_shape
            else:
                y_shape = (self.stored_steps_output[i],)+arg_shape
            y += [numpy.empty(y_shape, dtype=args[i+n_seqs].dtype)]
      seqs_mins = {}
      for j in xrange(n_seqs):
        if seqs_taps.has_key(j):
          seqs_mins.update({j:  min(seqs_taps[j])})

      outs_mins = {}
      initOuts_size = {}
      for j in xrange(n_outs):
        if outs_taps.has_key(j):
          outs_mins.update({j: min(outs_taps[j])})
          if self.outs_taps[j] != [-1]:
              initOuts_size.update({j: args[n_seqs+j].shape[0]})
          else:
              initOuts_size.update({j: 0})


      for i in xrange(n_steps):
        fn_args = []

        # sequences over which scan iterates
        # check to see if we are scaning them backwards or no
        _i = i
        if go_backwards:
            _i = n_steps-1-i
        for j in xrange(n_seqs):
          if seqs_taps.has_key(j):
            ls_taps = seqs_taps[j]
            min_tap = seqs_mins[j]
            for tap_value in ls_taps:
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
                    fn_args += [args[j+n_seqs] ]
                else:
                  k = i + sz + tap_value
                  if k < 0:
                     # past value not provided.. issue a warning and use 0s
                      fn_args += [numpy.zeros(args[j+n_seqs][0].shape)]
                      warning(('Past value %d for output %d not given in '
                               'inital out') % (j,tap_value))
                  else:
                    fn_args += [args[j+n_seqs][k]]
              else:
                if self.stored_steps_output[j] < 1:
                    fn_args += [y[j][i + tap_value]]
                elif self.stored_steps_output[j] == 1:
                    fn_args += [y[j] ]
                else:
                    raise NotImplementedError('This will be implemented in the near future')
        # get the non-iterable sequences
        fn_args += list(args[(n_seqs+n_outs):])
        # compute output
        something = fn(*fn_args)
        #update outputs
        for j in xrange(n_outs):
          if self.stored_steps_output[j] <1:
              y[j][i] = something[j]
          elif self.stored_steps_output[j] == 1:
              y[j] = something[j]
          else:
            raise NotImplementedError('This will be implemented in the near future')
      return y


    def grad(self, args, g_outs):

        raise NotImplementedError('This will be implemented in the near future');
        '''
        if True: 
           #((self.updates.keys() != []) or (self.inplace_map.keys() != [])\
           # or numpy.any(self.stored_steps_output)):
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
            gmap = theano.gradient.grad_sources_inputs( \
                        [(y,g_y)], theano.gof.graph.inputs([y]), False)
            def zero(p):
              return theano.tensor.TensorConstant(theano.tensor.TensorType(\
                      dtype=p.type.dtype, broadcastable=[]),
                      theano._asarray(0,dtype = p.type.dtype))

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
                  g_outs[i] = theano.tensor.zeros_like(y[i])

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


@gof.local_optimizer([None])
def scan_make_inplace(node):
    op = node.op
    if isinstance(op, Scan) and (not op.inplace) \
                            and (op.inplace_map.keys() != []):
        return Scan((op.inputs, op.outputs, op.givens ) , op.n_seqs,  
                    op.n_outs, op.inplace_map, op.seqs_taps, op.outs_taps, 
                    op.truncate_gradient, op.go_backwards, op.stored_steps_output,
                    inplace=True 
                      ).make_node(*node.inputs).outputs
    return False
        
        
optdb.register('scanOp_make_inplace', opt.in2out(scan_make_inplace,
               ignore_newtrees=True), 75, 'fast_run', 'inplace')



'''
class ScanGrad(theano.Op):
    """Gradient Op for Scan"""
    def __init__(self,(g_ins, g_outs) , n_seqs, n_outs, 
                 seqs_taps = {}, outs_taps= {}, truncate_gradient = -1):
        self.grad_fn = theano.function(g_ins, g_outs)
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
        return theano.Apply(self, list(args),
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



