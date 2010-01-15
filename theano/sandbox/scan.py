"""Provide Scan and related functions


 Scanning a function over sequential input(s) producing sequential output(s).

 Scanning is a general form of recurrence, which can be used for looping.

 The idea is that you 'scan' a function along some input sequence, producing 
 an output at each time-step that can be seen (but not modified) by the 
 function at the next time-step. (Technically, the function can see the 
 previous K  time-steps.)

 So for example, ``sum()`` could be computed by scanning the ``z+x_i`` 
 function over a list, given an initial state of ``z=0``. 

 Special cases:

    - A ``reduce()`` operation can be performed by returning only the last 
      output of a scan.
    
    - A ``map()`` operation can be performed by applying a function that 
      ignores each previous output.

 Often a for loop can be expressed as a scan() operation, and scan is the 
 closest that theano comes to looping.

 This module provides scanning functionality with the `Scan` Op.

"""
__docformat__ = 'restructedtext en'

import numpy 
import theano
from theano.tensor import opt
from theano import gof
from theano.compile import optdb

# Logging function for sending warning or info
import logging
_logger = logging.getLogger('theano.scan')
def warning(*msg):
    _logger.warning('WARNING theano.scan: '+' '.join(msg))
def info(*msg):
    _logger.info('INFO theano.scan: '+' '.join(msg))


# Hashing a list; list used by scan are list of numbers, therefore a list 
# can be hashed by hashing all elements in the list
def hash_list(list):
    hash_value = 0
    for v in list:
        hash_value ^= v
    return hash_value


# Hashing a dictionary; the dictionary used by scan has as keys numbers and 
# as values either numbers or list of numbers
def hash_dict(dictionary):
    hash_value = 0
    for k,v in dictionary,iteritems():
        # hash key
        hash_value ^= k
        if type(v) in (list,tuple):
            hash_value ^= hash_list(v)
        else:
            hash_value ^= v
    return hash_value


def scan(fn, sequnces, non_sequences, seed_values, inplace_map={}, 
         sequences_taps={}, outputs_taps = {},
         len = theano.tensor.zero(), force_gradient = False, 
         truncate_gradient = -1, go_backwards = False, mode = 'FAST_RUN'):
    '''The function creates a more intuitive interface to the scan op.

    This function first creates a scan op object, and afterwards applies it 
    to the input data. The scan operation iterates over X sequences producing
    Y outputs. The function that is applied recursively may consult several 
    previous outputs from the past as well as past values and future values 
    of the input. You can see it as havin the inputs :

        X sequences inptus x_1, x_2, .. x_X

        Y seeds/initial values ( u_1, u_2, .. u_Y) for the outputs

        W non sequences inputs w_1, w_2, .. w_W

    Outputs :
        
        Y sequence outputs y_1, y_2, .. y_Y

    Each otuput y_j computed one time step at a time according to the 
    formula:

    .. code-block:: python

      (y_1[t], y_2[t], .. y_Y[t]) = f( 
        x_1[t-K_1],.. x_1[t],x_1[t+1],.. x_1[t+L_1], # x_1 past and future 
                                                     #values
        x_2[t-K-2],.. x_2[t],x_2[t+1],.. x_2[t+L_2], # x_2 past and future 
                                                     # values
        ...                                          # ...
        y_1[t-1], y_1[t-2], .. y[t - T_1],           # past values of y_1
        y_2[t-1], y_2[t-2], .. y[t - T_2],,          # past values of y_2 
        ...
        w_1, w_2, .., w_W)                           # 'timeless' inputs 


    
    :param fn: fn is a lambda expression or a function that given a list of 
    symbolic inputs returns the update list and symbolic outputs list of the 
    function that shall be applied recursively. 

    :param sequences:list of sequences over which the scan op should iterate;
    sequnces length should also cover past and future taps; for example if 
    you also use for a sequence the past tap -3 and future tap +4, to total 
    length should be n+7, where first 3 values of sequence are those 
    corresponding to -3 -2 -1 and the last 4 values correspond to n+1 n+2 
    n+3 and n+4

    :param non_sequences: list of inputs over which it shouldn't iterate 

    :param seed_values: seeds (initial values) of the outputs; if past taps 
    are this seeds should contain enough values to cover this past values; 
    note that index 0 of a seed belongs to the largest past tap 
    
    :param inplace_map: a dictionary telling which output should be 
    computed in place of which input sequence ; input sequence has to be 
    of the same shape as the output

    :param sequence_taps: a dictionary telling for each sequence what past 
    and future taps it should use; past values should be negative, future
    taps positives; by default 0 is added in this dictionary (current value)
    if nothing is provided

    :param outputs_taps: a dictionary telling for each output what past 
    taps it should use (negative values); by default -1 is added to this 
    dictionary if nothing is provided

    :param len: a value (or theano scalar) describing for how many steps 
    the scan should iterate; 0 means that it should iterate over the entire
    length of the input sequence(s)

    :param force_gradient: a flag telling scan op that the gradient can be 
    computed even though inplace or updates are used - use this on your own
    risk

    :param truncate_gradient: tells for how many steps should scan go 
    back in time on the backward pass of backpropagation through time 

    :param go_backwards: a flag indicating if scan should iterate back from 
    the end of the sequence to the begining (if it is true) or from 0 to 
    the end

    :param mode: indicates the mode that should be used to compile the
    function that will be applied recursively

    '''


    # check if inputs are just single variables instead of lists     
    if not (type(sequences) in (list, tuple)):
        seqs = [sequences]
    elif seqs = sequences
        
    if not type(seed_values) in (list,tuple)):
        seeds = [seed_values]
    elif 
        seeds = seed_values
        
    if not (type(non_sequences) in (list,tuple)):
        non_seqs = [non_sequences]
    elif 
        non_seqs = non_sequences



    # compute number of sequences and number of seeds    
    n_seqs     = len(seqs)

    # see if there are outputs that do not feed anything back to the function
    # applied recursively
    outs_tapkeys = outputs_taps.keys()
    for k in outs_tapkeys.sort():
        if outputs_taps[k] == []
            # add empty lists where you have outputs that do not have past 
            # values
            seeds = seeds[:k] + [[]] + seeds[k:]

    n_seeds   = len(seeds)

    # update sequences_taps[idx] to contain 0 if it is not defined
    for i in xrange(n_seqs):
        if not sequences_taps.has_key(i):
            sequences_taps.update({i:[0]})
        # if input sequence is not actually used by the recursive function
        elif sequences_taps[i] == []:
            sequences_taps.__delitem__(i)
        elif not (sequences_taps[i] in (list,tuple)):
            sequences_taps[i] = [sequences_taps[i]]

    # update outputs_taps[idx] to contain -1 if it is not defined
    for i in xrange(n_seeds):
        if not outputs_taps.has_key(i):
            outputs_taps.update({i:-1})
        # if output sequence is not actually used as input to the recursive 
        # function
        elif outputs_taps[i] == []:
            outputs_taps.__delitem__(i)
        elif not(outputs_taps[i] in (list,tuple)):
            outputs_taps[i] = [outputs_taps[i]]


    # create theano inputs for the recursive function  
    args = []
    for (i,seq) in enumerate(seqs):
      if sequences_taps.has_key(i):
        for k in len(sequences_taps[i]):
            args += [seq[0].type() ]
    for (i,seed) in enumerate(seeds):
      if outputs_taps.has_key(i):
        for k in len(outputs_taps[i]):
            args += [seed[0].type() ]

    args += non_seqs
    next_outs, updates = fn(*args)

    # Create the Scan op object
    local_op = Scan( (args,next_outs, updates), n_seqs,n_seeds,inplace_map,
            sequences_taps, outputs_taps, force_gradient, truncate_gradient,
            go_backwards, mode)

    # Call the object on the input sequences, seeds, and non sequences
    return local_op( *(    [thenao.tensor.as_tensor(len)]  \
                         + seqs \
                         + seeds \
                         + non_seqs))




''' The class implementing the scan op 

The actual class. I would not recommend using it directly unless you really 
know what you are doing' 
'''
class Scan(theano.Op):
    def __init__(self,(inputs, outputs, updates),n_seqs, n_seeds,
                 inplace_map={}, seqs_taps={}, outs_taps={},
                 force_gradient = False, truncate_gradient = -1,
                 go_backwards = False, inplace=False):
        '''
        :param inputs: list of symbolic inputs of the function that will 
        be applied recursively 

        :param outputs: list of symbolic outputs for the function applied 
        recursively

        :param updates: list of updates for the function applied recursively

        :param n_seqs: number of sequences in the input over which it needs
        to iterate

        :param n_seeds: number of outputs (same as the number of seeds) 

        :param inplace_map: dictionary discribing which output should be 
        computed inplace of which input 

        :param seqs_taps: dictionary discribing which past and future taps
        of the input sequences are used by the recursive function

        :param outs_taps: dictionary discribing which past taps of the 
        outputs the recursive function is using 

        :param force_gradient: a flag indicating if the gradient is still 
        computable even though inplace operation or updates are used

        :param truncate_gradient: if different from -1 it tells after how 
        many steps in the backward pass of BPTT 
        '''
        

        # check inplace map
        for _out,_in in inplace_map.iteritems():
            if _out > n_seeds:
                raise ValueError(('Inplace map reffers to an unexisting'\
                          'output %d')% _out)
            if _in > n_seqs:
                raise ValueError(('Inplace map reffers to an unexisting'\
                          'input sequence %d')%_in)
            if (_in >= 0) and (min(seqs_taps[_in]) < 0):
                raise ValueError(('Input sequence %d uses past values that '\
                         'will be overwritten by inplace operation')%_in)


        #check sequences past taps
        for k,v in seqs_taps.map_iteritems():
          if k > n_seqs:
            raise ValueError(('Sequences past taps dictionary reffers to '
                    'an unexisting sequence %d')%k)

        #check outputs past taps
        for k,v in outs_taps.map_iteritems():
          if k > n_seeds:
            raise ValueError(('Sequences past taps dictionary reffers to '
                    'an unexisting sequence %d')%k)
          if max(v) > -1:
            raise ValueError(('Can not require future value %d of output'
                    '%d')%(k,max(v)))



        self.destroy_map = {}
        if inplace:
            self.destroy_map = inplace_map

        self.seqs_taps      = seqs_taps
        self.outs_taps      = outs_taps
        self.n_seqs         = n_seqs
        self.n_seeds        = n_seeds
        self.n_args         = n_seqs+n_seeds+1
        self.inplace_map    = inplace_map
        self.inplace        = inplace
        self.inputs         = inputs
        self.outputs        = outputs
        self.updates        = updates
        self.force_gradient = force_gradient
        self.truncate_gradient = truncate_gradient
        self.go_backwards   = go_backwards
    

        self.fn = theano.function(inputs,outputs, \
                                   updates = updates, mode = mode)

        g_y = [outputs[0].type()]
        g_args = theano.tensor.grad(outputs[0],inputs, g_cost = g_y[-1])
        # for all outputs compute gradients and then sum them up
        for y in outputs[1:]:
            g_y += [y.type()]
            g_args_y = theano.tensor.grad(y,inputs, g_cost=g_y[-1])
            for i in xrange(len(g_args)):
                g_args[i] += g_args_y[i]


        self.g_ins = g_y+inputs   
        self.g_outs = g_args


    def make_node(self,*inputs):
      n_args = len(inputs)
      if n_args < self.n_args :
         err = 'There should be at least '+str(self.n_args)+ 'arguments'
         raise ValueError(err)

      # Create list of output datatypes
      out_types = []
      for i in xrange(self.n_seqs+1, self.n_seqs+self.n_seeds+1):
         out_types += [theano.tensor.Tensor(dtype=inputs[i].dtype,\
                 broadcastable=(False,)+inputs[i].broadcastable[1:])()]
      return theano.Apply(self,inputs, out_types)


    def __eq__(self,other):
      rval = type(self) == type(other)
      if rval:
        rval = (self.inputs == other.inputs) and \
               (self.outputs ==  other.outputs) and \
               (self.updates == other.updates) and \
               (self.g_ins == other.g_ins) and \
               (self.g_outs == other.g_outs) and \
               (self.seqs_taps == other.seqs_taps) and \
               (self.outs_taps == other.outs_taps) and \
               (self.inplace_map == other.inplace_map) and \
               (self.n_seqs == other.n_seqs) and\
               (self.inplace == other.inplace) and\
               (self.go_backwards == other.go_backwards) and\
               (self.truncate_gradient == other.truncate_gradient) and\
               (self.force_gradient = other.force_gradient) and\
               (self.n_seeds == other.n_seeds) and\
               (self.n_args == other.n_args)
      return rval

    def __hash__(self):
      return hash(type(self)) ^ \
             hash(self.n_seqs) ^ \
             hash(self.n_seeds) ^ \
             hash(self.force_gradient) ^\
             hash(self.inplace) ^\
             hash(self.go_backwards) ^\
             hash(self.truncate_gradient) ^\
             hash(self.n_args) ^ \
             hash_list(self.outputs) ^ \
             hash_list(self.inputs) ^ \
             hash_list(g_ins) ^ \
             hash_list(h_outs) ^ \
             hash_dict(self.seqs_taps) ^\
             hash_dict(self.outs_taps) ^\
             hash_dict(self.inplace_map) ^\
             hash_dict(self.updates)




    def perform(self,node,args, outs):

        n_steps = 0 
        if (self.n_seqs ==0 ) and (args[0] == 0)
            raise ValueError('Scan does not know over how many steps it '
                'should iterate! No input sequence or number of steps to '
                'iterate given !')

        if (args[0] != 0):
            n_steps = args[0]
        
        for i in xrange(self.n_seqs):
          if self.seqs_taps.has_key(i):
              # compute actual length of the sequence ( we need to see what
              # past taps this sequence has, and leave room for them 
              seq_len = args[i+1].shape[0] + min(self.seqs_taps[i+1])
              if self.seqs_taps[i+1][2] > 0: 
                  # using future values, so need to end the sequence earlier
                  seq_len -= self.seqs_taps[i+1][2]
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

 
        # check lengths of seeds
        for i in xrange(self.n_seqs+1, \
                        self.n_seqs+self.n_seeds+1):
          if self.outs_taps.has_key(i-self.n_seqs-1):
            req_size = abs(min(self.outs_taps[i-self.n_seqs-1]))-1
            if args[i].shape[0] < req_size:
              warning(('Initial state for output %d has fewer values then '
                 'required by the maximal past value %d. Scan will use 0s'
                 ' for missing values')%(i-self.n_iterable-1,req_size))
            
        self.n_steps = n_steps
        y = self.scan(self.fn, args[1:],self.n_seqs, self.n_seeds, 
                 self.seqs_taps, self.outs_taps, n_steps, self.go_backwards, 
                 inplace_map)


        # write to storage
        for i in xrange(self.n_seeds):
            outs[i][0]=y[i]



    def scan(fn, args, n_seqs, n_seeds, seqs_taps, outs_taps,  n_steps, 
             go_backwards, inplace_map):
      y = []
      for i in xrange(self.n_seeds):
        if inplace_map.has_key(i) and (inplace_map[i] >= 0):
          y += [args[inplace_map[i]]]
        else:
          y_shape = (n_steps,)+args[i+self.n_seqs].shape[1:]
          y += [numpy.empty(y_shape,
                            dtype=args[i+self.n_seqs].dtype)]
      #iterate
      if go_backwards:
        the_range = xrange(n_steps-1,-1,-1)
      else:
        the_range = xrange(n_steps)

      seqs_mins = {}
      for j in xrange(self.n_seqs):
        if seqs_taps.has_key(j):
          seqs_mins.update({j:  min(seqs_taps[j])})

      outs_mins = {}
      seed_size = {}
      for j in xrange(self.n_seeds):
        if outs_taps.has_key(j):
          outs_mins.update({j: min(outs_taps[j])})
          seed_size.update({j: args[n_seqs+j].shape[0]})


      for i in the_range:
        fn_args = []

        # sequences over which scan iterates
        for j in xrange(self.n_seqs):
          if seqs_taps.has_key(j):
            ls_taps = seqs_taps[j]
            min_tap = seqs_mins[j]
            for tap_value in ls_taps:
                k = i - min_tap + tap_value
                fn_args += [args[j][k]]

        # seeds or past values of outputs
        for j in xrange(self.n_seeds):
          if outs_taps.has_key(j):
            ls_taps = outs_taps[j]
            min_tap = outs_mins[j]
            seed_sz = seed_size[j]
            for tap_value in ls_taps:
              if i + tap_value < 0:
                k = i + seed_sz + tap_value
                if k < 0
                  # past value not provided.. issue a warning and use 0s
                  fn_args += [numpy.zeros(args[j][0].shape)]
                  warning('Past value %d for output %d not given in seeds' %
                           (j,tap_value))
                else:
                  fn_args += [args[j][k]]
              else:
                fn_args += [y[j][i + tap_value]]

        # get the non-iterable sequences
        fn_args += list(args[(self.n_seqs+self.n_seedss):]
        # compute output
        something = fn(*fn_args)
        #update outputs 
        for j in xrange(self.n_seeds):
          y[j][i] = something[j]
      return y


    def grad(self, args, g_outs):
        if (not self.force_gradient) and \
           ((self.updates.keys() != []) or (self.inplace_map.keys() != [])):
            warning('Can not compute gradients if inplace or updates ' \
                    'are used. Use force_gradient if you know for sure '\
                    'that the gradient can be computed automatically.')
            return [None for i in inputs]
        else:
            # forward pass 
            y = self(*args)
            if not( type(y) in (list,tuple)):
                y = [y]
 

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
                              self.n_seeds,self.seqs_taps, self.outs_taps,
                              self.truncate_gradient)

            return g_scan(g_args)



@gof.local_optimizer([None])
def scan_make_inplace(node):
    op = node.op
    if isinstance(op, Scan) and (not op.inplace) \
                            and (op.inplace_map.keys() != []):
        return Scan((op.inputs, op.outputs, op.updates), op.n_seqs,  \
                    op.n_seeds, op.inplace_map, op.seqs_taps, op.outs_taps, \
                    op.force_gradient, op.truncate_gradient, \
                    op.go_backwards, inplace=True \
                      ).make_node(*node.inputs).outputs
    return False
        
        
optdb.register('scan_make_inplace', opt.in2out(scan_make_inplace,\
               ignore_newtrees=True), 75, 'fast_run', 'inplace')




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
                seed_size.update({j: g_seeds[j]..shape[0]})

            for i in the_range:
              # time slice of inputs
              _ins = []
              for j in xrange(self.n_seqs)
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
                        _outs += [seeds[j][[k]]
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




