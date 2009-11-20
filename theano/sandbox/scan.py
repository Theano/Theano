"""Provide Scan and related functions


Scanning a function over sequential input(s) producing sequential output(s).

Scanning is a general form of recurrence, which can be used for looping.

The idea is that you 'scan' a function along some input sequence, producing an output at each
time-step that can be seen (but not modified) by the function at the next time-step.
(Technically, the function can see the previous K time-steps.)

So for example, ``sum()`` could be computed by scanning the ``z+x_i`` function over a list,
given an initial state of ``z=0``. 

Special cases:

    - A ``reduce()`` operation can be performed by returning only the last output of a scan.
    
    - A ``map()`` operation can be performed by applying a function that ignores each previous
      output.

Often a for loop can be expressed as a scan() operation, and scan is the closest that theano
comes to looping.

This module provides scanning functionality with the `Scan` Op.

"""
__docformat__ = 'restructedtext en'

import traceback
import numpy 
import theano
import theano.compile.sandbox
from theano.tensor import opt
from theano import gof
from theano.compile import optdb

'''
 TODO : move out of sandbox !
'''

class Scan(theano.Op):
    """Scan a function `fn` over several inputs producing several outputs 

    This Op implements a generalization of scan in which `fn` may consult several previous
    outputs from the past, from positions (taps) relative to the current time.   The number of
    taps (T_j) to use for each output (y_j) must be provided when creating a Scan Op.

    Apply Inputs:

        X sequence inputs x_1, x_2, ... x_X

        Y initial states (u_1, u_2, ... u_Y) for our outputs. Each must have appropriate length
        (T_1, T_2, ..., T_Y).

        W other inputs w_1, w_2, ... w_W

    Apply Outputs:

        Y sequence outputs y_1, y_2, ... y_Y

    Each output y_j is computed one time-step at a time according to the formula:

    .. code-block:: python

        (y_1[t], y_2[t],.., y_Y[t]) = fn(
            x_1[t], x_2[t], ... x_X[t],          # X current input values
            y_1(t-1), y_1(t-2), .., y_1(t-T_1),  # T_1 previous outputs for y_1
            y_2(t-1), y_2(t-2), ..., y_2(t-T_2), # T_2 previous outputs for y_2
            ...,                                 # ...
            y_Y(t-1), y_Y(t-2), ..., y_Y(t-T_Y), # T_Y previous outputs for y_Y
            w_1, w_2,..., w_W)                   # W 'timeless' inputs

    So `fn` must accept X + T_1 + T_2 + ... + T_Y + W arguments.

    There are two high-level methods (`symbolic`, `compiled`) for creating a Scan Op besides
    the low-level `__init__` constructor.  ***Why would you call them?***

    When applying a Scan Op to theano Variables, the order of arguments is very important! When
    using the full flexibility of Scan there can be a lot of arguments, but it is essential to
    put them in the following order: 

     1. "Ignored inputs" (x_i with i < n_inplace_ignore) that will be overwritten by an inplace scan.

     2. Inputs that will be overwritten by an inplace scan (x_i with i < n_inplace)

     3. Remaining Inputs (x_i with i >= n_inplace)

     3. Output states (u_j) corresponding to the outputs that are computed inplace (j <
     n_inplace)

     4. Remaining output states not given in 3 (u_j with j >= n_inplace)

     5. Other inputs (w_1, w_2, ... w_W)


    Inplace Operation
    =================

    The Scan Op supports computing some (`n_inplace`) of the outputs y_j using the memory from
    corresponding inputs x_j.
    It is not possible to indicate precisely which outputs overwrite which inputs, but without
    loss of generality we assume that each of the first `n_inplace` outputs (y_j) overwrites
    the corresponding input (x_j).

    Note that using inplace computations destroys information, and may make it
    impossible to compute the gradient.
    As long as the function 'fn' does not update any of the other
    parameters (w_1,..) a gradient of this operation is supported.
    ***Who will care about this?  Someone just using the Op? Someone writing an inplace
    optimization?*** 

    Ignored Inputs
    ==============

    **** Behaviour?  Rationale?  Use case?

    """
    @classmethod
    def symbolic(cls,(in_args,out_args), n_ins, n_outs,\
                n_inplace=0, n_inplace_ignore=0, taps={},
                mode = 'FAST_RUN'):
        
        # if in_args is not a list assume it is just a variable and 
        # convert it to a list (if this is neither the case the code will 
        # raise an error somewhere else !)
        if not( type(in_args) in (list,tuple)):
            in_args = [in_args]
        # if out_args is not a list assume it is just a variable and 
        # convert it to a list 
        if not (type(out_args) in (list,tuple)):
            out_args = [out_args]
 
        # Create fn 
        my_fn   = theano.compile.sandbox.pfunc(in_args, out_args, mode = mode)

        # Create gradient function 
        gy_next  = [out_args[0].type()]
        g_inputs = theano.tensor.grad(out_args[0],in_args,g_cost=gy_next[-1])
        for y_next in out_args[1:] :
            gy_next +=[y_next.type()]
            g_ls = theano.tensor.grad(y_next,in_args,g_cost=gy_next[-1])
            for i in xrange(len(in_args)):
                g_inputs[i] += g_ls[i]
        g_fn=theano.compile.sandbox.pfunc(gy_next+in_args,g_inputs,
                             mode=mode)

    
        return cls(my_fn, g_fn, n_ins, n_outs,\
                   n_inplace,n_inplace_ignore, taps)

    @classmethod
    def compiled(cls,fn,n_ins, n_outs,\
            n_inplace=0, n_inplace_ignore=0, taps={}):
        """Return a Scan instance that will scan the callable `fn` over `n_ins` inputs and
        `n_outs` outputs.


        """
        return cls(fn, None, n_ins, n_outs, \
                   n_inplace, n_inplace_ignore, taps= taps)



    def __init__(self,fn,grad_fn,n_ins,n_outs,
                 n_inplace=0, n_inplace_ignore=0,                 
                 taps={}, inplace=False):
        """Create an instance of the scan class

        To use Scan, first you need to create it specifying the number of inputs, outputs,
        inplace outputs (see notes below), and inputs to be ignored, a dictionary describing
        the time taps used, the function that will be applied recursively and optionally, the
        gradient function (or a symbolic definition of the function and the op will compute the
        gradient on its own). Secondly you just call the op with a list of parameters.

        :param fn: compiled function that takes you from time step t-1 to t

        :param grad_fn: gradient of the function applied recursevly
 
        :param n_ins: number of inputs; in the list of arguments
        they start from 0 to 'n_ins'

        :param n_outs: number of outputs; in the list of arguments you 
        need to give the initial state of each outputs, this will be from 
        'n_ins' to 'n_outs'; each initial state should be a matrix where 
        the first dimension is time and should be sufficiently large to 
        cover the time taps. The matrix for an initial state should be 
        ordered such that if you use k delays, index 0 of matrix stands for 
        the value at time -k, index 1 for value at time 1-k, index 2 for 
        value at time 2-k and index k-1 for value at time -1

        :param n_inplace: indicates the number of outputs that should be 
        computed inplace; in the list of arguments there will be the first
        'n_inplace' outputs in place of the first 'n_inplace' inputs

        :param n_inplace_ignore: indicates the number of inputs that are 
        given just to be replaced by the inplace computation and which
        should not be given as arguments to the function applied 
        recursevly


        :param taps: a dictionary which for each output index gives
        a list of what taps it uses; a tap is given as an int, 
        where x stands for output(t - x); note that a past trace of 1 makes
        no sense, since you get that by default

        :param inplace: is used by the optimizer that allows the inplace 
        computation
        """
        if n_ins < 1:
            raise ValueError('Scan should iterate over at least on one input')

        if n_outs <1:
            raise ValueError('Scan should have at least one output')
        if (n_inplace > n_ins):
            raise ValueError('Number of inplace outputs should be smaller than '
                     'the number of inputs.')
        if (n_inplace < 0):
            raise ValueError('Number of inplace outputs should be larger '
                             'or equal to 0')
        if (n_inplace_ignore > n_inplace):
            raise ValueError('Number of inputs to ignore should not be '\
                             'larger than number of inplace outputs')
        if (n_inplace_ignore < 0):
            raise ValueError('n_inplace_ignore should be non-negative')

        self.destroy_map = {}
        if inplace:
            for i in xrange(n_inplace):
                self.destroy_map.update( {i:[i]} )

        for (k,v) in taps.iteritems():
            if k < 0 or k > n_outs:
                raise ValueError('Taps dictionary contains wrong key!')
            for vi in v:
                # why is it illegal to specify  vi < 2?  
                # what is special about vi == 1?
                #
                # Would it be simpler to just leave v alone if it is non-empty (checking that
                # all vi are >=1) and set v = [1] for all missing output keys?
              if vi < 2:
                raise ValueError('Taps dictionary contains wrong values!')

        self.taps   = taps
        self.n_ins  = n_ins
        self.n_outs = n_outs
        self.n_inplace = n_inplace
        self.inplace = inplace
        self.n_inplace_ignore = n_inplace_ignore
        self.fn = fn
        self.grad_fn = grad_fn

    def make_node(self,*inputs):
        """Create an node for the Scan operation

        :param inputs: list of inputs for the operations; they should be 
        at least 'self.n_ins'+'self.n_outs' arguments; first 'self.n_inplace'
        are inputs that are replaced inplace, followed by oter inputs up 
        to 'self.n_ins'; next 'self.n_outs' are ouputs followed by other 
        arguments that will be given to the function applied recursevly
        """

        n_args = len(inputs)
        min_n_args = self.n_ins+self.n_outs
        if n_args < min_n_args:
            err = 'There should be at least '+str(min_n_args)+ 'arguments'
            raise ValueError(err)

        # Create list of output datatypes
        out_types = []
        for i in xrange(self.n_ins,self.n_ins+self.n_outs):
            out_types += [theano.tensor.Tensor(dtype=inputs[i].dtype,\
                    broadcastable=(False,)+inputs[i].broadcastable[1:])()]
        return theano.Apply(self,inputs, out_types)




    def __eq__(self,other):
        rval = type(self) == type(other)
        if rval:
            rval = (self.fn is other.fn) and \
                   (self.grad_fn is other.grad_fn) and \
                   (self.n_ins == other.n_ins) and \
                   (self.n_outs == other.n_outs) and \
                   (self.n_inplace == other.n_inplace) and \
                   (self.n_inplace_ignore == other.n_inplace_ignore) and\
                   (self.inplace == other.inplace) and\
                   (self.taps == other.taps) 
        return rval

    def __hash__(self):
        # hash the taps dictionary
        taps_hash = 0
        for k,v in self.taps.iteritems():
            taps_hash ^= k
            for vi in v : 
                taps_hash ^= vi
            
        return hash(type(self)) ^ \
               hash(self.fn) ^ \
               hash(self.grad_fn) ^ \
               hash(self.n_ins) ^ \
               hash(self.n_outs) ^ \
               hash(self.n_inplace) ^ \
               hash(self.n_inplace_ignore) ^\
               hash(self.inplace) ^\
               taps_hash 




    def grad(self, inputs, g_outs):
        
        if self.grad_fn == None:
            print 'Warning! no gradient for the recursive function was given'
            return [None for i in inputs]
        else:
            y = self(*inputs)
            if not( type(y) in (list,tuple)):
                y = [y]
 
            for i in xrange(len(y)):
                if g_outs[i] == None:
                    g_outs[i] = theano.tensor.zeros_like(y[i])

            # Construct my gradient class: 
            gradScan = ScanGrad(self.grad_fn, 
                            self.n_ins- self.n_inplace_ignore, self.n_outs,
                            self.taps)

             
            args = g_outs + y + \
                   inputs[self.n_inplace_ignore:]
            
            grads = gradScan(*args)
            rval = [None for i in inputs[:self.n_inplace_ignore]]+grads
            return rval


    def perform(self,node,args, outs):

        # find number of timesteps, note that a precondition is to have 
        # atleast one input to iterate over
        n_steps = len(args[0])

        # check if we deal with a inplace operation 
        n_inplace = self.n_inplace
        n_inplace_ignore = self.n_inplace_ignore
        if not self.inplace: #if it was not optimized to work inplace
            n_inplace = 0

 
        # check lengths of inputs
        for i in xrange(self.n_ins):
            if args[i].shape[0] != n_steps:
                raise ValueError('All inputs should have n_steps length!')

        # check lengths of initial states
        for i in xrange(self.n_ins, self.n_ins+self.n_outs):
            req_size = 1
            if self.taps.has_key(i- self.n_ins):
                req_size = max(self.taps[i-self.n_ins])
            if len(args[i].shape) == 0:
              raise ValueError('Wrong initial state! ')
            if args[i].shape[0] < req_size:
              raise ValueError('Wrong initial state! ')

        # allocate space for the outputs 
        y = []
        # inplace outputs
        for i in xrange(n_inplace):
            y += [args[i]]
        # add outputs 
        for i in xrange(self.n_ins+n_inplace,self.n_ins+self.n_outs):
            y_shape = (n_steps,)+args[i].shape[1:]
            y += [numpy.empty(y_shape, dtype = args[i].dtype)]

        # iterate
        for i in xrange(n_steps):
            fn_args = []
            # get a time slice of inputs
            for j in xrange(n_inplace_ignore, self.n_ins):
                fn_args += [args[j][i]]
            
            # get past values of outputs (t-1 + taps)
            for j in xrange(self.n_outs):
                # get list of taps
                ls_taps = [1]
                if self.taps.has_key(j):
                    ls_taps += self.taps[j]
                maxVal = max(ls_taps)
                for tap_value in ls_taps:
                    if i - tap_value < 0:
                        fn_args += [args[j+self.n_ins][maxVal-tap_value+i]]
                    else:
                        fn_args += [y[j][i-tap_value]]

            # get the none iterable parameters
            fn_args += list(args[(self.n_ins+self.n_outs):])
            # compute output
            something = self.fn(*fn_args)
            # update y and inplace outputs
            for j in xrange(self.n_outs):
                y[j][i] = something[j]

        # write to storage
        for i in xrange(self.n_outs):
            outs[i][0]=y[i]



@gof.local_optimizer([None])
def scan_make_inplace(node):
    op = node.op
    if isinstance(op, Scan) and (not op.inplace) and (op.n_inplace>0):
        return Scan(op.fn, op.grad_fn, op.n_ins,\
                    op.n_outs, op.n_inplace, op.n_inplace_ignore,\
                    op.taps,inplace=True\
                                       ).make_node(*node.inputs).outputs
    return False

optdb.register('scan_make_inplace', opt.in2out(scan_make_inplace,\
               ignore_newtrees=True), 75, 'fast_run', 'inplace')




class ScanGrad(theano.Op):
    """Gradient Op for Scan"""

    def __init__(self, grad_fn, n_ins, n_outs, 
                 taps = {},inplace=False):
        self.grad_fn = grad_fn
        self.n_ins = n_ins # number of inputs of Scan op not of Grad Scan !!
        self.n_outs = n_outs # number of outs of Scan op not of Grad Scan !!
        self.inplace = inplace
        self.taps = taps
        self.destroy_map = {}
        if self.inplace:
          for i in xrange(self.n_outs):
            # claiming that output "-i" is destroying inputs is the way to
            # declare that no real output is aliased to any inputs.  We just
            # trash the inputs by using them as workspace.
            self.destroy_map.update( {-i:[i]})


    def __eq__(self,other): 
        rval = type(self) == type(other)
        if rval:
           rval = (self.grad_fn is other.grad_fn) and \
                  (self.n_ins == other.n_ins) and \
                  (self.n_outs == other.n_outs) and \
                  (self.inplace == other.inplace) and \
                  (self.taps == other.taps)
        return rval

    def __hash__(self):
        taps_hash = 0 
        for k,v in self.taps.iteritems():
            taps_hash ^= k
            for vi in v :
                taps_hash ^= vi

        return hash(type(self)) ^ \
               hash(self.grad_fn) ^ \
               hash(self.n_ins) ^ \
               hash(self.n_outs) ^ \
               hash(self.inplace) ^ taps_hash

    def make_node(self, *args):
        # input of the gradient op : 
        # | g_outs | y      | ins   | outs   | other_args |
        # | n_outs | n_outs | n_ins | n_outs | unknown    |
        # return 
        # | grad of ins | grad of outs | grad of other_args|
        # |   n_ins     |  n_outs      |  unknown          |
        return theano.Apply(self, list(args),
                    [i.type() for i in args[self.n_outs+self.n_outs:] ])

    def perform(self, node, args, storage):
            # get scan inputs
            inputs = args[self.n_outs+self.n_outs:]
            ins = inputs[:self.n_ins]
            initSt = inputs[self.n_ins:self.n_ins+self.n_outs]
            otherArgs = inputs[self.n_outs+self.n_ins:]
            
            # generate space for gradient 
            # not do if inplace !?
            g_ins   = [numpy.zeros_like(k) for k in ins]
            g_initSt = [numpy.zeros_like(k) for k in initSt]
            g_otherArgs = [numpy.zeros_like(k) for k in otherArgs]
            # get gradient from above
            g_outs = args[:self.n_outs]
            # we modify g_outs inplace ..
            if not self.inplace:
                g_outs = [gout.copy() for gout in g_outs]

            # get the output of the scan operation
            outs = args[self.n_outs:2*self.n_outs]

            # check for Nones (non - differentiable )
            #for i,g_o in enumerate(g_outs):
            #    if numpy.all(g_o == 0.):
            #        g_outs[i] = numpy.zeros_like(outs[i])

            # go back through time to 0 (use a time window !?)
            for i in xrange(len(ins[0])-1,-1,-1):
              # time slice of inputs
              _ins = [arg[i] for arg in ins]
              # time slice of outputs + taps
              _outs = []
              for j in xrange(self.n_outs):
                ls_taps = [1]
                if self.taps.has_key(j):
                    ls_taps += self.taps[j]
                maxVal = max(ls_taps)
                for tap_value in ls_taps:
                    if i - tap_value < 0:
                        _outs += [initSt[j][maxVal-tap_value+i]]
                    else:
                        _outs += [outs[j][i- tap_value]]

              g_out = [arg[i] for arg in g_outs]
              grad_args = g_out + _ins + _outs + otherArgs
              grads=self.grad_fn(*grad_args)
 
              # get gradient for inputs 
              for j in xrange(self.n_ins):
                g_ins[j][i] = grads[j]
              
              # get gradient for outputs
              pos = self.n_ins
              for j in xrange(self.n_outs):
                ls_taps = [1]
                if self.taps.has_key(j):
                    ls_taps += self.taps[j]
                maxVal = max(ls_taps)
                for tap_value in ls_taps:
                    if i - tap_value < 0:
                        g_initSt[j][maxVal-tap_value+i] += grads[pos]
                        pos +=1
                    else:
                       g_outs[j][i-tap_value]+= grads[pos]
                       pos += 1
              for j in xrange(len(g_otherArgs)):
                g_otherArgs[j] += grads[j+pos]
            # return the gradient 
            for i in xrange(len(g_ins)):
                storage[i][0] = g_ins[i] 

            for i in xrange(len(g_initSt)):
                storage[i+self.n_ins][0] = g_initSt[i]

            for i in xrange(len(g_otherArgs)):
                storage[i+self.n_ins+self.n_outs][0] = g_otherArgs[i]


@gof.local_optimizer([None])
def grad_scan_make_inplace(node):
    op = node.op
    if isinstance(op, ScanGrad) and (not op.inplace):
        return ScanGrad(op.grad_fn, op.n_ins, op.n_outs, op.taps, 
                   inplace=True).make_node(*node.inputs).outputs
    return False

optdb.register('grad_scan_make_inplace', opt.in2out(grad_scan_make_inplace,\
               ignore_newtrees=True), 75, 'fast_run', 'inplace')



