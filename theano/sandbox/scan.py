"""Provide Scan and related functions"""
__docformat__ = 'restructedtext en'

import traceback
import numpy
import theano

def scan1_lambda(lmbda, x, u, *other_inputs):
    """Scan function `lmbda` over `x`.

    :param lmbda: symbolic computation of the recursive function 'f' in the scan operation. This
    will be called with symbolic inputs, and a symbolic output is expected.  The type of the
    output should match that of y_{i-1}.

    :type lmbda: lambda x_i, y_{i-1}, *other_inputs : y_i

    :param x: iterable over which to scan

    :param u: initial value for y_{i-1}

    :param other_inputs: other variables that are inputs to our lambda expression

    :returns: lmbda scanned over x, starting at u.  (See `Scan1Env`)


    For example:

    .. code-block:: python

        u = dscalar('u')
        c = dscalar('c')
        x = dvector('x')

        y = scan_lambda(
                lambda x_i, y_prev, c: (x_i + y_prev) * c,
                x, u, c)

        f = theano.function([x,u, c], y)

        xval = numpy.asarray([1., 1, 1. , 1, 1])
        uval = numpy.asarray(2.)

        yval = f(xval, uval, 2.0)
        assert numpy.all(yval == [2.,    6.,   14.,   30.,   62.,  126.])

    """

    # construct the env used in the scan
    x_this = x[0].type()
    y_this = u.type()
    y_next = lmbda(x_this, y_this, *other_inputs)
    if y_next.type != u.type:
        raise TypeError('type of lambda recursion must match type of y_prev')
    env = theano.Env([x_this, y_this] + list(other_inputs), [y_next])

    #create a generic constant to hold our env
    env_var = theano.Constant(data=env, type=theano.generic)

    rval = scan1_env(*([env_var, x,u] + list(other_inputs)))
    return rval

class Scan1Env(theano.Op):
    """A Theano loop over one variable

    Scan1Env is less general than `Scan` because it permits looping only over one tensor.

    Scan1Env is defined to behave like this:

    .. code-block:: python
        
        #inputs
        x #a tensor with ndim >= 1
        u #a tensor that is like a row of y
        f #the function to scan over x

        y[0] = u
        for i in xrange(len(x)):
            y[i+1] = f(x[i], y[i])

        #outputs
        y # a tensor with one more leading-dimensional-slices than x
          # each leading-dimensional-slice of which is like u (in terms of shape and dtype)

    The Scan1Env Op works by representing `f` symbolically with an `Env`.

    :note: 
    The Op has two outputs: one for the output y, and one for the function compiled from the
    Env representation of 'f'.
    The second is intended to be a secret output, it is not returned by the 
    ``__call__`` method of this  Op.
    
    :todo: 
    Optimize for the case where y_this is not required to compute y_next.  
    This makes all the updates possible in parallel, it also makes the `u` argument to
    make_node un-necessary.

    """
    
    destroy_map = {}
    view_map = {}
    mode=None
    default_output = 0

    def make_node(self, env_var, x, u, *other_inputs):

        inputs = [x,u] + list(other_inputs)

        if hasattr(env_var, 'data'):
            env = env_var.data
            if len(env.inputs) != len(inputs):
                raise ValueError('Scan: Env has wrong number of inputs for scan')

            if len(env.outputs) != 1:
                raise ValueError('Scan: Env has wrong number of outputs for scan')

            if env.inputs[0].type != x[0].type:
                raise TypeError('Scan: Env input[0] type must match x[0].type')

            if env.inputs[1].type != u.type:
                raise TypeError('Scan: Env input[1] type must match u.type')

        # create the output type by adding a non-broadcastable dimension to u's type
        out_type = theano.tensor.Tensor(dtype=u.dtype, 
                broadcastable=[False] + list(u.broadcastable))

        return theano.Apply(self, [env_var]+inputs, [out_type(), theano.generic()])

    def grad(self, inputs, (g_y, g_fn)):
        assert g_fn is None
            
        y = self(*inputs)
        grads = scan1_grad(g_y, y, *inputs) 

        # trim off the output used to cache the compiled function
        grads_we_care_about = grads[:-1]

        return [None] + grads_we_care_about

    def perform(self, node, args, (y_out, fn_out)):

        env, x, u = args[:3]
        other_args = args[3:]

        #compile the env to a function if necessary
        if fn_out[0] is None:
            assert len(env.outputs) == 1
            fn_out[0] = theano.function(
                    inputs=env.inputs,
                    outputs=env.outputs[0],
                    mode=self.mode)
        fn = fn_out[0]

        # allocate the output ndarray y
        y_shape = (x.shape[0]+1,) + u.shape
        y = numpy.empty(y_shape, dtype=u.dtype)

        # do the scan
        y[0] = u
        for i, x_i in enumerate(x):
            something = fn(x_i, y[i], *other_args)
            y[i+1] = something

        # write to storage
        y_out[0] = y

scan1_env = Scan1Env()


class Scan1EnvGrad(theano.Op):
    """Gradient Op for Scan1Env"""

    def __init__(self, inplace=False):
        self.inplace = inplace
        if inplace:
            self.destroy_map = {1: [3]}

    def make_node(self, g_y, y, scan_env, x, u, *other_inputs):
        return theano.Apply(self,
                [g_y, y, scan_env, x, u] + list(other_inputs), 
                [x.type(), u.type()] + [oi.type() for oi in other_inputs] + [theano.generic()])

    def get_fn(self, scan_env, grad_storage):
        """Return the function to compute gradients during a backward scan

        :postcondition: grad_storage[-1][0] == fn
        """
        # identify the output storage for our compiled function
        fn_storage = grad_storage[-1]
        assert isinstance(scan_env, theano.gof.Env)

        # skip compilation if it's there
        if fn_storage[0] is None:

            # compile the grad function by doing symbolic gradient
            # on the scan Op's env
            y_next = scan_env.outputs[0]
            gy_next = y_next.type()
            inputs = scan_env.inputs # x_this, y_this, *rest
            g_inputs = theano.tensor.grad(y_next, inputs, g_cost=gy_next)

            fn_storage[0] = theano.function(
                    inputs=[gy_next] + inputs,
                    outputs=g_inputs)

        return fn_storage[0]

    def perform(self, node, args, grad_storage):

        #retrieve (or compute) the gradient function
        fn = self.get_fn(args[2], grad_storage)

        #unpack the args
        (g_y, y) = args[0:2]
        (x, u) = args[3:5]
        other_args = args[5:]

        #unpack grad_storage (outputs)
        gx_out, gu_out = grad_storage[0:2]
        g_other_storage = grad_storage[2:-1]

        assert len(other_args) == len(g_other_storage)

        # the algorithm below has to work in-place on g_y,
        # so here we just make a copy of it if we can't work 
        # in-place on the original.
        if not self.inplace:
            g_y = g_y.copy()

        # allocate space to hold the gradient on gx
        gx = numpy.zeros_like(x)
        
        # allocate space to hold the gradient on the other inputs
        g_other = [numpy.zeros_like(other) for other in other_args]

        # loop backward over the elements of x,
        # computing the gradient on several terms:
        # - x[i]
        # - y[i]
        # - other_inputs wrt y[i+1]
        for i in xrange(len(x)-1, -1, -1):
            #print 'x y gy_next', x[i], y[i], g_y[i+1]
            grads = fn(g_y[i+1], x[i], y[i], *other_args) 

            #gx[i] can be set directly from the computed gradient
            gx[i], gy_i = grads[0:2]

            # gy_i has to be added to the existing g_y[i]
            g_y[i] += gy_i

            #now increment the other-input gradient buffers
            assert len(g_other) == (len(grads)-2)
            for g_arg_buffer, g_arg in zip(g_other, grads[2:]):
                g_arg_buffer += g_arg

        #write results into storage locations
        gx_out[0] = gx
        gu_out[0] = g_y[0]
        assert len(g_other_storage) == len(g_other)
        for grad_storage, grad in zip(g_other_storage, g_other):
            grad_storage[0] = grad

scan1_grad = Scan1EnvGrad(inplace=False)
scan1_grad_inplace = Scan1EnvGrad(inplace=True)

#TODO: a specialize-phase optimization to swap in scan1_grad_inplace
