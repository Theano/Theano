#!/usr/bin/env python
import numpy as N
import theano
from theano import Op, Apply, tensor as T, Module, Method, Mode, compile
from theano.gof import OpSub, TopoOptimizer

from theano.printing import Print
from theano.tests import unittest_tools

####################
# Library-type stuff
####################

from theano.compile import module
from theano import tensor as T

class StochasticGradientDescent(module.FancyModule):
    """Fixed stepsize gradient descent"""
    def __init__(self, args, cost, params, gradients=None, stepsize=None, WEIRD_STUFF=True):
        """
        :param stepsize: the step to take in (negative) gradient direction
        :type stepsize: None, scalar value, or scalar TensorVariable
        """
        super(StochasticGradientDescent, self).__init__()
        self.WEIRD_STUFF = WEIRD_STUFF
        self.stepsize_init = None

        if stepsize is None:
            self.stepsize = (T.dscalar())
        elif isinstance(stepsize, T.TensorVariable):
            self.stepsize = stepsize
        else:
            if self.WEIRD_STUFF:
                #TODO: why is this necessary? why does the else clause not work?
#                self.stepsize = module.Member(T.dscalar(), init = stepsize)
                self.stepsize = (T.dscalar())
                self.stepsize_init = stepsize
            else:
#                self.stepsize = module.Member(T.value(stepsize))
                self.stepsize = (T.constant(stepsize))#work!

        if self.stepsize.ndim != 0:
            raise ValueError('stepsize must be a scalar', stepsize)

        self.params = params
        if gradients is None:
            self.gparams = T.grad(cost, self.params)
        else:
            self.gparams = gradients

        self.updates = dict((p, p - self.stepsize * g) for p, g in zip(self.params, self.gparams))

        self.step = module.Method(
                args, [],
                updates=self.updates)
        self.step_cost = module.Method(
                args, cost,
                updates=self.updates)
    def _instance_initialize(self, obj):
        if self.WEIRD_STUFF:
            obj.stepsize = self.stepsize_init
        else:
            pass


def sgd_minimizer(stepsize=None, **args):
    def m(i,c,p,g=None):
        return StochasticGradientDescent(i, c, p, stepsize=stepsize, **args)
    return m

class TanhRnn(Op):
    """
    This class implements the recurrent part of a recurrent neural network.

    There is not a neat way to include this in a more fine-grained way in Theano at the moment,
    so to get something working, I'm implementing a relatively complicated Op that could be
    broken down later into constituents.

    Anyway, this Op implements recursive computation of the form:

    .. latex-eqn:
        z_t &= \tanh( z_{t-1} A + x_{t-1})

    For z0 a vector, and x a TxM matrix, it returns a matrix z of shape (T+1, M),
    in which z[0] = z0.

    """
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, z0, A):
        """
        :type x:  matrix (each row is an x_t) (shape: (T, M))
        :type z0:  vector (the first row of output) (shape: M)
        :type A: matrix (M by M)

        """
        x = T.as_tensor_variable(x)
        z0 = T.as_tensor_variable(z0)
        A = T.as_tensor_variable(A)
        z = x.type() #make a new symbolic variable with the same type as x
        return Apply(self, [x, z0, A], [z])

    def perform(self, node, inp, out):
        x, z0, A = inp
        assert x is not None
        assert z0 is not None
        assert A is not None
        T,M = x.shape
        z = N.zeros((T+1, M))
        z[0] = z0
        for i in xrange(T):
            z[i+1] = N.tanh(N.dot(z[i], A) + x[i])
        out[0][0] = z

    def grad(self, inp, grads):
        x, z0, A = inp
        gz, = grads
        z = tanh_rnn(x, z0, A)
        gz_incl_rnn, gx = tanh_rnn_grad(A, z, gz)
        return [gx, gz_incl_rnn[0], (T.dot(z[:-1].T, gx))]
tanh_rnn = TanhRnn()

class TanhRnnGrad(Op):
    """Gradient calculation for TanhRnn"""
    view_map = {0: [2]}
    def __init__(self):
        pass

    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def make_node(self, A, z, gz):
        return Apply(self, [A,z,gz], (z.type(), gz.type()))

    def perform(self, node, inp, out):
        A, z, gz = inp
        Tp1,M = z.shape
        T = Tp1 - 1
        gx = N.zeros((T, M))

        for i in xrange(T-1, -1, -1):
            #back through the tanh
            gx[i] = gz[i+1] * (1.0 - z[i+1] * z[i+1])

        out[0][0] = gz
        out[1][0] = gx

    def __str__(self):
        return super(TanhRnnGrad, self).__str__()

tanh_rnn_grad = TanhRnnGrad()


#######################
# Experiment-type stuff
#######################



class ExampleRNN(Module):

    def __init__(self, n_vis, minimizer):
        super(ExampleRNN, self).__init__()

        self.n_vis = n_vis

        #recurrent weight matrix in latent space
        self.z0 = (T.dvector())
        self.w = (T.dmatrix())

        self.params = [self.z0, self.w]

        #input and target
        x, y = T.dmatrix(), T.dmatrix()

        z = tanh_rnn(x, self.z0, self.w)
        self.cost = T.sum(z[1:])

        # using the make_minimizer protocol
        self.minimizer = minimizer([x, y], self.cost, self.params)

    def _instance_initialize(self, obj):
        print 'INITIALIZE EXAMPLE RNN'
        n_vis = self.n_vis

        rng = N.random.RandomState(unittest_tools.fetch_seed(2342))

        obj.z0 = N.zeros(n_vis)
        obj.w = rng.randn(n_vis, n_vis) * 0.01
        obj.minimizer.initialize()

def test_example_rnn():
    minimizer_fn = sgd_minimizer(stepsize = 0.001)

    n_vis = 5
    n_out = 3
    n_hid = 4
    rnn_module = ExampleRNN(n_vis, minimizer_fn)

    rnn = rnn_module.make()

    rng = N.random.RandomState(unittest_tools.fetch_seed(7722342))
    x = rng.randn(10,n_vis)
    y = rng.randn(10,n_out)

    #set y to be like x with a lag of LAG
    LAG = 4
    y[LAG:] = x[:-LAG, 0:n_out]

    if 1:
        for i, node in enumerate(rnn.minimizer.step_cost.maker.env.toposort()):
            print i, node

    niter=1500
    if theano.config.mode=='DEBUG_MODE':
        niter=30

    for i in xrange(niter):
        if i % 100 == 0:
            print i, rnn.minimizer.step_cost(x, y), rnn.minimizer.stepsize
        else:
            rnn.minimizer.step_cost(x, y)
    if theano.config.mode=='DEBUG_MODE':
        assert rnn.minimizer.step_cost(x,y) < -.9 #it starts around -.28
    else:
        assert rnn.minimizer.step_cost(x,y) < -20 #it starts around -.28

def test_WEIRD_STUFF():
    n_vis = 3

    rng = N.random.RandomState(unittest_tools.fetch_seed(7722342))
    x = rng.randn(10,n_vis)
    y = rng.randn(10,n_vis)

    #set y to be like x with a lag of LAG
    LAG = 4
    y[LAG:] = x[:-LAG, 0:n_vis]

    minimizer_fn1 = sgd_minimizer(stepsize = 0.001, WEIRD_STUFF = False)
    minimizer_fn2 = sgd_minimizer(stepsize = 0.001, WEIRD_STUFF = True)
    rnn_module1 = ExampleRNN(n_vis, minimizer_fn1)
    rnn_module2 = ExampleRNN(n_vis, minimizer_fn2)
    rnn1 = rnn_module1.make(mode='FAST_RUN')
#    rnn2 = rnn_module1.make(mode='FAST_COMPILE')#work
#    rnn2 = rnn_module1.make(mode='FAST_RUN')#fail
    rnn2 = rnn_module2.make(mode=Mode('c|py', 'fast_run'))#fail
#    rnn2 = rnn_module1.make(mode=Mode('c|py', 'fast_run').excluding("inplace"))#work
#    rnn2 = rnn_module1.make(mode=Mode('c|py', 'fast_compile'))#work
#    rnn2 = rnn_module1.make(mode=Mode('py', 'fast_run_stable'))#work
#    rnn2 = rnn_module1.make(mode=Mode('py', 'merge'))#work
#    rnn2 = rnn_module1.make(mode=Mode('c|py', 'fast_run').excluding("inplace_opt"))#work
#    rnn2 = rnn_module1.make(mode=Mode('py', 'fast_run'))#fail
    m = Mode('py', 'fast_run')
    for n in m.optimizer: print n.name

    if 0:
        topo1=rnn1.minimizer.step_cost.maker.env.toposort()
        topo2=rnn2.minimizer.step_cost.maker.env.toposort()
        for i in range(len(topo1)):
            print '1',i, topo1[i]
            print '2',i, topo2[i]
    if 1:
        topo1=rnn1.minimizer.step.maker.env.toposort()
        topo2=rnn2.minimizer.step.maker.env.toposort()
        for i in range(len(topo1)):
            print '1',i, topo1[i]
            print '2',i, topo2[i]
    import theano.printing

    print len(rnn1.minimizer.step.maker.inputs)
    print len(rnn2.minimizer.step.maker.inputs)
    print rnn1.minimizer.step.maker.inputs
    print rnn2.minimizer.step.maker.inputs



#    for i in range(1,len(rnn1.minimizer.step.maker.inputs)):
#        print "valid update:",theano.printing.pp(rnn1.minimizer.step.maker.inputs[i].update),
#        print rnn1.minimizer.step.maker.inputs[i].update.name
#        print "other update",theano.printing.pp(rnn2.minimizer.step.maker.inputs[i].update),
#        print rnn2.minimizer.step.maker.inputs[i].update.name
#    print dir(rnn1.minimizer.step.maker.inputs[5].update)
#    print dir(rnn2.minimizer.step.maker.inputs[5].update)



    niter=3
    for i in xrange(niter):
        print rnn1.minimizer.step_cost(x, y)
        print rnn2.minimizer.step_cost(x, y)

    #    assert rnn1.n_vis != rnn2.n_vis or slef.n_hid != rnn2.n_hid or rnn1.n_out != rnn2.n_out
        assert (N.abs(rnn1.z0-rnn2.z0)<1e-8).all()
        print (N.abs(rnn1.w-rnn2.w)<1e-8).all()
        print (N.abs(rnn1.w-rnn2.w))
        print rnn1.w
        print rnn2.w
        assert (N.abs(rnn1.w-rnn2.w)<1e-8).all()

    #    assert b

if __name__ == '__main__':
#    from theano.tests import main
#    main(__file__)
#    test_example_rnn()
    test_WEIRD_STUFF()
