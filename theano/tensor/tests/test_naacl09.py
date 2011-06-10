#CUT-and-PASTE from pylearn.algorithms.daa

import theano
from theano import tensor as T
from theano.tensor import nnet as NN
from theano.compile import module
from theano.compile.mode import get_default_mode
from theano import config
from theano import tensor as T, sparse as S
import numpy as N
import sys
from theano.tests import unittest_tools

def cross_entropy(target, output, axis=1):
    """
    @todo: This is essentially duplicated as nnet_ops.binary_crossentropy
    @warning: OUTPUT and TARGET are reversed in nnet_ops.binary_crossentropy
    """
    return -T.mean(target * T.log(output) + (1 - target) * T.log(1 - output), axis=axis)
def quadratic(target, output, axis=1):
    return T.mean(T.sqr(target - output), axis=axis)

class QuadraticDenoisingAA(module.Module):
    """Quadratic de-noising Auto-encoder

    WRITEME

    Abstract base class. Requires subclass with functions:

    - build_corrupted_input()

    Introductory article about this model WRITEME.

    """

    def __init__(self,
            input = None,
#            regularize = False,
            tie_weights = False,
            n_quadratic_filters = 1,
            _w1 = None,
            _w2 = None,
            _b1 = None,
            _b2 = None,
            _qfilters = None,
            activation_function=NN.sigmoid,
            reconstruction_cost_function=cross_entropy):
        """
        :param input: WRITEME

        :param regularize: WRITEME

        :param tie_weights: WRITEME

        :param activation_function: WRITEME

        :param reconstruction_cost: Should return one cost per example (row)

        :todo: Default noise level for all daa levels

        """
        super(QuadraticDenoisingAA, self).__init__()

        self.random = T.RandomStreams()

        # MODEL CONFIGURATION
#        self.regularize = regularize
        self.tie_weights = tie_weights
        self.activation_function = activation_function
        self.reconstruction_cost_function = reconstruction_cost_function

        # ACQUIRE/MAKE INPUT
        if not input:
            input = T.matrix('input')
        #self.input = theano.External(input)
        self.input = (input)

        # HYPER-PARAMETERS
        #self.lr = theano.Member(T.scalar())
        self.lr = (T.scalar())

        # PARAMETERS
        if _qfilters is None:
            #self.qfilters = [theano.Member(T.dmatrix('q%i'%i)) for i in xrange(n_quadratic_filters)]
            self.qfilters = [(T.dmatrix('q%i'%i)) for i in xrange(n_quadratic_filters)]
        else:
            #self.qfilters = [theano.Member(q) for q in _qfilters]
            self.qfilters = [(q) for q in _qfilters]

        #self.w1 = theano.Member(T.matrix('w1')) if _w1 is None else theano.Member(_w1)
        if _w1 is None:
            self.w1 = (T.matrix('w1'))
        else: self.w1 = (_w1)
        if _w2 is None:
            if not tie_weights:
                #self.w2 = theano.Member(T.matrix())
                self.w2 = (T.matrix())
            else:
                self.w2 = self.w1.T
        else:
            #self.w2 = theano.Member(_w2)
            self.w2 = (_w2)
        #self.b1 = theano.Member(T.vector('b1')) if _b1 is None else theano.Member(_b1)
        if _b1 is None:
            self.b1 = (T.vector('b1'))
        else: self.b1 = (_b1)
        #self.b2 = theano.Member(T.vector('b2')) if _b2 is None else theano.Member(_b2)
        if _b2 is None:
            self.b2 = (T.vector('b2'))
        else: self.b2 = (_b2)

#        # REGULARIZATION COST
#        self.regularization = self.build_regularization()


        ### NOISELESS ###

        # HIDDEN LAYER
        def _act(x):
            if len(self.qfilters) > 0:
                qsum = 10e-10   # helps to control the gradient in the square-root below
                for qf in self.qfilters:
                    qsum = qsum + T.dot(x, qf)**2

                return T.dot(x, self.w1) + self.b1 + T.sqrt(qsum)
            else:
                return T.dot(x, self.w1) + self.b1

        self.hidden_activation = _act(self.input) #noise-free hidden

        self.hidden = self.hid_activation_function(self.hidden_activation)

        # RECONSTRUCTION LAYER
        self.output_activation = T.dot(self.hidden, self.w2) + self.b2
        self.output = self.out_activation_function(self.output_activation)

        # RECONSTRUCTION COST
        self.reconstruction_costs = self.build_reconstruction_costs(self.output)
        self.reconstruction_cost = T.mean(self.reconstruction_costs)

        # TOTAL COST
        self.cost = self.reconstruction_cost
#        if self.regularize:
#            self.cost = self.cost + self.regularization


        ### WITH NOISE ###
        self.corrupted_input = self.build_corrupted_input()

        # HIDDEN LAYER
        self.nhidden_activation = _act(self.corrupted_input)
        self.nhidden = self.hid_activation_function(self.nhidden_activation)

        # RECONSTRUCTION LAYER
        self.noutput_activation = T.dot(self.nhidden, self.w2) + self.b2
        self.noutput = self.out_activation_function(self.noutput_activation)

        # RECONSTRUCTION COST
        self.nreconstruction_costs = self.build_reconstruction_costs(self.noutput)
        self.nreconstruction_cost = T.mean(self.nreconstruction_costs)

        # TOTAL COST
        self.ncost = self.nreconstruction_cost
#        if self.regularize:
#            self.ncost = self.ncost + self.regularization


        # GRADIENTS AND UPDATES
        if self.tie_weights:
            self.params = [self.w1, self.b1, self.b2] + self.qfilters
        else:
            self.params = [self.w1, self.w2, self.b1, self.b2] + self.qfilters

        gradients = T.grad(self.ncost, self.params)
        updates = dict((p, p - self.lr * g) for p, g in zip(self.params, gradients))

        # INTERFACE METHODS
        #self.update = theano.Method(self.input, self.ncost, updates)
        #self.compute_cost = theano.Method(self.input, self.cost)
        #self.noisify = theano.Method(self.input, self.corrupted_input)
        #self.reconstruction = theano.Method(self.input, self.output)
        #self.representation = theano.Method(self.input, self.hidden)
        #self.reconstruction_through_noise = theano.Method(self.input, [self.corrupted_input, self.noutput])

        #self.validate = theano.Method(self.input, [self.cost, self.output])

    def _instance_initialize(self, obj, input_size, hidden_size, seed, lr, qfilter_relscale):
        print 'QDAA init'
        """
        qfilter_relscale is the initial range for any quadratic filters (relative to the linear
        filter's initial range)
        """
        if (input_size is None) ^ (hidden_size is None):
            raise ValueError("Must specify input_size and hidden_size or neither.")
        super(QuadraticDenoisingAA, self)._instance_initialize(obj, {})

        obj.random.initialize()
        R = N.random.RandomState(unittest_tools.fetch_seed(seed))
        if input_size is not None:
            sz = (input_size, hidden_size)
            inf = 1/N.sqrt(input_size)
            hif = 1/N.sqrt(hidden_size)
            obj.w1 = N.asarray(R.uniform(size = sz, low = -inf, high = inf),
                    dtype=config.floatX)
            if not self.tie_weights:
                obj.w2 = N.asarray(
                        R.uniform(size=list(reversed(sz)), low=-hif, high=hif),
                        dtype=config.floatX)
            obj.b1 = N.zeros(hidden_size, dtype=config.floatX)
            obj.b2 = N.zeros(input_size, dtype=config.floatX)
            obj.qfilters = [R.uniform(size = sz, low = -inf, high = inf) * qfilter_relscale \
                    for qf in self.qfilters]
        if seed is not None:
            obj.random.seed(seed)

        obj.lr = N.asarray(lr, dtype=config.floatX)

        obj.__hide__ = ['params']

#    def build_regularization(self):
#        """
#        @todo: Why do we need this function?
#        """
#        return T.zero() # no regularization!


class SigmoidXEQuadraticDenoisingAA(QuadraticDenoisingAA):
    """
    @todo: Merge this into the above.
    @todo: Default noise level for all daa levels
    """
    def setUp(self):
        unittest_tools.seed_rng()

    def build_corrupted_input(self):
        #self.noise_level = theano.Member(T.scalar())
        self.noise_level = (T.scalar())
        return self.random.binomial(T.shape(self.input), 1, 1 - self.noise_level) * self.input

    def hid_activation_function(self, activation):
        return self.activation_function(activation)

    def out_activation_function(self, activation):
        return self.activation_function(activation)

    def build_reconstruction_costs(self, output):
        return self.reconstruction_cost_function(self.input, output)

#    def build_regularization(self):
#        self.l2_coef = theano.Member(T.scalar())
#        if self.tie_weights:
#            return self.l2_coef * T.sum(self.w1 * self.w1)
#        else:
#            return self.l2_coef * (T.sum(self.w1 * self.w1) + T.sum(self.w2 * self.w2))

    def _instance_initialize(self, obj, input_size, hidden_size, noise_level, seed, lr, qfilter_relscale):
#        obj.l2_coef = 0.0
        obj.noise_level = N.asarray(noise_level, dtype=config.floatX)
        super(SigmoidXEQuadraticDenoisingAA, self)._instance_initialize(obj, input_size, hidden_size, seed, lr, qfilter_relscale)

QDAA = SigmoidXEQuadraticDenoisingAA

class Loss01(object):
    def loss_01(self, x, targ):
        return N.mean(self.classify(x) != targ)

class Module_Nclass(module.FancyModule):
    def _instance_initialize(mod_self, self, n_in, n_out, lr, seed):
        #self.component is the LogisticRegressionTemplate instance that built this guy.
        """
        @todo: Remove seed. Used only to keep Stacker happy.
        """

        self.w = N.zeros((n_in, n_out))
        self.b = N.zeros(n_out)
        self.lr = lr
        self.__hide__ = ['params']
        self.input_dimension = n_in
        self.output_dimension = n_out

    def __init__(self, x=None, targ=None, w=None, b=None, lr=None, regularize=False):
        super(Module_Nclass, self).__init__() #boilerplate

        #self.x = module.Member(x) if x is not None else T.matrix('input')
        if x is not None:
            self.x = (x)
        else: self.x = T.matrix('input')
        #self.targ = module.Member(targ) if targ is not None else T.lvector()
        if targ is not None:
            self.targ = (targ)
        else: self.targ = T.lvector()

        #self.w = module.Member(w) if w is not None else module.Member(T.dmatrix())
        if w is not None:
            self.w = (w)
        else: self.w = (T.dmatrix())
        #self.b = module.Member(b) if b is not None else module.Member(T.dvector())
        if b is not None:
            self.b = (b)
        else: self.b = (T.dvector())
        #self.lr = module.Member(lr) if lr is not None else module.Member(T.dscalar())
        if lr is not None:
            self.lr = (lr)
        else: self.lr = (T.dscalar())

        self.params = [p for p in [self.w, self.b] if p.owner is None]

        linear_output = T.dot(self.x, self.w) + self.b

        (xent, softmax, max_pr, argmax) = NN.crossentropy_softmax_max_and_argmax_1hot(
                linear_output, self.targ)
        sum_xent = T.sum(xent)

        self.softmax = softmax
        self.argmax = argmax
        self.max_pr = max_pr
        self.sum_xent = sum_xent

        # Softmax being computed directly.
        softmax_unsupervised = NN.softmax(linear_output)
        self.softmax_unsupervised = softmax_unsupervised

        #compatibility with current implementation of stacker/daa or something
        #TODO: remove this, make a wrapper
        self.cost = self.sum_xent
        self.input = self.x
        # TODO: I want to make output = linear_output.
        self.output = self.softmax_unsupervised

        #define the apply method
        self.pred = T.argmax(linear_output, axis=1)
        #self.apply = module.Method([self.input], self.pred)

        #self.validate = module.Method([self.input, self.targ], [self.cost, self.argmax, self.max_pr])
        #self.softmax_output = module.Method([self.input], self.softmax_unsupervised)

        if self.params:
            gparams = T.grad(sum_xent, self.params)

            #self.update = module.Method([self.input, self.targ], sum_xent,
                    #updates = dict((p, p - self.lr * g) for p, g in zip(self.params, gparams)))

class ConvolutionalMLP(module.FancyModule):
    def __init__(self,
            window_size,
            n_quadratic_filters,
            activation_function,
            reconstruction_cost_function,
            tie_weights = False,
#            _input,
#            _targ
            ):
        super(ConvolutionalMLP, self).__init__()

        #self.lr = module.Member(T.scalar())
        self.lr = (T.scalar())

        self.inputs = [T.dmatrix() for i in range(window_size)]
        self.targ = T.lvector()

        self.input_representations = []
        self.input_representations.append(QDAA(
                            input=self.inputs[0],
                            tie_weights = tie_weights,
                            n_quadratic_filters = n_quadratic_filters,
                            activation_function = activation_function,
                            reconstruction_cost_function = reconstruction_cost_function
                        )
        )

        for i in self.inputs[1:]:
            self.input_representations.append(
                            QDAA(
                                input=i,
                                tie_weights = tie_weights,
                                n_quadratic_filters = n_quadratic_filters,
                                activation_function = activation_function,
                                reconstruction_cost_function = reconstruction_cost_function,
                                _w1 = self.input_representations[0].w1,
                                _w2 = self.input_representations[0].w2,
                                _b1 = self.input_representations[0].b1,
                                _b2 = self.input_representations[0].b2,
                                _qfilters = self.input_representations[0].qfilters
                            )
            )
            assert self.input_representations[-1].w1 is self.input_representations[0].w1

        self.input_representation = T.concatenate([i.hidden for i in self.input_representations], axis=1)
        self.hidden = QDAA(
                        input = self.input_representation,
                        tie_weights = tie_weights,
                        n_quadratic_filters = n_quadratic_filters,
                        activation_function = activation_function,
                        reconstruction_cost_function = reconstruction_cost_function
                    )
        self.output = Module_Nclass(x=self.hidden.hidden, targ=self.targ)

        input_pretraining_params = [
                        self.input_representations[0].w1,
                        self.input_representations[0].w2,
                        self.input_representations[0].b1,
                        self.input_representations[0].b2
                        ] + self.input_representations[0].qfilters
        hidden_pretraining_params = [
                        self.hidden.w1,
                        self.hidden.w2,
                        self.hidden.b1,
                        self.hidden.b2
                        ] + self.hidden.qfilters
        input_pretraining_cost = sum(i.ncost for i in self.input_representations)
        hidden_pretraining_cost = self.hidden.ncost
        input_pretraining_gradients = T.grad(input_pretraining_cost,
                input_pretraining_params)
        hidden_pretraining_gradients = T.grad(hidden_pretraining_cost, hidden_pretraining_params)
        pretraining_updates = \
                dict((p, p - self.lr * g) for p, g in \
                zip(input_pretraining_params, input_pretraining_gradients) \
                + zip(hidden_pretraining_params, hidden_pretraining_gradients))

        self.pretraining_update = module.Method(self.inputs,
                [input_pretraining_cost, hidden_pretraining_cost],
                pretraining_updates)

        finetuning_params = \
                        [self.input_representations[0].w1, self.input_representations[0].b1] + self.input_representations[0].qfilters + \
                        [self.hidden.w1, self.hidden.b1] + self.hidden.qfilters + \
                        [self.output.w, self.output.b]
        finetuning_cost = self.output.cost
        finetuning_gradients = T.grad(finetuning_cost, finetuning_params)
        finetuning_updates = dict((p, p - self.lr * g) for p, g in zip(finetuning_params, finetuning_gradients))
        self.finetuning_update = module.Method(self.inputs + [self.targ], self.output.cost, finetuning_updates)

        #self.validate = module.Method(self.inputs + [self.targ], [self.output.cost, self.output.argmax, self.output.max_pr])
        #self.softmax_output = module.Method(self.inputs, self.output.softmax_unsupervised)

    def _instance_initialize(mod_self, self, input_size, input_representation_size, hidden_representation_size, output_size, lr, seed, noise_level, qfilter_relscale):

        R = N.random.RandomState(unittest_tools.fetch_seed(seed))

        self.input_size = input_size
        self.input_representation_size = input_representation_size
        self.hidden_representation_size = hidden_representation_size
        self.output_size = output_size

        self.lr = N.asarray(lr, dtype=config.floatX)
#        for layer in obj.layers:
#            if layer.lr is None:
#                layer.lr = lr
        assert self.input_representations[-1] is not self.input_representations[0]
        assert self.input_representations[-1].w1 is self.input_representations[0].w1

        for i in self.input_representations:
#            i.initialize(input_size=self.input_size, hidden_size=self.input_representation_size, seed=R.random_integers(2**30), noise_level=noise_level, qfilter_relscale=qfilter_relscale)
            i.initialize(input_size=self.input_size,
                    hidden_size=self.input_representation_size, noise_level=noise_level,
                    seed=int(R.random_integers(2**30)), lr=lr, qfilter_relscale=qfilter_relscale)
            print type(i.w1)
            assert isinstance(i.w1, N.ndarray)

        for i in self.input_representations[1:]:
            print type(i.w1)
            assert isinstance(i.w1, N.ndarray)
            assert (i.w1 == self.input_representations[0].w1).all()
            assert (i.w2 == self.input_representations[0].w2).all()
            assert (i.b1 == self.input_representations[0].b1).all()
            assert (i.b2 == self.input_representations[0].b2).all()
            assert N.all((a==b).all() for a, b in zip(i.qfilters, self.input_representations[0].qfilters))

        self.hidden.initialize(input_size=(len(self.inputs) * self.input_representation_size),
                hidden_size=self.hidden_representation_size, noise_level=noise_level,
                seed=int(R.random_integers(2**30)), lr=lr, qfilter_relscale=qfilter_relscale)

        self.output.initialize(n_in=self.hidden_representation_size, n_out=self.output_size, lr=lr, seed=R.random_integers(2**30))

def create(window_size=3,
        input_dimension=9,
        output_vocabsize=8,
        n_quadratic_filters=2,
        token_representation_size=5,
        concatenated_representation_size=7,
        lr=0.01,
        seed=123,
        noise_level=0.2,
        qfilter_relscale=0.1,
        compile_mode=None):
    """ Create a convolutional model. """
    activation_function = T.tanh

    architecture = ConvolutionalMLP( \
                window_size = window_size,
                n_quadratic_filters = n_quadratic_filters,
                activation_function = activation_function,
                reconstruction_cost_function = quadratic,
                tie_weights = False
            )

    backup = config.warn.sum_div_dimshuffle_bug
    config.warn.sum_div_dimshuffle_bug = False
    try:
        model = architecture.make(input_size=input_dimension, input_representation_size=token_representation_size, hidden_representation_size=concatenated_representation_size, output_size=output_vocabsize, lr=lr, seed=seed, noise_level=noise_level, qfilter_relscale=qfilter_relscale, mode=compile_mode)
    finally:
        config.warn.sum_div_dimshuffle_bug = backup
    return model

def create_realistic(window_size=3,#7,
        input_dimension=200,
        output_vocabsize=23,
        n_quadratic_filters=2,
        token_representation_size=150,
        concatenated_representation_size=400,
        lr=0.001,
        seed=123,
        noise_level=0.2,
        qfilter_relscale=0.1,
        compile_mode=None):
    """ Create a convolutional model. """
    activation_function = T.tanh

    architecture = ConvolutionalMLP( \
                window_size = window_size,
                n_quadratic_filters = n_quadratic_filters,
                activation_function = activation_function,
                reconstruction_cost_function = quadratic,
                tie_weights = False
            )
    model = architecture.make(input_size=input_dimension, input_representation_size=token_representation_size, hidden_representation_size=concatenated_representation_size, output_size=output_vocabsize, lr=lr, seed=seed, noise_level=noise_level, qfilter_relscale=qfilter_relscale, mode=compile_mode)
    return model

def test_naacl_model(iters_per_unsup=3, iters_per_sup=3,
        optimizer=None, realistic=False):
    print "BUILDING MODEL"
    import time
    t = time.time()

    if optimizer:
        mode = theano.Mode(linker='c|py', optimizer=optimizer)
    else: mode = get_default_mode()

    if mode.__class__.__name__ == 'DebugMode':
        iters_per_unsup=1
        iters_per_sup =1

    if realistic:
        m = create_realistic(compile_mode=mode)
    else:
        m = create(compile_mode=mode)

    print 'BUILD took %.3fs'%(time.time() - t)
    prog_str = []
    idx_of_node = {}
    for i, node in enumerate(m.pretraining_update.maker.env.toposort()):
        idx_of_node[node] = i
        if False and i > -1:
            print '   ', i, node, [(ii, idx_of_node.get(ii.owner, 'IN')) for ii in node.inputs]
        prog_str.append(str(node))
    #print input_pretraining_gradients[4].owner.inputs
    #print input_pretraining_gradients[4].owner.inputs[1].owner.inputs
    #sys.exit()

    print "PROGRAM LEN %i HASH %i"% (len(m.pretraining_update.maker.env.nodes), reduce(lambda a, b: hash(a) ^ hash(b),prog_str))

    rng = N.random.RandomState(unittest_tools.fetch_seed(23904))

    inputs = [rng.rand(10,m.input_size) for i in 1,2,3]
    targets = N.asarray([0,3,4,2,3,4,4,2,1,0])
    #print inputs

    print 'UNSUPERVISED PHASE'
    t = time.time()
    for i in xrange(3):
        for j in xrange(iters_per_unsup):
            m.pretraining_update(*inputs)
        s0, s1 = [str(j) for j in m.pretraining_update(*inputs)]
        print 'huh?', i, iters_per_unsup, iters_per_unsup * (i+1), s0, s1
    if iters_per_unsup == 3:
        assert s0.startswith('0.927793')#'0.403044')
        assert s1.startswith('0.068035')#'0.074898')
    print 'UNSUPERVISED took %.3fs'%(time.time() - t)

    print 'FINETUNING GRAPH'
    print 'SUPERVISED PHASE COSTS (%s)'%optimizer
    t = time.time()
    for i in xrange(3):
        for j in xrange(iters_per_unsup):
            m.finetuning_update(*(inputs + [targets]))
        s0 = str(m.finetuning_update(*(inputs + [targets])))
        print iters_per_sup * (i+1), s0
    if iters_per_sup == 10:
        s0f = float(s0)
        assert 19.7042 < s0f and s0f < 19.7043
    print 'SUPERVISED took %.3fs'%( time.time() - t)

def jtest_main():
    from theano import gof
    JTEST = theano.compile.mode.optdb.query(*sys.argv[2:])
    print 'JTEST', JTEST
    theano.compile.register_optimizer('JTEST', JTEST)
    optimizer = eval(sys.argv[1])
    test_naacl_model(optimizer, 10, 10, realistic=False)

def real_main():
    test_naacl_model()

def profile_main():
    # This is the main function for profiling
    # We've renamed our original main() above to real_main()
    import cProfile, pstats, StringIO
    prof = cProfile.Profile()
    prof = prof.runctx("real_main()", globals(), locals())
    stream = StringIO.StringIO()
    stats = pstats.Stats(prof)
    stats.sort_stats("time")  # Or cumulative
    stats.print_stats(80)  # 80 = how many to print
    # The rest is optional.
    # stats.print_callees()
    # stats.print_callers()

if __name__ == '__main__':
    #real_main()
    profile_main()
