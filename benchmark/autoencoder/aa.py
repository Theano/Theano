from __future__ import absolute_import
import numpy
import sys
import time

import theano
import theano.tensor as T

class Opt(object):
    merge = theano.gof.MergeOptimizer()
    gemm_opt_1 = theano.gof.TopoOptimizer(theano.tensor_opt.gemm_pattern_1)
    sqr_opt_0 = theano.gof.TopoOptimizer(
            theano.gof.PatternSub(
                (T.mul,'x', 'x'),
                (T.sqr, 'x')))
    ident_opt_0 = theano.gof.TopoOptimizer(
            theano.gof.PatternSub(
                (T.sqr, (T.sqrt,'x')),
                'x',
                allow_multiple_clients=True))
    ident_opt_1 = theano.gof.TopoOptimizer(
            theano.gof.PatternSub(
                (T.sqrt, (T.sqr,'x')),
                'x',
                allow_multiple_clients=True))
    ident_muldiv_0 = theano.gof.TopoOptimizer(
            theano.gof.PatternSub(
                (T.mul, 'x', (T.div,'y', 'x')),
                'y',
                allow_multiple_clients=True))
    ident_muldiv_1 = theano.gof.TopoOptimizer(
            theano.gof.PatternSub(
                (T.mul, (T.div,'y', 'x'), 'x'),
                'y',
                allow_multiple_clients=True))
    ident_muldiv_2 = theano.gof.TopoOptimizer(
            theano.gof.PatternSub(
                (T.div, (T.mul,'y', 'x'), 'x'),
                'y',
                allow_multiple_clients=True))
    ident_muldiv_3 = theano.gof.TopoOptimizer(
            theano.gof.PatternSub(
                (T.div, (T.mul,'y', 'x'), 'y'),
                'x',
                allow_multiple_clients=True))

    def __call__(self, env):
        self.merge(env)
        #eliminate identities
        if 0:
            print 'SKIPPING optimizations'
        else:

            self.ident_opt_0(env)
            self.ident_opt_1(env)
            self.ident_muldiv_0(env)
            self.ident_muldiv_1(env)
            self.ident_muldiv_2(env)
            self.ident_muldiv_3(env)
        
            self.gemm_opt_1(env)
            self.sqr_opt_0(env)

            self.merge(env)

def aa_fn(hid_fn, out_fn):

    x = T.matrix() # input, target
    w = T.matrix() # weights
    a = T.vector() # hid bias
    b = T.vector() # output bias

    hid = hid_fn(T.dot(x, w) + a)

    out = out_fn(T.dot(hid, w.T) + b)

    err = 0.5 * T.sum((out - x)**2)

    params = [w, a, b]

    gparams = T.grad(err, params)

    uparams = [T.sub_inplace(p, 0.01 * gp) for p, gp in zip(params, gparams)]

    return theano.function([x, w, a, b], [err] + uparams
            , linker = theano.gof.OpWiseCLinker()
            #, linker = theano.gof.PerformLinker()
            , optimizer = Opt() 
            )


aa_tanh_tanh = aa_fn(T.tanh, T.tanh)

neg, nout, nhid = [int(a) for a in sys.argv[1:]]

rng = numpy.random.RandomState(342)

x = (rng.rand(neg, nout)-0.5) * 1.5
w = rng.rand(nout, nhid)
a = rng.randn(nhid) * 0.0
b = rng.randn(nout) * 0.0

t = time.time()
for i in xrange(1000):
    err_and_stuff = aa_tanh_tanh(x, w, a, b)
print time.time() - t, err_and_stuff[0]

