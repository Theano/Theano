#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import numpy
import sys
import time

import theano
import theano.tensor as T
import theano.sandbox
from six.moves import xrange
from theano.compile import module, Mode, ProfileMode
from theano import gof, Op, Apply

from theano.tensor import blas, opt

# numpy: aa_numpy.py
# c : aa.cc


if 0:
    class Opt(object):
        merge = theano.gof.MergeOptimizer()
        gemm_opt_1 = theano.gof.TopoOptimizer(theano.tensor_opt.gemm_pattern_1)

        gemm_opt_2 = theano.gof.TopoOptimizer( # d -= a * (dot()+transpose(dot))
                theano.gof.PatternSub(
                    (
                        T.sub_inplace,
                        'd',
                        (
                            T.mul,
                            dict(pattern = (T.DimShuffle((), ['x', 'x'], inplace = True), 'a'),
                                allow_multiple_clients = True),
                            (
                                T.add,
                                (T.dot, 'b', 'c'),
                                (T.transpose_inplace, (T.dot, 'f', 'g'))
                            )
                        )
                    ),
                    (
                        T.gemm,
                        (
                            T.gemm,
                            'd',
                            (T.neg, 'a'),
                            (T.transpose_inplace, 'g'),
                            (T.transpose_inplace, 'f'),
                            T.constant(1.0)
                        ),
                        (T.neg, 'a'),
                        'b',
                        'c',
                        T.constant(1.0)
                    ),
                    allow_multiple_clients = False))

        sqr = []
        sqr.append( theano.gof.TopoOptimizer(
                theano.gof.PatternSub(
                    (T.mul,'x', 'x'),
                    (T.sqr, 'x'), allow_multiple_clients=True)))
        sqr.append(theano.gof.TopoOptimizer(
            theano.gof.PatternSub(
                (T.pow, 'x', (T.DimShuffle((), ['x', 'x'], inplace=True), T.constant(2))),
                (T.sqr, 'x'), allow_multiple_clients=True)))

        ident_opt_list = []
        ident_opt_list.append(  # remove explicit copies
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.tensor_copy, 'x'),
                        'x',
                        allow_multiple_clients=True)))
        ident_opt_list.append( # remove double-transpose
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.transpose_inplace, (T.transpose_inplace, 'x')),
                        'x',
                        allow_multiple_clients=True)))

        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.sqr, (T.sqrt,'x')),
                        'x',
                        allow_multiple_clients=True)))
        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.sqrt, (T.sqr,'x')),
                        'x',
                        allow_multiple_clients=True)))
        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.mul, 'x', (T.div,'y', 'x')),
                        'y',
                        allow_multiple_clients=True)))

        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.mul, (T.div,'y', 'x'), 'x'),
                        'y',
                        allow_multiple_clients=True)))

        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.div, (T.mul,'y', 'x'), 'x'),
                        'y',
                        allow_multiple_clients=True)))

        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.div, (T.mul,'y', 'x'), 'y'),
                        'x',
                        allow_multiple_clients=True)))

        def __call__(self, env):
            self.merge(env)
            #eliminate identities
            if 0:
                print('SKIPPING optimizations')
            else:

                for opt in self.ident_opt_list:
                    opt(env)

                for opt in self.sqr:
                    opt(env)

                self.gemm_opt_1(env)
                self.gemm_opt_2(env)

                self.merge(env)

def print_graph_linker(print_prog=True):
    if 1:
        imap = {None:'-'}
        def blah(i, node, thunk):
            imap[node] = str(i)
            if print_prog:# and node.op.__class__ is T.DimShuffle:
                if False and  node.op == T.DimShuffle((), ['x', 'x'], inplace = True):
                    print(node.op == T.DimShuffle((), ['x', 'x'],
                                                  inplace=True), end=' ')
                    print(node.inputs[0], type(node.inputs[0]), end=' ')
                    print(node.inputs[0].equals(T.constant(2)), end=' ')
                outputs = node.outputs
                inputs = theano.gof.graph.inputs(outputs)
                print('node ', i, node, end=' ')
                print(':'.join([imap[inp.owner] for inp in node.inputs]))
                #print theano.sandbox.pprint.pp.process_graph(inputs, outputs)
        return theano.sandbox.wraplinker.WrapLinkerMany(
                [theano.gof.OpWiseCLinker()],
                [theano.sandbox.wraplinker.run_all
                    ,blah
                    #,theano.sandbox.wraplinker.numpy_notall_isfinite
                    ])
    else:
        return theano.gof.OpWiseCLinker()


class M(module.Module):
    def __init__(self):
        super(M, self).__init__()

        x = T.matrix('x') # input, target
        self.w = module.Member(T.matrix('w')) # weights
        self.a = module.Member(T.vector('a')) # hid bias
        self.b = module.Member(T.vector('b')) # output bias

        self.hid = T.tanh(T.dot(x, self.w) + self.a)
        hid = self.hid

        self.out = T.tanh(T.dot(hid, self.w.T) + self.b)
        out = self.out

        self.err = 0.5 * T.sum((out - x)**2)
        err = self.err

        params = [self.w, self.a, self.b]

        gparams = T.grad(err, params)

        updates = [(p, p - 0.01 * gp) for p, gp in zip(params, gparams)]

        self.step = module.Method([x], err, updates=dict(updates))

mod = M()
mode = 'FAST_RUN'
#mode = ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
mode = Mode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker(nice_errors=True))
mode = Mode(optimizer='fast_run', linker='c')
mode = Mode(optimizer='fast_run', linker='c|py')
print(mod.pretty(mode=mode))
m = mod.make(mode=mode)

neg, nout, nhid, niter = [int(a) for a in sys.argv[1:]]
rng = numpy.random.RandomState(342)
m.w = rng.rand(nout, nhid)
m.a = rng.randn(nhid) * 0.0
m.b = rng.randn(nout) * 0.0

x = (rng.rand(neg, nout)-0.5) * 1.5

t = time.time()
for i in xrange(niter):
    err = m.step(x)
print('time: ',time.time() - t, 'err: ', err)
try:
    mode.print_summary()
    pass
except:
    pass


