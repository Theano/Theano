import os
import shutil
from tempfile import mkdtemp
import time
import unittest

import cPickle
import numpy
from numpy.testing import dec

import theano
import theano.sandbox.rng_mrg
from theano import tensor
from theano.compile.pfunc import rebuild_collect_shared
from theano.gof.python25 import any
from theano.tests  import unittest_tools as utt
from numpy.testing.noseclasses import KnownFailureTest

from test_utils import *
import theano.sandbox.scan_module as scan_module


class TestScan(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def test001_generator_one_scalar_output(self):
        def f_pow2(x_tm1):
            return 2 * x_tm1

        for n_steps in [-1, 1, 5, -5]:
            state = theano.tensor.scalar('state')
            output, updates = scan_module.scan(f_pow2,
                                               [],
                                               state,
                                               [],
                                               n_steps=n_steps,
                                               truncate_gradient=-1,
                                               go_backwards=False)
            my_f = theano.function([state],
                                   output,
                                   updates=updates,
                                   allow_input_downcast=True)
            if abs(n_steps) == 1:
                assert len([x for x in my_f.maker.env.toposort()
                        if isinstance(x.op, scan_module.scan_op.ScanOp)]) == 0

            rng = numpy.random.RandomState(utt.fetch_seed())
            state = rng.uniform()
            numpy_values = numpy.array([state * (2 ** (k + 1)) for k
                                        in xrange(abs(n_steps))])
            theano_values = my_f(state)
            assert numpy.allclose(numpy_values, theano_values)

    # simple rnn, one input, one state, weights for each; input/state
    # are vectors, weights are scalars
    def test002_one_sequence_one_output_and_weights(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        for n_steps in [-1, 1, 5, -5, None]:
            u = theano.tensor.vector('u')
            x0 = theano.tensor.scalar('x0')
            W_in = theano.tensor.scalar('win')
            W = theano.tensor.scalar('w')
            output, updates = scan_module.scan(f_rnn,
                                          u,
                                          x0,
                                          [W_in, W],
                                          n_steps=n_steps,
                                          truncate_gradient=-1,
                                          go_backwards=False)

            my_f = theano.function([u, x0, W_in, W],
                                   output,
                                   updates=updates,
                                   allow_input_downcast=True)
            if n_steps is not None and abs(n_steps) == 1:
                assert len([x for x in my_f.maker.env.toposort()
                        if isinstance(x.op, scan_module.scan_op.ScanOp)]) == 0
            # get random initial values
            rng = numpy.random.RandomState(utt.fetch_seed())
            v_u = rng.uniform(size=(8,), low=-5., high=5.)
            v_x0 = rng.uniform()
            W = rng.uniform()
            W_in = rng.uniform()

            # compute the output in numpy
            if n_steps is not None and n_steps < 0:
                _v_u = v_u[::-1]
            else:
                _v_u = v_u
            steps = 8
            if n_steps is not None:
                steps = abs(n_steps)

            v_out = numpy.zeros((8,))
            v_out[0] = _v_u[0] * W_in + v_x0 * W

            for step in xrange(1, steps):
                v_out[step] = _v_u[step] * W_in + v_out[step - 1] * W
            v_out = v_out[:steps]
            theano_values = my_f(v_u, v_x0, W_in, W)
            assert numpy.allclose(theano_values, v_out)
