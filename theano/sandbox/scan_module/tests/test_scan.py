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

        for n_steps in [-1,1, 5, -5]:
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



