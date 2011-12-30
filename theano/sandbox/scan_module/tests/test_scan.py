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

    # generator network, only one output , type scalar ; no sequence or
    # non sequence arguments
    def test001_generator_one_scalar_output(self):
        def f_pow2(x_tm1):
            return 2 * x_tm1
        state = theano.tensor.scalar('state')
        n_steps = theano.tensor.iscalar('nsteps')
        output, updates = scan_module.scan(f_pow2,
                                           [],
                                           state,
                                           [],
                                           n_steps=n_steps,
                                           truncate_gradient=-1,
                                           go_backwards=False)

        my_f = theano.function([state, n_steps],
                               output,
                               updates=updates,
                               allow_input_downcast=True)

        rng = numpy.random.RandomState(utt.fetch_seed())
        state = rng.uniform()
        steps = 5

        numpy_values = numpy.array([state * (2 ** (k + 1)) for k
                                    in xrange(steps)])
        theano_values = my_f(state, steps)
        assert numpy.allclose(numpy_values, theano_values)
