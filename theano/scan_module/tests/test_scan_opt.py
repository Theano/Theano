import numpy
import unittest

import theano
from theano import config
from theano import tensor as T
from theano.tests import unittest_tools as utt

mode = theano.compile.mode.get_mode(config.mode)


class TestGaussNewton(unittest.TestCase):
    """
    Regression test for code exhibiting various optimization errors.

    This test case is based on code by Sigurd Spieckermann.
    """
    def setUp(self):
        self.rng = numpy.random.RandomState(utt.fetch_seed())

    def _run(self, num_features, num_timesteps, batch_size, mode):
        # determine shapes of inputs and targets depending on the batch size
        if batch_size == 1:
            inputs_size = (num_timesteps, num_features)
            targets_size = (num_timesteps, 1)
        else:
            inputs_size = (num_timesteps, batch_size, num_features)
            targets_size = (num_timesteps, batch_size, 1)

        # make inputs and targets shared variables
        inputs = theano.shared(
            self.rng.uniform(size=inputs_size).astype(config.floatX),
            borrow=True)
        targets = theano.shared(
            self.rng.uniform(size=targets_size).astype(config.floatX),
            borrow=True)

        # create symbolic inputs and targets variables
        if batch_size == 1:
            x = T.matrix('inputs')
            t = T.matrix('targets')
        else:
            x = T.tensor3('inputs')
            t = T.tensor3('inputs')
        x.tag.test_value = inputs.get_value(borrow=True)
        t.tag.test_value = targets.get_value(borrow=True)

        # create a set of parameters for a simple RNN
        W_xh = theano.shared(
            (0.01 * self.rng.uniform(
                size=(num_features, 10))).astype(config.floatX),
            borrow=True)
        W_hh = theano.shared(
            (0.01 * self.rng.uniform(size=(10, 10))).astype(config.floatX),
            borrow=True)
        W_hy = theano.shared(
            (0.01 * self.rng.uniform(size=(10, 1))).astype(config.floatX),
            borrow=True)
        b_h = theano.shared(numpy.zeros(10).astype(config.floatX), borrow=True)
        b_y = theano.shared(numpy.zeros(1).astype(config.floatX), borrow=True)

        params = [W_xh, W_hh, W_hy, b_h, b_y]

        # recurrent function
        def step(x_t, h_tm1):
            h = T.tanh(T.dot(h_tm1, W_hh) + T.dot(x_t, W_xh) + b_h)
            return h

        # build recurrent graph
        if batch_size == 1:
            h_0 = T.alloc(0.0, 10).astype(config.floatX)
        else:
            h_0 = T.alloc(0.0, batch_size, 10).astype(config.floatX)
        h, updates = theano.scan(step,
                                 sequences=[x],
                                 outputs_info=[h_0])
        # network output
        y = T.dot(h, W_hy) + b_y

        # Create Gauss-Newton-Matrix object. Not really of any use here, but I
        # need it for Hessian-Free optimization.
        gn = GaussNewtonMatrix(y)

        # compute MSE
        cost = ((t - y) ** 2).sum(axis=1).mean()

        # Compute the cost at some other point in the parameter
        # space. Not really of any use here, but this is how I do it
        # during certain iterations of CG in the HF algorithm. There,
        # it's in fact `pi + current update proposal`.  For simplicity,
        # I just multiply by 2 here.
        cost_ = theano.clone(cost,
                             replace=dict([(pi, 2 * pi) for pi in params]))

        # Compute Gauss-Newton-Matrix times some vector `v` which is `p` in CG,
        # but for simplicity, I just take the parameters vector because it's
        # already there.
        Gv = gn(v=params, cost=cost, parameters=params, damp=T.constant(1.0))

        # compile Theano function
        f = theano.function([], [cost_] + Gv, givens={x: inputs, t: targets},
                            mode=mode)
        # execute
        f()

    def test_batch(self):
        # This runs fine. The batch size is set to something greater than 1,
        # i.e. the data is represented by a tensor3 object.
        self._run(100, 10, batch_size=5, mode=mode)

    def test_nobatch(self):
        # This used to give an error due to optimization "scan_merge_inouts".
        # The batch size is set to 1 and the data is represented by a matrix.
        # As of 2013-10-24, it still triggers an optimization error due to
        # "remove_constants_and_unused_inputs_scan".
        mode_exc = mode.excluding("remove_constants_and_unused_inputs_scan")
        self._run(100, 10, batch_size=1, mode=mode_exc)


class GaussNewtonMatrix(object):
    def __init__(self, s):
        # `s` is the linear network outputs, i.e. the network output
        # without having applied the activation function
        self._s = s

    def __call__(self, v, cost, parameters, damp):
        # compute Gauss-Newton Matrix right-multiplied by `v`
        Jv = T.Rop(self._s, parameters, v)
        HJv = T.grad(T.sum(T.grad(cost, self._s) * Jv), self._s,
                     consider_constant=[Jv])
        JHJv = T.grad(T.sum(HJv * self._s), parameters,
                      consider_constant=[HJv, Jv])

        # apply Tikhonov damping
        JHJv = [JHJvi + damp * vi for JHJvi, vi in zip(JHJv, v)]
        return JHJv
