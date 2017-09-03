from __future__ import absolute_import, print_function, division
import numpy as np
import unittest

import theano
from theano import config
from theano import tensor as T
from theano.scan_module.scan_op import Scan
from theano.tests import unittest_tools as utt

mode = theano.compile.mode.get_mode(config.mode)


class TestGaussNewton(unittest.TestCase):
    """
    Regression test for code exhibiting various optimization errors.

    This test case is based on code by Sigurd Spieckermann.
    """
    def setUp(self):
        self.rng = np.random.RandomState(utt.fetch_seed())

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
        b_h = theano.shared(np.zeros(10).astype(config.floatX), borrow=True)
        b_y = theano.shared(np.zeros(1).astype(config.floatX), borrow=True)

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
        self._run(100, 10, batch_size=1, mode=mode)


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


class TestPushOutScanOutputDot(object):
    """
    Test class for the PushOutScanOutput optimizer in the case where the inner
    function of a scan op has an output which is the result of a Dot product
    on a non-sequence matrix input to scan and a vector that is the result of
    computation in the inner function.
    """

    def test_dot_not_output(self):
        # Test the case where the vector input to the dot is not already an
        # output of the inner function.

        v = T.vector()
        m = T.matrix()
        output = T.dot(v, m)

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        f_opt = theano.function([v, m], T.jacobian(output, v), mode=opt_mode)

        no_opt_mode = mode.excluding("scanOp_pushout_output")
        f_no_opt = theano.function([v, m], T.jacobian(output, v), mode=no_opt_mode)

        # Ensure that the optimization was performed correctly in f_opt
        # The inner function of scan should have only one output and it should
        # not be the result of a Dot
        scan_node = [node for node in f_opt.maker.fgraph.toposort()
                     if isinstance(node.op, Scan)][0]
        assert len(scan_node.op.outputs) == 1
        assert not isinstance(scan_node.op.outputs[0], T.Dot)

        # Ensure that the function compiled with the optimization produces
        # the same results as the function compiled without
        v_value = np.random.random((4)).astype(config.floatX)
        m_value = np.random.random((4, 5)).astype(config.floatX)

        output_opt = f_opt(v_value, m_value)
        output_no_opt = f_no_opt(v_value, m_value)

        utt.assert_allclose(output_opt, output_no_opt)

    def test_dot_nitsot_output(self):
        # Test the case where the vector input to the dot is already a nitsot
        # output of the inner function.

        a = T.matrix()
        b = T.matrix()

        def inner_fct(vect, mat):
            vect_squared = vect ** 2
            return T.dot(vect_squared, mat), vect_squared

        outputs, updates = theano.scan(fn=inner_fct,
                                          outputs_info=[None]*2,
                                          sequences=a,
                                          non_sequences=b)

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        f_opt = theano.function([a, b], outputs, mode=opt_mode)

        no_opt_mode = mode.excluding("scanOp_pushout_output")
        f_no_opt = theano.function([a, b], outputs, mode=no_opt_mode)

        # Ensure that the optimization was performed correctly in f_opt
        # The inner function of scan should have only one output and it should
        # not be the result of a Dot
        scan_node = [node for node in f_opt.maker.fgraph.toposort()
                     if isinstance(node.op, Scan)][0]
        # NOTE: WHEN INFER_SHAPE IS REENABLED, BELOW THE SCAN MUST
        # HAVE ONLY 1 OUTPUT.
        assert len(scan_node.op.outputs) == 2
        assert not isinstance(scan_node.op.outputs[0], T.Dot)

        # Ensure that the function compiled with the optimization produces
        # the same results as the function compiled without
        a_value = np.random.random((3, 4)).astype(config.floatX)
        b_value = np.random.random((4, 5)).astype(config.floatX)

        output_opt = f_opt(a_value, b_value)
        output_no_opt = f_no_opt(a_value, b_value)

        utt.assert_allclose(output_opt[0], output_no_opt[0])
        utt.assert_allclose(output_opt[1], output_no_opt[1])

    def test_dot_sitsot_output(self):
        # Test the case where the vector input to the dot is not already a
        # non-nitsot (in this case a sitsot) output of the inner function.

        a = T.matrix()
        b = T.matrix()

        def inner_fct(seq1, previous_output1, nonseq1):
            output1 = previous_output1 + seq1
            output2 = T.dot(output1, nonseq1)
            return output1, output2

        outputs, updates = theano.scan(fn=inner_fct,
                                          outputs_info=[a[0], None],
                                          sequences=a,
                                          non_sequences=b)

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        f_opt = theano.function([a, b], outputs, mode=opt_mode)

        no_opt_mode = mode.excluding("scanOp_pushout_output")
        f_no_opt = theano.function([a, b], outputs, mode=no_opt_mode)

        # Ensure that the optimization was performed correctly in f_opt
        # The inner function of scan should have only one output and it should
        # not be the result of a Dot
        scan_node = [node for node in f_opt.maker.fgraph.toposort()
                     if isinstance(node.op, Scan)][0]
        assert len(scan_node.op.outputs) == 2
        assert not isinstance(scan_node.op.outputs[0], T.Dot)

        # Ensure that the function compiled with the optimization produces
        # the same results as the function compiled without
        a_value = np.random.random((3, 4)).astype(config.floatX)
        b_value = np.random.random((4, 5)).astype(config.floatX)

        output_opt = f_opt(a_value, b_value)
        output_no_opt = f_no_opt(a_value, b_value)

        utt.assert_allclose(output_opt[0], output_no_opt[0])
        utt.assert_allclose(output_opt[1], output_no_opt[1])


class TestPushOutSumOfDot():
    """
    Test case for the PushOutScanOutput optimizer in the case where the scan
    is used to compute the sum over the dot products between the corresponding
    elements of two list of matrices.
    """

    def test_machine_translation(self):
        # This test case comes from https://github.com/rizar/scan-grad-speed and
        # is an example of actual computation done with scan in the context of
        # machine translation
        #
        # 'dim' has been reduced from 1000 to 5 to make the test run faster

        # Parameters from an actual machine tranlation run
        batch_size = 80
        seq_len = 50
        n_words = 80 * 50
        dim = 5

        # Weight matrices
        U = theano.shared(np.random.normal(size=(dim, dim),
                                              scale=0.0001).astype(config.floatX))
        U.name = 'U'
        V = theano.shared(U.get_value())
        V.name = 'V'
        W = theano.shared(U.get_value())
        W.name = 'W'

        # Variables and their values
        x = T.tensor3('x')
        x_value = np.random.normal(size=(seq_len, batch_size, dim),
                                      scale=0.0001).astype(config.floatX)

        ri = T.tensor3('ri')
        ri_value = x_value

        zi = T.tensor3('zi')
        zi_value = x_value

        init = T.alloc(np.cast[config.floatX](0), batch_size, dim)
        def rnn_step1(
                # sequences
                x, ri, zi,
                # outputs_info
                h):
            pre_r = ri + h.dot(U)
            pre_z = zi + h.dot(V)
            r = T.nnet.sigmoid(pre_r)
            z = T.nnet.sigmoid(pre_z)

            after_r = r * h
            pre_h = x + after_r.dot(W)
            new_h = T.tanh(pre_h)

            res_h = z * new_h + (1 - z) * h
            return res_h

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        h, _ = theano.scan(rnn_step1, sequences=[x, ri, zi], n_steps=seq_len,
                           outputs_info=init, name='fpass1', mode=opt_mode)
        cost = h[-1].sum()
        grad1 = T.grad(cost, [U, V, W])
        f_opt = theano.function(inputs=[x, ri, zi], outputs=grad1,
                                mode=opt_mode)

        no_opt_mode = mode.excluding("scanOp_pushout_output")
        h, _ = theano.scan(rnn_step1, sequences=[x, ri, zi], n_steps=seq_len,
                outputs_info=init, name='fpass1', mode=no_opt_mode)
        cost = h[-1].sum()
        grad1 = T.grad(cost, [U, V, W])
        f_no_opt = theano.function(inputs=[x, ri, zi], outputs=grad1,
                                   mode=no_opt_mode)

        # Validate that the optimization has been applied
        scan_node_grad = [node for node in f_opt.maker.fgraph.toposort()
                     if isinstance(node.op, Scan)][1]

        for output in scan_node_grad.op.outputs:
            assert not (isinstance(output.owner.op, T.elemwise.Elemwise) and
                        any([isinstance(i, T.Dot) for i
                             in output.owner.inputs]))

        # Compare the outputs of the two functions on the same input data.
        f_opt_output = f_opt(x_value, ri_value, zi_value)
        f_no_opt_output = f_no_opt(x_value, ri_value, zi_value)
        utt.assert_allclose(f_opt_output, f_no_opt_output)

    def test_non_zero_init(self):
        # Test the case where the initial value for the nitsot output is non-zero

        input1 = T.tensor3()
        input2 = T.tensor3()
        input3 = T.tensor3()

        W = theano.shared(np.random.normal(size=(4, 5))).astype(config.floatX)
        U = theano.shared(np.random.normal(size=(6, 7))).astype(config.floatX)

        def inner_fct(seq1, seq2, seq3, previous_output):
            temp1 = T.dot(seq1, W) + seq3
            temp2 = T.dot(seq2, U)
            dot_output = T.dot(temp1, temp2)
            return previous_output + dot_output

        init = T.as_tensor_variable(np.random.normal(size=(3, 7)))

        # Compile the function twice, once with the optimization and once
        # without
        opt_mode = mode.including("scan")
        h, _ = theano.scan(inner_fct,
                sequences=[input1, input2, input3],
                outputs_info=init,
                mode=opt_mode)
        output = h[-1]
        f_opt = theano.function([input1, input2, input3], output,
                                mode=opt_mode)

        no_opt_mode = mode.excluding("scanOp_pushout_output")
        h, _ = theano.scan(inner_fct,
                sequences=[input1, input2, input3],
                outputs_info=init,
                mode=no_opt_mode)
        output = h[-1]
        f_no_opt = theano.function([input1, input2, input3], output,
                                    mode=no_opt_mode)

        # Ensure that the optimization has been applied for f_opt
        # TODO

        # Compare the outputs of the 2 functions
        input1_value = np.random.random((2, 3, 4)).astype(config.floatX)
        input2_value = np.random.random((2, 5, 6)).astype(config.floatX)
        input3_value = np.random.random((2, 3, 5)).astype(config.floatX)

        output_opt = f_opt(input1_value, input2_value, input3_value)
        output_no_opt = f_no_opt(input1_value, input2_value, input3_value)

        utt.assert_allclose(output_opt, output_no_opt)
