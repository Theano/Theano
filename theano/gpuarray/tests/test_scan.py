from __future__ import absolute_import, print_function, division
from unittest import TestCase

import numpy
from six.moves import xrange
import theano

from theano.tests import unittest_tools as utt
import theano.sandbox.rng_mrg
from ..basic_ops import GpuFromHost, HostFromGpu
from ..elemwise import GpuElemwise

from .config import mode_with_gpu, test_ctx_name


class T_Scan(TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_one_sequence_one_output_weights_gpu1(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.fvector('u')
        x0 = theano.tensor.fscalar('x0')
        W_in = theano.tensor.fscalar('win')
        W = theano.tensor.fscalar('w')

        mode = mode_with_gpu.excluding('InputToGpuOptimizer')
        output, updates = theano.scan(f_rnn,
                                      u,
                                      x0,
                                      [W_in, W],
                                      n_steps=None,
                                      truncate_gradient=-1,
                                      go_backwards=False,
                                      mode=mode)

        output = GpuFromHost(test_ctx_name)(output)
        f2 = theano.function([u, x0, W_in, W],
                             output,
                             updates=updates,
                             allow_input_downcast=True,
                             mode=mode)

        rng = numpy.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5., high=5.)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        v_u = numpy.asarray(v_u, dtype='float32')
        v_x0 = numpy.asarray(v_x0, dtype='float32')
        W = numpy.asarray(W, dtype='float32')
        W_in = numpy.asarray(W_in, dtype='float32')

        # compute the output in numpy
        v_out = numpy.zeros((4,))
        v_out[0] = v_u[0] * W_in + v_x0 * W
        for step in xrange(1, 4):
            v_out[step] = v_u[step] * W_in + v_out[step - 1] * W

        theano_values = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_values, v_out)

        # TO DEL
        topo = f2.maker.fgraph.toposort()
        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]

        topo = f2.maker.fgraph.toposort()
        assert sum([isinstance(node.op, HostFromGpu)
                    for node in topo]) == 0
        assert sum([isinstance(node.op, GpuFromHost)
                    for node in topo]) == 4

        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any([isinstance(node.op, GpuElemwise)
                    for node in scan_node_topo])
        assert not any([isinstance(node.op, HostFromGpu)
                        for node in scan_node_topo])
        assert not any([isinstance(node.op, GpuFromHost)
                        for node in scan_node_topo])

    # This second version test the second case in the optimizer to the gpu.
    def test_one_sequence_one_output_weights_gpu2(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.fvector('u')
        x0 = theano.tensor.fscalar('x0')
        W_in = theano.tensor.fscalar('win')
        W = theano.tensor.fscalar('w')
        output, updates = theano.scan(f_rnn,
                                      u,
                                      x0,
                                      [W_in, W],
                                      n_steps=None,
                                      truncate_gradient=-1,
                                      go_backwards=False,
                                      mode=mode_with_gpu)

        f2 = theano.function([u, x0, W_in, W],
                             output,
                             updates=updates,
                             allow_input_downcast=True,
                             mode=mode_with_gpu)

        # get random initial values
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5., high=5.)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        v_out = numpy.zeros((4,))
        v_out[0] = v_u[0] * W_in + v_x0 * W
        for step in xrange(1, 4):
            v_out[step] = v_u[step] * W_in + v_out[step - 1] * W

        theano_values = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_values, v_out)

        topo = f2.maker.fgraph.toposort()
        assert sum([isinstance(node.op, HostFromGpu)
                    for node in topo]) == 1
        assert sum([isinstance(node.op, GpuFromHost)
                    for node in topo]) == 4

        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any([isinstance(node.op, GpuElemwise)
                    for node in scan_node_topo])
        assert not any([isinstance(node.op, HostFromGpu)
                        for node in scan_node_topo])
        assert not any([isinstance(node.op, GpuFromHost)
                        for node in scan_node_topo])

    # This third test checks that scan can deal with a mixture of dtypes as
    # outputs when is running on GPU
    def test_gpu3_mixture_dtype_outputs(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return (u_t * W_in + x_tm1 * W,
                    theano.tensor.cast(u_t + x_tm1, 'int64'))

        u = theano.tensor.fvector('u')
        x0 = theano.tensor.fscalar('x0')
        W_in = theano.tensor.fscalar('win')
        W = theano.tensor.fscalar('w')
        output, updates = theano.scan(f_rnn,
                                      u,
                                      [x0, None],
                                      [W_in, W],
                                      n_steps=None,
                                      truncate_gradient=-1,
                                      go_backwards=False,
                                      mode=mode_with_gpu)

        f2 = theano.function([u, x0, W_in, W],
                             output,
                             updates=updates,
                             allow_input_downcast=True,
                             mode=mode_with_gpu)

        # get random initial values
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5., high=5.)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        v_out1 = numpy.zeros((4,))
        v_out2 = numpy.zeros((4,), dtype='int64')
        v_out1[0] = v_u[0] * W_in + v_x0 * W
        v_out2[0] = v_u[0] + v_x0
        for step in xrange(1, 4):
            v_out1[step] = v_u[step] * W_in + v_out1[step - 1] * W
            v_out2[step] = numpy.int64(v_u[step] + v_out1[step - 1])

        theano_out1, theano_out2 = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_out1, v_out1)
        utt.assert_allclose(theano_out2, v_out2)

        topo = f2.maker.fgraph.toposort()
        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        assert scan_node.op.gpua

        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert not any([isinstance(node.op, HostFromGpu)
                        for node in scan_node_topo])
        assert not any([isinstance(node.op, GpuFromHost)
                        for node in scan_node_topo])

    def test_gpu4_gibbs_chain(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_vsample = numpy.array(rng.binomial(1, .5, size=(3, 20),),
                                dtype='float32')
        vsample = theano.shared(v_vsample)
        trng = theano.sandbox.rng_mrg.MRG_RandomStreams(
            utt.fetch_seed())

        def f(vsample_tm1):
            return trng.binomial(vsample_tm1.shape, n=1, p=0.3,
                                 dtype='float32') * vsample_tm1

        theano_vsamples, updates = theano.scan(f,
                                               [],
                                               vsample,
                                               [],
                                               n_steps=10,
                                               truncate_gradient=-1,
                                               go_backwards=False,
                                               mode=mode_with_gpu)
        my_f = theano.function([],
                               theano_vsamples[-1],
                               updates=updates,
                               allow_input_downcast=True,
                               mode=mode_with_gpu)

        # I leave this to tested by debugmode, this test was anyway
        # more of does the graph compile kind of test
        my_f()
