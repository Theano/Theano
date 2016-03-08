from __future__ import absolute_import, print_function, division

import os
import shutil
import sys
from tempfile import mkdtemp
import time
import unittest
import copy

import six.moves.cPickle as pickle
from six.moves import xrange
import numpy
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises
from nose.tools import raises
from numpy.testing import dec

import theano
import theano.sandbox.rng_mrg
from theano import tensor
from theano.compile.pfunc import rebuild_collect_shared
from theano.tests import unittest_tools as utt
import theano.scalar.sharedvar
from theano.scan_module.scan_op import Scan
from theano.compat import PY3, OrderedDict
from theano.tests.unittest_tools import attr


'''
  Questions and notes about scan that should be answered :

   * Scan seems to do copies of every input variable. Is that needed?
   answer : probably not, but it doesn't hurt also ( what we copy is
   theano variables, which just cary information about the type / dimension
   of the data)


   * There is some of scan functionality that is not well documented
'''

if theano.config.mode == 'FAST_COMPILE':
    mode_with_opt = theano.compile.mode.get_mode('FAST_RUN')
else:
    mode_with_opt = theano.compile.mode.get_default_mode()
mode_with_gpu = mode_with_opt.including('gpu', 'scan')
if theano.config.mode in ('DEBUG_MODE', 'DebugMode'):
    mode_nodebug = theano.compile.mode.get_mode('FAST_RUN')
else:
    mode_nodebug = mode_with_opt
mode_with_gpu_nodebug = mode_nodebug.including('gpu', 'scan')


type_eps = {'float64': 1e-7,
            'float32': 3e-3}


class multiple_outputs_numeric_grad:
    """WRITEME"""
    def __init__(self, f, pt, ndarray_mask=None, eps=None):
        """Return the gradient of f at pt.

        This function computes the gradient by a one-sided finite differences
        of a fixed step size (eps).

        It is assumed that f(...) will return a scalar.
        :param eps: the stepsize for the finite differencing. None means
        input dtype-dependent. See `type_eps`.
        """

        def prod(inputs):
            rval = 1
            for i in inputs:
                rval *= i
            return rval
        packed_pt = False
        if not isinstance(pt, (list, tuple)):
            pt = [pt]
            packed_pt = True

        # This mask tells us if we are dealing with an ndarray input or
        # something else ( a random state ? ) with which we shouldn't really
        # mess up
        if not ndarray_mask:
            ndarray_mask = [True for x in pt]

        dtype_eps = type_eps['float64']

        for i, p in enumerate(pt):
            if ndarray_mask[i]:
                pt[i] = numpy.array(p)
                _eps = type_eps[str(pt[i].dtype)]
                if _eps > dtype_eps:
                    dtype_eps = _eps

        self.ndarray_mask = ndarray_mask
        # '''
        # Compute clean output:
        f_x = f(*pt)
        gx = []
        # now iterate over the elements of x and call f on those + delta x
        for i in xrange(len(pt)):
            if ndarray_mask[i]:
                # It is a ndarray that we can tweak
                if eps:
                    _eps = eps
                else:
                    _eps = dtype_eps
                if pt[i].ndim:
                    _g = []
                    # it has several dimensions:
                    for pos in xrange(prod(pt[i].shape)):
                        t = pt[i].copy()
                        t = t.flatten()
                        t[pos] += _eps
                        t = t.reshape(pt[i].shape)
                        f_eps = f(*(pt[:i] + [t] + pt[i + 1:]))
                        _g.append(numpy.asarray((f_eps - f_x) / _eps))
                    gx.append(numpy.asarray(_g).reshape(pt[i].shape))
                else:
                    t = numpy.array(pt[i] + _eps)
                    f_eps = f(*(pt[:i] + [t] + pt[i + 1:]))
                    gx.append(numpy.asarray((f_eps - f_x) / _eps))
        self.gx = gx

    @staticmethod
    def abs_rel_err(a, b, eps=1.0e-10):
        """Return a small number when a and b are close, relative to how big
        they are"""
        return abs(a - b) / (abs(a) + abs(b) + eps)

    def max_err(self, _g_pt):
        """Return the biggest relative error between g_pt and self.gx"""

        g_pt = []
        for i in xrange(len(_g_pt)):
            if self.ndarray_mask[i]:
                g_pt.append(_g_pt[i])
            elif isinstance(_g_pt[i], numpy.ndarray):
                assert numpy.all(_g_pt[i] == 0)
        if len(g_pt) != len(self.gx):
            raise ValueError('argument has wrong number of elements',
                             len(g_pt))
        errs = []

        for i, (a, b) in enumerate(zip(g_pt, self.gx)):
            if a.shape != b.shape:
                raise ValueError('argument element %i has wrong shape %s' %
                                 (i, str((a.shape, b.shape))))
            vv = multiple_outputs_numeric_grad.abs_rel_err(a, b)
            errs.append(numpy.max(
                multiple_outputs_numeric_grad.abs_rel_err(a, b)))
        if numpy.all(numpy.isfinite(errs)):
            return numpy.max(errs), numpy.argmax(errs)
        else:
            return numpy.inf, 0


# TODO: Test this function, and if it works,
# use it with the normal verify_grad rather than the
# copy-and-pasted one above.
# Also - add a reference to this technique in the
# verify_grad method so that other ops with multiple outputs can be tested.
# DONE - rp
def scan_project_sum(*args, **kwargs):
    rng = theano.tensor.shared_randomstreams.RandomStreams(123)
    scan_outputs, updates = theano.scan(*args, **kwargs)
    if type(scan_outputs) not in [list, tuple]:
        scan_outputs = [scan_outputs]
    # we should ignore the random-state updates so that
    # the uniform numbers are the same every evaluation and on every call
    rng.add_default_updates = False
    factors = [rng.uniform(size=s.shape, low=0.1, high=0.9) for s
               in scan_outputs]
    # Random values (?)
    return (sum([(s * f).sum() for s, f in zip(scan_outputs, factors)]),
            updates)


def asarrayX(value):
    return theano._asarray(value, dtype=theano.config.floatX)


def clone_optimized_graph(f):
    maker_ins = [x for x in f.maker.fgraph.inputs
                 if not isinstance(x, theano.tensor.sharedvar.SharedVariable)]
    inps, outs, _ = rebuild_collect_shared(f.maker.fgraph.outputs,
                                           maker_ins,
                                           copy_inputs_over=False)
    ins = [x for x in inps
           if not isinstance(x, theano.tensor.sharedvar.SharedVariable)]
    return (ins, outs)


def grab_scan_node(output):
    if output.owner is None:
        return None
    if output.owner.op.__class__.__name__ == 'Scan':
        return [output.owner]
    rval = []
    for i in output.owner.inputs:
        ri = grab_scan_node(i)
        if ri is not None:
            rval += ri
    if rval is []:
        return None
    else:
        return rval


def scan_nodes_from_fct(fct):
    nodes = fct.maker.fgraph.toposort()
    scan_nodes = [n for n in nodes if isinstance(n.op, Scan)]
    return scan_nodes


class T_Scan(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()
        super(T_Scan, self).setUp()

    # generator network, only one output , type scalar ; no sequence or
    # non sequence arguments
    @dec.skipif(
        isinstance(theano.compile.mode.get_default_mode(),
                   theano.compile.debugmode.DebugMode),
        ("This test fails in DebugMode, because it is not yet picklable."))
    def test_pickling(self):
        def f_pow2(x_tm1):
            return 2 * x_tm1

        state = theano.tensor.scalar('state')
        n_steps = theano.tensor.iscalar('nsteps')
        output, updates = theano.scan(f_pow2,
                                      [],
                                      state,
                                      [],
                                      n_steps=n_steps,
                                      truncate_gradient=-1,
                                      go_backwards=False)
        _my_f = theano.function([state, n_steps],
                                output,
                                updates=updates,
                                allow_input_downcast=True)

        # TESTING PICKLE-ing this function
        origdir = os.getcwd()
        tmpdir = None
        try:
            tmpdir = mkdtemp()
            os.chdir(tmpdir)

            with open('tmp_scan_test_pickle.pkl', 'wb') as f_out:
                pickle.dump(_my_f, f_out, protocol=-1)
            with open('tmp_scan_test_pickle.pkl', 'rb') as f_in:
                my_f = pickle.load(f_in)
        finally:
            # Get back to the original dir, and delete the temporary one.
            os.chdir(origdir)
            if tmpdir is not None:
                shutil.rmtree(tmpdir)

        rng = numpy.random.RandomState(utt.fetch_seed())
        state = rng.uniform()
        steps = 5

        numpy_values = numpy.array([state * (2 ** (k + 1)) for k
                                    in xrange(steps)])
        theano_values = my_f(state, steps)
        utt.assert_allclose(numpy_values, theano_values)

    # Test that the inner input_storage and output_storage are
    # properly cleared
    def test_inner_storage_leak(self):
        def f_pow2(x_tm1):
            return 2 * x_tm1

        state = theano.tensor.scalar('state')
        n_steps = theano.tensor.iscalar('nsteps')
        output, updates = theano.scan(f_pow2,
                                      [],
                                      state,
                                      [],
                                      n_steps=n_steps)

        f = theano.function([state, n_steps],
                            output,
                            updates=updates,
                            allow_input_downcast=True)

        scan_node = [node for node in f.maker.fgraph.toposort()
                     if isinstance(node.op, Scan)]

        assert len(scan_node) == 1
        scan_node = scan_node[0]

        # Make sure they start out as None
        assert all(i.value is None for i in scan_node.op.fn.input_storage)
        assert all(o.value is None for o in scan_node.op.fn.output_storage)

        rng = numpy.random.RandomState(utt.fetch_seed())
        state = rng.uniform()
        steps = 5

        f(state, steps)

        # And that they stay that way
        assert all(i.value is None for i in scan_node.op.fn.input_storage)
        assert all(o.value is None for o in scan_node.op.fn.output_storage)

    # generator network, only one output , type scalar ; no sequence or
    # non sequence arguments
    def test_generator_one_output_scalar(self):
        def f_pow2(x_tm1):
            return 2 * x_tm1

        state = theano.tensor.scalar('state')
        n_steps = theano.tensor.iscalar('nsteps')
        output, updates = theano.scan(f_pow2,
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
        utt.assert_allclose(numpy_values, theano_values)

    def test_subtensor_multiple_slices(self):
        # This addresses a bug reported by Matthias Zoehrer
        # the bug happens when you have multiple subtensors on the output of
        # scan (the bug requires the reshape to be produced, and it has
        # which has something to do with how the subtensors overlap
        def f_pow2(x_tm1):
            return 2 * x_tm1

        state = theano.tensor.vector('state')
        n_steps = theano.tensor.iscalar('nsteps')
        output, updates = theano.scan(f_pow2,
                                      [],
                                      state,
                                      [],
                                      n_steps=n_steps,
                                      truncate_gradient=-1,
                                      go_backwards=False)
        nw_shape = tensor.ivector('nw_shape')
        # Note that the output is reshaped to 3 dimensional tensor, and
        my_f = theano.function([state, n_steps, nw_shape],
                               [tensor.reshape(output, nw_shape, ndim=3)[:-2],
                                output[:-4]],
                               updates=updates,
                               allow_input_downcast=True)
        nodes = [x for x in my_f.maker.fgraph.toposort()
                 if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        # This assertation fails if savemem optimization failed on scan
        if theano.config.mode != "FAST_COMPILE":
            assert nodes[0].op._scan_savemem_visited
        rng = numpy.random.RandomState(utt.fetch_seed())
        my_f(rng.uniform(size=(3,)),
             4,
             numpy.int64([2, 2, 3]))

    @attr('slow')
    def test_only_nonseq_inputs(self):
        # Compile the Theano function
        n_steps = 2
        inp = tensor.matrix()
        broadcasted_inp, _ = theano.scan(lambda x: x,
                                         non_sequences=[inp],
                                         n_steps=n_steps)
        out = broadcasted_inp.sum()
        gr = tensor.grad(out, inp)
        fun = theano.function([inp], [broadcasted_inp, gr])

        # Execute the Theano function and compare outputs to the expected outputs
        inputs = numpy.array([[1, 2], [3, 4]], dtype=theano.config.floatX)
        expected_out1 = numpy.repeat(inputs[None], n_steps, axis=0)
        expected_out2 = numpy.ones(inputs.shape, dtype="int8") * n_steps

        out1, out2 = fun(inputs)
        utt.assert_allclose(out1, expected_out1)
        utt.assert_allclose(out2, expected_out2)

    # simple rnn, one input, one state, weights for each; input/state
    # are vectors, weights are scalars
    def test_one_sequence_one_output_weights(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.vector('u')
        x0 = theano.tensor.scalar('x0')
        W_in = theano.tensor.scalar('win')
        W = theano.tensor.scalar('w')

        output, updates = theano.scan(f_rnn,
                                      u,
                                      x0,
                                      [W_in, W],
                                      n_steps=None,
                                      truncate_gradient=-1,
                                      go_backwards=False)

        f2 = theano.function([u, x0, W_in, W],
                             output,
                             updates=updates,
                             allow_input_downcast=True)
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

    # simple rnn, one input, one state, weights for each; input/state
    # are vectors, weights are scalars; using shared variables
    def test_one_sequence_one_output_weights_shared(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        u = theano.tensor.vector('u')
        x0 = theano.tensor.scalar('x0')
        W_in = theano.shared(asarrayX(rng.uniform()), name='w_in')
        W = theano.shared(asarrayX(rng.uniform()), name='w')

        def f_rnn_shared(u_t, x_tm1, tmp_W_in, tmp_W):
            return u_t * tmp_W_in + x_tm1 * tmp_W

        output, updates = theano.scan(f_rnn_shared,
                                      u,
                                      x0,
                                      [W_in, W],
                                      n_steps=None,
                                      truncate_gradient=-1,
                                      go_backwards=False)
        f3 = theano.function([u, x0],
                             output,
                             updates=updates,
                             allow_input_downcast=True)
        # get random initial values

        v_u = rng.uniform(size=(4,), low=-5., high=5.)
        v_x0 = rng.uniform()
        # compute the output i numpy
        v_out = numpy.zeros((4,))
        v_out[0] = v_u[0] * W_in.get_value() + v_x0 * W.get_value()
        for step in xrange(1, 4):
            v_out[step] = (v_u[step] * W_in.get_value() +
                           v_out[step - 1] * W.get_value())

        theano_values = f3(v_u, v_x0)
        assert numpy.allclose(theano_values, v_out)

    # some rnn with multiple outputs and multiple inputs; other
    # dimension instead of scalars/vectors
    def test_multiple_inputs_multiple_outputs(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        vW = asarrayX(rng.uniform(size=(2, 2), low=-5., high=5.))
        vWout = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        vW_in1 = asarrayX(rng.uniform(size=(2, 2), low=-5., high=5.))
        v_u1 = asarrayX(rng.uniform(size=(3, 2), low=-5., high=5.))
        v_u2 = asarrayX(rng.uniform(size=(3,), low=-5., high=5.))
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        v_y0 = asarrayX(rng.uniform())

        W_in2 = theano.shared(vW_in2, name='win2')
        W = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name='wout')
        W_in1 = theano.tensor.matrix('win')
        u1 = theano.tensor.matrix('u1')
        u2 = theano.tensor.vector('u2')
        x0 = theano.tensor.vector('x0')
        y0 = theano.tensor.scalar('y0')

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, W_in1):
            return [theano.dot(u1_t, W_in1) + u2_t * W_in2 +
                    theano.dot(x_tm1, W), theano.dot(x_tm1, W_out)]

        outputs, updates = theano.scan(f_rnn_cmpl,
                                       [u1, u2],
                                       [x0, y0],
                                       W_in1,
                                       n_steps=None,
                                       truncate_gradient=-1,
                                       go_backwards=False)

        f4 = theano.function([u1, u2, x0, y0, W_in1],
                             outputs,
                             updates=updates,
                             allow_input_downcast=True)

        # compute the values in numpy
        v_x = numpy.zeros((3, 2), dtype=theano.config.floatX)
        v_y = numpy.zeros((3,), dtype=theano.config.floatX)
        v_x[0] = (numpy.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 +
                  numpy.dot(v_x0, vW))
        v_y[0] = numpy.dot(v_x0, vWout)
        for i in xrange(1, 3):
            v_x[i] = (numpy.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 +
                      numpy.dot(v_x[i - 1], vW))
            v_y[i] = numpy.dot(v_x[i - 1], vWout)

        (theano_x, theano_y) = f4(v_u1, v_u2, v_x0, v_y0, vW_in1)
        utt.assert_allclose(theano_x, v_x)
        utt.assert_allclose(theano_y, v_y)

    def test_multiple_outs_taps(self):
        l = 5
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size=(2,), low=-.2, high=.2))
        vW = asarrayX(rng.uniform(size=(2, 2), low=-.2, high=.2))
        vWout = asarrayX(rng.uniform(size=(2,), low=-.2, high=.2))
        vW_in1 = asarrayX(rng.uniform(size=(2, 2), low=-.2, high=.2))
        v_u1 = asarrayX(rng.uniform(size=(l, 2), low=-.2, high=.2))
        v_u2 = asarrayX(rng.uniform(size=(l + 2, 2), low=-.2, high=.2))
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-.2, high=.2))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = theano.shared(vW_in2, name='win2')
        W = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name='wout')
        W_in1 = theano.tensor.matrix('win')
        u1 = theano.tensor.matrix('u1')
        u2 = theano.tensor.matrix('u2')
        x0 = theano.tensor.vector('x0')
        y0 = theano.tensor.vector('y0')

        def f_rnn_cmpl(u1_t,
                       u2_tm1,
                       u2_t,
                       u2_tp1,
                       x_tm1,
                       y_tm1,
                       y_tm3,
                       W_in1):
            return [theano.dot(u1_t, W_in1) +
                    (u2_t + u2_tm1 * u2_tp1) * W_in2 +
                    theano.dot(x_tm1, W),
                    (y_tm1 + y_tm3) * theano.dot(x_tm1, W_out),
                    theano.dot(u1_t, W_in1)]

        outputs, updates = theano.scan(f_rnn_cmpl,
                                       [u1, dict(input=u2, taps=[-1, 0, 1])],
                                       [x0, dict(initial=y0, taps=[-1, -3]),
                                        None],
                                       W_in1,
                                       n_steps=None,
                                       truncate_gradient=-1,
                                       go_backwards=False)

        f = theano.function([u1, u2, x0, y0, W_in1],
                            outputs,
                            updates=updates,
                            allow_input_downcast=True)
        theano_out = f(v_u1,
                       v_u2,
                       v_x0,
                       v_y0,
                       vW_in1)

        ny0 = numpy.zeros((5, 2))
        ny1 = numpy.zeros((5,))
        ny2 = numpy.zeros((5, 2))
        ny0[0] = numpy.dot(v_u1[0], vW_in1) + \
                (v_u2[1] + v_u2[0] * v_u2[2]) * vW_in2 + numpy.dot(v_x0, vW)

        ny1[0] = (v_y0[2] + v_y0[0]) * numpy.dot(v_x0, vWout)
        ny2[0] = numpy.dot(v_u1[0], vW_in1)

        ny0[1] = numpy.dot(v_u1[1], vW_in1) + \
                (v_u2[2] + v_u2[1] * v_u2[3]) * vW_in2 + numpy.dot(ny0[0], vW)

        ny1[1] = (ny1[0] + v_y0[1]) * numpy.dot(ny0[0], vWout)
        ny2[1] = numpy.dot(v_u1[1], vW_in1)

        ny0[2] = numpy.dot(v_u1[2], vW_in1) + \
                (v_u2[3] + v_u2[2] * v_u2[4]) * vW_in2 + \
                numpy.dot(ny0[1], vW)
        ny1[2] = (ny1[1] + v_y0[2]) * numpy.dot(ny0[1], vWout)
        ny2[2] = numpy.dot(v_u1[2], vW_in1)

        ny0[3] = numpy.dot(v_u1[3], vW_in1) + \
                           (v_u2[4] + v_u2[3] * v_u2[5]) * vW_in2 + \
                           numpy.dot(ny0[2], vW)

        ny1[3] = (ny1[2] + ny1[0]) * numpy.dot(ny0[2], vWout)
        ny2[3] = numpy.dot(v_u1[3], vW_in1)

        ny0[4] = numpy.dot(v_u1[4], vW_in1) + \
                           (v_u2[5] + v_u2[4] * v_u2[6]) * vW_in2 + \
                           numpy.dot(ny0[3], vW)

        ny1[4] = (ny1[3] + ny1[1]) * numpy.dot(ny0[3], vWout)
        ny2[4] = numpy.dot(v_u1[4], vW_in1)

    def test_using_taps_sequence(self):
        # this test refers to a bug reported by Nicolas
        # Boulanger-Lewandowski June 6th
        x = theano.tensor.dvector()
        y, updates = theano.scan(lambda x: [x],
                                 sequences=dict(input=x, taps=[-1]),
                                 outputs_info=[None])
        inp = numpy.arange(5).astype('float64')
        rval = theano.function([x], y, updates=updates)(inp)
        assert numpy.all(rval == inp[:-1])

    def test_using_negative_taps_sequence(self):
        # This test refers to a bug reported on github on May 22 2015 by
        # user june-qijun
        def lp(x, x2):
            return x
        x = tensor.fvector('x')
        res, upd = theano.scan(lp,
                               sequences=dict(input=x, taps=[-2, -1]))
        f = theano.function([x], res, updates = upd)

        output =  f([1, 2, 3, 4, 5])
        expected_output = numpy.array([1, 2, 3], dtype="float32")
        utt.assert_allclose(output, expected_output)

    def test_connection_pattern(self):
        """Test connection_pattern() in the presence of recurrent outputs
        with multiple taps.

        This test refers to a bug signaled on the theano-users mailing list
        on March 10 2015 by David Schneider-Joseph.
        """
        def fn(a_m2, a_m1, b_m2, b_m1):
            return a_m1, b_m1

        a0 = theano.shared(numpy.arange(2))
        b0 = theano.shared(numpy.arange(2))

        (a, b), _ = theano.scan(fn,
                        outputs_info=[{'initial': a0, 'taps': [-2, -1]},
                                      {'initial': b0, 'taps': [-2, -1]}],
                        n_steps=2)

        tensor.grad(a[-1], a0)

        # Also validate that the mappings outer_inp_from_outer_out and
        # outer_inp_from_inner_inp produce the correct results
        scan_node = a.owner.inputs[0].owner

        result = scan_node.op.var_mappings['outer_inp_from_outer_out']
        expected_result = {0: 1, 1: 2}
        assert(result == expected_result)

        result = scan_node.op.var_mappings['outer_inp_from_inner_inp']
        expected_result = {0: 1, 1: 1, 2: 2, 3: 2}
        assert(result == expected_result)

    def test_connection_pattern2(self):
        # This tests for a crash in connection_pattern() when a scan node
        # has more than one mitmot (multiple input taps as well as
        # multiple output taps) output

        x = tensor.matrix()
        seq = tensor.vector()

        def inner_fct(seq, state_old, state_current):
            state_next = state_old * 2 + state_current + seq
            return state_next

        out, _ = theano.scan(inner_fct, sequences=seq,
                            outputs_info={'initial':x, 'taps':[-2,-1]})

        g_out = theano.grad(out.sum(), [seq, x])

        scan_node = g_out[0].owner.inputs[1].owner.inputs[1].owner.inputs[0].owner
        connection_pattern = scan_node.op.connection_pattern(scan_node)

        # Also validate that the mappings outer_inp_from_outer_out and
        # outer_inp_from_inner_inp produce the correct results
        scan_node = out.owner.inputs[0].owner

        result = scan_node.op.var_mappings['outer_inp_from_outer_out']
        expected_result = {0: 2}
        assert(result == expected_result)

        result = scan_node.op.var_mappings['outer_inp_from_inner_inp']
        expected_result = {0: 1, 1: 2, 2: 2}
        assert(result == expected_result)

    def test_grad_grad_mitsot_sitsot(self):
        # Test for an index error when taking the second derivative
        # through a Scan node with one sitsot and one mitsot.

        def inner_fct(mitsot_m2, mitsot_m1, sitsot):
            total = mitsot_m2 + mitsot_m1 + sitsot
            output = total ** 1.05
            return output, output

        inputs = [tensor.matrix(), tensor.vector()]
        outputs_info = [dict(initial=inputs[0], taps=[-2, -1]), inputs[1]]

        scan_outputs, updates = theano.scan(fn=inner_fct,
                                            outputs_info=outputs_info,
                                            n_steps=5)

        # Take the gradient of each output wrt its corresponding initial state
        gradients = [theano.grad(scan_outputs[0].sum(), inputs[0]),
                     theano.grad(scan_outputs[1].sum(), inputs[1])]

        # Take the gradient of the sum of gradients wrt the inputs
        sum_of_grads = sum([g.sum() for g in gradients])
        second_gradients = theano.grad(sum_of_grads, inputs[0])

    def test_verify_second_grad_sitsot(self):

        def get_sum_of_grad(inp):

            scan_outputs, updates = theano.scan(fn=lambda x: x * 2,
                                                outputs_info=[inp],
                                                n_steps=5)

            # Take the gradient of each output wrt its corresponding initial
            # state
            return theano.grad(scan_outputs.sum(), inp).sum()

        # Call verify_grad to ensure the correctness of the second gradients
        floatX = theano.config.floatX
        inputs_test_values = [numpy.random.random((3)).astype(floatX)]
        theano.tests.unittest_tools.verify_grad(get_sum_of_grad,
                                                inputs_test_values)

    def test_verify_second_grad_mitsot1(self):

        def inner_fct(mitsot_m2, sitsot):
            total = mitsot_m2 + sitsot
            output = total ** 1.02
            return output, output

        def get_sum_of_grad(input0, input1):
            outputs_info = [dict(initial=input0, taps=[-2]), input1]

            scan_outputs, updates = theano.scan(fn=inner_fct,
                                                outputs_info=outputs_info,
                                                n_steps=3)

            # Take the gradient of each output wrt its corresponding initial
            # state
            gradients = [theano.grad(scan_outputs[0].sum(), input0),
                         theano.grad(scan_outputs[1].sum(), input1)]

            return gradients[0].sum() + gradients[1].sum()

        # Call verify_grad to ensure the correctness of the second gradients
        floatX = theano.config.floatX
        inputs_test_values = [numpy.random.random((2, 3)).astype(floatX),
                              numpy.random.random((3)).astype(floatX)]
        theano.tests.unittest_tools.verify_grad(get_sum_of_grad,
                                                inputs_test_values)

    def test_grad_two_scans(self):

        # data input & output
        x = tensor.tensor3('x')
        t = tensor.imatrix('t')

        # forward pass
        W = theano.shared(
            numpy.random.randn(2, 2).astype('float32'),
            name="W", borrow=True)

        def forward_scanner(x_t):
            a2_t = tensor.dot(x_t, W)
            y_t = tensor.nnet.softmax_graph(a2_t)
            return y_t

        y, _ = theano.scan(fn=forward_scanner, sequences=x,
                           outputs_info=[None])

        # loss function
        def error_scanner(y_t, t_t):
            return tensor.mean(tensor.nnet.categorical_crossentropy(y_t, t_t))

        L, _ = theano.scan(fn=error_scanner, sequences=[y, t],
                           outputs_info=[None])
        L = tensor.mean(L)

        # backward pass
        gW = tensor.grad(L, [W])

    # simple rnn, one input, one state, weights for each; input/state are
    # vectors, weights are scalars; using shared variables and past
    # taps (sequences and outputs)
    def test_using_taps_input_output(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW = asarrayX(rng.uniform())
        vW_in = asarrayX(rng.uniform())
        vu = asarrayX(rng.uniform(size=(4,), low=-5., high=5.))
        vx0 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))

        u = theano.tensor.vector('u')
        x0 = theano.tensor.vector('x0')
        W_in = theano.shared(vW_in, name='w_in')
        W = theano.shared(vW, name='w')

        def f_rnn_shared(u_tm2, x_tm1, x_tm2):
            return u_tm2 * W_in + x_tm1 * W + x_tm2

        outputs, updates = theano.scan(f_rnn_shared,
                                       dict(input=u, taps=-2),
                                       dict(initial=x0, taps=[-1, -2]),
                                       [],
                                       n_steps=None,
                                       truncate_gradient=-1,
                                       go_backwards=False)

        f7 = theano.function([u, x0],
                             outputs,
                             updates=updates,
                             allow_input_downcast=True)
        theano_out = f7(vu, vx0)

        # compute output in numpy
        # a bit of explaining:
        # due to the definition of sequences taps in scan, v_0[0] is
        # actually v_0[-2], and v_0[1] is v_0[-1]. The values v_0[2]
        # and v_0[3] do not get uesd ( because you do not use v_0[t]
        # in scan) which might seem strange, but then again why not use
        # v_0[t] instead of v_0[t-2] in a real application ??
        # also vx0[0] corresponds to vx0[-2], vx0[1] to vx0[-1]
        numpy_out = numpy.zeros((2,))
        numpy_out[0] = vu[0] * vW_in + vx0[1] * vW + vx0[0]
        numpy_out[1] = vu[1] * vW_in + numpy_out[0] * vW + vx0[1]
        utt.assert_allclose(numpy_out, theano_out)

    # simple rnn, one input, one state, weights for each; input/state are
    # vectors, weights are scalars; using shared variables and past
    # taps (sequences and outputs) and future taps for sequences
    def test_past_future_taps_shared(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW = asarrayX(rng.uniform())
        vW_in = asarrayX(rng.uniform())
        vu = asarrayX(rng.uniform(size=(6,), low=-5., high=5.))
        vx0 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))

        u = theano.tensor.vector('u')
        x0 = theano.tensor.vector('x0')
        W_in = theano.shared(vW_in, name='w_in')
        W = theano.shared(vW, name='w')

        def f_rnn_shared(u_tm2, u_tp2, x_tm1, x_tm2):
            return (u_tm2 + u_tp2) * W_in + x_tm1 * W + x_tm2

        output, updates = theano.scan(f_rnn_shared,
                                      dict(input=u, taps=[-2, 2]),
                                      dict(initial=x0, taps=[-1, -2]),
                                      [],
                                      n_steps=None,
                                      truncate_gradient=-1,
                                      go_backwards=False)

        f8 = theano.function([u, x0],
                             output,
                             updates=updates,
                             allow_input_downcast=True)
        theano_out = f8(vu, vx0)
        # compute output in numpy
        numpy_out = numpy.zeros(2)
        # think of vu[0] as vu[-2], vu[4] as vu[2]
        # and vx0[0] as vx0[-2], vx0[1] as vx0[-1]
        numpy_out[0] = (vu[0] + vu[4]) * vW_in + vx0[1] * vW + vx0[0]
        numpy_out[1] = (vu[1] + vu[5]) * vW_in + numpy_out[0] * vW + vx0[1]
        utt.assert_allclose(numpy_out, theano_out)

    # simple rnn ; compute inplace version 1
    def test_inplace1(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW = asarrayX(numpy.random.uniform())
        vW_in = asarrayX(numpy.random.uniform())
        vu0 = asarrayX(rng.uniform(size=(3,), low=-5., high=5.))
        vu1 = asarrayX(rng.uniform(size=(3,), low=-5., high=5.))
        vu2 = asarrayX(rng.uniform(size=(3,), low=-5., high=5.))
        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())

        u0 = theano.tensor.vector('u0')
        u1 = theano.tensor.vector('u1')
        u2 = theano.tensor.vector('u2')
        mu0 = theano.In(u0, mutable=False)
        mu1 = theano.In(u1, mutable=True)
        mu2 = theano.In(u2, mutable=True)
        x0 = theano.tensor.scalar('x0')
        x1 = theano.tensor.scalar('y0')
        W_in = theano.shared(vW_in, 'Win')
        W = theano.shared(vW, 'W')
        mode = theano.compile.mode.get_mode(None).including('inplace')

        def f_rnn_shared(u0_t, u1_t, u2_t, x0_tm1, x1_tm1):
            return [u0_t * W_in + x0_tm1 * W + u1_t * u2_t,
                    u0_t * W_in + x1_tm1 * W + u1_t + u2_t]

        outputs, updates = theano.scan(f_rnn_shared,
                                       [u0, u1, u2],
                                       [dict(initial=x0, inplace=u2),
                                        dict(initial=x1, inplace=u1)],
                                       [],
                                       n_steps=None,
                                       truncate_gradient=-1,
                                       go_backwards=False,
                                       mode=mode)

        f9 = theano.function([mu0, mu1, mu2, x0, x1],
                             outputs,
                             updates=updates,
                             mode=mode,
                             allow_input_downcast=True)
        scan_node = [x for x in f9.maker.fgraph.toposort()
                     if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        assert 0 in scan_node[0].op.destroy_map.keys()
        assert 1 in scan_node[0].op.destroy_map.keys()
        # compute output in numpy
        numpy_x0 = numpy.zeros((3,))
        numpy_x1 = numpy.zeros((3,))
        numpy_x0[0] = vu0[0] * vW_in + vx0 * vW + vu1[0] * vu2[0]
        numpy_x1[0] = vu0[0] * vW_in + vx1 * vW + vu1[0] + vu2[0]
        for i in xrange(1, 3):
            numpy_x0[i] = (vu0[i] * vW_in + numpy_x0[i - 1] * vW +
                           vu1[i] * vu2[i])
            numpy_x1[i] = (vu0[i] * vW_in + numpy_x1[i - 1] * vW +
                           vu1[i] + vu2[i])

        # note theano computes inplace, so call function after numpy
        # equivalent is done
        (theano_x0, theano_x1) = f9(vu0, vu1, vu2, vx0, vx1)
        # assert that theano does what it should
        utt.assert_allclose(theano_x0, numpy_x0)
        utt.assert_allclose(theano_x1, numpy_x1)

    # simple rnn ; compute inplace version 2
    def test_inplace2(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW = asarrayX(numpy.random.uniform())
        vW_in = asarrayX(numpy.random.uniform())
        vu0 = asarrayX(rng.uniform(size=(3,), low=-5., high=5.))
        vu1 = asarrayX(rng.uniform(size=(4,), low=-5., high=5.))
        vu2 = asarrayX(rng.uniform(size=(5,), low=-5., high=5.))
        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())

        u0 = theano.tensor.vector('u0')
        u1 = theano.tensor.vector('u1')
        u2 = theano.tensor.vector('u2')
        mu0 = theano.In(u0, mutable=True)
        mu1 = theano.In(u1, mutable=True)
        mu2 = theano.In(u2, mutable=True)
        x0 = theano.tensor.scalar('x0')
        x1 = theano.tensor.scalar('y0')
        W_in = theano.shared(vW_in, 'Win')
        W = theano.shared(vW, 'W')
        mode = theano.compile.mode.get_mode(None).including('inplace')

        def f_rnn_shared(u0_t,
                         u1_t,
                         u1_tp1,
                         u2_tm1,
                         u2_t,
                         u2_tp1,
                         x0_tm1,
                         x1_tm1):
            return [u0_t * W_in + x0_tm1 * W + u1_t * u1_tp1,
                    u0_t * W_in + x1_tm1 * W + u2_tm1 + u2_t + u2_tp1]

        outputs, updates = theano.scan(f_rnn_shared,
                                       [u0,
                                        dict(input=u1, taps=[0, 1]),
                                        dict(input=u2, taps=[-1, 0, +1])],
                                       [dict(initial=x0), dict(initial=x1)],
                                       [],
                                       n_steps=None,
                                       truncate_gradient=-1,
                                       go_backwards=False,
                                       mode=mode)
        f9 = theano.function([mu0, mu1, mu2, x0, x1],
                             outputs,
                             updates=updates,
                             mode=mode,
                             allow_input_downcast=True)

        scan_node = [x for x in f9.maker.fgraph.toposort()
                     if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        assert 0 in scan_node[0].op.destroy_map.keys()
        assert 1 in scan_node[0].op.destroy_map.keys()
        # compute output in numpy
        numpy_x0 = numpy.zeros((3,))
        numpy_x1 = numpy.zeros((3,))
        numpy_x0[0] = vu0[0] * vW_in + vx0 * vW + vu1[0] * vu1[1]
        numpy_x1[0] = vu0[0] * vW_in + vx1 * vW + vu2[0] + vu2[1] + vu2[2]
        for i in xrange(1, 3):
            numpy_x0[i] = (vu0[i] * vW_in + numpy_x0[i - 1] * vW +
                           vu1[i] * vu1[i + 1])
            numpy_x1[i] = (vu0[i] * vW_in + numpy_x1[i - 1] * vW +
                           vu2[i] + vu2[i + 1] + vu2[i + 2])

        # note theano computes inplace, so call function after numpy
        # equivalent is done
        (theano_x0, theano_x1) = f9(vu0, vu1, vu2, vx0, vx1)
        # assert that theano does what it should
        utt.assert_allclose(theano_x0, numpy_x0)
        utt.assert_allclose(theano_x1, numpy_x1)

    def test_inplace3(self):
        rng = numpy.random.RandomState(utt.fetch_seed())

        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())
        x0 = theano.shared(vx0)
        x1 = theano.shared(vx1)
        outputs, updates = theano.scan(lambda x, y: (x + asarrayX(1),
                                                     y + asarrayX(1)),
                                       [],
                                       [x0, x1],
                                       n_steps=3)
        x0 = asarrayX(numpy.zeros((3,)))
        x0[0] = vx0
        x0 = theano.tensor.constant(x0)
        to_replace = outputs[0].owner.inputs[0].owner.inputs[1]
        outputs = theano.clone(outputs,
                               replace=[(to_replace, x0)])
        mode = theano.compile.mode.get_mode(None).including('inplace')
        f9 = theano.function([],
                             outputs,
                             updates=updates,
                             mode=mode)
        scan_node = [x for x in f9.maker.fgraph.toposort()
                     if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        assert 0 not in scan_node[0].op.destroy_map.keys()
        assert 1 in scan_node[0].op.destroy_map.keys()

    # Shared variable with updates
    def test_shared_arguments_with_updates(self):
        rng = numpy.random.RandomState(utt.fetch_seed())

        vW1 = asarrayX(rng.rand(2, 3))
        vW2 = asarrayX(rng.rand(3, 2))
        vu1 = asarrayX(rng.rand(3, 2))
        vu2 = asarrayX(rng.rand(3, 3))
        vy0 = asarrayX(rng.rand(3, 2))
        vy1 = asarrayX(rng.rand(2))
        vy2 = asarrayX(rng.rand(3))

        # Their is a bug when floatX=float32 when we remove this line.
        # The trace back is:
# Traceback (most recent call last):
#  File "/u/bastienf/repos/Theano/theano/tests/test_scan.py", line 434, in test_shared_arguments_with_updates
#    theano_y0,theano_y1,theano_y2 = f10(vu2, vy0)
#  File "/u/bastienf/repos/theano/compile/function_module.py", line 480, in __call__
#    self.fn()
#  File "/u/bastienf/repos/theano/compile/profilemode.py", line 59, in profile_f
#    raise_with_op(node)
#  File "/u/bastienf/repos/theano/compile/profilemode.py", line 52, in profile_f
#    th()
#  File "/u/bastienf/repos/theano/gof/cc.py", line 1141, in <lambda>
#    thunk = lambda p = p, i = node_input_storage, o = node_output_storage, n = node: p(n, [x[0] for x in i], o)
#  File "/u/bastienf/repos/theano/scan.py", line 922, in perform
#    inplace_map)
#  File "/u/bastienf/repos/theano/scan.py", line 1054, in scan
#    something = fn(*fn_args)
#  File "/u/bastienf/repos/theano/compile/function_module.py", line 458, in __call__
#    s.storage[0] = s.type.filter(arg, strict=s.strict)
#  File "/u/bastienf/repos/theano/tensor/basic.py", line 415, in filter
#    data = theano._asarray(data, dtype = self.dtype) #TODO - consider to pad shape with ones
#  File "/u/bastienf/repos/theano/misc/safe_asarray.py", line 30, in _asarray
#    rval = numpy.asarray(a, dtype=dtype, order=order)
#  File "/u/lisa/local/byhost/ceylon.iro.umontreal.ca//lib64/python2.5/site-packages/numpy/core/numeric.py", line 230, in asarray
#    return array(a, dtype, copy=False, order=order)
# TypeError: ('__array__() takes no arguments (1 given)', <theano.scan.Scan object at 0x3dbbf90>(?_steps, u1, u2, y0, y1, 0.0, W1, W2), 'Sequence id of Apply node=0')
#
#  This don't seam to be a theano related bug...
        vu1 = asarrayX(rng.rand(3, 2))

        W1 = theano.shared(vW1, 'W1')
        W2 = theano.shared(vW2, 'W2')
        u1 = theano.shared(vu1, 'u1')
        y1 = theano.shared(vy1, 'y1')

        def f(u1_t, u2_t, y0_tm3, y0_tm2, y0_tm1, y1_tm1):
            y0_t = (theano.dot(theano.dot(u1_t, W1), W2) + 0.1 * y0_tm1 +
                    0.33 * y0_tm2 + 0.17 * y0_tm3)
            y1_t = theano.dot(u2_t, W2) + y1_tm1
            y2_t = theano.dot(u1_t, W1)
            nwW1 = W1 + .1
            nwW2 = W2 + .05
            # return outputs followed by a list of updates
            return ([y0_t, y1_t, y2_t], [(W1, nwW1), (W2, nwW2)])

        u2 = theano.tensor.matrix('u2')
        y0 = theano.tensor.matrix('y0')
        outputs, updates = theano.scan(f,
                                       [u1, u2],
                                       [dict(initial=y0, taps=[-3, -2, -1]),
                                        y1,
                                        None],
                                       [],
                                       n_steps=None,
                                       go_backwards=False,
                                       truncate_gradient=-1)

        f10 = theano.function([u2, y0],
                              outputs,
                              updates=updates,
                              allow_input_downcast=True)
        allstuff = f10(vu2, vy0)
        theano_y0, theano_y1, theano_y2 = allstuff

        # do things in numpy
        numpy_y0 = numpy.zeros((6, 2))
        numpy_y1 = numpy.zeros((4, 2))
        numpy_y2 = numpy.zeros((3, 3))
        numpy_y0[:3] = vy0
        numpy_y1[0] = vy1
        numpy_W1 = vW1.copy()
        numpy_W2 = vW2.copy()
        for idx in xrange(3):
            numpy_y0[idx + 3] = numpy.dot(numpy.dot(vu1[idx, :], numpy_W1),
                                          numpy_W2) + \
                                0.1 * numpy_y0[idx + 2] + \
                                0.33 * numpy_y0[idx + 1] + \
                                0.17 * numpy_y0[idx]
            numpy_y1[idx + 1] = (numpy.dot(vu2[idx, :], numpy_W2) +
                                 numpy_y1[idx])
            numpy_y2[idx] = numpy.dot(vu1[idx, :], numpy_W1)
            numpy_W1 = numpy_W1 + .1
            numpy_W2 = numpy_W2 + .05

        utt.assert_allclose(theano_y0, numpy_y0[3:])
        utt.assert_allclose(theano_y1, numpy_y1[1:])
        utt.assert_allclose(theano_y2, numpy_y2)
        utt.assert_allclose(W1.get_value(), numpy_W1)
        utt.assert_allclose(W2.get_value(), numpy_W2)

    def test_grad_dtype_change(self):
        x = tensor.fscalar('x')
        y = tensor.fscalar('y')
        c = tensor.iscalar('c')

        def inner_fn(cond, x, y):
            new_cond = tensor.cast(tensor.switch(cond, x, y), 'int32')
            new_x = tensor.switch(cond, tensor.nnet.sigmoid(y * x), x)
            new_y = tensor.switch(cond, y, tensor.nnet.sigmoid(x))
            return new_cond, new_x, new_y

        values, _ = theano.scan(
            inner_fn,
            outputs_info=[c, x, y],
            n_steps=10,
            truncate_gradient=-1,
            go_backwards=False)
        gX, gY = tensor.grad(values[1].sum(), [x, y])
        f = theano.function([c, x, y], [gX, gY],
                            allow_input_downcast=True)
        # Check for runtime errors
        f(numpy.int32(0), numpy.float32(1.), numpy.float32(.5))

    def test_simple_shared_mrg_random(self):
        theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(utt.fetch_seed())

        values, updates = theano.scan(lambda: theano_rng.uniform((2,), -1, 1),
                                      [],
                                      [],
                                      [],
                                      n_steps=5,
                                      truncate_gradient=-1,
                                      go_backwards=False)
        my_f = theano.function([],
                               values,
                               updates=updates,
                               allow_input_downcast=True)

        # Just check for run-time errors
        theano_v = my_f()
        theano_v = my_f()

    def test_simple_shared_random(self):
        theano_rng = theano.tensor.shared_randomstreams.RandomStreams(
            utt.fetch_seed())

        values, updates = theano.scan(lambda: theano_rng.uniform((2,), -1, 1),
                                      [],
                                      [],
                                      [],
                                      n_steps=5,
                                      truncate_gradient=-1,
                                      go_backwards=False)
        my_f = theano.function([],
                               values,
                               updates=updates,
                               allow_input_downcast=True)

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2 ** 30)
        rng = numpy.random.RandomState(int(rng_seed))  # int() is for 32bit

        numpy_v = numpy.zeros((10, 2))
        for i in xrange(10):
            numpy_v[i] = rng.uniform(-1, 1, size=(2,))

        theano_v = my_f()
        utt.assert_allclose(theano_v, numpy_v[:5, :])
        theano_v = my_f()
        utt.assert_allclose(theano_v, numpy_v[5:, :])

    def test_gibbs_chain(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_W = numpy.array(rng.rand(20, 30) - .5, dtype='float32')
        v_vsample = numpy.array(rng.binomial(1, .5, size=(3, 20),),
                                dtype='float32')
        v_bvis = numpy.array(rng.rand(20) - .5, dtype='float32')
        v_bhid = numpy.array(rng.rand(30) - .5, dtype='float32')
        W = theano.shared(v_W, 'vW')
        bhid = theano.shared(v_bhid, 'vbhid')
        bvis = theano.shared(v_bvis, 'vbvis')
        vsample = theano.tensor.matrix(dtype='float32')
        trng = theano.tensor.shared_randomstreams.RandomStreams(
            utt.fetch_seed())

        def f(vsample_tm1):
            hmean_t = theano.tensor.nnet.sigmoid(
                theano.dot(vsample_tm1, W) + bhid)
            hsample_t = theano.tensor.cast(
                trng.binomial(hmean_t.shape, 1, hmean_t),
                dtype='float32')
            vmean_t = theano.tensor.nnet.sigmoid(
                theano.dot(hsample_t, W.T) + bvis)
            return theano.tensor.cast(
                trng.binomial(vmean_t.shape, 1, vmean_t),
                dtype='float32')

        theano_vsamples, updates = theano.scan(f,
                                               [],
                                               vsample,
                                               [],
                                               n_steps=10,
                                               truncate_gradient=-1,
                                               go_backwards=False)

        my_f = theano.function([vsample], theano_vsamples[-1],
                               updates=updates,
                               allow_input_downcast=True)

        _rng = numpy.random.RandomState(utt.fetch_seed())
        rng_seed = _rng.randint(2 ** 30)
        nrng1 = numpy.random.RandomState(int(rng_seed))  # int() is for 32bit

        rng_seed = _rng.randint(2 ** 30)
        nrng2 = numpy.random.RandomState(int(rng_seed))  # int() is for 32bit

        def numpy_implementation(vsample):
            for idx in range(10):
                hmean = 1. / (1. + numpy.exp(-(numpy.dot(vsample, v_W) +\
                        v_bhid)))
                hsample = numpy.array(nrng1.binomial(1,
                                                     hmean,
                                                     size=hmean.shape),
                                      dtype='float32')
                vmean = 1. / (1. + numpy.exp(-(numpy.dot(hsample, v_W.T) +\
                        v_bvis)))
                vsample = numpy.array(nrng2.binomial(1,
                                                     vmean,
                                                     size=vmean.shape),
                                      dtype='float32')

            return vsample

        t_result = my_f(v_vsample)
        n_result = numpy_implementation(v_vsample)
        utt.assert_allclose(t_result, n_result)

    def test_only_shared_no_input_no_output(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_state = asarrayX(rng.uniform())
        state = theano.shared(v_state, 'vstate')

        def f_2():
            return OrderedDict([(state, 2 * state)])
        n_steps = theano.tensor.iscalar('nstep')
        output, updates = theano.scan(f_2,
                                      [],
                                      [],
                                      [],
                                      n_steps=n_steps,
                                      truncate_gradient=-1,
                                      go_backwards=False)
        this_f = theano.function([n_steps],
                                 output,
                                 updates=updates,
                                allow_input_downcast=True)
        n_steps = 3
        this_f(n_steps)
        numpy_state = v_state * (2 ** (n_steps))
        utt.assert_allclose(state.get_value(), numpy_state)

    def test_map_functionality(self):
        def f_rnn(u_t):
            return u_t + 3

        u = theano.tensor.vector('u')

        outputs, updates = theano.scan(f_rnn,
                                       u,
                                       [],
                                       [],
                                       n_steps=None,
                                       truncate_gradient=-1,
                                       go_backwards=False)

        f2 = theano.function([u],
                             outputs,
                             updates=updates,
                             allow_input_downcast=True)
        rng = numpy.random.RandomState(utt.fetch_seed())

        v_u = rng.uniform(size=(5,), low=-5., high=5.)
        numpy_result = v_u + 3
        theano_result = f2(v_u)
        utt.assert_allclose(theano_result, numpy_result)

    def test_map(self):
        v = theano.tensor.vector('v')
        abs_expr, abs_updates = theano.map(
            lambda x: abs(x),
            v,
            [],
            truncate_gradient=-1,
            go_backwards=False)

        f = theano.function([v],
                            abs_expr,
                            updates=abs_updates,
                            allow_input_downcast=True)

        rng = numpy.random.RandomState(utt.fetch_seed())
        vals = rng.uniform(size=(10,), low=-5., high=5.)
        abs_vals = abs(vals)
        theano_vals = f(vals)
        utt.assert_allclose(abs_vals, theano_vals)

    def test_backwards(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.vector('u')
        x0 = theano.tensor.scalar('x0')
        W_in = theano.tensor.scalar('win')
        W = theano.tensor.scalar('w')

        output, updates = theano.scan(f_rnn,
                                      u,
                                      x0,
                                      [W_in, W],
                                      n_steps=None,
                                      truncate_gradient=-1,
                                      go_backwards=True)

        f2 = theano.function([u, x0, W_in, W],
                             output,
                             updates=updates,
                             allow_input_downcast=True)
        # get random initial values
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(4,), low=-5., high=5.)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        v_out = numpy.zeros((4,))
        v_out[0] = v_u[3] * W_in + v_x0 * W
        for step in xrange(1, 4):
            v_out[step] = v_u[3 - step] * W_in + v_out[step - 1] * W

        theano_values = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(theano_values, v_out)

    def test_reduce(self):
        v = theano.tensor.vector('v')
        s = theano.tensor.scalar('s')
        result, updates = theano.reduce(lambda x, y: x + y, v, s)

        f = theano.function([v, s],
                            result,
                            updates=updates,
                            allow_input_downcast=True)
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_v = rng.uniform(size=(5,), low=-5., high=5.)
        assert abs(numpy.sum(v_v) - f(v_v, 0.)) < 1e-3

    def test_grad_one_output(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.vector('u')
        x0 = theano.tensor.scalar('x0')
        W_in = theano.tensor.scalar('W_in')
        W = theano.tensor.scalar('W')

        cost, updates = scan_project_sum(f_rnn,
                                         u,
                                         x0,
                                         [W_in, W],
                                         n_steps=None,
                                         truncate_gradient=-1,
                                         go_backwards=False)
        gu, gx0, gW_in, gW = theano.tensor.grad(cost,
                                                [u, x0, W_in, W])
        grad_fn = theano.function(
            [u, x0, W_in, W],
            [gu, gx0, gW_in, gW],
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True)
        cost_fn = theano.function(
            [u, x0, W_in, W],
            cost,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True)

        # get random initial values
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_u = numpy.array(rng.uniform(size=(10,), low=-.5, high=.5),
                          dtype=theano.config.floatX)
        v_x0 = numpy.array(rng.uniform(), dtype=theano.config.floatX)
        W = numpy.array(rng.uniform(), dtype=theano.config.floatX)
        W_in = numpy.array(rng.uniform(), dtype=theano.config.floatX)
        analytic_grad = grad_fn(v_u, v_x0, W_in, W)

        num_grad = multiple_outputs_numeric_grad(
            cost_fn, [v_u, v_x0, W_in, W])
        max_err, max_err_pos = num_grad.max_err(analytic_grad)

        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                            (max_err, 1e-2, max_err_pos,
                             analytic_grad[max_err_pos],
                             num_grad.gx[max_err_pos]))

    def test_grad_multiple_outs(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size=(2,), low=-.1, high=.1))
        vW = asarrayX(rng.uniform(size=(2, 2), low=-.1, high=.1))
        vWout = asarrayX(rng.uniform(size=(2,), low=-.1, high=.1))
        vW_in1 = asarrayX(rng.uniform(size=(2, 2), low=-.1, high=.1))
        v_u1 = asarrayX(rng.uniform(size=(7, 2), low=-.1, high=.1))
        v_u2 = asarrayX(rng.uniform(size=(7,), low=-.1, high=.1))
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-.1, high=.1))
        v_y0 = asarrayX(rng.uniform())

        W_in2 = theano.shared(vW_in2, name='win2')
        W = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name='wout')
        W_in1 = theano.tensor.matrix('win')
        u1 = theano.tensor.matrix('u1')
        u2 = theano.tensor.vector('u2')
        x0 = theano.tensor.vector('x0')
        y0 = theano.tensor.scalar('y0')

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, W_in1):
            return [theano.dot(u1_t, W_in1) + u2_t * W_in2 + \
                    theano.dot(x_tm1, W), theano.dot(x_tm1, W_out)]

        cost, updates = scan_project_sum(f_rnn_cmpl,
                                         [u1, u2],
                                         [x0, y0],
                                         W_in1,
                                         n_steps=None,
                                         truncate_gradient=-1,
                                         go_backwards=False)
        vparams = [v_u1, v_u2, v_x0, v_y0, vW_in1]
        # y0 is actually not used in the computation of the cost
        params = [u1, u2, x0, y0, W_in1]
        gparams = theano.grad(cost, params,
                                     disconnected_inputs='ignore')

        grad_fn = theano.function([u1, u2, x0, y0, W_in1],
                                  gparams,
                                  updates=updates,
                                  no_default_updates=True,
                                 allow_input_downcast=True)
        cost_fn = theano.function([u1, u2, x0, y0, W_in1],
                                  cost,
                                  updates=updates,
                                  no_default_updates=True,
                                 allow_input_downcast=True)

        num_grad = multiple_outputs_numeric_grad(cost_fn,
                                                 [v_u1,
                                                  v_u2,
                                                  v_x0,
                                                  v_y0,
                                                  vW_in1])
        analytic_grad = grad_fn(v_u1, v_u2, v_x0, v_y0, vW_in1)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)

        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                            (max_err, 1e-2, max_err_pos,
                             analytic_grad[max_err_pos],
                             num_grad.gx[max_err_pos]))

    @attr('slow')
    def test_grad_multiple_outs_taps(self):
        l = 5
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size=(2,), low=-.2, high=.2))
        vW = asarrayX(rng.uniform(size=(2, 2), low=-.2, high=.2))
        vWout = asarrayX(rng.uniform(size=(2,), low=-.2, high=.2))
        vW_in1 = asarrayX(rng.uniform(size=(2, 2), low=-.2, high=.2))
        v_u1 = asarrayX(rng.uniform(size=(l, 2), low=-.2, high=.2))
        v_u2 = asarrayX(rng.uniform(size=(l + 2, 2), low=-.2, high=.2))
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-.2, high=.2))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = theano.shared(vW_in2, name='win2')
        W = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name='wout')
        W_in1 = theano.tensor.matrix('win')
        u1 = theano.tensor.matrix('u1')
        u2 = theano.tensor.matrix('u2')
        x0 = theano.tensor.vector('x0')
        y0 = theano.tensor.vector('y0')

        W_in1.tag.test_value = vW_in1
        u1.tag.test_value = v_u1
        u2.tag.test_value = v_u2
        x0.tag.test_value = v_x0
        y0.tag.test_value = v_y0

        def f_rnn_cmpl(u1_t,
                       u2_tm1,
                       u2_t,
                       u2_tp1,
                       x_tm1,
                       y_tm1,
                       y_tm3,
                       W_in1):
            return [theano.dot(u1_t, W_in1) +
                    (u2_t + u2_tm1 * u2_tp1) * W_in2 +
                    theano.dot(x_tm1, W),
                    (y_tm1 + y_tm3) * theano.dot(x_tm1, W_out),
                    theano.dot(u1_t, W_in1)]

        # We change the compute_test_value[_opt] flag to run the
        # assert in Scan.grad() of the new scan input sequence related
        # to outer_mitsot_outs, outer_sitsot_outs and
        # outer_nitsot_outs. This allow to test an old Scan bug.
        old1 = theano.config.compute_test_value
        old2 = theano.config.compute_test_value_opt
        theano.config.compute_test_value = 'raise'
        theano.config.compute_test_value_opt = 'raise'
        try:
            cost, updates = scan_project_sum(
                f_rnn_cmpl,
                [u1, dict(input=u2, taps=[-1, 0, 1])],
                [x0, dict(initial=y0, taps=[-1, -3]), None],
                W_in1,
                n_steps=None,
                truncate_gradient=-1,
                go_backwards=False)
            vparams = [v_u1, v_u2, v_x0, v_y0, vW_in1]
            params = [u1, u2, x0, y0, W_in1]
            gparams = theano.tensor.grad(cost, params)
            print(".", file=sys.stderr)
            cost_fn = theano.function([u1, u2, x0, y0, W_in1],
                                      cost,
                                      updates=updates,
                                      no_default_updates=True,
                                      allow_input_downcast=True)
            print(".", file=sys.stderr)
            grad_fn = theano.function([u1, u2, x0, y0, W_in1],
                                      gparams,
                                      updates=updates,
                                      no_default_updates=True,
                                      allow_input_downcast=True)
            print(".", file=sys.stderr)
        finally:
            theano.config.compute_test_value = old1
            theano.config.compute_test_value_opt = old2

        num_grad = multiple_outputs_numeric_grad(cost_fn,
                                                 [v_u1,
                                                  v_u2,
                                                  v_x0,
                                                  v_y0,
                                                  vW_in1])

        analytic_grad = grad_fn(v_u1, v_u2, v_x0, v_y0, vW_in1)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                            (max_err, 1e-2, max_err_pos,
                             analytic_grad[max_err_pos],
                             num_grad.gx[max_err_pos]))

    @attr('slow')
    def test_grad_multiple_outs_taps_backwards(self):
        l = 5
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size=(2,), low=-.2, high=.2))
        vW = asarrayX(rng.uniform(size=(2, 2), low=-.2, high=.2))
        vWout = asarrayX(rng.uniform(size=(2,), low=-.2, high=.2))
        vW_in1 = asarrayX(rng.uniform(size=(2, 2), low=-.2, high=.2))
        v_u1 = asarrayX(rng.uniform(size=(l, 2), low=-.2, high=.2))
        v_u2 = asarrayX(rng.uniform(size=(l + 2, 2), low=-.2, high=.2))
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-.2, high=.2))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = theano.shared(vW_in2, name='win2')
        W = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name='wout')
        W_in1 = theano.tensor.matrix('win')
        u1 = theano.tensor.matrix('u1')
        u2 = theano.tensor.matrix('u2')
        x0 = theano.tensor.vector('x0')
        y0 = theano.tensor.vector('y0')

        def f_rnn_cmpl(u1_t,
                       u2_tm1,
                       u2_t,
                       u2_tp1,
                       x_tm1,
                       y_tm1,
                       y_tm3,
                       W_in1):
            return [theano.dot(u1_t, W_in1) + \
                        (u2_t + u2_tm1 * u2_tp1) * W_in2 + \
                        theano.dot(x_tm1, W),
                    (y_tm1 + y_tm3) * theano.dot(x_tm1, W_out)]
        cost, updates = scan_project_sum(f_rnn_cmpl,
                                         [u1, dict(input=u2, taps=[-1, 0, 1])],
                                         [x0, dict(initial=y0, taps=[-1, -3])],
                                         W_in1,
                                         n_steps=None,
                                         truncate_gradient=-1,
                                         go_backwards=True)
        vparams = [v_u1, v_u2, v_x0, v_y0, vW_in1]
        params = [u1, u2, x0, y0, W_in1]
        gparams = theano.tensor.grad(cost, params)
        grad_fn = theano.function([u1, u2, x0, y0, W_in1],
                                  gparams,
                                  updates=updates,
                                  no_default_updates=True,
                                 allow_input_downcast=True)
        cost_fn = theano.function([u1, u2, x0, y0, W_in1],
                                  cost,
                                  updates=updates,
                                  no_default_updates=True,
                                  allow_input_downcast=True)

        num_grad = multiple_outputs_numeric_grad(cost_fn, [v_u1,
                                                           v_u2,
                                                           v_x0,
                                                           v_y0,
                                                           vW_in1])

        analytic_grad = grad_fn(v_u1, v_u2, v_x0, v_y0, vW_in1)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                            (max_err, 1e-2, max_err_pos,
                             analytic_grad[max_err_pos],
                             num_grad.gx[max_err_pos]))

    def test_grad_multiple_outs_some_uncomputable(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in = asarrayX(rng.uniform(size=(2, 2), low=-3., high=3.))
        v_u = asarrayX(rng.uniform(size=(5, 2), low=-3., high=3.))
        v_u2 = numpy.array([1, 3, 4, 6, 8], dtype='int32')
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-3., high=3.))

        W_in = theano.tensor.matrix('win')
        u = theano.tensor.matrix('u1')
        u2 = theano.tensor.ivector('u2')
        x0 = theano.tensor.vector('x0', dtype=theano.config.floatX)
        # trng  = theano.tensor.shared_randomstreams.RandomStreams(
        #                                               utt.fetch_seed())

        def f_rnn_cmpl(u_t, u2_t, x_tm1, W_in):
            trng1 = theano.tensor.shared_randomstreams.RandomStreams(123)
            x_t = theano.tensor.cast(u2_t, theano.config.floatX) +\
                    theano.dot(u_t, W_in) + x_tm1 + \
                    trng1.uniform(low=-1.1, high=1.1,
                                  dtype=theano.config.floatX)
            return x_t, 2 * u2_t

        cost, updates = scan_project_sum(f_rnn_cmpl,
                                         [u, u2],
                                         [x0, None],
                                         W_in,
                                         n_steps=None,
                                         truncate_gradient=-1,
                                         go_backwards=False)
        vparams = [v_u, v_u2, v_x0, vW_in]
        params = [u, u2, x0, W_in]
        gparams = theano.tensor.grad(cost, params)
        grad_fn = theano.function([u, u2, x0, W_in],
                                  gparams,
                                  updates=updates,
                                  no_default_updates=True,
                                  allow_input_downcast=True)
        cost_fn = theano.function([u, u2, x0, W_in],
                                  cost,
                                  updates=updates,
                                  no_default_updates=True,
                                  allow_input_downcast=True)

        def reset_rng_fn(fn, *args):
            for idx, arg in enumerate(fn.maker.expanded_inputs):
                if (arg.value and type(arg.value.data) == \
                    type(numpy.random.RandomState(123))):
                    obj = fn.maker.expanded_inputs[idx].value
                    obj.data = numpy.random.RandomState(123)
                    fn.maker.expanded_inputs[idx].value = obj
            return fn(*args)

        reset_rng_cost_fn = lambda *args: reset_rng_fn(cost_fn, *args)
        reset_rng_grad_fn = lambda *args: reset_rng_fn(grad_fn, *args)
        num_grad = multiple_outputs_numeric_grad(
            reset_rng_cost_fn,
            [v_u, v_u2, v_x0, vW_in],
            ndarray_mask=[True, False, True, True])
        analytic_grad = reset_rng_grad_fn(v_u, v_u2, v_x0, vW_in)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)

        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                            (max_err, 1e-2, max_err_pos,
                             analytic_grad[max_err_pos],
                             num_grad.gx[max_err_pos]))

        # Also validate that the mappings outer_inp_from_outer_out and
        # outer_inp_from_inner_inp produce the correct results
        scan_node = list(updates.values())[0].owner

        result = scan_node.op.var_mappings['outer_inp_from_outer_out']
        expected_result = {0: 3, 1: 5, 2: 4}
        assert(result == expected_result)

        result = scan_node.op.var_mappings['outer_inp_from_inner_inp']
        expected_result = {0: 1, 1: 2, 2: 3, 3: 4, 4: 6}
        assert(result == expected_result)

    def test_grad_multiple_outs_some_truncate(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in = asarrayX(rng.uniform(size=(2, 2), low=-.1, high=.1))
        v_u = asarrayX(rng.uniform(size=(5, 2), low=-.1, high=.1))
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-.1, high=.1))

        W_in = theano.tensor.matrix('win')
        u = theano.tensor.matrix('u1')
        x0 = theano.tensor.vector('x0')
        # trng  = theano.tensor.shared_randomstreams.RandomStreams(
        #                                               utt.fetch_seed())

        def f_rnn_cmpl(u_t, x_tm1, W_in):
            trng1 = theano.tensor.shared_randomstreams.RandomStreams(123)
            rnd_nb = trng1.uniform(low=-.1, high=.1)
            x_t = theano.dot(u_t, W_in) + x_tm1 + rnd_nb
            x_t = theano.tensor.cast(x_t, dtype=theano.config.floatX)
            return x_t

        cost, updates = scan_project_sum(f_rnn_cmpl,
                                         u,
                                         x0,
                                         W_in,
                                         n_steps=None,
                                         truncate_gradient=3,
                                         go_backwards=False)
        vparams = [v_u, v_x0, vW_in]
        params = [u, x0, W_in]
        gparams = theano.tensor.grad(cost, params)

        grad_fn = theano.function([u, x0, W_in],
                                  gparams,
                                  updates=updates,
                                  no_default_updates=True,
                                 allow_input_downcast=True)
        cost_fn = theano.function([u, x0, W_in],
                                  cost,
                                  updates=updates,
                                  no_default_updates=True,
                                 allow_input_downcast=True)

        def reset_rng_fn(fn, *args):
            for idx, arg in enumerate(fn.maker.expanded_inputs):
                if (arg.value and
                    isinstance(arg.value.data, numpy.random.RandomState)):
                    obj = fn.maker.expanded_inputs[idx].value
                    obj.data = numpy.random.RandomState(123)
                    fn.maker.expanded_inputs[idx].value = obj
            out = fn(*args)
            return out

        reset_rng_cost_fn = lambda *args: reset_rng_fn(cost_fn, *args)
        reset_rng_grad_fn = lambda *args: reset_rng_fn(grad_fn, *args)
        num_grad = multiple_outputs_numeric_grad(
            reset_rng_cost_fn, [v_u, v_x0, vW_in])
        analytic_grad = reset_rng_grad_fn(v_u, v_x0, vW_in)
        utt.assert_allclose(analytic_grad[0][:2], numpy.zeros((2, 2)))

    def test_grad_multiple_outs_some_disconnected(self):
        final_cost = self._grad_mout_helper(100, mode_nodebug)
        assert final_cost < 0.02

    def test_grad_multiple_outs_some_disconnected_2(self):
        # This is to try the network in DEBUG_MODE, but not fully
        # train it since that would take 3 hours
        self._grad_mout_helper(1, None)

    def _grad_mout_helper(self, n_iters, mode):
        # Created on Tue Oct 07 13:28:51 2014
        # @author: vaneetke
        rng = numpy.random.RandomState(utt.fetch_seed())
        n_hid = 3
        n_in = 1
        n_out = 1

        W_hh_v = asarrayX(rng.uniform(size=(n_hid, n_hid), low=-.01, high=.01))
        h0_v = asarrayX(rng.uniform(size=(2, n_hid), low=-.01, high=.01))
        b_h_v = asarrayX(rng.uniform(size=(n_hid), low=-.01, high=.01))
        W_ih_v = asarrayX(rng.uniform(size=(n_in, n_hid), low=-.01, high=.01))
        W_ho_v = asarrayX(rng.uniform(size=(n_hid, n_out), low=-.01, high=.01))
        b_o_v = asarrayX(rng.uniform(size=(n_out), low=-.01, high=.01))

        # parameters of the rnn
        b_h = theano.shared(b_h_v)
        h0 = theano.shared(h0_v)
        W_ih = theano.shared(W_ih_v)
        W_hh = theano.shared(W_hh_v)
        W_ho = theano.shared(W_ho_v)
        b_o = theano.shared(b_o_v)
        params = [W_ih, W_hh, b_h, W_ho, b_o, h0]

        # first dimension is time
        x = tensor.matrix()

        # sequences: x_t
        # prior results: h_tm2, h_tm1
        # non-sequences: W_ih, W_hh, W_ho, b_h
        def one_step(x_t, h_tm2, h_tm1, W_ih, W_hh, b_h, W_ho, b_o):
            h_t = tensor.tanh(theano.dot(x_t, W_ih)
                              + theano.dot(h_tm2, W_hh) + b_h)
            y_t = theano.dot(h_t, W_ho) + b_o
            return [h_t, y_t]

        # hidden and outputs of the entire sequence
        [h, y], _ = theano.scan(
            fn=one_step,
            sequences=dict(input=x),
            # corresponds to the return type of one_step
            outputs_info=[dict(initial=h0, taps=[-2, -1]), None],
            non_sequences=[W_ih, W_hh, b_h, W_ho, b_o],
            mode=mode)

        # target values
        t = tensor.matrix()

        # learning rate
        lr = asarrayX(0.1)
        learning_rate = theano.shared(lr)

        cost = ((0.5 * ((y - t) ** 2.0).mean())
                + (0.5 * (y.std() - t.std()) ** 2.0))

        gparams = theano.grad(cost, params)
        updates = [(param, param - gparam * learning_rate)
                   for param, gparam in zip(params, gparams)]
        learn_rnn_fn = theano.function(inputs=[x, t],
                                       outputs=cost,
                                       updates=updates,
                                       mode=mode)
        eval_rnn_fn = theano.function(inputs=[x],
                                      outputs=y,
                                      mode=mode)

        # artificial data
        x_v = numpy.arange(0., 10.49, 0.21, dtype=theano.config.floatX)
        x_v = x_v.reshape(len(x_v), 1)
        s_v = numpy.sin(x_v)
        t_v = numpy.roll(s_v, -1)[:-1]
        s_v = s_v[:-1]
        for i in xrange(n_iters):
            cost = learn_rnn_fn(s_v, t_v)
        pred = eval_rnn_fn(s_v)
        return cost

    def test_draw_as_input_to_scan(self):
        trng = theano.tensor.shared_randomstreams.RandomStreams(123)

        x = theano.tensor.matrix('x')
        y = trng.binomial(size=x.shape, p=x)
        z, updates = theano.scan(lambda a: a, non_sequences=y, n_steps=2)

        f = theano.function([x],
                            [y, z],
                            updates=updates,
                            allow_input_downcast=True)

        rng = numpy.random.RandomState(utt.fetch_seed())
        nx = rng.uniform(size=(10, 10))
        ny1, nz1 = f(nx)
        ny2, nz2 = f(nx)

        utt.assert_allclose([ny1, ny1], nz1)
        utt.assert_allclose([ny2, ny2], nz2)
        assert not numpy.allclose(ny1, ny2)

    def test_grad_of_shared(self):
        x1 = theano.shared(3.)
        x1.name = 'x1'
        x2 = theano.tensor.vector('x2')
        y, updates = theano.scan(
            lambda v: theano.tensor.cast(v * x1, theano.config.floatX),
            sequences=x2)
        m = theano.tensor.grad(y.sum(), x1)

        f = theano.function([x2], m, allow_input_downcast=True)
        utt.assert_allclose(f([2, 3]), 5)

    def test_computing_gradient(self):
        x1 = theano.tensor.scalar('x1')
        x2 = theano.shared(numpy.array([1, 2, 3, 4, 5]), name='x2')
        K = x2 * x1

        out, updates = theano.scan(lambda i, v: theano.tensor.grad(K[i], v),
                sequences=theano.tensor.arange(K.shape[0]),
                non_sequences=x1)
        f = theano.function([x1], out, allow_input_downcast=True)

        assert numpy.all(f(3.) != 0.)

    def test_shared_updates(self):
        X = theano.shared(numpy.array(1))

        out, updates = theano.scan(
            lambda: OrderedDict([(X, (X + 1))]),
            outputs_info=[],
            non_sequences=[],
            sequences=[],
            n_steps=10)

        f = theano.function([], [], updates=updates)
        f()
        assert X.get_value() == 11

    def test_memory_aliasing_updates(self):
        x = theano.shared(numpy.array(1))
        y = theano.shared(numpy.array(1))

        out, updates = theano.scan(
            lambda: OrderedDict([(x, x + 1), (y, x)]),
            outputs_info=[],
            non_sequences=[],
            sequences=[],
            n_steps=10)

        f = theano.function([], [], updates=updates)
        f()
        assert not numpy.may_share_memory(x.container.storage[0],
                                          y.container.storage[0])

        assert x.get_value() != y.get_value()

    def test_scan_output_padding(self):
        """
        Scan outputs are usually lists, whose entries correspond to the
        intermediate result. When n_steps=1, some extra machinery is
        required in order to mimic this interface. Scan thus calls
        tensor.shape_padleft on the inner function outputs.

        However, this is not the proper behavior for shared variables,
        they should not be padded in any way

        This unit test addresses the bug fix of changeset ba7157e95cb1.
        """
        a = theano.tensor.vector()
        init_a = theano.tensor.vector()
        b = theano.shared(numpy.random.rand(5, 4))

        def inner_func(a):
            return a + 1, OrderedDict([(b, 2 * b)])

        out, updates = theano.scan(
            inner_func,
            outputs_info=[OrderedDict([('initial', init_a)])],
            n_steps=1)
        out = out[-1]
        assert out.type.ndim == a.type.ndim
        assert updates[b].type.ndim == b.type.ndim

        out, updates = theano.scan(inner_func,
                                   outputs_info=[init_a],
                                   n_steps=1)
        assert out.type.ndim == a.type.ndim + 1
        assert updates[b].type.ndim == b.type.ndim

    def test_scan_extra_inputs_hessian(self):
        x = theano.tensor.vector('x')
        A = theano.tensor.matrix('A')
        fc1 = theano.shared(0.5, name='fc1')
        fc2 = theano.shared(0.9, name='fc2')
        y = fc1 * theano.dot(x * x, theano.dot(A, x))
        y.name = 'y'
        gy = theano.tensor.grad(y, x)
        gy.name = 'gy'
        hy, updates = theano.scan(
            lambda i, gy, x: theano.tensor.grad(gy[i] * fc2, x),
            sequences=theano.tensor.arange(gy.shape[0]),
            non_sequences=[gy, x])

        f = theano.function([x, A], hy, allow_input_downcast=True)
        vx = numpy.array([1., 1.], dtype=theano.config.floatX)
        vA = numpy.array([[1., 1.], [1., 0.]], dtype=theano.config.floatX)
        vR = numpy.array([[3.6, 1.8], [1.8, 0.9]], dtype=theano.config.floatX)
        out = f(vx, vA)

        utt.assert_allclose(out, vR)

    def test_cloning_no_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.vector('y')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = theano.clone(f1,
                          replace=None,
                          strict=True,
                          share_inputs=True)
        f2_inp = theano.gof.graph.inputs([f2])

        assert z  in f2_inp
        assert x  in f2_inp
        assert y  in f2_inp

    def test_cloning_no_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.vector('y')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = theano.clone(f1,
                          replace=None,
                          strict=True,
                          share_inputs=False)
        f2_inp = theano.gof.graph.inputs([f2])

        assert not z in f2_inp
        assert not x in f2_inp
        assert not y in f2_inp

    def test_cloning_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.vector('y')
        y2 = theano.tensor.vector('y2')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = theano.clone(f1,
                          replace=OrderedDict([(y, y2)]),
                          strict=True,
                          share_inputs=True)
        f2_inp = theano.gof.graph.inputs([f2])
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test_cloning_replace_not_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.fvector('y')
        y2 = theano.tensor.dvector('y2')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = theano.clone(f1,
                          replace=OrderedDict([(y, y2)]),
                          strict=False,
                          share_inputs=True)
        f2_inp = theano.gof.graph.inputs([f2])
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test_cloning_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.vector('y')
        y2 = theano.tensor.vector('y2')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = theano.clone(f1,
                          replace=[(y, y2)],
                          strict=True,
                          share_inputs=False)
        f2_inp = theano.gof.graph.inputs([f2])
        assert not z in f2_inp
        assert not x in f2_inp
        assert not y2 in f2_inp

    def test_cloning_replace_not_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.fvector('y')
        y2 = theano.tensor.dvector('y2')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = theano.clone(f1,
                          replace=[(y, y2)],
                          strict=False,
                          share_inputs=False)
        f2_inp = theano.gof.graph.inputs([f2])
        assert not z  in f2_inp
        assert not x  in f2_inp
        assert not y2 in f2_inp

    # TEST RE-ordering of inputs
    # some rnn with multiple outputs and multiple inputs; other
    # dimension instead of scalars/vectors
    def test_reordering(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        vW = asarrayX(rng.uniform(size=(2, 2), low=-5., high=5.))
        vWout = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        vW_in1 = asarrayX(rng.uniform(size=(2, 2), low=-5., high=5.))
        v_u1 = asarrayX(rng.uniform(size=(3, 2), low=-5., high=5.))
        v_u2 = asarrayX(rng.uniform(size=(3,), low=-5., high=5.))
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = theano.shared(vW_in2, name='win2')
        W = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name='wout')
        W_in1 = theano.tensor.matrix('win')
        u1 = theano.tensor.matrix('u1')
        u2 = theano.tensor.vector('u2')
        x0 = theano.tensor.vector('x0')
        y0 = theano.tensor.vector('y0')

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, y_tm3, W_in1):
            return [y_tm3 + 1,
                    y_tm3 + 2,
                    theano.dot(u1_t, W_in1) + u2_t * W_in2 + \
                        theano.dot(x_tm1, W),
                    y_tm1 + theano.dot(x_tm1, W_out)]

        outputs, updates = theano.scan(f_rnn_cmpl,
                                       [u1, u2],
                                       [None,
                                        None,
                                        x0,
                                        dict(initial=y0, taps=[-1, -3])],
                                       W_in1,
                                       n_steps=None,
                                       truncate_gradient=-1,
                                       go_backwards=False)

        f4 = theano.function([u1, u2, x0, y0, W_in1],
                             outputs,
                             updates=updates,
                             allow_input_downcast=True)

        # compute the values in numpy
        v_x = numpy.zeros((3, 2), dtype=theano.config.floatX)
        v_y = numpy.zeros((3,), dtype=theano.config.floatX)
        v_x[0] = numpy.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + \
                    numpy.dot(v_x0, vW)
        v_y[0] = numpy.dot(v_x0, vWout) + v_y0[2]
        for i in xrange(1, 3):
            v_x[i] = numpy.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + \
                        numpy.dot(v_x[i - 1], vW)
            v_y[i] = numpy.dot(v_x[i - 1], vWout) + v_y[i - 1]

        (theano_dump1, theano_dump2, theano_x, theano_y) = f4(v_u1,
                                                              v_u2,
                                                              v_x0,
                                                              v_y0,
                                                              vW_in1)

        utt.assert_allclose(theano_x, v_x)
        utt.assert_allclose(theano_y, v_y)

    def test_scan_as_tensor_on_gradients(self):
        """
        Bug reported by cityhall on scan when computing the gradients
        """
        to_scan = theano.tensor.dvector('to_scan')
        seq = theano.tensor.dmatrix('seq')
        f1 = theano.tensor.dscalar('f1')

        def scanStep(prev, seq, f1):
            return prev + f1 * seq

        scanned, _ = theano.scan(fn=scanStep,
                                 sequences=[seq],
                                 outputs_info=[to_scan],
                                 non_sequences=[f1])

        f_scan = theano.function(inputs=[to_scan, seq, f1],
                                 outputs=scanned,
                                 allow_input_downcast=True)

        t_grad = theano.tensor.grad(scanned.sum(),
                                    wrt=[to_scan, f1],
                                    consider_constant=[seq])
        f_grad = theano.function(inputs=[to_scan, seq, f1],
                                 outputs=t_grad,
                                 allow_input_downcast=True)

    def test_save_mem(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        vW = asarrayX(rng.uniform(size=(2, 2), low=-5., high=5.))
        vWout = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        vW_in1 = asarrayX(rng.uniform(size=(2, 2), low=-5., high=5.))
        v_u1 = asarrayX(rng.uniform(size=(8, 2), low=-5., high=5.))
        v_u2 = asarrayX(rng.uniform(size=(8,), low=-5., high=5.))
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = theano.shared(vW_in2, name='win2')
        W = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name='wout')
        W_in1 = theano.tensor.matrix('win')
        u1 = theano.tensor.matrix('u1')
        u2 = theano.tensor.vector('u2')
        x0 = theano.tensor.vector('x0')
        y0 = theano.tensor.vector('y0')

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, y_tm3, W_in1):
            return [y_tm3 + 1,
                    theano.dot(u1_t, W_in1) + u2_t * W_in2 + \
                        theano.dot(x_tm1, W),
                    y_tm1 + theano.dot(x_tm1, W_out)]

        _outputs, updates = theano.scan(f_rnn_cmpl,
                                       [u1, u2],
                                       [None,
                                        dict(initial=x0),
                                        dict(initial=y0, taps=[-1, -3])],
                                       W_in1,
                                       n_steps=None,
                                       truncate_gradient=-1,
                                       go_backwards=False)
        outputs = [_outputs[0][-1], _outputs[1][-1], _outputs[2][-1]]
        f4 = theano.function([u1, u2, x0, y0, W_in1],
                             outputs,
                             updates=updates,
                             allow_input_downcast=True)

        # compute the values in numpy
        v_x = numpy.zeros((8, 2), dtype=theano.config.floatX)
        v_y = numpy.zeros((8,), dtype=theano.config.floatX)
        v_x[0] = numpy.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + \
                        numpy.dot(v_x0, vW)
        v_y[0] = numpy.dot(v_x0, vWout) + v_y0[2]

        for i in xrange(1, 8):
            v_x[i] = numpy.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + \
                        numpy.dot(v_x[i - 1], vW)
            v_y[i] = numpy.dot(v_x[i - 1], vWout) + v_y[i - 1]

        (theano_dump, theano_x, theano_y) = f4(v_u1, v_u2, v_x0, v_y0, vW_in1)

        utt.assert_allclose(theano_x, v_x[-1:])
        utt.assert_allclose(theano_y, v_y[-1:])

    def caching_nsteps_by_scan_op(self):
        W = tensor.matrix('weights')
        initial = tensor.vector('initial')
        inpt = tensor.matrix('inpt')

        def one_step(x_t, h_tm1, W):
            expr = tensor.dot(h_tm1, W) + x_t
            return expr

        expr, _ = theano.scan(
          fn=one_step,
          sequences=[inpt],
          outputs_info=[initial],
          non_sequences=[W])

        sh = expr.shape[0]

        v1 = theano.shared(numpy.ones(5, dtype=theano.config.floatX))
        v2 = theano.shared(numpy.ones((5, 5), dtype=theano.config.floatX))
        shapef = theano.function([W],
                                 expr,
                                 givens=OrderedDict([(initial, v1),
                                         (inpt, v2)]))
        # First execution to cache n_steps
        shapef(numpy.ones((5, 5), dtype=theano.config.floatX))

        cost = expr.sum()
        d_cost_wrt_W = tensor.grad(cost, [W])
        f = theano.function(
            [W, inpt], d_cost_wrt_W,
            givens=OrderedDict([(initial, theano.shared(numpy.zeros(5)))]))

        rval = numpy.asarray([[5187989] * 5] * 5, dtype=theano.config.floatX)
        arg1 = numpy.ones((5, 5), dtype=theano.config.floatX)
        arg2 = numpy.ones((10, 5), dtype=theano.config.floatX)
        utt.assert_allclose(f(arg1, arg2), rval)

    def test_save_mem_reduced_number_of_steps(self):
        def f_rnn(u_t):
            return (u_t + 1.,
                    u_t + 2.,
                    u_t + 3.,
                    u_t + 4.,
                    u_t + 5.,
                    u_t + 6.,
                    u_t + 7.)

        u = theano.tensor.vector('u')
        idx = theano.tensor.iscalar('idx')
        jdx = theano.tensor.iscalar('jdx')
        [x1, x2, x3, x4, x5, x6, x7], updates = \
                theano.scan(f_rnn,
                            u,
                            n_steps=None,
                            truncate_gradient=-1,
                            go_backwards=False)

        f2 = theano.function([u, idx, jdx],
                             [x1[:2],
                              x2[4],
                              x3[idx],
                              x4[:idx],
                              x5[-10],
                              x6[-jdx],
                              x7[:-jdx]],
                              updates=updates,
                              allow_input_downcast=True)
        # get random initial values
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(20,), low=-5., high=5.)

        # compute the output in numpy
        tx1, tx2, tx3, tx4, tx5, tx6, tx7 = f2(v_u, 3, 15)

        utt.assert_allclose(tx1, v_u[:2] + 1.)
        utt.assert_allclose(tx2, v_u[4] + 2.)
        utt.assert_allclose(tx3, v_u[3] + 3.)
        utt.assert_allclose(tx4, v_u[:3] + 4.)
        utt.assert_allclose(tx5, v_u[-10] + 5.)
        utt.assert_allclose(tx6, v_u[-15] + 6.)
        utt.assert_allclose(tx7, v_u[:-15] + 7.)
        scan_node = f2.maker.fgraph.outputs[0].owner.inputs[0]

        # Maybe ugly, way to check if the optimization had
        # been applied

    def test_save_mem_store_steps(self):

        def f_rnn(u_t, x1_tm1, x1_tm3, x2_tm1, x3tm2, x3_tm1, x4_tm1):
            return (u_t + 1.,
                    u_t + 2.,
                    u_t + 3.,
                    u_t + 4.,
                    u_t + 5.,
                    u_t + 6.,
                    u_t + 7.)

        u = theano.tensor.vector('u')
        idx = theano.tensor.iscalar('idx')
        jdx = theano.tensor.iscalar('jdx')
        x10 = theano.tensor.vector('x10')
        x20 = theano.tensor.scalar('x20')
        x30 = theano.tensor.vector('x30')
        x40 = theano.tensor.scalar('x40')
        [x1, x2, x3, x4, x5, x6, x7], updates = \
                theano.scan(f_rnn,
                            u,
                            [None,
                             None,
                             None,
                             dict(initial=x10, taps=[-1, -2]),
                             x20,
                             dict(initial=x30, taps=[-1, -2]),
                             x40],
                            n_steps=None,
                            truncate_gradient=-1,
                            go_backwards=False)

        f2 = theano.function([u, x10, x20, x30, x40],
                             [x1[-7], x2[-3: -1], x3[-6:], x4[-1], x5[-1]],
                             updates=updates,
                             allow_input_downcast=True)

        # get random initial values
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(20,), low=-5., high=5.)

        # compute the output in numpy
        tx1, tx2, tx3, tx4, tx5 = f2(v_u, [0, 0], 0, [0, 0], 0)

        utt.assert_allclose(tx1, v_u[-7] + 1.)
        utt.assert_allclose(tx2, v_u[-3:-1] + 2.)
        utt.assert_allclose(tx3, v_u[-6:] + 3.)
        utt.assert_allclose(tx4, v_u[-1] + 4.)
        utt.assert_allclose(tx5, v_u[-1] + 5.)

    def test_use_scan_direct_output(self):
        # This test looks for a crash that happened when directly using the
        # recurrent output of a scan node instead of taking the result
        # returned by the scan() function

        # Obtain a compilation mode that will cause the test to fail if an
        # exception occurs in the optimization process
        on_opt_error = theano.config.on_opt_error
        theano.config.on_opt_error = "raise"
        mode = theano.compile.get_default_mode()
        theano.config.on_opt_error = on_opt_error

        x = tensor.scalar()
        seq = tensor.vector()
        outputs_info=[x, tensor.zeros_like(x)]
        (out1, out2), updates = theano.scan(lambda a, b, c : (a + b, b + c),
                                            sequences=seq,
                                            outputs_info=outputs_info,
                                            mode=mode)

        # Obtain a reference to the scan outputs before the subtensor and
        # compile a function with them as outputs
        assert isinstance(out1.owner.op, tensor.subtensor.Subtensor)
        assert isinstance(out2.owner.op, tensor.subtensor.Subtensor)
        out1_direct = out1.owner.inputs[0]
        out2_direct = out2.owner.inputs[0]
        fct = theano.function([x, seq],
                              [out1_direct[:-1], out2_direct[:-1]],
                              mode=mode)

        # Test the function to ensure valid outputs
        floatX = theano.config.floatX

        init_value = 5.0
        seq_value = numpy.arange(4, dtype=floatX)
        output1, output2 = fct(init_value, seq_value)

        expected_output1 = [init_value]
        expected_output2 = [0]
        for i in seq_value[:-1]:
            expected_output2.append(expected_output1[-1] +
                                    expected_output2[-1])
            expected_output1.append(expected_output1[-1] + i)

        utt.assert_allclose(output1, expected_output1)
        utt.assert_allclose(output2, expected_output2)

    def test_use_scan_direct_output2(self):
        # This test looks for a crash that happened when directly using the
        # recurrent output of a scan node associated with a state with a
        # state with broadcastable dimensions

        x = tensor.dcol()
        seq = tensor.dcol()
        outputs_info=[x, tensor.zeros_like(x)]
        (out1, out2), updates = theano.scan(lambda a, b, c : (a + b, a + c),
                                            sequences=seq,
                                            outputs_info=outputs_info)

        # Obtain a reference to the scan outputs before the subtensor and
        # compile a function with them as outputs
        assert isinstance(out1.owner.op, tensor.subtensor.Subtensor)
        assert isinstance(out2.owner.op, tensor.subtensor.Subtensor)
        out1_direct = out1.owner.inputs[0]
        out2_direct = out2.owner.inputs[0]
        fct = theano.function([x, seq],
                              [out1_direct, out2_direct])

        # Test that the function returns valid outputs
        x_val = numpy.arange(0, 4)[:, None]
        seq_val = numpy.arange(4, 8)[:, None]

        out1, out2 = fct(x_val, seq_val)

        expected_out1 = numpy.zeros((5, 4, 1))
        expected_out2 = numpy.zeros((5, 4, 1))
        for i in range(4):
            expected_out2[i + 1] = expected_out2[i] + seq_val[i]
        for i in range(5):
            expected_out1[i] = expected_out2[i] + x_val

        utt.assert_allclose(out1, expected_out1)
        utt.assert_allclose(out2, expected_out2)

    def test_infer_shape(self):
        # Test for a crash in scan.infer_shape when using both
        # an until condition and random sampling in the inner function.

        x = tensor.scalar()
        srng = theano.tensor.shared_randomstreams.RandomStreams(0)

        def inner_fct(previous_val):
            new_val = previous_val + srng.uniform()
            condition = theano.scan_module.until(previous_val > 5)
            return new_val, condition

        out, updates = theano.scan(inner_fct,
                                   outputs_info=x,
                                   n_steps=10)

        g_out = tensor.grad(out.sum(), x)
        fct = theano.function([x], [out, g_out])

        for i in xrange(-5, 5):
            output, g_output = fct(i)
            assert len(output) == g_output

    def test_infer_shape2(self):
        # Ensure that the shape inference can remove the Scan node in the
        # case of a complicated inner graph involving sequences and recurrent
        # states

        seq = tensor.lvector()
        sitsot_init = tensor.lscalar()
        mitsot_init = tensor.lvector()

        def step(seq1, sitsot_m1, mitsot_m2, mitsot_m1):
            # Every iteration, the sitsot state decreases and the mitsot state
            # increases such that their total value remains identical. This
            # is because this value will be used as the shape of a nitsot
            # output and the outputs of every iteration need to have the same
            # shape
            diff = mitsot_m1 + seq1
            next_mitsot_val = mitsot_m2 + diff
            next_sitsot_val = sitsot_m1 - diff
            nitsot_out = tensor.alloc(numpy.asarray(0., 'float32'),
                                      next_mitsot_val +
                                      next_sitsot_val)
            return next_sitsot_val, next_mitsot_val, nitsot_out

        out, updates = theano.scan(fn=step,
                                   sequences=seq,
                                   outputs_info=[sitsot_init,
                                                 {'initial': mitsot_init,
                                                  'taps': [-2, -1]},
                                                 None],
                                   n_steps=5)

        f = theano.function([seq, sitsot_init, mitsot_init], out[2].shape,
                            mode='FAST_RUN')
        # When Scan.infer_shape will cover more case, there will no scan left.
        assert(len(scan_nodes_from_fct(f)) == 1)

        # This generate a scan crash during execution.
        # output_shape = f(numpy.arange(5), 5, [1, 2])
        # assert(all(output_shape == (5, 6)))

    # The following test will fail in DebugMode if there are
    # some problems in Scan.infer_shape
    def test_remove_stuff(self):
        x = theano.tensor.vector('x')

        def lm(m):
            trng = theano.tensor.shared_randomstreams.RandomStreams(
                                                     utt.fetch_seed())
            return [2 * m + trng.uniform(low=-1.1, high=1.1,
                                      dtype=theano.config.floatX),
                    m + trng.uniform(size=[3])]

        [o1, o2], updates = theano.scan(lm,
                                        sequences=x,
                                        n_steps=None,
                                        truncate_gradient=-1,
                                        name='forward',
                                        go_backwards=False)
        go1 = theano.tensor.grad(o1.mean(), wrt=x)
        f = theano.function([x], go1, updates=updates,
                            allow_input_downcast=True, mode=mode_with_opt)
        self.assertTrue(numpy.allclose(f([1, 2, 3]), 2. / 3))

        topo = f.maker.fgraph.toposort()
        # this new assert is here to test if scan_merging works ..
        nb_scan = len([n for n in topo
            if isinstance(n.op, theano.scan_module.scan_op.Scan)])
        self.assertTrue(nb_scan == 1)
        nb_shape_i = len([n for n in topo
            if isinstance(n.op, theano.tensor.opt.Shape_i)])
        if theano.config.mode != 'FAST_COMPILE':
            self.assertTrue(nb_shape_i == 1)

    def test_merge(self):
        x = theano.tensor.vector()
        y = theano.tensor.vector()

        def sum(s):
            return s + 1

        sx, upx = theano.scan(sum, sequences=[x])
        sy, upy = theano.scan(sum, sequences=[y])

        f = theano.function([x, y], [sx, sy],
                            mode=mode_with_opt.excluding('scanOp_pushout_seqs_ops'))
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(
            n.op, theano.scan_module.scan_op.Scan)]
        self.assertTrue(len(scans) == 2)

        sx, upx = theano.scan(sum, sequences=[x], n_steps=2)
        sy, upy = theano.scan(sum, sequences=[y], n_steps=3)

        f = theano.function([x, y], [sx, sy],
                            mode=mode_with_opt.excluding('scanOp_pushout_seqs_ops'))
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(
            n.op, theano.scan_module.scan_op.Scan)]
        self.assertTrue(len(scans) == 2)

        sx, upx = theano.scan(sum, sequences=[x], n_steps=4)
        sy, upy = theano.scan(sum, sequences=[y], n_steps=4)

        f = theano.function([x, y], [sx, sy],
                            mode=mode_with_opt.excluding('scanOp_pushout_seqs_ops'))
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(
            n.op, theano.scan_module.scan_op.Scan)]
        self.assertTrue(len(scans) == 1)

        sx, upx = theano.scan(sum, sequences=[x])
        sy, upy = theano.scan(sum, sequences=[x])

        f = theano.function([x], [sx, sy],
                            mode=mode_with_opt.excluding('scanOp_pushout_seqs_ops'))
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(
            n.op, theano.scan_module.scan_op.Scan)]
        self.assertTrue(len(scans) == 1)

        sx, upx = theano.scan(sum, sequences=[x])
        sy, upy = theano.scan(sum, sequences=[x], mode='FAST_COMPILE')

        f = theano.function([x], [sx, sy],
                            mode=mode_with_opt.excluding('scanOp_pushout_seqs_ops'))
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(
            n.op, theano.scan_module.scan_op.Scan)]
        self.assertTrue(len(scans) == 1)

        sx, upx = theano.scan(sum, sequences=[x])
        sy, upy = theano.scan(sum, sequences=[x], truncate_gradient=1)

        f = theano.function([x], [sx, sy],
                            mode=mode_with_opt.excluding('scanOp_pushout_seqs_ops'))
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(
            n.op, theano.scan_module.scan_op.Scan)]
        self.assertTrue(len(scans) == 2)

    def test_merge_3scans(self):
        # This test checks a case where we have 3 scans, two of them
        # cannot be merged together, but the third one can be merged with
        # either.
        x = theano.tensor.vector()
        y = theano.tensor.vector()

        def sum(s):
            return s + 1

        sx, upx = theano.scan(sum, sequences=[x], n_steps=4, name='X')
        # We need to use an expression of y rather than y so the toposort
        # comes up with the 'Y' scan last.
        sy, upy = theano.scan(sum, sequences=[2 * y + 2], n_steps=4, name='Y')
        sz, upz = theano.scan(sum, sequences=[sx], n_steps=4, name='Z')

        f = theano.function(
            [x, y], [sy, sz],
            mode=mode_with_opt.excluding('scanOp_pushout_seqs_ops'))
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(
            n.op, theano.scan_module.scan_op.Scan)]
        self.assertTrue(len(scans) == 2)

        rng = numpy.random.RandomState(utt.fetch_seed())
        x_val = rng.uniform(size=(4,)).astype(theano.config.floatX)
        y_val = rng.uniform(size=(4,)).astype(theano.config.floatX)
        # Run it so DebugMode can detect optimization problems.
        f(x_val, y_val)

    def test_pushout_seqs(self):

        def init_predictive_output(inputs,targets,hyp,x_star,s_star):
            E = hyp.shape[0]

            def init_K(i,X,Y):
                XX = X.sum(1).reshape((X.shape[0], 1))
                K = (XX + XX.T)
                return K.sum()

            beta, K_updts = theano.scan(init_K, sequences=tensor.arange(E),
                                        non_sequences=[inputs,targets])

            # mean
            def predict_mean_i(i,x_star,s_star,X,beta,h):
                n,D = tensor.shape(X)
                # rescale every dimension by the corresponding inverse lengthscale
                iL = tensor.diag(h[i,:D])
                inp = (X - x_star).dot(iL)

                # compute the mean
                B = iL.dot(s_star).dot(iL)
                t = inp.dot(B)

                lb = (inp * t).sum() + beta.sum()

                Mi = tensor.sum(lb) * h[i,D]
                return Mi

            (M), M_updts = theano.scan( predict_mean_i ,
                                        sequences=tensor.arange(E),
                                        non_sequences=[x_star,s_star,inputs,beta,hyp] )
            return M

        # some initializations
        hypx = numpy.log(numpy.tile([1,1,1,1,1,1,0.01], (3,1)))

        # variables used in the following expressions
        hyp = theano.shared(hypx)
        inputs = tensor.dmatrix('X')
        targets = tensor.dmatrix('Y')
        x_star = tensor.dvector('x_star')
        s_star = tensor.dmatrix('s_star')

        M = init_predictive_output(inputs,targets,hyp,x_star,s_star)

        X = numpy.random.random((10,4))
        Y = numpy.random.random((10,3))
        test_m = numpy.random.random((4,))
        test_s = numpy.eye(4)

        # Compute expected outputs (jacobian of M wrt x_star)
        dfdm = theano.function([inputs,targets,x_star,s_star],
                               [tensor.grad(M[0],x_star),
                                tensor.grad(M[1],x_star),
                                tensor.grad(M[2],x_star)])
        expected_output = dfdm(X,Y,test_m,test_s)

        # equivalent code for the jacobian using scan
        dMdm, dMdm_updts = theano.scan(lambda i,M,x: tensor.grad(M[i],x),
                                       sequences=tensor.arange(M.shape[0]),
                                       non_sequences=[M,x_star])
        dfdm = theano.function([inputs,targets,x_star,s_star],
                               [dMdm[0], dMdm[1], dMdm[2]])
        scan_output = dfdm(X,Y,test_m,test_s)

        # equivalent code for the jacobian using tensor.jacobian
        dMdm_j = tensor.jacobian(M,x_star)
        dfdm_j = theano.function([inputs,targets,x_star,s_star],
                                 [dMdm_j[0], dMdm_j[1], dMdm_j[2]])
        jacobian_outputs = dfdm_j(X,Y,test_m,test_s)

        utt.assert_allclose(expected_output, scan_output)
        utt.assert_allclose(expected_output, jacobian_outputs)

    @theano.configparser.change_flags(on_opt_error='raise')
    def test_pushout_seqs2(self):
        # This test for a bug with PushOutSeqScan that was reported on the
        # theano-user mailing list where the optimization raised an exception
        # when applied on this graph.
        x = tensor.matrix()
        outputs, updates = theano.scan(
            lambda x: [x*x, tensor.constant(0).copy().copy()],
            n_steps=2,
            sequences=[],
            non_sequences=[],
            outputs_info=[x, None])

        # Compile a theano function where any optimization error will lead to
        # an exception being raised
        theano.function([x], outputs, updates=updates)

    @theano.configparser.change_flags(on_opt_error='raise')
    def test_pushout_nonseq(self):
        # Test case originally reported by Daniel Renshaw. The crashed occured
        # during the optimization PushOutNonSeqScan when it attempted to
        # a scan node with two outputs but only providing a replacement for
        # one of those outputs. This led the optimization to raise an
        # exception.

        outputs, _ = theano.scan(lambda x: (x * x, x),
                                 non_sequences=[2], n_steps=2)
        f = theano.function(inputs=[], outputs=outputs)

        outs = f()
        expected_outs = [[4, 4], [2, 2]]
        utt.assert_allclose(outs, expected_outs)

    def test_sequence_dict(self):
        # Test that we can specify sequences as a dictionary with
        # only the 'input' key
        def incr(s):
            return s + 1

        x = theano.tensor.vector()
        sx, upx = theano.scan(
            fn=incr,
            sequences=[{'input': x}])
        f = theano.function([x], sx)

    def test_hash(self):
        x = theano.tensor.vector()
        y = theano.tensor.vector()
        scan1, updates = theano.scan(lambda _x: _x + 1, x)
        scan2, updates = theano.scan(lambda _x: _x + 1, y)
        assert scan1.owner.op == scan2.owner.op
        assert hash(scan1.owner.op) == hash(scan2.owner.op)

    def test_same(self):
        # This test is checking a bug discovered by Arnaud and it is based
        # on his code

        x = theano.tensor.fmatrix('x')

        mem_val = numpy.zeros((2,), dtype='float32')
        memory = theano.shared(mem_val)
        W = theano.shared(numpy.random.random((5, 2)).astype('float32'))

        def f(inp, mem):
            i = theano.tensor.join(0, inp, mem)
            d = theano.tensor.dot(i, W)
            return d, d

        outs, updts = theano.scan(f, sequences=[x],
                                  non_sequences=[],
                                  outputs_info=[None, memory])

        f = theano.function([x], outs[0])
        f2 = theano.function([x], outs[1])

        x_val = numpy.random.random((4, 3)).astype('float32')

        f_vals = f(x_val)
        memory.set_value(mem_val)
        f2_vals = f2(x_val)
        utt.assert_allclose(f_vals, f2_vals)

    def test_reduce_memory_consumption(self):

        x = theano.shared(numpy.asarray(
            numpy.random.uniform(size=(10,)), dtype=theano.config.floatX))
        o, _ = theano.reduce(lambda v, acc: acc + v,
                             x,
                             theano.tensor.constant(
                                 numpy.asarray(0.,
                                               dtype=theano.config.floatX)))
        mode = theano.compile.mode.FAST_RUN
        mode = mode.excluding('inplace')
        f1 = theano.function([], o, mode=mode)
        inputs, outputs = clone_optimized_graph(f1)

        scan_nodes = grab_scan_node(outputs[0])
        assert scan_nodes is not None
        scan_node = scan_nodes[0]
        f1 = theano.function(inputs, scan_node.inputs[2])

        # Originally, the shape would have been 1 due to the SaveMem
        # optimization reducing the size to the number of taps (in this case
        # 1) provided to the inner function. Now, because of the memory-reuse
        # feature in Scan it can be 2 because SaveMem needs to keep a
        # larger buffer to avoid aliasing between the inputs and the outputs.
        if theano.config.scan.allow_output_prealloc:
            assert f1().shape[0] == 2
        else:
            assert f1().shape[0] == 1

        gx = theano.tensor.grad(o, x)
        f2 = theano.function([], gx)
        utt.assert_allclose(f2(), numpy.ones((10,)))

    def test_foldl_memory_consumption(self):
        x = theano.shared(numpy.asarray(
            numpy.random.uniform(size=(10,)), dtype=theano.config.floatX))
        o, _ = theano.foldl(lambda v, acc: acc + v,
                            x,
                            theano.tensor.constant(
                                numpy.asarray(0.,
                                              dtype=theano.config.floatX)))

        mode = theano.compile.mode.FAST_RUN
        mode = mode.excluding('inplace')
        f0 = theano.function([], o, mode=mode)
        inputs, outputs = clone_optimized_graph(f0)

        scan_nodes = grab_scan_node(outputs[0])
        assert scan_nodes is not None
        scan_node = scan_nodes[0]
        f1 = theano.function(inputs, scan_node.inputs[2])

        # Originally, the shape would have been 1 due to the SaveMem
        # optimization reducing the size to the number of taps (in this case
        # 1) provided to the inner function. Now, because of the memory-reuse
        # feature in Scan it can be 2 because SaveMem needs to keep a
        # larger buffer to avoid aliasing between the inputs and the outputs.
        if theano.config.scan.allow_output_prealloc:
            assert f1().shape[0] == 2
        else:
            assert f1().shape[0] == 1

        gx = theano.tensor.grad(o, x)
        f2 = theano.function([], gx)
        utt.assert_allclose(f2(), numpy.ones((10,)))

    def test_foldr_memory_consumption(self):

        x = theano.shared(numpy.asarray(
            numpy.random.uniform(size=(10,)), dtype=theano.config.floatX))
        o, _ = theano.foldr(lambda v, acc: acc + v,
                            x,
                            theano.tensor.constant(
                                numpy.asarray(0.,
                                              dtype=theano.config.floatX)))

        mode = theano.compile.mode.FAST_RUN
        mode = mode.excluding('inplace')
        f1 = theano.function([], o, mode=mode)
        inputs, outputs = clone_optimized_graph(f1)

        scan_nodes = grab_scan_node(outputs[0])
        assert scan_nodes is not None
        scan_node = scan_nodes[0]
        f1 = theano.function(inputs, scan_node.inputs[2])

        # Originally, the shape would have been 1 due to the SaveMem
        # optimization reducing the size to the number of taps (in this case
        # 1) provided to the inner function. Now, because of the memory-reuse
        # feature in Scan it can be 2 because SaveMem needs to keep a
        # larger buffer to avoid aliasing between the inputs and the outputs.
        if theano.config.scan.allow_output_prealloc:
            assert f1().shape[0] == 2
        else:
            assert f1().shape[0] == 1

        gx = theano.tensor.grad(o, x)
        f2 = theano.function([], gx)
        utt.assert_allclose(f2(), numpy.ones((10,)))

    @attr('slow')
    def test_rop2(self):
        seed = utt.fetch_seed()
        rng = numpy.random.RandomState(seed)
        floatX = theano.config.floatX
        v_u = numpy.array(rng.uniform(size=(3, 5)) - .5, dtype=floatX)
        v_W = numpy.array(rng.uniform(size=(5, 5)) - .5, dtype=floatX)
        v_h0 = numpy.array(rng.uniform(size=(5,)) - .5, dtype=floatX)

        v_eu = numpy.array(rng.uniform(size=(3, 5)) - .5, dtype=floatX)
        v_eW = numpy.array(rng.uniform(size=(5, 5)) - .5, dtype=floatX)
        v_eh0 = numpy.array(rng.uniform(size=(5,)) - .5, dtype=floatX)

        def rnn_fn(_u, _y, _W):

            srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
            tmp_val = _u + _y + srng.uniform(size=v_h0.shape) *\
                        numpy.asarray(1e-6, dtype=floatX)
            sl_o = theano.tensor.tanh(theano.tensor.dot(_W, tmp_val))
            return sl_o, tmp_val

        u = theano.tensor.matrix('U')
        h0 = theano.tensor.vector('h0')
        W = theano.tensor.matrix('W')

        _u = theano.tensor.specify_shape(u, v_u.shape)
        _u.name = '_U'
        _h0 = theano.tensor.specify_shape(h0, v_h0.shape)
        _h0.name = '_h0'
        _W = theano.tensor.specify_shape(W, v_W.shape)
        _W.name = '_W'

        [o, _], _ = theano.scan(rnn_fn,
                           sequences=_u,
                           outputs_info=[_h0, None],
                           non_sequences=_W,
                           name='rnn_fn')
        o = o[-1]
        eu = theano.tensor.matrix('eu')
        eh0 = theano.tensor.vector('eh0')
        eW = theano.tensor.matrix('eW')

        nwo_u = theano.tensor.Rop(o, _u, eu)
        nwo_h0 = theano.tensor.Rop(o, _h0, eh0)
        nwo_W = theano.tensor.Rop(o, _W, eW)
        fn_rop = theano.function([u, h0, W, eu, eh0, eW],
                                 [nwo_u, nwo_h0, nwo_W, o],
                                 on_unused_input='ignore')
        vnu, vnh0, vnW, vno = fn_rop(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)

        n2o_u, _ = theano.scan(lambda i, o, u, h0, W, eu: \
                                (theano.tensor.grad(o[i], u) * eu).sum(),
                              sequences=tensor.arange(o.shape[0]),
                              non_sequences=[o, u, h0, W, eu],
                              name='jacobU')

        n2o_h0, _ = theano.scan(lambda i, o, u, h0, W, eh0: \
                                  (theano.tensor.grad(o[i], h0) * eh0).sum(),
                              sequences=tensor.arange(o.shape[0]),
                              non_sequences=[o, u, h0, W, eh0],
                              name='jacobh')

        n2o_W, _ = theano.scan(lambda i, o, u, h0, W, eW: \
                                  (theano.tensor.grad(o[i], W) * eW).sum(),
                              sequences=tensor.arange(o.shape[0]),
                              non_sequences=[o, u, h0, W, eW],
                             name='jacobW')

        fn_test = theano.function([u, h0, W, eu, eh0, eW],
                                  [n2o_u, n2o_h0, n2o_W, o],
                                  on_unused_input='ignore')

        tnu, tnh0, tnW, tno = fn_test(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)
        utt.assert_allclose(vnu, tnu, atol=1e-6)
        utt.assert_allclose(vnh0, tnh0, atol=1e-6)
        utt.assert_allclose(vnW, tnW, atol=1e-6)

    def test_rop(self):
        seed = utt.fetch_seed()
        rng = numpy.random.RandomState(seed)
        floatX = theano.config.floatX
        v_u = numpy.array(rng.uniform(size=(20, 5)), dtype=floatX)
        v_W = numpy.array(rng.uniform(size=(5, 5)), dtype=floatX)
        v_h0 = numpy.array(rng.uniform(size=(5,)), dtype=floatX)

        v_eu = numpy.array(rng.uniform(size=(20, 5)), dtype=floatX)
        v_eW = numpy.array(rng.uniform(size=(5, 5)), dtype=floatX)
        v_eh0 = numpy.array(rng.uniform(size=(5,)), dtype=floatX)

        def rnn_fn(_u, _y, _W):
            sl_o = theano.tensor.tanh(theano.tensor.dot(_W, (_u + _y)))
            return sl_o

        u = theano.tensor.matrix('U')
        h0 = theano.tensor.vector('h0')
        W = theano.tensor.matrix('W')

        _u = theano.tensor.specify_shape(u, v_u.shape)
        _u.name = '_U'
        _h0 = theano.tensor.specify_shape(h0, v_h0.shape)
        _h0.name = '_h0'
        _W = theano.tensor.specify_shape(W, v_W.shape)
        _W.name = '_W'

        o, _ = theano.scan(rnn_fn,
                           sequences=_u,
                           outputs_info=_h0,
                           non_sequences=_W,
                           name='rnn_fn')
        o = o[-1]
        eu = theano.tensor.matrix('eu')
        eh0 = theano.tensor.vector('eh0')
        eW = theano.tensor.matrix('eW')

        nwo_u = theano.tensor.Rop(o, _u, eu)
        nwo_h0 = theano.tensor.Rop(o, _h0, eh0)
        nwo_W = theano.tensor.Rop(o, _W, eW)
        fn_rop = theano.function([u, h0, W, eu, eh0, eW],
                                 [nwo_u, nwo_h0, nwo_W],
                                 on_unused_input='ignore')

        n2o_u, _ = theano.scan(lambda i, o, u, h0, W, eu: \
                                (theano.tensor.grad(o[i], u) * eu).sum(),
                              sequences=tensor.arange(o.shape[0]),
                              non_sequences=[o, u, h0, W, eu],
                              name='jacobU')

        n2o_h0, _ = theano.scan(lambda i, o, u, h0, W, eh0: \
                                  (theano.tensor.grad(o[i], h0) * eh0).sum(),
                              sequences=tensor.arange(o.shape[0]),
                              non_sequences=[o, u, h0, W, eh0],
                              name='jacobh')

        n2o_W, _ = theano.scan(lambda i, o, u, h0, W, eW: \
                                  (theano.tensor.grad(o[i], W) * eW).sum(),
                              sequences=tensor.arange(o.shape[0]),
                              non_sequences=[o, u, h0, W, eW],
                             name='jacobW')

        fn_test = theano.function([u, h0, W, eu, eh0, eW],
                                  [n2o_u, n2o_h0, n2o_W],
                                  on_unused_input='ignore')

        vnu, vnh0, vnW = fn_rop(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)
        tnu, tnh0, tnW = fn_test(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)

        utt.assert_allclose(vnu, tnu, atol=1e-6)
        utt.assert_allclose(vnh0, tnh0, atol=1e-6)
        utt.assert_allclose(vnW, tnW, atol=1e-6)

    def test_pushout_dot(self):
        W = tensor.matrix('W')
        h = tensor.matrix('h')

        o, _ = theano.scan(lambda hi, him1, W: (hi, tensor.dot(hi+him1, W)),
                           outputs_info=[tensor.zeros([h.shape[1]]), None],
                           sequences=[h],
                           non_sequences=[W])

        f = theano.function([W, h], o, mode=mode_with_opt)

        scan_nodes = [x for x in f.maker.fgraph.toposort()
                     if isinstance(x.op,
                                   theano.scan_module.scan_op.Scan)]
        assert len(scan_nodes) == 1
        scan_op = scan_nodes[0].op
        assert not any(isinstance(n.op, tensor.Dot) for n in
                       scan_op.fn.maker.fgraph.apply_nodes)

    def test_pushout_all(self):
        W1 = tensor.matrix('W1')
        W2 = tensor.matrix('W2')
        h0 = tensor.vector('h0')

        def lambda_fn(h, W1, W2):
            return tensor.dot(h, W1 + W2)

        o, _ = theano.scan(lambda_fn,
                           non_sequences=[h0, W1, W2],
                           n_steps=5)

        f = theano.function([h0, W1, W2], o, mode=mode_with_opt)

        scan_nodes = [x for x in f.maker.fgraph.toposort()
                     if isinstance(x.op,
                                   theano.scan_module.scan_op.Scan)]
        assert len(scan_nodes) == 0

        seed = utt.fetch_seed()
        rng = numpy.random.RandomState(seed)
        floatX = theano.config.floatX
        v_h = numpy.array(rng.uniform(size=(2,)), dtype=floatX)
        v_W1 = numpy.array(rng.uniform(size=(2, 2)), dtype=floatX)
        v_W2 = numpy.array(rng.uniform(size=(2, 2)), dtype=floatX)

        v_out = numpy.dot(v_h, v_W1 + v_W2)
        sol = numpy.zeros((5, 2))
        # This line is here to make sol have the same shape as the output of
        # theano. Note that what we ask theano to do is to repeat the 2
        # elements vector v_out 5 times
        sol[:, :] = v_out
        utt.assert_allclose(sol, f(v_h, v_W1, v_W2))

    def test_pushout_while(self):
        # Ensure that the optimizations for Scan that push computation out of
        # the Scan don't alter the result for 'as_while' scans.

        W1 = tensor.matrix('W1')
        W2 = tensor.matrix('W2')
        step_indices = tensor.vector('step_indices')

        def lambda_fn(step_idx, W1, W2):
            until_condition = theano.scan_module.until(step_idx > 2)
            return tensor.dot(W1, W2), until_condition

        # Compile a function with the optimization
        o, _ = theano.scan(lambda_fn,
                           sequences=[step_indices, W1],
                           non_sequences=[W2],
                           n_steps=5)

        f = theano.function([W1, W2, step_indices], o, mode=mode_with_opt)

        # Compule an theano function without the optimization
        o, _ = theano.scan(lambda_fn,
                           sequences=[step_indices, W1],
                           non_sequences=[W2],
                           n_steps=5, mode='FAST_COMPILE')

        f_ref = theano.function([W1, W2, step_indices], o, mode='FAST_COMPILE')

        # Compare the results of the two implementations
        input_values = [numpy.random.random((5, 5)).astype("float32"),
                        numpy.random.random((5, 5)).astype("float32"),
                        numpy.arange(5).astype("float32")]

        out = f(*input_values)
        out_ref = f_ref(*input_values)
        utt.assert_allclose(out, out_ref)

    def test_pushout(self):
        W1 = tensor.matrix('W1')
        W2 = tensor.matrix('W2')
        h0 = tensor.vector('h0')

        def lambda_fn(h, W1, W2):
            return tensor.dot(h, W1 + W2)

        o, _ = theano.scan(lambda_fn,
                           outputs_info=h0,
                           non_sequences=[W1, W2],
                           n_steps=5)

        f = theano.function([h0, W1, W2], o, mode=mode_with_opt)

        scan_node = [x for x in f.maker.fgraph.toposort()
                     if isinstance(x.op,
                                   theano.scan_module.scan_op.Scan)][0]
        assert len([x for x in scan_node.op.fn.maker.fgraph.toposort()
                    if isinstance(x.op, theano.tensor.Elemwise)]) == 0

    def test_pushout_nomodif(self):
        inp = tensor.matrix('inp')

        def fn(i, i_tm1):
            return i + 10, i_tm1

        ([i_t, i_tm1], _) = theano.scan(
            fn, sequences=[inp],
            outputs_info=[numpy.asarray([0.0, 0.0], theano.config.floatX),
                          None])
        f = theano.function([inp], [i_t, i_tm1])
        val = numpy.arange(10).reshape(5, 2).astype(theano.config.floatX)
        ret = f(val)
        utt.assert_allclose(ret[0], val + 10)
        utt.assert_allclose(ret[1], [[0.,  0.],
                                     [10., 11.],
                                     [12., 13.],
                                     [14., 15.],
                                     [16., 17.]])

    def test_alloc_inputs1(self):
        W1 = tensor.matrix('W1')
        W2 = tensor.matrix('W2')
        h0 = tensor.vector('h0')

        def lambda_fn(h, W1, W2):
            return tensor.dot(h, W1 * W2)
        o, _ = theano.scan(lambda_fn,
                           outputs_info=h0,
                           non_sequences=[W1, tensor.zeros_like(W2)],
                           n_steps=5)

        f = theano.function([h0, W1, W2], o, mode=mode_with_opt)
        scan_node = [x for x in f.maker.fgraph.toposort()
                     if isinstance(x.op,
                                   theano.scan_module.scan_op.Scan)][0]
        assert len([x for x in scan_node.op.fn.maker.fgraph.toposort()
                    if isinstance(x.op, theano.tensor.Elemwise)]) == 0

    def test_alloc_inputs2(self):
        raise SkipTest("This tests depends on an optimization for "
                       "scan that has not been implemented yet.")
        W1 = tensor.matrix()
        W2 = tensor.matrix()
        h0 = tensor.vector()

        def lambda_fn(W1, h, W2):
            return W1 * tensor.dot(h, W2)

        o, _ = theano.scan(lambda_fn,
                           sequences=tensor.zeros_like(W1),
                           outputs_info=h0,
                           non_sequences=[tensor.zeros_like(W2)],
                           n_steps=5)

        f = theano.function([h0, W1, W2], o, mode=mode_with_opt)
        scan_node = [x for x in f.maker.fgraph.toposort()
                     if isinstance(x.op,
                                   theano.scan_module.scan_op.Scan)][0]

        assert len([x for x in scan_node.op.fn.maker.fgraph.toposort()
                    if isinstance(x.op, theano.tensor.Elemwise)]) == 0

    def test_alloc_inputs3(self):
        _W1 = tensor.matrix()
        _W2 = tensor.matrix()
        _h0 = tensor.vector()

        W1 = tensor.specify_shape(_W1, (3, 3))
        W2 = tensor.specify_shape(_W2, (3, 3))
        h0 = tensor.specify_shape(_h0, (3,))

        def lambda_fn(W1, h, W2):
            return W1 * tensor.dot(h, W2)

        o, _ = theano.scan(lambda_fn,
                           sequences=tensor.zeros_like(W1),
                           outputs_info=h0,
                           non_sequences=[tensor.zeros_like(W2)],
                           n_steps=5)

        f = theano.function([_h0, _W1, _W2], o, mode=mode_with_opt)
        scan_node = [x for x in f.maker.fgraph.toposort()
                     if isinstance(x.op,
                                   theano.scan_module.scan_op.Scan)][0]

        assert len(scan_node.op.inputs) == 1

    def test_while0(self):
        x = tensor.vector('x')

        def lambda_fn(x_t):
            return x_t + 1, theano.scan_module.until(x_t > 3)
        o, _ = theano.scan(lambda_fn, x)
        f = theano.function([x], o)
        vx = numpy.zeros((50,), dtype=theano.config.floatX)
        vx[23] = 4
        out = f(vx)
        assert len(out) == 24

    def test_while1(self):
        x = tensor.vector('x')

        def lambda_fn(x_t):
            return x_t + 1, theano.scan_module.until(x_t > 3)
        o, _ = theano.scan(lambda_fn, x)
        o2, _ = theano.scan(lambda x_t: x_t + 2, x)

        f = theano.function([x], [o, o2], mode=mode_with_opt)
        vx = numpy.zeros((50,), dtype=theano.config.floatX)
        vx[23] = 4
        out, out2 = f(vx)
        assert len(out) == 24
        assert numpy.all(out2 == vx + 2)
        lssc = [x for x in f.maker.fgraph.toposort()
                if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        # One scan node gets optimnized out
        assert len(lssc) == 1

    @dec.skipif(True,
                        ("This test fails because not typed outputs_info "
                         "are always gived the smallest dtype. There is "
                         "no upcast of outputs_info in scan for now."))
    def test_outputs_info_not_typed(self):
        # This was ticket 766

        coefficients = theano.tensor.vector("coefficients")
        x = tensor.scalar("x")
        max_coefficients_supported = 10000

        # Generate the components of the polynomial
        full_range = theano.tensor.arange(max_coefficients_supported)
        components, updates = theano.scan(
            fn=lambda coeff, power, free_var: coeff * (free_var ** power),
            sequences=[coefficients, full_range],
            non_sequences=x)
        polynomial1 = components.sum()
        polynomial2, updates = theano.scan(
            fn=lambda coeff, power, prev, free_var: \
                            prev + coeff * (free_var ** power),
            outputs_info=theano.tensor.constant(0, dtype='floatX'),
            sequences=[coefficients, full_range],
            non_sequences=x)

        # python int
        polynomial3, updates = theano.scan(
            fn=lambda coeff, power, prev, free_var: \
                            prev + coeff * (free_var ** power),
            outputs_info=0,
            sequences=[coefficients, full_range],
            non_sequences=x)

        # python float
        polynomial4, updates = theano.scan(
            fn=lambda coeff, power, prev, free_var: \
                            prev + coeff * (free_var ** power),
            outputs_info=0.,
            sequences=[coefficients, full_range],
            non_sequences=x)

        calculate_polynomial = theano.function(
            inputs=[coefficients, x],
            outputs=[polynomial1,
                     polynomial2[-1],
                     polynomial3[-1],
                     polynomial4[-1]])

        test_coeff = numpy.asarray([1, 0, 2], dtype=theano.config.floatX)
        # This will be tested by DEBUG_MODE
        out = calculate_polynomial(test_coeff, 3)
        assert out[0] == 19
        assert out[1] == 19
        assert out[2] == 19
        assert out[4] == 19
        # 19.0

    def test_crash_nonseq_grad(self):
        # Test case was originally reported by Bitton Tenessi. It crashed
        # during the grad operation and this tests validates that it now
        # raises a NullTypeGradError instead because the gradient relies on
        # the intermediary states of the random number generators used in the
        # test. The test case was modified from the original for simplicity

        rand_stream = tensor.shared_randomstreams.RandomStreams()
        inp = tensor.matrix()
        norm_inp = inp / tensor.sum(inp, axis=0)

        def unit_dropout(out_idx):
            def stochastic_pooling(in_idx):
                # sample the input matrix for each column according to the
                # column values
                pvals = norm_inp.T
                sample = rand_stream.multinomial(n=1, pvals=pvals)
                return inp + sample

            pooled, updates_inner = theano.scan(fn=stochastic_pooling,
                                        sequences=tensor.arange(inp.shape[0]))

            # randomly add stuff to units
            rand_nums = rand_stream.binomial(size=pooled.shape)
            return pooled + rand_nums, updates_inner

        out, updates_outer = theano.scan(unit_dropout,
                                     sequences=[tensor.arange(inp.shape[0])])

        assert_raises(theano.gradient.NullTypeGradError,
                      tensor.grad, out.sum(), inp)

    def test_bugFunctioProvidesIntermediateNodesAsInputs(self):
        # This is a bug recently reported by Ilya
        # made it CPU friendly
        V = tensor.ftensor3('INPUT')
        orig = tensor.fmatrix('PARAM')
        # = gpu_from_host(orig)  # <-- this doesn't work
        W = orig + 2  # <-- has same effect but it works on CPU as well
        # W = T.fmatrix('PARAM') # <-- this line works

        def one_step(v, W):
            o = v + 1 + W.sum()  # <-- this doesn't work
            # o = v + 1  # <-- this line works
            return o

        OS, updates = theano.scan(
            fn=one_step,
            sequences=V,
            outputs_info=[None],
            non_sequences=[W])

        O = OS.sum() + W.sum()

        # This bug manifests itself by not allowing the function to compile,
        # so if it compiles it means the test pass
        f = theano.function([V, W], O)

    def test_while2(self):
        x = tensor.vector('x')

        def lambda_fn(x_t):
            return x_t + 1, theano.scan_module.until(x_t > 3)
        o, _ = theano.scan(lambda_fn, x)
        o2, _ = theano.scan(lambda x_t: (x_t + 2,
                                         theano.scan_module.until(x_t > 3)),
                            x)

        f = theano.function([x], [o, o2], mode=mode_with_opt)
        vx = numpy.zeros((50,), dtype=theano.config.floatX)
        vx[23] = 4
        out, out2 = f(vx)
        assert len(out) == 24
        assert len(out2) == 24
        lssc = [x for x in f.maker.fgraph.toposort()
                if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        assert len(lssc) == 1

    def test_while_infershape(self):
        x = tensor.vector('x')

        def lambda_fn(x_t):
            return x_t + 1, theano.scan_module.until(x_t > 3)
        o, _ = theano.scan(lambda_fn, x)

        f = theano.function([x], o.shape[0], mode=mode_with_opt)
        vx = numpy.zeros((50,), dtype=theano.config.floatX)
        vx[23] = 4
        out = f(vx)
        assert out == 24

    def test_infershape_seq_shorter_nsteps(self):
        raise SkipTest("This is a generic problem with "
                       "infershape that has to be discussed "
                       "and figured out")
        x = tensor.vector('x')
        [o1, o2], _ = theano.scan(lambda x, y: (x + 1, y + x),
                         sequences=x,
                         outputs_info=[None, x[0]],
                         n_steps=20)

        f = theano.function([x],
                            [o1.shape[0], o2.shape[0]],
                            mode=mode_with_opt)

        vx = numpy.ones((10,), dtype=theano.config.floatX)
        out1, out2 = f(vx)
        assert out1 == 10
        assert out2 == 10
        lssc = [x for x in f.maker.fgraph.toposort()
                if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        assert len(lssc) == 0

    def test_infershape_nsteps_smaller_seq_length(self):
        x = tensor.vector('x')
        [o1, o2], _ = theano.scan(lambda x, y: (x + 1, y + x),
                         sequences=x,
                         outputs_info=[None, x[0]],
                         n_steps=20)

        f = theano.function([x],
                            [o1.shape[0], o2.shape[0]],
                            mode=mode_with_opt)

        vx = numpy.ones((30,), dtype=theano.config.floatX)
        o1, o2 = f(vx)
        assert o1 == 20
        assert o2 == 20
        lssc = [x for x in f.maker.fgraph.toposort()
                if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        assert len(lssc) == 0

    def test_oinp_iinp_iout_oout_mappings(self):
        # Test the mapping produces by
        # ScanOp.get_oinp_iinp_iout_oout_mappings()

        rng = theano.tensor.shared_randomstreams.RandomStreams(123)

        def inner_fct(seq, mitsot, sitsot, nitsot, nseq):
            random_scalar = rng.uniform((1,))[0]
            total = seq + mitsot + sitsot + nitsot + nseq + random_scalar
            return total, total, total

        # Assemble a scan with one sequence, one mitsot, one sitsot, one nitsot
        # a non-sequence and a random state to test the mappings.
        seq = [tensor.vector()]
        non_seq = [tensor.scalar()]
        outputs_info = [dict(initial=tensor.vector(), taps=[-3, -1]),
                        tensor.scalar(), None]

        scan_outputs, _ = theano.scan(fn=inner_fct, sequences=seq,
                                      outputs_info=outputs_info,
                                      non_sequences=non_seq)

        # Compare the mappings with the expected values
        scan_node = scan_outputs[0].owner.inputs[0].owner
        mappings = scan_node.op.var_mappings

        assert mappings['inner_inp_from_outer_inp'] == {0 : [], 1 : [0],
                                                        2 : [1, 2], 3 : [3],
                                                        4 : [4], 5 : [],
                                                        6 : [5]}
        assert mappings['inner_out_from_outer_inp'] == {0 : [], 1 : [],
                                                        2 : [0], 3 : [1],
                                                        4 : [3], 5 : [2],
                                                        6 : []}
        assert mappings['outer_out_from_outer_inp'] == {0 : -1, 1 : -1,
                                                        2 : 0, 3 : 1,
                                                        4 : 3, 5 : 2,
                                                        6 : -1}

        assert mappings['outer_inp_from_inner_inp'] == {0 : 1, 1 : 2,
                                                        2 : 2, 3 : 3,
                                                        4 : 4, 5 : 6}
        assert mappings['inner_out_from_inner_inp'] == {0 : [], 1 : [0],
                                                        2 : [0], 3 : [1],
                                                        4 : [3], 5 : []}
        assert mappings['outer_out_from_inner_inp'] == {0 : -1, 1 : 0,
                                                        2 : 0, 3 : 1,
                                                        4 : 3, 5 : -1}

        assert mappings['outer_inp_from_inner_out'] == {0 : 2, 1 : 3,
                                                        2 : 5, 3 : 4}
        assert mappings['inner_inp_from_inner_out'] == {0 : [1, 2], 1 : [3],
                                                        2 : [], 3 : [4]}
        assert mappings['outer_out_from_inner_out'] == {0 : 0, 1 : 1,
                                                        2 : 2, 3 : 3}

        assert mappings['outer_inp_from_outer_out'] == {0 : 2, 1 : 3,
                                                        2 : 5, 3 : 4}
        assert mappings['inner_inp_from_outer_out'] == {0 : [1, 2], 1 : [3],
                                                        2 : [], 3 : [4]}
        assert mappings['inner_out_from_outer_out'] == {0 : [0], 1 : [1],
                                                        2 : [2], 3 : [3]}

    def test_grad_duplicate_outputs(self):
        # This test validates that taking the gradient of a scan, in which
        # multiple outputs are the same theano variable, works.

        def inner_fct(inp1, inp2, inp3):
            total = inp1 + inp2 + inp3
            return total, total

        # Assemble the scan
        seq = tensor.matrix()
        out_init = tensor.matrix()
        non_seq = tensor.vector()

        outputs_info = ([None, dict(initial=out_init, taps=[-3])])

        scan_outputs, _ = theano.scan(fn=inner_fct, sequences=seq,
                                      outputs_info=outputs_info,
                                      non_sequences=non_seq)

        # Attempt to take various gradients
        g_output0 = theano.grad(scan_outputs[0].sum(), [seq, out_init, non_seq])
        g_output1 = theano.grad(scan_outputs[1].sum(), [seq, out_init, non_seq])

        # Compile the function
        fct = theano.function([seq, out_init, non_seq],
                              g_output0 + g_output1)

        # Run the function and validate the outputs
        dtype = theano.config.floatX
        seq_value = numpy.random.random((10, 3)).astype(dtype)
        out_init_value = numpy.random.random((3, 3)).astype(dtype)
        non_seq_value = numpy.random.random((3)).astype(dtype)

        outputs =  fct(seq_value, out_init_value, non_seq_value)

        expected_g_seq = numpy.array([[4, 4, 4],
                                      [3, 3, 3],
                                      [3, 3, 3],
                                      [3, 3, 3],
                                      [2, 2, 2],
                                      [2, 2, 2],
                                      [2, 2, 2],
                                      [1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]])
        expected_g_out_init = expected_g_seq[:3]
        expected_g_non_seq = numpy.array([22, 22, 22])

        utt.assert_allclose(outputs[0], expected_g_seq)
        utt.assert_allclose(outputs[1], expected_g_out_init)
        utt.assert_allclose(outputs[2], expected_g_non_seq)
        utt.assert_allclose(outputs[3], expected_g_seq)
        utt.assert_allclose(outputs[4], expected_g_out_init)
        utt.assert_allclose(outputs[5], expected_g_non_seq)

    def test_grad_duplicate_outputs_connection_pattern(self):
        # This test checks for a crash in scan.connection_pattern when taking
        # the grad of a scan with certain combinations of outputs.

        def inner_fct(inp1, inp2, inp3, inp4, inp5, inp6):
            total = inp1 + inp2 + inp3 + inp4 + inp5 + inp6
            return total, total, total, total, total, total

        # Assemble the scan
        out_init = [tensor.vector(), tensor.vector(),
                    tensor.matrix(), tensor.matrix()]

        outputs_info = ([None, None, out_init[0], out_init[1],
                        dict(initial=out_init[2], taps=[-2, -1]),
                        dict(initial=out_init[3], taps=[-2, -1])])

        scan_outputs, _ = theano.scan(fn=inner_fct, outputs_info=outputs_info,
                                      n_steps=10)

        g_output0 = theano.grad(scan_outputs[0].sum(), out_init[1])

        # Validate the connnection pattern is as it should be
        node = scan_outputs[0].owner
        connection_pattern = node.op.connection_pattern(node)
        expected_connection_pattern = [[(j in [1, 2, 3, 4]) for i in range(6)]
                                       for j in range(7)]

        assert connection_pattern == expected_connection_pattern

    def test_grad_multiple_seqs_different_nsteps(self):
        # Example provided Michael Forbes
        # This test assures that we clip the sequences to n_steps before
        # computing the gradient (so that when we reverse them we actually
        # get the right values in
        c = theano.tensor.vector('c')
        x = theano.tensor.scalar('x')
        _max_coefficients_supported = 1000
        full_range = theano.tensor.arange(_max_coefficients_supported)
        components, updates = theano.scan(
            fn=lambda coeff, power, free_var: coeff * (free_var ** power),
            outputs_info=None,
            sequences=[c, full_range],
            non_sequences=x)
        P = components.sum()
        dP = theano.tensor.grad(P, x)
        tf = theano.function([c, x], dP)
        assert tf([1.0, 2.0, -3.0, 4.0], 2.0) == 38

    def test_grad_of_grad_of_state(self):
        # Example provided Michael Forbes
        # This tests ensures that we can compute gradients through cost
        # defines in terms of gradients of scan
        c = theano.tensor.vector('c')
        x = theano.tensor.scalar('x')
        _max_coefficients_supported = 1000
        full_range = theano.tensor.arange(_max_coefficients_supported)
        components, updates = theano.scan(
            fn=lambda coeff, power, free_var: coeff * (free_var ** power),
            outputs_info=None,
            sequences=[c, full_range],
            non_sequences=x)
        P = components.sum()
        dP = theano.tensor.grad(P, x).sum()
        ddP = theano.tensor.grad(dP, x)
        tf = theano.function([c, x], ddP)
        assert tf([1.0, 2.0, -3.0, 4.0], 2.0) == 42

    def test_return_steps(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        vW = asarrayX(rng.uniform(size=(2, 2), low=-5., high=5.))
        vWout = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        vW_in1 = asarrayX(rng.uniform(size=(2, 2), low=-5., high=5.))
        v_u1 = asarrayX(rng.uniform(size=(8, 2), low=-5., high=5.))
        v_u2 = asarrayX(rng.uniform(size=(8,), low=-5., high=5.))
        v_x0 = asarrayX(rng.uniform(size=(2,), low=-5., high=5.))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = theano.shared(vW_in2, name='win2')
        W = theano.shared(vW, name='w')
        W_out = theano.shared(vWout, name='wout')
        W_in1 = theano.tensor.matrix('win')
        u1 = theano.tensor.matrix('u1')
        u2 = theano.tensor.vector('u2')
        x0 = theano.tensor.vector('x0')
        y0 = theano.tensor.vector('y0')

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, y_tm3, W_in1):
            return [y_tm3 + 1,
                    theano.dot(u1_t, W_in1) + u2_t * W_in2 + \
                        theano.dot(x_tm1, W),
                    y_tm1 + theano.dot(x_tm1, W_out)]

        rval, updates = theano.scan(f_rnn_cmpl,
                                    [u1, u2],
                                    [None,
                                     dict(initial=x0),
                                     dict(initial=y0, taps=[-1, -3])],
                                    W_in1,
                                    n_steps=None,
                                    truncate_gradient=-1,
                                    go_backwards=False)

        outputs = []
        outputs += [rval[0][-3:]]
        outputs += [rval[1][-2:]]
        outputs += [rval[2][-4:]]
        f4 = theano.function([u1, u2, x0, y0, W_in1],
                             outputs,
                             updates=updates,
                             allow_input_downcast=True)

        # compute the values in numpy
        v_x = numpy.zeros((8, 2), dtype=theano.config.floatX)
        v_y = numpy.zeros((8,), dtype=theano.config.floatX)
        v_x[0] = numpy.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + \
                    numpy.dot(v_x0, vW)
        v_y[0] = numpy.dot(v_x0, vWout) + v_y0[2]

        for i in xrange(1, 8):
            v_x[i] = numpy.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + \
                        numpy.dot(v_x[i - 1], vW)
            v_y[i] = numpy.dot(v_x[i - 1], vWout) + v_y[i - 1]

        (theano_dump, theano_x, theano_y) = f4(v_u1, v_u2, v_x0, v_y0, vW_in1)

        utt.assert_allclose(theano_x, v_x[-2:])
        utt.assert_allclose(theano_y, v_y[-4:])

    def test_opt_order(self):
        """
        Verify that scan optimizations are applied before blas
        optimizations.
        This is needed as otherwise, the dot won't become a dot22
        so it will be slower and won't get transferred to the gpu.
        """
        x = theano.tensor.matrix('x')
        A = theano.tensor.matrix('A')

        z, updates = theano.scan(
            theano.dot,
            sequences=[],
            non_sequences=[x, A],
            n_steps=2)
        f = theano.function([x, A], z)
        topo = f.maker.fgraph.toposort()
        if theano.config.mode != "FAST_COMPILE":
            assert any([isinstance(node.op, tensor.blas.Dot22)
                        for node in topo])

        vx = numpy.array([[1., 1.], [2., 2.]], dtype=theano.config.floatX)
        vA = numpy.array([[1., 1.], [1., 0.]], dtype=theano.config.floatX)
        vR = numpy.array([[[2, 1], [4, 2]], [[2, 1], [4, 2]]],
                         dtype=theano.config.floatX)
        utt.assert_allclose(f(vx, vA), vR)

    def test_savemem_opt(self):
        y0 = theano.shared(numpy.ones((2, 10)))
        [y1, y2], updates = theano.scan(lambda y: [y, y],
                                         outputs_info=[dict(initial=y0,
                                                            taps=[-2]), None],
                                        n_steps=5)
        rval = theano.function([], y2.sum())()

    def test_savemem_opt_0_step(self):
        # Test a case where the savemem optimization has the opportunity to
        # lower the number of steps of a Scan to 0. It tests that the
        # optimization doesn't do so since Scan nodes with 0
        # steps are not currently supported and doing so would result in a
        # crash during the function execution.

        def inner_scan_step(x_t_t, h_tm1, w):
            return tensor.dot(h_tm1, w) + x_t_t

        def outer_scan_step(x_t, w):
            h, _ = theano.scan(inner_scan_step,
                            sequences=[x_t[1:]],
                            outputs_info=[x_t[0]],
                            non_sequences=[w],
                            strict=True,
                            name="the_inner_scan")
            return h

        def get_outputs(x, w):
            features, _ = theano.scan(outer_scan_step,
                                    sequences=[x],
                                    non_sequences=[w],
                                    strict=True,
                                    name="the_outer_scan")

            return_val =  tensor.grad(features.sum(), w)
            return return_val

        # Compile the theano function
        x = tensor.tensor3('x')
        w = tensor.matrix('w')
        f = theano.function(inputs=[x, w], outputs=get_outputs(x, w))

        # Test the function to ensure it returns valid results
        x_value = numpy.random.random((2, 2, 3)).astype(theano.config.floatX)
        w_value = numpy.random.random((3, 3)).astype(theano.config.floatX)
        expected_output = numpy.tile(x_value[:, 0].sum(0), (3, 1)).transpose()

        output = f(x_value, w_value)
        utt.assert_allclose(output, expected_output)


    def test_grad_multiple_taps_state(self):
        # The test is based on the code provided by Timothy Lillicrap

        def onestep(xdl, xprev, w):
            xnew = w + xprev
            return xnew

        xinit = tensor.tensor3('xinit')
        w = tensor.matrix('w')
        (xseq, updates) = theano.scan(
            n_steps=10,
            fn=onestep,
            outputs_info=[dict(initial=xinit, taps=[-4, -1])],
            non_sequences=w)
        loss = (xseq[-1] ** 2).sum()
        cost_fn = theano.function([xinit, w],
                                  loss,
                                  no_default_updates=True,
                                  allow_input_downcast=True)

        gw, gx = tensor.grad(loss, [w, xinit])
        grad_fn = theano.function([xinit, w], [gx, gw],
                                 allow_input_downcast=True)
        rng = numpy.random.RandomState(utt.fetch_seed())
        # If numbers are small, the gradients with respect to x are small
        # and the numeric differentiation becomes unstable.
        # To fix this issue I ensure we are sampling numbers larger in
        # absolute value than 1.
        v_x = numpy.array(rng.uniform(size=(5, 2, 2), low=1., high=3.),
                           dtype=theano.config.floatX)
        # Making some entries to be negative.
        pos = rng.uniform(size=(5, 2, 2), low=0., high=1) < .5
        v_x[pos] = -1 * v_x[pos]
        v_w = numpy.array(rng.uniform(size=(2, 2), low=1., high=3.),
                          dtype=theano.config.floatX)
        pos = rng.uniform(size=(2, 2), low=0., high=1.) < .5
        v_w[pos] = -1 * v_w[pos]
        analytic_grad = grad_fn(v_x, v_w)
        num_grad = multiple_outputs_numeric_grad(cost_fn,
                                                 [v_x, v_w])
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        if max_err > 1e-2:
            raise Exception(theano.tensor.verify_grad.E_grad,
                            (max_err, 1e-2, max_err_pos,
                             analytic_grad[max_err_pos],
                             num_grad.gx[max_err_pos]))

    def test_grad_numeric_shared(self):
        shared_var = theano.shared(numpy.float32(1.))

        def inner_fn():
            return [], OrderedDict(
                [(shared_var, shared_var + numpy.float32(1.))])
        _, updates = theano.scan(inner_fn,
                                 n_steps=10,
                                 truncate_gradient=-1,
                                 go_backwards=False)
        cost = list(updates.values())[0]
        g_sh = tensor.grad(cost, shared_var)
        fgrad = theano.function([], g_sh)
        assert fgrad() == 1

    def test_rop_mitmot(self):
        # this test is a copy paste from the script given by Justin Bayer to
        # reproduce this bug
        # We have 2 parameter groups with the following shapes.
        W1shape = (1, 3)
        W2shape = (3, 3)

        n_pars = 1 * 3 + 3 * 3

        # Allocate big parameter array.
        pars = theano.shared(numpy.empty(n_pars))

        # Assign slices.
        W1 = pars[:3].reshape(W1shape)
        W2 = pars[3:].reshape(W2shape)

        # Define recurrent model. We are using a model where each input is a
        # tensor
        # of shape (T, B, D) where T is the number of timesteps, B is the
        # number of
        # sequences iterated over in parallel and D is the dimensionality of
        # each
        # item at a timestep.

        inpt = tensor.tensor3('inpt')
        target = tensor.tensor3('target')

        # Make these flat in order to be able to use dot products instead of
        # tensordot,
        # which is slower.
        inpt_flat = inpt.reshape((inpt.shape[0] * inpt.shape[1],
                                  inpt.shape[2]))
        hidden_flat = tensor.dot(inpt_flat, W1)
        hidden = hidden_flat.reshape((inpt.shape[0], inpt.shape[1], 3))

        transfer = tensor.nnet.sigmoid

        hidden_rec, _ = theano.scan(
                lambda x, h_tm1: transfer(tensor.dot(h_tm1, W2) + x),
                sequences=hidden,
                outputs_info=[tensor.zeros_like(hidden[0])])

        hidden_rec_flat = hidden_rec.reshape(
                    (hidden_rec.shape[0] * hidden_rec.shape[1],
                     hidden_rec.shape[2]))

        cost = ((hidden_rec - target) ** 2).mean()
        d_cost_wrt_pars = tensor.grad(cost, pars)

        p = tensor.dvector()
        Hp = tensor.Rop(d_cost_wrt_pars, pars, p)

    def test_seq_tap_bug_jeremiah(self):
        inp = numpy.arange(10).reshape(-1, 1).astype(theano.config.floatX)
        exp_out = numpy.zeros((10, 1)).astype(theano.config.floatX)
        exp_out[4:] = inp[:-4]

        def onestep(x, x_tm4):
            return x, x_tm4

        seq = tensor.matrix()
        initial_value = theano.shared(numpy.zeros((4, 1),
                                                  dtype=theano.config.floatX))
        outputs_info = [OrderedDict(
            [('initial', initial_value), ('taps', [-4])]), None]
        results, updates = theano.scan(fn=onestep,
                                       sequences=seq,
                                       outputs_info=outputs_info)

        f = theano.function([seq], results[1])
        assert numpy.all(exp_out == f(inp))

    def test_borrow_bug_jeremiah(self):
        # This tests two things. The first is a bug occuring when scan wrongly
        # used the borrow flag. The second thing it that Scan's infer_shape()
        # method will be able to remove the Scan node from the graph in this
        # case.

        inp = numpy.arange(10).reshape(-1, 1).astype(theano.config.floatX)
        exp_out = numpy.zeros((10, 1)).astype(theano.config.floatX)
        exp_out[4:] = inp[:-4]

        def onestep(x, x_tm4):
            return x, x_tm4

        seq = tensor.matrix()
        initial_value = theano.shared(numpy.zeros((4, 1),
                                                  dtype=theano.config.floatX))
        outputs_info = [OrderedDict([('initial', initial_value),
                                     ('taps', [-4])]), None]
        results, _ = theano.scan(fn=onestep,
                                       sequences=seq,
                                       outputs_info=outputs_info)
        sharedvar = theano.shared(numpy.zeros((1, 1),
                                              dtype=theano.config.floatX))
        updates = OrderedDict([(sharedvar, results[0][-1:])])

        f = theano.function([seq], results[1], updates=updates)

        # This fails if scan uses wrongly the borrow flag
        assert numpy.all(exp_out == f(inp))

        # This fails if Scan's infer_shape() is unable to remove the Scan
        # node from the graph.
        f_infershape = theano.function([seq], results[1].shape,
                                       mode='FAST_RUN')
        scan_nodes_infershape = scan_nodes_from_fct(f_infershape)
        assert(len(scan_nodes_infershape) == 0)

    def test_memory_reuse_with_outputs_as_inputs(self):
        # Test the memory pre-allocation feature in scan for the following
        # cases :
        #  - An output of the inner graph is also an input of the inner graph
        #  - An output of the inner graph is not an input in the unoptimized
        #    graph but it could becomes the case in the optimized graph due to
        #    the optimizations.
        #  - An output of the inner graph is obtained through a view op on an
        #    input of the inner graph and the view op is removed by the
        #    optimization process
        #  - An output of the inner graph is obtained through a view op on an
        #    input of the inner graph and the view op is NOT removed by the
        #    optimization process
        #  - An output of the inner graph is not obtained through any of the
        #    previously mentionned cases (standard case)

        def inner_fn(tap_m3, tap_m2, tap_m1):
            return (tap_m2, (tap_m1 * 1),
                    theano.gradient.disconnected_grad(tap_m2),
                    theano.tensor.opt.assert_(tap_m2, 1),
                    tap_m3 + tap_m2 + tap_m1)

        init = theano.tensor.matrix()
        outputs_info = [None, None, None, None,
                        dict(initial=init, taps=[-3, -2, -1])]

        out, _ = theano.scan(inner_fn, outputs_info=outputs_info, n_steps=3)
        fct = theano.function([init], out)

        # Compare obtained outputs with expected outputs
        floatX = theano.config.floatX
        outputs = fct(numpy.arange(9, dtype=floatX).reshape(3,3))

        states = numpy.array([[0, 1, 2],
                              [3, 4, 5],
                              [6, 7, 8],
                              [9, 12, 15],
                              [18, 23, 28],
                              [33, 42, 51]],dtype=floatX)
        expected_outputs = [states[1:4], states[2:5], states[1:4],
                            states[1:4], states[3:6]]

        utt.assert_allclose(outputs, expected_outputs)

    def test_grad_connectivity_matrix(self):
        def inner_fn(x_tm1, y_tm1, z_tm1):
            x_tm1.name = 'x'
            y_tm1.name = 'y'
            z_tm1.name = 'z'
            return x_tm1 ** 2, y_tm1, x_tm1 + 1
        x0 = tensor.vector('X')
        y0 = tensor.vector('y0')
        z0 = tensor.vector('Z')
        [x, y, z], _ = theano.scan(inner_fn,
                                 outputs_info=[x0, y0, z0],
                                 n_steps=10)
        cost = (x + y + z).sum()

        gx0 = tensor.grad(cost, x0)  # defined
        gy0 = tensor.grad(cost, y0)  # defined
        self.assertRaises(ValueError, tensor.grad, cost, z0)
        cost = x.sum()
        self.assertRaises(ValueError, tensor.grad, cost, y0)

    def test_disconnected_gradient(self):
        v = tensor.vector('v')
        m = tensor.matrix('m')
        u0 = tensor.zeros((7,))

        [u, m2], _ = theano.scan(lambda _, u: [u, v],
                                 sequences=m,
                                 outputs_info=[u0, None])
        # This used to raise an exception with older versions becasue for a
        # disconnected gradient a non disconnected type was returned
        tensor.grad((m * m2).sum(), v)

    def test_disconnected_gradient2(self):
        v = tensor.vector('v')
        m = tensor.matrix('m')
        u0 = tensor.zeros((7,))

        [u, m2], _ = theano.scan(lambda x, u: [x+u, u+v],
                                 sequences=m,
                                 outputs_info=[u0, None])
        # This used to raise an exception with older versions becasue
        # scan could not detect the connection between `m2` and `x`
        tensor.grad(m2.sum(), m)

    def test_disconnected_gradient3(self):
        # This tests for a crash that would occur sometimes when taking the
        # gradient through a scan with a non-recurrent output which would
        # receive a disconnected gradient

        v = tensor.dvector('v')

        def step(seq):
            out1 = seq + 1
            out2 = out1 + 1
            return out1, out2

        [out1, out2], _ = theano.scan(step, sequences=v)
        gv = tensor.grad(out2.sum(), [v])
        f = theano.function([v], gv)

        # Ensure the output of the function is valid
        output = f(numpy.random.random(5))
        utt.assert_allclose(output, numpy.ones(5))

    def test_dot_optimization(self):
        A = tensor.matrix('A')
        B = tensor.matrix('B')
        S, _ = theano.scan(lambda x1, x2, u: u + tensor.dot(x1, x2),
                           sequences=[A.dimshuffle(0, 1, 'x'),
                                        B.dimshuffle(0, 'x', 1)],
                           outputs_info=[tensor.zeros_like(A)])
        f = theano.function([A, B], S.owner.inputs[0][-1])
        rng = numpy.random.RandomState(utt.fetch_seed())
        vA = rng.uniform(size=(5, 5)).astype(theano.config.floatX)
        vB = rng.uniform(size=(5, 5)).astype(theano.config.floatX)
        utt.assert_allclose(f(vA, vB), numpy.dot(vA.T, vB))

    def test_pregreedy_optimizer(self):
        W = tensor.zeros((5, 4))
        bv = tensor.zeros((5,))
        bh = tensor.zeros((4,))
        v = tensor.matrix('v')
        (bv_t, bh_t), _ = theano.scan(lambda _: [bv, bh], sequences=v,
                                      outputs_info=[None, None])
        chain, _ = theano.scan(
            lambda x: tensor.dot(tensor.dot(x, W) + bh_t, W.T) + bv_t,
            outputs_info=v,
            n_steps=2)
        theano.function([v], chain)(numpy.zeros((3, 5),
                                                dtype=theano.config.floatX))

    def test_savemem_does_not_duplicate_number_of_scan_nodes(self):
        var = tensor.ones(())
        values, _ = theano.scan(lambda x: ([x], (),
                                           theano.scan_module.until(x)),
                                outputs_info=[var], n_steps=2)

        tmp_fn = theano.function([var], values)
        scan_nodes = [x for x in tmp_fn.maker.fgraph.toposort()
                      if isinstance(x.op,
                                    theano.scan_module.scan_op.Scan)]
        assert len(scan_nodes) == 1

    def test_eliminate_seqs(self):
        U = tensor.vector('U')
        sh = theano.shared(asarrayX(2.))
        x1 = tensor.vector('x1')
        x2 = tensor.scalar('x2')

        def rec_fn(*args):
            u_t = args[0]
            return [(u_t + 1,  # mitsot
                     u_t + 2,  # sitsot
                     u_t + 3),  # nitsot
                    {sh: u_t + 4}]  # shared

        [X1, X2, X3], updates = theano.scan(
            rec_fn,
            U,
            [dict(initial=x1, taps=[-1, -3]), x2, None],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False)
        f = theano.function([U, x1, x2], [X1, X2, X3],
                            updates=updates,
                            mode=theano.Mode(linker='py'),
                            allow_input_downcast=True)
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_u = asarrayX(rng.uniform(size=(5,)))
        outs = f(v_u, [0, 0, 0], 0)
        utt.assert_allclose(outs[0], v_u + 1)
        utt.assert_allclose(outs[1], v_u + 2)
        utt.assert_allclose(outs[2], v_u + 3)
        utt.assert_allclose(sh.get_value(), v_u[-1] + 4)

    def test_eliminate_nonseqs(self):
        W = tensor.scalar('W')
        sh = theano.shared(asarrayX(2.))
        x1 = tensor.vector('x1')
        x2 = tensor.scalar('x2')

        def rec_fn(*args):
            w = args[-1]
            return [(w + 1.,  # mitsot
                     w + 2.,  # sitsot
                     w + 3.),  # nitsot
                    {sh: w + 4.}]  # shared

        [X1, X2, X3], updates = theano.scan(
            rec_fn,
            [],
            [dict(initial=x1, taps=[-1, -3]), x2, None],
            W,
            n_steps=5,
            truncate_gradient=-1,
            go_backwards=False)
        f = theano.function([W, x1, x2], [X1, X2, X3],
                            updates=updates,
                            mode=theano.Mode(linker='py'),
                            allow_input_downcast=True)
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_w = asarrayX(rng.uniform())
        outs = f(v_w, [0, 0, 0], 0)
        utt.assert_allclose(outs[0], v_w + 1)
        utt.assert_allclose(outs[1], v_w + 2)
        utt.assert_allclose(outs[2], v_w + 3)
        utt.assert_allclose(sh.get_value(), v_w + 4)

    def test_grad_bug_disconnected_input(self):
        W = theano.shared(numpy.zeros((3, 3)), name='W')
        v = theano.tensor.ivector(name='v')
        y, _ = theano.scan(lambda i, W: W[i], sequences=v, outputs_info=None, non_sequences=W)

        # This used to raise an exception
        f = theano.function([v], theano.tensor.grad(y.sum(), W))
        utt.assert_allclose(f([1, 2]), [[0, 0, 0], [1, 1, 1], [1, 1, 1]])

    def test_clone(self):
        def test(x, y, mention_y):
            if mention_y:
                d = 0.1 + 0 * y
            else:
                d = 0.1
            out = theano.clone(y, replace={x: x + d})
            # theano.printing.debugprint(out)
            return theano.function([], out)()

        x = theano.shared(numpy.asarray(0., dtype=theano.config.floatX))
        utt.assert_allclose(test(x, tensor.sum((x+1)**2), mention_y=False),
                              1.21000003815)
        utt.assert_allclose(test(x, tensor.sum((x+1)**2), mention_y=True),
                              1.21000003815)

    def test_grad_find_input(self):
        w = theano.shared(numpy.array(0, dtype='float32'), name='w')
        init = tensor.fscalar('init')

        out, _ = theano.scan(
                fn=lambda prev: w,
                outputs_info=init,
                n_steps=2,
        )
        tensor.grad(out[-1], w)

    def test_scan_merge_nodes(self):
        inps = tensor.vector()
        state = tensor.scalar()
        y1, _ = theano.scan(lambda x, y: x*y,
                            sequences=inps,
                            outputs_info=state,
                            n_steps=5)

        y2, _ = theano.scan(lambda x, y : (x+y, theano.scan_module.until(x > 0)),
                            sequences=inps,
                            outputs_info=state,
                            n_steps=5)
        scan_node1 = y1.owner.inputs[0].owner
        assert isinstance(scan_node1.op, theano.scan_module.scan_op.Scan)
        scan_node2 = y2.owner.inputs[0].owner
        assert isinstance(scan_node2.op, theano.scan_module.scan_op.Scan)
        opt_obj = theano.scan_module.scan_opt.ScanMerge()
        # Test the method belongs_to of this class. Specifically see if it
        # detects the two scan_nodes as not being similar
        assert not opt_obj.belongs_to_set(scan_node1, [scan_node2])
        assert not opt_obj.belongs_to_set(scan_node2, [scan_node1])

    def test_remove_constants_and_unused_inputs_scan_non_seqs(self):
        # Test the opt remove_constants_and_unused_inputs_scan for
        # non sequences.
        W = theano.tensor.matrix(name='W')
        v = theano.tensor.ivector(name='v')
        y1, _ = theano.scan(lambda i, W: W[i], sequences=v,
                            outputs_info=None, non_sequences=[W])
        y2, _ = theano.scan(lambda i, _, W: W[i], sequences=v,
                            outputs_info=None, non_sequences=[W[0], W])
        y3, _ = theano.scan(lambda i, W, _: W[i], sequences=v,
                            outputs_info=None, non_sequences=[W, W[0]])
        y4, _ = theano.scan(lambda i, _, _2, W: W[i], sequences=v,
                            outputs_info=None, non_sequences=[W[0], W[0], W])
        y5, _ = theano.scan(lambda i, _, W, _2: W[i], sequences=v,
                            outputs_info=None, non_sequences=[W[0], W, W[0]])
        y6, _ = theano.scan(lambda i, W, _, _2: W[i], sequences=v,
                            outputs_info=None, non_sequences=[W, W[0], W[0]])
        # TODO: y7 have problem during run time. I think it should
        # raise an error during the scan construction.
        # y7, _ = theano.scan(lambda i, W, _, _2: W[i], sequences=v,
        #                    outputs_info=None, non_sequences=[v, W[0], W])
        for out in [y1, y2, y3, y4, y5, y6]:
            # This used to raise an exception
            f = theano.function([W, v], out, mode=mode_with_opt)
            f(numpy.zeros((3, 3), dtype=theano.config.floatX), [1, 2])

            scan_nodes = scan_nodes_from_fct(f)
            assert len(scan_nodes) == 1
            scan_node = scan_nodes[0]

            # The first input is the number of iteration.
            assert (len(scan_node.inputs[1:]) ==
                    len(set(scan_node.inputs[1:])))
            inp = scan_node.op.inner_non_seqs(scan_node.op.inputs)
            assert len(inp) == 1
            assert (len(inp) == len(set(inp)))
            inp = scan_node.op.outer_non_seqs(scan_node)
            assert len(inp) == 1
            assert (len(inp) == len(set(inp)))

    def test_remove_constants_and_unused_inputs_scan_seqs(self):
        # Test the opt remove_constants_and_unused_inputs_scan for sequences.
        W = theano.tensor.matrix(name='W')
        v = theano.tensor.ivector(name='v')
        vv = theano.tensor.matrix(name='vv')
        y1, _ = theano.scan(lambda i, W: W[i], sequences=v,
                            outputs_info=None, non_sequences=[W])
        y2, _ = theano.scan(lambda i, _, W: W[i], sequences=[v, v],
                            outputs_info=None, non_sequences=W)
        y3, _ = theano.scan(lambda i, _, W: W[i], sequences=[v, vv[0]],
                            outputs_info=None, non_sequences=W)
        y4, _ = theano.scan(lambda _, i, W: W[i], sequences=[vv[0], v],
                            outputs_info=None, non_sequences=W)
        y5, _ = theano.scan(lambda _, i, _2, W: W[i], sequences=[vv, v, vv[0]],
                            outputs_info=None, non_sequences=W)
        y6, _ = theano.scan(lambda _, _2, i, W: W[i], sequences=[vv[0], vv, v],
                            outputs_info=None, non_sequences=W)
        y7, _ = theano.scan(lambda i, _, _2, W: W[i],
                            sequences=[v, vv[0], vv[0]],
                            outputs_info=None, non_sequences=W)
        y8, _ = theano.scan(lambda _, i, W, _2, _3: W[i], sequences=[vv[0], v],
                            outputs_info=None, non_sequences=[W, W[0], W[0]])
        for out in [y1, y2, y3, y4, y5, y6, y7, y8]:
            # This used to raise an exception
            f = theano.function([W, v, vv], out, on_unused_input='ignore',
                                mode=mode_with_opt)
            f(numpy.zeros((3, 3), theano.config.floatX),
              [1, 2],
              numpy.zeros((3, 3), theano.config.floatX))

            scan_nodes = scan_nodes_from_fct(f)
            assert len(scan_nodes) == 1
            scan_node = scan_nodes[0]

            # The first input is the number of iteration.
            assert (len(scan_node.inputs[1:]) ==
                    len(set(scan_node.inputs[1:])))
            inp = scan_node.op.inner_seqs(scan_node.op.inputs)
            assert len(inp) == 1
            inp = scan_node.op.outer_seqs(scan_node)
            assert len(inp) == 1
            inp = scan_node.op.inner_non_seqs(scan_node.op.inputs)
            assert len(inp) == 1
            inp = scan_node.op.outer_non_seqs(scan_node)
            assert len(inp) == 1

    @attr('slow')
    def test_hessian_bug_grad_grad_two_scans(self):
        # Bug reported by Bitton Tenessi
        # NOTE : The test to reproduce the bug reported by Bitton Tenessi
        # was modified from its original version to be faster to run.

        W = tensor.fvector(name='W')
        n_steps = tensor.iscalar(name='Nb_steps')

        def loss_outer(sum_outer, W):

            def loss_inner(sum_inner, W):

                return sum_inner + (W**2).sum()

            result_inner, _ = theano.scan(
                fn=loss_inner,
                outputs_info=tensor.as_tensor_variable(
                    numpy.asarray(0, dtype=numpy.float32)),
                non_sequences=[W],
                n_steps=1,
            )
            return sum_outer + result_inner[-1]

        result_outer, _ = theano.scan(
            fn=loss_outer,
            outputs_info=tensor.as_tensor_variable(
                numpy.asarray(0, dtype=numpy.float32)),
            non_sequences=[W],
            n_steps=n_steps,
        )

        cost = result_outer[-1]
        H = theano.gradient.hessian(cost, W)
        print(".", file=sys.stderr)
        f = theano.function([W, n_steps], H)
        f(numpy.ones((8,), dtype='float32'), 1)

    def test_strict_mode(self):
        n = 10

        w = numpy.array([[-1, 2], [3, -4]]).astype(theano.config.floatX)
        w_ = theano.shared(w)
        x0 = numpy.array([1, 2]).astype(theano.config.floatX)
        x0_ = tensor.vector(name='x0', dtype=theano.config.floatX)

        def _scan_loose(x):
            return tensor.dot(x, w_)

        def _scan_strict(x, w_ns):
            return tensor.dot(x, w_ns)

        ret_loose = theano.scan(_scan_loose,
                              sequences=[],
                              outputs_info=[x0_],
                              n_steps=n,
                              strict=False)
        f_loose = theano.function([x0_], ret_loose[0][-1])

        ret_strict = theano.scan(_scan_strict,
                               sequences=[],
                               outputs_info=[x0_],
                               non_sequences=[w_],
                               n_steps=n,
                               strict=True)
        f_strict = theano.function([x0_], ret_strict[0][-1])

        result_loose = f_loose(x0)
        result_strict = f_strict(x0)

        diff = (abs(result_loose - result_strict)).mean()

        assert diff <= type_eps[theano.config.floatX]

    @raises(theano.gof.fg.MissingInputError)
    def test_strict_mode_ex(self):
        n = 10

        w = numpy.array([[-1, 2], [3, -4]]).astype(theano.config.floatX)
        w_ = theano.shared(w)
        x0 = numpy.array([1, 2]).astype(theano.config.floatX)
        x0_ = tensor.vector(name='x0', dtype=theano.config.floatX)

        def _scan_loose(x):
            return tensor.dot(x, w_)

        ret_strict = theano.scan(_scan_loose,
                               sequences=[],
                               outputs_info=[x0_],
                               n_steps=n,
                               strict=True)

        f_strict = theano.function([x0_], ret_strict[0][-1])
        result_strict = f_strict(x0)

    def test_monitor_mode(self):
        # Test that it is possible to pass an instance of MonitorMode
        # to the inner function
        k = tensor.iscalar("k")
        A = tensor.vector("A")

        # Build a MonitorMode that counts how many values are greater than 10
        def detect_large_outputs(i, node, fn):
            for output in fn.outputs:
                if isinstance(output[0], numpy.ndarray):
                    detect_large_outputs.large_count += (output[0] > 10).sum()
        detect_large_outputs.large_count = 0

        mode = theano.compile.MonitorMode(post_func=detect_large_outputs)

        # Symbolic description of the result
        result, updates = theano.scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=tensor.ones_like(A),
            non_sequences=A,
            n_steps=k,
            mode=mode)

        final_result = result[-1]

        f = theano.function(inputs=[A, k],
                            outputs=final_result,
                            updates=updates)
        f(numpy.asarray([2, 3, .1, 0, 1], dtype=theano.config.floatX), 4)

        # There should be 3 outputs greater than 10: prior_result[0] at step 3,
        # and prior_result[1] at steps 2 and 3.
        if theano.config.mode in ["DEBUG_MODE", "DebugMode"]:
            # DebugMode will run all the intermediate nodes, so we
            # should expect a multiple of 3, not exactly 3.
            assert detect_large_outputs.large_count % 3 == 0

        else:
            assert detect_large_outputs.large_count == 3


class ScanGpuTests:
    """ This class defines a number of tests for Scan on GPU as well as a few
    helper functions for these tests. The GPU tests defined in this class are
    independant of the GPU backend used. Because of this, a class inheriting
    from ScanGpuTests should define the following attributes and methods to
    make the tests run on a specific backend :
    - self.gpu_backend : Reference to the backend module
    - self.mode_with_opt : Compilation mode to force usage of the gpu backend
    - self.is_scan_on_gpu(node) : Method to determine is a scan node has been
                                  moved to run on a gpu under the specific
                                  backend. Returns a boolean.
    """

    # as test_one_sequence_one_output_weights, but on the gpu
    # This first version test the first case in the optimizer to the gpu.
    def test_one_sequence_one_output_weights_gpu1(self):

        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.fvector('u')
        x0 = theano.tensor.fscalar('x0')
        W_in = theano.tensor.fscalar('win')
        W = theano.tensor.fscalar('w')

        # The following line is needed to have the first case being used
        # Otherwise, it is the second that is tested.
        mode = self.mode_with_gpu.excluding('InputToGpuOptimizer')
        output, updates = theano.scan(f_rnn,
                                      u,
                                      x0,
                                      [W_in, W],
                                      n_steps=None,
                                      truncate_gradient=-1,
                                      go_backwards=False,
                                      mode=mode)

        output = self.gpu_backend.gpu_from_host(output)
        f2 = theano.function([u, x0, W_in, W],
                             output,
                             updates=updates,
                             allow_input_downcast=True,
                             mode=self.mode_with_gpu)

        # get random initial values
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
        assert sum([isinstance(node.op, self.gpu_backend.HostFromGpu)
                    for node in topo]) == 0
        assert sum([isinstance(node.op, self.gpu_backend.GpuFromHost)
                    for node in topo]) == 4

        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any([isinstance(node.op, self.gpu_backend.GpuElemwise)
                    for node in scan_node_topo])
        assert not any([isinstance(node.op, self.gpu_backend.HostFromGpu)
                        for node in scan_node_topo])
        assert not any([isinstance(node.op, self.gpu_backend.GpuFromHost)
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
                                      mode=self.mode_with_gpu)

        f2 = theano.function([u, x0, W_in, W],
                             output,
                             updates=updates,
                             allow_input_downcast=True,
                             mode=self.mode_with_gpu)

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
        assert sum([isinstance(node.op, self.gpu_backend.HostFromGpu)
                    for node in topo]) == 1
        assert sum([isinstance(node.op, self.gpu_backend.GpuFromHost)
                    for node in topo]) == 4

        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any([isinstance(node.op, self.gpu_backend.GpuElemwise)
                    for node in scan_node_topo])
        assert not any([isinstance(node.op, self.gpu_backend.HostFromGpu)
                        for node in scan_node_topo])
        assert not any([isinstance(node.op, self.gpu_backend.GpuFromHost)
                        for node in scan_node_topo])

    # This third test checks that scan can deal with a mixture of dtypes as
    # outputs when is running on GPU
    def test_gpu3_mixture_dtype_outputs(self):

        def f_rnn(u_t, x_tm1, W_in, W):
            return (u_t * W_in + x_tm1 * W,
                    tensor.cast(u_t + x_tm1, 'int64'))

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
                                      mode=self.mode_with_gpu)

        f2 = theano.function([u, x0, W_in, W],
                             output,
                             updates=updates,
                             allow_input_downcast=True,
                             mode=self.mode_with_gpu)

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
        assert self.is_scan_on_gpu(scan_node)

    def test_gibbs_chain(self):
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
                                               mode=self.mode_with_gpu)
        my_f = theano.function([],
                               theano_vsamples[-1],
                               updates=updates,
                               allow_input_downcast=True,
                               mode=self.mode_with_gpu)

        # I leave this to tested by debugmode, this test was anyway more of
        # doest the graph compile kind of test
        t_result = my_f()

    def test_gpu_memory_usage(self):
        # This test validates that the memory usage of the defined theano
        # function is reasonnable when executed on the GPU. It checks for
        # a bug in which one of scan's optimization was not applied which
        # made the scan node compute large and unnecessary outputs which
        # brought memory usage on the GPU to ~12G.

        # Dimensionality of input and output data (not one-hot coded)
        n_in = 100
        n_out = 100
        # Number of neurons in hidden layer
        n_hid = 4000

        # Number of minibatches
        mb_size = 2
        # Time steps in minibatch
        mb_length = 200

        # Define input variables
        xin = tensor.ftensor3(name='xin')
        yout = tensor.ftensor3(name='yout')

        # Initialize the network parameters
        floatX = theano.config.floatX
        U = theano.shared(numpy.zeros((n_in, n_hid), dtype="float32"),
                        name='W_xin_to_l1')
        V = theano.shared(numpy.zeros((n_hid, n_hid), dtype="float32"),
                        name='W_l1_to_l1')
        W = theano.shared(numpy.zeros((n_hid, n_out), dtype="float32"),
                        name='W_l1_to_l2')
        nparams = [U, V, W]

        # Build the forward pass
        l1_base = tensor.dot(xin, U)

        def scan_l(baseline, last_step):
            return baseline + tensor.dot(last_step, V)

        zero_output = tensor.alloc(numpy.asarray(0., dtype="float32"),
                                   mb_size, n_hid)

        l1_out, _ = theano.scan(scan_l, sequences=[l1_base],
                                outputs_info=[zero_output],
                                mode=self.mode_with_gpu_nodebug)

        l2_out = tensor.dot(l1_out, W)

        # Compute the cost and take the gradient wrt params
        cost = tensor.sum((l2_out - yout) ** 2)
        grads = tensor.grad(cost, nparams)
        updates = list(zip(nparams, (n - g for n, g in zip(nparams, grads))))

        # Compile the theano function
        feval_backprop = theano.function([xin, yout], cost, updates=updates,
                                         mode=self.mode_with_gpu_nodebug)

        # Validate that the PushOutScanOutput optimization has been applied
        # by checking the number of outputs of the grad Scan node in the
        # compiled function.
        nodes = feval_backprop.maker.fgraph.toposort()
        scan_nodes = [n for n in nodes if isinstance(
                      n.op, theano.scan_module.scan_op.Scan)]

        # The grad scan is always the 2nd one according to toposort. If the
        # optimization has been applied, it has 2 outputs, otherwise 3.
        grad_scan_node = scan_nodes[1]
        assert len(grad_scan_node.outputs) == 2

        # Call the theano function to ensure the absence of a memory error
        feval_backprop(numpy.zeros((mb_length, mb_size, n_in),
                                   dtype="float32"),
                       numpy.zeros((mb_length, mb_size, n_out),
                                   dtype="float32"))

    def test_memory_reuse_gpudimshuffle(self):
        # Test the memory pre-allocation feature in scan when one output is
        # the result of a GpuDimshuffle (because an optimization in
        # GpuDimshuffle can cause issues with the memory pre-allocation
        # where it falsely thinks that a pre-allocated memory region has
        # been used when it hasn't).
        def inner_fn(seq1, recurrent_out):
            temp = seq1 + recurrent_out.sum()
            output1 = temp.dimshuffle(1, 0)
            output2 = temp.sum() + recurrent_out
            return output1, output2

        input1 = theano.tensor.ftensor3()
        init = theano.tensor.ftensor3()
        outputs_info = [None, init]

        out, _ = theano.scan(inner_fn, sequences=[input1],
                             outputs_info=outputs_info,
                             mode=self.mode_with_gpu)

        out1 = out[0].flatten()
        out2 = out[1].flatten()

        fct = theano.function([input1, init], [out1, out2],
                              mode=self.mode_with_gpu)

        output = fct(numpy.ones((2, 1, 1), dtype="float32"),
                     numpy.ones((1, 1, 1), dtype="float32"))

        expected_output = (numpy.array([2, 4], dtype="float32"),
                           numpy.array([3, 7], dtype="float32"))
        utt.assert_allclose(output, expected_output)


class T_Scan_Cuda(unittest.TestCase, ScanGpuTests):
    """This class takes the gpu tests for scan that are defined in
    class ScanGpuTests and runs them using the cuda backend. It also adds
    tests specific to the cuda backend
    """

    def __init__(self, *args, **kwargs):
        from theano.sandbox import cuda
        self.gpu_backend = cuda
        self.mode_with_gpu = mode_with_gpu
        self.mode_with_gpu_nodebug = mode_with_gpu_nodebug
        super(T_Scan_Cuda, self).__init__(*args, **kwargs)

    def setUp(self):
        # Skip the test if cuda is not available
        if not self.gpu_backend.cuda_available:
            raise SkipTest('Optional package cuda disabled')

        utt.seed_rng()
        super(T_Scan_Cuda, self).setUp()

    def is_scan_on_gpu(self, node):
        return node.op.info.get('gpu', False)

    def test_inconsistent_inner_fct(self):
        # Test that scan can detect inconsistencies in the inner graph and
        # raises an appropriate exception. The pickled file used in this test
        # relies on the cuda backend.

        # This test has not been extensively tested for Python 3 so it should
        # be skipped if python version is >=3
        version = sys.version_info
        if version >= (3,):
            raise SkipTest("This test relies on a pickled file produced with "
                           "Python 2. The current python version "
                           "(%i.%i.%i.%i) is >= 3 so the test will be "
                           "skipped." % (version.major, version.minor,
                           version.micro, version.serial))

        # When unpickled, the scan op should perform validation on its inner
        # graph, detect the inconsistencies and raise a TypeError
        folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(folder, "inconsistent_scan.pkl")
        assert_raises(TypeError, pickle.load, open(path, "r"))

    def test_consistent_inner_fct(self):
        # Test that scan does not falsely detect inconsistencies in a valid
        # inner graph

        rs = theano.sandbox.rng_mrg.MRG_RandomStreams(use_cuda=True)
        output, _ = theano.scan(lambda : rs.uniform((3,), dtype="float32"),
                                n_steps=3)
        pickle.loads(pickle.dumps(output))

        # Also ensure that, after compilation, the Scan has been moved
        # on the gpu
        fct = theano.function([], output, mode=self.mode_with_gpu)
        scan_nodes = scan_nodes_from_fct(fct)
        assert len(scan_nodes) == 1
        assert self.is_scan_on_gpu(scan_nodes[0])


class T_Scan_Gpuarray(unittest.TestCase, ScanGpuTests):
    """This class takes the gpu tests for scan that are defined in
    class ScanGpuTests and runs them using the gpuarray backend.
    """

    def __init__(self, *args, **kwargs):
        from theano.sandbox import gpuarray
        self.gpu_backend = gpuarray

        # This is unfortunate, but required
        def gpu_from_host(v):
            return gpuarray.GpuFromHost(None)(v)
        self.gpu_backend.gpu_from_host = gpu_from_host

        self.mode_with_gpu = mode_with_opt.including('gpuarray', 'scan')
        self.mode_with_gpu_nodebug = mode_nodebug.including('gpuarray', 'scan')
        super(T_Scan_Gpuarray, self).__init__(*args, **kwargs)

    def setUp(self):
        # Skip the test if pygpu is not available
        if not self.gpu_backend.pygpu_activated:
            raise SkipTest('Optional package pygpu disabled')

        utt.seed_rng()
        super(T_Scan_Gpuarray, self).setUp()

    def is_scan_on_gpu(self, node):
        return node.op.info.get('gpua', False)



def test_speed():
    #
    # This function prints out the speed of very simple recurrent
    # calculations implemented in various ways.  In DebugMode this will
    # test the correctness of the optimizations applied, but generally
    # correctness-testing is not the goal of this test.
    #
    # To be honest, it isn't really a unit test so much as a tool for testing
    # approaches to scan.
    #
    # The computation being tested here is a recurrent addition.
    #
    #
    # We need the CVM for this speed test
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")

    r = numpy.arange(10000).astype(theano.config.floatX).reshape(1000, 10)

    t0 = time.time()
    for i in xrange(1, 1000):
        r[i] += r[i - 1]
    t1 = time.time()
    print('python', t1 - t0)

    r = numpy.arange(10000).astype(theano.config.floatX).reshape(1000, 10)
    t0 = time.time()
    r_i = iter(r[1:])
    r_ii = iter(r[:-1])
    if PY3:
        while True:
            try:
                tmp = next(r_i)
                tmp += next(r_ii)
            except StopIteration:
                break
    else:
        while True:
            try:
                tmp = next(r_i)
                tmp += next(r_ii)
            except StopIteration:
                break
    t1 = time.time()
    print('python with builtin iterator', t1 - t0)

    if 1:
        r = numpy.arange(10000).astype(theano.config.floatX).reshape(1000, 10)
        s_r = tensor.matrix()
        s_y, updates = theano.scan(fn=lambda ri, rii: ri + rii,
                sequences=[s_r[1:]],
                outputs_info=tensor.constant(r[0]),
                mode=theano.Mode(linker='cvm'))
        assert not updates
        f = theano.function([s_r], s_y)

        t2 = time.time()
        f(r)
        t3 = time.time()
        print('theano (scan, cvm)', t3 - t2)

    if 1:
        r = numpy.arange(10000).astype(theano.config.floatX).reshape(-1, 10)
        shared_r = theano.shared(r)
        s_i = theano.shared(numpy.array(1))
        s_rinc = tensor.inc_subtensor(shared_r[s_i], shared_r[s_i - 1],
                tolerate_inplace_aliasing=True)
        # theano.printing.debugprint(s_rinc)
        f = theano.function([],
                            [],
                            updates=OrderedDict([
                                (s_i, s_i + 1),
                                (shared_r, s_rinc)]),
                           mode=theano.Mode(linker='cvm'))
        f._check_for_aliased_inputs = False
        t2 = time.time()
        f_fn = f.fn
        for i in xrange(998):
            f_fn()
        f()  # 999 to update the profiling timers
        t3 = time.time()
        print('theano (updates, cvm)', t3 - t2)
        # print shared_r.get_value()


def test_speed_rnn():

    #
    # This function prints out the speed of recurrent neural network
    # calculations implemented in various ways.  In DebugMode this will
    # test the correctness of the optimizations applied, but generally
    # correctness-testing is not the goal of this test.
    #
    # To be honest, it isn't really a unit test so much as a tool for testing
    # approaches to scan.
    #
    # The computation being tested here is a repeated tanh of a matrix-vector
    # multiplication - the heart of an ESN or RNN.
    #

    # We need the CVM for this speed test
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")

    L = 10000
    N = 50

    numpy.random.seed(2523452)
    r = numpy.arange(L * N).astype(theano.config.floatX).reshape(L, N)
    w = numpy.random.randn(N, N).astype(theano.config.floatX)

    t0 = time.time()
    for i in xrange(1, L):
        r[i] = numpy.tanh(numpy.dot(r[i - 1], w))
    t1 = time.time()
    print('python', t1 - t0)

    if 1:
        r = numpy.arange(L * N).astype(theano.config.floatX).reshape(L, N)
        s_r = tensor.matrix()
        s_y, updates = theano.scan(
                fn=lambda ri, rii: tensor.tanh(tensor.dot(rii, w)),
                sequences=[s_r[1:]],
                outputs_info=tensor.constant(r[0]),
                mode=theano.Mode(linker='cvm'))
        assert not updates
        f = theano.function([s_r], s_y, mode=theano.Mode(linker='cvm'))

        t2 = time.time()
        f(r)
        t3 = time.time()
        print('theano (scan, cvm)', t3 - t2)

    if 1:
        r = numpy.arange(L * N).astype(theano.config.floatX).reshape(L, N)
        s_w = theano.shared(w)
        shared_r = theano.shared(r)
        s_i = theano.scalar.sharedvar.shared(1)
        s_rinc = tensor.inc_subtensor(
                shared_r[s_i],
                theano.tensor.tanh(
                    theano.tensor.dot(
                        shared_r[s_i - 1],
                        w)),
                tolerate_inplace_aliasing=True)
        f = theano.function([], [],
                updates=OrderedDict([
                    (s_i, s_i + 1),
                    (shared_r, s_rinc)]),
                mode=theano.Mode(linker='cvm'))
        # theano.printing.debugprint(f)
        f_fn = f.fn
        # print f_fn
        t2 = time.time()
        f_fn(n_calls=L - 2)
        f()  # 999 to update the profiling timers
        t3 = time.time()
        print('theano (updates, cvm)', t3 - t2)
        # print shared_r.get_value()


def test_speed_batchrnn():

    #
    # This function prints out the speed of recurrent neural network
    # calculations implemented in various ways.

    # We force the mode to theano.Mode(linker='cvm'). If you manually
    # change this code to use DebugMode this will test the correctness
    # of the optimizations applied, but generally correctness-testing
    # is not the goal of this test.
    #
    # To be honest, it isn't really a unit test so much as a tool for testing
    # approaches to scan.
    #
    # The computation being tested here is a repeated tanh of a matrix-vector
    # multiplication - the heart of an ESN or RNN.
    #

    # We need the CVM for this speed test
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")
    L = 100
    B = 50
    N = 400

    numpy.random.seed(2523452)
    r = numpy.arange(B * L * N).astype(theano.config.floatX).reshape(L, B, N)
    w = numpy.random.randn(N, N).astype(theano.config.floatX)

    t0 = time.time()
    for i in xrange(1, L):
        r[i] = numpy.tanh(numpy.dot(r[i - 1], w))
    t1 = time.time()
    print('python', t1 - t0)

    if 1:
        r = numpy.arange(B * L * N).astype(
            theano.config.floatX).reshape(L, B, N)
        s_w = theano.shared(w)
        shared_r = theano.shared(r)
        s_i = theano.scalar.sharedvar.shared(1)
        s_rinc = tensor.inc_subtensor(
                shared_r[s_i],
                theano.tensor.tanh(
                    theano.tensor.dot(
                        shared_r[s_i - 1],
                        w)),
                tolerate_inplace_aliasing=True)
        f = theano.function([],
                            [],
                            updates=[
                                (s_i, s_i + 1),
                                (shared_r, s_rinc)],
                mode=theano.Mode(linker='cvm'))
        # theano.printing.debugprint(f)
        f_fn = f.fn
        # print f_fn
        t2 = time.time()
        f_fn(n_calls=L - 2)
        f()  # 999 to update the profiling timers
        t3 = time.time()
        print('theano (updates, cvm)', t3 - t2)


if __name__ == '__main__':
    #'''
    print(' Use nosetests to run these tests ')
    '''
    scan_tst = T_Scan()
    #''
    print 1
    scan_tst.test_generator_one_output_scalar()
    #''
    print 2
    scan_tst.test_one_sequence_one_output_weights()

    #''
    print 3
    scan_tst.test_one_sequence_one_output_weights_shared()

    #''
    print 4
    scan_tst.test_multiple_inputs_multiple_outputs()
    #''
    print 5
    scan_tst.test_using_taps_input_output()

    #''
    print 6
    scan_tst.test_past_future_taps_shared()
    #''
    print 7
    scan_tst.test_inplace1()
    #''
    print 8
    scan_tst.test_inplace2()
    #''
    print 9
    scan_tst.test_shared_arguments_with_updates()

    print 10
    scan_tst.test_simple_shared_random()

    print 11
    scan_tst.test_only_shared_no_input_no_output()

    print 12
    scan_tst.test_map_functionality()

    print 13
    scan_tst.test_map()
    #''
    print 14
    scan_tst.test_backwards()
    #''

    print 15
    scan_tst.test_reduce()

    print 15.5
    scan_tst.test_save_mem()
    #''
    print 16
    scan_tst.test_grad_one_output()
    #''
    print 17
    scan_tst.test_grad_multiple_outs()
    #''
    print 17.5
    scan_tst.test_multiple_outs_taps()
    #''
    print 18
    scan_tst.test_grad_multiple_outs_taps()
    #''
    print 19
    scan_tst.test_grad_multiple_outs_taps_backwards()
    #''
    print 20
    scan_tst.test_grad_multiple_outs_some_uncomputable()
    #''
    print 21
    scan_tst.test_grad_multiple_outs_some_truncate()
    #''
    print 22
    scan_tst.test_grad_of_shared()
    #''
    print 23
    scan_tst.test_computing_gradient()
    #''
    print 24
    scan_tst.test_scan_output_padding()

    print 25
    scan_tst.test_scan_extra_inputs_hessian()
    #''
    print 26
    scan_tst.test_cloning_no_replace_strict_copy_inputs()

    print 27
    scan_tst.test_cloning_no_replace_strict_not_copy_inputs()

    print 28
    scan_tst.test_cloning_replace_strict_copy_inputs()

    print 29
    scan_tst.test_cloning_replace_not_strict_copy_inputs()

    print 30
    scan_tst.test_cloning_replace_strict_not_copy_inputs()

    print 31
    scan_tst.test_cloning_replace_not_strict_not_copy_inputs()
    #''
    print 32
    scan_tst.test_draw_as_input_to_scan()
    #''
    print 33
    scan_tst.test_reordering()
    #''
    print 34
    scan_tst.test_return_steps()
    #''
    print 35
    scan_tst.test_scan_as_tensor_on_gradients()
    #''
    print 36
    scan_tst.test_save_mem_reduced_number_of_steps()
    #''
    print 37
    scan_tst.test_save_mem_store_steps()
    #'''


def test_compute_test_value():
    # Verify that test values can be used with scan.
    backup = theano.config.compute_test_value
    theano.config.compute_test_value = 'raise'
    try:
        x = tensor.vector('x')
        xv = numpy.ones(3, dtype=theano.config.floatX)
        x.tag.test_value = xv
        y = theano.shared(numpy.arange(3, dtype=theano.config.floatX),
                          name='y')
        z, updates = theano.scan(
                fn=lambda u, v: u + v,
                sequences=[x, y])
        assert not updates
        z.name = 'z'
        # The gradient computation used to crash before 6af465e.
        g = tensor.grad(z.sum(), x)
        #f = theano.function([x], g)
        # print f(xv)
    finally:
        theano.config.compute_test_value = backup


def test_compute_test_value_nonseq():
    # Verify that test values can be used for non_sequences with scan.
    backup = theano.config.compute_test_value
    theano.config.compute_test_value = 'raise'
    try:
        x = tensor.vector('x')
        xv = numpy.ones(3, dtype=theano.config.floatX)
        x.tag.test_value = xv
        y = theano.shared(
                numpy.arange(9, dtype=theano.config.floatX).reshape(3, 3),
                name='y')
        z, updates = theano.scan(
                fn=lambda u, v: u + v,
                sequences=[x],
                non_sequences=[y])
        assert not updates
        z.name = 'z'
        # The gradient computation used to crash before 6af465e.
        g = tensor.grad(z.sum(), x)
        #f = theano.function([x], g)
        # print f(xv)
    finally:
        theano.config.compute_test_value = backup


def test_compute_test_value_grad():
    # Test case originally reported by Bitton Tenessi
    # https://groups.google.com/d/msg/theano-users/fAP3i2CbskQ/3OgBf4yjqiQJ
    WEIGHT = numpy.array([1, 2, 1, 3, 4, 1, 5, 6, 1, 7, 8, 1],
                         dtype='float32')

    old_compute_test_val = theano.config.compute_test_value
    old_exception_verbosity = theano.config.exception_verbosity
    try:
        theano.config.compute_test_value = 'raise'
        theano.config.exception_verbosity = 'high'

        W_flat = tensor.fvector(name='W')
        W_flat.tag.test_value = WEIGHT
        W = W_flat.reshape((2, 2, 3))

        outputs_mi = tensor.as_tensor_variable(
                numpy.asarray(0, dtype='float32'))
        outputs_mi.tag.test_value = numpy.asarray(0, dtype='float32')

        def loss_mi(mi, sum_mi, W):
            outputs_ti = tensor.as_tensor_variable(
                    numpy.asarray(0, dtype='float32'))
            outputs_ti.tag.test_value = numpy.asarray(0, dtype='float32')

            def loss_ti(ti, sum_ti, mi, W):
                return W.sum().sum().sum() + sum_ti

            result_ti, _ = theano.scan(
                    fn=loss_ti,
                    outputs_info=outputs_ti,
                    sequences=tensor.arange(W.shape[1], dtype='int32'),
                    non_sequences=[mi, W],
                    )
            lossmi = result_ti[-1]
            return sum_mi + lossmi

        result_mi, _ = theano.scan(
                fn=loss_mi,
                outputs_info=outputs_mi,
                sequences=tensor.arange(W.shape[0], dtype='int32'),
                non_sequences=[W],
                )

        loss = result_mi[-1]
        tensor.grad(loss, W_flat)
    finally:
        theano.config.compute_test_value = old_compute_test_val
        theano.config.exception_verbosity = old_exception_verbosity


def test_compute_test_value_grad_cast():
    # Test for test values when variables have to be casted
    # Reported by Daniel Renshaw at
    # https://groups.google.com/d/topic/theano-users/o4jK9xDe5WI/discussion
    floatX = theano.config.floatX
    backup = theano.config.compute_test_value
    theano.config.compute_test_value = 'raise'
    try:
        h = tensor.matrix('h')
        h.tag.test_value = numpy.array([[1, 2, 3, 4], [5, 6, 7, 8]],
                                       dtype=floatX)

        w = theano.shared(numpy.random.randn(4, 3).astype(floatX), name='w')

        outputs, _ = theano.scan(lambda i, h, w: (theano.dot(h[i], w), i),
                                 outputs_info=[None, 0], non_sequences=[h, w],
                                 n_steps=3)

        theano.grad(outputs[0].sum(), w)
    finally:
        theano.config.compute_test_value = backup


def test_constant_folding_n_steps():
    # The following code used to crash at revision 2060b8f, in the constant
    # folding optimization step.
    res, _ = theano.scan(lambda x: x * 2,
                         outputs_info=tensor.ones(()),
                         # The constant `n_steps` was causing the crash.
                         n_steps=10)
    on_opt_error = theano.config.on_opt_error
    theano.config.on_opt_error = 'raise'
    try:
        theano.function([], res)()
    finally:
        theano.config.on_opt_error = on_opt_error
