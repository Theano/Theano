import os
import shutil
from tempfile import mkdtemp
import time
import unittest

import cPickle
import numpy
from nose.plugins.skip import SkipTest
from numpy.testing import dec

import theano
import theano.sandbox.rng_mrg
from theano import tensor
from theano.compile.pfunc import rebuild_collect_shared
from theano.gof.python25 import any
from theano.tests  import unittest_tools as utt
import theano.scalar.sharedvar
from theano.gof.python25 import OrderedDict

from numpy.testing.noseclasses import KnownFailureTest


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


class multiple_outputs_numeric_grad:
    """WRITEME"""
    type_eps = {'float64': 1e-7,
            'float32': 3e-3}

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

        dtype_eps = multiple_outputs_numeric_grad.type_eps['float64']

        for i, p in enumerate(pt):
            if ndarray_mask[i]:
                pt[i] = numpy.array(p)
                _eps = multiple_outputs_numeric_grad.type_eps[str(
                                            pt[i].dtype)]
                if _eps > dtype_eps:
                    dtype_eps = _eps

        self.ndarray_mask = ndarray_mask
        #'''
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


#TODO: Test this function, and if it works,
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


class T_Scan(unittest.TestCase):
#class T_Scan(object):

    def setUp(self):
        utt.seed_rng()

    # generator network, only one output , type scalar ; no sequence or
    # non sequence arguments
    @dec.knownfailureif(
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

        ### TESTING PICKLE-ing this function
        origdir = os.getcwd()
        tmpdir = None
        try:
            tmpdir = mkdtemp()
            os.chdir(tmpdir)

            f_out = open('tmp_scan_test_pickle.pkl', 'wb')
            try:
                cPickle.dump(_my_f, f_out, protocol=-1)
            finally:
                f_out.close()
            f_in = open('tmp_scan_test_pickle.pkl', 'rb')
            try:
                my_f = cPickle.load(f_in)
            finally:
                f_in.close()
        finally:
            # Get back to the orinal dir, and delete temporary one.
            os.chdir(origdir)
            if tmpdir is not None:
                shutil.rmtree(tmpdir)

        rng = numpy.random.RandomState(utt.fetch_seed())
        state = rng.uniform()
        steps = 5

        numpy_values = numpy.array([state * (2 ** (k + 1)) for k
                                    in xrange(steps)])
        theano_values = my_f(state, steps)
        assert numpy.allclose(numpy_values, theano_values)

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
        assert numpy.allclose(numpy_values, theano_values)

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
        assert numpy.allclose(theano_values, v_out)

    # as test_one_sequence_one_output_weights, but on the gpu
    # This first version test the first case in the optimizer to the gpu.
    def test_one_sequence_one_output_weights_gpu1(self):
        from theano.sandbox import cuda
        if cuda.cuda_available == False:
            raise SkipTest('Optional package cuda disabled')

        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = theano.tensor.fvector('u')
        x0 = theano.tensor.fscalar('x0')
        W_in = theano.tensor.fscalar('win')
        W = theano.tensor.fscalar('w')

        # The following line is needed to have the first case being used
        # Otherwise, it is the second that is tested.
        mode = mode_with_gpu.excluding('InputToGpuOptimizer')
        output, updates = theano.scan(f_rnn,
                                      u,
                                      x0,
                                      [W_in, W],
                                      n_steps=None,
                                      truncate_gradient=-1,
                                      go_backwards=False,
                                      mode=mode)

        output = theano.sandbox.cuda.gpu_from_host(output)
        f2 = theano.function([u, x0, W_in, W],
                             output,
                             updates=updates,
                             allow_input_downcast=True,
                             mode=mode)

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
        assert numpy.allclose(theano_values, v_out), (theano_values, v_out,
                                                      theano_values - v_out)

        # TO DEL
        topo = f2.maker.fgraph.toposort()
        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]

        topo = f2.maker.fgraph.toposort()
        assert sum([isinstance(node.op, theano.sandbox.cuda.HostFromGpu)
                    for node in topo]) == 0
        assert sum([isinstance(node.op, theano.sandbox.cuda.GpuFromHost)
                    for node in topo]) == 4

        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any([isinstance(node.op, theano.sandbox.cuda.GpuElemwise)
                    for node in scan_node_topo])
        assert not any([isinstance(node.op, theano.sandbox.cuda.HostFromGpu)
                        for node in scan_node_topo])
        assert not any([isinstance(node.op, theano.sandbox.cuda.GpuFromHost)
                        for node in scan_node_topo])

    # This second version test the second case in the optimizer to the gpu.
    def test_one_sequence_one_output_weights_gpu2(self):
        from theano.sandbox import cuda
        if cuda.cuda_available == False:
            raise SkipTest('Optional package cuda disabled')

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
        assert numpy.allclose(theano_values, v_out)

        topo = f2.maker.fgraph.toposort()
        assert sum([isinstance(node.op, theano.sandbox.cuda.HostFromGpu)
                    for node in topo]) == 1
        assert sum([isinstance(node.op, theano.sandbox.cuda.GpuFromHost)
                    for node in topo]) == 4

        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        scan_node_topo = scan_node.op.fn.maker.fgraph.toposort()

        # check that there is no gpu transfer in the inner loop.
        assert any([isinstance(node.op, theano.sandbox.cuda.GpuElemwise)
                    for node in scan_node_topo])
        assert not any([isinstance(node.op, theano.sandbox.cuda.HostFromGpu)
                        for node in scan_node_topo])
        assert not any([isinstance(node.op, theano.sandbox.cuda.GpuFromHost)
                        for node in scan_node_topo])

    # This third test checks that scan can deal with a mixture of dtypes as
    # outputs when is running on GPU
    def test_gpu3_mixture_dtype_outputs(self):
        from theano.sandbox import cuda
        if cuda.cuda_available == False:
            raise SkipTest('Optional package cuda disabled')

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
        assert numpy.allclose(theano_out1, v_out1)
        assert numpy.allclose(theano_out2, v_out2)

        topo = f2.maker.fgraph.toposort()
        scan_node = [node for node in topo
                     if isinstance(node.op, theano.scan_module.scan_op.Scan)]
        assert len(scan_node) == 1
        scan_node = scan_node[0]
        assert scan_node.op.gpu

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
            v_out[step] = v_u[step] * W_in.get_value() + \
                    v_out[step - 1] * W.get_value()

        theano_values = f3(v_u, v_x0)
        assert  numpy.allclose(theano_values, v_out)

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
            return [theano.dot(u1_t, W_in1) + u2_t * W_in2 + \
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
        v_x[0] = numpy.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + \
                    numpy.dot(v_x0, vW)
        v_y[0] = numpy.dot(v_x0, vWout)
        for i in xrange(1, 3):
            v_x[i] = numpy.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + \
                        numpy.dot(v_x[i - 1], vW)
            v_y[i] = numpy.dot(v_x[i - 1], vWout)

        (theano_x, theano_y) = f4(v_u1, v_u2, v_x0, v_y0, vW_in1)
        assert numpy.allclose(theano_x, v_x), (theano_x, v_x, theano_x - v_x)
        assert numpy.allclose(theano_y, v_y), (theano_y, v_y, theano_y - v_y)

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
            return [theano.dot(u1_t, W_in1) + \
                        (u2_t + u2_tm1 * u2_tp1) * W_in2 + \
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
        assert numpy.allclose(numpy_out, theano_out)

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
        assert numpy.allclose(numpy_out, theano_out)

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
        mu0 = theano.Param(u0, mutable=False)
        mu1 = theano.Param(u1, mutable=True)
        mu2 = theano.Param(u2, mutable=True)
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
            numpy_x0[i] = vu0[i] * vW_in + numpy_x0[i - 1] * vW + \
                    vu1[i] * vu2[i]
            numpy_x1[i] = vu0[i] * vW_in + numpy_x1[i - 1] * vW + \
                    vu1[i] + vu2[i]

        # note theano computes inplace, so call function after numpy
        # equivalent is done
        (theano_x0, theano_x1) = f9(vu0, vu1, vu2, vx0, vx1)
        # assert that theano does what it should
        assert numpy.allclose(theano_x0, numpy_x0), (theano_x0, numpy_x0,
                                                     theano_x0 - numpy_x0)
        assert numpy.allclose(theano_x1, numpy_x1), (theano_x1, numpy_x1,
                                                     theano_x1 - numpy_x1)
        # assert that it was done in place

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Old way of doing inplace operations is deprecated .. tests don't
        # make sense anymore.

        ##assert numpy.allclose( theano_x0 , vu2)
        ## assert numpy.allclose( theano_x1 , vu1)

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
        mu0 = theano.Param(u0, mutable=True)
        mu1 = theano.Param(u1, mutable=True)
        mu2 = theano.Param(u2, mutable=True)
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
            numpy_x0[i] = vu0[i] * vW_in + numpy_x0[i - 1] * vW + \
                    vu1[i] * vu1[i + 1]
            numpy_x1[i] = vu0[i] * vW_in + numpy_x1[i - 1] * vW + \
                    vu2[i] + vu2[i + 1] + vu2[i + 2]

        # note theano computes inplace, so call function after numpy
        # equivalent is done
        (theano_x0, theano_x1) = f9(vu0, vu1, vu2, vx0, vx1)
        # assert that theano does what it should
        assert numpy.allclose(theano_x0, numpy_x0), (theano_x0, numpy_x0)
        assert numpy.allclose(theano_x1, numpy_x1), (theano_x1, numpy_x1)
        # assert that it was done in place
        # not that x0 should not be inplace of vu2 because you are using
        # past values of u2, and therefore you are not allowed to work
        # inplace !!

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Old way of doing inplace operations is deprecated .. tests don't
        # make sense anymore.
        #assert not numpy.allclose( theano_x0 , vu2[1:4])
        #assert numpy.allclose( theano_x1 , vu1[0:3])

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
#Traceback (most recent call last):
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
#TypeError: ('__array__() takes no arguments (1 given)', <theano.scan.Scan object at 0x3dbbf90>(?_steps, u1, u2, y0, y1, 0.0, W1, W2), 'Sequence id of Apply node=0')
#
#  This don't seam to be a theano related bug...
        vu1 = asarrayX(rng.rand(3, 2))

        W1 = theano.shared(vW1, 'W1')
        W2 = theano.shared(vW2, 'W2')
        u1 = theano.shared(vu1, 'u1')
        y1 = theano.shared(vy1, 'y1')

        def f(u1_t, u2_t, y0_tm3, y0_tm2, y0_tm1, y1_tm1):
            y0_t = theano.dot(theano.dot(u1_t, W1), W2) + 0.1 * y0_tm1 + \
                    0.33 * y0_tm2 + 0.17 * y0_tm3
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
            numpy_y0[idx + 3] = numpy.dot(\
                                          numpy.dot(vu1[idx, :], numpy_W1), \
                                          numpy_W2) + \
                                0.1 * numpy_y0[idx + 2] + \
                                0.33 * numpy_y0[idx + 1] + \
                                0.17 * numpy_y0[idx]
            numpy_y1[idx + 1] = numpy.dot(vu2[idx, :], numpy_W2) +\
                                numpy_y1[idx]
            numpy_y2[idx] = numpy.dot(vu1[idx, :], numpy_W1)
            numpy_W1 = numpy_W1 + .1
            numpy_W2 = numpy_W2 + .05

        assert numpy.allclose(theano_y0, numpy_y0[3:])
        assert numpy.allclose(theano_y1, numpy_y1[1:])
        assert numpy.allclose(theano_y2, numpy_y2)
        assert numpy.allclose(W1.get_value(), numpy_W1)
        assert numpy.allclose(W2.get_value(), numpy_W2)

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
        assert numpy.allclose(theano_v, numpy_v[:5, :])
        theano_v = my_f()
        assert numpy.allclose(theano_v, numpy_v[5:, :])

    def test_cuda_gibbs_chain(self):
        from theano.sandbox import cuda
        if cuda.cuda_available == False:
            raise SkipTest('Optional package cuda disabled')

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

        # I leave this to tested by debugmode, this test was anyway more of
        # doest the graph compile kind of test
        t_result = my_f()

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
        assert numpy.allclose(t_result, n_result)

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
        assert numpy.allclose(state.get_value(), numpy_state)

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
        assert numpy.allclose(theano_result, numpy_result)

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
        assert numpy.allclose(abs_vals, theano_vals)

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
        assert numpy.allclose(theano_values, v_out)

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
                    (max_err, 1e-2, max_err_pos))

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
                    (max_err, 1e-2, max_err_pos))

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
                    (y_tm1 + y_tm3) * theano.dot(x_tm1, W_out),
                    theano.dot(u1_t, W_in1)]
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
                    (max_err, 1e-2, max_err_pos))

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
                    (max_err, 1e-2, max_err_pos))

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
                    (max_err, 1e-2, max_err_pos))

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
        assert numpy.allclose(analytic_grad[0][:2], numpy.zeros((2, 2)))

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

        assert numpy.allclose([ny1, ny1], nz1)
        assert numpy.allclose([ny2, ny2], nz2)
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
        assert numpy.allclose(f([2, 3]), 5)

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

        However, this is not the proper behavior for:
        * shared variables : these should not be padded in any way
        * when return_steps is explicitely set to 1. Output should NOT be
          a list, but a tensor corresponding to the result of the last
          iteration.

        This unit test addresses the bug fix of changeset ba7157e95cb1.

        !!! test lost some of its meaning because return_steps has been
        deprecated !!!
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

        assert numpy.allclose(out, vR)

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
                          copy_inputs=True)
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
                          copy_inputs=False)
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
                          copy_inputs=True)
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
                          copy_inputs=True)
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
                          copy_inputs=False)
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
                          copy_inputs=False)
        f2_inp = theano.gof.graph.inputs([f2])
        assert not z  in f2_inp
        assert not x  in f2_inp
        assert not y2 in f2_inp

    ### TEST RE-ordering of inputs
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

        assert numpy.allclose(theano_x, v_x)
        assert numpy.allclose(theano_y, v_y)

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

        assert numpy.allclose(theano_x, v_x[-1:])
        assert numpy.allclose(theano_y, v_y[-1:])

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
        f = theano.function([W, inpt], d_cost_wrt_W,
                             givens=OrderedDict([(initial, theano.shared(numpy.zeros(5)))]))

        rval = numpy.asarray([[5187989] * 5] * 5, dtype=theano.config.floatX)
        arg1 = numpy.ones((5, 5), dtype=theano.config.floatX)
        arg2 = numpy.ones((10, 5), dtype=theano.config.floatX)
        assert numpy.allclose(f(arg1, arg2), rval)

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

        assert numpy.allclose(tx1, v_u[:2] + 1.)
        assert numpy.allclose(tx2, v_u[4] + 2.)
        assert numpy.allclose(tx3, v_u[3] + 3.)
        assert numpy.allclose(tx4, v_u[:3] + 4.)
        assert numpy.allclose(tx5, v_u[-10] + 5.)
        assert numpy.allclose(tx6, v_u[-15] + 6.)
        assert numpy.allclose(tx7, v_u[:-15] + 7.)
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

        assert numpy.allclose(tx1, v_u[-7] + 1.)
        assert numpy.allclose(tx2, v_u[-3:-1] + 2.)
        assert numpy.allclose(tx3, v_u[-6:] + 3.)
        assert numpy.allclose(tx4, v_u[-1] + 4.)
        assert numpy.allclose(tx5, v_u[-1] + 5.)

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
        # For this to work we need an optimization that it will be pushed in
        # a new pull request
        self.assertTrue(nb_scan == 2)
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

        f = theano.function([x, y], [sx, sy], mode=mode_with_opt)
        topo = f.maker.fgraph.toposort()
        scans = filter(lambda n: isinstance(
            n.op, theano.scan_module.scan_op.Scan), topo)
        self.assertTrue(len(scans) == 2)

        sx, upx = theano.scan(sum, sequences=[x], n_steps=2)
        sy, upy = theano.scan(sum, sequences=[y], n_steps=3)

        f = theano.function([x, y], [sx, sy], mode=mode_with_opt)
        topo = f.maker.fgraph.toposort()
        scans = filter(lambda n: isinstance(
            n.op, theano.scan_module.scan_op.Scan), topo)
        self.assertTrue(len(scans) == 2)

        sx, upx = theano.scan(sum, sequences=[x], n_steps=4)
        sy, upy = theano.scan(sum, sequences=[y], n_steps=4)

        f = theano.function([x, y], [sx, sy], mode=mode_with_opt)
        topo = f.maker.fgraph.toposort()
        scans = filter(lambda n: isinstance(
            n.op, theano.scan_module.scan_op.Scan), topo)
        self.assertTrue(len(scans) == 1)

        sx, upx = theano.scan(sum, sequences=[x])
        sy, upy = theano.scan(sum, sequences=[x])

        f = theano.function([x], [sx, sy], mode=mode_with_opt)
        topo = f.maker.fgraph.toposort()
        scans = filter(lambda n:
                       isinstance(n.op, theano.scan_module.scan_op.Scan), topo)
        self.assertTrue(len(scans) == 1)

        sx, upx = theano.scan(sum, sequences=[x])
        sy, upy = theano.scan(sum, sequences=[x], mode='FAST_COMPILE')

        f = theano.function([x], [sx, sy],
                            mode=mode_with_opt)
        topo = f.maker.fgraph.toposort()
        scans = filter(lambda n:
                       isinstance(n.op, theano.scan_module.scan_op.Scan), topo)
        self.assertTrue(len(scans) == 1)

        sx, upx = theano.scan(sum, sequences=[x])
        sy, upy = theano.scan(sum, sequences=[x], truncate_gradient=1)

        f = theano.function([x], [sx, sy], mode=mode_with_opt)
        topo = f.maker.fgraph.toposort()
        scans = filter(lambda n:
                       isinstance(n.op, theano.scan_module.scan_op.Scan), topo)
        self.assertTrue(len(scans) == 2)

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
        memory = theano.shared(mem_val.copy())
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
        memory.set_value(mem_val.copy())
        f2_vals = f2(x_val)
        assert numpy.allclose(f_vals, f2_vals)

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
        assert f1().shape[0] == 1
        gx = theano.tensor.grad(o, x)
        f2 = theano.function([], gx)
        assert numpy.allclose(f2(), numpy.ones((10,)))

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
        assert f1().shape[0] == 1
        gx = theano.tensor.grad(o, x)
        f2 = theano.function([], gx)
        assert numpy.allclose(f2(), numpy.ones((10,)))

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
        assert f1().shape[0] == 1
        gx = theano.tensor.grad(o, x)
        f2 = theano.function([], gx)
        assert numpy.allclose(f2(), numpy.ones((10,)))

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

        vnu, vnh0, vnW, vno = fn_rop(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)
        tnu, tnh0, tnW, tno = fn_test(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)
        assert numpy.allclose(vnu, tnu, atol=1e-6)
        assert numpy.allclose(vnh0, tnh0, atol=1e-6)
        assert numpy.allclose(vnW, tnW, atol=1e-6)

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

        assert numpy.allclose(vnu, tnu, atol=1e-6)
        assert numpy.allclose(vnh0, tnh0, atol=1e-6)
        assert numpy.allclose(vnW, tnW, atol=1e-6)

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
        assert numpy.allclose(sol, f(v_h, v_W1, v_W2))

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
        raise KnownFailureTest((
            "This tests depends on an optimization for scan "
            "that has not been implemented yet."))
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

        f = theano.function([x], [o, o2])
        vx = numpy.zeros((50,), dtype=theano.config.floatX)
        vx[23] = 4
        out, out2 = f(vx)
        print 'len_out', len(out)
        assert len(out) == 24
        assert numpy.all(out2 == vx + 2)
        lssc = [x for x in f.maker.fgraph.toposort()
                if isinstance(x.op, theano.scan_module.scan_op.Scan)]
        assert len(lssc) == 2

    @dec.knownfailureif(True,
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

    def test_bugFunctioProvidesIntermediateNodesAsInputs(self):
        # This is a bug recently reported by Ilya
        # made it CPU friendly
        V = tensor.ftensor3('INPUT')
        orig = tensor.fmatrix('PARAM')
        # = gpu_from_host(orig)  # <-- this doesn't work
        W = orig + 2  # <-- has same effect but it works on CPU as well
        #W = T.fmatrix('PARAM') # <-- this line works

        def one_step(v, W):
            o = v + 1 + W.sum()  # <-- this doesn't work
            #o = v + 1  # <-- this line works
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
        raise KnownFailureTest('This is a generic problem with infershape'
                               ' that has to be discussed and figured out')
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

        assert numpy.allclose(theano_x, v_x[-2:])
        assert numpy.allclose(theano_y, v_y[-4:])

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
        assert numpy.allclose(f(vx, vA), vR)

    def test_savemem_opt(self):
        y0 = theano.shared(numpy.ones((2, 10)))
        [y1, y2], updates = theano.scan(lambda y: [y, y],
                                         outputs_info=[dict(initial=y0,
                                                            taps=[-2]), None],
                                        n_steps=5)
        rval = theano.function([], y2.sum())()

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
                    (max_err, 1e-2, max_err_pos))

    def test_grad_numeric_shared(self):
        shared_var = theano.shared(numpy.float32(1.))

        def inner_fn():
            return [], OrderedDict([(shared_var, shared_var + numpy.float32(1.))])
        _, updates = theano.scan(inner_fn,
                                 n_steps=10,
                                 truncate_gradient=-1,
                                 go_backwards=False)
        cost = updates.values()[0]
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
        outputs_info = [OrderedDict([('initial', initial_value), ('taps', [-4])]), None]
        results, updates = theano.scan(fn=onestep,
                                       sequences=seq,
                                       outputs_info=outputs_info)

        f = theano.function([seq], results[1])
        assert numpy.all(exp_out == f(inp))

    def test_borrow_bug_jeremiah(self):
        # This test fails if scan uses wrongly the borrow flag
        inp = numpy.arange(10).reshape(-1, 1).astype(theano.config.floatX)
        exp_out = numpy.zeros((10, 1)).astype(theano.config.floatX)
        exp_out[4:] = inp[:-4]

        def onestep(x, x_tm4):
            return x, x_tm4

        seq = tensor.matrix()
        initial_value = theano.shared(numpy.zeros((4, 1),
                                                  dtype=theano.config.floatX))
        outputs_info = [OrderedDict([('initial', initial_value), ('taps', [-4])]), None]
        results, _ = theano.scan(fn=onestep,
                                       sequences=seq,
                                       outputs_info=outputs_info)
        sharedvar = theano.shared(numpy.zeros((1, 1),
                                              dtype=theano.config.floatX))
        updates = OrderedDict([(sharedvar, results[0][-1:])])

        f = theano.function([seq], results[1], updates=updates)
        assert numpy.all(exp_out == f(inp))

    def test_grad_connectivity_matrix(self):
        def inner_fn(x_tm1, y_tm1, z_tm1):
            x_tm1.name = 'x'
            y_tm1.name = 'y'
            z_tm1.name = 'z'
            return x_tm1 ** 2, x_tm1 + y_tm1, x_tm1 + 1

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

    def test_savemem_does_not_duplicate_number_of_scan_nodes(self):
        var = tensor.ones(())
        values, _ = theano.scan(lambda x: ([x], (), theano.scan_module.until(x)),
                                          outputs_info=[var], n_steps=2)

        tmp_fn = theano.function([var], values)
        scan_nodes = [x for x in tmp_fn.maker.fgraph.toposort()
                      if isinstance(x.op,
                                    theano.scan_module.scan_op.Scan)]
        assert len(scan_nodes) == 1



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
    #We need the CVM for this speed test
    if not theano.config.cxx:
        raise SkipTest("G++ not available, so we need to skip this test.")

    r = numpy.arange(10000).astype(theano.config.floatX).reshape(1000, 10)

    t0 = time.time()
    for i in xrange(1, 1000):
        r[i] += r[i - 1]
    t1 = time.time()
    print 'python', t1 - t0

    r = numpy.arange(10000).astype(theano.config.floatX).reshape(1000, 10)
    t0 = time.time()
    r_i = iter(r[1:])
    r_ii = iter(r[:-1])
    while True:
        try:
            tmp = r_i.next()
            tmp += r_ii.next()
        except StopIteration:
            break
    t1 = time.time()
    print 'python with builtin iterator', t1 - t0

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
        print 'theano (scan, cvm)', t3 - t2

    if 1:
        r = numpy.arange(10000).astype(theano.config.floatX).reshape(-1, 10)
        shared_r = theano.shared(r)
        s_i = theano.shared(numpy.array(1))
        s_rinc = tensor.inc_subtensor(shared_r[s_i], shared_r[s_i - 1],
                tolerate_inplace_aliasing=True)
        theano.printing.debugprint(s_rinc)
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
        print 'theano (updates, cvm)', t3 - t2
        print shared_r.get_value()


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

    #We need the CVM for this speed test
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
    print 'python', t1 - t0

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
        print 'theano (scan, cvm)', t3 - t2

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
        #theano.printing.debugprint(f)
        f_fn = f.fn
        #print f_fn
        t2 = time.time()
        f_fn(n_calls=L - 2)
        f()  # 999 to update the profiling timers
        t3 = time.time()
        print 'theano (updates, cvm)', t3 - t2
        #print shared_r.get_value()


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

    #We need the CVM for this speed test
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
    print 'python', t1 - t0

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
        #theano.printing.debugprint(f)
        f_fn = f.fn
        #print f_fn
        t2 = time.time()
        f_fn(n_calls=L - 2)
        f()  # 999 to update the profiling timers
        t3 = time.time()
        print 'theano (updates, cvm)', t3 - t2


if __name__ == '__main__':
    #'''
    print ' Use nosetests to run these tests '
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
        y = theano.shared(numpy.arange(3, dtype=theano.config.floatX), name='y')
        z, _ = theano.scan(
                fn=lambda u, v: u + v,
                sequences=[x, y])
        assert not _
        z.name = 'z'
        # The gradient computation used to crash before 6af465e.
        g = tensor.grad(z.sum(), x)
        #f = theano.function([x], g)
        #print f(xv)
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
