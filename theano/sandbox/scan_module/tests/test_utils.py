import cPickle
import numpy
import unittest

import theano
from theano.compile.pfunc import rebuild_collect_shared
import theano.sandbox.scan_module as scan_module

if theano.config.mode == 'FAST_COMPILE':
    mode_with_opt = theano.compile.mode.get_mode('FAST_RUN')
else:
    mode_with_opt = theano.compile.mode.get_default_mode()
mode_with_gpu = mode_with_opt.including('gpu', 'scan')


# TODO: this should replace the verify_grad in gradient.py
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
                raise ValueError('argument element %i has wrong shape %s' % \
                                 (i, str((a.shape, b.shape))))
            vv = multiple_outputs_numeric_grad.abs_rel_err(a, b)
            errs.append(numpy.max(
                multiple_outputs_numeric_grad.abs_rel_err(a, b)))
        if numpy.all(numpy.isfinite(errs)):
            return numpy.max(errs), numpy.argmax(errs)
        else:
            return numpy.inf, 0


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
    return (sum([(s * f).sum() for s, f in zip(scan_outputs, factors)]),
            updates)


def asarrayX(value):
    return theano._asarray(value, dtype=theano.config.floatX)


def clone_optimized_graph(f):
    maker_ins = [x for x in f.maker.env.inputs
                 if not isinstance(x, theano.tensor.sharedvar.SharedVariable)]
    inps, outs, _ = rebuild_collect_shared(f.maker.env.outputs,
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


class TestScanUtils(unittest.TestCase):

    def test001_cloning_no_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.vector('y')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = scan_module.scan_utils.clone(f1,
                                          replace=None,
                                          strict=True,
                                          copy_inputs=True)
        f2_inp = theano.gof.graph.inputs([f2])

        assert z in f2_inp
        assert x in f2_inp
        assert y in f2_inp

    def test002_cloning_no_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.vector('y')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = scan_module.scan_utils.clone(f1,
                                          replace=None,
                                          strict=True,
                                          copy_inputs=False)
        f2_inp = theano.gof.graph.inputs([f2])

        assert not z in f2_inp
        assert not x in f2_inp
        assert not y in f2_inp

    def test003_cloning_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.vector('y')
        y2 = theano.tensor.vector('y2')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = scan_module.scan_utils.clone(f1,
                                          replace={y: y2},
                                          strict=True,
                                          copy_inputs=True)
        f2_inp = theano.gof.graph.inputs([f2])
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test004_cloning_replace_not_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.fvector('y')
        y2 = theano.tensor.dvector('y2')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = scan_module.scan_utils.clone(f1,
                                          replace={y: y2},
                                          strict=False,
                                          copy_inputs=True)
        f2_inp = theano.gof.graph.inputs([f2])
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test005_cloning_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.vector('y')
        y2 = theano.tensor.vector('y2')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = scan_module.scan_utils.clone(f1,
                                          replace={y: y2},
                                          strict=True,
                                          copy_inputs=False)
        f2_inp = theano.gof.graph.inputs([f2])
        assert not z in f2_inp
        assert not x in f2_inp
        assert not y2 in f2_inp

    def test006_cloning_replace_not_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = theano.tensor.vector('x')
        y = theano.tensor.fvector('y')
        y2 = theano.tensor.dvector('y2')
        z = theano.shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = scan_module.scan_utils.clone(f1,
                                          replace={y: y2},
                                          strict=False,
                                          copy_inputs=False)
        f2_inp = theano.gof.graph.inputs([f2])
        assert not z in f2_inp
        assert not x in f2_inp
        assert not y2 in f2_inp
