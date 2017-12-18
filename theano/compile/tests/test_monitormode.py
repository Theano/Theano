from __future__ import absolute_import, print_function, division
import sys

import numpy as np
from six.moves import StringIO

import theano


def test_detect_nan():
    # Test the code snippet example that detects NaN values.

    nan_detected = [False]

    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if np.isnan(output[0]).any():
                print('*** NaN detected ***')
                theano.printing.debugprint(node)
                print('Inputs : %s' % [input[0] for input in fn.inputs])
                print('Outputs: %s' % [output[0] for output in fn.outputs])
                nan_detected[0] = True
                break

    x = theano.tensor.dscalar('x')
    f = theano.function([x], [theano.tensor.log(x) * x],
                        mode=theano.compile.MonitorMode(
                            post_func=detect_nan))
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        f(0)  # log(0) * 0 = -inf * 0 = NaN
    finally:
        sys.stdout = old_stdout
    assert nan_detected[0]


def test_optimizer():
    # Test that we can remove optimizer

    nan_detected = [False]

    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if np.isnan(output[0]).any():
                print('*** NaN detected ***')
                theano.printing.debugprint(node)
                print('Inputs : %s' % [input[0] for input in fn.inputs])
                print('Outputs: %s' % [output[0] for output in fn.outputs])
                nan_detected[0] = True
                break

    x = theano.tensor.dscalar('x')
    mode = theano.compile.MonitorMode(post_func=detect_nan)
    mode = mode.excluding('fusion')
    f = theano.function([x], [theano.tensor.log(x) * x],
                        mode=mode)
    # Test that the fusion wasn't done
    assert len(f.maker.fgraph.apply_nodes) == 2
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        f(0)  # log(0) * 0 = -inf * 0 = NaN
    finally:
        sys.stdout = old_stdout

    # Test that we still detect the nan
    assert nan_detected[0]


def test_not_inplace():
    # Test that we can remove optimizers including inplace optimizers

    nan_detected = [False]

    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if np.isnan(output[0]).any():
                print('*** NaN detected ***')
                theano.printing.debugprint(node)
                print('Inputs : %s' % [input[0] for input in fn.inputs])
                print('Outputs: %s' % [output[0] for output in fn.outputs])
                nan_detected[0] = True
                break

    x = theano.tensor.vector('x')
    mode = theano.compile.MonitorMode(post_func=detect_nan)
    # mode = mode.excluding('fusion', 'inplace')
    mode = mode.excluding('local_elemwise_fusion',
                          'inplace_elemwise_optimizer')
    o = theano.tensor.outer(x, x)
    out = theano.tensor.log(o) * o
    f = theano.function([x], [out],
                        mode=mode)

    # Test that the fusion wasn't done
    assert len(f.maker.fgraph.apply_nodes) == 5
    assert not f.maker.fgraph.toposort()[-1].op.destroy_map
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        f([0, 0])  # log(0) * 0 = -inf * 0 = NaN
    finally:
        sys.stdout = old_stdout

    # Test that we still detect the nan
    assert nan_detected[0]
