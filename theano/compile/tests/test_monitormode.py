import numpy

import theano


def test_detect_nan():
    """
    Test the code snippet example that detects NaN values.
    """
    nan_detected = [False]

    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if numpy.isnan(output[0]).any():
                print '*** NaN detected ***'
                theano.printing.debugprint(node)
                print 'Inputs : %s' % [input[0] for input in fn.inputs]
                print 'Outputs: %s' % [output[0] for output in fn.outputs]
                nan_detected[0] = True
                break

    x = theano.tensor.dscalar('x')
    f = theano.function([x], [theano.tensor.log(x) * x],
                        mode=theano.compile.MonitorMode(
                            post_func=detect_nan))
    f(0)  # log(0) * 0 = -inf * 0 = NaN
    assert nan_detected[0]


def test_optimizers():
    """
    Test that we can remove optimizers
    """
    nan_detected = [False]

    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if numpy.isnan(output[0]).any():
                print '*** NaN detected ***'
                theano.printing.debugprint(node)
                print 'Inputs : %s' % [input[0] for input in fn.inputs]
                print 'Outputs: %s' % [output[0] for output in fn.outputs]
                nan_detected[0] = True
                break

    x = theano.tensor.dscalar('x')
    mode = theano.compile.MonitorMode(post_func=detect_nan)
    mode = mode.excluding('fusion')
    f = theano.function([x], [theano.tensor.log(x) * x],
                        mode=mode)
    # Test that the fusion wasn't done
    assert len(f.maker.fgraph.nodes) == 2
    f(0)  # log(0) * 0 = -inf * 0 = NaN

    # Test that we still detect the nan
    assert nan_detected[0]
