from __future__ import absolute_import, print_function, division

from nose.plugins.skip import SkipTest

import theano
from theano.compile.mode import Mode, AddFeatureOptimizer
from theano.gof.toolbox import NoOutputFromInplace
import theano.tensor as T


def test_no_output_from_implace():

    x = T.matrix()
    y = T.matrix()
    a = T.dot(x, y)
    b = T.tanh(a)

    # Ensure that the elemwise op that produces the output is inplace when
    # using a mode that does not include the optimization
    fct_no_opt = theano.function([x, y], b, mode="FAST_RUN")
    op = fct_no_opt.maker.fgraph.outputs[0].owner.op
    assert (hasattr(op, 'destroy_map') and 0 in op.destroy_map)

    if not theano.config.cxx:
        raise SkipTest("Need cxx for this test")
    # Ensure that the elemwise op that produces the output is not inplace when
    # using a mode that includes the optimization
    opt = AddFeatureOptimizer(NoOutputFromInplace())
    mode_opt = Mode(linker="cvm", optimizer="fast_run").register((opt, 49.9))

    fct_opt = theano.function([x, y], b, mode=mode_opt)
    op = fct_opt.maker.fgraph.outputs[0].owner.op
    assert (not hasattr(op, 'destroy_map') or 0 not in op.destroy_map)


def test_including():
    mode = theano.Mode(optimizer='merge')
    mode.including('fast_compile')
