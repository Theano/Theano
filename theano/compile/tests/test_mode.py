import theano
from theano.compile.mode import Mode
import theano.tensor as T

def test_no_output_from_implace():

    x = T.matrix()
    y = T.matrix()
    a = T.dot(x, y)
    b = T.tanh(a)

    # Ensure that the elemwise op that produces the output is inplace when
    # using a mode that does not include the optimization
    fct_no_opt = theano.function([x,y], b, mode="FAST_RUN")
    op = fct_no_opt.maker.fgraph.outputs[0].owner.op
    assert (hasattr(op, 'destroy_map') and 0 in op.destroy_map)

    # Ensure that the elemwise op that produces the output is not inplace when
    # using a mode that includes the optimization
    mode_opt = Mode(linker="cvm", optimizer="fast_run")
    mode_opt = mode_opt.including("add_no_output_from_inplace")

    fct_opt = theano.function([x,y], b, mode=mode_opt)
    op = fct_opt.maker.fgraph.outputs[0].owner.op
    assert (not hasattr(op, 'destroy_map') or 0 not in op.destroy_map)
