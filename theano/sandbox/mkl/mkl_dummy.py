
"""
Example of local OPT
"""
from __future__ import absolute_import, print_function, division
import theano
from theano import gof


class dummyOP(gof.Op):
    """
    Example to show MKL OP
    """
    __props__ = ()

    def __init__(self, *args):
        pass

    def make_node(self, inp):
        x = theano.tensor.as_tensor_variable(inp)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        print("Intel theano : dummy OP forward")
        x = inputs[0]
        z = theano.tensor.nnet.sigmoid(x)
        outputs[0] = z


dummy_op = dummyOP()
