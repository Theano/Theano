"""Apply for use with Tensors that implements shape propagation via variable.tag.shape
"""
import sys
from theano import gof

def ishape(v):
    try:
        return (True, v.tag.shape)
    except AttributeError:
        return (False, (None,)*v.type.ndim)


class Apply(gof.Apply):
    def __init__(self, op, inputs, outputs):
        super(Apply, self).__init__(op, inputs, outputs)
        if not inputs:
            return
        # if any input has any shape info, then propagate it
        try:
            provided, ishapes = zip(*[ishape(i) for i in inputs])
        except AttributeError:
            # i.type.ndim didn't make sense for some i
            return
        if provided == [False for i in inputs]:
            # no input had a tag.shape
            return
        try:
            infer_shape = op.infer_shape
        except AttributeError:
            # op has no infer_shape, that's fine
            return

        try:
            oshapes = infer_shape(self, ishapes)
        except NotImplementedError:
            return

        for o, oshp in zip(outputs, oshapes):
            o.tag.shape = oshp


