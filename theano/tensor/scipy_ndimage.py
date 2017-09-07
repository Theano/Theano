import theano
from theano.gradient import DisconnectedType
from scipy.ndimage.interpolation import zoom
import numpy as np

class Zoom(theano.Op):
    # Properties attribute
    __props__ = ()

    def __init__(self):
        super(Zoom, self).__init__()

    def make_node(self, x, ref):
        x = theano.tensor.as_tensor_variable(x)
        ref = theano.tensor.as_tensor_variable(ref)
        return theano.Apply(self, [x, ref], [x.type()])

    # Python implementation:
    def perform(self, node, inputs, outputs_storage):
        x, ref = inputs
        z = outputs_storage[0]
        assert x.ndim == ref.ndim
        assert x.shape[0] == 1
        #
        resize_factors = [
            ref.shape[2] / x.shape[2],
            ref.shape[3] / x.shape[3]
            ]
        #
        final_shape = (1, x.shape[1], ref.shape[2], ref.shape[3])
        z[0] = np.empty(final_shape, dtype=np.float32)

        # Loop over channels
        for i in range(x.shape[1]):
            z[0][0,i] = zoom(x[0,i], resize_factors, order=0)

    def infer_shape(self, node, i0_shapes):
        return i0_shapes[1],

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inputs, output_grads):
        op = Zoom()
        x, ref = inputs
        grads = output_grads[0]
        return op(output_grads[0], x), DisconnectedType()()
