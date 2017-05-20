from __future__ import absolute_import, print_function, division
import numpy as np
import theano

class DoubleOp(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return self.__class__.__name__
    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])
    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = x * 2

x = theano.tensor.matrix()

f = theano.function([x], DoubleOp()(x))

inp = np.random.rand(5,5)
out = f(inp)
assert np.allclose(inp*2, out)
print(inp)
print(out)
