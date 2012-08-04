
# Theano tutorial
# Solution to Exercise in section 'Extending Theano'


import theano


# 1. Op returns x * y

class ProdOp(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x, y):
        x = theano.tensor.as_tensor_variable(x)
        y = theano.tensor.as_tensor_variable(y)
        outdim = x.ndim
        output = (theano.tensor.TensorType
                  (dtype=theano.scalar.upcast(x.dtype, y.dtype),
                      broadcastable=[False] * outdim)())
        return theano.Apply(self, inputs=[x, y], outputs=[output])

    def perform(self, node, inputs, output_storage):
        x, y = inputs
        z = output_storage[0]
        z[0] = x * y

    def infer_shape(self, node, i0_shapes):
        return [i0_shapes[0]]

    def grad(self, inputs, output_grads):
        return [output_grads[0] * inputs[1], output_grads[0] * inputs[0]]


# 2. Op returns x + y and x - y

class SumDiffOp(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x, y):
        x = theano.tensor.as_tensor_variable(x)
        y = theano.tensor.as_tensor_variable(y)
        outdim = x.ndim
        output1 = (theano.tensor.TensorType
                  (dtype=theano.scalar.upcast(x.dtype, y.dtype),
                      broadcastable=[False] * outdim)())
        output2 = (theano.tensor.TensorType
                  (dtype=theano.scalar.upcast(x.dtype, y.dtype),
                      broadcastable=[False] * outdim)())
        return theano.Apply(self, inputs=[x, y], outputs=[output1, output2])

    def perform(self, node, inputs, output_storage):
        x, y = inputs
        z1, z2 = output_storage
        z1[0] = x + y
        z2[0] = x - y

    def infer_shape(self, node, i0_shapes):
        return [i0_shapes[0], i0_shapes[0]]

    def grad(self, inputs, output_grads):
        return [output_grads[0] + output_grads[1],
                output_grads[0] - output_grads[1]]


# 3. Testing apparatus

import numpy
from theano.gof import Op, Apply
from theano import tensor, function, printing
from theano.tests import unittest_tools as utt


class TestOp(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):

        super(TestOp, self).setUp()
        # adapt the choice of the next instruction to the op under test
        self.op_class = ProdOp  # case 1
        #self.op_class = SumDiffOp  # case 2

    def test_perform(self):

        x = theano.tensor.matrix()
        y = theano.tensor.matrix()
        f = theano.function([x, y], self.op_class()(x, y))
        import numpy
        x_val = numpy.random.rand(5, 4)
        y_val = numpy.random.rand(5, 4)
        out = f(x_val, y_val)
        # adapt the choice of the next instruction to the op under test
        assert numpy.allclose(x_val * y_val, out)  # case 1
        #assert numpy.allclose([x_val + y_val, x_val - y_val], out)  # case 2

    def test_gradient(self):

        utt.verify_grad(self.op_class(), [numpy.random.rand(5, 4),
                                numpy.random.rand(5, 4)],
                        n_tests=1, rng=TestOp.rng)

    def test_infer_shape(self):

        x = tensor.dmatrix()
        y = tensor.dmatrix()

        # adapt the choice of the next instruction to the op under test
       
        self._compile_and_check([x, y], [self.op_class()(x, y)],  # case 1
                                [numpy.random.rand(5, 6),
                                 numpy.random.rand(5, 6)],
                                self.op_class)
        """
        
        self._compile_and_check([x, y], self.op_class()(x, y),  # case 2
                                [numpy.random.rand(5, 6),
                                 numpy.random.rand(5, 6)],
                                self.op_class)
        """

if __name__ == "__main__":

    t = TestOp('setUp')
    t.setUp()
    t.test_perform()
    # comment out next instruction in case 2 since autotesting of
    # gradient of multiple output functions is not implemented yet
    t.test_gradient() # enable in case 1, disable in case 2
    t.test_infer_shape()
