from __future__ import absolute_import, print_function, division
import numpy
import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.tests.breakpoint import PdbBreakpoint


class TestPdbBreakpoint(utt.InferShapeTester):

    def setUp(self):

        super(TestPdbBreakpoint, self).setUp()

        # Sample computation that involves tensors with different numbers
        # of dimensions
        self.input1 = T.fmatrix()
        self.input2 = T.fscalar()
        self.output = T.dot((self.input1 - self.input2),
                            (self.input1 - self.input2).transpose())

        # Declare the conditional breakpoint
        self.breakpointOp = PdbBreakpoint("Sum of output too high")
        self.condition = T.gt(self.output.sum(), 1000)
        (self.monitored_input1,
         self.monitored_input2,
         self.monitored_output) = self.breakpointOp(self.condition,
                                                    self.input1,
                                                    self.input2, self.output)

    def test_infer_shape(self):

        input1_value = numpy.arange(6).reshape(2, 3).astype("float32")
        input2_value = 10.0

        self._compile_and_check([self.input1, self.input2],
                                [self.monitored_input1,
                                 self.monitored_input2,
                                 self.monitored_output],
                                [input1_value, input2_value],
                                PdbBreakpoint)

    def test_grad(self):

        input1_value = numpy.arange(9).reshape(3, 3).astype("float32")
        input2_value = 10.0

        grads = [T.grad(self.monitored_input1.sum(), self.input1),
                 T.grad(self.monitored_input2.sum(), self.input2)]

        # Add self.monitored_input1 as an output to the Theano function to
        # prevent Theano from optimizing the PdbBreakpoint op out of the
        # function graph
        fct = theano.function([self.input1, self.input2],
                              grads + [self.monitored_input1])

        gradients = fct(input1_value, input2_value)[:-1]

        expected_gradients = [numpy.ones((3, 3), dtype="float32"),
                              numpy.array(1., dtype="float32")]

        for i in range(len(gradients)):
            numpy.testing.assert_allclose(gradients[i], expected_gradients[i])

    def test_fprop(self):

        input1_value = numpy.arange(9).reshape(3, 3).astype("float32")
        input2_value = 10.0
        fct = theano.function([self.input1, self.input2],
                              [self.monitored_input1, self.monitored_input2])

        output = fct(input1_value, input2_value)
        numpy.testing.assert_allclose(output[0], input1_value)
        numpy.testing.assert_allclose(output[1], input2_value)

    def test_connection_pattern(self):

        node = self.monitored_output.owner
        connection_pattern = self.breakpointOp.connection_pattern(node)
        expected_pattern = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        assert connection_pattern == expected_pattern
