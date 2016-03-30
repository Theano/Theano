from __future__ import absolute_import, print_function, division

import unittest

import numpy

from theano import gof, tensor, function
from theano.tests import unittest_tools as utt


class Minimal(gof.Op):
    # TODO : need description for class

    # if the Op has any attributes, consider using them in the eq function.
    # If two Apply nodes have the same inputs and the ops compare equal...
    # then they will be MERGED so they had better have computed the same thing!

    def __init__(self):
        # If you put things here, think about whether they change the outputs
        # computed by # self.perform()
        #  - If they do, then you should take them into consideration in
        #    __eq__ and __hash__
        #  - If they do not, then you should not use them in
        #    __eq__ and __hash__

        super(Minimal, self).__init__()

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, *args):
        # HERE `args` must be THEANO VARIABLES
        return gof.Apply(op=self, inputs=args, outputs=[tensor.lscalar()])

    def perform(self, node, inputs, out_):
        output, = out_
        # HERE `inputs` are PYTHON OBJECTS

        # do what you want here,
        # but do not modify any of the arguments [inplace].
        print("perform got %i arguments" % len(inputs))

        print("Max of input[0] is ", numpy.max(inputs[0]))

        # return some computed value.
        # do not return something that is aliased to one of the inputs.
        output[0] = numpy.asarray(0, dtype='int64')

minimal = Minimal()


# TODO: test dtype conversion
# TODO: test that invalid types are rejected by make_node
# TODO: test that each valid type for A and b works correctly


class T_minimal(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState(utt.fetch_seed(666))

    def test0(self):
        A = tensor.matrix()
        b = tensor.vector()

        print('building function')
        f = function([A, b], minimal(A, A, b, b, A))
        print('built')

        Aval = self.rng.randn(5, 5)
        bval = numpy.arange(5, dtype=float)
        f(Aval, bval)
        print('done')
