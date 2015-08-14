""" test code snippet in the Theano tutorials.
"""
from __future__ import print_function

import os
import shutil
import unittest

from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest
import numpy
from numpy import array

import theano
import theano.tensor as T
from theano import function, compat

from six.moves import xrange
from theano import config
from theano.tests import unittest_tools as utt
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

class T_typedlist(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/library/typed_list.html
    # Theano/doc/library/typed_list.txt
    # Any change you do here must also be done in the documentation !

    def test_typedlist_basic(self):
        import theano.typed_list

        tl = theano.typed_list.TypedListType(theano.tensor.fvector)()
        v = theano.tensor.fvector()
        o = theano.typed_list.append(tl, v)
        f = theano.function([tl, v], o)
        output = f([[1, 2, 3], [4, 5]], [2])

        # Validate ouput is as expected
        expected_output = [numpy.array([1, 2, 3], dtype="float32"),
                           numpy.array([4, 5], dtype="float32"),
                           numpy.array([2], dtype="float32")]

        assert len(output) == len(expected_output)
        for i in range(len(output)):
            utt.assert_allclose(output[i], expected_output[i])

    def test_typedlist_with_scan(self):
        import theano.typed_list

        a = theano.typed_list.TypedListType(theano.tensor.fvector)()
        l = theano.typed_list.length(a)
        s, _ = theano.scan(fn=lambda i, tl: tl[i].sum(),
                        non_sequences=[a],
                        sequences=[theano.tensor.arange(l, dtype='int64')])

        f = theano.function([a], s)
        output = f([[1, 2, 3], [4, 5]])

        # Validate ouput is as expected
        expected_output = numpy.array([6, 9], dtype="float32")
        utt.assert_allclose(output, expected_output)
