from __future__ import absolute_import, print_function, division
import os
import shutil
import unittest
from tempfile import mkdtemp

import numpy
from numpy.testing import assert_allclose
from nose.plugins.skip import SkipTest

import theano
import theano.sandbox.cuda as cuda_ndarray

from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.misc.pkl_utils import dump, load, StripPickler


class T_dump_load(unittest.TestCase):
    def setUp(self):
        # Work in a temporary directory to avoid cluttering the repository
        self.origdir = os.getcwd()
        self.tmpdir = mkdtemp()
        os.chdir(self.tmpdir)

    def tearDown(self):
        # Get back to the original dir, and delete the temporary one
        os.chdir(self.origdir)
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)

    def test_dump_load(self):
        if not cuda_ndarray.cuda_enabled:
            raise SkipTest('Optional package cuda disabled')

        x = CudaNdarraySharedVariable('x', CudaNdarrayType((1, 1), name='x'),
                                      [[1]], False)

        with open('test', 'wb') as f:
            dump(x, f)

        with open('test', 'rb') as f:
            x = load(f)

        assert x.name == 'x'
        assert_allclose(x.get_value(), [[1]])

    def test_dump_load_mrg(self):
        rng = MRG_RandomStreams(use_cuda=cuda_ndarray.cuda_enabled)

        with open('test', 'wb') as f:
            dump(rng, f)

        with open('test', 'rb') as f:
            rng = load(f)

        assert type(rng) == MRG_RandomStreams

    def test_dump_zip_names(self):
        foo_1 = theano.shared(0, name='foo')
        foo_2 = theano.shared(1, name='foo')
        foo_3 = theano.shared(2, name='foo')
        with open('model.zip', 'wb') as f:
            dump((foo_1, foo_2, foo_3, numpy.array(3)), f)
        keys = list(numpy.load('model.zip').keys())
        assert keys == ['foo', 'foo_2', 'foo_3', 'array_0', 'pkl']
        foo_3 = numpy.load('model.zip')['foo_3']
        assert foo_3 == numpy.array(2)
        with open('model.zip', 'rb') as f:
            foo_1, foo_2, foo_3, array = load(f)
        assert array == numpy.array(3)


class TestStripPickler(unittest.TestCase):
    def setUp(self):
        # Work in a temporary directory to avoid cluttering the repository
        self.origdir = os.getcwd()
        self.tmpdir = mkdtemp()
        os.chdir(self.tmpdir)

    def tearDown(self):
        # Get back to the original dir, and delete the temporary one
        os.chdir(self.origdir)
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)

    def test0(self):
        with open('test.pkl', 'wb') as f:
            m = theano.tensor.matrix()
            dest_pkl = 'my_test.pkl'
            f = open(dest_pkl, 'wb')
            strip_pickler = StripPickler(f, protocol=-1)
            strip_pickler.dump(m)
