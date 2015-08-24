import numpy as np
import os.path as pt
from tempfile import mkstemp
import unittest

import theano as th
import theano.d3viz as d3v
from theano.d3viz.tests import models


class TestD3Viz(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(0)

    def check(self, f):
        _, html_file = mkstemp('.html')
        d3v.d3viz(f, html_file)
        assert pt.getsize(html_file) > 0

    def test_mlp(self):
        m = models.Mlp(rng=self.rng)
        f = th.function(m.inputs, m.outputs)
        self.check(f)

    def test_mlp_profiled(self):
        m = models.Mlp(rng=self.rng)
        f = th.function(m.inputs, m.outputs, profile=True)
        x_val = self.rng.normal(0, 1, (1000, m.nfeatures))
        f(x_val)
        self.check(f)

    def test_ofg(self):
        m = models.Ofg()
        f = th.function(m.inputs, m.outputs)
        self.check(f)

    def test_ofg_nested(self):
        m = models.OfgNested()
        f = th.function(m.inputs, m.outputs)
        self.check(f)
