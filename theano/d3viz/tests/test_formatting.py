from __future__ import absolute_import, print_function, division

import numpy as np
import unittest

import theano as th
from theano.d3viz.formatting import PyDotFormatter
from theano.d3viz.tests import models

from nose.plugins.skip import SkipTest
from theano.d3viz.formatting import pydot_imported, pydot_imported_msg
if not pydot_imported:
    raise SkipTest('pydot not available: ' + pydot_imported_msg)


class TestPyDotFormatter(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(0)

    def node_counts(self, graph):
        node_types = [node.get_attributes()['node_type']
                      for node in graph.get_nodes()]
        a, b = np.unique(node_types, return_counts=True)
        nc = dict(zip(a, b))
        return nc

    def test_mlp(self):
        m = models.Mlp()
        f = th.function(m.inputs, m.outputs)
        pdf = PyDotFormatter()
        graph = pdf(f)
        expected = 11
        if th.config.mode == "FAST_COMPILE":
            expected = 12
        self.assertEqual(len(graph.get_nodes()), expected)
        nc = self.node_counts(graph)

        if th.config.mode == "FAST_COMPILE":
            assert nc['apply'] == 6
        else:
            assert nc['apply'] == 5
        assert nc['output'] == 1

    def test_ofg(self):
        m = models.Ofg()
        f = th.function(m.inputs, m.outputs)
        pdf = PyDotFormatter()
        graph = pdf(f)
        assert len(graph.get_nodes()) == 10
        sub_graphs = graph.get_subgraph_list()
        assert len(sub_graphs) == 2
        ofg1, ofg2 = sub_graphs
        if th.config.mode == "FAST_COMPILE":
            assert len(ofg1.get_nodes()) == 9
        else:
            assert len(ofg1.get_nodes()) == 5
        assert len(ofg1.get_nodes()) == len(ofg2.get_nodes())

    def test_ofg_nested(self):
        m = models.OfgNested()
        f = th.function(m.inputs, m.outputs)
        pdf = PyDotFormatter()
        graph = pdf(f)
        assert len(graph.get_nodes()) == 7
        assert len(graph.get_subgraph_list()) == 1
        ofg1 = graph.get_subgraph_list()[0]
        assert len(ofg1.get_nodes()) == 6
        assert len(ofg1.get_subgraph_list()) == 1
        ofg2 = ofg1.get_subgraph_list()[0]
        assert len(ofg2.get_nodes()) == 4
