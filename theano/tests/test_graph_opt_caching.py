import unittest
import theano
import theano.tensor as T
from theano.gof.graph import is_same_graph
from theano.gof import FunctionGraph
from theano.scan_module.scan_utils import equal_computations

def graphs_equal(x, y):
    # check if two graphs are equal
    # x: outputs list of graph A
    # y: outputs list of graph B

    pass


def test_graph_equivalence():
    # Test if equivalent graphs are in fact equivalent
    # by using some functions in Theano

    # graph g1
    g1_a = T.fmatrix('inputs')
    g1_b = T.fmatrix('inputs')
    g1_y = T.sum(g1_a+g1_b)
    g1_yy = T.sum(g1_a + g1_b)
    
    g2_x = T.fmatrix('inputs')
    g2_y = g2_x.sum()

    g3_a = T.fmatrix('inputs')
    g3_b = T.fmatrix('inputs')
    g3_y = T.sum(g3_a+g3_b)

    assert is_same_graph(g1_y, g2_y) == False
    # This does not work.
    assert is_same_graph(g1_y, g1_y)
    assert is_same_graph(g1_y, g1_yy)
    assert is_same_graph(g1_y, g3_y, givens={g1_a: g3_a, g1_b: g3_b})
    l1 = theano.gof.graph.inputs([g1_y])
    l2 = theano.gof.graph.inputs([g3_y])
    assert len(l1) == len(l2)
    
    #FunctionGraph([], g1_y)
    
    #assert graphs_equal(g1_y, g3_y) == True
    #assert graphs_equal(g1_y, g2_y) == False
    
def test_graph_optimization_caching():
    # 
    x = T.fmatrix('inputs')
    y = x.sum()
    f = theano.function(inputs=[x],outputs=y)
    

if __name__ == '__main__':
    test_graph_optimization_caching()
    #test_graph_equivalence()
