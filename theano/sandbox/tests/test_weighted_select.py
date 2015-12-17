import numpy
from theano import config, function, tensor
from theano.sandbox import multinomial

def test_select_distinct():
    p = tensor.fmatrix()
    u = tensor.fvector()
    n = tensor.iscalar()
    m = multinomial.WeightedSelectionFromUniform('auto')(p, u, n)
    
    f = function([p, u, n], m, allow_input_downcast=True)
    
    n_elements = 1000
    numpy.random.seed(12345)
    for i in [5, 10, 50, 100, 500]:
        uni = numpy.random.rand(i).astype(config.floatX)
        pvals = numpy.random.randint(1,100,(1,n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        res = f(pvals, uni, i)
        res = numpy.squeeze(res)
        assert len(res) == i
        assert numpy.all(numpy.in1d(numpy.unique(res), res)), res

def test_select_all():
    p = tensor.fmatrix()
    u = tensor.fvector()
    n = tensor.iscalar()
    m = multinomial.WeightedSelectionFromUniform('auto')(p, u, n)
    
    f = function([p, u, n], m, allow_input_downcast=True)
    
    n_elements = 1000
    numpy.random.seed(12345)
    for _ in range(100):
        uni = numpy.random.rand(n_elements).astype(config.floatX)
        pvals = numpy.random.randint(1,100,(1,n_elements)).astype(config.floatX)
        pvals /= pvals.sum(1)
        res = f(pvals, uni, n_elements)
        res = numpy.squeeze(res)
        assert len(res) == n_elements
        assert numpy.all(numpy.in1d(numpy.unique(res), res)), res

        
def test_select_proportional_to_weight():
    p = tensor.fmatrix()
    u = tensor.fvector()
    n = tensor.iscalar()
    m = multinomial.WeightedSelectionFromUniform('auto')(p, u, n)
    
    f = function([p, u, n], m, allow_input_downcast=True)

    n_elements = 100
    n_selected = 10
    mean_rtol = 0.04
    numpy.random.seed(12345)
    pvals = numpy.random.randint(1,100,(1,n_elements)).astype(config.floatX)
    pvals /= pvals.sum(1)
    avg_pvals = numpy.zeros((n_elements,))
    
    for rep in range(1000):
        uni = numpy.random.rand(n_selected).astype(config.floatX)
        res = f(pvals, uni, n_selected)
        res = numpy.squeeze(res)
        # print res
        avg_pvals[res] += 1
    avg_pvals /= avg_pvals.sum()
    
    print avg_pvals
    print numpy.squeeze(pvals)
    assert numpy.mean(abs(avg_pvals - pvals)) < mean_rtol
