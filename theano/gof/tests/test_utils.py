import theano
from theano.gof.utils import give_variables_names, unique, remove


def test_give_variables_names():
    x = theano.tensor.matrix('x')
    y = x + 1
    z = theano.tensor.dot(x, y)
    variables = (x, y, z)
    give_variables_names(variables)
    assert all(var.name for var in variables)
    assert unique([var.name for var in variables])


def test_give_variables_names_idempotence():
    x = theano.tensor.matrix('x')
    y = x + 1
    z = theano.tensor.dot(x, y)
    variables = (x, y, z)

    give_variables_names(variables)
    names = [var.name for var in variables]
    give_variables_names(variables)
    names2 = [var.name for var in variables]

    assert names == names2


def test_give_variables_names_small():
    x = theano.tensor.matrix('x')
    y = theano.tensor.dot(x, x)
    fgraph = theano.FunctionGraph((x,), (y,))
    give_variables_names(fgraph.variables)
    assert all(var.name for var in fgraph.variables)
    assert unique([var.name for var in fgraph.variables])


def test_remove():
    def even(x):
        return x % 2 == 0

    def odd(x):
        return x % 2 == 1
    # The list are needed as with python 3, remove and filter return generators
    # and we can't compare generators.
    assert list(remove(even, range(5))) == list(filter(odd, range(5)))
