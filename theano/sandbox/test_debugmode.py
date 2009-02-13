import theano
import theano.tensor
import debugmode

def test0():
    x = theano.tensor.dvector()
    f = theano.function([x], (2.*x + 7) / 2., mode=debugmode.OptCheck())
    print f([1,2])
