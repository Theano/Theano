from theano.compile.module import *
import theano.tensor as T

def test_whats_up_with_submembers():
    class Blah(FancyModule):
        def __init__(self, stepsize):
            super(Blah, self).__init__()
            self.stepsize = Member(T.value(stepsize))
            x = T.dscalar()
            
            self.step = Method([x], x - self.stepsize)

    B = Blah(0.0)
    b = B.make(mode='FAST_RUN')
    b.step(1.0)
    print b.stepsize
    assert b.stepsize == 0.0
