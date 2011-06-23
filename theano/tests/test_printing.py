"""
This is a REALLY PARTIAL TEST.

I did them to help debug stuff.

"""


import theano
import theano.tensor as tensor


def test_pydotprint_cond_highlight():
    x = tensor.dvector()
    f = theano.function([x], x*2)
    f([1,2,3,4])

    theano.printing.pydotprint(f, cond_highlight = True)
