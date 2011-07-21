"""
This is a REALLY PARTIAL TEST.

I did them to help debug stuff.

"""
import logging
import StringIO

import theano
import theano.tensor as tensor


def test_pydotprint_cond_highlight():
    assert len(theano.theano_logger.handlers) == 1

    x = tensor.dvector()
    f = theano.function([x], x*2)
    f([1,2,3,4])

    s = StringIO.StringIO()
    new_handler = logging.StreamHandler(s)
    new_handler.setLevel(logging.DEBUG)
    orig_handler = theano.theano_logger.handlers[0]

    theano.theano_logger.removeHandler(orig_handler)
    theano.theano_logger.addHandler(new_handler)
    try:
        theano.printing.pydotprint(f, cond_highlight = True)
    finally:
        theano.theano_logger.addHandler(orig_handler)
        theano.theano_logger.removeHandler(new_handler)

    assert s.getvalue() == 'pydotprint: cond_highlight is set but there is no IfElse node in the graph\n'
