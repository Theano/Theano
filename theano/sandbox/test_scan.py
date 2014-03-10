import theano
import numpy
import scan


def test_001():
    x0 = theano.tensor.fvector('x0')
    state = theano.tensor.unbroadcast(
        theano.tensor.shape_padleft(x0), 0)
    out, _ = scan.scan(lambda x:x+numpy.float32(1),
                           states = state,
                           n_steps = 5)
    fn = theano.function([x0], out[0])
    val_x0 = numpy.float32([1,2,3])
    assert numpy.all(fn(val_x0) == val_x0 +5)


def test_002():
    x0 = theano.tensor.fvector('x0')
    state = theano.tensor.alloc(
        theano.tensor.constant(numpy.float32(0)),
        6,
        x0.shape[0])
    state = theano.tensor.set_subtensor(state[0], x0)

    out, _ = scan.scan(lambda x:x+numpy.float32(1),
                           states = state,
                           n_steps = 5)
    fn = theano.function([x0], out)
    val_x0 = numpy.float32([1,2,3])
    assert numpy.all(fn(val_x0)[-1] == val_x0 +5)
    assert numpy.all(fn(val_x0)[0] == val_x0)


def test_003():
    x0 = theano.tensor.fvector('x0')
    sq = theano.tensor.fvector('sq')
    state = theano.tensor.alloc(
        theano.tensor.constant(numpy.float32(0)),
        6,
        x0.shape[0])
    state = theano.tensor.set_subtensor(state[0], x0)

    out, _ = scan.scan(lambda s, x:x+s,
                           sequences=sq,
                           states = state,
                           n_steps = 5)
    fn = theano.function([sq, x0], out)
    val_x0 = numpy.float32([1,2,3])
    val_sq = numpy.float32([1,2,3,4,5])
    assert numpy.all(fn(val_sq, val_x0)[-1] == val_x0 +15)
    assert numpy.all(fn(val_sq, val_x0)[0] == val_x0)

def test_004():
    sq = theano.tensor.fvector('sq')
    nst = theano.tensor.iscalar('nst')
    out, _ = scan.scan(lambda s:s+numpy.float32(1),
                           sequences=sq,
                           states = [],
                           n_steps = nst)
    fn = theano.function([sq,nst], out)
    val_sq = numpy.float32([1,2,3,4,5])
    assert numpy.all(fn(val_sq, 5) == val_sq +1)

def test_005():
    sq = theano.tensor.fvector('sq')
    nst = theano.tensor.iscalar('nst')
    out, _ = scan.scan(lambda s: s+numpy.float32(1),
                       sequences=sq,
                       states = [None],
                       n_steps = nst)
    fn = theano.function([sq, nst], out)
    val_sq = numpy.float32([1,2,3,4,5])
    assert numpy.all(fn(val_sq, 5) == val_sq +1)


if __name__=='__main__':
    test_001()
    test_002()
    test_003()
    test_004()
    test_005()

