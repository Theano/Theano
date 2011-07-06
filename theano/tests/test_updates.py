import theano
from theano.updates import Updates
import theano.tensor as T


def test_updates_setitem():
    ok = True

    up = Updates()
    sv = theano.shared('asdf')

    # keys have to be SharedVariables
    try:
        up[5] = 7
        ok = False
    except TypeError:
        ok = True
    assert ok

    # keys have to be SharedVariables
    try:
        up[T.vector()] = 7
        ok = False
    except TypeError:
        ok = True
    assert ok

    # keys have to be SharedVariables
    up[theano.shared(88)] = 7


def test_updates_add():

    up1 = Updates()
    up2 = Updates()

    a = theano.shared('a')
    b = theano.shared('b')


    assert not up1 + up2

    up1[a] = 5

    # test that addition works
    assert up1
    assert up1 + up2
    assert not up2

    assert len(up1+up2)==1
    assert (up1 + up2)[a] == 5

    up2[b] = 7
    assert up1
    assert up1 + up2
    assert up2

    assert len(up1+up2)==2
    assert (up1 + up2)[a] == 5
    assert (up1 + up2)[b] == 7

    assert a in (up1 + up2)
    assert b in (up1 + up2)

    # this works even though there is a collision
    # because values all match
    assert len(up1 + up1 + up1)==1

    up2[a] = 8 # a gets different value in up1 and up2
    try:
        up1 + up2
        assert 0
    except KeyError:
        pass

    # reassigning to a key works fine right?
    up2[a] = 10


