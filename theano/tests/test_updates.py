from __future__ import absolute_import, print_function, division
import unittest

import theano
from theano.updates import OrderedUpdates
import theano.tensor as T


class test_ifelse(unittest.TestCase):

    def test_updates_init(self):
        self.assertRaises(TypeError, OrderedUpdates, dict(d=3))

        sv = theano.shared('asdf')
        OrderedUpdates({sv: 3})

    def test_updates_setitem(self):

        up = OrderedUpdates()

        # keys have to be SharedVariables
        self.assertRaises(TypeError, up.__setitem__, 5, 7)
        self.assertRaises(TypeError, up.__setitem__, T.vector(), 7)

        up[theano.shared(88)] = 7

    def test_updates_add(self):

        up1 = OrderedUpdates()
        up2 = OrderedUpdates()

        a = theano.shared('a')
        b = theano.shared('b')

        assert not up1 + up2

        up1[a] = 5

        # test that addition works
        assert up1
        assert up1 + up2
        assert not up2

        assert len(up1 + up2) == 1
        assert (up1 + up2)[a] == 5

        up2[b] = 7
        assert up1
        assert up1 + up2
        assert up2

        assert len(up1 + up2) == 2
        assert (up1 + up2)[a] == 5
        assert (up1 + up2)[b] == 7

        assert a in (up1 + up2)
        assert b in (up1 + up2)

        # this works even though there is a collision
        # because values all match
        assert len(up1 + up1 + up1) == 1

        up2[a] = 8  # a gets different value in up1 and up2
        try:
            up1 + up2
            assert 0
        except KeyError:
            pass

        # reassigning to a key works fine right?
        up2[a] = 10
