from __future__ import absolute_import, print_function, division
import numpy as np

from six import integer_types
import theano as th
import theano.tensor as T


class Mlp(object):

    def __init__(self, nfeatures=100, noutputs=10, nhiddens=50, rng=None):
        if rng is None:
            rng = 0
        if isinstance(rng, integer_types):
            rng = np.random.RandomState(rng)
        self.rng = rng
        self.nfeatures = nfeatures
        self.noutputs = noutputs
        self.nhiddens = nhiddens

        x = T.dmatrix('x')
        wh = th.shared(self.rng.normal(0, 1, (nfeatures, nhiddens)),
                       borrow=True)
        bh = th.shared(np.zeros(nhiddens), borrow=True)
        h = T.nnet.sigmoid(T.dot(x, wh) + bh)

        wy = th.shared(self.rng.normal(0, 1, (nhiddens, noutputs)))
        by = th.shared(np.zeros(noutputs), borrow=True)
        y = T.nnet.softmax(T.dot(h, wy) + by)

        self.inputs = [x]
        self.outputs = [y]


class OfgNested(object):

    def __init__(self):
        x, y, z = T.scalars('xyz')
        e = x * y
        op = th.OpFromGraph([x, y], [e])
        e2 = op(x, y) + z
        op2 = th.OpFromGraph([x, y, z], [e2])
        e3 = op2(x, y, z) + z

        self.inputs = [x, y, z]
        self.outputs = [e3]


class Ofg(object):

    def __init__(self):
        x, y, z = T.scalars('xyz')
        e = T.nnet.sigmoid((x + y + z)**2)
        op = th.OpFromGraph([x, y, z], [e])
        e2 = op(x, y, z) + op(z, y, x)

        self.inputs = [x, y, z]
        self.outputs = [e2]


class OfgSimple(object):

    def __init__(self):
        x, y, z = T.scalars('xyz')
        e = T.nnet.sigmoid((x + y + z)**2)
        op = th.OpFromGraph([x, y, z], [e])
        e2 = op(x, y, z)

        self.inputs = [x, y, z]
        self.outputs = [e2]
