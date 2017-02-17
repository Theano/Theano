from __future__ import absolute_import, print_function, division
import itertools
import unittest
import numpy
import theano
from theano import tensor
from theano.scan_module.scan_utils import equal_computations, map_variables
from theano.tensor.type_other import NoneConst


def test_equal_compuations():
    # This was a bug report by a Theano user.
    c = NoneConst
    assert equal_computations([c], [c])
    m = theano.tensor.matrix()
    max_argmax1 = theano.tensor.max_and_argmax(m)
    max_argmax2 = theano.tensor.max_and_argmax(m)
    assert equal_computations(max_argmax1, max_argmax2)


#################
# map_variables #
#################

class TestMapVariables(unittest.TestCase):
    @staticmethod
    def replacer(graph):
        return getattr(graph.tag, "replacement", graph)

    def test_leaf(self):
        a = tensor.scalar("a")
        b = tensor.scalar("b")
        c = tensor.scalar("c")

        b.tag.replacement = c

        u = a + b
        v, = map_variables(self.replacer, [u])

        assert u.owner.inputs == [a, b]
        assert v.owner.inputs == [a, c]

    def test_leaf_inside_scan(self):
        x = tensor.vector('x')
        y = tensor.scalar('y')
        z = tensor.scalar('z')

        y.tag.replacement = z

        s, _ = theano.scan(lambda x: x * y, sequences=x)
        s2, = map_variables(self.replacer, [s])

        f = theano.function([x, y, z], [s, s2])
        rval = f(x=numpy.array([1, 2, 3], dtype=numpy.float32), y=1, z=2)
        assert numpy.array_equal(rval, [[1, 2, 3], [2, 4, 6]])

    def test_scan(self):
        x = tensor.vector('x')

        # we will insert a subgraph involving these variables into the inner
        # graph of scan. since they were not previously in the inner graph,
        # they are like non_sequences to scan(). scan() infers these and
        # imports them into the inner graph properly, and map_variables()
        # should do this as well.
        outer = tensor.scalar("outer")
        shared = theano.shared(
            numpy.array(1., dtype=theano.config.floatX),
            name="shared")
        constant = tensor.constant(1, name="constant")

        # z will equal 1 so multiplying by it doesn't change any values
        z = outer * (shared + constant)

        def step(x, a):
            r = a + x
            r.tag.replacement = z * (a - x)
            return r

        s, _ = theano.scan(step, sequences=x,
                           outputs_info=[numpy.array(0.)])
        # ensure z is owned by the outer graph so map_variables() will need to
        # jump through additional hoops to placate FunctionGraph.
        t = z * s
        s2, = map_variables(self.replacer, [t])
        t2 = z * s2

        f = theano.function([x, outer], [t, t2])
        rval = f(x=numpy.array([1, 2, 3], dtype=numpy.float32), outer=0.5)
        assert numpy.array_equal(rval, [[1, 3, 6], [-1, -3, -6]])

    def test_scan_with_shared_update(self):
        x = tensor.vector('x')

        # counts how many times its value is used
        counter = theano.shared(0, name="shared")
        counter.update = counter + 1

        def step(x, a):
            r = a + x
            # introducing a shared variable with an update into the
            # inner graph is unsupported and the code must crash rather
            # than silently produce the wrong result.
            r.tag.replacement = counter * (a - x)
            return r

        s, _ = theano.scan(step, sequences=x,
                           outputs_info=[numpy.array(0.)])
        self.assertRaises(NotImplementedError,
                          map_variables, self.replacer, [s])

    def test_scan_with_shared_update2(self):
        x = tensor.vector('x')

        # counts how many times its value is used
        counter = theano.shared(0, name="shared")
        counter.update = counter + 1

        def step(x, a):
            r = a + x
            # introducing a shared variable with an update into the
            # inner graph is unsupported and the code must crash rather
            # than silently produce the wrong result.
            r.tag.replacement = counter * (a - x)
            # the shared variable was already present, but the
            # replacement changes the number of times it is used,
            # which would have to change the updates, which is
            # unsupported.
            return r + counter

        s, _ = theano.scan(step, sequences=x,
                           outputs_info=[numpy.array(0.)])
        self.assertRaises(NotImplementedError,
                          map_variables, self.replacer, [s])

    def test_opfromgraph(self):
        # as with the scan tests above, insert foreign inputs into the
        # inner graph.
        outer = tensor.scalar("outer")
        shared = theano.shared(
            numpy.array(1., dtype=theano.config.floatX),
            name="shared")
        constant = tensor.constant(1., name="constant")
        z = outer * (shared + constant)

        # construct the inner graph
        a = tensor.scalar()
        b = tensor.scalar()
        r = a + b
        r.tag.replacement = z * (a - b)

        # construct the outer graph
        c = tensor.scalar()
        d = tensor.scalar()
        u = theano.OpFromGraph([a, b], [r])(c, d)
        t = z * u
        v, = map_variables(self.replacer, [t])
        t2 = z * v

        f = theano.function([c, d, outer], [t, t2])
        for m, n in itertools.combinations(range(10), 2):
            assert f(m, n, outer=0.5) == [m + n, m - n]

        # test that the unsupported case of replacement with a shared
        # variable with updates crashes
        shared.update = shared + 1
        self.assertRaises(NotImplementedError,
                          map_variables, self.replacer, [t])
