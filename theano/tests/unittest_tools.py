from copy import copy, deepcopy
import sys

import numpy

import theano.tensor as T
from theano.configparser import config, AddConfigVar, StrParam
try:
    from nose.plugins.skip import SkipTest
except ImportError:
    class SkipTest(Exception):
        """
        Skip this test
        """


AddConfigVar('unittests.rseed',
        "Seed to use for randomized unit tests. Special value 'random' means using a seed of None.",
        StrParam(666),
        in_c_key=False)


def fetch_seed(pseed=None):
    """
    Returns the seed to use for running the unit tests.
    If an explicit seed is given, it will be used for seeding numpy's rng.
    If not, it will use config.unittest.rseed (its default value is 666).
    If config.unittest.rseed is set to "random", it will seed the rng with None,
    which is equivalent to seeding with a random seed.

    Useful for seeding RandomState objects.
    >>> rng = numpy.random.RandomState(unittest_tools.fetch_seed())
    """

    seed = pseed or config.unittests.rseed
    if seed=='random':
        seed = None

    try:
        if seed:
            seed = int(seed)
        else:
            seed = None
    except ValueError:
        print >> sys.stderr, 'Error: config.unittests.rseed contains '\
                'invalid seed, using None instead'
        seed = None

    return seed


def seed_rng(pseed=None):
    """
    Seeds numpy's random number generator with the value returned by fetch_seed.
    Usage: unittest_tools.seed_rng()
    """

    seed = fetch_seed(pseed)
    if pseed and pseed!=seed:
        print >> sys.stderr, 'Warning: using seed given by config.unittests.rseed=%i'\
                'instead of seed %i given as parameter' % (seed, pseed)
    numpy.random.seed(seed)
    return seed


def verify_grad(op, pt, n_tests=2, rng=None, *args, **kwargs):
    """
    Wrapper for tensor/basic.py:verify_grad
    Takes care of seeding the random number generator if None is given
    """
    if rng is None:
        seed_rng()
        rng = numpy.random
    T.verify_grad(op, pt, n_tests, rng, *args, **kwargs)

#
# This supports the following syntax:
#
# try:
#     verify_grad(...)
# except verify_grad.E_grad, e:
#     print e.num_grad.gf
#     print e.analytic_grad
#     raise
#
verify_grad.E_grad = T.verify_grad.E_grad


class TestOptimizationMixin(object):
    def assertFunctionContains(self, f, op, min=1, max=sys.maxint):
        toposort = f.maker.env.toposort()
        matches = [node for node in toposort if node.op == op]
        assert (min <= len(matches) <= max), (toposort, matches, str(op), min, max)

    def assertFunctionContains0(self, f, op):
        return self.assertFunctionContains(f, op, min=0, max=0)

    def assertFunctionContains1(self, f, op):
        return self.assertFunctionContains(f, op, min=1, max=1)

    def assertFunctionContainsN(self, f, op, N):
        return self.assertFunctionContains(f, op, min=N, max=N)

    def SkipTest(self, msg='Skip this test'):
        raise SkipTest(msg)


# This object name should not start with Test.
# Otherwise nosetests will execute it!
class T_OpContractMixin(object):
    # self.ops should be a list of instantiations of an Op class to test.
    # self.other_op should be an op which is different from every op
    other_op = T.add

    def copy(self, x):
        return copy(x)

    def deepcopy(self, x):
        return deepcopy(x)

    def clone(self, op):
        raise NotImplementedError('return new instance like `op`')

    def test_eq(self):
        for i, op_i in enumerate(self.ops):
            assert op_i == op_i
            assert op_i == self.copy(op_i)
            assert op_i == self.deepcopy(op_i)
            assert op_i == self.clone(op_i)
            assert op_i != self.other_op
            for j, op_j in enumerate(self.ops):
                if i == j: continue
                assert op_i != op_j

    def test_hash(self):
        for i, op_i in enumerate(self.ops):
            h_i = hash(op_i)
            assert h_i == hash(op_i)
            assert h_i == hash(self.copy(op_i))
            assert h_i == hash(self.deepcopy(op_i))
            assert h_i == hash(self.clone(op_i))
            assert h_i != hash(self.other_op)
            for j, op_j in enumerate(self.ops):
                if i == j: continue
                assert op_i != hash(op_j)

    def test_name(self):
        for op in self.ops:
            s = str(op)    # show that str works
            assert s       # names should not be empty
