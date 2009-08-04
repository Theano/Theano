import unittest
import numpy
import theano.tensor as T

import os, sys

def fetch_seed(pseed=None):
    """
    Returns the seed to use for running the unit tests.
    If an explicit seed is given, it will be used for seending numpy's rng.
    If not, it will try to get a seed from the THEANO_UNITTEST_SEED variable.
    If THEANO_UNITTEST_SEED is set to "random", it will seed the rng. with None,
    which is equivalent to seeding with a random seed.
    If THEANO_UNITTEST_SEED is not defined, it will use a default seed of 666.

    Useful for seeding RandomState objects.
    >>> rng = numpy.random.RandomState(unittest_tools.fetch_seed())
    """
     
    seed = pseed or os.getenv("THEANO_UNITTEST_SEED", 666)
    if seed=='random':
      seed = None
    #backport
    #seed = None if seed=='random' else seed

    try:
        if seed:
          seed = int(seed)
        else:
          seed = None
        #backport
        #seed = int(seed) if seed else None
    except ValueError:
        print >> sys.stderr, 'Error: THEANO_UNITTEST_SEED contains '\
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
        print >> sys.stderr, 'Warning: using seed given by THEANO_UNITTEST_SEED=%i'\
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

