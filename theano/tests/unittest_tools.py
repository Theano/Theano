import unittest
import numpy

import os, sys

def fetch_seed(pseed=None):
    seed = os.getenv("THEANO_UNITTEST_SEED", pseed)
    try:
        seed = int(seed) if seed else None
    except ValueError:
        print >> sys.stderr, 'Error: THEANO_UNITTEST_SEED contains '\
                'invalid seed, using None instead'
        seed = None

    return seed

def seed_rng(pseed=None):

    seed = fetch_seed(pseed)
    if pseed and pseed!=seed:
        print >> sys.stderr, 'Warning: using seed given by THEANO_UNITTEST_SEED=%i'\
                'instead of seed %i given as parameter' % (seed, pseed)
    numpy.random.seed(seed)
    return seed
