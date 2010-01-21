"""Define the tensor toplevel"""
__docformat__ = "restructuredtext en"

from basic import *

if config.floatX == 'float32':
    # change the default casting behaviour for python floats to always cast to float32
    autocast_float.dtypes = ('float32',)

import opt
import blas
import xlogx

import raw_random, randomstreams
import shared_randomstreams
from randomstreams import \
    RandomStreams

random = RandomStreams(seed=0xBAD5EED)
"""Imitate the numpy.random symbol with a tensor.random one"""

from elemwise import \
    DimShuffle, Elemwise, CAReduce

import sharedvar # adds shared-variable constructors

import nnet # used for softmax, sigmoid, etc.





