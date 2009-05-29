"""Define the tensor toplevel"""
__docformat__ = "restructuredtext en"

from basic import *

import opt
import blas
import xlogx

import raw_random, randomstreams
from randomstreams import \
    RandomStreams

random = RandomStreams(seed=0xBAD5EED)
"""Imitate the numpy.random symbol with a tensor.random one"""

from elemwise import \
    DimShuffle, Elemwise, CAReduce






