"""Define the tensor toplevel"""
__docformat__ = "restructuredtext en"

from basic import *

import opt
import opt_uncanonicalize
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
from sharedvar import tensor_constructor as shared

import nnet # used for softmax, sigmoid, etc.



