"""Define the tensor toplevel"""
__docformat__ = "restructuredtext en"

from basic import *

import opt
import blas

import raw_random, rmodule
from rmodule import \
    RandomKit, RModule

random = RandomKit('random')
"""Imitate the numpy.random symbol with a tensor.random one"""

from elemwise import \
    DimShuffle, Elemwise, CAReduce






