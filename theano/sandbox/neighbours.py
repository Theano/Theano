"""
Neighbours was moved into theano.tensor.nnet.neighbours.
This file was created for compatibility.
"""
from __future__ import absolute_import, print_function, division
from theano.tensor.nnet.neighbours import (images2neibs, neibs2images,
                                           Images2Neibs)

__all__ = ["images2neibs", "neibs2images", "Images2Neibs"]
