#!/usr/bin/env python
# Theano tutorial
# Solution to Exercise in section 'Baby Steps - Algebra'

from __future__ import absolute_import, print_function, division
import theano
a = theano.tensor.vector()  # declare variable
b = theano.tensor.vector()  # declare variable
out = a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression
f = theano.function([a, b], out)   # compile function
print(f([1, 2], [4, 5]))  # prints [ 25.  49.]
