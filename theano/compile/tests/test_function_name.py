from __future__ import absolute_import, print_function, division

import unittest
import os
import re

import theano
from theano import tensor


class FunctionName(unittest.TestCase):
    def test_function_name(self):
        x = tensor.vector('x')
        func = theano.function([x], x + 1.)

        regex = re.compile(os.path.basename('.*test_function_name.pyc?:14'))
        assert(regex.match(func.name) is not None)
