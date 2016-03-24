"""Symbolic Op for raising an exception."""
from __future__ import absolute_import, print_function, division
from theano import gof

__authors__ = "James Bergstra"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__ = "3-clause BSD License"
__contact__ = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"


class Raise(gof.Op):
    """Op whose perform() raises an exception.
    """
    __props__ = ('msg', 'exc')

    def __init__(self, msg="", exc=NotImplementedError):
        """
        msg - the argument to the exception
        exc - an exception class to raise in self.perform
        """
        self.msg = msg
        self.exc = exc

    def __str__(self):
        return "Raise{%s(%s)}" % (self.exc, self.msg)

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, out_storage):
        raise self.exc(self.msg)
