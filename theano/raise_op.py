"""Symbolic Op for raising an exception."""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"
import gof

class Raise(gof.Op):
    """Op whose perform() raises an exception.
    """
    def __init__(self, msg="", exc=NotImplementedError):
        """
        msg - the argument to the exception
        exc - an exception class to raise in self.perform
        """
        self.msg = msg
        self.exc = exc
    def __eq__(self, other):
        # Note: the msg does not technically have to be in the hash and eq
        # because it doesn't affect the return value.
        return (type(self) == type(other)
                and self.msg == other.msg
                and self.exc == other.exc)
    def __hash__(self):
        return hash((type(self), self.msg, self.exc))
    def __str__(self):
        return "Raise{%s(%s)}"%(self.exc, self.msg)
    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])
    def perform(self, node, inputs, out_storage):
        raise self.exc(self.msg)


