"""Defines Updates object for storing a (SharedVariable, new_value) mapping.

"""
__authors__   = "theano-dev"
__copyright__ = "(c) 2010, Universite de Montreal"
__license__   = "3-clause BSD License"
__contact__   = "theano-dev <theano-dev@googlegroups.com>"

__docformat__ = "restructuredtext en"

from theano.compile.sharedvalue  import SharedVariable
import logging
logger = logging.getLogger('theano.updates')

class Updates(dict):
    """
    Dict-like mapping from SharedVariable keys to their new values.

    This mapping supports the use of the "+" operator for the union of updates.
    """
    def __setitem__(self, key, value):
        if isinstance(key, SharedVariable):

            #TODO: consider doing error-checking on value.
            # insist that it is a Theano variable? Have the right type?
            # This could have weird consequences - for example a
            # GPU SharedVariable is customarily associated with a TensorType
            # value. Should it be cast to a GPU value right away?  Should
            # literals be transformed into constants immediately?

            return super(Updates, self).__setitem__(key, value)
        else:
            raise TypeError('Updates keys must inherit from SharedVariable', key)
    def update(self, other):
        for key, val in dict(other).iteritems():
            if key in self:
                if self[key] == val:
                    continue
                raise KeyError('Collision', key)
            self[key] = val # __setitem__ does type-checking

    def __add__(self, other):
        rval = Updates()
        rval.update(self)
        rval.update(other)
        return rval

    def __radd__(other, self):
        rval = Updates()
        rval.update(other)
        rval.update(self)
        return rval

