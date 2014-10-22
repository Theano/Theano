import numpy as np

import theano
from theano.gof import Type, hashtype
from theano.tensor.type import TensorType


class MaskedTensorType(Type):
    broadcastable = property(lambda self: self.ttype.broadcastable)
    ndim = property(lambda self: len(self.ttype.broadcastable))
    dtype = property(lambda self: self.ttype.dtype)

    def __init__(self, ttype):
        if not isinstance(ttype, TensorType):
            raise TypeError('Expected a Theano TensorType')
        self.ttype = ttype
        self.mask_ttype = TensorType('int8', ttype.broadcastable)

    def __str__(self):
        return 'Masked{}'.format(self.ttype)

    def filter(self, x, strict=False, allow_downcast=None):
        if strict:
            if not isinstance(x, np.ma.MaskedArray):
                raise TypeError('Expected a masked array')
        if allow_downcast:
            x = np.ma.array(theano._asarray(x.data, dtype=self.dtype),
                            mask=x.mask)
        if self.ttype.is_valid_value(x.data):
            # TODO Check mask for validity?
            return np.ma.array(x.data, mask=x.mask)
        else:
            raise TypeError

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.ttype == other.ttype)

    def __hash__(self):
        return hashtype(self) ^ hash(self.ttype) ^ hash(self.mask_ttype)

    def values_eq(self, a, b):
        if not self.ttype.values_eq(a.data, b.data):
            return False
        if not self.mask_ttype.values_eq(a.mask, b.mask):
            return False
        return True

    def may_share_memory(self, a, b):
        if a is b:
            return True
        if self.ttype.may_share_memory(a.data, b.data):
            return True
        if self.mask_ttype.may_share_memory(a.mask, b.mask):
            return True
        return False
