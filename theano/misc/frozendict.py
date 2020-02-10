# License : https://github.com/slezica/python-frozendict/blob/master/LICENSE.txt

from __future__ import absolute_import, print_function, division
import collections
import operator
import functools

from theano.compat import Mapping


class frozendict(Mapping):
    """
    An immutable wrapper around dictionaries that implements the complete :py:class:`collections.abc.Mapping`
    interface. It can be used as a drop-in replacement for dictionaries where immutability and ordering are desired.
    """

    dict_cls = dict

    def __init__(self, *args, **kwargs):
        self._dict = self.dict_cls(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self._dict)

    def __hash__(self):
        if self._hash is None:
            hashes = map(hash, self.items())
            self._hash = functools.reduce(operator.xor, hashes, 0)

        return self._hash


class FrozenOrderedDict(frozendict):
    """
    A FrozenDict subclass that maintains key order
    """

    dict_cls = collections.OrderedDict
