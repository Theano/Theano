"""
Helper functions to make gof backwards compatible (tested on python 2.4 and 2.5)
"""

import sys
if sys.version_info[:2] < (2,5):
    def all(iterable):
        for element in iterable:
            if not element:
                return False
        return True
    def any(iterable):
        for element in iterable:
            if element:
                return True
        return False
    def partial(func, *args, **keywords):
        def newfunc(*fargs, **fkeywords):
            newkeywords = keywords.copy()
            newkeywords.update(fkeywords)
            return func(*(args + fargs), **newkeywords)
        newfunc.func = func
        newfunc.args = args
        newfunc.keywords = keywords
        return newfunc
    class defaultdict(dict):
        def __init__(self, default_factory=None, *a, **kw):
            if (default_factory is not None and
                not hasattr(default_factory, '__call__')):
                raise TypeError('first argument must be callable')
            dict.__init__(self, *a, **kw)
            self.default_factory = default_factory
        def __getitem__(self, key):
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                return self.__missing__(key)
        def __missing__(self, key):
            if self.default_factory is None:
                raise KeyError(key)
            self[key] = value = self.default_factory()
            return value
        def __reduce__(self):
            if self.default_factory is None:
                args = tuple()
            else:
                args = self.default_factory,
            # consider replacing items() with iteritems()
            return type(self), args, None, None, self.items()
        def copy(self):
            return self.__copy__()
        def __copy__(self):
            return type(self)(self.default_factory, self)
        def __deepcopy__(self, memo):
            import copy
            return type(self)(self.default_factory,
                              copy.deepcopy(self.items()))
        def __repr__(self):
            return 'defaultdict(%s, %s)' % (self.default_factory,
                                            dict.__repr__(self))

else:
     # Only bother with this else clause and the __all__ line if you are putting
     # this in a separate file.
     import __builtin__
     all = __builtin__.all
     any = __builtin__.any
     import functools, collections
     partial = functools.partial
     defaultdict = collections.defaultdict
__all__ = ['all', 'any']
