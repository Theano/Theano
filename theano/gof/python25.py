"""
Helper functions to make gof backwards compatible (tested on python 2.4 and 2.5)
"""

import collections
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

    class deque(collections.deque):
        """
        Custom deque class to implement the `remove` method.
        """
        def remove(self, item):
            found = None
            for i, x in enumerate(self):
                if x == item:
                    found = i
                    break
            if found is None:
                raise ValueError('item not found in deque')
            # To remove an item, we rotate the queue until it is the first item
            # in the queue, we pop it, and finally we rotate back the queue.
            self.rotate(-found)
            self.popleft()
            self.rotate(found)

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
    deque = collections.deque

__all__ = ['all', 'any']

if sys.version_info[:2] < (2,6):
    # Borrowed from Python docs
    def combinations(iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = range(r)
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)


    def product(*args, **kwds):
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        pools = map(tuple, args) * kwds.get('repeat', 1)
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)

else:
    from itertools import combinations, product
