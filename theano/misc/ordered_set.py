from __future__ import absolute_import, print_function, division

try:
    from collections.abc import MutableSet
except ImportError:
    # this raises an DeprecationWarning on py37 and will become
    # an Exception in py38
    from collections import MutableSet
import types
import weakref

from six import string_types


def check_deterministic(iterable):
    # Most places where OrderedSet is used, theano interprets any exception
    # whatsoever as a problem that an optimization introduced into the graph.
    # If I raise a TypeError when the DestoryHandler tries to do something
    # non-deterministic, it will just result in optimizations getting ignored.
    # So I must use an assert here. In the long term we should fix the rest of
    # theano to use exceptions correctly, so that this can be a TypeError.
    if iterable is not None:
        if not isinstance(iterable, (
                list, tuple, OrderedSet,
                types.GeneratorType, string_types)):
            if len(iterable) > 1:
                # We need to accept length 1 size to allow unpickle in tests.
                raise AssertionError(
                    "Get an not ordered iterable when one was expected")

# Copyright (C) 2009 Raymond Hettinger
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# {{{ http://code.activestate.com/recipes/576696/ (r5)


class Link(object):
    # This make that we need to use a different pickle protocol
    # then the default.  Othewise, there is pickling errors
    __slots__ = 'prev', 'next', 'key', '__weakref__'

    def __getstate__(self):
        # weakref.proxy don't pickle well, so we use weakref.ref
        # manually and don't pickle the weakref.
        # We restore the weakref when we unpickle.
        ret = [self.prev(), self.next()]
        try:
            ret.append(self.key)
        except AttributeError:
            pass
        return ret

    def __setstate__(self, state):
        self.prev = weakref.ref(state[0])
        self.next = weakref.ref(state[1])
        if len(state) == 3:
            self.key = state[2]


class OrderedSet(MutableSet):
    'Set the remembers the order elements were added'
    # Big-O running times for all methods are the same as for regular sets.
    # The internal self.__map dictionary maps keys to links in a doubly linked list.
    # The circular doubly linked list starts and ends with a sentinel element.
    # The sentinel element never gets deleted (this simplifies the algorithm).
    # The prev/next links are weakref proxies (to prevent circular references).
    # Individual links are kept alive by the hard reference in self.__map.
    # Those hard references disappear when a key is deleted from an OrderedSet.

    # Added by IG-- pre-existing theano code expected sets
    #   to have this method
    def update(self, iterable):
        check_deterministic(iterable)
        self |= iterable

    def __init__(self, iterable=None):
        # Checks added by IG
        check_deterministic(iterable)
        self.__root = root = Link()         # sentinel node for doubly linked list
        root.prev = root.next = weakref.ref(root)
        self.__map = {}                     # key --> link
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.__map)

    def __contains__(self, key):
        return key in self.__map

    def add(self, key):
        # Store new key in a new link at the end of the linked list
        if key not in self.__map:
            self.__map[key] = link = Link()
            root = self.__root
            last = root.prev
            link.prev, link.next, link.key = last, weakref.ref(root), key
            last().next = root.prev = weakref.ref(link)

    def union(self, s):
        check_deterministic(s)
        n = self.copy()
        for elem in s:
            if elem not in n:
                n.add(elem)
        return n

    def intersection_update(self, s):
        l = []
        for elem in self:
            if elem not in s:
                l.append(elem)
        for elem in l:
            self.remove(elem)
        return self

    def difference_update(self, s):
        check_deterministic(s)
        for elem in s:
            if elem in self:
                self.remove(elem)
        return self

    def copy(self):
        n = OrderedSet()
        n.update(self)
        return n

    def discard(self, key):
        # Remove an existing item using self.__map to find the link which is
        # then removed by updating the links in the predecessor and successors.
        if key in self.__map:
            link = self.__map.pop(key)
            link.prev().next = link.next
            link.next().prev = link.prev

    def __iter__(self):
        # Traverse the linked list in order.
        root = self.__root
        curr = root.next()
        while curr is not root:
            yield curr.key
            curr = curr.next()

    def __reversed__(self):
        # Traverse the linked list in reverse order.
        root = self.__root
        curr = root.prev()
        while curr is not root:
            yield curr.key
            curr = curr.prev()

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        if last:
            key = next(reversed(self))
        else:
            key = next(iter(self))
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        # Note that we implement only the comparison to another
        # `OrderedSet`, and not to a regular `set`, because otherwise we
        # could have a non-symmetric equality relation like:
        #       my_ordered_set == my_set and my_set != my_ordered_set
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        elif isinstance(other, set):
            # Raise exception to avoid confusion.
            raise TypeError(
                'Cannot compare an `OrderedSet` to a `set` because '
                'this comparison cannot be made symmetric: please '
                'manually cast your `OrderedSet` into `set` before '
                'performing this comparison.')
        else:
            return NotImplemented

# end of http://code.activestate.com/recipes/576696/ }}}

if __name__ == '__main__':
    print(list(OrderedSet('abracadaba')))
    print(list(OrderedSet('simsalabim')))
    print(OrderedSet('boom') == OrderedSet('moob'))
    print(OrderedSet('boom') == 'moob')
