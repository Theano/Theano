MutableSet = None
try:
    from collections import MutableSet
except ImportError:
    # Python 2.4
    pass
from theano.gof.python25 import OrderedDict
import types

def check_deterministic(iterable):
    # Most places where OrderedSet is used, theano interprets any exception
    # whatsoever as a problem that an optimization introduced into the graph.
    # If I raise a TypeError when the DestoryHandler tries to do something
    # non-deterministic, it will just result in optimizations getting ignored.
    # So I must use an assert here. In the long term we should fix the rest of
    # theano to use exceptions correctly, so that this can be a TypeError.
    if iterable is not None:
        assert isinstance(iterable, (
            list, tuple, OrderedSet, types.GeneratorType, basestring))

if MutableSet is not None:
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
    ## {{{ http://code.activestate.com/recipes/576696/ (r5)
    import collections
    from weakref import proxy

    class Link(object):
        __slots__ = 'prev', 'next', 'key', '__weakref__'

    class OrderedSet(collections.MutableSet):
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
            root.prev = root.next = root
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
                link.prev, link.next, link.key = last, root, key
                last.next = root.prev = proxy(link)

        def discard(self, key):
            # Remove an existing item using self.__map to find the link which is
            # then removed by updating the links in the predecessor and successors.
            if key in self.__map:
                link = self.__map.pop(key)
                link.prev.next = link.next
                link.next.prev = link.prev

        def __iter__(self):
            # Traverse the linked list in order.
            root = self.__root
            curr = root.next
            while curr is not root:
                yield curr.key
                curr = curr.next

        def __reversed__(self):
            # Traverse the linked list in reverse order.
            root = self.__root
            curr = root.prev
            while curr is not root:
                yield curr.key
                curr = curr.prev

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

    ## end of http://code.activestate.com/recipes/576696/ }}}

else:
    # Python 2.4
    class OrderedSet(object):
        """
        An implementation of OrderedSet based on the keys of
        an OrderedDict.
        """
        def __init__(self, iterable=None):
            self.data = OrderedDict()
            if iterable is not None:
                self.update(iterable)

        def update(self, container):
            check_deterministic(container)
            for elem in container:
                self.add(elem)

        def add(self, key):
            self.data[key] = None

        def __len__(self):
            return len(self.data)

        def __contains__(self, key):
            return key in self.data

        def discard(self, key):
            if key in self.data:
                del self.data[key]

        def remove(self, key):
            if key in self.data:
                del self.data[key]
            else:
                raise KeyError(key)

        def __iter__(self):
            return self.data.__iter__()

        def __reversed__(self):
            return self.data.__reversed__()

        def pop(self, last=True):
            raise NotImplementedError()

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

        # NB: Contrary to the other implementation above, we do not override
        # the `__del__` method. On one hand, this is not needed since this
        # implementation does not add circular references. Moreover, one should
        # not clear the underlying dictionary holding the data as soon as the
        # ordered set is cleared from memory, because there may still be
        # pointers to this dictionary.


if __name__ == '__main__':
    print list(OrderedSet('abracadaba'))
    print list(OrderedSet('simsalabim'))
    print OrderedSet('boom') == OrderedSet('moob')
    print OrderedSet('boom') == 'moob'
