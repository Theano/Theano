MutableSet = None
try:
    from collections import MutableSet
except ImportError:
    # Python 2.4
    pass
from theano.gof.python25 import OrderedDict

if MutableSet is not None:
    # From http://code.activestate.com/recipes/576694/
    KEY, PREV, NEXT = range(3)

    class OrderedSet(MutableSet):

        # Added by IG-- pre-existing theano code expected sets
        #   to have this method
        def update(self, container):
            # only allowed ordered containers
            assert isinstance(container, (list, OrderedSet))
            for elem in container:
                self.add(elem)

        def __init__(self, iterable=None):
            self.end = end = []
            end += [None, end, end]         # sentinel node for doubly linked list
            self.map = {}                   # key --> [key, prev, next]
            if iterable is not None:
                self |= iterable

        def __len__(self):
            return len(self.map)

        def __contains__(self, key):
            return key in self.map

        def add(self, key):
            if key not in self.map:
                end = self.end
                curr = end[PREV]
                curr[NEXT] = end[PREV] = self.map[key] = [key, curr, end]

        def discard(self, key):
            if key in self.map:
                key, prev, next = self.map.pop(key)
                prev[NEXT] = next
                next[PREV] = prev

        def __iter__(self):
            end = self.end
            curr = end[NEXT]
            while curr is not end:
                yield curr[KEY]
                curr = curr[NEXT]

        def __reversed__(self):
            end = self.end
            curr = end[PREV]
            while curr is not end:
                yield curr[KEY]
                curr = curr[PREV]

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
            if isinstance(other, OrderedSet):
                return len(self) == len(other) and list(self) == list(other)
            return set(self) == set(other)

        def __del__(self):
            self.clear()                    # remove circular references
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
            # only allowed ordered containers
            assert isinstance(container, (list, OrderedSet))
            for elem in container:
                self.add(elem)

        def add(self, key):
            self.data[key] = None

        def __len__(self, key):
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
            return type(self) == type(other) and \
                    self.data == other.data

        def __del__(self):
            # Remove circular references
            self.data.clear()






if __name__ == '__main__':
    print(OrderedSet('abracadaba'))
    print(OrderedSet('simsalabim'))
