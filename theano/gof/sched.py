from graph import list_of_nodes
from theano.gof.python25 import any, defaultdict


## {{{ http://code.activestate.com/recipes/578231/ (r1)
# Copyright (c) Oren Tirosh 2012
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(defaultdict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__
## end of http://code.activestate.com/recipes/578231/ }}}


def make_depends():
    @memodict
    def depends((a, b)):
        """ Returns True if a depends on b """
        return (any(bout in a.inputs for bout in b.outputs)
                 or any(depends((ainp.owner, b)) for ainp in a.inputs
                                                  if ainp.owner))
    return depends

def make_dependence_cmp():
    """ Create a comparator to represent the dependence of nodes in a graph """

    depends = make_depends()

    def dependence(a, b):
        """ A cmp function for nodes in a graph - does a depend on b?

        Returns positive number if a depends on b
        Returns negative number if b depends on a
        Returns 0 otherwise
        """
        if depends((a, b)): return  1
        if depends((b, a)): return -1
        return 0

    return dependence

def reverse_dict(d):
    """ Reverses direction of dependence dict

    >>> d = {'a': (1, 2), 'b': (2, 3), 'c':()}
    >>> reverse_dict(d)
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}
    """
    result = {}
    for key in d:
        for val in d[key]:
            result[val] = result.get(val, tuple()) + (key, )
    return result

def _toposort(edges):
    """ Topological sort algorithm by Kahn [1] - O(nodes + vertices)

    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges

    >>> _toposort({1: {2, 3}, 2: (3, )})
    [1, 2, 3]

    Closely follows the wikipedia page [2]

    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = dict((k, set(val)) for k, val in incoming_edges.items())
    S = set((v for v in edges if v not in incoming_edges))
    L = []

    while S:
        n = S.pop()
        L.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S.add(m)
    if any(incoming_edges.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    return L

def posort(l, *cmps):
    """ Partially ordered sort with multiple comparators

    Given a list of comparators order the elements in l so that the comparators
    are satisfied as much as possible giving precedence to earlier comparators.

    inputs:
        l - an iterable of nodes in a graph
        cmps - a sequence of comparator functions that describe which nodes
               should come before which others

    outputs:
        a list of nodes which satisfy the comparators as much as possible.

    >>> lower_tens = lambda a, b: a/10 - b/10 # prefer lower numbers div 10
    >>> prefer evens = lambda a, b: a%2 - b%2 # prefer even numbers
    >>> posort(range(20), lower_tens, prefer_evens)
    [0, 8, 2, 4, 6, 1, 3, 5, 7, 9, 16, 18, 10, 12, 14, 17, 19, 11, 13, 15]

    implemented with _toposort """
    comes_before = dict((a, set()) for a in l)
    comes_after  = dict((a, set()) for a in l)

    def add_links(a, b): # b depends on a
        comes_after[a].add(b)
        comes_after[a].update(comes_after[b])
        for c in comes_before[a]:
            comes_after[c].update(comes_after[a])
        comes_before[b].add(a)
        comes_before[b].update(comes_before[a])
        for c in comes_after[b]:
            comes_before[c].update(comes_before[b])

    def check():
        """ Tests for cycles in manufactured edges """
        for a in l:
            for b in l:
                assert not(b in comes_after[a] and a in comes_after[b])

    for cmp in cmps:
        for a in l:
            for b in l:
                if cmp(a, b) < 0: # a wants to come before b
                    # if this wouldn't cause a cycle and isn't already known
                    if not b in comes_before[a] and not b in comes_after[a]:
                        add_links(a, b)
    # check() # debug code

    return _toposort(comes_after)

def sort_apply_nodes(inputs, outputs, cmps):
    """ Order a graph of apply nodes according to a list of comparators

    The following example sorts first by dependence of nodes (this is a
    topological sort) and then by lexicographical ordering (nodes that start
    with 'E' come before nodes that start with 'I' if there is no dependence.

    >>> from theano.gof.graph import sort_apply_nodes, dependence
    >>> from theano.tensor import matrix, dot
    >>> x = matrix('x')
    >>> y = dot(x*2, x+1)
    >>> str_cmp = lambda a, b: cmp(str(a), str(b)) # lexicographical sort
    >>> sort_apply_nodes([x], [y], cmps=[dependence, str_cmp])
    [Elemwise{add,no_inplace}(x, InplaceDimShuffle{x,x}.0),
     InplaceDimShuffle{x,x}(TensorConstant{2}),
     Elemwise{mul,no_inplace}(x, InplaceDimShuffle{x,x}.0),
     InplaceDimShuffle{x,x}(TensorConstant{1}),
     dot(Elemwise{mul,no_inplace}.0, Elemwise{add,no_inplace}.0)]
    """

    return posort(list_of_nodes(inputs, outputs), *cmps)

def sort_schedule_fn(*cmps):
    """ Make a schedule function from comparators

    See also:
        sort_apply_nodes
    """
    dependence = make_dependence_cmp()
    cmps = (dependence,) + cmps
    def schedule(fgraph):
        """ Order nodes in a FunctionGraph """
        return sort_apply_nodes(fgraph.inputs, fgraph.outputs, cmps)
    return schedule

def key_to_cmp(key):
    def key_cmp(a, b):
        return cmp(key(a), key(b))
    return key_cmp
