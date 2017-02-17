from __future__ import absolute_import, print_function, division
from theano.gof.sched import (make_dependence_cmp, sort_apply_nodes,
                              reverse_dict, _toposort, posort)

from theano import tensor
from theano.gof.graph import io_toposort
from theano.compat import cmp


def test_dependence():
    dependence = make_dependence_cmp()

    x = tensor.matrix('x')
    y = tensor.dot(x * 2, x + 1)
    nodes = io_toposort([x], [y])

    for a, b in zip(nodes[:-1], nodes[1:]):
        assert dependence(a, b) <= 0


def test_sort_apply_nodes():
    x = tensor.matrix('x')
    y = tensor.dot(x * 2, x + 1)

    def str_cmp(a, b):
        return cmp(str(a), str(b))  # lexicographical sort

    nodes = sort_apply_nodes([x], [y], cmps=[str_cmp])

    for a, b in zip(nodes[:-1], nodes[1:]):
        assert str(a) <= str(b)


def test_reverse_dict():
    d = {'a': (1, 2), 'b': (2, 3), 'c': ()}
    # Python 3.3 enable by default random hash for dict.
    # This change the order of traversal, so this can give 2 outputs
    assert (reverse_dict(d) == {1: ('a',), 2: ('a', 'b'), 3: ('b',)} or
            reverse_dict(d) == {1: ('a',), 2: ('b', 'a'), 3: ('b',)})


def test__toposort():
    edges = {1: set((4, 6, 7)), 2: set((4, 6, 7)),
             3: set((5, 7)), 4: set((6, 7)), 5: set((7,))}
    order = _toposort(edges)
    assert not any(a in edges.get(b, ()) for i, a in enumerate(order)
                   for b in order[i:])


def test_posort_easy():
    nodes = "asdfghjkl"

    def mycmp(a, b):
        if a < b:
            return -1
        elif a > b:
            return 1
        else:
            return 0

    assert posort(nodes, mycmp) == list("adfghjkls")


def test_posort():
    l = list(range(1, 20))
    cmps = [lambda a, b: a % 10 - b % 10,
            lambda a, b: (a / 10) % 2 - (b / 10) % 2,
            lambda a, b: a - b]
    assert (posort(l, *cmps) ==
            [10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19])
