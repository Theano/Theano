
from copy import copy

from result import BrokenLink, BrokenLinkError
from op import Op
import utils


__all__ = ['inputs',
           'results_and_orphans', 'results', 'orphans',
           'ops',
           'clone', 'clone_get_equiv',
           'io_toposort',
           'as_string',
           'Graph']


def inputs(o, repair = False):
    """
    o -> list of output Results

    Returns the set of inputs necessary to compute the outputs in o
    such that input.owner is None.
    """
    results = set()
    def seek(r):
        if isinstance(r, BrokenLink):
            raise BrokenLinkError
        op = r.owner
        if op is None:
            results.add(r)
        else:
            for i in range(len(op.inputs)):
                try:
                    seek(op.inputs[i])
                except BrokenLinkError:
                    if repair:
                        op.refresh()
                        seek(op.inputs[i])
                    else:
                        raise
    for output in o:
        seek(output)
    return results


def results_and_orphans(i, o):
    """
    i -> list of input Results
    o -> list of output Results

    Returns the pair (results, orphans). The former is the set of
    Results that are involved in the subgraph that lies between i and
    o. This includes i, o, orphans(i, o) and all results of all
    intermediary steps from i to o. The second element of the returned
    pair is orphans(i, o).
    """
    results = set(o)
    results.update(i)
    incomplete_paths = []

    def helper(r, path):
        if isinstance(r, BrokenLink):
            raise BrokenLinkError
        if r in i:
            results.update(path)
        elif r.owner is None:
            incomplete_paths.append(path)
        else:
            op = r.owner
            for r2 in op.inputs:
                helper(r2, path + [r2])

    for output in o:
        helper(output, [])

    orphans = set()
    for path in incomplete_paths:
        for r in path:
            if r not in results:
                orphans.add(r)
                break

    return results, orphans


def ops(i, o):
    """
    i -> list of input Results
    o -> list of output Results

    Returns the set of ops that are contained within the subgraph
    that lies between i and o, including the owners of the Results in
    o and intermediary ops between i and o, but not the owners of the
    Results in i.
    """
    ops = set()
    results, orphans = results_and_orphans(i, o)
    for r in results:
        if r not in i and r not in orphans:
            ops.add(r.owner)
    return ops


def results(i, o):
    """
    i -> list of input Results
    o -> list of output Results

    Returns the set of Results that are involved in the subgraph
    that lies between i and o. This includes i, o, orphans(i, o)
    and all values of all intermediary steps from i to o.
    """
    return results_and_orphans(i, o)[0]


def orphans(i, o):
    """
    i -> list of input Results
    o -> list of output Results

    Returns the set of Results which one or more Results in o depend
    on but are neither in i nor in the subgraph that lies between
    i and o.

    e.g. orphans([x], [(x+y).out]) => [y]
    """
    return results_and_orphans(i, o)[1]


def clone(i, o):
    """
    i -> list of input Results
    o -> list of output Results

    Copies the subgraph contained between i and o and returns the
    outputs of that copy (corresponding to o). The input Results in
    the list are _not_ copied and the new graph refers to the
    originals.
    """
    new_o, equiv = clone_get_equiv(i, o)
    return new_o


def clone_get_equiv(i, o, copy_inputs = False):
    """
    i -> list of input Results
    o -> list of output Results

    Returns (new_o, equiv) where new_o are the outputs of a copy of
    the whole subgraph bounded by i and o and equiv is a dictionary
    that maps the original ops and results found in the subgraph to
    their copy (akin to deepcopy's memo). See clone for more details.
    """

    d = {}

    for op in ops(i, o):
        d[op] = copy(op)

    for old_op, op in d.items():
        for old_output, output in zip(old_op.outputs, op.outputs):
            d[old_output] = output
        for i, input in enumerate(op.inputs):
            owner = input.owner
            if owner in d:
                op._inputs[i] = d[owner].outputs[input._index]

    return [[d[output] for output in o], d]


def io_toposort(i, o, orderings = {}):
    """
    i -> list of input Results
    o -> list of output Results
    orderings -> {op: [requirements for op]} (defaults to {})

    Returns an ordered list of Ops that belong in the subgraph between
    i and o which respects the following constraints:
    - all inputs in i are assumed to be already computed
    - the Ops that compute an Op's inputs must be computed before it
    - the orderings specified in the optional orderings parameter must be satisfied

    Note that this function does not take into account ordering information
    related to destructive operations or other special behavior.
    """
    prereqs_d = copy(orderings)
    all = ops(i, o)
    for op in all:
        prereqs_d.setdefault(op, set()).update(set([input.owner for input in op.inputs if input.owner and input.owner in all]))
#        prereqs_d[op] = set([input.owner for input in op.inputs if input.owner and input.owner in all])
    return utils.toposort(prereqs_d)


def as_string(i, o):
    """
    i -> list of input Results
    o -> list of output Results

    Returns a string representation of the subgraph between i and o. If the same
    Op is used by several other ops, the first occurrence will be marked as
    '*n -> description' and all subsequent occurrences will be marked as '*n',
    where n is an id number (ids are attributed in an unspecified order and only
    exist for viewing convenience).
    """

    multi = set()
    seen = set()
    for op in ops(i, o):
        for input in op.inputs:
            op2 = input.owner
            if input in i or op2 is None:
                continue
            if op2 in seen:
                multi.add(op2)
            else:
                seen.add(input.owner)
    multi = [x for x in multi]
    done = set()

    def multi_index(x):
        try:
            return multi.index(x) + 1
        except:
            return 999

    def describe(x, first = False):
        if isinstance(x, Result):
            done.add(x)
            if x.owner is not None and x not in i:
                op = x.owner
                idx = op.outputs.index(x)
                if idx:
                    s = describe(op, first) + "." + str(idx)
                else:
                    s = describe(op, first)
                return s
            else:
                return str(id(x))
                
        elif isinstance(x, Op):
            if x in done:
                return "*%i" % multi_index(x)
            else:
                done.add(x)
                if not first and hasattr(x, 'name') and x.name is not None:
                    return x.name
                s = x.__class__.__name__ + "(" + ", ".join([describe(v) for v in x.inputs]) + ")"
                if x in multi:
                    return "*%i -> %s" % (multi_index(x), s)
                else:
                    return s
        
        else:
            raise TypeError("Cannot print type: %s" % x.__class__)

    return "[" + ", ".join([describe(x, True) for x in o]) + "]"


# Op.__str__ = lambda self: as_string(inputs(self.outputs), self.outputs)[1:-1]
# Result.__str__ = lambda self: as_string(inputs([self]), [self])[1:-1]



class Graph:

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def ops(self):
        return ops(self.inputs, self.outputs)

    def values(self):
        return values(self.inputs, self.outputs)

    def orphans(self):
        return orphans(self.inputs, self.outputs)

    def io_toposort(self):
        return io_toposort(self.inputs, self.outputs)

    def toposort(self):
        return self.io_toposort()

    def clone(self):
        o = clone(self.inputs, self.outputs)
        return Graph(self.inputs, o)

    def __str__(self):
        return as_string(self.inputs, self.outputs)








