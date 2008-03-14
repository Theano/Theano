
from copy import copy

import utils


__all__ = ['inputs',
           'results_and_orphans', 'results', 'orphans',
           'ops',
           'clone', 'clone_get_equiv',
           'io_toposort',
           'default_leaf_formatter', 'default_node_formatter',
           'op_as_string',
           'as_string',
           'Graph']


is_result = utils.attr_checker('owner', 'index')
is_op = utils.attr_checker('inputs', 'outputs')


def inputs(o):
    """
    o -> list of output Results

    Returns the set of inputs necessary to compute the outputs in o
    such that input.owner is None.
    """
    results = set()
    def seek(r):
        op = r.owner
        if op is None:
            results.add(r)
        else:
            for input in op.inputs:
                seek(input)
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
            if r.owner is not None:
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


def clone(i, o, copy_inputs = False):
    """
    i -> list of input Results
    o -> list of output Results
    copy_inputs -> if True, the inputs will be copied (defaults to False)

    Copies the subgraph contained between i and o and returns the
    outputs of that copy (corresponding to o).
    """
    equiv = clone_get_equiv(i, o)
    return [equiv[output] for output in o]


def clone_get_equiv(i, o, copy_inputs_and_orphans = False):
    """
    i -> list of input Results
    o -> list of output Results
    copy_inputs_and_orphans -> if True, the inputs and the orphans
         will be replaced in the cloned graph by copies available in
         the equiv dictionary returned by the function (copy_inputs
         defaults to False)

    Returns equiv a dictionary mapping each result and op in the
    graph delimited by i and o to a copy (akin to deepcopy's memo).
    """

    d = {}

    for input in i:
        if copy_inputs_and_orphans:
            d[input] = copy(input)
        else:
            d[input] = input

    def clone_helper(result):
        if result in d:
            return d[result]
        op = result.owner
        if not op:
            if copy_inputs_and_orphans:
                d[result] = copy(result)
            else:
                d[result] = result
            return d[result]
        else:
            new_op = op.clone_with_new_inputs(*[clone_helper(input) for input in op.inputs])
            d[op] = new_op
            for output, new_output in zip(op.outputs, new_op.outputs):
                d[output] = new_output
            return d[result]

    for output in o:
        clone_helper(output)

    return d


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
    return utils.toposort(prereqs_d)


default_leaf_formatter = str
default_node_formatter = lambda op, argstrings: "%s(%s)" % (op.__class__.__name__,
                                                            ", ".join(argstrings))

def op_as_string(i, op,
                 leaf_formatter = default_leaf_formatter,
                 node_formatter = default_node_formatter):
    strs = as_string(i, op.inputs, leaf_formatter, node_formatter)
    return node_formatter(op, strs)


def as_string(i, o,
              leaf_formatter = default_leaf_formatter,
              node_formatter = default_node_formatter):
    """
    i -> list of input Results
    o -> list of output Results
    leaf_formatter -> function that takes a result and returns a string to describe it
    node_formatter -> function that takes an op and the list of strings corresponding
                      to its arguments and returns a string to describe it

    Returns a string representation of the subgraph between i and o. If the same
    op is used by several other ops, the first occurrence will be marked as
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
        return multi.index(x) + 1

    def describe(r):
        if r.owner is not None and r not in i:
            op = r.owner
            idx = op.outputs.index(r)
            if idx == op._default_output_idx:
                idxs = ""
            else:
                idxs = "::%i" % idx
            if op in done:
                return "*%i%s" % (multi_index(x), idxs)
            else:
                done.add(op)
                s = node_formatter(op, [describe(input) for input in op.inputs])
                if op in multi:
                    return "*%i -> %s" % (multi_index(x), s)
                else:
                    return s
        else:
            return leaf_formatter(r)

    return [describe(output) for output in o]




class Graph:
    """
    Object-oriented wrapper for all the functions in this module.
    """

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








