
from copy import copy

import utils
from utils import object2
        

def deprecated(f):
    printme = True
    def g(*args, **kwargs):
        if printme:
            print 'gof.graph.%s deprecated: April 29' % f.__name__
            printme = False
        return f(*args, **kwargs)
    return g


class Apply(object2):
    """
    Note: it is illegal for an output element to have an owner != self
    """
    #__slots__ = ['op', 'inputs', 'outputs']
    def __init__(self, op, inputs, outputs):
        self.op = op
        self.inputs = []

        ## filter inputs to make sure each element is a Result
        for input in inputs:
            if isinstance(input, Result):
                self.inputs.append(input)
#             elif isinstance(input, Type):
#                 self.inputs.append(Result(input, None, None))
            else:
                raise TypeError("The 'inputs' argument to Apply must contain Result instances, not %s" % input)
        self.outputs = []
        ## filter outputs to make sure each element is a Result
        for i, output in enumerate(outputs):
            if isinstance(output, Result):
                if output.owner is None:
                    output.owner = self
                    output.index = i
                elif output.owner is not self or output.index != i:
                    raise ValueError("All output results passed to Apply must belong to it.")
                self.outputs.append(output)
#             elif isinstance(output, Type):
#                 self.outputs.append(Result(output, self, i))
            else:
                raise TypeError("The 'outputs' argument to Apply must contain Result instances with no owner, not %s" % output)
    def default_output(self):
        print 'default_output deprecated: April 29'
        """
        Returns the default output for this Node, typically self.outputs[0].
        Depends on the value of node.op.default_output
        """
        do = self.op.default_output
        if do < 0:
            raise AttributeError("%s does not have a default output." % self.op)
        elif do > len(self.outputs):
            raise AttributeError("default output for %s is out of range." % self.op)
        return self.outputs[do]
    out = property(default_output, 
                   doc = "Same as self.outputs[0] if this Op's has_default_output field is True.")
    def __str__(self):
        return op_as_string(self.inputs, self)
    def __repr__(self):
        return str(self)
    def __asapply__(self):
        return self
    nin = property(lambda self: len(self.inputs))
    nout = property(lambda self: len(self.outputs))


class Result(object2):
    #__slots__ = ['type', 'owner', 'index', 'name']
    def __init__(self, type, owner = None, index = None, name = None):
        self.type = type
        self.owner = owner
        self.index = index
        self.name = name

    def __str__(self):
        if self.name is not None:
            return self.name
        if self.owner is not None:
            op = self.owner.op
            if self.index == op.default_output:
                return str(self.owner.op) + ".out"
            else:
                return str(self.owner.op) + "." + str(self.index)
        else:
            return "?::" + str(self.type)
    def __repr__(self):
        return str(self)
    @deprecated
    def __asresult__(self):
        return self

class Constant(Result):
    #__slots__ = ['data']
    def __init__(self, type, data, name = None):
        Result.__init__(self, type, None, None, name)
        self.data = type.filter(data)
        self.indestructible = True
    def equals(self, other):
        # this does what __eq__ should do, but Result and Apply should always be hashable by id
        return isinstance(other, Constant) and self.signature() == other.signature()
    def signature(self):
        return (self.type, self.data)
    def __str__(self):
        if self.name is not None:
            return self.name
        return str(self.data) #+ "::" + str(self.type)

@deprecated
def as_result(x):
    if isinstance(x, Result):
        return x
#     elif isinstance(x, Type):
#         return Result(x, None, None)
    elif hasattr(x, '__asresult__'):
        r = x.__asresult__()
        if not isinstance(r, Result):
            raise TypeError("%s.__asresult__ must return a Result instance" % x, (x, r))
        return r
    else:
        raise TypeError("Cannot wrap %s in a Result" % x)

@deprecated
def as_apply(x):
    if isinstance(x, Apply):
        return x
    elif hasattr(x, '__asapply__'):
        node = x.__asapply__()
        if not isinstance(node, Apply):
            raise TypeError("%s.__asapply__ must return an Apply instance" % x, (x, node))
        return node
    else:
        raise TypeError("Cannot map %s to Apply" % x)
    



@deprecated
def inputs(o):
    """
    @type o: list
    @param o: output L{Result}s

    Returns the set of inputs necessary to compute the outputs in o
    such that input.owner is None.
    """
    print 'gof.graph.inputs deprecated: April 29'
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


def results_and_orphans(i, o, except_unreachable_input=False):
    """
    @type i: list
    @param i: input L{Result}s
    @type o: list
    @param o: output L{Result}s

    Returns the pair (results, orphans). The former is the set of
    L{Result}s that are involved in the subgraph that lies between i and
    o. This includes i, o, orphans(i, o) and all results of all
    intermediary steps from i to o. The second element of the returned
    pair is orphans(i, o).
    """
    results = set()
    i = set(i)
    results.update(i)
    incomplete_paths = []
    reached = set()

    def helper(r, path):
        if r in i:
            reached.add(r)
            results.update(path)
        elif r.owner is None:
            incomplete_paths.append(path)
        else:
            op = r.owner
            for r2 in op.inputs:
                helper(r2, path + [r2])

    for output in o:
        helper(output, [output])

    orphans = set()
    for path in incomplete_paths:
        for r in path:
            if r not in results:
                orphans.add(r)
                break

    if except_unreachable_input and len(i) != len(reached):
        raise Exception(results_and_orphans.E_unreached)

    results.update(orphans)

    return results, orphans
results_and_orphans.E_unreached = 'there were unreachable inputs'


def ops(i, o):
    """
    @type i: list
    @param i: input L{Result}s
    @type o: list
    @param o: output L{Result}s

    Returns the set of ops that are contained within the subgraph
    that lies between i and o, including the owners of the L{Result}s in
    o and intermediary ops between i and o, but not the owners of the
    L{Result}s in i.
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
    @type i: list
    @param i: input L{Result}s
    @type o: list
    @param o: output L{Result}s

    Returns the set of Results that are involved in the subgraph
    that lies between i and o. This includes i, o, orphans(i, o)
    and all values of all intermediary steps from i to o.
    """
    return results_and_orphans(i, o)[0]


def orphans(i, o):
    """
    @type i: list
    @param i: input L{Result}s
    @type o: list
    @param o: output L{Result}s

    Returns the set of Results which one or more Results in o depend
    on but are neither in i nor in the subgraph that lies between
    i and o.

    e.g. orphans([x], [(x+y).out]) => [y]
    """
    return results_and_orphans(i, o)[1]


def clone(i, o, copy_inputs = False):
    """
    @type i: list
    @param i: input L{Result}s
    @type o: list
    @param o: output L{Result}s
    @type copy_inputs: bool
    @param copy_inputs: if True, the inputs will be copied (defaults to False)

    Copies the subgraph contained between i and o and returns the
    outputs of that copy (corresponding to o).
    """
    equiv = clone_get_equiv(i, o)
    return [equiv[output] for output in o]


def clone_get_equiv(i, o, copy_inputs_and_orphans = False):
    """
    @type i: list
    @param i: input L{Result}s
    @type o: list
    @param o: output L{Result}s
    @type copy_inputs_and_orphans: bool
    @param copy_inputs_and_orphans: if True, the inputs and the orphans
         will be replaced in the cloned graph by copies available
         in the equiv dictionary returned by the function
         (copy_inputs_and_orphans defaults to False)

    @rtype: a dictionary
    @return: equiv mapping each L{Result} and L{Op} in the
    graph delimited by i and o to a copy (akin to deepcopy's memo).
    """

    d = {}

    for input in i:
        if copy_inputs_and_orphans:
            cpy = copy(input)
            cpy.owner = None
            cpy.index = None
            d[input] = cpy
        else:
            d[input] = input

    def clone_helper(result):
        if result in d:
            return d[result]
        node = result.owner
        if node is None: # result is an orphan
            if copy_inputs_and_orphans:
                cpy = copy(result)
                cpy.owner = None
                cpy.index = None
                d[result] = cpy
            else:
                d[result] = result
            return d[result]
        else:
            new_node = copy(node)
            new_node.inputs = [clone_helper(input) for input in node.inputs]
            new_node.outputs = []
            for output in node.outputs:
                new_output = copy(output)
                new_output.owner = new_node
                new_node.outputs.append(new_output)
#             new_node = Apply(node.op,
#                              [clone_helper(input) for input in node.inputs],
#                              [output.type for output in node.outputs])
            d[node] = new_node
            for output, new_output in zip(node.outputs, new_node.outputs):
                d[output] = new_output
            return d[result]

    for output in o:
        clone_helper(output)

    return d

#     d = {}

#     for input in i:
#         if copy_inputs_and_orphans:
#             d[input] = copy(input)
#         else:
#             d[input] = input

#     def clone_helper(result):
#         if result in d:
#             return d[result]
#         op = result.owner
#         if not op: # result is an orphan
#             if copy_inputs_and_orphans:
#                 d[result] = copy(result)
#             else:
#                 d[result] = result
#             return d[result]
#         else:
#             new_op = op.clone_with_new_inputs(*[clone_helper(input) for input in op.inputs])
#             d[op] = new_op
#             for output, new_output in zip(op.outputs, new_op.outputs):
#                 d[output] = new_output
#             return d[result]

#     for output in o:
#         clone_helper(output)

#     return d


def io_toposort(i, o, orderings = {}):
    """
    @type i: list
    @param i: input L{Result}s
    @type o: list
    @param o: output L{Result}s
    @param orderings: {op: [requirements for op]} (defaults to {})

    @rtype: ordered list
    @return: L{Op}s that belong in the subgraph between i and o which
    respects the following constraints:
     - all inputs in i are assumed to be already computed
     - the L{Op}s that compute an L{Op}'s inputs must be computed before it
     - the orderings specified in the optional orderings parameter must be satisfied

    Note that this function does not take into account ordering information
    related to destructive operations or other special behavior.
    """
    prereqs_d = copy(orderings)
    all = ops(i, o)
    for op in all:
        asdf = set([input.owner for input in op.inputs if input.owner and input.owner in all])
        prereqs_d.setdefault(op, set()).update(asdf)
    return utils.toposort(prereqs_d)


default_leaf_formatter = str
default_node_formatter = lambda op, argstrings: "%s(%s)" % (op.op,
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
    @type i: list
    @param i: input L{Result}s
    @type o: list
    @param o: output L{Result}s
    @type leaf_formatter: function
    @param leaf_formatter: takes a L{Result} and returns a string to describe it
    @type node_formatter: function
    @param node_formatter: takes an L{Op} and the list of strings
    corresponding to its arguments and returns a string to describe it

    Returns a string representation of the subgraph between i and o. If the same
    op is used by several other ops, the first occurrence will be marked as
    '*n -> description' and all subsequent occurrences will be marked as '*n',
    where n is an id number (ids are attributed in an unspecified order and only
    exist for viewing convenience).
    """

    orph = orphans(i, o)
    
    multi = set()
    seen = set()
    for output in o:
        op = output.owner
        if op in seen:
            multi.add(op)
        else:
            seen.add(op)
    for op in ops(i, o):
        for input in op.inputs:
            op2 = input.owner
            if input in i or input in orph or op2 is None:
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
        if r.owner is not None and r not in i and r not in orph:
            op = r.owner
            idx = op.outputs.index(r)
            if idx == op.op.default_output:
                idxs = ""
            else:
                idxs = "::%i" % idx
            if op in done:
                return "*%i%s" % (multi_index(op), idxs)
            else:
                done.add(op)
                s = node_formatter(op, [describe(input) for input in op.inputs])
                if op in multi:
                    return "*%i -> %s" % (multi_index(op), s)
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








