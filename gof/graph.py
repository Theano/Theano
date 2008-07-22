
from copy import copy
from collections import deque

import utils


class Apply(utils.object2):
    """
    Represents the application of an Op on input Results, producing output
    Results. These should be instantiated by an Op's make_node function.
    """
    #__slots__ = ['op', 'inputs', 'outputs']
    def __init__(self, op, inputs, outputs):
        """
        Sets self.op, self.inputs, self.outputs to the respective parameter
        in the arguments list.

        The owner field of each output in the outputs list will be set to
        self.

        Note: it is illegal for an output element to have an owner that is
        not None, unless it already points to self.
        """
        self.op = op
        self.inputs = []
        self.tag = utils.scratchpad()

        ## filter inputs to make sure each element is a Result
        for input in inputs:
            if isinstance(input, Result):
                self.inputs.append(input)
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
            else:
                raise TypeError("The 'outputs' argument to Apply must contain Result instances with no owner, not %s" % output)
    def default_output(self):
        """
        Returns the default output for this node. If there is only one
        output, it will be returned. Else, it will consult the value of
        node.op.default_output to decide which output to return.
        """
        do = getattr(self.op, 'default_output', None)
        if do is None:
            if len(self.outputs) == 1:
                return self.outputs[0]
            else:
                raise AttributeError("%s.default_output should be an output index." % self.op)
        elif do < 0 or do >= len(self.outputs):
            raise AttributeError("%s.default_output is out of range." % self.op)
        return self.outputs[do]
    out = property(default_output, 
                   doc = "same as self.default_output()")
    def __str__(self):
        return op_as_string(self.inputs, self)
    def __repr__(self):
        return str(self)
    def __asapply__(self):
        return self
    def clone(self):
#         cp = copy(self)
#         cp.outputs = [output.clone() for output in self.outputs]
#         for output in cp.outputs:
#             output.owner = cp
#         return cp
        cp = self.__class__(self.op, self.inputs, [output.clone() for output in self.outputs])
        cp.tag = copy(self.tag)
        return cp
    def clone_with_new_inputs(self, inputs, strict = True):
        """
        Returns an Apply node with the same op but different inputs. Unless
        strict is False, the type fields of all the inputs must be
        equal to the current ones.

        If strict is True, the outputs of the clone will have the same type as
        the outputs of self. Else, it depends on the types of the new inputs
        and the behavior of the op wrt that.
        """
#         if check_type:
#             for curr, new in zip(self.inputs, inputs):
#                 if not curr.type == new.type:
#                     raise TypeError("Cannot change the type of this input.", curr, new)
#         new_node = self.clone()
#         new_node.inputs = inputs
#         return new_node
        remake_node = False
        for curr, new in zip(self.inputs, inputs):
            if not curr.type == new.type:
                if strict:
                    raise TypeError("Cannot change the type of this input.", curr, new)
                else:
                    remake_node = True
        if remake_node:
            new_node = self.op.make_node(*inputs)
            new_node.tag = copy(self.tag).__update__(new_node.tag)
        else:
            new_node = self.clone()
            new_node.inputs = inputs
        return new_node

    nin = property(lambda self: len(self.inputs), doc = 'same as len(self.inputs)')
    nout = property(lambda self: len(self.outputs), doc = 'same as len(self.outputs)')


class Result(utils.object2):
    """
    Represents the result of some computation (pointed to by its owner field),
    or an input to the graph (if owner is None)
    """
    #__slots__ = ['type', 'owner', 'index', 'name']
    def __init__(self, type, owner = None, index = None, name = None):
        self.tag = utils.scratchpad()
        self.type = type
        if owner is not None and not isinstance(owner, Apply):
            raise TypeError("owner must be an Apply instance", owner)
        self.owner = owner
        if index is not None and not isinstance(index, int):
            raise TypeError("index must be an int", index)
        self.index = index
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string", name)
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
            return "<?>::" + str(self.type)
    def __repr__(self):
        return str(self)
    def clone(self):
        #return copy(self)
        cp = self.__class__(self.type, None, None, self.name)
        cp.tag = copy(self.tag)
        return cp

class Value(Result):
    """
    Result with a data field. The data field is filtered by what is
    provided in the constructor for the Value's type field.

    Its owner field is always None.
    """
    #__slots__ = ['data']
    def __init__(self, type, data, name = None):
        Result.__init__(self, type, None, None, name)
        self.data = type.filter(data)
    def __str__(self):
        if self.name is not None:
            return self.name
        return "<" + str(self.data) + ">" #+ "::" + str(self.type)
    def clone(self):
        return self.__class__(self.type, copy(self.data), self.name)
    def __set_owner(self, value):
        if value is not None:
            raise ValueError("Value instances cannot have an owner.")
    owner = property(lambda self: None, __set_owner)

class Constant(Value):
    """
    Same as Value, but the data it contains cannot be modified.
    """
    #__slots__ = ['data']
    def __init__(self, type, data, name = None):
        Value.__init__(self, type, data, name)
    def equals(self, other):
        # this does what __eq__ should do, but Result and Apply should always be hashable by id
        return type(other) == type(self) and self.signature() == other.signature()
    def signature(self):
        return (self.type, self.data)
    def __str__(self):
        if self.name is not None:
            return self.name
        return str(self.data) #+ "::" + str(self.type)




def stack_search(start, expand, mode='bfs', build_inv = False):
    """Search through L{Result}s, either breadth- or depth-first
    @type start: deque
    @param start: search from these nodes
    @type explore: function
    @param explore: when we get to a node, add explore(node) to the list of
                    nodes to visit.  This function should return a list, or None
    @rtype: list of L{Result}
    @return: the list of L{Result}s in order of traversal.
    
    @note: a L{Result} will appear at most once in the return value, even if it
    appears multiple times in the start parameter.  

    @postcondition: every element of start is transferred to the returned list.
    @postcondition: start is empty.
    """

    if mode not in ('bfs', 'dfs'):
        raise ValueError('mode should be bfs or dfs', mode)
    rval_set = set()
    rval_list = list()
    if mode is 'bfs': start_pop = start.popleft
    else: start_pop = start.pop
    expand_inv = {}
    while start:
        l = start_pop()
        if id(l) not in rval_set:
            rval_list.append(l)
            rval_set.add(id(l))
            expand_l = expand(l)
            if expand_l:
                if build_inv:
                    for r in expand_l:
                        expand_inv.setdefault(r, []).append(l)
                start.extend(expand_l)
    assert len(rval_list) == len(rval_set)
    if build_inv:
        return rval_list, expand_inv
    return rval_list


def inputs(result_list):
    """
    @type result_list: list of L{Result}
    @param result_list: output L{Result}s (from which to search backward through owners)
    @returns: the list of L{Result}s with no owner, in the order found by a
    left-recursive depth-first search started at the L{Result}s in result_list.

    """
    def expand(r):
        if r.owner:
            l = list(r.owner.inputs)
            l.reverse()
            return l
    dfs_results = stack_search(deque(result_list), expand, 'dfs')
    rval = [r for r in dfs_results if r.owner is None]
    #print rval, _orig_inputs(o)
    return rval


def results_and_orphans(i, o):
    """
    """
    def expand(r):
        if r.owner and r not in i:
            l = list(r.owner.inputs) + list(r.owner.outputs)
            l.reverse()
            return l
    results = stack_search(deque(o), expand, 'dfs')
    orphans = [r for r in results if r.owner is None and r not in i]
    return results, orphans


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


def clone(i, o, copy_inputs = True):
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
    equiv = clone_get_equiv(i, o, copy_inputs)
    return [equiv[input] for input in i], [equiv[output] for output in o]


def clone_get_equiv(i, o, copy_inputs_and_orphans = True):
    """
    @type i: list
    @param i: input L{Result}s
    @type o: list
    @param o: output L{Result}s
    @type copy_inputs_and_orphans: bool
    @param copy_inputs_and_orphans: if True, the inputs and the orphans
         will be replaced in the cloned graph by copies available
         in the equiv dictionary returned by the function
         (copy_inputs_and_orphans defaults to True)

    @rtype: a dictionary
    @return: equiv mapping each L{Result} and L{Op} in the
    graph delimited by i and o to a copy (akin to deepcopy's memo).
    """

    d = {}

    for input in i:
        if copy_inputs_and_orphans:
            cpy = input.clone()
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
                cpy = result.clone()
                d[result] = cpy
            else:
                d[result] = result
            return d[result]
        else:
            new_node = node.clone_with_new_inputs([clone_helper(input) for input in node.inputs])
            d[node] = new_node
            for output, new_output in zip(node.outputs, new_node.outputs):
                d[output] = new_output
            return d[result]

    for output in o:
        clone_helper(output)

    return d


# def clone_with_new_inputs(i, o, new_i):
#     equiv = clone_with_new_inputs_get_equiv(i, o, new_i)
#     return [equiv[input] for input in i], [equiv[output] for output in o]


# def clone_with_new_inputs_get_equiv(i, o, new_i, copy_orphans = True):
#     # note: this does not exactly mirror Apply.clone_with_new_inputs
#     # here it is possible to give different types to new_i and then
#     # make_node is called on the ops instead of clone_with_new_inputs
#     # whenever the type is different.

#     d = {}

#     for input, new_input in zip(i, new_i):
#         d[input] = new_input

#     def clone_helper(result):
#         if result in d:
#             return d[result]
#         node = result.owner
#         if node is None: # result is an orphan
#             if copy_orphans:
#                 cpy = result.clone()
#                 d[result] = cpy
#             else:
#                 d[result] = result
#             return d[result]
#         else:
#             cloned_inputs = [clone_helper(input) for input in node.inputs]
#             if any(input != cloned_input for input, cloned_input in zip(node.inputs, cloned_inputs)):
#                 new_node = node.op.make_node(*cloned_inputs)
#             else:
#                 new_node = node.clone_with_new_inputs(cloned_inputs)
#             d[node] = new_node
#             for output, new_output in zip(node.outputs, new_node.outputs):
#                 d[output] = new_output
#             return d[result]

#     for output in o:
#         clone_helper(output)

#     return d


def clone_with_equiv(i, o, d, missing_input_policy = 'fail', orphan_policy = 'copy'):

    def clone_helper(result):
        if result in d:
            return d[result]
        node = result.owner
        if node is None: # result is an input or an orphan not in d
            if isinstance(result, Value):
                if orphan_policy == 'copy':
                    d[result] = copy(result)
                elif orphan_policy == 'keep':
                    d[result] = result
                else:
                    raise ValueError("unknown orphan_policy: '%s'" % orphan_policy)
            else:
                if missing_input_policy == 'fail':
                    raise ValueError("missing input: %s" % result)
                elif missing_input_policy == 'keep':
                    d[result] = result
                else:
                    raise ValueError("unknown missing_input_policy: '%s'" % missing_input_policy)
            return d[result]
        else:
            cloned_inputs = [clone_helper(input) for input in node.inputs]
            if all(input is cloned_input for input, cloned_input in zip(node.inputs, cloned_inputs)):
                new_node = node
            else:
                new_node = node.clone_with_new_inputs(cloned_inputs, strict = False)
#             if any(input != cloned_input for input, cloned_input in zip(node.inputs, cloned_inputs)):
#                 new_node = node.op.make_node(*cloned_inputs)
#             else:
#                 new_node = node.clone_with_new_inputs(cloned_inputs)
            d[node] = new_node
            for output, new_output in zip(node.outputs, new_node.outputs):
                d[output] = new_output
            return d[result]

    for output in o:
        clone_helper(output)

    return [d[input] for input in i], [d[output] for output in o]


def general_toposort(r_out, deps):
    """
    @note: deps(i) should behave like a pure function (no funny business with
    internal state)

    @note: deps(i) can/should be cached by the deps function to be fast
    """
    deps_cache = {}
    def _deps(io):
        if io not in deps_cache:
            d = deps(io)
            if d:
                deps_cache[io] = list(d)
            else:
                deps_cache[io] = d
            return d
        else:
            return deps_cache[io]

    assert isinstance(r_out, (tuple, list, deque))

    reachable, clients = stack_search( deque(r_out), _deps, 'dfs', True)
    sources = deque([r for r in reachable if not deps_cache.get(r, None)])

    rset = set()
    rlist = []
    while sources:
        node = sources.popleft()
        if node not in rset:
            rlist.append(node)
            rset.add(node)
            for client in clients.get(node, []):
                deps_cache[client] = [a for a in deps_cache[client] if a is not node]
                if not deps_cache[client]:
                    sources.append(client)

    if len(rlist) != len(reachable):
        print ''
        print reachable
        print rlist

        raise 'failed to complete topological sort of given nodes'

    return rlist


def io_toposort(i, o, orderings = {}):
    iset = set(i)
    def deps(obj):
        rval = []
        if obj not in iset:
            if isinstance(obj, Result): 
                if obj.owner:
                    rval = [obj.owner]
            if isinstance(obj, Apply):
                rval = list(obj.inputs)
            rval.extend(orderings.get(obj, []))
        else:
            assert not orderings.get(obj, [])
        return rval
    topo = general_toposort(o, deps)
    return [o for o in topo if isinstance(o, Apply)]




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

    i = set(i)

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
            if len(op.outputs) == 1:
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

