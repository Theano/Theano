"""
Node classes (`Apply`, `Result`) and expression graph algorithms.

To read about what theano graphs are from a user perspective, have a look at
`graph.html <../doc/graph.html>`__.

"""

__docformat__ = "restructuredtext en"

from copy import copy
from collections import deque

import utils


class Apply(utils.object2):
    """
    An :term:`Apply` instance is a node in an expression graph which represents the application
    of an `Op` to some input `Result` nodes, producing some output `Result` nodes.
    
    This class is typically instantiated by an Op's make_node() function, which is typically
    called by that Op's __call__() function.

    An Apply instance serves as a simple structure with three important attributes:

    - :literal:`inputs` :  a list of `Result` nodes that represent the arguments of the expression,

    - :literal:`outputs` : a list of `Result` nodes that represent the result of the expression, and

    - :literal:`op` : an `Op` instance that determines the nature of the expression being applied.

    The driver `compile.function` uses Apply's inputs attribute together with Result's owner
    attribute to search the expression graph and determine which inputs are necessary to
    compute the function's outputs.

    A `Linker` uses the Apply instance's `op` field to compute the results.

    Comparing with the Python language, an `Apply` instance is theano's version of a function
    call (or expression instance) whereas `Op` is theano's version of a function definition.

    """

    def __init__(self, op, inputs, outputs):
        """Initialize attributes

        :Parameters:
         `op` : `Op` instance
           initialize self.op
         `inputs` : list of Result instances
           initialize self.inputs
         `outputs` : list of Result instances
           initialize self.outputs

        :note:
            The owner field of each output in the outputs list will be set to self.

        :note:
            If an output element has an owner that is neither None nor self, then a ValueError
            exception will be raised.

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
        """Returns the default output for this node.
        
        :rtype:
            Result instance 

        :return:
            an element of self.outputs, typically self.outputs[0].

        :note:
            may raise AttributeError self.op.default_output is out of range, or if there are
            multiple outputs and self.op.default_output does not exist.

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
                   doc = "alias for self.default_output()")
    """Alias for self.default_output()"""

    def __str__(self):
        return op_as_string(self.inputs, self)

    def __repr__(self):
        return str(self)

    def __asapply__(self):
        return self

    def clone(self):
        """Duplicate this Apply instance with inputs = self.inputs.

        :return:
            a new Apply instance (or subclass instance) with new outputs.

        :note:
            tags are copied from self to the returned instance.
        """
        cp = self.__class__(self.op, self.inputs, [output.clone() for output in self.outputs])
        cp.tag = copy(self.tag)
        return cp

    def clone_with_new_inputs(self, inputs, strict = True):
        """Duplicate this Apply instance in a new graph.

        :param inputs: list of Result instances to use as inputs.

        :type strict: Bool

        :param strict: 
            If True, the type fields of all the inputs must be equal to the current ones, and
            returned outputs are guaranteed to have the same types as self.outputs.  If False,
            then there's no guarantee that the clone's outputs will have the same types as
            self.outputs, and cloning may not even be possible (it depends on the Op).

        :returns: an Apply instance with the same op but different outputs.

        """
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

    #convenience properties
    nin = property(lambda self: len(self.inputs), doc = 'same as len(self.inputs)')
    """property: Number of inputs"""

    nout = property(lambda self: len(self.outputs), doc = 'same as len(self.outputs)')
    """property: Number of outputs"""


class Result(utils.object2):
    """
    A :term:`Result` is a node in an expression graph that represents a variable.

    The inputs and outputs of every `Apply` are `Result` instances.
    The input and output arguments to create a `function` are also `Result` instances.
    A `Result` is like a strongly-typed variable in some other languages; each `Result` contains a
    reference to a `Type` instance that defines the kind of value the `Result` can take in a
    computation.

    A `Result` is a container for four important attributes:

    - :literal:`type` a `Type` instance defining the kind of value this `Result` can have,

    - :literal:`owner` either None (for graph roots) or the `Apply` instance of which `self` is an output,

    - :literal:`index` the integer such that :literal:`owner.outputs[index] is this_result` (ignored if `owner` is None)

    - :literal:`name` a string to use in pretty-printing and debugging.

    There are a few kinds of Results to be aware of: A Result which is the output of a symbolic
    computation has a reference to the Apply instance to which it belongs (property: owner) and
    the position of itself in the owner's output list (property: index).

    - `Result` (this base type) is typically the output of a symbolic computation,
    
    - `Value` (a subclass) adds a default :literal:`value`, and requires that owner == None

    - `Constant` (a subclass) which adds a default and un-replacable :literal:`value`, and
      requires that owner == None

    A Result which is the output of a symbolic computation will have an owner != None.

    Code Example
    ============

    .. python::

        import theano
        from theano import tensor

        a = tensor.constant(1.5)        # declare a symbolic constant
        b = tensor.fscalar()            # declare a symbolic floating-point scalar

        c = a + b                       # create a simple expression

        f = theano.function([b], [c])   # this works because a has a value associated with it already

        assert 4.0 == f(2.5)            # bind 2.5 to an internal copy of b and evaluate an internal c

        theano.function([a], [c])       # compilation error because b (required by c) is undefined

        theano.function([a,b], [c])     # compilation error because a is constant, it can't be an input

        d = tensor.value(1.5)           # create a value similar to the constant 'a'
        e = d + b
        theano.function([d,b], [e])     # this works.  d's default value of 1.5 is ignored.

    The python variables :literal:`a,b,c` all refer to instances of type `Result`.
    The `Result` refered to by `a` is also an instance of `Constant`.

    `compile.function` uses each `Apply` instance's `inputs` attribute
    together with each Result's `owner` field to determine which inputs are necessary to compute the function's outputs.

    """
    #__slots__ = ['type', 'owner', 'index', 'name']
    def __init__(self, type, owner = None, index = None, name = None):
        """Initialize type, owner, index, name.

        :type type: a Type instance
        :param type: 
            the type governs the kind of data that can be associated with this variable

        :type owner: None or Apply instance
        :param owner: the Apply instance which computes the value for this variable

        :type index: None or int
        :param index: the position of this Result in owner.outputs

        :type name: None or str
        :param name: a string for pretty-printing and debugging

        """
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
        """WRITEME"""
        if self.name is not None:
            return self.name
        if self.owner is not None:
            op = self.owner.op
            if self.index == op.default_output:
                return str(self.owner.op) + ".out"
            else:
                return str(self.owner.op) + "." + str(self.index)
        else:
            return "<%s>" % str(self.type)
    def __repr__(self):
        return str(self)
    def clone(self):
        """Return a new Result like self.

        :rtype: Result instance
        :return: a new Result instance (or subclass instance) with no owner or index.

        :note: tags are copied to the returned instance.
        :note: name is copied to the returned instance.
        """
        #return copy(self)
        cp = self.__class__(self.type, None, None, self.name)
        cp.tag = copy(self.tag)
        return cp

class Value(Result):
    """
    A :term:`Value` is a `Result` with a default value.

    Its owner field is always None. And since it has a default value, a `Value` instance need
    not be named as an input to `compile.function`.

    This kind of node is useful because when a value is known at compile time, more
    optimizations are possible.

    """
    #__slots__ = ['data']
    def __init__(self, type, data, name = None):
        """Initialize self.

        :note: 
            The data field is filtered by what is provided in the constructor for the Value's
            type field.

        WRITEME

        """
        Result.__init__(self, type, None, None, name)
        self.data = type.filter(data)
    def __str__(self):
        """WRITEME"""
        if self.name is not None:
            return self.name
        return "<" + str(self.data) + ">" #+ "::" + str(self.type)
    def clone(self):
        """WRITEME"""
        return self.__class__(self.type, copy(self.data), self.name)
    def __set_owner(self, value):
        """WRITEME

        :Exceptions:
         - `ValueError`: if `value` is not `None`
        """
        if value is not None:
            raise ValueError("Value instances cannot have an owner.")
    owner = property(lambda self: None, __set_owner)

class Constant(Value):
    """
    A :term:`Constant` is a `Value` that cannot be changed at runtime.

    Constant nodes make eligible numerous optimizations: constant inlining in C code, constant folding, etc.
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
    """Search through a graph, either breadth- or depth-first

    :type start: deque
    :param start: search from these nodes
    :type expand: callable
    :param expand: 
        when we get to a node, add expand(node) to the list of nodes to visit.  This function
        should return a list, or None
    :rtype: list of `Result` or `Apply` instances (depends on `expend`)
    :return: the list of nodes in order of traversal.
    
    :note:
        a node will appear at most once in the return value, even if it appears multiple times
        in the start parameter.  

    :postcondition: every element of start is transferred to the returned list.
    :postcondition: start is empty.

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


def inputs(result_list, blockers = None):
    """Return the inputs required to compute the given Results.

    :type result_list: list of `Result` instances
    :param result_list:
        output `Result` instances from which to search backward through owners
    :rtype: list of `Result` instances
    :returns: 
        input nodes with no owner, in the order found by a left-recursive depth-first search
        started at the nodes in `result_list`.

    """
    def expand(r):
        if r.owner and (not blockers or r not in blockers):
            l = list(r.owner.inputs)
            l.reverse()
            return l
    dfs_results = stack_search(deque(result_list), expand, 'dfs')
    rval = [r for r in dfs_results if r.owner is None]
    #print rval, _orig_inputs(o)
    return rval


def results_and_orphans(i, o):
    """WRITEME
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
    """ WRITEME

    :type i: list
    :param i: input L{Result}s
    :type o: list
    :param o: output L{Result}s

    :returns:
        the set of ops that are contained within the subgraph that lies between i and o,
        including the owners of the L{Result}s in o and intermediary ops between i and o, but
        not the owners of the L{Result}s in i.
    """
    ops = set()
    results, orphans = results_and_orphans(i, o)
    for r in results:
        if r not in i and r not in orphans:
            if r.owner is not None:
                ops.add(r.owner)
    return ops


def results(i, o):
    """ WRITEME

    :type i: list
    :param i: input L{Result}s
    :type o: list
    :param o: output L{Result}s

    :returns:
        the set of Results that are involved in the subgraph that lies between i and o. This
        includes i, o, orphans(i, o) and all values of all intermediary steps from i to o.
    """
    return results_and_orphans(i, o)[0]


def orphans(i, o):
    """ WRITEME

    :type i: list
    :param i: input L{Result}s
    :type o: list
    :param o: output L{Result}s

    :returns:
        the set of Results which one or more Results in o depend on but are neither in i nor in
        the subgraph that lies between i and o.

    e.g. orphans([x], [(x+y).out]) => [y]
    """
    return results_and_orphans(i, o)[1]


def clone(i, o, copy_inputs = True):
    """ WRITEME

    :type i: list
    :param i: input L{Result}s
    :type o: list
    :param o: output L{Result}s
    :type copy_inputs: bool
    :param copy_inputs: if True, the inputs will be copied (defaults to False)

    Copies the subgraph contained between i and o and returns the
    outputs of that copy (corresponding to o).
    """
    equiv = clone_get_equiv(i, o, copy_inputs)
    return [equiv[input] for input in i], [equiv[output] for output in o]


def clone_get_equiv(i, o, copy_inputs_and_orphans = True):
    """ WRITEME

    :type i: list
    :param i: input L{Result}s
    :type o: list
    :param o: output L{Result}s
    :type copy_inputs_and_orphans: bool
    :param copy_inputs_and_orphans: 
        if True, the inputs and the orphans will be replaced in the cloned graph by copies
        available in the equiv dictionary returned by the function (copy_inputs_and_orphans
        defaults to True)

    :rtype: a dictionary
    :return:
        equiv mapping each L{Result} and L{Op} in the graph delimited by i and o to a copy
        (akin to deepcopy's memo).
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

    for apply in io_toposort(i, o):
        for input in apply.inputs:
            if input not in d:
                if copy_inputs_and_orphans:
                    cpy = input.clone()
                    d[input] = cpy
                else:
                    d[input] = input

        new_apply = apply.clone_with_new_inputs([d[i] for i in apply.inputs])
        d[apply] = new_apply
        for output, new_output in zip(apply.outputs, new_apply.outputs):
            d[output] = new_output

    for output in o:
        if output not in d:
            d[output] = output.clone()

    return d

def general_toposort(r_out, deps, debug_print = False):
    """WRITEME

    :note: 
        deps(i) should behave like a pure function (no funny business with internal state)

    :note: 
        deps(i) can/should be cached by the deps function to be fast
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
        if debug_print:
            print ''
            print reachable
            print rlist
        raise ValueError('graph contains cycles')

    return rlist


def io_toposort(i, o, orderings = {}):
    """WRITEME
    """
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
    """WRITEME"""
    strs = as_string(i, op.inputs, leaf_formatter, node_formatter)
    return node_formatter(op, strs)


def as_string(i, o,
              leaf_formatter = default_leaf_formatter,
              node_formatter = default_node_formatter):
    """WRITEME

    :type i: list
    :param i: input `Result` s
    :type o: list
    :param o: output `Result` s
    :type leaf_formatter: function
    :param leaf_formatter: takes a `Result`  and returns a string to describe it
    :type node_formatter: function
    :param node_formatter: 
        takes an `Op`  and the list of strings corresponding to its arguments and returns a
        string to describe it

    :rtype: str
    :returns:
        Returns a string representation of the subgraph between i and o. If the same op is used
        by several other ops, the first occurrence will be marked as :literal:`*n ->
        description` and all subsequent occurrences will be marked as :literal:`*n`, where n is
        an id number (ids are attributed in an unspecified order and only exist for viewing
        convenience).
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


def view_roots(r):
    """
    Utility function that returns the leaves of a search through
    consecutive view_map()s.

    WRITEME
    """
    owner = r.owner
    if owner is not None:
        try:
            view_map = owner.op.view_map
            view_map = dict([(owner.outputs[o], i) for o, i in view_map.items()])
        except AttributeError:
            return [r]
        if r in view_map:
            answer = []
            for i in view_map[r]:
                answer += view_roots(owner.inputs[i])
            return answer
        else:
            return [r]
    else:
        return [r]


