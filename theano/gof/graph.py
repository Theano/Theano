"""
Node classes (`Apply`, `Variable`) and expression graph algorithms.

To read about what theano graphs are from a user perspective, have a look at
`graph.html <../doc/graph.html>`__.

"""

__docformat__ = "restructuredtext en"


from copy import copy

import theano
import warnings
from theano.gof import utils
from theano.gof.python25 import any, deque
from theano.misc.ordered_set import OrderedSet

# Lazy imports to avoid circular dependencies.
is_same_graph_with_merge = None
equal_computations = None


class Node(utils.object2):
    """A Node in a theano graph.
    Graphs contain two kinds of Nodes--
    Variable and Apply.
    Edges in the graph are not explicitly represented.
    Instead each Node keeps track of its parents via
    Variable.owner / Apply.inputs and its children
    via Variable.clients / Apply.outputs.
    """

    def get_parents(self):
        """ Return a list of the parents of this node.
        Should return a copy--i.e., modifying the return
        value should not modify the graph structure."""

        raise NotImplementedError()


class Apply(Node):
    """
    An :term:`Apply` instance is a node in an expression graph which represents the application
    of an `Op` to some input `Variable` nodes, producing some output `Variable` nodes.

    This class is typically instantiated by an Op's make_node() function, which is typically
    called by that Op's __call__() function.

    An Apply instance serves as a simple structure with three important attributes:

    - :literal:`inputs` :  a list of `Variable` nodes that represent the arguments of the expression,

    - :literal:`outputs` : a list of `Variable` nodes that represent the variable of the expression, and

    - :literal:`op` : an `Op` instance that determines the nature of the expression being applied.

    The driver `compile.function` uses Apply's inputs attribute together with Variable's owner
    attribute to search the expression graph and determine which inputs are necessary to
    compute the function's outputs.

    A `Linker` uses the Apply instance's `op` field to compute the variables.

    Comparing with the Python language, an `Apply` instance is theano's version of a function
    call (or expression instance) whereas `Op` is theano's version of a function definition.

    """

    def __init__(self, op, inputs, outputs):
        """Initialize attributes

        :Parameters:
         `op` : `Op` instance
           initialize self.op
         `inputs` : list of Variable instances
           initialize self.inputs
         `outputs` : list of Variable instances
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

        if not isinstance(inputs, (list, tuple)):
            raise TypeError("The inputs of an Apply must be a list or tuple")

        if not isinstance(outputs, (list, tuple)):
            raise TypeError("The output of an Apply must be a list or tuple")

        ## filter inputs to make sure each element is a Variable
        for input in inputs:
            if isinstance(input, Variable):
                self.inputs.append(input)
            else:
                raise TypeError("The 'inputs' argument to Apply must contain Variable instances, not %s" % input)
        self.outputs = []
        ## filter outputs to make sure each element is a Variable
        for i, output in enumerate(outputs):
            if isinstance(output, Variable):
                if output.owner is None:
                    output.owner = self
                    output.index = i
                elif output.owner is not self or output.index != i:
                    raise ValueError("All output variables passed to Apply must belong to it.")
                self.outputs.append(output)
            else:
                raise TypeError("The 'outputs' argument to Apply must contain Variable instances with no owner, not %s" % output)

    def default_output(self):
        """Returns the default output for this node.

        :rtype:
            Variable instance

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

    def env_getter(self):
        warnings.warn("Apply.env is deprecated, it has been renamed 'fgraph'",
                stacklevel=2)
        return self.fgraph

    def env_setter(self, value):
        warnings.warn("Apply.env is deprecated, it has been renamed 'fgraph'",
                stacklevel=2)
        self.fgraph = value

    def env_deleter(self):
        warnings.warn("Apply.env is deprecated, it has been renamed 'fgraph'",
                stacklevel=2)
        del self.fgraph

    env = property(env_getter, env_setter, env_deleter)

    out = property(default_output,
                   doc="alias for self.default_output()")
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

    def clone_with_new_inputs(self, inputs, strict=True):
        """Duplicate this Apply instance in a new graph.

        :param inputs: list of Variable instances to use as inputs.

        :type strict: Bool

        :param strict:
            If True, the type fields of all the inputs must be equal
            to the current ones (or compatible, for instance Tensor /
            CudaNdarray of the same dtype and broadcastable patterns,
            in which case they will be converted into current Type), and
            returned outputs are guaranteed to have the same types as
            self.outputs.  If False, then there's no guarantee that the
            clone's outputs will have the same types as self.outputs,
            and cloning may not even be possible (it depends on the Op).

        :returns: an Apply instance with the same op but different outputs.

        """
        assert isinstance(inputs, (list, tuple))
        remake_node = False
        new_inputs = inputs[:]
        for i, (curr, new) in enumerate(zip(self.inputs, new_inputs)):
            if not curr.type == new.type:
                if strict:
                    # If compatible, casts new into curr.type
                    new_inputs[i] = curr.type.filter_variable(new)
                else:
                    remake_node = True
        if remake_node:
            new_node = self.op.make_node(*new_inputs)
            new_node.tag = copy(self.tag).__update__(new_node.tag)
        else:
            new_node = self.clone()
            new_node.inputs = new_inputs
        return new_node

    def get_parents(self):
        return list(self.inputs)

    #convenience properties
    nin = property(lambda self: len(self.inputs), doc='same as len(self.inputs)')
    """property: Number of inputs"""

    nout = property(lambda self: len(self.outputs), doc='same as len(self.outputs)')
    """property: Number of outputs"""


class Variable(Node):
    """
    A :term:`Variable` is a node in an expression graph that represents a variable.

    The inputs and outputs of every `Apply` (theano.gof.Apply) are `Variable` instances.
    The input and output arguments to create a `function` are also `Variable` instances.
    A `Variable` is like a strongly-typed variable in some other languages; each `Variable` contains a
    reference to a `Type` instance that defines the kind of value the `Variable` can take in a
    computation.

    A `Variable` is a container for four important attributes:

    - :literal:`type` a `Type` instance defining the kind of value this `Variable` can have,

    - :literal:`owner` either None (for graph roots) or the `Apply` instance of which `self` is an output,

    - :literal:`index` the integer such that :literal:`owner.outputs[index] is this_variable` (ignored if `owner` is None)

    - :literal:`name` a string to use in pretty-printing and debugging.

    There are a few kinds of Variables to be aware of: A Variable which is the output of a symbolic
    computation has a reference to the Apply instance to which it belongs (property: owner) and
    the position of itself in the owner's output list (property: index).

    - `Variable` (this base type) is typically the output of a symbolic computation,

    - `Constant` (a subclass) which adds a default and un-replaceable :literal:`value`, and
      requires that owner is None

    - `TensorVariable` subclass of Variable that represents a numpy.ndarray object

    - `SharedTensorVariable` Shared version of TensorVariable

    - `SparseVariable` subclass of Variable that represents a scipy.sparse.{csc,csr}_matrix object

    - `CudaNdarrayVariable` subclass of Variable that represents our object on the GPU that is a subset of numpy.ndarray

    - `RandomVariable`

    A Variable which is the output of a symbolic computation will have an owner
    not equal to None.

    Using the Variables' owner field and the Apply nodes' inputs fields, one can navigate a graph
    from an output all the way to the inputs. The opposite direction is not possible until an
    FunctionGraph has annotated the Variables with the clients field, ie, before the compilation process
    has begun a Variable does not know which Apply nodes take it as input.

    **Code Example**

    .. code-block:: python

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

    The python variables :literal:`a,b,c` all refer to instances of type `Variable`.
    The `Variable` refered to by `a` is also an instance of `Constant`.

    `compile.function` uses each `Apply` instance's `inputs` attribute
    together with each Variable's `owner` field to determine which inputs are necessary to compute the function's outputs.

    """
    #__slots__ = ['type', 'owner', 'index', 'name']
    def __init__(self, type, owner=None, index=None, name=None):
        """Initialize type, owner, index, name.

        :type type: a Type instance
        :param type:
            the type governs the kind of data that can be associated with this variable

        :type owner: None or Apply instance
        :param owner: the Apply instance which computes the value for this variable

        :type index: None or int
        :param index: the position of this Variable in owner.outputs

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
        if name is not None and not isinstance(name, basestring):
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
        """Return a new Variable like self.

        :rtype: Variable instance
        :return: a new Variable instance (or subclass instance) with no owner or index.

        :note: tags are copied to the returned instance.
        :note: name is copied to the returned instance.
        """
        #return copy(self)
        cp = self.__class__(self.type, None, None, self.name)
        cp.tag = copy(self.tag)
        return cp

    def __lt__(self, other):
        raise NotImplementedError('Subclasses of Variable must provide __lt__',
                                  self.__class__.__name__)

    def __le__(self, other):
        raise NotImplementedError('Subclasses of Variable must provide __le__',
                                  self.__class__.__name__)

    def __gt__(self, other):
        raise NotImplementedError('Subclasses of Variable must provide __gt__',
                                  self.__class__.__name__)

    def __ge__(self, other):
        raise NotImplementedError('Subclasses of Variable must provide __ge__',
                                  self.__class__.__name__)

    def get_parents(self):
        if self.owner is not None:
            return [self.owner]
        return []

    def env_getter(self):
        warnings.warn("Variable.env is deprecated, it has been renamed 'fgraph'",
                stacklevel=2)
        return self.fgraph

    def env_setter(self, value):
        warnings.warn("Variable.env is deprecated, it has been renamed 'fgraph'",
                stacklevel=2)
        self.fgraph = value

    def env_deleter(self):
        warnings.warn("Variable.env is deprecated, it has been renamed 'fgraph'",
                stacklevel=2)
        del self.fgraph

    def eval(self, inputs_to_values=None):
        """ Evaluates this variable.

        inputs_to_values: a dictionary mapping theano Variables to values.
        """

        if inputs_to_values is None:
            inputs_to_values = {}

        if not hasattr(self, '_fn'):
            self._fn_inputs = inputs_to_values.keys()
            self._fn = theano.function(self._fn_inputs, self)
        args = [inputs_to_values[param] for param in self._fn_inputs]

        rval = self._fn(*args)

        return rval

    env = property(env_getter, env_setter, env_deleter)


class Constant(Variable):
    """
    A :term:`Constant` is a `Variable` with a `value` field that cannot be changed at runtime.

    Constant nodes make eligible numerous optimizations: constant inlining in C code, constant folding, etc.
    """
    #__slots__ = ['data']
    def __init__(self, type, data, name=None):
        """Initialize self.

        :note:
            The data field is filtered by what is provided in the constructor for the Constant's
            type field.

        WRITEME

        """
        Variable.__init__(self, type, None, None, name)
        self.data = type.filter(data)

    def equals(self, other):
        # this does what __eq__ should do, but Variable and Apply should always be hashable by id
        return isinstance(other, Constant) and self.signature() == other.signature()

    def signature(self):
        return (self.type, self.data)

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            name = str(self.data)
            if len(name) > 20:
                name = name[:10] + '...' + name[-10]
            return 'Constant{%s}' % name

    def clone(self):
        """
        We clone this object, but we don't clone the data to lower memory requirement
        We suppose that the data will never change.
        """
        cp = self.__class__(self.type, self.data, self.name)
        cp.tag = copy(self.tag)
        return cp

    def __set_owner(self, value):
        """WRITEME

        :Exceptions:
         - `ValueError`: if `value` is not `None`
        """
        if value is not None:
            raise ValueError("Constant instances cannot have an owner.")

    owner = property(lambda self: None, __set_owner)
    value = property(lambda self: self.data,
            doc='read-only data access method')

    # index is not defined, because the `owner` attribute must necessarily be None


def stack_search(start, expand, mode='bfs', build_inv=False):
    """Search through a graph, either breadth- or depth-first

    :type start: deque
    :param start: search from these nodes
    :type expand: callable
    :param expand:
        when we get to a node, add expand(node) to the list of nodes to visit.
        This function should return a list, or None
    :rtype: list of `Variable` or `Apply` instances (depends on `expend`)
    :return: the list of nodes in order of traversal.

    :note:
        a node will appear at most once in the return value, even if it
        appears multiple times in the start parameter.

    :postcondition: every element of start is transferred to the returned list.
    :postcondition: start is empty.

    """

    if mode not in ('bfs', 'dfs'):
        raise ValueError('mode should be bfs or dfs', mode)
    rval_set = set()
    rval_list = list()
    if mode == 'bfs':
        start_pop = start.popleft
    else:
        start_pop = start.pop
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


def ancestors(variable_list, blockers=None):
    """Return the variables that contribute to those in variable_list (inclusive).

    :type variable_list: list of `Variable` instances
    :param variable_list:
        output `Variable` instances from which to search backward through owners
    :rtype: list of `Variable` instances
    :returns:
        all input nodes, in the order found by a left-recursive depth-first search
        started at the nodes in `variable_list`.

    """
    def expand(r):
        if r.owner and (not blockers or r not in blockers):
            return reversed(r.owner.inputs)
    dfs_variables = stack_search(deque(variable_list), expand, 'dfs')
    return dfs_variables


def inputs(variable_list, blockers=None):
    """Return the inputs required to compute the given Variables.

    :type variable_list: list of `Variable` instances
    :param variable_list:
        output `Variable` instances from which to search backward through owners
    :rtype: list of `Variable` instances
    :returns:
        input nodes with no owner, in the order found by a left-recursive depth-first search
        started at the nodes in `variable_list`.

    """
    vlist = ancestors(variable_list, blockers)
    rval = [r for r in vlist if r.owner is None]
    return rval


def variables_and_orphans(i, o):
    """WRITEME
    """
    def expand(r):
        if r.owner and r not in i:
            l = list(r.owner.inputs) + list(r.owner.outputs)
            l.reverse()
            return l
    variables = stack_search(deque(o), expand, 'dfs')
    orphans = [r for r in variables if r.owner is None and r not in i]
    return variables, orphans


def ops(i, o):
    """ WRITEME

    :type i: list
    :param i: input L{Variable}s
    :type o: list
    :param o: output L{Variable}s

    :returns:
        the set of ops that are contained within the subgraph that lies between i and o,
        including the owners of the L{Variable}s in o and intermediary ops between i and o, but
        not the owners of the L{Variable}s in i.
    """
    ops = set()
    variables, orphans = variables_and_orphans(i, o)
    for r in variables:
        if r not in i and r not in orphans:
            if r.owner is not None:
                ops.add(r.owner)
    return ops


def variables(i, o):
    """ WRITEME

    :type i: list
    :param i: input L{Variable}s
    :type o: list
    :param o: output L{Variable}s

    :returns:
        the set of Variables that are involved in the subgraph that lies between i and o. This
        includes i, o, orphans(i, o) and all values of all intermediary steps from i to o.
    """
    return variables_and_orphans(i, o)[0]


def orphans(i, o):
    """ WRITEME

    :type i: list
    :param i: input L{Variable}s
    :type o: list
    :param o: output L{Variable}s

    :returns:
        the set of Variables which one or more Variables in o depend on but are neither in i nor in
        the subgraph that lies between i and o.

    e.g. orphans([x], [(x+y).out]) => [y]
    """
    return variables_and_orphans(i, o)[1]


def clone(i, o, copy_inputs=True):
    """
    Copies the subgraph contained between i and o.

    :type i: list
    :param i: input L{Variable}s
    :type o: list
    :param o: output L{Variable}s
    :type copy_inputs: bool
    :param copy_inputs: if True, the inputs will be copied (defaults to True)

    Returns the inputs and outputs of that copy.
    """
    equiv = clone_get_equiv(i, o, copy_inputs)
    return [equiv[input] for input in i], [equiv[output] for output in o]


def clone_get_equiv(inputs, outputs,
        copy_inputs_and_orphans=True,
        memo=None):
    """
    Return a dictionary that maps from Variable and Apply nodes in the
    original graph to a new node (a clone) in a new graph.

    This function works by recursively cloning inputs... rebuilding a directed
    graph from the bottom (inputs) up to eventually building new outputs.

    Parameters
    ----------
    inputs: a list of Variables
    outputs: a list of Variables
    copy_inputs_and_orphans: bool
        True means to create the cloned graph from new input and constant
            nodes (the bottom of a feed-upward graph),
        False means to clone a graph that is rooted at the original input
            nodes.
    memo: None or dict
        Optionally start with a partly-filled dictionary for the return value.
        If a dictionary is passed, this function will work in-place on that
        dictionary and return it.

    """

    if memo is None:
        memo = {}

    # clone the inputs if necessary
    for input in inputs:
        if copy_inputs_and_orphans:
            cpy = input.clone()
            cpy.owner = None
            cpy.index = None
            memo.setdefault(input, cpy)
        else:
            memo.setdefault(input, input)

    # go through the inputs -> outputs graph cloning as we go
    for apply in io_toposort(inputs, outputs):
        for input in apply.inputs:
            if input not in memo:
                if copy_inputs_and_orphans:
                    cpy = input.clone()
                    memo[input] = cpy
                else:
                    memo[input] = input

        new_apply = apply.clone_with_new_inputs([memo[i] for i in apply.inputs])
        memo.setdefault(apply, new_apply)
        for output, new_output in zip(apply.outputs, new_apply.outputs):
            memo.setdefault(output, new_output)

    # finish up by cloning any remaining outputs (it can happen)
    for output in outputs:
        if output not in memo:
            memo[output] = output.clone()

    return memo


def general_toposort(r_out, deps, debug_print=False):
    """WRITEME

    :note:
        deps(i) should behave like a pure function (no funny business with internal state)

    :note:
        deps(i) will be cached by this function (to be fast)

    :note:
        The order of the return value list is determined by the order of nodes returned by the deps() function.
    """
    deps_cache = {}

    def _deps(io):
        if io not in deps_cache:
            d = deps(io)
            if d:
                if not isinstance(d, (list, OrderedSet)):
                    raise TypeError("Non-deterministic collections here make"
                            " toposort non-deterministic.")
                deps_cache[io] = list(d)
            else:
                deps_cache[io] = d
            return d
        else:
            return deps_cache[io]

    assert isinstance(r_out, (tuple, list, deque))

    reachable, clients = stack_search(deque(r_out), _deps, 'dfs', True)
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


def io_toposort(inputs, outputs, orderings=None):
    """WRITEME

    inputs: a list or tuple of Variable instances
    outputs: a list or tuple of Variable instances

    orderings: a dictionary
                key: Apply instance
                value: list of Apply instance

                it is important that the value be
                a container with a deterministic iteration
                order. no sets allowed!

    """
    if orderings is None:
        orderings = {}

    #the inputs are used only here in the function that decides what 'predecessors' to explore
    iset = set(inputs)

    def deps(obj):
        rval = []
        if obj not in iset:
            if isinstance(obj, Variable):
                if obj.owner:
                    rval = [obj.owner]
            elif isinstance(obj, Apply):
                rval = list(obj.inputs)
            rval.extend(orderings.get(obj, []))
        else:
            assert not orderings.get(obj, [])
        return rval

    topo = general_toposort(outputs, deps)
    return [o for o in topo if isinstance(o, Apply)]


default_leaf_formatter = str
default_node_formatter = lambda op, argstrings: "%s(%s)" % (op.op,
                                                            ", ".join(argstrings))


def is_same_graph(var1, var2, givens=None, debug=False):
    """
    Return True iff Variables `var1` and `var2` perform the same computation.

    By 'performing the same computation', we mean that they must share the same
    graph, so that for instance this function will return False when comparing
    (x * (y * z)) with ((x * y) * z).

    The current implementation is not efficient since, when possible, it
    verifies equality by calling two different functions that are expected to
    return the same output. The goal is to verify this assumption, to
    eventually get rid of one of them in the future.

    :param var1: The first Variable to compare.

    :param var2: The second Variable to compare.

    :param givens: Similar to the `givens` argument of `theano.function`, it
    can be used to perform substitutions in the computational graph of `var1`
    and `var2`. This argument is associated to neither `var1` nor `var2`:
    substitutions may affect both graphs if the substituted variable is present
    in both.

    :param debug: If True, then an exception is raised when we are in a
    situation where the `equal_computations` implementation cannot be called.
    This parameter is intended to be used in tests only, to make sure we
    properly test both implementations.

    Examples:

        ======  ======  ======  ======
        var1    var2    givens  output
        ======  ======  ======  ======
        x + 1   x + 1   {}      True
        x + 1   y + 1   {}      False
        x + 1   y + 1   {x: y}  True
        ======  ======  ======  ======
    """
    # Lazy import.
    if givens is None:
        givens = {}
    global equal_computations, is_same_graph_with_merge
    if equal_computations is None:
        from theano.gof.opt import is_same_graph_with_merge
        from theano.scan_module.scan_utils import equal_computations
    # Convert `givens` to dictionary.
    if not isinstance(givens, dict):
        givens = dict(givens)
    # Get result from the merge-based function.
    rval1 = is_same_graph_with_merge(var1=var1, var2=var2, givens=givens)
    # Get result from the function `equal_computations` from scan_utils.
    use_equal_computations = True
    if givens:
        # We need to build the `in_xs` and `in_ys` lists. To do this, we need
        # to be able to tell whether a variable belongs to the computational
        # graph of `var1` or `var2`.
        # The typical case we want to handle is when `to_replace` belongs to
        # one of these graphs, and `replace_by` belongs to the other one. In
        # other situations, the current implementation of `equal_computations`
        # is probably not appropriate, so we do not call it.
        ok = True
        in_xs = []
        in_ys = []
        # Compute the sets of all variables found in each computational graph.
        inputs_var = map(inputs, ([var1], [var2]))
        all_vars = [set(variables(v_i, v_o))
                    for v_i, v_o in ((inputs_var[0], [var1]),
                                     (inputs_var[1], [var2]))]

        def in_var(x, k):
            # Return True iff `x` is in computation graph of variable `vark`.
            return x in all_vars[k - 1]

        for to_replace, replace_by in givens.iteritems():
            # Map a substitution variable to the computational graphs it
            # belongs to.
            inside = dict((v, [in_var(v, k) for k in (1, 2)])
                          for v in (to_replace, replace_by))
            if (inside[to_replace][0] and not inside[to_replace][1] and
                inside[replace_by][1] and not inside[replace_by][0]):
                # Substitute variable in `var1` by one from `var2`.
                in_xs.append(to_replace)
                in_ys.append(replace_by)
            elif (inside[to_replace][1] and not inside[to_replace][0] and
                  inside[replace_by][0] and not inside[replace_by][1]):
                # Substitute variable in `var2` by one from `var1`.
                in_xs.append(replace_by)
                in_ys.append(to_replace)
            else:
                ok = False
                break
        if not ok:
            # We cannot directly use `equal_computations`.
            if debug:
                raise AssertionError(
                    'When `debug` is True we want to make sure we are also '
                    'using the `equal_computations` implementation')
            use_equal_computations = False
    else:
        in_xs = None
        in_ys = None
    if use_equal_computations:
        rval2 = equal_computations(xs=[var1], ys=[var2],
                                   in_xs=in_xs, in_ys=in_ys)
        assert rval2 == rval1
    return rval1


def op_as_string(i, op,
                 leaf_formatter=default_leaf_formatter,
                 node_formatter=default_node_formatter):
    """WRITEME"""
    strs = as_string(i, op.inputs, leaf_formatter, node_formatter)
    return node_formatter(op, strs)


def as_string(i, o,
              leaf_formatter=default_leaf_formatter,
              node_formatter=default_node_formatter):
    """WRITEME

    :type i: list
    :param i: input `Variable` s
    :type o: list
    :param o: output `Variable` s
    :type leaf_formatter: function
    :param leaf_formatter: takes a `Variable`  and returns a string to describe it
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


def list_of_nodes(inputs, outputs):
    """ Return the apply nodes of the graph between inputs and outputs """
    return stack_search(
            deque([o.owner for o in outputs]),
            lambda o: [inp.owner for inp in o.inputs
                           if inp.owner
                           and not any(i in inp.owner.outputs for i in inputs)])
