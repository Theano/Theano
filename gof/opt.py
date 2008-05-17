"""
Defines the base class for optimizations as well as a certain
amount of useful generic optimization tools.
"""


import graph
from env import InconsistencyError
import utils
import unify
import toolbox
import op


class Optimizer:
    """
    An L{Optimizer} can be applied to an L{Env} to transform it.
    It can represent an optimization or in general any kind
    of transformation you could apply to an L{Env}.
    """

    def apply(self, env):
        """
        Applies the optimization to the provided L{Env}. It may use all
        the methods defined by the L{Env}. If the L{Optimizer} needs
        to use a certain tool, such as an L{InstanceFinder}, it can do
        so in its L{add_requirements} method.
        """
        pass

    def optimize(self, env):
        """
        This is meant as a shortcut to::
          env.satisfy(opt)
          opt.apply(env)
        """
        self.add_requirements(env)
        self.apply(env)

    def __call__(self, env):
        """
        Same as self.optimize(env)
        """
        return self.optimize(env)

    def add_requirements(self, env):
        """
        Add features to the env that are required to apply the optimization.
        For example:
          env.extend(History())
          env.extend(MyFeature())
          etc.
        """
        pass


DummyOpt = Optimizer()
DummyOpt.__doc__ = "Does nothing."


class SeqOptimizer(Optimizer, list):
    """
    Takes a list of L{Optimizer} instances and applies them
    sequentially.
    """

    def __init__(self, *opts):
        if len(opts) == 1 and isinstance(opts[0], (list, tuple)):
            opts = opts[0]
        list.__init__(opts)

    def apply(self, env):
        """
        Applies each L{Optimizer} in self in turn.
        """
        for optimizer in self:
            optimizer.optimize(env)

    def __str__(self):
        return "SeqOpt(%s)" % list.__str__(self)

    def __repr__(self):
        return list.__repr__(self)



class LocalOptimizer(Optimizer):
    """
    Generic L{Optimizer} class that considers local parts of
    the L{Env}. It must be subclassed and should override the
    following two methods:
     - candidates(env) -> returns a set of ops that can be
       optimized
     - apply_on_node(env, node) -> for each node in candidates,
       this function will be called to perform the actual
       optimization.
    """

    def candidates(self, env):
        """
        Must return a set of nodes that can be optimized.
        """
        raise utils.AbstractFunctionError()

    def apply_on_node(self, env, node):
        """
        For each node in candidates, this function will be called to
        perform the actual optimization.
        """
        raise utils.AbstractFunctionError()

    def apply(self, env):
        """
        Calls self.apply_on_op(env, op) for each op in self.candidates(env).
        """
        for node in self.candidates(env):
            if node in env.nodes:
                self.apply_on_node(env, node)



class OpSpecificOptimizer(LocalOptimizer):
    """
    Generic L{Optimizer} that applies only to ops of a certain
    type. The type in question is accessed through L{self.op}.
    op can also be a class variable of the subclass.
    """

    def add_requirements(self, env):
        try:
            env.extend(toolbox.NodeFinder())
            env.extend(toolbox.ReplaceValidate())
        except: pass

    def candidates(self, env):
        """
        Returns all nodes that have L{self.op} in their op field.
        """
        return env.get_nodes(self.op)




class OpSubOptimizer(Optimizer):
    """
    Replaces all applications of a certain op by the application of
    another op that take the same inputs as what they are replacing.

    e.g. OpSubOptimizer(add, sub) ==> add(div(x, y), add(y, x)) -> sub(div(x, y), sub(y, x))
    
    OpSubOptimizer requires the following features:
      - NodeFinder
      - ReplaceValidate
    """

    def add_requirements(self, env):
        """
        Requires the following features:
          - NodeFinder
          - ReplaceValidate
        """
        try:
            env.extend(toolbox.NodeFinder())
            env.extend(toolbox.ReplaceValidate())
        except: pass

    def __init__(self, op1, op2, failure_callback = None):
        """
        op1.make_node and op2.make_node must take the same number of
        inputs and have the same number of outputs.
        
        If failure_callback is not None, it will be called whenever
        the Optimizer fails to do a replacement in the graph. The
        arguments to the callback are: (node, replacement, exception)
        """
        self.op1 = op1
        self.op2 = op2
        self.failure_callback = failure_callback

    def apply(self, env):
        """
        Replaces all applications of self.op1 by applications of self.op2
        with the same inputs.
        """
        candidates = env.get_nodes(self.op1)

        for node in candidates:
            try:
                repl = self.op2.make_node(*node.inputs)
                assert len(node.outputs) == len(repl.outputs)
                for old, new in zip(node.outputs, repl.outputs):
                    env.replace_validate(old, new)
            except Exception, e:
                if self.failure_callback is not None:
                    self.failure_callback(node, repl, e)

    def str(self):
        return "%s -> %s" % (self.op1, self.op2)



class OpRemover(Optimizer):
    """
    @todo untested
    Removes all applications of an op by transferring each of its
    outputs to the corresponding input.
    """

    def add_requirements(self, env):
        try:
            env.extend(toolbox.NodeFinder())
            env.extend(toolbox.ReplaceValidate())
        except: pass

    def __init__(self, op, failure_callback = None):
        """
        Applications of the op must have as many inputs as outputs.
        
        If failure_callback is not None, it will be called whenever
        the Optimizer fails to remove an operation in the graph. The
        arguments to the callback are: (node, exception)
        """
        self.op = op
        self.failure_callback = failure_callback

    def apply(self, env):
        """
        Removes all applications of self.op.
        """
        candidates = env.get_nodes(self.op)

        for node in candidates:
            try:
                assert len(node.inputs) == len(node.outputs)
                for input, output in zip(node.inputs, node.outputs):
                    env.replace(output, input)
            except Exception, e:
                if self.failure_callback is not None:
                    self.failure_callback(node, e)
                pass

    def str(self):
        return "f(%s(x)) -> f(x)" % self.op



class PatternOptimizer(OpSpecificOptimizer):
    """
    @todo update
    
    Replaces all occurrences of the input pattern by the output pattern:

     input_pattern ::= (op, <sub_pattern1>, <sub_pattern2>, ...)
     input_pattern ::= dict(pattern = <input_pattern>,
                            constraint = <constraint>)
     sub_pattern ::= input_pattern
     sub_pattern ::= string
     sub_pattern ::= a Constant instance
     constraint ::= lambda env, expr: additional matching condition
     
     output_pattern ::= (op, <output_pattern1>, <output_pattern2>, ...)
     output_pattern ::= string

    Each string in the input pattern is a variable that will be set to
    whatever expression is found in its place. If the same string is
    used more than once, the same expression must be found in those
    places. If a string used in the input pattern is used in the
    output pattern, the matching expression will be inserted in its
    place. The input pattern cannot just be a string but the output
    pattern can.

    If you put a constant result in the input pattern, there will be a
    match iff a constant result with the same value and the same type
    is found in its place.

    You can add a constraint to the match by using the dict(...)  form
    described above with a 'constraint' key. The constraint must be a
    function that takes the env and the current Result that we are
    trying to match and returns True or False according to an
    arbitrary criterion.

    Examples:
     PatternOptimizer((add, 'x', 'y'), (add, 'y', 'x'))
     PatternOptimizer((multiply, 'x', 'x'), (square, 'x'))
     PatternOptimizer((subtract, (add, 'x', 'y'), 'y'), 'x')
     PatternOptimizer((power, 'x', Constant(double, 2.0)), (square, 'x'))
     PatternOptimizer((boggle, {'pattern': 'x',
                                'constraint': lambda env, expr: expr.type == scrabble}),
                      (scrabble, 'x'))
    """

    def __init__(self, in_pattern, out_pattern, allow_multiple_clients = False, failure_callback = None):
        """
        Creates a PatternOptimizer that replaces occurrences of
        in_pattern by occurrences of out_pattern.
        
        If failure_callback is not None, if there is a match but a
        replacement fails to occur, the callback will be called with
        arguments (result_to_replace, replacement, exception).

        If allow_multiple_clients is False, he pattern matching will
        fail if one of the subpatterns has more than one client.
        """
        self.in_pattern = in_pattern
        self.out_pattern = out_pattern
        if isinstance(in_pattern, (list, tuple)):
            self.op = self.in_pattern[0]
        elif isinstance(in_pattern, dict):
            self.op = self.in_pattern['pattern'][0]
        else:
            raise TypeError("The pattern to search for must start with a specific Op instance.")
        self.__doc__ = self.__class__.__doc__ + "\n\nThis instance does: " + str(self) + "\n"
        self.failure_callback = failure_callback
        self.allow_multiple_clients = allow_multiple_clients

    def apply_on_node(self, env, node):
        """
        Checks if the graph from node corresponds to in_pattern. If it does,
        constructs out_pattern and performs the replacement.
        """
        def match(pattern, expr, u, first = False):
            if isinstance(pattern, (list, tuple)):
                if expr.owner is None:
                    return False
                if not (expr.owner.op == pattern[0]) or (not self.allow_multiple_clients and not first and env.nclients(expr) > 1):
                    return False
                if len(pattern) - 1 != len(expr.owner.inputs):
                    return False
                for p, v in zip(pattern[1:], expr.owner.inputs):
                    u = match(p, v, u)
                    if not u:
                        return False
            elif isinstance(pattern, dict):
                try:
                    real_pattern = pattern['pattern']
                    constraint = pattern['constraint']
                except KeyError:
                    raise KeyError("Malformed pattern: %s (expected keys pattern and constraint)" % pattern)
                if constraint(env, expr):
                    return match(real_pattern, expr, u, False)
            elif isinstance(pattern, str):
                v = unify.Var(pattern)
                if u[v] is not v and u[v] is not expr:
                    return False
                else:
                    u = u.merge(expr, v)
            elif isinstance(pattern, graph.Constant) and isinstance(expr, graph.Constant) and pattern.equals(expr):
                return u
            else:
                return False
            return u

        def build(pattern, u):
            if isinstance(pattern, (list, tuple)):
                args = [build(p, u) for p in pattern[1:]]
                return pattern[0](*args)
            elif isinstance(pattern, str):
                return u[unify.Var(pattern)]
            else:
                return pattern

        u = match(self.in_pattern, node.out, unify.Unification(), True)
        if u:
            try:
                # note: only replaces the default 'out' port if it exists
                p = self.out_pattern
                new = 'unassigned' # this is for the callback if build fails
                new = build(p, u)
                env.replace(node.out, new)
            except Exception, e:
                if self.failure_callback is not None:
                    self.failure_callback(node.out, new, e)
                pass

    def __str__(self):
        def pattern_to_str(pattern):
            if isinstance(pattern, (list, tuple)):
                return "%s(%s)" % (str(pattern[0]), ", ".join([pattern_to_str(p) for p in pattern[1:]]))
            elif isinstance(pattern, dict):
                return "%s subject to %s" % (pattern_to_str(pattern['pattern']), str(pattern['constraint']))
            else:
                return str(pattern)
        return "%s -> %s" % (pattern_to_str(self.in_pattern), pattern_to_str(self.out_pattern))




class _metadict:
    # dict that accepts unhashable keys
    # uses an associative list
    # for internal use only
    def __init__(self):
        self.d = {}
        self.l = []
    def __getitem__(self, item):
        return self.get(item, None)
    def __setitem__(self, item, value):
        try:
            self.d[item] = value
        except:
            self.l.append((item, value))
    def get(self, item, default):
        try:
            return self.d[item]
        except:
            for item2, value in self.l:
                try:
                    if item == item2:
                        return value
                    if item.equals(item2):
                        return value
                except:
                    if item is item2:
                        return value
            else:
                return default
    def clear(self):
        self.d = {}
        self.l = []
    def __str__(self):
        return "(%s, %s)" % (self.d, self.l)


class MergeOptimizer(Optimizer):
    """
    Merges parts of the graph that are identical, i.e. parts that
    take the same inputs and carry out the asme computations so we
    can avoid doing them more than once. Also merges results that
    are constant.
    """

    def add_requirements(self, env):
        try:
            env.extend(toolbox.ReplaceValidate())
        except: pass

    def apply(self, env):
        cid = _metadict()     #result -> result.desc()  (for constants)
        inv_cid = _metadict() #desc -> result (for constants)
        for i, r in enumerate([r for r in env.results if isinstance(r, graph.Constant)]):
            sig = r.signature()
            other_r = inv_cid.get(sig, None)
            if other_r is not None:
                env.replace(r, other_r)
            else:
                cid[r] = sig
                inv_cid[sig] = r
        # we clear the dicts because the Constants signatures are not necessarily hashable
        # and it's more efficient to give them an integer cid like the other Results
        cid.clear()
        inv_cid.clear()
        for i, r in enumerate(r for r in env.results if r.owner is None):
            cid[r] = i
            inv_cid[i] = r

        for node in graph.io_toposort(env.inputs, env.outputs):
            node_cid = (node.op, tuple([cid[input] for input in node.inputs]))
            dup = inv_cid.get(node_cid, None)
            success = False
            if dup is not None:
                success = True
                try:
                    env.replace_all_validate(zip(node.outputs, dup.outputs))
                except InconsistencyError, e:
                    success = False
            if not success:
                cid[node] = node_cid
                inv_cid[node_cid] = node
                for i, output in enumerate(node.outputs):
                    ref = (i, node_cid)
                    cid[output] = ref
                    inv_cid[ref] = output


def MergeOptMerge(opt):
    """
    Returns an Optimizer that merges the graph then applies the
    optimizer in opt and then merges the graph again in case the
    opt introduced additional similarities.
    """
    merger = MergeOptimizer()
    return SeqOptimizer([merger, opt, merger])





class LocalOptimizer:

    def applies(self, node):
        raise utils.AbstractFunctionError()

    def transform(self, node):
        raise utils.AbstractFunctionError()


class ExpandMacro:

    def applies(self, node):
        return isinstance(node.op, op.Macro)

    def transform(self, node):
        return node.op.expand(node)


from collections import deque

class TopDownOptimizer(Optimizer):

    def __init__(self, local_opt, ignore_newtrees = False):
        self.local_opt = local_opt
        self.ignore_newtrees = ignore_newtrees

    def apply(self, env):
        ignore_newtrees = self.ignore_newtrees
        q = deque()
        class Updater:
            def on_attach(self, env):
                for node in graph.io_toposort(env.inputs, env.outputs):
                    q.appendleft(node)
            if not ignore_newtrees:
                def on_import(self, env, node):
                    q.appendleft(node)
            def on_prune(self, env, node):
                if node is not current_node:
                    q.remove(node)
        u = Updater()
        env.extend(u)
        while q:
            node = q.popleft()
            current_node = node
            if not self.local_opt.applies(node):
                continue
            replacements = self.local_opt.transform(node)
            for output, replacement in zip(node.outputs, replacements):
                env.replace_validate(output, replacement)
        env.remove_feature(u)

    def add_requirements(self, env):
        try:
            env.extend(toolbox.ReplaceValidate())
        except: pass

expand_macros = TopDownOptimizer(ExpandMacro())





