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
from copy import copy
from collections import deque


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

    def optimize(self, env, *args, **kwargs):
        """
        This is meant as a shortcut to::
          env.satisfy(opt)
          opt.apply(env)
        """
        self.add_requirements(env)
        self.apply(env, *args, **kwargs)

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
        env.extend(toolbox.ReplaceValidate())

    def apply(self, env):
        cid = _metadict()     #result -> result.desc()  (for constants)
        inv_cid = _metadict() #desc -> result (for constants)
        for i, r in enumerate([r for r in env.results if isinstance(r, graph.Constant)]):
            sig = r.signature()
            other_r = inv_cid.get(sig, None)
            if other_r is not None:
                env.replace_validate(r, other_r)
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



########################
### Local Optimizers ###
########################

class LocalOptimizer(utils.object2):

    def transform(self, node):
        raise utils.AbstractFunctionError()


class LocalOptGroup(LocalOptimizer):

    def __init__(self, optimizers):
        self.opts = optimizers
        self.reentrant = any(getattr(opt, 'reentrant', True), optimizers)
        self.retains_inputs = all(getattr(opt, 'retains_inputs', False), optimizers)

    def transform(self, node):
        for opt in self.opts:
            repl = opt.transform(node)
            if repl is not False:
                return repl


class LocalOpKeyOptGroup(LocalOptGroup):

    def __init__(self, optimizers):
        if any(not hasattr(opt, 'op_key'), optimizers):
            raise TypeError("All LocalOptimizers passed here must have an op_key method.")
        CompositeLocalOptimizer.__init__(self, optimizers)
    
    def op_key(self):
        return [opt.op_key() for opt in self.opts]


class ExpandMacro(LocalOptimizer):

    def __init__(self, filter = None):
        if filter is None:
            self.filter = lambda node: True
        else:
            self.filter = filter

    def transform(self, node):
        if not isinstance(node.op, op.Macro) or not self.filter(node):
            return False
        return node.op.expand(node)


class OpSub(LocalOptimizer):
    """
    Replaces the application of a certain op by the application of
    another op that take the same inputs as what they are replacing.

    e.g. OpSub(add, sub) ==> add(div(x, y), add(y, x)) -> sub(div(x, y), sub(y, x))
    """

    reentrant = False      # an OpSub does not apply to the nodes it produces
    retains_inputs = True  # all the inputs of the original node are transferred to the outputs

    def __init__(self, op1, op2, transfer_tags = True):
        """
        op1.make_node and op2.make_node must take the same number of
        inputs and have the same number of outputs.
        """
        self.op1 = op1
        self.op2 = op2
        self.transfer_tags = transfer_tags

    def op_key(self):
        return self.op1

    def transform(self, node):
        if node.op != self.op1:
            return False
        repl = self.op2.make_node(*node.inputs)
        if self.transfer_tags:
            repl.tag = copy(node.tag)
            for output, new_output in zip(node.outputs, repl.outputs):
                new_output.tag = copy(output.tag)
        return repl.outputs

    def str(self):
        return "%s -> %s" % (self.op1, self.op2)


class OpRemove(LocalOptimizer):
    """
    Removes all applications of an op by transferring each of its
    outputs to the corresponding input.
    """

    reentrant = False      # no nodes are added at all

    def __init__(self, op):
        """
        op1.make_node and op2.make_node must take the same number of
        inputs and have the same number of outputs.
        """
        self.op = op

    def op_key(self):
        return self.op

    def transform(self, node):
        if node.op != self.op:
            return False
        return node.inputs

    def str(self):
        return "%s(x) -> x" % (self.op)


class PatternSub(LocalOptimizer):
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

    def __init__(self, in_pattern, out_pattern, allow_multiple_clients = False):
        """
        Creates a PatternOptimizer that replaces occurrences of
        in_pattern by occurrences of out_pattern.

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
        self.allow_multiple_clients = allow_multiple_clients

    def op_key(self):
        return self.op

    def transform(self, node):
        """
        Checks if the graph from node corresponds to in_pattern. If it does,
        constructs out_pattern and performs the replacement.
        """
        if node.op != self.op:
            return False
        def match(pattern, expr, u, first = False):
            if isinstance(pattern, (list, tuple)):
                if expr.owner is None:
                    return False
                if not (expr.owner.op == pattern[0]) or (not self.allow_multiple_clients and not first and len(expr.clients) > 1):
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
                if constraint(expr):
                    return match(real_pattern, expr, u, False)
                else:
                    return False
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
            p = self.out_pattern
            new = build(p, u)
            return [new]
        else:
            return False

    def __str__(self):
        def pattern_to_str(pattern):
            if isinstance(pattern, (list, tuple)):
                return "%s(%s)" % (str(pattern[0]), ", ".join([pattern_to_str(p) for p in pattern[1:]]))
            elif isinstance(pattern, dict):
                return "%s subject to %s" % (pattern_to_str(pattern['pattern']), str(pattern['constraint']))
            else:
                return str(pattern)
        return "%s -> %s" % (pattern_to_str(self.in_pattern), pattern_to_str(self.out_pattern))



##################
### Navigators ###
##################

# Use the following classes to apply LocalOptimizers


class NavigatorOptimizer(Optimizer):

    def __init__(self, local_opt, ignore_newtrees = 'auto', failure_callback = None):
        self.local_opt = local_opt
        if ignore_newtrees == 'auto':
            self.ignore_newtrees = not getattr(local_opt, 'reentrant', True)
        else:
            self.ignore_newtrees = ignore_newtrees
        self.failure_callback = failure_callback

    def attach_updater(self, env, importer, pruner):
        if self.ignore_newtrees:
            importer = None
        
        if importer is None and pruner is None:
            return None

        class Updater:
            if importer is not None:
                def on_import(self, env, node):
                    importer(node)
            if pruner is not None:
                def on_prune(self, env, node):
                    pruner(node)
        u = Updater()
        env.extend(u)
        return u

    def detach_updater(self, env, u):
        if u is not None:
            env.remove_feature(u)

    def process_node(self, env, node):
        replacements = self.local_opt.transform(node)
        if replacements is False or replacements is None:
            return
        repl_pairs = zip(node.outputs, replacements)
        try:
            env.replace_all_validate(repl_pairs)
        except Exception, e:
            if self.failure_callback is not None:
                self.failure_callback(e, self, repl_pairs)
            else:
                raise

    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())



class TopoOptimizer(NavigatorOptimizer):

    def __init__(self, local_opt, order = 'in_to_out', ignore_newtrees = False, failure_callback = None):
        if order not in ['out_to_in', 'in_to_out']:
            raise ValueError("order must be 'out_to_in' or 'in_to_out'")
        self.order = order
        NavigatorOptimizer.__init__(self, local_opt, ignore_newtrees, failure_callback)

    def apply(self, env, start_from = None):
        if start_from is None: start_from = env.outputs
        q = deque(graph.io_toposort(env.inputs, start_from))
        def importer(node):
            q.append(node)
        def pruner(node):
            if node is not current_node:
                try: q.remove(node)
                except ValueError: pass
        
        u = self.attach_updater(env, importer, pruner)
        try:
            while q:
                if self.order == 'out_to_in':
                    node = q.pop()
                else:
                    node = q.popleft()
                current_node = node
                self.process_node(env, node)
        except:
            self.detach_updater(env, u)
            raise
        


class OpKeyOptimizer(NavigatorOptimizer):

    def __init__(self, local_opt, ignore_newtrees = False, failure_callback = None):
        if not hasattr(local_opt, 'op_key'):
            raise TypeError("LocalOptimizer for OpKeyOptimizer must have an 'op_key' method.")
        NavigatorOptimizer.__init__(self, local_opt, ignore_newtrees, failure_callback)
    
    def apply(self, env):
        op = self.local_opt.op_key()
        if isinstance(op, (list, tuple)):
            q = reduce(list.__iadd__, map(env.get_nodes, op))
        else:
            q = list(env.get_nodes(op))
        def importer(node):
            if node.op == op: q.append(node)
        def pruner(node):
            if node is not current_node and node.op == op:
                try: q.remove(node)
                except ValueError: pass
        u = self.attach_updater(env, importer, pruner)
        try:
            while q:
                node = q.pop()
                current_node = node
                self.process_node(env, node)
        except:
            self.detach_updater(env, u)
            raise

    def add_requirements(self, env):
        """
        Requires the following features:
          - NodeFinder
          - ReplaceValidate
        """
        NavigatorOptimizer.add_requirements(self, env)
        env.extend(toolbox.NodeFinder())


def keep_going(exc, nav, repl_pairs):
    pass


##############################
### Pre-defined optimizers ###
##############################

def ExpandMacros(filter = None):
    return TopoOptimizer(ExpandMacro(filter = filter),
                         order = 'in_to_out',
                         ignore_newtrees = False)
