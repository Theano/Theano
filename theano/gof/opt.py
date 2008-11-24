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
from collections import deque, defaultdict
import destroyhandler as dh
import sys

_optimizer_idx = [0]

def _list_of_nodes(env):
    return graph.io_toposort(env.inputs, env.outputs)

class Optimizer(object):
    """WRITEME
    An L{Optimizer} can be applied to an L{Env} to transform it.
    It can represent an optimization or in general any kind
    of transformation you could apply to an L{Env}.
    """

    def __hash__(self):
        if not hasattr(self, '_optimizer_idx'):
            self._optimizer_idx = _optimizer_idx[0]
            _optimizer_idx[0] += 1
        return self._optimizer_idx

    def apply(self, env):
        """WRITEME
        Applies the optimization to the provided L{Env}. It may use all
        the methods defined by the L{Env}. If the L{Optimizer} needs
        to use a certain tool, such as an L{InstanceFinder}, it can do
        so in its L{add_requirements} method.
        """
        pass

    def optimize(self, env, *args, **kwargs):
        """WRITEME
        This is meant as a shortcut to::
          opt.add_requirements(env)
          opt.apply(env)
        """
        self.add_requirements(env)
        self.apply(env, *args, **kwargs)

    def __call__(self, env):
        """WRITEME
        Same as self.optimize(env)
        """
        return self.optimize(env)

    def add_requirements(self, env):
        """WRITEME
        Add features to the env that are required to apply the optimization.
        For example:
          env.extend(History())
          env.extend(MyFeature())
          etc.
        """
        pass


class FromFunctionOptimizer(Optimizer):
    """WRITEME"""
    def __init__(self, fn):
        self.apply = fn
    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())

def optimizer(f):
    """decorator for FromFunctionOptimizer"""
    return FromFunctionOptimizer(f)



class SeqOptimizer(Optimizer, list):
    #inherit from Optimizer first to get Optimizer.__hash__
    """WRITEME
    Takes a list of L{Optimizer} instances and applies them
    sequentially.
    """

    def __init__(self, *opts, **kw):
        """WRITEME"""
        if len(opts) == 1 and isinstance(opts[0], (list, tuple)):
            opts = opts[0]
        self[:] = opts
        self.failure_callback = kw.pop('failure_callback', None)

    def apply(self, env):
        """WRITEME
        Applies each L{Optimizer} in self in turn.
        """
        for optimizer in self:
            try:
                optimizer.optimize(env)
            except Exception, e:
                if self.failure_callback:
                    self.failure_callback(e, self, optimizer)
                    continue
                else:
                    raise

    def __eq__(self, other):
        #added to override the list's __eq__ implementation
        return id(self) == id(other)

    def __neq__(self, other):
        #added to override the list's __neq__ implementation
        return id(self) != id(other)


    def __str__(self):
        return "SeqOpt(%s)" % list.__str__(self)

    def __repr__(self):
        return list.__repr__(self)



class _metadict:
    """WRITEME"""
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
            for i, (key,val) in enumerate(self.l):
                if key == item:
                    self.l[i] = (item, value)
                    return
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
    """WRITEME
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
                if r.name: other_r.name = r.name
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

        for node in _list_of_nodes(env):
            node_cid = (node.op, tuple([cid[input] for input in node.inputs]))
            dup = inv_cid.get(node_cid, None)
            success = False
            if dup is not None:
                success = True
                pairs = zip(node.outputs, dup.outputs)
                for output, new_output in pairs:
                    if output.name and not new_output.name:
                        new_output.name = output.name
                try:
                    env.replace_all_validate(pairs)
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
    """WRITEME
    Returns an Optimizer that merges the graph then applies the
    optimizer in opt and then merges the graph again in case the
    opt introduced additional similarities.
    """
    merger = MergeOptimizer()
    return SeqOptimizer([merger, opt, merger])



########################
### Local Optimizers ###
########################

class LocalOptimizer(object):
    """A class for node-based optimizations.

    Instances should implement the transform function, 
    and be passed to configure a env-based Optimizer instance.
    """

    def __hash__(self):
        if not hasattr(self, '_optimizer_idx'):
            self._optimizer_idx = _optimizer_idx[0]
            _optimizer_idx[0] += 1
        return self._optimizer_idx

    def transform(self, node):
        """Transform a subgraph whose output is `node`.

        Subclasses should implement this function so that it returns one of two
        kinds of things:

        - False to indicate that no optimization can be applied to this `node`; or

        - <list of results> to use in place of `node`'s outputs in the greater graph.

        :type node: an Apply instance

        """

        raise utils.AbstractFunctionError()


class FromFunctionLocalOptimizer(LocalOptimizer):
    """WRITEME"""
    def __init__(self, fn, tracks = []):
        self.transform = fn
        self._tracks = tracks
    def tracks(self):
        return self._tracks
    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())
    def __str__(self):
        return getattr(self, 'name', '<FromFunctionLocalOptimizer instance>')

def local_optimizer(*tracks):
    def decorator(f):
        """WRITEME"""
        rval = FromFunctionLocalOptimizer(f, tracks)
        rval.__name__ = f.__name__
        return rval
    return decorator


class LocalOptGroup(LocalOptimizer):
    """WRITEME"""

    def __init__(self, *optimizers):
        self.opts = optimizers
        self.reentrant = any(getattr(opt, 'reentrant', True) for opt in optimizers)
        self.retains_inputs = all(getattr(opt, 'retains_inputs', False) for opt in optimizers)

    def transform(self, node):
        for opt in self.opts:
            repl = opt.transform(node)
            if repl:
                return repl


class _LocalOpKeyOptGroup(LocalOptGroup):
    """WRITEME"""

    def __init__(self, optimizers):
        if any(not hasattr(opt, 'op_key'), optimizers):
            raise TypeError("All LocalOptimizers passed here must have an op_key method.")
        CompositeLocalOptimizer.__init__(self, optimizers)
    
    def op_key(self):
        return [opt.op_key() for opt in self.opts]


class OpSub(LocalOptimizer):
    """WRITEME
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

    def tracks(self):
        return [[self.op1]]

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
    """WRITEME
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

    def tracks(self):
        return [[self.op]]

    def transform(self, node):
        if node.op != self.op:
            return False
        return node.inputs

    def str(self):
        return "%s(x) -> x" % (self.op)


class PatternSub(LocalOptimizer):
    """WRITEME
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

    def tracks(self):
        def helper(pattern, sofar):
            if isinstance(pattern, (list, tuple)):
                sofar = sofar + (pattern[0],)
                return reduce(tuple.__add__,
                              tuple(helper(p, sofar) for p in pattern[1:]),
                              ())
            elif isinstance(pattern, dict):
                return helper(pattern['pattern'], sofar)
            else:
                return (sofar,)
        return set(helper(self.in_pattern, ()))

    def transform(self, node):
        """
        Checks if the graph from node corresponds to in_pattern. If it does,
        constructs out_pattern and performs the replacement.
        """
        if node.op != self.op:
            return False
        def match(pattern, expr, u, allow_multiple_clients = False):
            if isinstance(pattern, (list, tuple)):
                if expr.owner is None:
                    return False
                if not (expr.owner.op == pattern[0]) or (not allow_multiple_clients and len(expr.clients) > 1):
                    return False
                if len(pattern) - 1 != len(expr.owner.inputs):
                    return False
                for p, v in zip(pattern[1:], expr.owner.inputs):
                    u = match(p, v, u, self.allow_multiple_clients)
                    if not u:
                        return False
            elif isinstance(pattern, dict):
                try:
                    real_pattern = pattern['pattern']
                except KeyError:
                    raise KeyError("Malformed pattern: %s (expected key 'pattern')" % pattern)
                constraint = pattern.get('constraint', lambda expr: True)
                if constraint(expr):
                    return match(real_pattern, expr, u, pattern.get('allow_multiple_clients', False))
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
                return pattern.clone()

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
                return "%s subject to %s" % (pattern_to_str(pattern['pattern']), str(pattern.get('constraint', 'no conditions')))
            else:
                return str(pattern)
        return "%s -> %s" % (pattern_to_str(self.in_pattern), pattern_to_str(self.out_pattern))

    def __repr__(self):
        return str(self)



##################
### Navigators ###
##################

# Use the following classes to apply LocalOptimizers


class NavigatorOptimizer(Optimizer):
    """Abstract class
    
    """

    def __init__(self, local_opt, ignore_newtrees = 'auto', failure_callback = None):
        """
        :param local_opt:  a LocalOptimizer to apply over a Env.
        :param ignore_newtrees: 
            - True: new subgraphs returned by an optimization is not a candidate for optimization
            - False: new subgraphs returned by an optimization is a candidate for optimization
            - 'auto': let the local_opt set this parameter via its 'reentrant' attribute.
        :param failure_callback:
            a function that takes (exception, navigator, [(old, new),
            (old,new),...]) and we call it if there's an exception.
              
            If the trouble is from local_opt.transform(), the new variables will be 'None'.

            If the trouble is from validation (the new types don't match for
            example) then the new variables will be the ones created by
            transform().

            If this parameter is None, then exceptions are not caught here (raised normally).
        """
        self.local_opt = local_opt
        if ignore_newtrees == 'auto':
            self.ignore_newtrees = not getattr(local_opt, 'reentrant', True)
        else:
            self.ignore_newtrees = ignore_newtrees
        self.failure_callback = failure_callback

    def attach_updater(self, env, importer, pruner, chin = None):
        """Install some Env listeners to help the navigator deal with the ignore_trees-related functionality.

        :param importer: function that will be called whenever when optimizations add stuff to the graph.
        :param pruner: function to be called when optimizations remove stuff from graph.
        :param chin: "on change input" called whenever an node's inputs change.

        :returns: The Env plugin that handles the three tasks.  Keep this around so that you can detach later!

        """
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
            if chin is not None:
                def on_change_input(self, env, node, i, r, new_r):
                    chin(node, i, r, new_r)

        u = Updater()
        env.extend(u)
        return u

    def detach_updater(self, env, u):
        """Undo the work of attach_updater.

        :param u: a return-value of attach_updater

        :returns: None.
        """
        if u is not None:
            env.remove_feature(u)

    def process_node(self, env, node, lopt = None):
        lopt = lopt or self.local_opt
        try:
            replacements = lopt.transform(node)
        except Exception, e:
            if self.failure_callback is not None:
                self.failure_callback(e, self, [(x, None) for x in node.outputs])
                return False
            else:
                raise
        if replacements is False or replacements is None:
            return False
        repl_pairs = zip(node.outputs, replacements)
        try:
            env.replace_all_validate(repl_pairs)
            return True
        except Exception, e:
            if self.failure_callback is not None:
                self.failure_callback(e, self, repl_pairs)

                #DEBUG DONT PUSH
                #print lopt 
                #print dir(lopt)
                #raise
                #END

                return False
            else:
                raise

    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())



class TopoOptimizer(NavigatorOptimizer):
    """WRITEME"""

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
        self.detach_updater(env, u)


class OpKeyOptimizer(NavigatorOptimizer):
    """WRITEME"""

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
        self.detach_updater(env, u)

    def add_requirements(self, env):
        """
        Requires the following features:
          - NodeFinder
          - ReplaceValidate
        """
        NavigatorOptimizer.add_requirements(self, env)
        env.extend(toolbox.NodeFinder())



from utils import D

class EquilibriumOptimizer(NavigatorOptimizer):
    def __init__(self,
                 local_optimizers,
                 failure_callback = None,
                 max_depth = None,
                 max_use_ratio = None):
        """
        :param max_use_ratio: each optimizer can be applied at most (size of graph * this number)

        """

        super(EquilibriumOptimizer, self).__init__(
            None,
            ignore_newtrees = True,
            failure_callback = failure_callback)

        self.local_optimizers = local_optimizers
        self.max_depth = max_depth
        self.max_use_ratio = max_use_ratio

    def apply(self, env, start_from = None):
        if start_from is None:
            start_from = env.outputs
        changed = True
        max_use_abort = False
        process_count = {}

        while changed and not max_use_abort:
            changed = False

            q = deque(graph.io_toposort(env.inputs, start_from))

            max_use = len(q) * self.max_use_ratio
            def importer(node):
                q.append(node)
            def pruner(node):
                if node is not current_node:
                    try: q.remove(node)
                    except ValueError: pass
            
            u = self.attach_updater(env, importer, pruner)
            try:
                while q:
                    node = q.pop()
                    current_node = node
                    for lopt in self.local_optimizers:
                        process_count.setdefault(lopt, 0)
                        if process_count[lopt] > max_use:
                            max_use_abort = True
                        else:
                            lopt_change = self.process_node(env, node, lopt)
                            process_count[lopt] += 1 if lopt_change else 0
                            changed |= lopt_change
            except:
                self.detach_updater(env, u)
                raise
            self.detach_updater(env, u)
        if max_use_abort:
            print >> sys.stderr, "WARNING: EquilibriumOptimizer max'ed out"


def keep_going(exc, nav, repl_pairs):
    """WRITEME"""
    pass


import traceback
def warn(exc, nav, repl_pairs):
    """WRITEME"""
    traceback.print_exc()


#################
### Utilities ###
#################

def _check_chain(r, chain):
    """WRITEME"""
    chain = list(reversed(chain))
    while chain:
        elem = chain.pop()
        if elem is None:
            if not r.owner is None:
                return False
        elif r.owner is None:
            return False
        elif isinstance(elem, op.Op):
            if not r.owner.op == elem:
                return False
        else:
            try:
                if issubclass(elem, op.Op) and not isinstance(r.owner.op, elem):
                    return False
            except TypeError:
                return False
        if chain:
            r = r.owner.inputs[chain.pop()]
    return r

def check_chain(r, *chain):
    """WRITEME"""
    if isinstance(r, graph.Apply):
        r = r.outputs[0]
    return _check_chain(r, reduce(list.__iadd__, ([x, 0] for x in chain)))





############
### Misc ###
############

class InplaceOptimizer(Optimizer):

    def __init__(self, inplace):
        self.inplace = inplace

    def apply(self, env):
        self.inplace(env)

    def add_requirements(self, env):
        env.extend(dh.DestroyHandler())


class PureThenInplaceOptimizer(Optimizer):

    def __init__(self, pure, inplace):
        self.pure = pure
        self.inplace = inplace

    def apply(self, env):
        self.pure(env)
        env.extend(dh.DestroyHandler())
        self.inplace(env)



