
from op import Op
from graph import Constant
from type import Type
from env import InconsistencyError
import utils
import unify
import toolbox
import ext


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
        to use a certain tool, such as an L{InstanceFinder}, it should
        set the L{__env_require__} field to a list of what needs to be
        registered with the L{Env}.
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
        return self.optimize(env)

    def add_requirements(self, env):
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
     - apply_on_op(env, op) -> for each op in candidates,
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
            if env.has_node(node):
                self.apply_on_node(env, node)



class OpSpecificOptimizer(LocalOptimizer):
    """
    Generic L{Optimizer} that applies only to ops of a certain
    type. The type in question is accessed through L{self.op}.
    op can also be a class variable of the subclass.
    """

    def add_requirements(self, env):
        env.extend(toolbox.NodeFinder(env))

    def candidates(self, env):
        """
        Returns all instances of L{self.op}.
        """
        return env.get_nodes(self.op)




class OpSubOptimizer(Optimizer):
    """
    Replaces all L{Op}s of a certain type by L{Op}s of another type that
    take the same inputs as what they are replacing.

    e.g. OpSubOptimizer(add, sub) ==> add(div(x, y), add(y, x)) -> sub(div(x, y), sub(y, x))
    """

    def add_requirements(self, env):
        env.extend(toolbox.NodeFinder(env))

    def __init__(self, op1, op2, failure_callback = None):
        """
        op1 and op2 must both be Op subclasses, they must both take
        the same number of inputs and they must both have the same
        number of outputs.
        """
        self.op1 = op1
        self.op2 = op2
        self.failure_callback = failure_callback

    def apply(self, env):
        """
        Replaces all occurrences of self.op1 by instances of self.op2
        with the same inputs.
        
        If failure_callback is not None, it will be called whenever
        the Optimizer fails to do a replacement in the graph. The
        arguments to the callback are: (op1_instance, replacement, exception)
        """
        candidates = env.get_nodes(self.op1)

        for node in candidates:
            try:
                repl = self.op2.make_node(*node.inputs)
                assert len(node.outputs) == len(repl.outputs)
                for old, new in zip(node.outputs, repl.outputs):
                    env.replace(old, new)
            except Exception, e:
                if self.failure_callback is not None:
                    self.failure_callback(node, repl, e)
                pass

    def str(self):
        return "%s -> %s" % (self.op1, self.op2)



class OpRemover(Optimizer):
    """
    @todo untested
    Removes all ops of a certain type by transferring each of its
    outputs to the corresponding input.
    """

    def add_requirements(self, env):
        env.extend(toolbox.NodeFinder(env))

    def __init__(self, op, failure_callback = None):
        """
        opclass is the class of the ops to remove. It must take as
        many inputs as outputs.
        """
        self.op = op
        self.failure_callback = failure_callback

    def apply(self, env):
        """
        Removes all occurrences of self.opclass.
        
        If self.failure_callback is not None, it will be called whenever
        the Optimizer fails to remove an operation in the graph. The
        arguments to the callback are: (opclass_instance, exception)
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
    
    Replaces all occurrences of the input pattern by the output pattern::

     input_pattern ::= (OpClass, <sub_pattern1>, <sub_pattern2>, ...)
     input_pattern ::= dict(pattern = <input_pattern>,
                            constraint = <constraint>)
     sub_pattern ::= input_pattern
     sub_pattern ::= string
     sub_pattern ::= a Result r such that r.constant is True
     constraint ::= lambda env, expr: additional matching condition
     
     output_pattern ::= (OpClass, <output_pattern1>, <output_pattern2>, ...)
     output_pattern ::= string

    Each string in the input pattern is a variable that will be set to
    whatever expression is found in its place. If the same string is
    used more than once, the same expression must be found in those
    places. If a string used in the input pattern is used in the
    output pattern, the matching expression will be inserted in its
    place. The input pattern cannot just be a string but the output
    pattern can.

    If you put a constant result in the input pattern, there will be a
    match iff a constant result with the same value is found in its
    place.

    You can add a constraint to the match by using the dict(...)  form
    described above with a 'constraint' key. The constraint must be a
    function that takes the env and the current Result that we are
    trying to match and returns True or False according to an
    arbitrary criterion.

    Examples:
     PatternOptimizer((Add, 'x', 'y'), (Add, 'y', 'x'))
     PatternOptimizer((Multiply, 'x', 'x'), (Square, 'x'))
     PatternOptimizer((Subtract, (Add, 'x', 'y'), 'y'), 'x')
     PatternOptimizer((Power, 'x', Double(2.0, constant = True)), (Square, 'x'))
     PatternOptimizer((Boggle, {'pattern': 'x',
                                'constraint': lambda env, expr: expr.owner.scrabble == True}),
                      (Scrabble, 'x'))
    """

    def __init__(self, in_pattern, out_pattern, allow_multiple_clients = False, failure_callback = None):
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
        Checks if the graph from op corresponds to in_pattern. If it does,
        constructs out_pattern and performs the replacement.

        If self.failure_callback is not None, if there is a match but a
        replacement fails to occur, the callback will be called with
        arguments (results_to_replace, replacement, exception).

        If self.allow_multiple_clients is False, he pattern matching will fail
        if one of the subpatterns has more than one client.
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
            elif isinstance(pattern, Constant) and isinstance(expr, Constant) and pattern.equals(expr):
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



# class ConstantFinder(Optimizer):
#     """
#     Sets as constant every orphan that is not destroyed.
#     """
    
#     def apply(self, env):
#         if env.has_feature(ext.DestroyHandler(env)):
#             for r in env.orphans():
#                 if not env.destroyers(r):
#                     r.indestructible = True
#                     r.constant = True
# #             for r in env.inputs:
# #                 if not env.destroyers(r):
# #                     r.indestructible = True
#         else:
#             for r in env.orphans():
#                 r.indestructible = True
#                 r.constant = True
# #             for r in env.inputs:
# #                 r.indestructible = True

import graph

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

    def apply(self, env):
        cid = _metadict()     #result -> result.desc()  (for constants)
        inv_cid = _metadict() #desc -> result (for constants)
        for i, r in enumerate(env.orphans.union(env.inputs)):
            if isinstance(r, Constant):
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
        for i, r in enumerate(env.orphans.union(env.inputs)):
            cid[r] = i
            inv_cid[i] = r

        for node in env.io_toposort():
            node_cid = (node.op, tuple([cid[input] for input in node.inputs]))
            dup = inv_cid.get(node_cid, None)
            success = False
            if dup is not None:
                success = True
                d = dict(zip(node.outputs, dup.outputs))
                try:
                    env.replace_all(d)
                except Exception, e:
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



### THE FOLLOWING OPTIMIZERS ARE NEITHER USED NOR TESTED BUT PROBABLY WORK AND COULD BE USEFUL ###

# class MultiOptimizer(Optimizer):

#     def __init__(self, **opts):
#         self._opts = []
#         self.ord = {}
#         self.name_to_opt = {}
#         self.up_to_date = True
#         for name, opt in opts:
#             self.register(name, opt, after = [], before = [])

#     def register(self, name, opt, **relative):
#         self.name_to_opt[name] = opt
        
#         after = relative.get('after', [])
#         if not isinstance(after, (list, tuple)):
#             after = [after]
        
#         before = relative.get('before', [])
#         if not isinstance(before, (list, tuple)):
#             before = [before]
        
#         self.up_to_date = False

#         if name in self.ord:
#             raise Exception("Cannot redefine optimization: '%s'" % name)
        
#         self.ord[name] = set(after)
        
#         for postreq in before:
#             self.ord.setdefault(postreq, set()).add(name)

#     def get_opts(self):
#         if not self.up_to_date:
#             self.refresh()
#         return self._opts

#     def refresh(self):
#         self._opts = [self.name_to_opt[name] for name in utils.toposort(self.ord)]
#         self.up_to_date = True

#     def apply(self, env):
#         for opt in self.opts:
#             opt.apply(env)

#     opts = property(get_opts)



# class TaggedMultiOptimizer(MultiOptimizer):
    
#     def __init__(self, **opts):
#         self.tags = {}
#         MultiOptimizer.__init__(self, **opts)

#     def register(self, name, opt, tags = [], **relative):
#         tags = set(tags)
#         tags.add(name)
#         self.tags[opt] = tags
#         MultiOptimizer.register(self, name, opt, **relative)

#     def filter(self, whitelist, blacklist):
#         return [opt for opt in self.opts
#                 if self.tags[opt].intersection(whitelist)
#                 and not self.tags[opt].intersection(blacklist)]

#     def whitelist(self, *tags):
#         return [opt for opt in self.opts if self.tags[opt].intersection(tags)]

#     def blacklist(self, *tags):
#         return [opt for opt in self.opts if not self.tags[opt].intersection(tags)]



# class TagFilterMultiOptimizer(Optimizer):

#     def __init__(self, all, whitelist = None, blacklist = None):
#         self.all = all
        
#         if whitelist is not None:
#             self.whitelist = set(whitelist)
#         else:
#             self.whitelist = None

#         if blacklist is not None:
#             self.blacklist = set(blacklist)
#         else:
#             self.blacklist = set()

#     def use_whitelist(self, use = True):
#         if self.whitelist is None and use:
#             self.whitelist = set()

#     def allow(self, *tags):
#         if self.whitelist is not None:
#             self.whitelist.update(tags)
#         self.blacklist.difference_update(tags)

#     def deny(self, *tags):
#         if self.whitelist is not None:
#             self.whitelist.difference_update(tags)
#         self.blacklist.update(tags)

#     def dont_care(self, *tags):
#         if self.whitelist is not None:
#             self.whitelist.difference_update(tags)
#         self.blacklist.difference_update(tags)

#     def opts(self):
#         if self.whitelist is not None:
#             return self.all.filter(self.whitelist, self.blacklist)
#         else:
#             return self.all.blacklist(*[tag for tag in self.blacklist])
    
#     def apply(self, env):
#         for opt in self.opts():
#             opt.apply(env)
