
from op import Op
from env import InconsistencyError
import utils
import unify
import features
import ext


class Optimizer:

    def apply(self, env):
        pass

    def optimize(self, env):
        env.satisfy(self)
        self.apply(env)

    def __call__(self, env):
        self.optimize(env)


DummyOpt = Optimizer()



class SeqOptimizer(Optimizer, list):

    def apply(self, env):
        for optimizer in self:
            optimizer.optimize(env)

    def __str__(self):
        return "SeqOpt(%s)" % list.__str__(self)

    def __repr__(self):
        return list.__repr__(self)



class LocalOptimizer(Optimizer):

    def candidates(self, env):
        return env.ops()

    def apply_on_op(self, env, op):
        raise Exception("Please override this function.")

    def apply(self, env):
        for op in self.candidates(env):
            if env.has_op(op):
                self.apply_on_op(env, op)

#         no_change_listener = graph.changed is None
#         while(True):
#             exprs = self.candidates(graph)
#             graph.changed(False)
#             for expr in exprs:
#                 self.apply_on_op(graph, expr)
#                 if no_change_listener or graph.changed:
#                     break
#             else:
#                 break


class OpSpecificOptimizer(LocalOptimizer):

    __env_require__ = features.InstanceFinder

    opclass = Op

    def candidates(self, env):
        return env.get_instances_of(self.opclass)




class OpSubOptimizer(Optimizer):

    __env_require__ = features.InstanceFinder

    def __init__(self, op1, op2):
        if not op1.has_default_output:
            raise TypeError("OpSubOptimizer must be used with Op instances that have a default output.")
        # note: op2 must have the same input signature as op1
        self.op1 = op1
        self.op2 = op2

    def apply(self, env):
        candidates = env.get_instances_of(self.op1)

        for op in candidates:
            try:
                # note: only replaces the default 'out' port if it exists
                r = self.op2(*op.inputs)
                if isinstance(r, Op):
                    r = r.out
                env.replace(op.out, r)
            except InconsistencyError, e:
                print "Warning: OpSubOpt failed to transform %s into %s: %s" % (op, self.op2, str(e)) # warning is for debug
                pass



class OpRemover(Optimizer):

    __env_require__ = features.InstanceFinder

    def __init__(self, opclass):
        self.opclass = opclass

    def apply(self, env):
        candidates = env.get_instances_of(self.opclass)

        for op in candidates:
            try:
                assert len(op.inputs) == len(op.outputs)
                for input, output in zip(op.inputs, op.outputs):
                    env.replace(output, input)
            except InconsistencyError, e:
                print "Warning: OpRemover failed to remove %s: %s" % (op, str(e)) # warning is for debug
                pass



class PatternOptimizer(OpSpecificOptimizer):
    """
    Replaces all occurrences of the first pattern by the second pattern.
    """

    def __init__(self, in_pattern, out_pattern):
        self.in_pattern = in_pattern
        self.out_pattern = out_pattern
        self.opclass = self.in_pattern[0]
        self.__doc__ = self.__class__.__doc__ + "\n\nThis instance does: " + str(self) + "\n"

    def apply_on_op(self, env, op):

        def match(pattern, expr, u, first = False):
            if isinstance(pattern, (list, tuple)):
                if not issubclass(expr.owner.__class__, pattern[0]) or (not first and env.nclients(expr.owner) > 1):
                    return False
                if len(pattern) - 1 != len(expr.owner.inputs):
                    return False
                for p, v in zip(pattern[1:], expr.owner.inputs):
                    u = match(p, v, u)
                    if not u:
                        return False
            elif isinstance(pattern, str):
                v = unify.Var(pattern)
                if u[v] is not v and u[v] is not expr:
                    return False
                else:
                    u = u.merge(expr, v)
            else:
                if pattern != expr:
                    return False
                return u
            return u

        def build(pattern, u):
            if isinstance(pattern, (list, tuple)):
                return pattern[0](*[build(p, u) for p in pattern[1:]])
            elif isinstance(pattern, str):
                return u[unify.Var(pattern)]
            else:
                return pattern

        u = match(self.in_pattern, op.out, unify.Unification(), True)
        if u:
            try:
                # note: only replaces the default 'out' port if it exists
                new = build(self.out_pattern, u)
                if isinstance(new, Op):
                    new = new.out
                env.replace(op.out, new)
            except InconsistencyError, e:
                print "Warning: '%s' failed to apply on %s: %s" % (self, op, str(e)) # warning is for debug
                pass


    def __str__(self):
        def pattern_to_str(pattern):
            if isinstance(pattern, (list, tuple)):
                return "%s(%s)" % (pattern[0].__name__, ", ".join([pattern_to_str(p) for p in pattern[1:]]))
            else:
                return str(pattern)
        return "%s -> %s" % (pattern_to_str(self.in_pattern), pattern_to_str(self.out_pattern))



class MergeOptimizer(Optimizer):

    def apply(self, env):
        cid = {}
        inv_cid = {}
        for i, r in enumerate(env.inputs.union(env.orphans())):
            cid[r] = i
            inv_cid[i] = r

        for op in env.io_toposort():
            op_cid = (op.__class__, tuple([cid[input] for input in op.inputs]))
            dup = inv_cid.get(op_cid, None)
            if dup is None:
                cid[op] = op_cid
                inv_cid[op_cid] = op
                for i, output in enumerate(op.outputs):
                    ref = (i, op_cid)
                    cid[output] = ref
                    inv_cid[ref] = output
            else:
                for output, other_output in zip(op.outputs, dup.outputs):
                    #print "replacing: %s %s" % (repr(output.owner), repr(other_output.owner))
                    env.replace(output, other_output)



def MergeOptMerge(opt):
    merger = MergeOptimizer()
    return SeqOptimizer([merger, opt, merger])



class MultiOptimizer(Optimizer):

    def __init__(self, **opts):
        self._opts = []
        self.ord = {}
        self.name_to_opt = {}
        self.up_to_date = True
        for name, opt in opts:
            self.register(name, opt, after = [], before = [])

    def register(self, name, opt, **relative):
        self.name_to_opt[name] = opt
        
        after = relative.get('after', [])
        if not isinstance(after, (list, tuple)):
            after = [after]
        
        before = relative.get('before', [])
        if not isinstance(before, (list, tuple)):
            before = [before]
        
        self.up_to_date = False

        if name in self.ord:
            raise Exception("Cannot redefine optimization: '%s'" % name)
        
        self.ord[name] = set(after)
        
        for postreq in before:
            self.ord.setdefault(postreq, set()).add(name)

    def get_opts(self):
        if not self.up_to_date:
            self.refresh()
        return self._opts

    def refresh(self):
        self._opts = [self.name_to_opt[name] for name in utils.toposort(self.ord)]
        self.up_to_date = True

    def apply(self, env):
        for opt in self.opts:
            opt.apply(env)

    opts = property(get_opts)



class TaggedMultiOptimizer(MultiOptimizer):
    
    def __init__(self, **opts):
        self.tags = {}
        MultiOptimizer.__init__(self, **opts)

    def register(self, name, opt, tags = [], **relative):
        tags = set(tags)
        tags.add(name)
        self.tags[opt] = tags
        MultiOptimizer.register(self, name, opt, **relative)

    def filter(self, whitelist, blacklist):
        return [opt for opt in self.opts
                if self.tags[opt].intersection(whitelist)
                and not self.tags[opt].intersection(blacklist)]

    def whitelist(self, *tags):
        return [opt for opt in self.opts if self.tags[opt].intersection(tags)]

    def blacklist(self, *tags):
        return [opt for opt in self.opts if not self.tags[opt].intersection(tags)]



class TagFilterMultiOptimizer(Optimizer):

    def __init__(self, all, whitelist = None, blacklist = None):
        self.all = all
        
        if whitelist is not None:
            self.whitelist = set(whitelist)
        else:
            self.whitelist = None

        if blacklist is not None:
            self.blacklist = set(blacklist)
        else:
            self.blacklist = set()

    def use_whitelist(self, use = True):
        if self.whitelist is None and use:
            self.whitelist = set()

    def allow(self, *tags):
        if self.whitelist is not None:
            self.whitelist.update(tags)
        self.blacklist.difference_update(tags)

    def deny(self, *tags):
        if self.whitelist is not None:
            self.whitelist.difference_update(tags)
        self.blacklist.update(tags)

    def dont_care(self, *tags):
        if self.whitelist is not None:
            self.whitelist.difference_update(tags)
        self.blacklist.difference_update(tags)

    def opts(self):
        if self.whitelist is not None:
            return self.all.filter(self.whitelist, self.blacklist)
        else:
            return self.all.blacklist(*[tag for tag in self.blacklist])
    
    def apply(self, env):
        for opt in self.opts():
            opt.apply(env)
