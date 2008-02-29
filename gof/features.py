
# from copy import copy
# from op import Op
# import result
# import graph

import utils

# from random import shuffle


__all__ = ['Feature',
           'Listener',
           'Constraint',
           'Orderings',
           'Tool',
           'uniq_features',
#           'EquivTool',
#           'InstanceFinder',
#           'PrintListener',
#           'ChangeListener',
           ]



class Feature(object):
    
    def __init__(self, env):
        """
        Initializes the Feature's env field to the parameter
        provided.
        """
        self.env = env


class Listener(Feature):
    """
    When registered by an env, each listener is informed of any op
    entering or leaving the subgraph (which happens at construction
    time and whenever there is a replacement).
    """

    def on_import(self, op):
        """
        This method is called by the env whenever a new op is
        added to the graph.
        """
        raise utils.AbstractFunctionError()
    
    def on_prune(self, op):
        """
        This method is called by the env whenever an op is
        removed from the graph.
        """
        raise utils.AbstractFunctionError()

    def on_rewire(self, clients, r, new_r):
        """
        clients -> (op, i) pairs such that op.inputs[i] is new_r
                   but used to be r
        r -> the old result that was used by the ops in clients
        new_r -> the new result that is now used by the ops in clients

        Note that the change from r to new_r is done before this
        method is called.
        """
        raise utils.AbstractFunctionError()


class Constraint(Feature):
    """
    When registered by an env, a Constraint can restrict the ops that
    can be in the subgraph or restrict the ways ops interact with each
    other.
    """

    def validate(self):
        """
        Raises an L{InconsistencyError} if the env is currently
        invalid from the perspective of this object.
        """
        raise utils.AbstractFunctionError()


class Orderings(Feature):
    """
    When registered by an env, an Orderings object can provide supplemental
    ordering constraints to the subgraph's topological sort.
    """

    def orderings(self):
        """
        Returns {op: set(ops that must be evaluated before this op), ...}
        This is called by env.orderings() and used in env.toposort() but
        not in env.io_toposort().
        """
        raise utils.AbstractFunctionError()


class Tool(Feature):
    """
    A Tool can extend the functionality of an env so that, for example,
    optimizations can have access to efficient ways to search the graph.
    """

    def publish(self):
        """
        This is only called once by the env, when the Tool is added.
        Adds methods to env.
        """
        raise utils.AbstractFunctionError()



def uniq_features(_features, *_rest):
    """Return a list such that no element is a subclass of another"""
    # used in Env.__init__ to 
    features = [x for x in _features]
    for other in _rest:
        features += [x for x in other]
    res = []
    while features:
        feature = features.pop()
        for feature2 in features:
            if issubclass(feature2, feature):
                break
        else:
            res.append(feature)
    return res



# MOVE TO LIB

# class EquivTool(Listener, Tool, dict):

#     def on_rewire(self, clients, r, new_r):
#         repl = self(new_r)
#         if repl is r:
#             self.ungroup(r, new_r)
#         elif repl is not new_r:
#             raise Exception("Improper use of EquivTool!")
#         else:
#             self.group(new_r, r)

#     def publish(self):
#         self.env.equiv = self

#     def group(self, main, *keys):
#         "Marks all the keys as having been replaced by the Result main."
#         keys = [key for key in keys if key is not main]
#         if self.has_key(main):
#             raise Exception("Only group results that have not been grouped before.")
#         for key in keys:
#             if self.has_key(key):
#                 raise Exception("Only group results that have not been grouped before.")
#             if key is main:
#                 continue
#             self.setdefault(key, main)

#     def ungroup(self, main, *keys):
#         "Undoes group(main, *keys)"
#         keys = [key for key in keys if key is not main]
#         for key in keys:
#             if self[key] is main:
#                 del self[key]

#     def __call__(self, key):
#         "Returns the currently active replacement for the given key."
#         next = self.get(key, None)
#         while next:
#             key = next
#             next = self.get(next, None)
#         return key


# class InstanceFinder(Listener, Tool, dict):

#     def __init__(self, env):
#         self.env = env

#     def all_bases(self, cls):
#         return utils.all_bases(cls, lambda cls: issubclass(cls, Op))
# #        return [cls for cls in utils.all_bases(cls) if issubclass(cls, Op)]

#     def on_import(self, op):
#         for base in self.all_bases(op.__class__):
#             self.setdefault(base, set()).add(op)

#     def on_prune(self, op):
#         for base in self.all_bases(op.__class__):
#             self[base].remove(op)
#             if not self[base]:
#                 del self[base]

#     def __query__(self, cls):
#         all = [x for x in self.get(cls, [])]
#         shuffle(all) # this helps a lot for debugging because the order of the replacements will vary
#         while all:
#             next = all.pop()
#             if next in self.env.ops():
#                 yield next

#     def query(self, cls):
#         return self.__query__(cls)

#     def publish(self):
#         self.env.get_instances_of = self.query



# class PrintListener(Listener):

#     def __init__(self, env, active = True):
#         self.env = env
#         self.active = active
#         if active:
#             print "-- initializing"

#     def on_import(self, op):
#         if self.active:
#             print "-- importing: %s" % graph.as_string(self.env.inputs, op.outputs)

#     def on_prune(self, op):
#         if self.active:
#             print "-- pruning: %s" % graph.as_string(self.env.inputs, op.outputs)

#     def on_rewire(self, clients, r, new_r):
#         if self.active:
#             if r.owner is None:
#                 rg = id(r) #r.name
#             else:
#                 rg = graph.as_string(self.env.inputs, r.owner.outputs)
#             if new_r.owner is None:
#                 new_rg = id(new_r) #new_r.name
#             else:
#                 new_rg = graph.as_string(self.env.inputs, new_r.owner.outputs)
#             print "-- moving from %s to %s" % (rg, new_rg)



# class ChangeListener(Listener):

#     def __init__(self, env):
#         self.change = False

#     def on_import(self, op):
#         self.change = True

#     def on_prune(self, op):
#         self.change = True

#     def on_rewire(self, clients, r, new_r):
#         self.change = True

#     def __call__(self, value = "get"):
#         if value == "get":
#             return self.change
#         else:
#             self.change = value

