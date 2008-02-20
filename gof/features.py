
from copy import copy
from op import Op
import result
import graph
import utils

from random import shuffle


__all__ = ['Feature',
           'Listener',
           'Constraint',
           'Orderings',
           'Tool',
           'EquivTool',
           'InstanceFinder',
           'PrintListener',
           'ChangeListener',
           ]



class Feature(object):

    def __init__(self, env):
        self.env = env


class Listener(Feature):

    def on_import(self, op):
        pass

    def on_prune(self, op):
        pass

    def on_rewire(self, clients, r, new_r):
        pass


class Constraint(Feature):

    def validate(self):
        return True


class Orderings(Feature):

    def orderings(self):
        return {}


class Tool(Feature):

    def publish(self):
        pass


# class Preprocessor(Feature):

#     def transform(self, inputs, outputs):
#         return inputs, outputs

#     def __call__(self, inputs, outputs):
#         return self.transform(inputs, outputs)


# class Optimization(object):

#     def require(self):
#         return []

#     def apply(self, env):
#         pass

#     def __call__(self, env):
#         return self.apply(env)




# Optimization
#  * require <tool_class>*
#  * apply

# Prog
#  * __init__
#    * inputs
#    * outputs
#    * listeners, constraints, orderings
#      * dispatched by isinstance Listener, etc.
#    * {tool_class: preferred_implementation, ...}



class EquivTool(Listener, Tool, dict):

    def on_rewire(self, clients, r, new_r):
        repl = self(new_r)
        if repl is r:
            self.ungroup(r, new_r)
        elif repl is not new_r:
            raise Exception("Improper use of EquivTool!")
        else:
            self.group(new_r, r)

    def publish(self):
        self.env.equiv = self

    def group(self, main, *keys):
        "Marks all the keys as having been replaced by the Result main."
        keys = [key for key in keys if key is not main]
        if self.has_key(main):
            raise Exception("Only group results that have not been grouped before.")
        for key in keys:
            if self.has_key(key):
                raise Exception("Only group results that have not been grouped before.")
            if key is main:
                continue
            self.setdefault(key, main)

    def ungroup(self, main, *keys):
        "Undoes group(main, *keys)"
        keys = [key for key in keys if key is not main]
        for key in keys:
            if self[key] is main:
                del self[key]

    def __call__(self, key):
        "Returns the currently active replacement for the given key."
        next = self.get(key, None)
        while next:
            key = next
            next = self.get(next, None)
        return key



class InstanceFinder(Listener, Tool, dict):

    def __init__(self, env):
        self.env = env

    def all_bases(self, cls):
        return utils.all_bases(cls, lambda cls: issubclass(cls, Op))
#        return [cls for cls in utils.all_bases(cls) if issubclass(cls, Op)]

    def on_import(self, op):
        for base in self.all_bases(op.__class__):
            self.setdefault(base, set()).add(op)

    def on_prune(self, op):
        for base in self.all_bases(op.__class__):
            self[base].remove(op)
            if not self[base]:
                del self[base]

    def __query__(self, cls):
        all = [x for x in self.get(cls, [])]
        shuffle(all) # this helps a lot for debugging because the order of the replacements will vary
        while all:
            next = all.pop()
            if next in self.env.ops():
                yield next

    def query(self, cls):
        return self.__query__(cls)

    def publish(self):
        self.env.get_instances_of = self.query



class PrintListener(Listener):

    def __init__(self, env, active = True):
        self.env = env
        self.active = active
        if active:
            print "-- initializing"

    def on_import(self, op):
        if self.active:
            print "-- importing: %s" % graph.as_string(self.env.inputs, op.outputs)

    def on_prune(self, op):
        if self.active:
            print "-- pruning: %s" % graph.as_string(self.env.inputs, op.outputs)

    def on_rewire(self, clients, r, new_r):
        if self.active:
            if r.owner is None:
                rg = id(r) #r.name
            else:
                rg = graph.as_string(self.env.inputs, r.owner.outputs)
            if new_r.owner is None:
                new_rg = id(new_r) #new_r.name
            else:
                new_rg = graph.as_string(self.env.inputs, new_r.owner.outputs)
            print "-- moving from %s to %s" % (rg, new_rg)



class ChangeListener(Listener):

    def __init__(self, env):
        self.change = False

    def on_import(self, op):
        self.change = True

    def on_prune(self, op):
        self.change = True

    def on_rewire(self, clients, r, new_r):
        self.change = True

    def __call__(self, value = "get"):
        if value == "get":
            return self.change
        else:
            self.change = value







# class SuperFinder(Listener, Tool, dict):

#     def __init__(self, env, props):
#         self.env = env
#         self.props = props

#     def on_import(self, op):
#         for prop, value in self.props(op).items():
#             self.setdefault(prop, {}).setdefault(value, set()).add(op)

#     def on_prune(self, op):
#         for prop, value in self.props(op).items():
#             self[prop][value].remove(op)
#             if len(self[prop][value]) == 0:
#                 del self[prop][value]
#                 if len(self[prop]) == 0:
#                     del self[prop]

#     def __query__(self, order, template):
#         all = []
#         for prop, value in template.items():
#             all += [x for x in self.get(prop, {}).get(value, set())]
#         # If not None, the order option requires the order listener to be included in the env under the name 'order'
#         if order == 'o->i':
#             all.sort(lambda op1, op2: self.env.order[op1].__cmp__(self.env.order[op2]))
#         elif order == 'i->o':
#             all.sort(lambda op1, op2: self.env.order[op2].__cmp__(self.env.order[op1]))
#         while all:
#             next = all.pop()
#             if next in self.env.ops():
#                 yield next

#     def query(self, **template):
#         return self.__query__(None, template)

#     def query_downstream(self, **template):
#         return self.__query__('i->o', template)

#     def query_upstream(self, **template):
#         return self.__query__('o->i', template)

#     def publish(self):
#         self.env.query = self.query






