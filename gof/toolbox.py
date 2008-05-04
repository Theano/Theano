
from random import shuffle
import utils
from functools import partial
import graph


class Bookkeeper:
    
    def on_attach(self, env):
        for node in graph.io_toposort(env.inputs, env.outputs):
            self.on_import(env, node)

    def on_deattach(self, env):
        for node in graph.io_toposort(env.inputs, env.outputs):
            self.on_prune(env, node)


class History:

    def __init__(self):
        self.history = {}

    def on_attach(self, env):
        if hasattr(env, 'checkpoint') or hasattr(env, 'revert'):
            raise Exception("History feature is already present or in conflict with another plugin.")
        self.history[env] = []
        env.checkpoint = lambda: len(self.history[env])
        env.revert = partial(self.revert, env)

    def on_deattach(self, env):
        del env.checkpoint
        del env.revert
        del self.history[env]

    def on_change_input(self, env, node, i, r, new_r):
        if self.history[env] is None:
            return
        h = self.history[env]
        h.append(lambda: env.change_input(node, i, r))
    
    def revert(self, env, checkpoint):
        """
        Reverts the graph to whatever it was at the provided
        checkpoint (undoes all replacements).  A checkpoint at any
        given time can be obtained using self.checkpoint().
        """
        h = self.history[env]
        self.history[env] = None
        while len(h) > checkpoint:
            f = h.pop()
            f()
        self.history[env] = h


class Validator:

    def on_attach(self, env):
        if hasattr(env, 'validate'):
            raise Exception("Validator feature is already present or in conflict with another plugin.")
        env.validate = lambda: env.execute_callbacks('validate')
        def consistent():
            try:
                env.validate()
                return True
            except:
                return False
        env.consistent = consistent

    def on_deattach(self, env):
        del env.validate
        del env.consistent


class ReplaceValidate(History, Validator):

    def on_attach(self, env):
        History.on_attach(self, env)
        Validator.on_attach(self, env)
        for attr in ('replace_validate', 'replace_all_validate'):            
            if hasattr(env, attr):
                raise Exception("ReplaceValidate feature is already present or in conflict with another plugin.")
        env.replace_validate = partial(self.replace_validate, env)
        env.replace_all_validate = partial(self.replace_all_validate, env)

    def on_deattach(self, env):
        History.on_deattach(self, env)
        Validator.on_deattach(self, env)
        del env.replace_validate
        del env.replace_all_validate

    def replace_validate(self, env, r, new_r):
        self.replace_all_validate(env, [(r, new_r)])

    def replace_all_validate(self, env, replacements):
        chk = env.checkpoint()
        for r, new_r in replacements:
            env.replace(r, new_r)
        try:
            env.validate()
        except:
            env.revert(chk)
            raise


class NodeFinder(dict, Bookkeeper):

    def __init__(self):
        self.env = None
    
    def on_attach(self, env):
        if self.env is not None:
            raise Exception("A NodeFinder instance can only serve one Env.")
        if hasattr(env, 'get_nodes'):
            raise Exception("NodeFinder is already present or in conflict with another plugin.")
        self.env = env
        env.get_nodes = partial(self.query, env)
        Bookkeeper.on_attach(self, env)

    def on_deattach(self, env):
        if self.env is not env:
            raise Exception("This NodeFinder instance was not attached to the provided env.")
        self.env = None
        del env.get_nodes
        Bookkeeper.on_deattach(self, env)

    def on_import(self, env, node):
        try:
            self.setdefault(node.op, []).append(node)
        except TypeError: #node.op is unhashable
            return

    def on_prune(self, env, node):
        try:
            nodes = self[node.op]
        except TypeError: #node.op is unhashable
            return
        nodes.remove(node)
        if not nodes:
            del self[node.op]

    def query(self, env, op):
        try:
            all = self.get(op, [])
        except TypeError:
            raise TypeError("%s in unhashable and cannot be queried by the optimizer" % op)
        all = list(all)
        while all:
            next = all.pop()
            if next in env.nodes:
                yield next


class PrintListener(object):

    def __init__(self, active = True):
        self.active = active
    
    def on_attach(self, env):
        if self.active:
            print "-- attaching to: ", env

    def on_deattach(self, env):
        if self.active:
            print "-- deattaching from: ", env

    def on_import(self, env, node):
        if self.active:
            print "-- importing: %s" % node

    def on_prune(self, env, node):
        if self.active:
            print "-- pruning: %s" % node

    def on_change_input(self, env, node, i, r, new_r):
        if self.active:
            print "-- changing (%s.inputs[%s]) from %s to %s" % (node, i, r, new_r)









# class EquivTool(dict):

#     def __init__(self, env):
#         self.env = env

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
#         self.env.set_equiv = self.set_equiv

#     def unpublish(self):
#         del self.env.equiv
#         del self.env.set_equiv

#     def set_equiv(self, d):
#         self.update(d)

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
#         return utils.all_bases(cls, lambda cls: cls is not object)

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



# class DescFinder(Listener, Tool, dict):

#     def __init__(self, env):
#         self.env = env

#     def on_import(self, op):
#         self.setdefault(op.desc(), set()).add(op)

#     def on_prune(self, op):
#         desc = op.desc()
#         self[desc].remove(op)
#         if not self[desc]:
#             del self[desc]

#     def __query__(self, desc):
#         all = [x for x in self.get(desc, [])]
#         shuffle(all) # this helps for debugging because the order of the replacements will vary
#         while all:
#             next = all.pop()
#             if next in self.env.ops():
#                 yield next

#     def query(self, desc):
#         return self.__query__(desc)

#     def publish(self):
#         self.env.get_from_desc = self.query




### UNUSED AND UNTESTED ###

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


