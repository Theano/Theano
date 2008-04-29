
from random import shuffle
import utils


class EquivTool(dict):

    def __init__(self, env):
        self.env = env

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
        self.env.set_equiv = self.set_equiv

    def unpublish(self):
        del self.env.equiv
        del self.env.set_equiv

    def set_equiv(self, d):
        self.update(d)

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


class NodeFinder(dict):

    def __init__(self, env):
        self.env = env

    def on_import(self, node):
        try:
            self.setdefault(node.op, set()).add(node)
        except TypeError:
            pass

    def on_prune(self, node):
        try:
            self[node.op].remove(node)
        except TypeError:
            return
        if not self[node.op]:
            del self[node.op]

    def query(self, op):
        try:
            all = self.get(op, [])
        except TypeError:
            raise TypeError("%s in unhashable and cannot be queried by the optimizer" % op)
        all = [x for x in all]
        shuffle(all) # this helps a lot for debugging because the order of the replacements will vary
        while all:
            next = all.pop()
            if self.env.has_node(next):
                yield next

    def publish(self):
        self.env.get_nodes = self.query

    def __eq__(self, other):
        return isinstance(other, NodeFinder) and self.env is other.env


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



class PrintListener(object):

    def __init__(self, env, active = True):
        self.env = env
        self.active = active
        if active:
            print "-- initializing"

    def on_import(self, node):
        if self.active:
            print "-- importing: %s" % node

    def on_prune(self, node):
        if self.active:
            print "-- pruning: %s" % node

    def on_rewire(self, clients, r, new_r):
        if self.active:
            if r.owner is not None: r = r.owner
            if new_r.owner is not None: new_r = new_r.owner
            print "-- moving from %s to %s" % (r, new_r)


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


