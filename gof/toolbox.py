
from features import Listener, Tool
from random import shuffle
import utils

__all__ = ['EquivTool',
          'InstanceFinder',
          'PrintListener',
          'ChangeListener',
           ]


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
        self.env.set_equiv = self.set_equiv

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


class InstanceFinder(Listener, Tool, dict):

    def __init__(self, env):
        self.env = env

    def all_bases(self, cls):
        return utils.all_bases(cls, lambda cls: cls is not object)

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
            print "-- importing: %s" % op

    def on_prune(self, op):
        if self.active:
            print "-- pruning: %s" % op

    def on_rewire(self, clients, r, new_r):
        if self.active:
            if r.owner is not None: r = r.owner
            if new_r.owner is not None: new_r = new_r.owner
            print "-- moving from %s to %s" % (r, new_r)



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


