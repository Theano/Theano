import sys

from theano.gof.python25 import partial

import graph


class AlreadyThere(Exception):
    pass


class Bookkeeper:

    def on_attach(self, env):
        for node in graph.io_toposort(env.inputs, env.outputs):
            self.on_import(env, node)

    def on_detach(self, env):
        for node in graph.io_toposort(env.inputs, env.outputs):
            self.on_prune(env, node)


class History:

    def __init__(self):
        self.history = {}

    def on_attach(self, env):
        if hasattr(env, 'checkpoint') or hasattr(env, 'revert'):
            raise AlreadyThere("History feature is already present or in"
                               " conflict with another plugin.")
        self.history[env] = []
        env.checkpoint = lambda: len(self.history[env])
        env.revert = partial(self.revert, env)

    def on_detach(self, env):
        del env.checkpoint
        del env.revert
        del self.history[env]

    def on_change_input(self, env, node, i, r, new_r, reason=None):
        if self.history[env] is None:
            return
        h = self.history[env]
        h.append(lambda: env.change_input(node, i, r,
                                          reason=("Revert", reason)))

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
            raise AlreadyThere("Validator feature is already present or in"
                               " conflict with another plugin.")
        env.validate = lambda: env.execute_callbacks('validate')

        def consistent():
            try:
                env.validate()
                return True
            except:
                return False
        env.consistent = consistent

    def on_detach(self, env):
        del env.validate
        del env.consistent


class ReplaceValidate(History, Validator):

    def on_attach(self, env):
        History.on_attach(self, env)
        Validator.on_attach(self, env)
        for attr in ('replace_validate', 'replace_all_validate'):
            if hasattr(env, attr):
                raise AlreadyThere("ReplaceValidate feature is already present"
                                   " or in conflict with another plugin.")
        env.replace_validate = partial(self.replace_validate, env)
        env.replace_all_validate = partial(self.replace_all_validate, env)

    def on_detach(self, env):
        History.on_detach(self, env)
        Validator.on_detach(self, env)
        del env.replace_validate
        del env.replace_all_validate

    def replace_validate(self, env, r, new_r, reason=None):
        self.replace_all_validate(env, [(r, new_r)], reason=reason)

    def replace_all_validate(self, env, replacements, reason=None):
        chk = env.checkpoint()
        for r, new_r in replacements:
            try:
                env.replace(r, new_r, reason=reason)
            except Exception, e:
                if ('The type of the replacement must be the same' not in
                    str(e) and 'does not belong to this Env' not in str(e)):
                    out = sys.stderr
                    print >> out, "<<!! BUG IN ENV.REPLACE OR A LISTENER !!>>",
                    print >> out, type(e), e, reason
                # this might fail if the error is in a listener:
                # (env.replace kinda needs better internal error handling)
                env.revert(chk)
                raise
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
            raise AlreadyThere("NodeFinder is already present or in conflict"
                               " with another plugin.")
        self.env = env
        env.get_nodes = partial(self.query, env)
        Bookkeeper.on_attach(self, env)

    def on_detach(self, env):
        if self.env is not env:
            raise Exception("This NodeFinder instance was not attached to the"
                            " provided env.")
        self.env = None
        del env.get_nodes
        Bookkeeper.on_detach(self, env)

    def on_import(self, env, node):
        try:
            self.setdefault(node.op, []).append(node)
        except TypeError:  # node.op is unhashable
            return
        except Exception, e:
            print >> sys.stderr, 'OFFENDING node', type(node), type(node.op)
            try:
                print >> sys.stderr, 'OFFENDING node hash', hash(node.op)
            except:
                print >> sys.stderr, 'OFFENDING node not hashable'
            raise e

    def on_prune(self, env, node):
        try:
            nodes = self[node.op]
        except TypeError:  # node.op is unhashable
            return
        nodes.remove(node)
        if not nodes:
            del self[node.op]

    def query(self, env, op):
        try:
            all = self.get(op, [])
        except TypeError:
            raise TypeError("%s in unhashable and cannot be queried by the"
                            " optimizer" % op)
        all = list(all)
        return all


class PrintListener(object):

    def __init__(self, active=True):
        self.active = active

    def on_attach(self, env):
        if self.active:
            print "-- attaching to: ", env

    def on_detach(self, env):
        if self.active:
            print "-- detaching from: ", env

    def on_import(self, env, node):
        if self.active:
            print "-- importing: %s" % node

    def on_prune(self, env, node):
        if self.active:
            print "-- pruning: %s" % node

    def on_change_input(self, env, node, i, r, new_r, reason=None):
        if self.active:
            print "-- changing (%s.inputs[%s]) from %s to %s" % (
                node, i, r, new_r)


class PreserveNames:
    def on_change_input(self, env, mode, i, r, new_r, reason=None):
        if r.name is not None and new_r.name is None:
            new_r.name = r.name
