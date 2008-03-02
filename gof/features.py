
import utils


__all__ = ['Feature',
           'Listener',
           'Constraint',
           'Orderings',
           'Tool',
           'uniq_features',
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


