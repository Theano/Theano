
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
        Initializes the L{Feature}'s env field to the parameter
        provided.
        """
        self.env = env


class Listener(Feature):
    """
    When registered by an L{Env}, each listener is informed of any L{Op}
    entering or leaving the subgraph (which happens at construction
    time and whenever there is a replacement).
    """

    def on_import(self, op):
        """
        This method is called by the L{Env} whenever a new L{Op} is
        added to the graph.
        """
        raise utils.AbstractFunctionError()
    
    def on_prune(self, op):
        """
        This method is called by the L{Env} whenever an L{Op} is
        removed from the graph.
        """
        raise utils.AbstractFunctionError()

    def on_rewire(self, clients, r, new_r):
        """
        @param clients: (op, i) pairs such that op.inputs[i] is new_r but used to be r
        @param r: the old result that was used by the L{Op}s in clients
        @param new_r: the new result that is now used by the L{Op}s in clients

        Note that the change from r to new_r is done before this
        method is called.
        """
        raise utils.AbstractFunctionError()


class Constraint(Feature):
    """
    When registered by an L{Env}, a L{Constraint} can restrict the L{Op}s that
    can be in the subgraph or restrict the ways L{Op}s interact with each
    other.
    """

    def validate(self):
        """
        Raises an L{InconsistencyError} if the L{Env} is currently
        invalid from the perspective of this object.
        """
        raise utils.AbstractFunctionError()


class Orderings(Feature):
    """
    When registered by an L{Env}, an L{Orderings} object can provide supplemental
    ordering constraints to the subgraph's topological sort.
    """

    def orderings(self):
        """
        Returns {op: set(ops that must be evaluated before this op), ...}
        This is called by L{Env.orderings}() and used in L{Env.toposort}() but
        not in L{Env.io_toposort}().
        """
        raise utils.AbstractFunctionError()


class Tool(Feature):
    """
    A L{Tool} can extend the functionality of an L{Env} so that, for example,
    optimizations can have access to efficient ways to search the graph.
    """

    def publish(self):
        """
        This is only called once by the L{Env}, when the L{Tool} is added.
        Adds methods to L{Env}.
        """
        raise utils.AbstractFunctionError()



def uniq_features(_features, *_rest):
    """Return a list such that no element is a subclass of another"""
    # used in Env.__init__
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


