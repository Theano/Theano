__authors__   = "Olivier Delalleau"
__contact__   = "delallea@iro"


"""
Utility classes for singletons.
"""

class Singleton(object):

    """Base class for Singletons."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(
                                cls, *args, **kwargs)
        return cls._instance


class DeepCopiableFunction(Singleton):

    """
    Utility class to work around Python 2.4 limitations.

    The problem is that trying to deep copy a function in Python 2.4 raises
    an exception. In Python 2.5+, deepcopying a function simply returns the
    same function (there is no actual copy being performed).

    Instances of subclasses are like "deepcopy-able functions": they are
    callable singleton objects.

    For instance if you plan to deepcopy a function f, instead of:
        def f():
            return 0
    write instead:
        class F(DeepCopiableFunction):
            def __call__(self):
                return 0
        f = F()
    """

    def __deepcopy__(self, memo):
        # Simply return a pointer to self (same as in Python 2.5+).
        return self

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        # Since it is a singleton there should be no two different instances
        # of this class. However it looks like this can actually happen under
        # Linux when pickling with protocol 0 => need to look into this. Until
        # then, the assert below is commented and it might happen that two
        # instances actually exist (which should not be an issue unless someone
        # does a comparison with the 'is' operator).
        #assert self is other
        return True

    def __hash__(self):
        # Required for Ops that contain such functions.
        return hash(type(self))

