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


class DeepCopyableFunction(Singleton):

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
        class F(DeepCopyableFunction):
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
        # of this class.
        assert self is other
        return True

    def __hash__(self):
        # Required for Ops that contain such functions.
        return hash(type(self))

    def __getstate__(self):
        # Only implemented to ensure __setstate__ will always be called
        # after unpickling such an object, even when __dict__ is empty.
        if self.__dict__:
            return self.__dict__
        else:
            # We need to make sure we return a value whose boolean cast is not
            # False, otherwise __setstate__ will not be called.
            return True

    def __setstate__(self, state):
        # Only implemented to enforce the "singletonness" of this class.
        if state is not True:
            self.__dict__.update(state)
        if self is not type(self)._instance:
            raise AssertionError(
                    "Unpickling a singleton should yield this singleton. If "
                    "this is not the case, you may have pickled it with "
                    "protocol 0. You will need to use a higher protocol to "
                    "properly unpickle this object.")
