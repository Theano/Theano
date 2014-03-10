"""Define RandomStreams, providing random number variables for Theano
graphs.

"""
__docformat__ = "restructuredtext en"

import numpy

from theano.compile import In, Component
from theano.gof import Container
from theano.tensor import raw_random
import warnings


def deprecation_warning():
    # Make sure the warning is displayed only once.
    if deprecation_warning.already_displayed:
        return

    warnings.warn((
        "RandomStreams is deprecated and will be removed in release 0.7. "
        "Use shared_randomstreams.RandomStreams or "
        "MRG_RandomStreams instead."),
        stacklevel=3)
    deprecation_warning.already_displayed = True

deprecation_warning.already_displayed = False


class RandomStreamsInstance(object):
    """RandomStreamsInstance"""
    def __init__(self, random_streams, memo, default_seed):
        self.random_streams = random_streams
        self.memo = memo
        self.default_seed = default_seed

    def initialize(self, seed=None):
        """Initialize each random stream

        :param seed: each random stream will be assigned a unique
        state that depends deterministically on this value.

        :type seed: None or integer in range 0 to 2**30

        :rtype: None

        """
        self.seed(seed)

    def seed(self, seed=None):
        """Re-initialize each random stream

        :param seed: each random stream will be assigned a unique
        state that depends deterministically on this value.

        :type seed: None or integer in range 0 to 2**30

        :rtype: None

        """
        if seed is None:
            seed = self.default_seed
        #backport
        #seed = self.default_seed if seed is None else seed
        seedgen = numpy.random.RandomState(seed)
        for old_r, new_r in self.random_streams.random_state_variables:
            old_r_seed = seedgen.randint(2 ** 30)
            old_r_container = self.memo[old_r].value
            if old_r_container.value is None:
                #the cast to int here makes it work on 32bit machines,
                #not sure why
                old_r_container.value = numpy.random.RandomState(
                    int(old_r_seed))
            else:
                #the cast to int here makes it work on 32bit machines,
                #not sure why
                old_r_container.value.seed(int(old_r_seed))

    def __getitem__(self, item):
        """Retrieve the numpy RandomState instance associated with a
        particular stream

        :param item: a variable of type RandomStateType, associated
        with this RandomStream

        :rtype: numpy RandomState (or None, before initialize)

        """
        for old_r, new_r in self.random_streams.random_state_variables:
            if item is old_r:
                container = self.memo[item].value
                return container.value
        raise KeyError(item)

    def __setitem__(self, item, val):
        """Set the numpy RandomState instance associated with a
        particular stream

        :param item: a variable of type RandomStateType, associated
        with this RandomStream

        :param val: the new value
        :type val: numpy RandomState

        :rtype:  None

        """
        if type(val) is not numpy.random.RandomState:
            raise TypeError('only values of type RandomState are permitted',
                            val)
        for old_r, new_r in self.random_streams.random_state_variables:
            if item is old_r:
                container = self.memo[item].value
                container.value = val
                return
        raise KeyError(item)


class RandomStreams(Component, raw_random.RandomStreamsBase):
    """Module component with similar interface to numpy.random
    (numpy.random.RandomState)

    """

    def __init__(self, seed=None, no_warn=False):
        """:type seed: None or int

        :param seed: a default seed to initialize the RandomState
        instances after build.  See `RandomStreamsInstance.__init__`
        for more details.

        """
        if not no_warn:
            deprecation_warning()
        super(RandomStreams, self).__init__(no_warn=True)

        # A list of pairs of the form (input_r, output_r).  This will be
        # over-ridden by the module instance to contain stream generators.
        self.random_state_variables = []

        # Instance variable should take None or integer value.  Used to seed the
        # random number generator that provides seeds for member streams
        self.default_instance_seed = seed

    def allocate(self, memo):
        """override `Component.allocate` """
        for old_r, new_r in self.random_state_variables:
            if old_r in memo:
                assert memo[old_r].update is new_r
            else:
                memo[old_r] = In(old_r,
                        value=Container(old_r, storage=[None]),
                        update=new_r,
                        mutable=True)

    def build(self, mode, memo):
        """override `Component.build` """
        if self not in memo:
            memo[self] = RandomStreamsInstance(self, memo,
                                               self.default_instance_seed)
        return memo[self]

    def gen(self, op, *args, **kwargs):
        """Create a new random stream in this container.

        :param op: a RandomFunction instance to

        :param args: interpreted by `op`

        :param kwargs: interpreted by `op`

        :returns: The symbolic random draw part of op()'s return
        value.  This function stores the updated RandomStateType
        Variable for use at `build` time.

        :rtype: TensorVariable

        """
        random_state_variable = raw_random.random_state_type()
        new_r, out = op(random_state_variable, *args, **kwargs)
        out.rng = random_state_variable
        self.random_state_variables.append((random_state_variable, new_r))
        return out
