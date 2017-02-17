"""
Define RandomStreams, providing random number variables for Theano
graphs.

"""
from __future__ import absolute_import, print_function, division

import copy

import numpy

from theano.compile.sharedvalue import (SharedVariable, shared_constructor,
                                        shared)
from theano.tensor import raw_random

__docformat__ = "restructuredtext en"


class RandomStateSharedVariable(SharedVariable):
    pass


@shared_constructor
def randomstate_constructor(value, name=None, strict=False,
                            allow_downcast=None, borrow=False):
    """
    SharedVariable Constructor for RandomState.

    """
    if not isinstance(value, numpy.random.RandomState):
        raise TypeError
    if not borrow:
        value = copy.deepcopy(value)
    return RandomStateSharedVariable(
        type=raw_random.random_state_type,
        value=value,
        name=name,
        strict=strict,
        allow_downcast=allow_downcast)


class RandomStreams(raw_random.RandomStreamsBase):
    """
    Module component with similar interface to numpy.random
    (numpy.random.RandomState)

    Parameters
    ----------
    seed: None or int
        A default seed to initialize the RandomState
        instances after build.  See `RandomStreamsInstance.__init__`
        for more details.

    """

    def updates(self):
        return list(self.state_updates)

    def __init__(self, seed=None):
        super(RandomStreams, self).__init__()
        # A list of pairs of the form (input_r, output_r).  This will be
        # over-ridden by the module instance to contain stream generators.
        self.state_updates = []
        # Instance variable should take None or integer value. Used to seed the
        # random number generator that provides seeds for member streams.
        self.default_instance_seed = seed
        # numpy.RandomState instance that gen() uses to seed new streams.
        self.gen_seedgen = numpy.random.RandomState(seed)

    def seed(self, seed=None):
        """
        Re-initialize each random stream.

        Parameters
        ----------
        seed : None or integer in range 0 to 2**30
            Each random stream will be assigned a unique state that depends
            deterministically on this value.

        Returns
        -------
        None

        """
        if seed is None:
            seed = self.default_instance_seed

        seedgen = numpy.random.RandomState(seed)
        for old_r, new_r in self.state_updates:
            old_r_seed = seedgen.randint(2 ** 30)
            old_r.set_value(numpy.random.RandomState(int(old_r_seed)),
                            borrow=True)

    def __getitem__(self, item):
        """
        Retrieve the numpy RandomState instance associated with a particular
        stream.

        Parameters
        ----------
        item
            A variable of type RandomStateType, associated
            with this RandomStream.

        Returns
        -------
        numpy RandomState (or None, before initialize)

        Notes
        -----
        This is kept for compatibility with `tensor.randomstreams.RandomStreams`.
        The simpler syntax ``item.rng.get_value()`` is also valid.

        """
        return item.get_value(borrow=True)

    def __setitem__(self, item, val):
        """
        Set the numpy RandomState instance associated with a particular stream.

        Parameters
        ----------
        item
            A variable of type RandomStateType, associated with this
            RandomStream.

        val : numpy RandomState
            The new value.

        Returns
        -------
        None

        Notes
        -----
        This is kept for compatibility with `tensor.randomstreams.RandomStreams`.
        The simpler syntax ``item.rng.set_value(val)`` is also valid.

        """
        item.set_value(val, borrow=True)

    def gen(self, op, *args, **kwargs):
        """
        Create a new random stream in this container.

        Parameters
        ----------
        op
            A RandomFunction instance to
        args
            Interpreted by `op`.
        kwargs
            Interpreted by `op`.

        Returns
        -------
        Tensor Variable
            The symbolic random draw part of op()'s return value.
            This function stores the updated RandomStateType Variable
            for use at `build` time.

        """
        seed = int(self.gen_seedgen.randint(2 ** 30))
        random_state_variable = shared(numpy.random.RandomState(seed))
        # Add a reference to distinguish from other shared variables
        random_state_variable.tag.is_rng = True
        new_r, out = op(random_state_variable, *args, **kwargs)
        out.rng = random_state_variable
        out.update = (random_state_variable, new_r)
        self.state_updates.append(out.update)
        random_state_variable.default_update = new_r
        return out
