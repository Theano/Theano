"""Define RModule, a Module providing random number streams in Theano graphs."""
__docformat__ = "restructuredtext en"
import sys
import functools
from functools import partial
from collections import deque

import numpy

from ..compile import (SymbolicInputKit, SymbolicInput, 
        Module, KitComponent, module, Method, Member, In, Component)
from ..gof import Container

from ..tensor import raw_random

class RandomKit(SymbolicInputKit):

    def __init__(self, name, value = None):
        super(RandomKit, self).__init__(name)
        self.value = value

    def gen(self, op, *args, **kwargs):
        random_state_result = raw_random.random_state_type()
        new_r, out = op(random_state_result, *args, **kwargs)
        self.add_input(SymbolicInput(random_state_result, update = new_r))
        out.rng = new_r
        out.auto = self
        return out

    def distribute(self, value, indices, containers):
        rg = partial(numpy.random.RandomState(int(value)).randint, 2**30)
        elems = deque(zip(indices, containers))
        i = 0
        while elems:
            index, container = elems.popleft()
            while i <= index:
                curr = rg()
                i += 1
            rs = numpy.random.RandomState(int(curr))
            container.data = rs

    def binomial(self, *args, **kwargs):
        return self.gen(raw_random.binomial, *args, **kwargs)

    def uniform(self, *args, **kwargs):
        return self.gen(raw_random.uniform, *args, **kwargs)

    def normal(self, *args, **kwargs):
        return self.gen(raw_random.normal, *args, **kwargs)

    def random_integers(self, *args, **kwargs):
        return self.gen(raw_random.random_integers, *args, **kwargs)



rk = RandomKit('rk', 0xBAD5EED)


class RModule(Module):
    """Module providing random number streams in Theano graphs."""

    def __init__(self, components = {}, **kwcomponents):
        super(RModule, self).__init__(components, **kwcomponents)
        self.random = RandomKit('rkit')
        self._rkit = KitComponent(self.random)

    def __wrapper__(self, x):
        x = module.wrap(x)
        if isinstance(x, Method):
            x.kits += [self.random]
        return x

    def _instance_seed(self, inst, seed, recursive = True):
        seedgen = numpy.random.RandomState(seed)
        if recursive:
            #Here, we recurse through all the components (inst2) contained in (inst)
            #and seeds each subcomponent that is an RModule
            
            
            for path, c in self.flat_components_map(True):
                if isinstance(c, RModule):
                    inst2 = inst
                    for name in path:
                        inst2 = inst2[name]
                    # A Kit (c._rkit.kit) contains a list of io.SymbolicIn instances
                    # and the distribute method takes a value (seed), a list of indices
                    # and a list of corresponding Container instances. In this
                    # situation it will reseed all the rngs using the containers
                    # associated to them.
                    c._rkit.kit.distribute(seedgen.random_integers(2**30),
                                           xrange(len(inst2._rkit)), inst2._rkit)
        else:
            self._rkit.kit.distribute(seedgen.random_integers(2**30), xrange(len(inst._rkit)), inst._rkit)


class RandomStreamsInstance(object):
    """RandomStreamsInstance"""
    def __init__(self, random_streams, memo, default_seed):
        self.random_streams = random_streams
        self.memo = memo
        self.default_seed = default_seed

    def initialize(self, seed=None):
        """Initialize each random stream

        :param seed: each random stream will be assigned a unique state that depends
        deterministically on this value.

        :type seed: None or integer in range 0 to 2**30

        :rtype: None
        """
        self.seed(seed)

    def seed(self, seed=None):
        """Re-initialize each random stream
        
        :param seed: each random stream will be assigned a unique state that depends
        deterministically on this value.

        :type seed: None or integer in range 0 to 2**30

        :rtype: None
        """
        seed = self.default_seed if seed is None else seed
        seedgen = numpy.random.RandomState(seed)
        for old_r, new_r in self.random_streams.random_state_results:
            old_r_seed = seedgen.randint(2**30)
            old_r_container = self.memo[old_r].value
            if old_r_container.value is None:
                #the cast to int here makes it work on 32bit machines, not sure why
                old_r_container.value = numpy.random.RandomState(int(old_r_seed))
            else:
                #the cast to int here makes it work on 32bit machines, not sure why
                old_r_container.value.seed(int(old_r_seed))

    def __getitem__(self, item):
        """Retrieve the numpy RandomState instance associated with a particular stream

        :param item: a result of type RandomStateType, associated with this RandomStream

        :rtype: numpy RandomState (or None, before initialize)

        """
        for old_r, new_r in self.random_streams.random_state_results:
            if item is old_r:
                container = self.memo[item].value
                return container.value
        raise KeyError(item)

    def __setitem__(self, item, val):
        """Set the numpy RandomState instance associated with a particular stream

        :param item: a result of type RandomStateType, associated with this RandomStream

        :param val: the new value
        :type val: numpy RandomState

        :rtype:  None

        """
        if type(val) is not numpy.random.RandomState:
            raise TypeError('only values of type RandomState are permitted', val)
        for old_r, new_r in self.random_streams.random_state_results:
            if item is old_r:
                container = self.memo[item].value
                container.value = val
                return
        raise KeyError(item)



class RandomStreams(Component):
    """Module with similar interface to numpy.random (numpy.random.RandomState)"""

    random_state_results = []
    """A list of pairs of the form (input_r, output_r).  This will be over-ridden by the module
    instance to contain stream generators.
    """

    default_instance_seed = None
    """Instance variable should take None or integer value.  Used to seed the random number
    generator that provides seeds for member streams"""

    def __init__(self, seed=None):
        super(RandomStreams, self).__init__()
        self.random_state_results = []
        self.default_instance_seed = seed

    def allocate(self, memo):
        for old_r, new_r in self.random_state_results:
            assert old_r not in memo
            memo[old_r] = In(old_r, 
                    value=Container(old_r, storage=[None]),
                    update=new_r,
                    mutable=True)

    def build(self, mode, memo):
        #print 'MODE', mode
        #returns a list of containers
        return RandomStreamsInstance(self, memo, self.default_instance_seed)

    def gen(self, op, *args, **kwargs):
        random_state_result = raw_random.random_state_type()
        new_r, out = op(random_state_result, *args, **kwargs)
        out.rng = random_state_result
        self.random_state_results.append((random_state_result, new_r))
        return out

    def binomial(self, *args, **kwargs):
        """Return a symbolic binomial sample

        """
        return self.gen(raw_random.binomial, *args, **kwargs)

    def uniform(self, *args, **kwargs):
        return self.gen(raw_random.uniform, *args, **kwargs)

    def normal(self, *args, **kwargs):
        return self.gen(raw_random.normal, *args, **kwargs)

    def random_integers(self, *args, **kwargs):
        return self.gen(raw_random.random_integers, *args, **kwargs)

