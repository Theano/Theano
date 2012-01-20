"""Define RModule, a Module providing random number streams in Theano graphs."""
__docformat__ = "restructuredtext en"
import sys
if sys.version_info[:2] >= (2,5):
  from functools import partial
else:
  from theano.gof.python25 import partial

import numpy
from copy import copy

from theano.compile import (SymbolicInputKit, SymbolicInput, 
        Module, module, Method, Member, In, Component)
from theano.gof import Container
from theano.gof.python25 import deque

from theano.tensor import raw_random

class KitComponent(Component):
    """
    Represents a SymbolicInputKit (see io.py).
    """
    
    def __init__(self, kit):
        super(KitComponent, self).__init__()
        self.kit = kit

    def allocate(self, memo):
        """
        Allocates a Container for each input in the kit. Sets a key in
        the memo that maps the SymbolicInputKit to the list of
        Containers.
        """
        for input in self.kit.sinputs:
            r = input.variable
            if r not in memo:
                input = copy(input)
                input.value = Container(r, storage = [None])
                memo[r] = input

    def build(self, mode, memo):
        return [memo[i.variable].value for i in self.kit.sinputs]


class RandomKit(SymbolicInputKit):

    def __init__(self, name, value = None):
        super(RandomKit, self).__init__(name)
        self.value = value

    def gen(self, op, *args, **kwargs):
        random_state_variable = raw_random.random_state_type()
        new_r, out = op(random_state_variable, *args, **kwargs)
        self.add_input(SymbolicInput(random_state_variable, update = new_r))
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

