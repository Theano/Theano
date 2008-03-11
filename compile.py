"""Convenient driver of the graph construction, optimization, and linking phases"""

import gof
from copy import copy


class Prog:
    def __init__(self, 
            inputs,
            outputs, 
            features=[],
            optimizer=None, #TODO: put together some default optimizations
            linker_cls=gof.link.PerformLinker,
            keep_locals=False):

        env = gof.env.Env(inputs, outputs, features, consistency_check = True)

        if None is not optimizer:
            optimizer.optimize(env)

        linker = linker_cls(env)

        if keep_locals: # useful flag for debugging
            self.__dict__.update(locals())
        self.fn  = linker.make_function(False)

    def __call__(self, *args):
        return self.fn(*args)


