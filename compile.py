"""Convenient driver of graph construction, optimization, and linking."""

import gof

#TODO: put together some default optimizations

_optimizations = None

def prog_py_opt(inputs, outputs, features=[]):
    """Return an optimized graph running purely python implementations"""
    return Prog(intputs, outputs, features, _optimizations, gof.link.PerformLinker, False)

def prog_opt(inputs, outputs, features=[]):
    """Return a fast implementation"""
    return Prog(intputs, outputs, features, _optimizations, gof.link.PerformLinker, False)

class Prog:
    """An 'executable' compiled from a graph

    This class is meant to be used as a function: the idea is to use
    __call__(*args) and it will compute your graph's function on the args and
    return the value(s) corresponding to the output(s).
    
    Attributes
    fn - the return value of linker.make_function(False)

    Additional Attributes if keep_locals == True
    inputs - inputs in the env
    outputs - outputs in the env
    features - features to add to the env
    linker_cls - the linker class
    linker - the linker allocated from env
    env - The env passed to the linker
    """
    def __init__(self, 
            inputs,
            outputs, 
            features=[],
            optimizer=None,
            linker_cls=gof.link.PerformLinker,
            keep_locals=True):

        env = gof.env.Env(inputs, outputs, features, consistency_check = True)

        if None is not optimizer:
            optimizer.optimize(env)

        linker = linker_cls(env)

        if keep_locals: # useful flag for debugging
            self.__dict__.update(locals())

        self.fn  = linker.make_function(False)

    def __call__(self, *args):
        return self.fn(*args)


