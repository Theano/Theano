"""Convenient driver of graph construction, optimization, and linking."""

import gof

#TODO: put together some default optimizations (TRAC #67)

_optimizations = None

def exec_py_opt(inputs, outputs, features=[]):
    """Return an optimized graph running purely python implementations"""
    return Function(intputs, outputs, features, _optimizations, gof.link.PerformLinker, False)

def exec_opt(inputs, outputs, features=[]):
    """Return a fast implementation"""
    return Function(intputs, outputs, features, _optimizations, gof.link.PerformLinker, False)

def _mark_indestructible(results):
    for r in results:
        r.indestructible = True

class Function:
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
    def __init__(self, inputs, outputs,
            features = [],
            optimizer = None,
            linker_cls = gof.link.PerformLinker,
            unpack_single = True,
            except_unreachable_input = True,
            keep_locals = True):
        """ Copy the graph, optimize, and link it.
        Parameters:
        inputs - a list of results to be this function's inputs
        outputs - a list of results to be this function's outputs
        features - features to add to the env
        optimizer - an optimizer to apply to the copied graph, before linking
        linker_cls - a callable that takes an env and returns a Linker
        unpack_single - unpack return value lists of length 1
                      - see  Linker.make_function
        keep_locals - add the local variables from __init__ to the class
        """

        _mark_indestructible(outputs)

        if len(inputs) != len(set(inputs)):
            raise Exception('duplicate inputs')
        if len(outputs) != len(set(outputs)):
            raise Exception('duplicate outputs')

        #evaluate the orphans, and put these values into the clone of the env

        orphans = list(gof.graph.results_and_orphans(inputs, outputs,
            except_unreachable_input=except_unreachable_input)[1])
        orphan_data = eval_outputs(orphans, unpack_single=False)

        #print 'orphans', orphans

        #print 'ops', gof.graph.ops(inputs, outputs)
        env = gof.env.Env(inputs, outputs, features, consistency_check = True)

        #print 'orphans in env', env.orphans()

        env = env.clone(clone_inputs=True)

        #print 'orphans after clone', env.orphans()

        for d, o in zip(orphan_data, env.orphans()):
            #print 'assigning orphan value', d
            o.data = d

        # optimize and link the cloned env
        if None is not optimizer:
            optimizer.optimize(env)
        linker = linker_cls(env)

        if keep_locals:# useful flag for debugging!
            self.__dict__.update(locals())

        self.fn  = linker.make_function(inplace=True, 
                unpack_single=unpack_single)

    def __call__(self, *args):
        return self.fn(*args)

def eval_outputs(outputs,
        features = [],
        optimizer = None,
        linker_cls = gof.link.PerformLinker,
        unpack_single = True,
        keep_locals = True):

    if len(outputs) == 0:
        #print 'returning with no inputs'
        if unpack_single:
            return None
        else:
            return []

    inputs = list(gof.graph.inputs(outputs))
    in_data = [i.data for i in inputs if i.data is not None]
    #print 'in_data = ', in_data
    if len(inputs) != len(in_data):
        raise Exception('some input data is unknown')

    env = gof.env.Env(inputs, outputs, features, consistency_check = True)
    env = env.clone(clone_inputs=True)

    _mark_indestructible(env.outputs)
    if None is not optimizer:
        optimizer.optimize(env)
    linker = linker_cls(env)
    fn = linker.make_function(inplace=True, unpack_single=unpack_single)
    rval = fn(*in_data)
    return rval


