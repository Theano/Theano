"""Convenient driver of graph construction, optimization, and linking."""

import numpy
import gof

#TODO: put together some default optimizations (TRAC #67)

def exec_py_opt(inputs, outputs, features=[]):
    """Return an optimized graph running purely python implementations"""
    return Function(intputs, outputs, features, exec_py_opt.optimizer, gof.link.PerformLinker, False)
exec_py_opt.optimizer = None

def exec_opt(inputs, outputs, features=[]):
    """Return a fast implementation"""
    return Function(intputs, outputs, features, exec_opt.optimizer, gof.link.PerformLinker, False)
exec_opt.optimizer = None

class _DefaultOptimizer(object):
    const = gof.opt.ConstantFinder()
    merge = gof.opt.MergeOptimizer()
    def __call__(self, env):
        self.const(env)
        self.merge(env)
default_optimizer = _DefaultOptimizer()
        
def _mark_indestructible(results):
    for r in results:
        r.indestructible = True

def linker_cls_python_and_c(env):
    """Use this as the linker_cls argument to Function.__init__ to compare
    python and C implementations"""
    def checker(x, y):
        x, y = x.data, y.data
        if isinstance(x, numpy.ndarray) or isinstance(y, numpy.ndarray):
            if x.dtype != y.dtype or x.shape != y.shape or numpy.any(abs(x - y) > 1e-10):
                raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})
        else:
            if x != y:
                raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})
    return gof.DualLinker(env, checker)


class Function:
    """
    An 'executable' compiled from a graph

    This class is meant to be used as a function: the idea is to use
    __call__(*args) and it will compute your graph's function on the args and
    return the value(s) corresponding to the output(s).
    
    @ivar fn: the return value of L{linker.make_function}(False)

    Additional Attributes if keep_locals == True
    inputs - inputs in the env
    outputs - outputs in the env
    features - features to add to the env
    linker_cls - the linker class
    linker - the linker allocated from env
    env - The env passed to the linker

    @note: B{Re: Memory ownership, aliasing, re-use:}
    That the objects returned by L{Function.__call__}(self, *args) are owned
    by self, and that in general these outputs might be overwritten (in-place)
    by subsequent calls to L{self.__call__}(*args).  Why?  This behaviour is
    necessary for inplace operations to work, and L{Function}'s linker might re-use
    memory from one execution to the next in order to make each execution faster.

    """
    def __init__(self, inputs, outputs,
            features = [],
            optimizer = default_optimizer,
            linker_cls = gof.link.PerformLinker,
            profiler = None,
            unpack_single = True,
            except_unreachable_input = True,
            keep_locals = True):
        """
        Copy the graph, optimize, and link it.

        @param inputs: a list of results to be this function's inputs
        @param outputs: a list of results to be this function's outputs
        @param features: features to add to the env
        @param optimizer: an optimizer to apply to the copied graph, before linking
        @param linker_cls: a callable that takes an env and returns a Linker
        @param profiler: a L{Profiler} for the produced function (only valid if the
                   linker_cls's make_function takes a profiler argument)
        @param unpack_single: unpack return value lists of length 1. @see: L{Linker.make_function}
        @param keep_locals: add the local variables from __init__ to the class
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
        env = gof.env.Env(inputs, outputs, features + [gof.EquivTool], consistency_check = True)

        #print 'orphans in env', env.orphans()

        env = env.clone(clone_inputs=True)

        #print 'orphans after clone', env.orphans()

        for d, o in zip(orphan_data, [env.equiv(orphan) for orphan in orphans]):
            #print 'assigning orphan value', d
            o.data = d

        # optimize and link the cloned env
        if None is not optimizer:
            optimizer(env)

        linker = linker_cls(env)

        if keep_locals:# useful flag for debugging!
            self.__dict__.update(locals())

        if profiler is None:
            self.fn  = linker.make_function(inplace=True,
                                            unpack_single=unpack_single)
        else:
            self.fn  = linker.make_function(inplace=True,
                                            unpack_single=unpack_single,
                                            profiler=profiler)
        self.inputs = env.inputs
        self.outputs = env.outputs
        self.features = features
        self.optimizer = optimizer
        self.linker_cls = linker_cls
        self.profiler = profiler
        self.unpack_single = unpack_single
        self.except_unreachable_input = except_unreachable_input
        self.keep_locals = keep_locals

    def __call__(self, *args):
        return self.fn(*args)

    def __copy__(self):
        return Function(self.inputs, self.outputs,
                        features = self.features,
                        optimizer = self.optimizer,
                        linker_cls = self.linker_cls,
                        profiler = self.profiler,
                        unpack_single = self.unpack_single,
                        except_unreachable_input = self.except_unreachable_input,
                        keep_locals = self.keep_locals)


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
        optimizer(env)
    linker = linker_cls(env)
    fn = linker.make_function(inplace=True, unpack_single=unpack_single)
    rval = fn(*in_data)
    return rval


def infer_reuse_pattern(env, outputs_to_disown):
    do_not_reuse = outputs_to_disown
    seen = set()
    def walk(r):
        if env.edge(r) or r in seen:
            return
        seen.add(r)
        do_not_reuse.append(r)
        op = r.owner
        dmap = op.destroy_map() if hasattr(op, 'destroy_map') else {}
        vmap = op.view_map() if hasattr(op, 'view_map') else {}
        cat = lambda x, y: list(x) + list(y)
        for r2 in reduce(cat, dmap.values()) + reduce(cat, vmap.values()):
            accumulate(r2)
    for output in outputs_to_disown:
        walk(output)
    return do_not_reuse



# StateFunction([x, y], [e], (w, w + lr * bla()))




# class _Function:

#     def __init__(self,
#                  inputs,
#                  outputs,
#                  optimizer,
#                  linker_type = 'py',
#                  unpack_single = True,
#                  except_unreachable_input = True,
#                  disposable_inputs = [],
#                  borrow_outputs = []):




#         _mark_indestructible(outputs)

#         if len(inputs) != len(set(inputs)):
#             raise Exception('duplicate inputs')
#         if len(outputs) != len(set(outputs)):
#             raise Exception('duplicate outputs')

#         orphans = list(gof.graph.results_and_orphans(inputs, outputs,
#             except_unreachable_input=except_unreachable_input)[1])
#         orphan_data = eval_outputs(orphans, unpack_single=False)

#         env = gof.env.Env(inputs, outputs, features + [gof.EquivTool], consistency_check = True)

#         env = env.clone(clone_inputs=True)

#         for d, o in zip(orphan_data, [env.equiv(orphan) for orphan in orphans]):
#             o.data = d

#         # optimize and link the cloned env
#         if None is not optimizer:
#             optimizer(env)

#         linker = linker_cls(env)

#         if keep_locals:# useful flag for debugging!
#             self.__dict__.update(locals())

#         if profiler is None:
#             self.fn  = linker.make_function(inplace=True,
#                                             unpack_single=unpack_single)
#         else:
#             self.fn  = linker.make_function(inplace=True,
#                                             unpack_single=unpack_single,
#                                             profiler=profiler)
#         self.inputs = env.inputs
#         self.outputs = env.outputs
#         self.features = features
#         self.optimizer = optimizer
#         self.linker_cls = linker_cls
#         self.profiler = profiler
#         self.unpack_single = unpack_single
#         self.except_unreachable_input = except_unreachable_input
#         self.keep_locals = keep_locals

#     def __call__(self, *args):
#         return self.fn(*args)

#     def __copy__(self):
#         return Function(self.inputs, self.outputs,
#                         features = self.features,
#                         optimizer = self.optimizer,
#                         linker_cls = self.linker_cls,
#                         profiler = self.profiler,
#                         unpack_single = self.unpack_single,
#                         except_unreachable_input = self.except_unreachable_input,
#                         keep_locals = self.keep_locals)












# class StateFunction:

#     def __init__(self, inputs, outputs, *states):
#         in_states, out_states = zip(*states)
#         env = 

    
