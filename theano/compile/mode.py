
import numpy
from .. import gof

def check_equal(x, y):
    """
    Returns True iff x[0] and y[0] are equal (checks the dtype and
    shape if x and y are numpy.ndarray instances). Used internally.
    """
    x, y = x[0], y[0]
    if isinstance(x, numpy.ndarray) and isinstance(y, numpy.ndarray):
        if x.dtype != y.dtype or x.shape != y.shape or numpy.any(abs(x - y) > 1e-10):
            raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})
    else:
        if x != y:
            raise Exception("Output mismatch.", {'performlinker': x, 'clinker': y})





# If a string is passed as the linker argument in the constructor for
# Mode, it will be used as the key to retrieve the real linker in this
# dictionary
predefined_linkers = {
    'py'   : gof.PerformLinker(),
    'c'    : gof.CLinker(),
    'c|py' : gof.OpWiseCLinker(),
    'c&py' : gof.DualLinker(checker = check_equal)
    }

default_linker = 'c|py'

def register_linker(name, linker):
    """Add a `Linker` which can be referred to by `name` in `Mode`."""
    if name in predefined_linkers:
        raise ValueError('Linker name already taken: %s' % name)
    predefined_linkers[name] = linker





# If a string is passed as the optimizer argument in the constructor
# for Mode, it will be used as the key to retrieve the real optimizer
# in this dictionary
OPT_FAST_RUN = gof.Query(include = ['fast_run'])
OPT_FAST_RUN_STABLE = OPT_FAST_RUN.requiring('stable')
OPT_FAST_COMPILE = gof.Query(include = ['fast_compile'])

predefined_optimizers = {
    None    : lambda env: None,
    'merge' : gof.MergeOptimizer(),
    'fast_run' : OPT_FAST_RUN,
    'fast_run_stable' : OPT_FAST_RUN_STABLE,
    'fast_compile' : OPT_FAST_COMPILE
    }
default_optimizer = 'merge'

def register_optimizer(name, opt):
    """Add a `Optimizer` which can be referred to by `name` in `Mode`."""
    if name in predefined_optimizers:
        raise ValueError('Optimizer name already taken: %s' % name)
    predefined_optimizers[name] = opt

class AddDestroyHandler(gof.Optimizer):
    def apply(self, env):
        pass
    def add_requirements(self, env):
        super(AddDestroyHandler, self).add_requirements(env)
        env.extend(gof.DestroyHandler())

optdb = gof.SequenceDB()
optdb.register('merge1', gof.MergeOptimizer(), 0, 'fast_run', 'fast_compile')
optdb.register('canonicalize', gof.EquilibriumDB(), 1, 'fast_run')
optdb.register('specialize', gof.EquilibriumDB(), 2, 'fast_run')
optdb.register('merge2', gof.EquilibriumDB(), 49, 'fast_run')
optdb.register('add_destroy_handler', AddDestroyHandler(), 49.5, 'fast_run', 'inplace')
optdb.register('merge3', gof.EquilibriumDB(), 100, 'fast_run')


class Mode(object):
    """
    The Mode represents a way to optimize and then link a computation
    graph.

     * optimizer -> a structure of type Optimizer. An Optimizer may
       simplify the math, put similar computations together, improve
       numerical stability and various other improvements.
     * linker -> a structure of type Linker. A Linker decides which
       implementations to use (C or Python, for example) and how to
       string them together to perform the computation.

    See predefined_linkers, predefined_optimizers and also
    predefined_modes.
    """
    
    def __init__(self, linker = default_linker, optimizer = default_optimizer):
        self.__setstate__((linker, optimizer))

    def __getstate__(self):
        return (self.provided_linker, self.provided_optimizer)

    def __setstate__(self, (linker, optimizer)):
        self.provided_linker = linker
        self.provided_optimizer = optimizer
        if isinstance(linker, str) or linker is None:
            linker = predefined_linkers[linker]
        self.linker = linker
        if isinstance(optimizer, str) or optimizer is None:
            optimizer = predefined_optimizers[optimizer]
        if isinstance(optimizer, gof.Query):
            self.provided_optimizer = optimizer
        self._optimizer = optimizer

    def __str__(self):
        return "Mode(linker = %s, optimizer = %s)" % (self.provided_linker, self.provided_optimizer)

    def __get_optimizer(self):
        if isinstance(self._optimizer, gof.Query):
            return optdb.query(self._optimizer)
        else:
            return self._optimizer

    optimizer = property(__get_optimizer)

    def including(self, *tags):
        return Mode(self.provided_linker, self.provided_optimizer.including(*tags))

    def excluding(self, *tags):
        return Mode(self.provided_linker, self.provided_optimizer.excluding(*tags))

    def requiring(self, *tags):
        return Mode(self.provided_linker, self.provided_optimizer.requiring(*tags))

# If a string is passed as the mode argument in function or
# FunctionMaker, the Mode will be taken from this dictionary using the
# string as the key

FAST_COMPILE = Mode('py', 'fast_compile')
FAST_RUN = Mode('c|py', 'fast_run')

predefined_modes = {'FAST_COMPILE': FAST_COMPILE,
                    'FAST_RUN': FAST_RUN}
default_mode = 'FAST_COMPILE'

def register_mode(name, mode):
    """Add a `Mode` which can be referred to by `name` in `function`."""
    if name in predefined_modes:
        raise ValueError('Mode name already taken: %s' % name)
    predefined_modes[name] = mode

