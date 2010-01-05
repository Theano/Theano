"""WRITEME
"""
import os, logging

import numpy
from theano import gof
import theano.config as config

_logger = logging.getLogger('theano.compile.mode')

def check_equal(x, y):
    """
    Returns True iff x[0] and y[0] are equal (checks the dtype and
    shape if x and y are numpy.ndarray instances). Used internally.
    """
    #I put the import here to allow using theano without scipy.
    import scipy.sparse as sp
    x, y = x[0], y[0]
   
    # TODO: bug in current scipy, two sparse matrices are never equal, remove when moving to 0.7
    if sp.issparse(x):
        x = x.todense()
    if sp.issparse(y):
        y = y.todense()

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
    'c|py' : gof.OpWiseCLinker(allow_gc=True),
    'c|py_nogc' : gof.OpWiseCLinker(allow_gc=False),
    'c&py' : gof.DualLinker(checker = check_equal)
    }

#Keep default_linker the same as the one for default_mode
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
#Keep default_optimizer the same as the one for default_mode
default_optimizer = 'fast_run'

def register_optimizer(name, opt):
    """Add a `Optimizer` which can be referred to by `name` in `Mode`."""
    if name in predefined_optimizers:
        raise ValueError('Optimizer name already taken: %s' % name)
    predefined_optimizers[name] = opt

class OutputGuard(gof.Op):
    destroy_map = {0:[0]}
    view_map = {0:[0]}
    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def perform(self, node, (x,), (z,)):
        z[0] = x
    def __str__(self):
        return '%s' % self.__class__.__name__
    def c_code(self, node, nodename, (x,), (z,), sub):
        return """
        Py_XDECREF(%(z)s);
        %(z)s = %(x)s;
        Py_XINCREF(%(z)s);
        """ %locals()
    def c_code_cache_version(self):
        return (1,)
_output_guard = OutputGuard()

class AddDestroyHandler(gof.Optimizer):
    """This optimizer performs two important functions:

    1) it has a 'requirement' of the destroyhandler.  This means that the env will include it
    as a feature for this optimization, and keep this feature enabled for subsequent
    optimizations.  All optimizations that work inplace on any of their inputs must run *after*
    this optimization to ensure that the DestroyHandler has been included in the env.

    2) It tries to replace each output with an Op that purports to destroy it (but it won't I
    promise).   If this replacement succeeds it means that there is a bug in theano.  It should
    not be possible to destroy outputs.
    """
    def apply(self, env):
        for o in env.outputs:
            try:
                env.replace_validate(o, _output_guard(o), reason='output_guard')
                _logger.warning("Output variable %s required output_guard,"
                        " how was this output left unprotected against destructive operations?"
                        % o)
            except gof.InconsistencyError:
                #this output is already impossible to destroy. no guard necessary
                pass
    def add_requirements(self, env):
        super(AddDestroyHandler, self).add_requirements(env)
        env.extend(gof.DestroyHandler())

optdb = gof.SequenceDB()
optdb.register('merge1', gof.MergeOptimizer(), 0, 'fast_run', 'fast_compile')
optdb.register('canonicalize', gof.EquilibriumDB(), 1, 'fast_run')
optdb.register('specialize', gof.EquilibriumDB(), 2, 'fast_run')
optdb.register('merge2', gof.MergeOptimizer(), 49, 'fast_run')
optdb.register('add_destroy_handler', AddDestroyHandler(), 49.5, 'fast_run', 'inplace')
optdb.register('merge3', gof.MergeOptimizer(), 100, 'fast_run')


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
FAST_RUN_NOGC = Mode("c|py_nogc", 'fast_run')
SANITY_CHECK = [Mode('c|py', None),
                Mode('c|py', 'fast_run')]
predefined_modes = {'FAST_COMPILE': FAST_COMPILE,
                    'FAST_RUN': FAST_RUN,
                    'FAST_RUN_NOGC':FAST_RUN_NOGC,
                    'SANITY_CHECK': SANITY_CHECK}


##
# The default mode used by functions and modules is read from the environment
# variable THEANO_DEFAULT_MODE. Unit tests will run using this value. If the env. var.
# is not set, it will default to 'FAST_RUN'
# keep default_mode.optimizer==default_optimizer and default_mode.linker==default_linker!
##
default_mode = config.THEANO_DEFAULT_MODE

def get_mode(string):
    if string is None: string = default_mode
    if not isinstance(string, str): return string #it is already a mode...
    if not predefined_modes.has_key(string):
        raise Exception("No predefixed mode exist for string: %s"%string)
    return predefined_modes[string]

def get_default_mode():
    return get_mode(default_mode)

def register_mode(name, mode):
    """Add a `Mode` which can be referred to by `name` in `function`."""
    if name in predefined_modes:
        raise ValueError('Mode name already taken: %s' % name)
    predefined_modes[name] = mode

