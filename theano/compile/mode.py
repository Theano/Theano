"""WRITEME
"""
import os, logging, warnings

import numpy, theano
from theano import gof
import theano.gof.vm
from theano.configparser import config, AddConfigVar, StrParam, EnumStr


_logger = logging.getLogger('theano.compile.mode')

AddConfigVar('optimizer_excluding',
        "When using the default mode, we will remove optimizer with that tag. Separate many tags with ':'.",
        StrParam("", allow_override=False),
        in_c_key=False)
AddConfigVar('optimizer_including',
        "When using the default mode, we will add optimizer with that tag. Separate many tags with ':'.",
        StrParam("", allow_override=False),
        in_c_key=False)
AddConfigVar('optimizer_requiring',
        "When using the default mode, we will require optimizer with that tag. Separate many tags with ':'.",
        StrParam("", allow_override=False),
        in_c_key=False)

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
    'c&py' : gof.DualLinker(checker = check_equal),
    'vm'   : gof.vm.VM_Linker(allow_gc=True, use_cloop=False),
    'cvm'   : gof.vm.VM_Linker(allow_gc=True, use_cloop=True),
    'vm_nogc' : gof.vm.VM_Linker(allow_gc=False, use_cloop=False),
    'cvm_nogc': gof.vm.VM_Linker(allow_gc=False, use_cloop=True),
    }


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
OPT_STABILIZE = gof.Query(include = ['fast_run'])
OPT_STABILIZE.position_cutoff = 1.5000001

predefined_optimizers = {
    None    : lambda env: None,
    'None'    : lambda env: None,
    'merge' : gof.MergeOptimizer(),
    'fast_run' : OPT_FAST_RUN,
    'fast_run_stable' : OPT_FAST_RUN_STABLE,
    'fast_compile' : OPT_FAST_COMPILE,
    'stabilize': OPT_STABILIZE
    }

def register_optimizer(name, opt):
    """Add a `Optimizer` which can be referred to by `name` in `Mode`."""
    if name in predefined_optimizers:
        raise ValueError('Optimizer name already taken: %s' % name)
    predefined_optimizers[name] = opt

def register_OutputGuard_c_code(type):
    OutputGuard.c_code_types.append(type)

class OutputGuard(gof.Op):
    """
    This op is used only internally by Theano.

    Only the AddDestroyHandler optimizer tries to insert them in the graph.

    This Op is declared as destructive while it is not destroying
    anything. It returns a view. This is used to prevent destruction of
    the output variables of a Theano function.

    There is a mechanism in Theano that should prevent this, but the use
    of OutputGuard adds a safeguard: it may be possible for some optimization
    run before the add_destroy_handler phase to bypass this mechanism, by
    making in-place optimizations.

    TODO: find a current full explanation.
    """
    destroy_map = {0:[0]}
    view_map = {0:[0]}
    c_code_types = []

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = x
    def __str__(self):
        return '%s' % self.__class__.__name__

    def c_code(self, node, nodename, inp, out, sub):
        x, = inp
        z, = out
        if isinstance(node.inputs[0].type, theano.scalar.Scalar):
            # Scalars are C objects on the stacks, and should not be inc/decrefed
            return """
            %(z)s = %(x)s;
            """ % locals()
        elif (isinstance(node.inputs[0].type, tuple(self.c_code_types))):
            # These are Python object types
            return """
            Py_XDECREF(%(z)s);
            %(z)s = %(x)s;
            Py_XINCREF(%(z)s);
            """ % locals()

        # Else, no C code for you
        return super(OutputGuard, self).c_code(node, nodename, inp, out, sub)

    def c_code_cache_version(self):
        return (2,)

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
                _logger.info("Output variable %s required output_guard,"
                        " how was this output left unprotected against destructive operations?"
                        % o)
            except gof.InconsistencyError:
                #this output is already impossible to destroy. no guard necessary
                pass
    def add_requirements(self, env):
        super(AddDestroyHandler, self).add_requirements(env)
        env.extend(gof.DestroyHandler())

class PrintCurrentEnv(gof.Optimizer):
    """This optimizer is for debugging.

    Toss it into the optimization pipeline to see the state of things at any given point.
    """
    def __init__(self, header):
        self.header =header
    def apply(self, env):
        import theano.printing
        print "PrintCurrentEnv:", self.header
        theano.printing.debugprint(env.outputs)

optdb = gof.SequenceDB()
optdb.register('merge1', gof.MergeOptimizer(),
        0, 'fast_run', 'fast_compile')
optdb.register('canonicalize', gof.EquilibriumDB(),         # rearranges elemwise expressions
        1, 'fast_run', 'fast_compile')
optdb.register('merge1.2', gof.MergeOptimizer(skip_const_merge=False),
        1.2, 'fast_run', 'fast_compile')
optdb.register('Print1.21', PrintCurrentEnv('Post-canonicalize'),
        1.21,)# 'fast_run', 'fast_compile')

optdb.register('stabilize', gof.EquilibriumDB(),            # replace unstable subgraphs
        1.5, 'fast_run')
optdb.register('Print1.51', PrintCurrentEnv('Post-stabilize'),
        1.51,) #'fast_run', 'fast_compile')
optdb.register('specialize', gof.EquilibriumDB(),           # misc special cases for speed
        2, 'fast_run')
optdb.register('Print2.01', PrintCurrentEnv('Post-specialize'),
        2.01, )#'fast_run', 'fast_compile')
optdb.register('uncanonicalize', gof.EquilibriumDB(),# misc special cases for speed that break canonicalization
        3, 'fast_run')
optdb.register('specialize_device', gof.EquilibriumDB(),           # misc special cases for speed that are dependent on the device.
        48.6, 'fast_run')#must be after gpu stuff at 48.5
optdb.register('merge2', gof.MergeOptimizer(),              # especially constant merge
        49, 'fast_run')
optdb.register('add_destroy_handler', AddDestroyHandler(),
        49.5, 'fast_run', 'inplace')
optdb.register('merge3', gof.MergeOptimizer(),              # final pass just to make sure
        100, 'fast_run')


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

    def __init__(self, linker = config.linker, optimizer = config.optimizer):
        self.__setstate__((linker, optimizer))
        #self.provided_optimizer - typically the `optimizer` arg.  But if the `optimizer` arg is
        #    keyword corresponding to a predefined Query, then this stores the query
        #self._optimizer - typically same as provided_optimizer??

        #self.__get_optimizer - returns self._optimizer (possibly querying optdb with self._optimizer)
        #self.optimizer - property that returns __get_optimizer()

    def __getstate__(self):
        return (self.provided_linker, self.provided_optimizer)

    def __setstate__(self, state):
        linker, optimizer = state
        self.provided_linker = linker
        self.provided_optimizer = optimizer
        if isinstance(linker, basestring) or linker is None:
            linker = predefined_linkers[linker]
        self.linker = linker
        if isinstance(optimizer, basestring) or optimizer is None:
            optimizer = predefined_optimizers[optimizer]
        if isinstance(optimizer, gof.Query):
            self.provided_optimizer = optimizer
        self._optimizer = optimizer
        self.call_time = 0
        self.fn_time = 0
        linker.mode = self #TODO: WHY IS THIS HERE?
        self.optimizer_time = 0
        self.linker_time = 0

    def __str__(self):
        return "Mode(linker = %s, optimizer = %s)" % (self.provided_linker, self.provided_optimizer)

    def __get_optimizer(self):
        if isinstance(self._optimizer, gof.Query):
            return optdb.query(self._optimizer)
        else:
            return self._optimizer

    optimizer = property(__get_optimizer)

    def get_linker_optimizer(self, linker, optimizer):
        if isinstance(linker, basestring) or linker is None:
            linker = predefined_linkers[linker]
        if isinstance(optimizer, basestring) or optimizer is None:
            optimizer = predefined_optimizers[optimizer]
        return (linker, optimizer)

    def including(self, *tags):
        link, opt = self.get_linker_optimizer(self.provided_linker, self.provided_optimizer)
        #N.B. opt might be a Query instance, not sure what else it might be...
        #     string? Optimizer? OptDB? who knows???
        return self.__class__(linker=link, optimizer=opt.including(*tags))

    def excluding(self, *tags):
        link, opt = self.get_linker_optimizer(self.provided_linker, self.provided_optimizer)
        return self.__class__(linker=link, optimizer=opt.excluding(*tags))

    def requiring(self, *tags):
        link, opt = self.get_linker_optimizer(self.provided_linker, self.provided_optimizer)
        return self.__class__(linker=link, optimizer=opt.requiring(*tags))

# If a string is passed as the mode argument in function or
# FunctionMaker, the Mode will be taken from this dictionary using the
# string as the key
FAST_COMPILE = Mode('py', 'fast_compile')
FAST_RUN = Mode('c|py', 'fast_run')

predefined_modes = {'FAST_COMPILE': FAST_COMPILE,
                    'FAST_RUN': FAST_RUN,
                    }

instanciated_default_mode=None
def get_mode(orig_string):
    if orig_string is None:
        string = config.mode
    else:
        string = orig_string
    if not isinstance(string, basestring):
        return string #it is hopefully already a mode...

    global instanciated_default_mode
    # The default mode is cached. However, config.mode can change
    # If instanciated_default_mode has the right class, use it.
    if orig_string is None and instanciated_default_mode:
        if predefined_modes.has_key(string):
            default_mode_class = predefined_modes[string].__class__.__name__
        else:
            default_mode_class = string
        if (instanciated_default_mode.__class__.__name__ ==
                default_mode_class):
            return instanciated_default_mode

    if string in ['Mode','ProfileMode','DebugMode']:
        if string == 'DebugMode':
            #need to import later to break circular dependency.
            from debugmode import DebugMode
            #DebugMode use its own linker.
            ret = DebugMode(optimizer=config.optimizer)
        else:
            # The import is needed in case string is ProfileMode
            from profilemode import ProfileMode,prof_mode_instance_to_print
            ret = eval(string+'(linker=config.linker, optimizer=config.optimizer)')
    elif predefined_modes.has_key(string):
        ret = predefined_modes[string]
    else:
        raise Exception("No predefined mode exist for string: %s"%string)

    if orig_string is None:
        # Build and cache the default mode
        if theano.config.optimizer_excluding:
            ret = ret.excluding(*theano.config.optimizer_excluding.split(':'))
        if theano.config.optimizer_including:
            ret = ret.including(*theano.config.optimizer_including.split(':'))
        if theano.config.optimizer_requiring:
            ret = ret.requiring(*theano.config.optimizer_requiring.split(':'))
        instanciated_default_mode = ret

    #must tell python to print the summary at the end.
    if string == 'ProfileMode':
        #need to import later to break circular dependency.
        prof_mode_instance_to_print.append(ret)

    return ret

def get_default_mode():
    return get_mode(None)

# Removed: use config.mode instead.
#default_mode = config.mode

def register_mode(name, mode):
    """Add a `Mode` which can be referred to by `name` in `function`."""
    if name in predefined_modes:
        raise ValueError('Mode name already taken: %s' % name)
    predefined_modes[name] = mode
