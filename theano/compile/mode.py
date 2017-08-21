"""
WRITEME

"""
from __future__ import absolute_import, print_function, division
import logging
import warnings

import theano
from theano import gof
import theano.gof.vm
from theano import config
from six import string_types
from theano.compile.function_module import Supervisor


_logger = logging.getLogger('theano.compile.mode')


# If a string is passed as the linker argument in the constructor for
# Mode, it will be used as the key to retrieve the real linker in this
# dictionary
predefined_linkers = {
    'py': gof.PerformLinker(),  # Use allow_gc Theano flag
    'c': gof.CLinker(),  # Don't support gc. so don't check allow_gc
    'c|py': gof.OpWiseCLinker(),  # Use allow_gc Theano flag
    'c|py_nogc': gof.OpWiseCLinker(allow_gc=False),
    'vm': gof.vm.VM_Linker(use_cloop=False),  # Use allow_gc Theano flag
    'cvm': gof.vm.VM_Linker(use_cloop=True),  # Use allow_gc Theano flag
    'vm_nogc': gof.vm.VM_Linker(allow_gc=False, use_cloop=False),
    'cvm_nogc': gof.vm.VM_Linker(allow_gc=False, use_cloop=True)}


def register_linker(name, linker):
    """Add a `Linker` which can be referred to by `name` in `Mode`."""
    if name in predefined_linkers:
        raise ValueError('Linker name already taken: %s' % name)
    predefined_linkers[name] = linker


# If a string is passed as the optimizer argument in the constructor
# for Mode, it will be used as the key to retrieve the real optimizer
# in this dictionary
exclude = []
if not theano.config.cxx:
    exclude = ['cxx_only']
OPT_NONE = gof.Query(include=[], exclude=exclude)
# Even if multiple merge optimizer call will be there, this shouldn't
# impact performance.
OPT_MERGE = gof.Query(include=['merge'], exclude=exclude)
OPT_FAST_RUN = gof.Query(include=['fast_run'], exclude=exclude)
OPT_FAST_RUN_STABLE = OPT_FAST_RUN.requiring('stable')
# We need fast_compile_gpu here.  As on the GPU, we don't have all
# operation that exist in fast_compile, but have some that get
# introduced in fast_run, we want those optimization to also run in
# fast_compile+gpu. We can't tag them just as 'gpu', as this would
# exclude them if we exclude 'gpu'.
OPT_FAST_COMPILE = gof.Query(include=['fast_compile', 'fast_compile_gpu'],
                             exclude=exclude)
OPT_STABILIZE = gof.Query(include=['fast_run'], exclude=exclude)
OPT_STABILIZE.position_cutoff = 1.5000001
OPT_NONE.name = 'OPT_NONE'
OPT_MERGE.name = 'OPT_MERGE'
OPT_FAST_RUN.name = 'OPT_FAST_RUN'
OPT_FAST_RUN_STABLE.name = 'OPT_FAST_RUN_STABLE'
OPT_FAST_COMPILE.name = 'OPT_FAST_COMPILE'
OPT_STABILIZE.name = 'OPT_STABILIZE'

OPT_O2 = OPT_FAST_COMPILE.including('fusion')
OPT_O3 = OPT_FAST_RUN.excluding('inplace')
OPT_UNSAFE = OPT_O3.including('unsafe')

OPT_O2.name = 'OPT_O2'
OPT_O3.name = 'OPT_O3'
OPT_UNSAFE.name = 'OPT_UNSAFE'

predefined_optimizers = {
    None: OPT_NONE,
    'None': OPT_NONE,
    'merge': OPT_MERGE,
    'o4': OPT_FAST_RUN,
    'o3': OPT_O3,
    'o2': OPT_O2,
    'o1': OPT_FAST_COMPILE,
    'unsafe': OPT_UNSAFE,
    'fast_compile': OPT_FAST_COMPILE,
    'fast_run': OPT_FAST_RUN,
    'fast_run_stable': OPT_FAST_RUN_STABLE,
    'stabilize': OPT_STABILIZE}


def register_optimizer(name, opt):
    """Add a `Optimizer` which can be referred to by `name` in `Mode`."""
    if name in predefined_optimizers:
        raise ValueError('Optimizer name already taken: %s' % name)
    predefined_optimizers[name] = opt


class AddDestroyHandler(gof.Optimizer):
    """
    This optimizer performs two important functions:

    1) It has a 'requirement' of the destroyhandler. This means that the fgraph
    will include it as a feature for this optimization, and keep this feature
    enabled for subsequent optimizations. All optimizations that work inplace
    on any of their inputs must run *after* this optimization to ensure that
    the DestroyHandler has been included in the fgraph.

    2) It tries to replace each output with an Op that purports to destroy it
    (but it won't I promise). If this replacement succeeds it means that
    there is a bug in theano. It should not be possible to destroy outputs.

    """
    def apply(self, fgraph):
        supervisor_added = False
        for feature in fgraph._features:
            if isinstance(feature, Supervisor):
                supervisor_added = True
                break
        if not supervisor_added:
            warnings.warn("WARNING: Supervisor is not added. Please build a FunctionGraph"
                          "via theano.compile.function_module.std_graph()"
                          "or add the Supervisor class manually.",
                          stacklevel=3)

    def add_requirements(self, fgraph):
        super(AddDestroyHandler, self).add_requirements(fgraph)
        fgraph.attach_feature(gof.DestroyHandler())


class AddFeatureOptimizer(gof.Optimizer):
    """
    This optimizer adds a provided feature to the function graph.
    """

    def __init__(self, feature):
        self.feature = feature

    def add_requirements(self, fgraph):
        super(AddFeatureOptimizer, self).add_requirements(fgraph)
        fgraph.attach_feature(self.feature)


class PrintCurrentFunctionGraph(gof.Optimizer):
    """
    This optimizer is for debugging.

    Toss it into the optimization pipeline to see the state of things at any
    given point.

    """
    def __init__(self, header):
        self.header = header

    def apply(self, fgraph):
        import theano.printing
        print("PrintCurrentFunctionGraph:", self.header)
        theano.printing.debugprint(fgraph.outputs)


optdb = gof.SequenceDB()
optdb.register('merge1', gof.MergeOptimizer(),
               0, 'fast_run', 'fast_compile', 'merge')

# After scan1 opt at 0.5 and before ShapeOpt at 1
# This should only remove nodes.
# The opt should not do anything that need shape inference.
# New nodes that don't have infer_shape need that the original node
# also don't have infer_shape
local_useless = gof.optdb.LocalGroupDB(apply_all_opts=True, profile=True)
optdb.register(
    'useless',
    gof.optdb.TopoDB(local_useless,
                     failure_callback=gof.opt.NavigatorOptimizer.warn_inplace),
    0.6, 'fast_run', 'fast_compile')

optdb.register('merge1.1', gof.MergeOptimizer(),
               0.65, 'fast_run', 'fast_compile', 'merge')

# rearranges elemwise expressions
optdb.register('canonicalize', gof.EquilibriumDB(ignore_newtrees=False),
               1, 'fast_run', 'fast_compile', 'canonicalize_db')
# Register in the canonizer Equilibrium as a clean up opt the merge opt.
# Without this, as the equilibrium have ignore_newtrees=False, we
# won't merge all nodes if it is set as a global optimizer with
# final_opt=True.

# We need a new instance of MergeOptimizer to don't have its name
# changed by other usage of it.
optdb['canonicalize'].register("merge", gof.opt.MergeOptimizer(), 'fast_run',
                               "fast_compile", cleanup=True)

optdb.register('merge1.2', gof.MergeOptimizer(),
               1.2, 'fast_run', 'fast_compile', 'merge')

optdb.register('Print1.21', PrintCurrentFunctionGraph('Post-canonicalize'),
               1.21,)  # 'fast_run', 'fast_compile')

# replace unstable subgraphs
optdb.register('stabilize', gof.EquilibriumDB(),
               1.5, 'fast_run')

optdb.register('Print1.51', PrintCurrentFunctionGraph('Post-stabilize'),
               1.51,)  # 'fast_run', 'fast_compile')

# misc special cases for speed
optdb.register('specialize', gof.EquilibriumDB(),
               2, 'fast_run', 'fast_compile_gpu')

# misc special cases for speed that break canonicalization
optdb.register('uncanonicalize', gof.EquilibriumDB(),
               3, 'fast_run')

# misc special cases for speed that are dependent on the device.
optdb.register('specialize_device', gof.EquilibriumDB(),
               48.6, 'fast_compile', 'fast_run')  # must be after gpu stuff at 48.5

# especially constant merge
optdb.register('merge2', gof.MergeOptimizer(),
               49, 'fast_run', 'merge')

optdb.register('add_destroy_handler', AddDestroyHandler(),
               49.5, 'fast_run', 'inplace')

# final pass just to make sure
optdb.register('merge3', gof.MergeOptimizer(),
               100, 'fast_run', 'merge')

if theano.config.check_stack_trace in ['raise', 'warn', 'log']:
    _tags = ('fast_run', 'fast_compile')

if theano.config.check_stack_trace == 'off':
    _tags = ()

optdb.register('CheckStackTrace',
               gof.CheckStackTraceOptimization(), -1, *_tags)
del _tags


class Mode(object):
    """
    The Mode represents a way to optimize and then link a computation graph.

    Parameters
    ----------
    optimizer : a structure of type Optimizer
        An Optimizer may simplify the math, put similar computations together,
        improve numerical stability and various other improvements.
    linker : a structure of type Linker
        A Linker decides which implementations to use (C or Python, for example)
        and how to string them together to perform the computation.

    See Also
    --------
    predefined_linkers
    predefined_optimizers
    predefined_modes

    """

    def __init__(self, linker=None, optimizer='default'):
        if linker is None:
            linker = config.linker
        if optimizer is 'default':
            optimizer = config.optimizer
        Mode.__setstate__(self, (linker, optimizer))

        # self.provided_optimizer - typically the `optimizer` arg.
        # But if the `optimizer` arg is keyword corresponding to a predefined
        # Query, then this stores the query
        # self._optimizer - typically same as provided_optimizer??

        # self.__get_optimizer - returns self._optimizer (possibly querying
        # optdb with self._optimizer)
        # self.optimizer - property that returns __get_optimizer()

    def __getstate__(self):
        return (self.provided_linker, self.provided_optimizer)

    def __setstate__(self, state):
        linker, optimizer = state
        self.provided_linker = linker
        self.provided_optimizer = optimizer
        if isinstance(linker, string_types) or linker is None:
            linker = predefined_linkers[linker]
        self.linker = linker
        if isinstance(optimizer, string_types) or optimizer is None:
            optimizer = predefined_optimizers[optimizer]
        if isinstance(optimizer, gof.Query):
            self.provided_optimizer = optimizer
        self._optimizer = optimizer
        self.call_time = 0
        self.fn_time = 0

    def __str__(self):
        return "%s(linker = %s, optimizer = %s)" % (self.__class__.__name__,
                                                    self.provided_linker,
                                                    self.provided_optimizer)

    def __get_optimizer(self):
        if isinstance(self._optimizer, gof.Query):
            return optdb.query(self._optimizer)
        else:
            return self._optimizer

    optimizer = property(__get_optimizer)

    def get_linker_optimizer(self, linker, optimizer):
        if isinstance(linker, string_types) or linker is None:
            linker = predefined_linkers[linker]
        if isinstance(optimizer, string_types) or optimizer is None:
            optimizer = predefined_optimizers[optimizer]
        return (linker, optimizer)

    def including(self, *tags):
        link, opt = self.get_linker_optimizer(self.provided_linker,
                                              self.provided_optimizer)
        # N.B. opt might be a Query instance, not sure what else it might be...
        #     string? Optimizer? OptDB? who knows???
        return self.clone(optimizer=opt.including(*tags), linker=link)

    def register(self, *optimizations):
        """Adds new optimization instances to a mode.

        This method adds new optimization instances to a compilation mode. It
        works like the `including()` method but takes as inputs optimization
        instances to add instead of tags.

        Parameters
        ----------
        optimizations :
            Every element of `optimizations` is a tuple containing an
            optimization instance and a floating point value indicating the
            position at which to insert the optimization in the mode.

        Returns
        -------
        Mode
            Copy of the current Mode which includes the provided
            optimizations.
        """

        link, opt = self.get_linker_optimizer(self.provided_linker,
                                              self.provided_optimizer)
        return self.clone(optimizer=opt.register(*optimizations))

    def excluding(self, *tags):
        link, opt = self.get_linker_optimizer(self.provided_linker,
                                              self.provided_optimizer)
        return self.clone(optimizer=opt.excluding(*tags), linker=link)

    def requiring(self, *tags):
        link, opt = self.get_linker_optimizer(self.provided_linker,
                                              self.provided_optimizer)
        return self.clone(optimizer=opt.requiring(*tags), linker=link)

    def clone(self, link_kwargs=None, optimizer="", **kwargs):
        """
        Create a new instance of this Mode.

        Keyword arguments can be provided for the linker,
        in which case its `clone` method will be called with these
        arguments.

        """
        if link_kwargs is None:
            link_kwargs = {}
        new_linker = self.linker.clone(**link_kwargs)

        if optimizer == "":
            optimizer = self.provided_optimizer
        new_mode = type(self)(linker=new_linker,
                              optimizer=optimizer)
        return new_mode


# If a string is passed as the mode argument in function or
# FunctionMaker, the Mode will be taken from this dictionary using the
# string as the key
# Use VM_linker to allow lazy evaluation by default.
FAST_COMPILE = Mode(theano.gof.vm.VM_Linker(use_cloop=False, c_thunks=False),
                    'fast_compile')
if theano.config.cxx:
    FAST_RUN = Mode('cvm', 'fast_run')
else:
    FAST_RUN = Mode('vm', 'fast_run')

predefined_modes = {'FAST_COMPILE': FAST_COMPILE,
                    'FAST_RUN': FAST_RUN,
                    }

instantiated_default_mode = None


def get_mode(orig_string):
    if orig_string is None:
        string = config.mode
    else:
        string = orig_string
    if not isinstance(string, string_types):
        return string  # it is hopefully already a mode...

    global instantiated_default_mode
    # The default mode is cached. However, config.mode can change
    # If instantiated_default_mode has the right class, use it.
    if orig_string is None and instantiated_default_mode:
        if string in predefined_modes:
            default_mode_class = predefined_modes[string].__class__.__name__
        else:
            default_mode_class = string
        if (instantiated_default_mode.__class__.__name__ ==
                default_mode_class):
            return instantiated_default_mode

    if string in ['Mode', 'DebugMode', 'NanGuardMode']:
        if string == 'DebugMode':
            # need to import later to break circular dependency.
            from .debugmode import DebugMode
            # DebugMode use its own linker.
            ret = DebugMode(optimizer=config.optimizer)
        elif string == 'NanGuardMode':
            # need to import later to break circular dependency.
            from .nanguardmode import NanGuardMode
            # NanGuardMode use its own linker.
            ret = NanGuardMode(True, True, True, optimizer=config.optimizer)
        else:
            # TODO: Can't we look up the name and invoke it rather than using eval here?
            ret = eval(string +
                       '(linker=config.linker, optimizer=config.optimizer)')
    elif string in predefined_modes:
        ret = predefined_modes[string]
    else:
        raise Exception("No predefined mode exist for string: %s" % string)

    if orig_string is None:
        # Build and cache the default mode
        if theano.config.optimizer_excluding:
            ret = ret.excluding(*theano.config.optimizer_excluding.split(':'))
        if theano.config.optimizer_including:
            ret = ret.including(*theano.config.optimizer_including.split(':'))
        if theano.config.optimizer_requiring:
            ret = ret.requiring(*theano.config.optimizer_requiring.split(':'))
        instantiated_default_mode = ret

    return ret


def get_default_mode():
    return get_mode(None)


def register_mode(name, mode):
    """
    Add a `Mode` which can be referred to by `name` in `function`.

    """
    if name in predefined_modes:
        raise ValueError('Mode name already taken: %s' % name)
    predefined_modes[name] = mode
