"""
gof.py

gof stands for Graph Optimization Framework.

The gof submodule of theano implements a framework
for manipulating programs described as graphs. The
gof module defines basic theano graph concepts:
    -Apply nodes, which represent the application
of an Op to Variables. Together these make up a
graph.
    -The Type, needed for Variables to make sense.
    -The FunctionGraph, which defines how a subgraph
should be interpreted to implement a function.
    -The Thunk, a callable object that becames part
of the executable emitted by theano.
    -Linkers/VMs, the objects that call Thunks in
sequence in order to execute a theano program.

Conceptually, gof is intended to be sufficiently abstract
that it could be used to implement a language other than
theano. ie, theano is a domain-specific language for
numerical computation, created by implementing
tensor Variables and Ops that perform mathematical functions.
A different kind of domain-specific language could be
made by using gof with different Variables and Ops.
In practice, gof and the rest of theano are somewhat more
tightly intertwined.

Currently, gof also contains much of the C compilation
functionality. Ideally this should be refactored into
a different submodule.

For more details and discussion, see the theano-dev
e-mail thread "What is gof?".

"""
from __future__ import absolute_import, print_function, division

from theano.gof.cc import \
    CLinker, OpWiseCLinker, DualLinker, HideC

from theano.gof.fg import \
    CachedConstantError, InconsistencyError, MissingInputError, FunctionGraph

from theano.gof.destroyhandler import \
    DestroyHandler

from theano.gof.graph import \
    Apply, Variable, Constant, view_roots

from theano.gof.link import \
    Container, Linker, LocalLinker, PerformLinker, WrapLinker, WrapLinkerMany

from theano.gof.op import \
    Op, OpenMPOp, PureOp, COp, ops_with_inner_function

from theano.gof.opt import (
    Optimizer,
    optimizer, inplace_optimizer,
    SeqOptimizer,
    MergeOptimizer,
    LocalOptimizer, local_optimizer, LocalOptGroup,
    OpSub, OpRemove, PatternSub,
    NavigatorOptimizer, TopoOptimizer, EquilibriumOptimizer,
    OpKeyOptimizer)

from theano.gof.optdb import \
    DB, LocalGroupDB, Query, \
    EquilibriumDB, SequenceDB, ProxyDB

from theano.gof.toolbox import \
    Feature, \
    Bookkeeper, History, Validator, ReplaceValidate, NodeFinder,\
    PrintListener, ReplacementDidntRemovedError, NoOutputFromInplace

from theano.gof.type import \
    Type, Generic, generic

from theano.gof.utils import \
    hashtype, object2, MethodNotDefined

import theano

if theano.config.cmodule.preload_cache:
    cc.get_module_cache()
