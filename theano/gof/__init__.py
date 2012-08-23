"""
gof.py

gof stands for Graph Optimization Framework

The gof submodule of theano implements a framework
for manipulating programs described as graphs. The
gof module defines basic theano graph concepts:
    -Apply nodes, which represent the application
of an Op to Variables. Together these make up a
graph.
    -The Type, needed for Variables to make sense
    -The FunctionGraph, which defines how a subgraph
should be interpreted to implement a function
    -The Thunk, a callable object that becames part
of the executable emitted by theano
    -Linkers/VMs, the objects that call Thunks in
sequence in order to execute a theano program

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
e-mail thread "What is gof?"
"""

import sys

from cc import \
    CLinker, OpWiseCLinker, DualLinker

import compiledir # adds config vars

from fg import \
    InconsistencyError, MissingInputError, FunctionGraph

from destroyhandler import \
    DestroyHandler

from graph import \
    Apply, Variable, Constant, view_roots

from link import \
    Container, Linker, LocalLinker, PerformLinker, WrapLinker, WrapLinkerMany

from op import \
    Op, OpenMPOp, PureOp, ops_with_inner_function

from opt import (Optimizer, optimizer, SeqOptimizer,
    MergeOptimizer, MergeOptMerge,
    LocalOptimizer, local_optimizer, LocalOptGroup,
    OpSub, OpRemove, PatternSub,
    NavigatorOptimizer, TopoOptimizer, EquilibriumOptimizer,
    InplaceOptimizer, PureThenInplaceOptimizer,
    OpKeyOptimizer)

from optdb import \
    DB, Query, \
    EquilibriumDB, SequenceDB, ProxyDB

from toolbox import \
    Feature, \
    Bookkeeper, History, Validator, ReplaceValidate, NodeFinder,\
    PrintListener, ReplacementDidntRemovedError

from type import \
    Type, Generic, generic

from utils import \
    object2, MethodNotDefined

