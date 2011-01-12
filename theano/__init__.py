"""
Theano is an optimizing compiler in Python, built to evaluate complicated expressions
(especially matrix-valued ones) as quickly as possible.
Theano compiles expression graphs (see :doc:`graph` ) that are built by Python code.
The expressions in these graphs are called `Apply` nodes and the variables in these graphs are called `Variable` nodes.

You compile a graph by calling `function`, which takes a graph, and returns a callable object.
One of theano's most important features is that `function` can transform your graph before
compiling it.
It can replace simple expressions with faster or more numerically stable implementations.

To learn more, check out:

- Joseph Turian's n00b walk through ( :wiki:`UserBasic` )

- an introduction to extending theano ( :wiki:`UserAdvanced` )

- Terminology Glossary (:wiki:`glossary`)

- Index of Howto documents (:wiki:`IndexHowto`)

- Op List (:doc:`oplist`)

"""

__docformat__ = "restructuredtext en"

# Set a default logger. It is important to do this before importing some other
# theano code, since this code may want to log some messages.
import logging
logging_default_handler = logging.StreamHandler()
logging.getLogger("theano").addHandler(logging_default_handler)
logging.getLogger("theano").setLevel(logging.WARNING)

import configparser, configdefaults

config = configparser.TheanoConfigParser()

import gof

from gof import \
     CLinker, OpWiseCLinker, DualLinker, Linker, LocalLinker, PerformLinker, \
     Container, \
     InconsistencyError, Env, \
     Apply, Variable, Constant, Value, \
     Op, \
     opt, \
     toolbox, \
     Type, Generic, generic, \
     object2, utils

from compile import \
    SymbolicInput, In, \
    SymbolicOutput, Out, \
    Mode, \
    predefined_modes, predefined_linkers, predefined_optimizers, \
    FunctionMaker, function, OpFromGraph, \
    Component, External, Member, Method, \
    Composite, ComponentList, ComponentDict, Module, \
    ProfileMode, \
    Param, shared

from misc.safe_asarray import _asarray

import numpy.testing
test = numpy.testing.Tester().test

FancyModule = Module

from printing import \
    pprint, pp

from scan import scan,map, reduce, foldl, foldr

import tensor
import scalar
#import sparse #we don't import by default as we don't want to force having scipy installed.
import gradient
import gof

if config.device.startswith('gpu') or config.init_gpu_device.startswith('gpu'):
    import theano.sandbox.cuda

## import scalar_opt

### This is defined here because it is designed to work across symbolic datatypes
#   (Sparse and Tensor)
def dot(l, r):
    """Return a symbolic matrix/dot product between l and r """
    rval = NotImplemented
    e0, e1 = None, None

    if rval == NotImplemented and hasattr(l, '__dot__'):
        try:
            rval = l.__dot__(r)
        except Exception, e0:
            rval = NotImplemented
    if rval == NotImplemented and hasattr(r, '__rdot__'):
        try:
            rval = r.__rdot__(l)
        except Exception, e1:
            rval = NotImplemented
    if rval == NotImplemented:
        raise NotImplementedError("Dot failed for the following reaons:", (e0, e1))
    return rval


# Version information
try:
    import theano.version
    __version__ = theano.version.version
except ImportError:
    __version__ = "unknown"
