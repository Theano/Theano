"""
Theano is an optimizing compiler in Python, built to evaluate
complicated expressions (especially matrix-valued ones) as quickly as
possible.  Theano compiles expression graphs (see :doc:`graph` ) that
are built by Python code. The expressions in these graphs are called
`Apply` nodes and the variables in these graphs are called `Variable`
nodes.

You compile a graph by calling `function`, which takes a graph, and
returns a callable object.  One of theano's most important features is
that `function` can transform your graph before compiling it.  It can
replace simple expressions with faster or more numerically stable
implementations.

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

theano_logger = logging.getLogger("theano")
logging_default_handler = logging.StreamHandler()
logging_default_formatter = logging.Formatter(
        fmt='%(levelname)s (%(name)s): %(message)s')
logging_default_handler.setFormatter(logging_default_formatter)
theano_logger.addHandler(logging_default_handler)
theano_logger.setLevel(logging.WARNING)

import configparser
import configdefaults

config = configparser.TheanoConfigParser()

# Version information.
import theano.version
__version__ = theano.version.version

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

import compile

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
import scan_module
from scan_module import scan, map, reduce, foldl, foldr, clone

from updates import Updates

import tensor
import scalar
#we don't import by default as we don't want to force having scipy installed.
#import sparse
import gradient
from gradient import Rop, Lop, grad
import gof

if config.device.startswith('gpu') or config.init_gpu_device.startswith('gpu'):
    import theano.sandbox.cuda

# Use config.numpy to call numpy.seterr
import numpy
if config.numpy.seterr_all == 'None':
    _all = None
else:
    _all = config.numpy.seterr_all
if config.numpy.seterr_divide == 'None':
    _divide = None
else:
    _divide = config.numpy.seterr_divide
if config.numpy.seterr_over == 'None':
    _over = None
else:
    _over = config.numpy.seterr_over
if config.numpy.seterr_under == 'None':
    _under = None
else:
    _under = config.numpy.seterr_under
if config.numpy.seterr_invalid == 'None':
    _invalid = None
else:
    _invalid = config.numpy.seterr_invalid
numpy.seterr(
        all=_all,
        divide=_divide,
        over=_over,
        under=_under,
        invalid=_invalid)
del _all, _divide, _over, _under, _invalid

## import scalar_opt

### This is defined here because it is designed to work across symbolic
#   datatypes (Sparse and Tensor)


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
        raise NotImplementedError("Dot failed for the following reasons:",
                                  (e0, e1))
    return rval
