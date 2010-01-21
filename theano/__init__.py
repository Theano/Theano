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
     object2, utils, \
     set_compiledir, get_compiledir, clear_compiledir

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


FancyModule = Module

from printing import \
    pprint, pp

import tensor
import scalar
import sparse
import gradient
import gof
import floatX
floatX.set_floatX()

import config

#if THEANO_GPU not defined: don't automaticcaly importe cuda
#if THEANO_GPU defined to something else then "": automatically import cuda
#   he will init cuda automatically if THEANO_GPU is not -1 or GPU
#if cuda.use() and THEANO_GPU not defined or defined to "": init to device 0.
#if THEANO_GPU defined to "-1" or "CPU", automatically import cuda, but don't init it.
if config.THEANO_GPU not in [None,""]:
    import theano.sandbox.cuda

## import scalar_opt

import subprocess as _subprocess
import imp as _imp
def __src_version__():
    """Return compact identifier of module code.

    @return: compact identifier of module code.
    @rtype: string

    @note: This function tries to establish that the source files and the repo
    are synchronized.  It raises an Exception if there are un-tracked '.py'
    files, or if there are un-committed modifications.  This implementation uses
    "hg id" to establish this.  The code returned by "hg id" is not affected by
    hg pull, but pulling might remove the " tip" string which might have
    appeared.  This implementation ignores the  " tip" information, and only
    uses the code.

    @note: This implementation is assumes that the import directory is under
    version control by mercurial.

    """

    #
    # NOTE
    #
    # If you find bugs in this function, please update the __src_version__
    # function in pylearn, and email either theano-dev or pylearn-dev so that
    # people can update their experiment dirs (the output of this function is
    # meant to be hard-coded in external files).
    #


    if not hasattr(__src_version__, 'rval'):
        #print 'name:', __name__
        location = _imp.find_module(__name__)[1]
        #print 'location:', location

        status = _subprocess.Popen(('hg','st'),cwd=location,stdout=_subprocess.PIPE).communicate()[0]
        #status_codes = [line[0] for line in  if line and line[0] != '?']
        for line in status.split('\n'):
            if not line: continue
            if line[0] != '?':
                raise Exception('Uncommitted modification to "%s" in %s (%s)'
                        %(line[2:], __name__,location))
            if line[0] == '?' and line[-3:] == '.py':
                raise Exception('Untracked file "%s" in %s (%s)'
                        %(line[2:], __name__, location))

        hg_id = _subprocess.Popen(('hg','id'),cwd=location,stdout=_subprocess.PIPE).communicate()[0]

        #This asserts my understanding of hg id return values
        # There is mention in the doc that it might return two parent hash codes
        # but I've never seen it, and I dont' know what it means or how it is
        # formatted.
        tokens = hg_id.split(' ')
        assert len(tokens) <= 2
        assert len(tokens) >= 1
        assert tokens[0][-1] != '+' # the trailing + indicates uncommitted changes
        if len(tokens) == 2:
            assert tokens[1] == 'tip\n'

        __src_version__.rval = tokens[0]

    return __src_version__.rval

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


### 
#   Set a default logger
#
import logging
logging_default_handler = logging.StreamHandler()
logging.getLogger("theano").addHandler(logging_default_handler)
logging.getLogger("theano").setLevel(logging.WARNING)


