"""
Theano is an optimizing compiler in Python, built to evaluate complicated expressions
(especially matrix-valued ones) as quickly as possible. 
Theano compiles expression graphs (see :doc:`graph` ) that are built by Python code.
The expressions in these graphs are called `Apply` nodes and the variables in these graphs are called `Result` nodes.

You compile a graph by calling `function`, which takes a graph, and returns a callable object.
One of theano's most important features is that `function` can transform your graph before
compiling it.
It can replace simple expressions with faster or more numerically stable implementations.

To learn more, check out:
- Joseph Turian's n00b walk through ( :wiki:`UserBasic` )

- an introduction to extending theano ( :wiki:`UserAdvanced` )

- Terminology Glossary (:wiki:`TerminologyGlossary`)

- Index of Howto documents (:wiki:`HowtoIndex`)

"""

__docformat__ = "restructuredtext en"

from gof import \
     CLinker, OpWiseCLinker, DualLinker, Linker, LocalLinker, PerformLinker, Profiler, \
     InconsistencyError, Env, \
     Apply, Result, Constant, Value, \
     Op, \
     opt, \
     toolbox, \
     Type, Generic, generic, \
     object2, utils

from compile import function, eval_outputs, fast_compute, OpFromGraph

import tensor
import tensor_random
import scalar
import sparse
import gradient
import elemwise
import tensor_opt

## import scalar_opt

import subprocess as _subprocess
import imp as _imp
def __src_version__():
    """Return compact identifier of module code.

    @return: compact identifier of module code.
    @rtype: string

    @note: This function tries to establish that the source files and the repo
    are syncronized.  It raises an Exception if there are un-tracked '.py'
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

