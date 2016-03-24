"""We don't have real tests for the cache, but it would be great to make them!

But this one tests a current behavior that isn't good: the c_code isn't
deterministic based on the input type and the op.

"""
from __future__ import absolute_import, print_function, division

import numpy

import theano
from theano.gof.cmodule import GCC_compiler


class MyOp(theano.compile.ops.DeepCopyOp):
    nb_called = 0

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, name, inames, onames, sub):
        MyOp.nb_called += 1
        iname, = inames
        oname, = onames
        fail = sub['fail']
        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            rand = numpy.random.rand()
            return ("""printf("%(rand)s\\n");""" + code) % locals()
        # Else, no C code
        return super(theano.compile.ops.DeepCopyOp, self).c_code(
            node, name, inames, onames, sub)


def test_inter_process_cache():
    """When an op with c_code, but no version. If we have 2 apply node
    in the graph with different inputs variable(so they don't get
    merged) but the inputs variable have the same type, do we reuse
    the same module? Even if they would generate different c_code?
    Currently this test show that we generate the c_code only once.

    This is to know if the c_code can add information specific to the
    node.inputs[*].owner like the name of the variable.

    """

    x, y = theano.tensor.dvectors('xy')
    f = theano.function([x, y], [MyOp()(x), MyOp()(y)])
    f(numpy.arange(60), numpy.arange(60))
    if theano.config.mode == 'FAST_COMPILE' or theano.config.cxx == "":
        assert MyOp.nb_called == 0
    else:
        assert MyOp.nb_called == 1

    # What if we compile a new function with new variables?
    x, y = theano.tensor.dvectors('xy')
    f = theano.function([x, y], [MyOp()(x), MyOp()(y)])
    f(numpy.arange(60), numpy.arange(60))
    if theano.config.mode == 'FAST_COMPILE' or theano.config.cxx == "":
        assert MyOp.nb_called == 0
    else:
        assert MyOp.nb_called == 1


def test_flag_detection():
    # Check that the code detecting blas flags does not raise any exception.
    # It used to happen on python 3 because of improper string handling,
    # but was not detected because that path is not usually taken,
    # so we test it here directly.
    GCC_compiler.try_flags(["-lblas"])
