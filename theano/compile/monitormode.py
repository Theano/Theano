from __future__ import absolute_import, print_function, division
# Note: this code was initially copied from the 'pyutools' package by its
# original author, and re-licensed under Theano's license.
import numpy

import theano
from theano.compile.mode import Mode


class MonitorMode(Mode):
    """
    `MonitorMode` is a debug mode to easily step through function execution.

    Its default behavior is to behave like the 'FAST_RUN' mode. By providing
    either a `pre_func` (called before a node is executed) or a `post_func`
    (called after a node is executed) monitoring function, the user can inspect
    node behavior.

    A typical use case is to detect the introduction of NaN values in a graph.
    For an example of such a use case, see doc/tutorial/debug_faq.txt.

    Parameters
    ----------
    pre_func
        A function to call before executing a thunk, with arguments:
        - the thunk index
        - the Apply node
        - the thunk to be called
    post_func
        A function to call after executing a thunk, with the same three
        arguments as `pre_func`.
    optimizer
        The optimizer to use. One may use for instance 'fast_compile' to skip
        optimizations.
    linker
        DO NOT USE. This mode uses its own linker. The parameter is needed to
        allow selecting optimizers to use.

    """

    def __init__(self, pre_func=None, post_func=None,
                 optimizer='default', linker=None):
        self.pre_func = pre_func
        self.post_func = post_func
        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()],
                                                [self.eval])
        if optimizer == 'default':
            optimizer = theano.config.optimizer
        if (linker is not None and
                not isinstance(linker.mode, MonitorMode)):
            raise Exception("MonitorMode can only use its own linker! You "
                            "should not provide one.", linker)

        super(MonitorMode, self).__init__(wrap_linker, optimizer=optimizer)

    def __getstate__(self):
        lnk, opt = super(MonitorMode, self).__getstate__()
        return (lnk, opt, self.pre_func, self.post_func)

    def __setstate__(self, state):
        lnk, opt, pre_func, post_func = state
        self.pre_func = pre_func
        self.post_func = post_func
        super(MonitorMode, self).__setstate__((lnk, opt))

    def eval(self, i, node, fn):
        """
        The method that calls the thunk `fn`.

        """
        if self.pre_func is not None:
            self.pre_func(i, node, fn)
        fn()
        if self.post_func is not None:
            self.post_func(i, node, fn)

    def clone(self, link_kwargs=None, optimizer="", **kwargs):
        """
        Create a new instance of this Mode.

        Keyword arguments can be provided for the linker, but they will be
        ignored, because ProfileMode needs to use its own linker.

        """
        if optimizer == "":
            optimizer = self.provided_optimizer
        new_mode = type(self)(pre_func=self.pre_func,
                              post_func=self.post_func,
                              linker=None,
                              optimizer=optimizer)
        return new_mode


def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], numpy.random.RandomState) and
                numpy.isnan(output[0]).any()):
            print('*** NaN detected ***')
            theano.printing.debugprint(node)
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            break
