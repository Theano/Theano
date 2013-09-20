# Note: this code was initially copied from the 'pyutools' package by its
# original author, and re-licensed under Theano's license.


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
    """

    def __init__(self, pre_func=None, post_func=None,
                 optimizer='default', linker=None):
        """
        Constructor.

        :param pre_func: A function to call before executing a thunk, with
            arguments:
                - the thunk index
                - the Apply node
                - the thunk to be called

        :param post_func: A function to call after executing a thunk, with the
            same three arguments as `pre_func`.

        :param optimizer: The optimizer to use. One may use for instance
            'fast_compile' to skip optimizations.

        :param linker: DO NOT USE. This mode uses its own linker.
            The parameter is needed to allow selecting optimizers to use.
        """
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

    def eval(self, i, node, fn):
        """
        The method that calls the thunk `fn`.
        """
        if self.pre_func is not None:
            self.pre_func(i, node, fn)
        fn()
        if self.post_func is not None:
            self.post_func(i, node, fn)

    def including(self, *tags):
        ret = super(MonitorMode, self).including(*tags)
        ret.pre_func = self.pre_func
        ret.post_func = self.post_func
        return ret

    def excluding(self, *tags):
        ret = super(MonitorMode, self).excluding(*tags)
        ret.pre_func = self.pre_func
        ret.post_func = self.post_func
        return ret

    def requiring(self, *tags):
        ret = super(MonitorMode, self).requiring(*tags)
        ret.pre_func = self.pre_func
        ret.post_func = self.post_func
        return ret
