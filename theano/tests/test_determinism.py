from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from theano.tests.record import RecordMode, Record
from theano.tests import disturb_mem
import numpy as np
import theano
from theano.printing import var_descriptor

from theano import config, shared
from six import StringIO
from six.moves import xrange


def sharedX(x, name=None):
    x = np.cast[config.floatX](x)
    return shared(x, name)


def test_determinism_1():

    # Tests that repeatedly running a script that compiles and
    # runs a function does exactly the same thing every time it
    # is run, even when the memory addresses of the objects involved
    # change.
    # This specific script is capable of catching a bug where
    # FunctionGraph.toposort was non-deterministic.

    def run(replay, log=None):

        if not replay:
            log = StringIO()
        else:
            log = StringIO(log)
        record = Record(replay=replay, file_object=log)

        disturb_mem.disturb_mem()

        mode = RecordMode(record=record)

        b = sharedX(np.zeros((2,)), name='b')
        channels = OrderedDict()

        disturb_mem.disturb_mem()

        v_max = b.max(axis=0)
        v_min = b.min(axis=0)
        v_range = v_max - v_min

        updates = []
        for i, val in enumerate([
                v_max.max(),
                v_max.min(),
                v_range.max(),
                ]):
            disturb_mem.disturb_mem()
            s = sharedX(0., name='s_' + str(i))
            updates.append((s, val))

        for var in theano.gof.graph.ancestors(update for _, update in updates):
            if var.name is not None and var.name is not 'b':
                if var.name[0] != 's' or len(var.name) != 2:
                    var.name = None

        for key in channels:
            updates.append((s, channels[key]))
        f = theano.function([], mode=mode, updates=updates,
                            on_unused_input='ignore', name='f')
        for output in f.maker.fgraph.outputs:
            mode.record.handle_line(var_descriptor(output) + '\n')
        disturb_mem.disturb_mem()
        f()

        mode.record.f.flush()

        if not replay:
            return log.getvalue()

    log = run(0)
    # Do several trials, since failure doesn't always occur
    # (Sometimes you sample the same outcome twice in a row)
    for i in xrange(10):
        run(1, log)

if __name__ == '__main__':
    test_determinism_1()
