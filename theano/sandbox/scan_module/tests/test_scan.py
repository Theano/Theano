import os
import shutil
from tempfile import mkdtemp
import time
import sys
import unittest

import cPickle
import numpy
from numpy.testing import dec

import theano
import theano.sandbox.rng_mrg
from theano import tensor
from theano.compile.pfunc import rebuild_collect_shared
from theano.gof.python25 import any
from theano.tests  import unittest_tools as utt
from numpy.testing.noseclasses import KnownFailureTest

from test_utils import *
import theano.sandbox.scan_module as scan_module
from theano.sandbox.scan_module.scan_op import ScanOp


class TestScan(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def new_run(self,
              inputs_info,
              states_info,
              parameters_info,
              n_outputs,
              n_shared_updates):
        """Generates a test for scan.

        :param inputs_info: list of lists of dictionaries
            Each list of dictionary represents one input sequence. Each
            dictionary is one tap of that sequence. The dictionary has two
            keys. ``use`` is either True or False, and it indicates if this
            tap should be used in the inner graph or not. ``tap`` is the tap
            value.
        :param states_info: list of lists of dictionaries
            see param ``inputs_info``. ``states_info`` has the same
            semantics, just that it is for states and not for inputs
        :param paramters_info: list of dictionary
            Each dictionary is a different parameter. It has only one key,
            namely ``use`` which says if the parameter should be used
            internally or not
        :param n_outputs: int
            Number of pure outputs for scan
        :param n_shared_updates: int
            Number of shared variable with updates. They are all numeric.

        """
        # Check the scan node has at least one output
        if n_outputs + n_shared_updates + len(states_info) == 0:
            return

        rng = numpy.random.RandomState(utt.fetch_seed())
        n_ins = len(inputs_info)
        inputs = [tensor.matrix('u%d' % k) for k in xrange(n_ins)]
        scan_inputs = []
        for inp, info in zip(inputs, inputs_info):
            scan_inputs.append(dict(input=inp, taps=[x['tap'] for x in
                                                     info]))
        n_states = len(states_info)
        scan_states = []
        states = []
        for info in states_info:
            if len(info) == 1 and info[0]['tap'] == -1:
                state = tensor.vector('x%d' % k)
                states.append(state)
                scan_states.append(state)
            else:
                state = tensor.matrix('x%d' % k)
                states.append(state)
                scan_states.append(
                    dict(initial=state, taps=[x['tap'] for x in info]))
        n_parameters = len(parameters_info)
        parameters = [tensor.vector('p%d' % k) for k in xrange(n_parameters)]
        original_shared_values = []
        shared_vars = []

        for k in xrange(n_shared_updates):
            data = rng.uniform(size=(4,)).astype(theano.config.floatX)
            original_shared_values.append(data)
            shared_vars.append(theano.shared(data, name='z%d' % k))

        def inner_function(*args):
            """
            Functions that constructs the inner graph of scan
            """
            arg_pos = 0
            to_add = None
            for in_info in inputs_info:
                for info in in_info:
                    arg = args[arg_pos]
                    arg_pos += 1
                    # Construct dummy graph around input
                    if info['use']:
                        if to_add is None:
                            to_add = arg * 2
                        else:
                            to_add = to_add + arg * 2
            states_out = [to_add] * n_states
            for dx, st_info in enumerate(states_info):
                for info in st_info:
                    arg = args[arg_pos]
                    arg_pos += 1
                    if info['use']:
                        states_out[dx] = states_out[dx] + arg * 3
            for info in paramters_info:
                arg = args[arg_pos]
                arg_pos += 1
                if info['use']:
                    if to_add is None:
                        to_add = arg * 4
                    else:
                        to_add = to_add + arg * 4
            shared_outs = [sh * 5 + to_add for sh in shared_vars]
            states_out = [x + to_add for x in states_out]
            pure_outs = [to_add ** 2 for x in xrange(n_outs)]
            return states_out + pure_outs, dict(zip(shared_vars,
                                                    shared_outs))

        def execute_inner_graph(*args):
            """
            Functions that computes numerically the values that scan should
            return
            """
            # Check if you need to go back in time over the sequences (the
            # first argument is n_steps, the second is go_backwards)
            n_steps = args[0]
            invert = False
            if n_steps < 0 or args[1]:
                new_ins = [x[::-1] for x in args[2: 2 + n_ins]]
            n_steps = abs(n_steps)
            # Simplify the inputs by slicing them according to the taps
            nw_inputs = []
            for inp, info in zip(new_ins, inputs_info):
                taps = [x['tap'] for x in info]
                nw_inputs += [inp[abs(numpy.min(taps)) + k:] for k in taps]
            # Simplify the states by slicing them according to the taps.
            # Note that if the memory buffer for the inputs and outputs is
            # the same, by changing the outputs we also change the outputs
            nw_states_inputs = []
            nw_states_outs = []
            for st, info in zip(args[2 + n_ins:2 + n_ins + n_states],
                                states_info):
                taps = [x['tap'] for x in info]
                membuf = numpy.zeros((n_steps + numpy.max(abs(taps)), 4))
                membuf[:numpy.max(abs(taps))] = st[:numpy.max(abs(taps))]
                nw_states_inputs += [membuf[numpy.max(abs(taps)) + k:]
                                     for k in taps]
                nw_states_outs.append(membuf[numpy.max(abs(taps)):])

            paramters = args[2 + n_ins + n_states:]
            out_mem_buffers = [numpy.zeros((n_steps, 4)) for k in n_outs]
            shared_values = [x.copy() for x in original_shared_values]
            for step in xrange(n_steps):
                arg_pos = 0
                to_add = None
                for in_info in inputs_info:
                    for info in in_info:
                        arg = nw_inputs[arg_pos][step]
                        arg_pos += 1
                        # Construct dummy graph around input
                        if info['use']:
                            if to_add is None:
                                to_add = arg * 2
                            else:
                                to_add = to_add + arg * 2
                states_out = [to_add] * n_states
                arg_pos = 0
                for dx, st_info in enumerate(states_info):
                    nw_states_outs[dx][step] = to_add
                    for info in st_info:
                        arg = nw_states_inputs[arg_pos][step]
                        arg_pos += 1
                        if info['use']:
                            nw_states_outs[dx][step] += arg * 3
                for arg, info in zip(parameters, paramters_info):
                    if info['use']:
                        if to_add is None:
                            to_add = arg * 4
                        else:
                            to_add = to_add + arg * 4

                shared_values = [sh * 5 + to_add for sh in shared_values]
                for state in nw_states_outs:
                    state[step] += to_add
                for out in out_mem_buffers:
                    out[step] = to_add ** 2
            return nw_states_outs + out_mem_buffers, shared_values

        for n_steps in [-1, 1, 5, -5, None]:
            for go_backwards in [True, False]:
                outputs, updates = scan_module.scan(
                    inner_function,
                    sequences=scan_inputs,
                    outputs_info=scan_states,
                    non_sequences=parameters,
                    n_steps=n_steps,
                    go_backwards=go_backwards,
                    truncate_gradient=-1)
            my_f = theano.function(inputs + states + parameters,
                                   outputs,
                                   updates=updates,
                                   allow_input_downcast=True)

            if n_steps is not None and abs(n_steps) == 1:
                assert len([x for x in my_f.maker.env.toposort()
                        if isinstance(x.op, scan_module.scan_op.ScanOp)]) == 0
            # Generating data
            # Scenario 1 : Good fit shapes
            inputs_values = []
            for info in inputs_info:
                taps = [x['tap'] for x in info]
                offset = abs(numpy.min([x for x in taps if x < 0]))
                offset += numpy.max([x for x in taps if x > 0])
                data = rng.uniform(size=(n_steps + offset, 4))
                inputs_values.append(data)
            state_values = []
            for info in states_info:
                taps = [x['tap'] for x in info]
                offset = abs(numpy.min(taps))
                data = rng.uniform(size=(offset, 4))
                state_values.append(data)
            param_values = [rng.uniform(size=(4,)) for k in
                            xrange(n_parameters)]
            for var, val in zip(shared_vars, original_shared_values):
                var.set_value(val)
            theano_outs = my_f(*(inputs_values + state_values +
                                 param_values))
            args = ([n_steps, go_backwards] +
                    input_values +
                    state_values +
                    param_values)
            rvals = execute_inner_graph(*args)
            numpy_outs, numpy_shared = rvals
            assert len(numpy_outs) == len(theano_outs)
            assert len(numpy_shared) == len(shared_vars)
            for th_out, num_out in zip(theano_outs, numpy_outs):
                assert numpy.allclose(th_out, num_out)
            for th_out, num_out in zip(shared_outs, numpy_shared):
                assert numpy.allclose(th_out.get_value(), num_out)
            # Scenario 2 : Loose fit (sequences longer then required)
            inputs_values = []
            for pos, info in enumerate(inputs_info):
                taps = [x['tap'] for x in info]
                offset = abs(numpy.min([x for x in taps if x < 0]))
                offset += numpy.max([x for x in taps if x > 0])
                data = rng.uniform(size=(n_steps + offset + pos + 1, 4))
                inputs_values.append(data)
            state_values = []
            for pos, info in enumerate(states_info):
                taps = [x['tap'] for x in info]
                offset = abs(numpy.min(taps))
                data = rng.uniform(size=(offset + pos + 1, 4))
                state_values.append(data)
            param_values = [rng.uniform(size=(4,)) for k in
                            xrange(n_parameters)]
            for var, val in zip(shared_vars, original_shared_values):
                var.set_value(val)
            theano_outs = my_f(*(inputs_values + state_values +
                                 param_values))
            args = ([n_steps, go_backwards] +
                    input_values +
                    state_values +
                    param_values)
            rvals = execute_inner_graph(*args)
            numpy_outs, numpy_shared = rvals
            assert len(numpy_outs) == len(theano_outs)
            assert len(numpy_shared) == len(shared_vars)
            for th_out, num_out in zip(theano_outs, numpy_outs):
                assert numpy.allclose(th_out, num_out)
            for th_out, num_out in zip(shared_outs, numpy_shared):
                assert numpy.allclose(th_out.get_value(), num_out)
            # Scenario 3 : Less data then required
            inputs_values = []
            for pos, info in enumerate(inputs_info):
                taps = [x['tap'] for x in info]
                offset = abs(numpy.min([x for x in taps if x < 0]))
                offset += numpy.max([x for x in taps if x > 0])
                data = rng.uniform(size=(n_steps + offset - 1, 4))
                inputs_values.append(data)
            state_values = []
            for pos, info in enumerate(states_info):
                taps = [x['tap'] for x in info]
                offset = abs(numpy.min(taps))
                data = rng.uniform(size=(offset - 1, 4))
                state_values.append(data)
            param_values = [rng.uniform(size=(4,)) for k in
                            xrange(n_parameters)]
            for var, val in zip(shared_vars, original_shared_values):
                var.set_value(val)
            self.assertRaises(Exception, my_f,
                              inputs + state_values + param_values)

    def test000_generate_tests(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        all_inputs_info = [[]]
        possible_taps_use_pairs = [[dict(tap=0, use=True)],
                                   [dict(tap=0, use=False)],
                                   [dict(tap=-3, use=True),
                                        dict(tap=-1, use=True)],
                                   [dict(tap=-3, use=True),
                                        dict(tap=-1, use=False)],
                                   [dict(tap=-3, use=False),
                                        dict(tap=-1, use=False)],
                                   [dict(tap=-2, use=True),
                                        dict(tap=0, use=True)],
                                   [dict(tap=-2, use=False),
                                        dict(tap=0, use=True)],
                                   [dict(tap=-2, use=False),
                                        dict(tap=0, use=False)],
                                   [dict(tap=0, use=True),
                                        dict(tap=3, use=True)],
                                   [dict(tap=2, use=True),
                                        dict(tap=3, use=True)],
                                   [dict(tap=-2, use=True),
                                        dict(tap=3, use=True)]]
        for n_ins in [1,2]:
            # Randomly pick up 4*n_ins combinations of arguments
            for k in xrange(4*n_ins):
                inp = []
                for inp_nb in xrange(n_ins):
                    pos = rng.randint(len(possible_taps_use_pairs))
                    inp.append(possible_taps_use_pairs[pos])
                all_inputs_info.append(inp)
        all_states_info = [[]]
        possible_taps_use_pairs = [[dict(tap=-1, use=True)],
                                   [dict(tap=-1, use=False)],
                                   [dict(tap=-3, use=True)],
                                   [dict(tap=-3, use=False)],
                                   [dict(tap=-3, use=True),
                                        dict(tap=-1, use=True)],
                                   [dict(tap=-3, use=True),
                                        dict(tap=-1, use=False)],
                                   [dict(tap=-3, use=False),
                                        dict(tap=-1, use=False)],
                                   [dict(tap=-4, use=True),
                                        dict(tap=-2, use=True)],
                                   [dict(tap=-4, use=False),
                                        dict(tap=-2, use=True)]]
        for n_ins in [1,2]:
            # Randomly pick up 4*n_ins combinations of arguments
            for k in xrange(4*n_ins):
                state = []
                for state_nb in xrange(n_ins):
                    pos = rng.randint(len(possible_taps_use_pairs))
                    state.append(possible_taps_use_pairs[pos])
                all_states_info.append(state)
        all_parameters_info = [[],
                           [dict(use=False)],
                           [dict(use=True)],
                           [dict(use=True), dict(use=True)],
                           [dict(use=True), dict(use=False)]]

        for n_outputs in [0,1,2]:
            for n_shared_updates in [0,1,2]:
                for n_random_combinations in xrange(14):
                    pos_inp = rng.randint(len(all_inputs_info))
                    pos_st = rng.randint(len(all_states_info))
                    pos_param = rng.randint(len(all_parameters_info))
                    self.new_run(inputs_info=all_inputs_info[pos_inp],
                             states_info=all_states_info[pos_st],
                             parameters_info=all_parameters_info[pos_param],
                             n_outputs=n_outputs,
                             n_shared_updates=n_shared_updates)



    def test001_generator_one_scalar_output(self):
        def f_pow2(x_tm1):
            return 2 * x_tm1

        for n_steps in [-1, 1, 5, -5]:
            state = theano.tensor.scalar('state')
            output, updates = scan_module.scan(f_pow2,
                                               [],
                                               state,
                                               [],
                                               n_steps=n_steps,
                                               truncate_gradient=-1,
                                               go_backwards=False)
            my_f = theano.function([state],
                                   output,
                                   updates=updates,
                                   allow_input_downcast=True)
            if abs(n_steps) == 1:
                assert len([x for x in my_f.maker.env.toposort()
                        if isinstance(x.op, scan_module.scan_op.ScanOp)]) == 0

            rng = numpy.random.RandomState(utt.fetch_seed())
            state = rng.uniform()
            numpy_values = numpy.array([state * (2 ** (k + 1)) for k
                                        in xrange(abs(n_steps))])
            theano_values = my_f(state)
            assert numpy.allclose(numpy_values, theano_values)

    # simple rnn, one input, one state, weights for each; input/state
    # are vectors, weights are scalars
    def test002_one_sequence_one_output_and_weights(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W
        u = theano.tensor.vector('u')
        x0 = theano.tensor.scalar('x0')
        W_in = theano.tensor.scalar('win')
        W = theano.tensor.scalar('w')
        output, updates = scan_module.scan(f_rnn,
                                      u,
                                      x0,
                                      [W_in, W],
                                      n_steps=n_steps,
                                      truncate_gradient=-1,
                                      go_backwards=False)

        my_f = theano.function([u, x0, W_in, W],
                               output,
                               updates=updates,
                               allow_input_downcast=True)
        if n_steps is not None and abs(n_steps) == 1:
            assert len([x for x in my_f.maker.env.toposort()
                    if isinstance(x.op, scan_module.scan_op.ScanOp)]) == 0
        # get random initial values
        rng = numpy.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(size=(8,), low=-5., high=5.)
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        if n_steps is not None and n_steps < 0:
            _v_u = v_u[::-1]
        else:
            _v_u = v_u
        steps = 8
        if n_steps is not None:
            steps = abs(n_steps)

        v_out = numpy.zeros((8,))
        v_out[0] = _v_u[0] * W_in + v_x0 * W

        for step in xrange(1, steps):
            v_out[step] = _v_u[step] * W_in + v_out[step - 1] * W
        v_out = v_out[:steps]
        theano_values = my_f(v_u, v_x0, W_in, W)
        assert numpy.allclose(theano_values, v_out)

    def test003_multiple_inputs_multiple_outputs(self):
        pass

    def test004_collect_parameters_outer_graph(self):
        pass

    def test005_multiple_taps(self):
        pass

    def test006_updates(self):
        pass
