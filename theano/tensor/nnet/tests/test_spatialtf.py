from __future__ import absolute_import, print_function, division

from nose.plugins.skip import SkipTest
import numpy as np
import theano
import theano.ifelse
from theano import tensor as T
from theano import config
from theano.tests import unittest_tools as utt
from theano.tensor.nnet.spatialtf import spatialtf, TransformerGrid, TransformerSampler, TransformerGradI, \
    TransformerGradT
from theano.tensor.nnet.symbolic_spatialtf import theano_spatialtf, theano_spatialtf_grid, theano_spatialtf_from_grid


def assert_raises(exception_classes, callable, *args):
    try:
        callable(*args)
    except Exception as e:
        if not isinstance(e, tuple(exception_classes)):
            raise e
    else:
        raise AssertionError('assert_raises', exception_classes, callable.__name__, *args)


valid_gradients = DOUT_DINPUT, DOUT_DTHETA, DOUT_DGRID, DGRID_DTHETA = (
    'out_input', 'out_theta', 'out_grid', 'grid_theta')

symbolic_functions = {}


def get_symb_and_cpu_functions(mode):
    # Returns a couple of theano functions:
    # (function with NumPy implementation, function with symbolic (Theano variables) implementation).

    global symbolic_functions
    fn_key = (mode, None)
    if fn_key in symbolic_functions:
        return symbolic_functions[fn_key]

    t_inp = T.tensor4('inp')
    t_theta = T.tensor3('theta')
    t_scale_height = T.scalar('scale_height')
    t_scale_width = T.scalar('scalar_width')

    symb_out_var = theano_spatialtf(t_inp, t_theta, t_scale_width, t_scale_height)
    cpu_out_var = spatialtf(t_inp, t_theta, t_scale_width, t_scale_height)
    symb_out_fn = theano.function([t_inp, t_theta, t_scale_width, t_scale_height], symb_out_var,
                                  mode=mode)
    cpu_out_fn = theano.function([t_inp, t_theta, t_scale_width, t_scale_height], cpu_out_var, mode=mode)

    functions = (cpu_out_fn, symb_out_fn)
    symbolic_functions[fn_key] = functions
    return functions


def get_symb_and_cpu_grad_functions(mode, for_grad, grid_op, sampler_op):
    global symbolic_functions
    fn_key = (mode, for_grad)
    if fn_key in symbolic_functions:
        return symbolic_functions[fn_key]

    assert for_grad in valid_gradients

    t_inp = T.tensor4('inp')
    t_theta = T.tensor3('theta')
    t_scale_height = T.scalar('scale_height')
    t_scale_width = T.scalar('scalar_width')

    def get_dout_dgrid(sampler_callable, inp, grid):
        out = sampler_callable(inp, grid)
        scalar = T.sum(out)
        return T.grad(scalar, grid)

    def get_dgrid_dtheta(grid, theta):
        scalar = T.sum(grid)
        return T.grad(scalar, theta)

    def get_grad_scalar(sampler_callable, inp, theta, scale_width, scale_height, wrt):
        out = sampler_callable(inp, theta, scale_width, scale_height)
        scalar = T.sum(out)
        return T.grad(scalar, wrt)

    if 'grid' in for_grad:
        out_height = T.ceil(t_inp.shape[2] * t_scale_height)
        out_width = T.ceil(t_inp.shape[3] * t_scale_width)
        t_out_dims = [T.cast(v, 'int64') for v in (t_inp.shape[0], t_inp.shape[1], out_height, out_width)]

        t_grid = grid_op()(t_theta, t_out_dims)
        t_symb_grid = theano_spatialtf_grid(t_theta, t_out_dims)

        if for_grad == DOUT_DGRID:
            t_grad_scalar = get_dout_dgrid(sampler_op(), t_inp, t_grid)
            t_symb_grad_scalar = get_dout_dgrid(theano_spatialtf_from_grid, t_inp, t_symb_grid)
        elif for_grad == DGRID_DTHETA:
            t_grad_scalar = get_dgrid_dtheta(t_grid, t_theta)
            t_symb_grad_scalar = get_dgrid_dtheta(t_symb_grid, t_theta)
        else:
            raise ValueError('Cannot create spatialf grid-based grad functions for', for_grad)

    else:
        if for_grad == DOUT_DINPUT:
            t_wrt = t_inp
        elif for_grad == DOUT_DTHETA:
            t_wrt = t_theta
        else:
            raise ValueError('Cannot create spatialtf gradient functions for', for_grad)

        t_grad_scalar = get_grad_scalar(spatialtf, t_inp, t_theta, t_scale_width, t_scale_height, t_wrt)
        t_symb_grad_scalar = get_grad_scalar(theano_spatialtf, t_inp, t_theta, t_scale_width, t_scale_height, t_wrt)

    fn_symb = theano.function([t_inp, t_theta, t_scale_width, t_scale_height], t_symb_grad_scalar, mode=mode)
    fn_cpu = theano.function([t_inp, t_theta, t_scale_width, t_scale_height], t_grad_scalar, mode=mode)

    functions = (fn_cpu, fn_symb)
    symbolic_functions[fn_key] = functions
    return functions


class TestTransformer(object):
    mode = None
    n_tests = 10
    transformer_grid_op = TransformerGrid
    transformer_sampler_op = TransformerSampler
    transformer_grad_i_op = TransformerGradI
    transformer_grad_t_op = TransformerGradT

    some_transformations = [
        [[1, 0, 0],
         [0, 1, 0]],
        [[-1, 0, 0],
         [0, -1, 0]],
        [[1, 0, -1],
         [0, 1, -1]],
        [[1, 1, 1],
         [1, 1, 1]],
        [[-1.90120868, 1.48872078, -4.01530816],
         [8.27449531, 1.75634363, 6.66776181]]
    ]
    cases_count = 15
    inp_cases = []
    transform_cases = []

    def __init__(self):
        utt.seed_rng()
        self.inp_cases = self._get_inp_shape_cases(self.cases_count)
        self.transform_cases = self._get_transform_cases(self.cases_count - len(self.some_transformations))

    def setUp(self):
        utt.seed_rng()

    def _skip_dgrid(self, msg=''):
        if msg:
            msg = ' ' + msg
        raise SkipTest('CPU spatial transformation gradient wrt grid '
                       'is not currently well implemented.' + msg)

    def _get_inp_shape_cases(self, count):
        return [self._generate_inp_shape() for i in range(count)]

    def _get_transform_cases(self, count):
        return [np.asarray(t) for t in self.some_transformations] + [self._generate_transformation() for i in
                                                                     range(count)]

    def _generate_inp_shape(self):
        return (
            np.random.randint(1, 11), np.random.randint(1, 6), np.random.randint(10, 101), np.random.randint(10, 101))

    def _generate_transformation(self):
        return np.random.uniform(-10, 10, (2, 3)).astype(theano.config.floatX)

    def check_cpu_function(self, cpu_out_fn):
        count_grid_ops = 0
        count_sampler_ops = 0
        for node in cpu_out_fn.maker.fgraph.apply_nodes:
            if isinstance(node.op, self.transformer_grid_op):
                count_grid_ops += 1
            elif isinstance(node.op, self.transformer_sampler_op):
                count_sampler_ops += 1
        assert count_grid_ops == 1, count_grid_ops
        assert count_sampler_ops == 1, count_sampler_ops

    def check_cpu_grad_function(self, cpu_out_fn, grad_name):
        count_ops = 0
        if grad_name in (DOUT_DINPUT, DOUT_DGRID):
            look_for = self.transformer_grad_i_op
        elif grad_name in (DOUT_DTHETA, DGRID_DTHETA):
            look_for = self.transformer_grad_t_op
        else:
            raise ValueError('Cannot check CPU graph for unknown gradient', grad_name)
        for node in cpu_out_fn.maker.fgraph.apply_nodes:
            if isinstance(node.op, look_for):
                count_ops += 1
        assert count_ops == 1, count_ops

    def getInputs(self, inp_shape, transform):
        inp = np.random.random(inp_shape).astype(config.floatX)
        theta = np.asarray(inp_shape[0] * [transform], dtype=config.floatX)
        scale_height = np.random.random()
        scale_width = np.random.random()
        return (inp, theta, scale_width, scale_height)

    def get_out_dims(self, inp_shape, scale_width, scale_height):
        return tuple(int(v) for v in (inp_shape[0], inp_shape[1],
                                      np.ceil(inp_shape[2] * scale_height),
                                      np.ceil(inp_shape[3] * scale_width)))

    def compare_symbolic_vs_numpy(self, case_index, for_grad=None):
        # Compare CPU implementation with symbolic implementation.
        inp_shape = self.inp_cases[case_index]
        transform = self.transform_cases[case_index]
        inp_shape = (1, 2, 3, 4)
        transform = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=theano.config.floatX)  # + 1e-8
        inp, theta, scale_width, scale_height = self.getInputs(inp_shape, transform)

        if for_grad is None:
            fn_cpu, fn_symb = get_symb_and_cpu_functions(self.mode)
            self.check_cpu_function(fn_cpu)
        else:
            fn_cpu, fn_symb = get_symb_and_cpu_grad_functions(self.mode, for_grad, self.transformer_grid_op,
                                                              self.transformer_sampler_op)
            self.check_cpu_grad_function(fn_cpu, for_grad)

        symb_out = fn_symb(inp, theta, scale_width, scale_height)
        cpu_out = fn_cpu(inp, theta, scale_width, scale_height)

        try:
            utt.assert_allclose(cpu_out, symb_out)
        except Exception as e:
            more_exception_infos = """
Failing case %d (grad: %s):
Input shape: %s
Transform: %s
""" % (case_index, for_grad, str(inp_shape), str(transform.flatten()))
            raise Exception(more_exception_infos + str(e))

    def compare_symbolic_vs_numpy_grad_dout_dtheta(self, case_index):
        self.compare_symbolic_vs_numpy(case_index, DOUT_DTHETA)

    def compare_symbolic_vs_numpy_grad_dout_dinput(self, case_index):
        self.compare_symbolic_vs_numpy(case_index, DOUT_DINPUT)

    def compare_symbolic_vs_numpy_grad_dout_dgrid(self, case_index):
        self.compare_symbolic_vs_numpy(case_index, DOUT_DGRID)

    def compare_symbolic_vs_numpy_grad_dgrid_dtheta(self, case_index):
        self.compare_symbolic_vs_numpy(case_index, DGRID_DTHETA)

    def test_symbolic_vs_numpy_grad_dout_dtheta(self):
        self._skip_dgrid('dout/dtheta depends on dout/dgrid.')
        for test_case_index in range(len(self.inp_cases)):
            yield (self.compare_symbolic_vs_numpy_grad_dout_dtheta, test_case_index)

    def test_symbolic_vs_numpy_grad_dout_dinput(self):
        for test_case_index in range(len(self.inp_cases)):
            yield (self.compare_symbolic_vs_numpy_grad_dout_dinput, test_case_index)

    def test_symbolic_vs_numpy_grad_dout_dgrid(self):
        self._skip_dgrid()
        for test_case_index in range(len(self.inp_cases)):
            yield (self.compare_symbolic_vs_numpy_grad_dout_dgrid, test_case_index)

    def test_symbolic_vs_numpy_grad_dgrid_dtheta(self):
        for test_case_index in range(len(self.inp_cases)):
            yield (self.compare_symbolic_vs_numpy_grad_dgrid_dtheta, test_case_index)

    def test_symbolic_vs_numpy(self):
        for test_case_index in range(len(self.inp_cases)):
            yield (self.compare_symbolic_vs_numpy, test_case_index)

    def test_invalid_shapes(self):
        inputs = T.tensor4('inputs')
        theta = T.tensor3('theta')

        st = spatialtf(inputs, theta)
        st_func = theano.function([inputs, theta], st, mode=self.mode)

        inputs_val = np.ones((3, 5, 7, 7), dtype=theano.config.floatX)

        def try_theta_shp(theta_shp):
            theta_val = np.ones(theta_shp, dtype=theano.config.floatX)
            return st_func(inputs_val, theta_val)

        # the theta shape for this input should be (3, 2, 3)
        try_theta_shp((3, 2, 3))

        # incorrect parameter dimensions
        assert_raises([ValueError, RuntimeError], try_theta_shp, (3, 1, 3))
        assert_raises([ValueError, RuntimeError], try_theta_shp, (3, 2, 1))

        # number of rows does not match the number of input rows
        assert_raises([ValueError, RuntimeError], try_theta_shp, (1, 2, 3))
        assert_raises([ValueError, RuntimeError], try_theta_shp, (4, 2, 3))
