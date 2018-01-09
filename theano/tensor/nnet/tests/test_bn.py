from __future__ import absolute_import, print_function, division
import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
import numpy as np
from collections import OrderedDict

from theano.tensor.nnet import bn


def test_BNComposite():
    try:
        orig = theano.config.compute_test_value

        theano.config.compute_test_value = 'raise'

        def bn_ref(x, G, B, M, V):
            n = (x - M) / V
            return n * G + B

        np.random.seed(1234)
        X = 1 + np.random.random([10, 20]).astype('float32')
        B = 1 + np.random.random([20]).astype('float32')
        G = 1 + np.random.random([20]).astype('float32')
        M = 1 + np.random.random([20]).astype('float32')
        V = 1 + np.random.random([20]).astype('float32')

        x = theano.tensor.matrix('x')
        b = theano.tensor.vector('b')
        g = theano.tensor.vector('g')
        m = theano.tensor.vector('m')
        v = theano.tensor.vector('v')

        x.tag.test_value = np.random.rand(2, 2).astype(theano.config.floatX)
        b.tag.test_value = np.random.rand(2).astype(theano.config.floatX)
        g.tag.test_value = np.random.rand(2).astype(theano.config.floatX)
        m.tag.test_value = np.random.rand(2).astype(theano.config.floatX)
        v.tag.test_value = np.random.rand(2).astype(theano.config.floatX)

        bn_ref_op = bn_ref(x, g, b, m, v)
        f_ref = theano.function([x, b, g, m, v], [bn_ref_op])
        res_ref = f_ref(X, G, B, M, V)
        for mode in ['low_mem', 'high_mem']:
            bn_op = bn.batch_normalization(x, g, b, m, v, mode=mode)
            f = theano.function([x, b, g, m, v], [bn_op])
            res = f(X, G, B, M, V)
            utt.assert_allclose(res_ref, res)
    finally:
        theano.config.compute_test_value = orig


def test_batch_normalization():

    def bn_ref(x, G, B, M, V):
        n = (x - M) / V
        return n * G + B

    np.random.seed(1234)
    X = 1 + np.random.random([10, 20]).astype('float32')
    B = 1 + np.random.random([20]).astype('float32')
    G = 1 + np.random.random([20]).astype('float32')
    M = 1 + np.random.random([20]).astype('float32')
    V = 1 + np.random.random([20]).astype('float32')

    x = theano.tensor.matrix('x')
    b = theano.tensor.vector('b')
    g = theano.tensor.vector('g')
    m = theano.tensor.vector('m')
    v = theano.tensor.vector('v')

    bn_ref_op = bn_ref(x, g, b, m, v)
    f_ref = theano.function([x, g, b, m, v], [bn_ref_op])
    res_ref = f_ref(X, G, B, M, V)
    for mode in ['low_mem', 'high_mem']:
        bn_op = bn.batch_normalization(x, g, b, m, v, mode=mode)
        f = theano.function([x, g, b, m, v], [bn_op])
        res = f(X, G, B, M, V)
        utt.assert_allclose(res_ref, res)

        def bn_f(inputs, gamma, beta, mean, std):
            return bn.batch_normalization(inputs, gamma, beta, mean, std, mode=mode)
        utt.verify_grad(bn_f, [X, G, B, M, V])

    bn_ref_op = bn_ref(x, g, b, x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True))
    f_ref = theano.function([x, b, g], [bn_ref_op])
    res_ref = f_ref(X, G, B)
    for mode in ['low_mem', 'high_mem']:
        bn_op = bn.batch_normalization(x, g, b, x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True), mode=mode)
        f = theano.function([x, b, g], [bn_op])
        res = f(X, G, B)
        utt.assert_allclose(res_ref, res)

        def bn_f(inputs, gamma, beta, mean, std):
            return bn.batch_normalization(inputs, gamma, beta, mean, std, mode=mode)
        utt.verify_grad(bn_f, [X, G, B,
                               X.mean(axis=0)[np.newaxis], X.std(axis=0)[np.newaxis]])


def test_bn_feature_maps():

    def bn_ref(x, G, B, M, V):
        n = (x - M) / V
        return n * G + B

    np.random.seed(1234)
    X = 1 + np.random.random([2, 3, 4, 4]).astype('float32')
    B = 1 + np.random.random([3]).astype('float32')
    G = 1 + np.random.random([3]).astype('float32')
    M = 1 + np.random.random([3]).astype('float32')
    V = 1 + np.random.random([3]).astype('float32')

    x = theano.tensor.tensor4('x')
    b = theano.tensor.vector('b')
    g = theano.tensor.vector('g')
    m = theano.tensor.vector('m')
    v = theano.tensor.vector('v')

    bn_ref_op = bn_ref(x,
                       g.dimshuffle('x', 0, 'x', 'x'),
                       b.dimshuffle('x', 0, 'x', 'x'),
                       m.dimshuffle('x', 0, 'x', 'x'),
                       v.dimshuffle('x', 0, 'x', 'x'))
    f_ref = theano.function([x, b, g, m, v], [bn_ref_op])
    res_ref = f_ref(X, G, B, M, V)

    for mode in ['low_mem', 'high_mem']:
        bn_op = bn.batch_normalization(x,
                                       g.dimshuffle('x', 0, 'x', 'x'),
                                       b.dimshuffle('x', 0, 'x', 'x'),
                                       m.dimshuffle('x', 0, 'x', 'x'),
                                       v.dimshuffle('x', 0, 'x', 'x'),
                                       mode=mode)
        f = theano.function([x, b, g, m, v], [bn_op])
        res = f(X, G, B, M, V)
        utt.assert_allclose(res_ref, res)

        def conv_bn(inputs, gamma, beta, mean, std):
            return bn.batch_normalization(inputs,
                                          gamma.dimshuffle('x', 0, 'x', 'x'),
                                          beta.dimshuffle('x', 0, 'x', 'x'),
                                          mean.dimshuffle('x', 0, 'x', 'x'),
                                          std.dimshuffle('x', 0, 'x', 'x'),
                                          mode=mode)
        utt.verify_grad(conv_bn, [X, G, B, M, V])


def test_batch_normalization_train():
    utt.seed_rng()

    for axes in ('per-activation', 'spatial', (1, 2, 3, 4)):
        for vartype in (T.tensor5, T.tensor3, T.vector):
            x, scale, bias, running_mean, running_var = (vartype(n)
                                                         for n in ('x', 'scale', 'bias',
                                                                   'running_mean',
                                                                   'running_var'))
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used
            running_average_factor = 0.3

            # remove non-existing axes
            if isinstance(axes, tuple):
                axes = tuple(i for i in axes if i < ndim)
            if len(axes) == 0:
                continue

            # forward pass
            out, x_mean, x_invstd, out_running_mean, out_running_var = \
                bn.batch_normalization_train(
                    x, scale, bias, axes, eps,
                    running_average_factor, running_mean, running_var)
            # reference forward pass
            if axes == 'per-activation':
                axes2 = (0,)
            elif axes == 'spatial':
                axes2 = (0,) + tuple(range(2, ndim))
            else:
                axes2 = axes
            x_mean2 = x.mean(axis=axes2, keepdims=True)
            x_var2 = x.var(axis=axes2, keepdims=True)
            x_invstd2 = T.inv(T.sqrt(x_var2 + eps))
            scale2 = T.addbroadcast(scale, *axes2)
            bias2 = T.addbroadcast(bias, *axes2)
            out2 = (x - x_mean2) * (scale2 * x_invstd2) + bias2
            m = T.cast(T.prod(x.shape) / T.prod(scale.shape), theano.config.floatX)
            out_running_mean2 = running_mean * (1 - running_average_factor) + \
                x_mean2 * running_average_factor
            out_running_var2 = running_var * (1 - running_average_factor) + \
                (m / (m - 1)) * x_var2 * running_average_factor
            # backward pass
            dy = vartype('dy')
            grads = T.grad(None, wrt=[x, scale, bias], known_grads={out: dy})
            # reference backward pass
            grads2 = T.grad(None, wrt=[x, scale, bias], known_grads={out2: dy})
            # second-order backward pass
            dx = vartype('dinputs')
            dscale = vartype('dscale')
            dbias = vartype('dbias')
            grad_grads = T.grad(None, wrt=[x, dy, scale], known_grads=OrderedDict(
                {grads[0]: dx, grads[1]: dscale, grads[2]: dbias}),
                consider_constant=[x, dy, scale, bias, x_mean, x_invstd, running_mean, running_var],
                return_disconnected='zero')
            # reference second-order backward pass
            grad_grads2 = T.grad(None, wrt=[x, dy, scale], known_grads=OrderedDict(
                {grads2[0]: dx, grads2[1]: dscale, grads2[2]: dbias}),
                consider_constant=[x, dy, scale, bias, x_mean2, x_var2, running_mean, running_var],
                return_disconnected='zero')
            # compile
            f = theano.function([x, scale, bias, running_mean, running_var, dy, dx, dscale, dbias],
                                [out, x_mean, x_invstd, out_running_mean, out_running_var,
                                 out2, x_mean2, x_invstd2, out_running_mean2, out_running_var2] +
                                grads + grads2 + grad_grads + grad_grads2)
            # check if the abstract Ops have been replaced
            assert not any([isinstance(n.op, (bn.AbstractBatchNormTrain,
                                              bn.AbstractBatchNormInference,
                                              bn.AbstractBatchNormTrainGrad))
                            for n in f.maker.fgraph.toposort()])
            # run
            for data_shape in ((5, 10, 30, 40, 10), (4, 3, 1, 1, 1), (2, 3, 5, 5, 5)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(1 if d in axes2 else s
                                    for d, s in enumerate(data_shape))
                X = 4 + 3 * np.random.randn(*data_shape).astype(theano.config.floatX)
                Dy = -1 + 2 * np.random.randn(*data_shape).astype(theano.config.floatX)
                Scale = np.random.randn(*param_shape).astype(theano.config.floatX)
                Bias = np.random.randn(*param_shape).astype(theano.config.floatX)
                Running_mean = np.random.randn(*param_shape).astype(theano.config.floatX)
                Running_var = np.random.randn(*param_shape).astype(theano.config.floatX)
                Dx = 4 + 3 * np.random.randn(*data_shape).astype(theano.config.floatX)
                Dscale = -1 + 2 * np.random.randn(*param_shape).astype(theano.config.floatX)
                Dbias = np.random.randn(*param_shape).astype(theano.config.floatX)

                outputs = f(X, Scale, Bias, Running_mean, Running_var, Dy, Dx, Dscale, Dbias)
                # compare outputs
                utt.assert_allclose(outputs[0], outputs[0 + 5])  # out
                utt.assert_allclose(outputs[1], outputs[1 + 5])  # mean
                utt.assert_allclose(outputs[2], outputs[2 + 5])  # invstd
                utt.assert_allclose(outputs[3], outputs[3 + 5])  # running_mean
                utt.assert_allclose(np.nan_to_num(outputs[4]),
                                    np.nan_to_num(outputs[4 + 5]))  # running_var
                # compare gradients
                utt.assert_allclose(outputs[10], outputs[10 + 3], atol=1e-4)  # dx
                utt.assert_allclose(outputs[11], outputs[11 + 3], rtol=2e-4, atol=1e-4)  # dscale
                utt.assert_allclose(outputs[12], outputs[12 + 3])  # dbias
                # compare second-order gradients
                utt.assert_allclose(outputs[16], outputs[16 + 3], atol=1e-4)  # ddx
                utt.assert_allclose(outputs[17], outputs[17 + 3])  # ddy
                utt.assert_allclose(outputs[18], outputs[18 + 3], rtol=3e-4, atol=1e-4)  # ddscale


def test_batch_normalization_train_grad_grad():
    utt.seed_rng()

    for axes in ('per-activation', 'spatial', (1, 2, 3, 4)):
        for vartype in (T.tensor5, T.tensor4, T.tensor3, T.matrix, T.vector):
            # run these experiments with float64 for sufficient numerical stability
            x, dy, scale, x_mean, x_invstd = (vartype(n, dtype='float64')
                                              for n in ('x', 'dy', 'scale',
                                                        'x_mean', 'x_invstd'))
            ndim = x.ndim

            # reference forward pass
            if axes == 'per-activation':
                axes = (0,)
            elif axes == 'spatial':
                axes = (0,) + tuple(range(2, ndim))
            else:
                # remove non-existing axes
                axes = tuple(i for i in axes if i < ndim)
            if len(axes) == 0:
                continue

            def bn_grad_wrt_inputs_f(x, dy, scale, x_mean, x_invstd):
                g_inputs, g_scale, g_bias = bn.AbstractBatchNormTrainGrad(axes)(x, dy, scale, x_mean, x_invstd)
                return g_inputs

            def bn_grad_wrt_scale_f(x, dy, scale, x_mean, x_invstd):
                g_inputs, g_scale, g_bias = bn.AbstractBatchNormTrainGrad(axes)(x, dy, scale, x_mean, x_invstd)
                return g_scale

            def bn_grad_wrt_bias_f(x, dy, scale, x_mean, x_invstd):
                g_inputs, g_scale, g_bias = bn.AbstractBatchNormTrainGrad(axes)(x, dy, scale, x_mean, x_invstd)
                return g_bias

            # run
            for data_shape in ((4, 3, 3, 3, 3), (4, 3, 1, 1, 1), (2, 3, 5, 3, 2)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(1 if d in axes else s
                                    for d, s in enumerate(data_shape))
                # force float64 for sufficient numerical stability
                x_val = 4 + 3 * np.random.randn(*data_shape).astype('float64')
                dy_val = -1 + 2 * np.random.randn(*data_shape).astype('float64')
                scale_val = np.random.randn(*param_shape).astype('float64')
                x_mean_val = np.random.randn(*param_shape).astype('float64')
                x_invstd_val = np.random.randn(*param_shape).astype('float64')

                utt.verify_grad(bn_grad_wrt_inputs_f, [x_val, dy_val, scale_val, x_mean_val, x_invstd_val], abs_tol=5e-4, rel_tol=5e-4)
                utt.verify_grad(bn_grad_wrt_scale_f, [x_val, dy_val, scale_val, x_mean_val, x_invstd_val])
                utt.verify_grad(bn_grad_wrt_bias_f, [x_val, dy_val, scale_val, x_mean_val, x_invstd_val])


def test_batch_normalization_train_without_running_averages():
    # compile and run batch_normalization_train without running averages
    utt.seed_rng()

    x, scale, bias, dy = T.tensor4('x'), T.tensor4('scale'), T.tensor4('bias'), T.tensor4('dy')
    data_shape = (5, 10, 30, 25)
    param_shape = (1, 10, 30, 25)

    # forward pass
    out, x_mean, x_invstd = bn.batch_normalization_train(x, scale, bias, 'per-activation')
    # backward pass
    grads = T.grad(None, wrt=[x, scale, bias], known_grads={out: dy})
    # compile
    f = theano.function([x, scale, bias, dy], [out, x_mean, x_invstd] + grads)
    # check if the abstract Ops have been replaced
    assert not any([isinstance(n.op, (bn.AbstractBatchNormTrain,
                                      bn.AbstractBatchNormInference,
                                      bn.AbstractBatchNormTrainGrad))
                    for n in f.maker.fgraph.toposort()])
    # run
    X = 4 + 3 * np.random.randn(*data_shape).astype(theano.config.floatX)
    Dy = -1 + 2 * np.random.randn(*data_shape).astype(theano.config.floatX)
    Scale = np.random.randn(*param_shape).astype(theano.config.floatX)
    Bias = np.random.randn(*param_shape).astype(theano.config.floatX)
    f(X, Scale, Bias, Dy)


def test_batch_normalization_train_broadcast():
    for axes in ('per-activation', 'spatial', (1, 2, 3, 4)):
        for vartype in (T.tensor5, T.tensor4, T.tensor3, T.matrix, T.vector):
            x = vartype('x')
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used
            running_average_factor = 0.3

            # remove non-existing axes
            if isinstance(axes, tuple):
                axes = tuple(i for i in axes if i < ndim)
            if len(axes) == 0:
                continue

            # convert axes to explicit list
            if axes == 'per-activation':
                axes2 = (0,)
            elif axes == 'spatial':
                axes2 = (0,) + tuple(range(2, ndim))
            else:
                axes2 = axes

            # compute axes for parameter tensors
            non_bc_axes = tuple(i for i in range(ndim) if i not in axes2)
            params_dimshuffle = ['x'] * ndim
            for i, axis in enumerate(non_bc_axes):
                params_dimshuffle[axis] = i

            # construct non-broadcasted parameter variables
            param_type = T.TensorType(x.dtype, (False,) * len(non_bc_axes))
            scale, bias, running_mean, running_var = (param_type(n)
                                                      for n in ('scale', 'bias',
                                                                'running_mean',
                                                                'running_var'))

            # broadcast parameter variables
            scale_bc = scale.dimshuffle(params_dimshuffle)
            bias_bc = bias.dimshuffle(params_dimshuffle)
            running_mean_bc = running_mean.dimshuffle(params_dimshuffle)
            running_var_bc = running_var.dimshuffle(params_dimshuffle)

            # batch_normalization_train with original, non-broadcasted variables
            train_non_bc = \
                bn.batch_normalization_train(
                    x, scale, bias, axes, eps,
                    running_average_factor, running_mean, running_var)
            # batch_normalization_train with broadcasted variables
            train_bc = \
                bn.batch_normalization_train(
                    x, scale_bc, bias_bc, axes, eps,
                    running_average_factor, running_mean_bc, running_var_bc)
            train_bc = tuple([train_bc[0]] +  # out
                             [r.dimshuffle(non_bc_axes) for r in train_bc[1:]])

            # batch_normalization_test with original, non-broadcasted variables
            test_non_bc = \
                bn.batch_normalization_test(
                    x, scale, bias, running_mean, running_var, axes, eps)
            # batch_normalization_test with broadcasted variables
            test_bc = \
                bn.batch_normalization_test(
                    x, scale_bc, bias_bc, running_mean_bc, running_var_bc, axes, eps)

            # subtract the results of the non-broadcasted and broadcasted calls
            results_non_bc = train_non_bc + (test_non_bc,)
            results_bc = train_bc + (test_bc,)
            results = [abs(r - r_bc) for (r, r_bc) in zip(results_non_bc, results_bc)]

            # compile to compute all differences
            f = theano.function([x, scale, bias, running_mean, running_var],
                                T.sum(sum(results)))

            # the paired ops are exactly the same, so the optimizer should have
            # collapsed the sum of differences to a constant zero
            nodes = f.maker.fgraph.toposort()
            if theano.config.mode != "FAST_COMPILE":
                assert len(nodes) == 1
                assert isinstance(nodes[0].op, theano.compile.DeepCopyOp)
            inputs = [np.asarray(np.random.rand(*((4,) * n)), x.dtype)
                      for n in [x.ndim, scale.ndim, bias.ndim,
                                running_mean.ndim, running_var.ndim]]
            assert 0.0 == f(*inputs)


def test_batch_normalization_test():
    for axes in ('per-activation', 'spatial', (1, 2, 3, 4)):
        for vartype in (T.tensor5, T.tensor3, T.vector):
            x, scale, bias, mean, var = (vartype(n)
                                         for n in ('x', 'scale', 'bias', 'mean', 'var'))
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used

            # remove non-existing axes
            if isinstance(axes, tuple):
                axes = tuple(i for i in axes if i < ndim)
            if len(axes) == 0:
                continue

            # forward pass
            out = bn.batch_normalization_test(x, scale, bias, mean,
                                              var, axes, eps)
            # reference forward pass
            if axes == 'per-activation':
                axes2 = (0,)
            elif axes == 'spatial':
                axes2 = (0,) + tuple(range(2, ndim))
            else:
                axes2 = axes
            scale2, bias2, mean2, var2 = (T.addbroadcast(t, *axes2)
                                          for t in (scale, bias, mean, var))
            out2 = (x - mean2) * (scale2 / T.sqrt(var2 + eps)) + bias2
            # backward pass
            dy = vartype('dy')
            grads = T.grad(None, wrt=[x, scale, bias, mean, var], known_grads={out: dy})
            # reference backward pass
            grads2 = T.grad(None, wrt=[x, scale, bias, mean, var], known_grads={out2: dy})
            # compile
            f = theano.function([x, scale, bias, mean, var, dy],
                                [out, out2] + grads + grads2)
            # check if the abstract Ops have been replaced
            assert not any([isinstance(n.op, (bn.AbstractBatchNormTrain,
                                              bn.AbstractBatchNormInference,
                                              bn.AbstractBatchNormTrainGrad))
                            for n in f.maker.fgraph.toposort()])
            # run
            for data_shape in ((10, 20, 30, 40, 10), (4, 3, 1, 1, 1), (1, 1, 5, 5, 5)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(1 if d in axes2 else s
                                    for d, s in enumerate(data_shape))
                X = 4 + 3 * np.random.randn(*data_shape).astype(theano.config.floatX)
                Dy = -1 + 2 * np.random.randn(*data_shape).astype(theano.config.floatX)
                Scale = np.random.randn(*param_shape).astype(theano.config.floatX)
                Bias = np.random.randn(*param_shape).astype(theano.config.floatX)
                Mean = np.random.randn(*param_shape).astype(theano.config.floatX)
                Var = np.random.rand(*param_shape).astype(theano.config.floatX)
                outputs = f(X, Scale, Bias, Mean, Var, Dy)
                # compare outputs
                utt.assert_allclose(outputs[0], outputs[1])  # out
                # compare gradients
                utt.assert_allclose(outputs[2], outputs[2 + 5], atol=4e-5)  # dx
                utt.assert_allclose(outputs[3], outputs[3 + 5], atol=4e-5)  # dscale
                utt.assert_allclose(outputs[4], outputs[4 + 5])  # dbias
                utt.assert_allclose(outputs[5], outputs[5 + 5])  # dmean
                utt.assert_allclose(outputs[6], outputs[6 + 5], rtol=2e-3, atol=4e-5)  # dvar


def test_batch_normalization_broadcastable():
    # check if the broadcastable pattern is preserved by the optimizations
    x, dy, scale, bias, mean, var = (T.scalar(n).dimshuffle(['x'] * 5)
                                     for n in ('x', 'dy', 'scale', 'bias', 'mean', 'var'))

    # forward pass
    out_train, x_mean, x_invstd = bn.batch_normalization_train(x, scale, bias, 'spatial')
    out_test = bn.batch_normalization_test(x, scale, bias, mean, var, 'spatial')
    # backward pass
    grads_train = T.grad(None, wrt=[x, scale, bias], known_grads={out_train: dy})
    grads_test = T.grad(None, wrt=[x, scale, bias], known_grads={out_test: dy})
    # compile
    f = theano.function([x, scale, bias, mean, var, dy],
                        [out_train, x_mean, x_invstd, out_test] + grads_train + grads_test)
    assert not any([isinstance(n.op, (bn.AbstractBatchNormTrain,
                                      bn.AbstractBatchNormInference,
                                      bn.AbstractBatchNormTrainGrad))
                    for n in f.maker.fgraph.toposort()])
