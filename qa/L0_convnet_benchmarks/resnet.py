from __future__ import absolute_import, print_function
from collections import OrderedDict
import time
import six
from six import iteritems
from six.moves import range

import numpy
import theano
import theano.tensor as tensor
from theano import config

import lasagne
from lasagne.utils import floatX
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax


worker = None

def load_data():
    """
    create synthetic data
    """
    train_targets = numpy.random.randint(1000, size=(2048,1))
    train_data = numpy.random.random((train_targets.shape[0],3,224,224))
    valid_targets = numpy.random.randint(1000, size=(1024,1))
    valid_data = numpy.random.random((valid_targets.shape[0],3,224,224))
    test_targets = numpy.random.randint(1000, size=(1024,1))
    test_data = numpy.random.random((test_targets.shape[0],3,224,224))

    rval = ([numpy_floatX(train_data), numpy_int32(train_targets)],
            [numpy_floatX(valid_data), numpy_int32(valid_targets)],
            [numpy_floatX(test_data), numpy_int32(test_targets)])
    return rval


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def numpy_int32(data):
    return data.astype(numpy.int32)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def infer_bc_pattern(shape):
    pattern = [True if i == 1 else False for i in shape]
    return tuple(pattern)


def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % p.name,
                             broadcastable=infer_bc_pattern(p.get_value().shape))
               for p in tparams]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams, gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in iteritems(tparams)]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in iteritems(tparams)]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in iteritems(tparams)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in iteritems(tparams)]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in iteritems(tparams)]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in iteritems(tparams)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in iteritems(tparams)]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    names : list of string
        Names of the layers in block

    num_filters : int
        Number of filters in convolution layer

    filter_size : int
        Size of filters in convolution layer

    stride : int
        Stride of convolution layer

    pad : int
        Padding of convolution layer

    use_bias : bool
        Whether to use bias in conlovution layer

    nonlin : function
        Nonlinearity type of Nonlinearity layer

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, stride, pad,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]



simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block

    ratio_size : float
        Scale factor of filter size

    has_left_branch : bool
        if True, then left branch contains simple block

    upscale_factor : float
        Scale factor of filter bank at the output of residual block

    ix : int
        Id of residual block

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter),
        1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter),
            1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix


def build_resnet():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 2, 3, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2,
                             pad=0, mode='max', ignore_border=False)

    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name],
                                                              1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name],
                                                              1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name],
                                                              1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name],
                                                              1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name],
                                                              1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name],
                                                              1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name],
                                                              1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name],
                                                              1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)

    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)

    return net


def split_data_for_worker(dataset, splits, name):
    data = dataset[0][splits[name][0]:splits[name][1]]
    targets = dataset[1][splits[name][0]:splits[name][1]]
    return [data, targets]


def pred_error(f_pred, data, iterator):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    """
    valid_err = 0.
    i = 0
    for _, valid_index in iterator:
        x = [data[0][t] for t in valid_index]
        y = [data[1][t] for t in valid_index]
        valid_err += f_pred(x, y)
        i += 1

    return valid_err / i


def train_resnet(
    batch_size=64,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    validFreq=5,
    lrate=1e-4,
    optimizer=sgd,
):
    print(theano.config.profile)
    # Each worker needs the same seed in order to draw the same parameters.
    # This will also make them shuffle the batches the same way, but splits are
    # different so doesnt matter
#    seed = worker.send_req('seed')
#    numpy.random.seed(seed)

    print('Loading data')
    train, valid, test = load_data()

    print('Building model')
    resnet = build_resnet()
    params = lasagne.layers.get_all_params(resnet.values(), trainable=True)

#    print("Using all_reduce worker's interface!")
#    asgd = AverageSGD(worker)
 #   asgd.make_rule(params)
#    print("Params init done")

    x = tensor.ftensor4('x')
    y = tensor.imatrix('y')

    prob = lasagne.layers.get_output(resnet['prob'], x, deterministic=False)
    cost = tensor.nnet.categorical_crossentropy(prob, y.flatten()).mean()

    grads = tensor.grad(cost, wrt=params)
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, params, grads,
                                        x, y, cost)

    v_prob = lasagne.layers.get_output(resnet['prob'], x, deterministic=True)
    v_mc = tensor.mean(tensor.neq(tensor.argmax(prob, axis=1), y.flatten()))
    f_pred = theano.function([x,y], v_mc)

    print('Optimization')
#    splits = worker.send_req('splits', {'train_len': train[0].shape[0],
#                                        'valid_len': valid[0].shape[0],
#                                        'test_len' : test[0].shape[0]})
#    train = split_data_for_worker(train, splits, 'train_splits')
#    valid = split_data_for_worker(valid, splits, 'valid_splits')
#    test = split_data_for_worker(test, splits, 'test_splits')

    kf_valid = get_minibatches_idx(valid[0].shape[0], valid_batch_size)
    kf_test = get_minibatches_idx(test[0].shape[0], valid_batch_size)

    def train_iter():
        while True:
            kf = get_minibatches_idx(train[0].shape[0], batch_size, shuffle=False)
            for _, train_index in kf:
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]
                yield x, y

    train_it = train_iter()
    nb_train = train[0].shape[0] // batch_size

    # first pass in function so it doesnt bias the next time count
    # because of the dnn flags
    dummy_x = numpy_floatX(numpy.random.random((batch_size,3,224,224)))
    dummy_y = numpy_int32(numpy.random.randint(1000, size=(batch_size,1)))
    dumz = f_grad_shared(dummy_x, dummy_y)

    epoch = 0
    while True:
        for i in range(nb_train):
            x, y = next(train_it)
            func_time = time.time()
            cost = f_grad_shared(x, y)
            f_update(lrate)
            print("Func call time", time.time() - func_time)
            overhead_time = time.time()
            print("Overhead time", time.time() - overhead_time)

        print('Train cost:', cost)

        if epoch > 10:
            break
        epoch += 1


if __name__ == '__main__':
    train_resnet()
