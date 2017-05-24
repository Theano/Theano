import sys
import theano
import math
import collections
import theano.tensor as T
import numpy as np
from baselayers import ReluLayer, SoftmaxLayer, DropoutLayer, DataLayer, Conv2DLayer, MaxPool2DLayer, LocalResponseNormalization2DLayer, ConcatLayer, GlobalPoolLayer, DenseLayer, AveragePool2DLayer

# MLSL based OP
from theano.sandbox.mlsl.multinode import collect


def aux_tower(name, input_layer,input_shape, drop_flag):
    net = {}
    params = []
    weight_types = []

    # input shape = (14x14x512or528)
    net['aux_tower_pool'] = AveragePool2DLayer(name,
    input_layer, input_shape, pool_size=5, stride=3, pad=0, ignore_border=False)
    output_shape = net['aux_tower_pool'].get_output_shape_for()

    # output shape = (4x4x512or528)
    filter_shape = (128, output_shape[1], 1, 1)
    sub_layer_name = name+'/conv'
    net['aux_tower_1x1'] = Conv2DLayer(name+'_conv',net['aux_tower_pool'].output, output_shape, filter_shape, flip_filters=False)
    params +=  [net['aux_tower_1x1'].W, net['aux_tower_1x1'].b]
    weight_types += net['aux_tower_1x1'].weight_types
    output_shape = net['aux_tower_1x1'].get_output_shape_for()

    net['FC_1'] = DenseLayer(net['aux_tower_1x1'].output, output_shape, 1024)
    sub_layer_name = name+'/fc'
    params +=  [net['FC_1'].W, net['FC_1'].b]
    weight_types += net['FC_1'].weight_types
    output_shape = net['FC_1'].get_output_shape_for()
    net['relu'] = ReluLayer(name+'_relu_fc',net['FC_1'].output, output_shape)
    output_shape = net['relu'].get_output_shape_for()

    if drop_flag == True:
        net['aux_dropout'] = DropoutLayer(net['relu'].output, prob_drop=0.7)
        DenseLayerInput = net['aux_dropout'].output
    else:
        DenseLayerInput = net['relu'].output
    net['FC_2'] = DenseLayer(DenseLayerInput, output_shape, 1000)
    sub_layer_name = name+'/classifier'
    params +=  [net['FC_2'].W, net['FC_2'].b]
    weight_types += net['FC_2'].weight_types

    net['classifier'] = SoftmaxLayer(name,net['FC_2'].output)

    return params, weight_types, net['classifier'].p_y_given_x, net['classifier'].negative_log_likelihood, net['classifier']

def build_inception_module_4in(inception_name, x,y,xx,yy, input_shape, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    params = []
    weight_types = []
    input_shapes = [1,1,1,1]

    filter_shape = (nfilters[1], input_shape[1], 1, 1)
    sub_layer_name=inception_name+'/1x1'
    net['1x1'] = Conv2DLayer(inception_name+'_1x1',x, input_shape, filter_shape, flip_filters=False)
    params +=  [net['1x1'].W, net['1x1'].b]
    weight_types += net['1x1'].weight_types
    output_shape = net['1x1'].get_output_shape_for()
    input_shapes[0] = output_shape

    filter_shape = (nfilters[2], input_shape[1], 1, 1)
    sub_layer_name=inception_name+'/3x3_reduce'
    net['3x3_reduce'] = Conv2DLayer(inception_name+'_3x3_reduce',y, input_shape, filter_shape, flip_filters=False)
    params += [net['3x3_reduce'].W, net['3x3_reduce'].b]
    weight_types += net['3x3_reduce'].weight_types
    output_shape = net['3x3_reduce'].get_output_shape_for()

    filter_shape = (nfilters[3], output_shape[1], 3, 3)
    sub_layer_name=inception_name+'/3x3'
    net['3x3'] = Conv2DLayer(inception_name+'_3x3',net['3x3_reduce'].output, output_shape, filter_shape, padsize=1, flip_filters=False)
    params += [net['3x3'].W, net['3x3'].b]
    weight_types += net['3x3'].weight_types
    output_shape = net['3x3'].get_output_shape_for()
    input_shapes[1] = output_shape

    filter_shape = (nfilters[4], input_shape[1], 1, 1)
    sub_layer_name=inception_name+'/5x5_reduce'
    net['5x5_reduce'] = Conv2DLayer(inception_name+'_5x5_reduce',xx, input_shape, filter_shape, flip_filters=False)
    params += [net['5x5_reduce'].W, net['5x5_reduce'].b]
    weight_types += net['5x5_reduce'].weight_types
    output_shape = net['5x5_reduce'].get_output_shape_for()

    filter_shape = (nfilters[5], output_shape[1], 5, 5)
    sub_layer_name=inception_name+'/5x5'
    net['5x5'] = Conv2DLayer(inception_name+'_5x5',net['5x5_reduce'].output, output_shape, filter_shape, padsize=2, flip_filters=False)
    params += [net['5x5'].W, net['5x5'].b]
    weight_types += net['5x5'].weight_types
    output_shape = net['5x5'].get_output_shape_for()
    input_shapes[2] = output_shape

    net['pool'] = MaxPool2DLayer(inception_name,yy, input_shape, pool_size=3, stride=1, pad=1)
    output_shape = net['pool'].get_output_shape_for()

    filter_shape = (nfilters[0], output_shape[1], 1, 1)
    sub_layer_name=inception_name+'/pool_proj'
    net['pool_proj'] = Conv2DLayer(inception_name+'_pool_proj',net['pool'].output, output_shape, filter_shape, flip_filters=False)
    params += [net['pool_proj'].W, net['pool_proj'].b]
    weight_types += net['pool_proj'].weight_types
    output_shape = net['pool_proj'].get_output_shape_for()
    input_shapes[3] = output_shape

    net['output'] = ConcatLayer(inception_name,[
        net['1x1'].output,
        net['3x3'].output,
        net['5x5'].output,
        net['pool_proj'].output,
        ], input_shapes)

    return net['output'], params, weight_types


def build_inception_module(inception_name, input_layer, input_shape, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)

    net = {}
    params = []
    weight_types = []
    input_shapes = [1,1,1,1]

    filter_shape = (nfilters[1], input_shape[1], 1, 1)
    sub_layer_name=inception_name+'/1x1'
    net['1x1'] = Conv2DLayer(inception_name+'_1x1',input_layer, input_shape, filter_shape, flip_filters=False)
    params +=  [net['1x1'].W, net['1x1'].b]
    weight_types += net['1x1'].weight_types
    output_shape = net['1x1'].get_output_shape_for()
    input_shapes[0] = output_shape

    filter_shape = (nfilters[2], input_shape[1], 1, 1)
    sub_layer_name=inception_name+'/3x3_reduce'
    net['3x3_reduce'] = Conv2DLayer(inception_name+'_3x3_reduce',input_layer, input_shape, filter_shape, flip_filters=False)
    params += [net['3x3_reduce'].W, net['3x3_reduce'].b]
    weight_types += net['3x3_reduce'].weight_types
    output_shape = net['3x3_reduce'].get_output_shape_for()

    filter_shape = (nfilters[3], output_shape[1], 3, 3)
    sub_layer_name=inception_name+'/3x3'
    net['3x3'] = Conv2DLayer(inception_name+'_3x3',net['3x3_reduce'].output, output_shape, filter_shape, padsize=1, flip_filters=False)
    params += [net['3x3'].W, net['3x3'].b]
    weight_types += net['3x3'].weight_types
    output_shape = net['3x3'].get_output_shape_for()
    input_shapes[1] = output_shape

    filter_shape = (nfilters[4], input_shape[1], 1, 1)
    sub_layer_name=inception_name+'/5x5_reduce'
    net['5x5_reduce'] = Conv2DLayer(inception_name+'_5x5_reduce',input_layer, input_shape, filter_shape, flip_filters=False)
    params += [net['5x5_reduce'].W, net['5x5_reduce'].b]
    weight_types += net['5x5_reduce'].weight_types
    output_shape = net['5x5_reduce'].get_output_shape_for()

    filter_shape = (nfilters[5], output_shape[1], 5, 5)
    sub_layer_name=inception_name+'/5x5'
    net['5x5'] = Conv2DLayer(inception_name+'_5x5',net['5x5_reduce'].output, output_shape, filter_shape, padsize=2, flip_filters=False)
    params += [net['5x5'].W, net['5x5'].b]
    weight_types += net['5x5'].weight_types
    output_shape = net['5x5'].get_output_shape_for()
    input_shapes[2] = output_shape

    net['pool'] = MaxPool2DLayer(inception_name,input_layer, input_shape, pool_size=3, stride=1, pad=1)
    output_shape = net['pool'].get_output_shape_for()

    filter_shape = (nfilters[0], output_shape[1], 1, 1)
    sub_layer_name=inception_name+'/pool_proj'
    net['pool_proj'] = Conv2DLayer(inception_name+'_pool_proj',net['pool'].output, output_shape, filter_shape, flip_filters=False)
    params += [net['pool_proj'].W, net['pool_proj'].b]
    weight_types += net['pool_proj'].weight_types
    output_shape = net['pool_proj'].get_output_shape_for()
    input_shapes[3] = output_shape

    net['output'] = ConcatLayer(inception_name,[
        net['1x1'].output,
        net['3x3'].output,
        net['5x5'].output,
        net['pool_proj'].output,
        ], input_shapes)

    return net['output'], params, weight_types

class googlenet(object):

    def __init__(self, input_shape, drop_flag):
        self.input_shape = input_shape
        x = T.tensor4('x', dtype=theano.config.floatX)
        y = T.vector('y', dtype='int32')
        net = {}
        params = []
        weight_types = []
        all_output = collections.OrderedDict()
        net['input'] = x

        filter_shape = (64, input_shape[1], 7, 7)
        net['conv1/7x7_s2'] = Conv2DLayer('conv1_7x7',net['input'], input_shape, filter_shape, convstride=2, padsize=3, flip_filters=False)
        params += [net['conv1/7x7_s2'].W, net['conv1/7x7_s2'].b]
        weight_types += net['conv1/7x7_s2'].weight_types
        output_shape = net['conv1/7x7_s2'].get_output_shape_for()
        all_output['conv1/7x7_s2'] = net['conv1/7x7_s2'].output

        net['pool1/3x3_s2'] = MaxPool2DLayer('pool1',net['conv1/7x7_s2'].output, output_shape, pool_size=3, stride=2, ignore_border=False)
        output_shape = net['pool1/3x3_s2'].get_output_shape_for()
        all_output['pool1/3x3_s2'] = net['pool1/3x3_s2'].output


        net['pool1/norm1'] = LocalResponseNormalization2DLayer(net['pool1/3x3_s2'].output, output_shape, alpha=0.0001, k=1)
        output_shape = net['pool1/norm1'].get_output_shape_for()
        all_output['pool1/norm1'] = net['pool1/norm1'].output

        filter_shape = (64, output_shape[1], 1, 1)
        net['conv2/3x3_reduce'] = Conv2DLayer('conv2_3x3_reduce',net['pool1/norm1'].output, output_shape, filter_shape, flip_filters=False)
        params += [net['conv2/3x3_reduce'].W, net['conv2/3x3_reduce'].b]
        weight_types += net['conv2/3x3_reduce'].weight_types
        output_shape = net['conv2/3x3_reduce'].get_output_shape_for()
        all_output['conv2/3x3_reduce'] = net['conv2/3x3_reduce'].output

        filter_shape = (192, output_shape[1], 3, 3)
        net['conv2/3x3'] = Conv2DLayer('conv2_3x3',net['conv2/3x3_reduce'].output, output_shape, filter_shape, padsize=1, flip_filters=False)
        params += [net['conv2/3x3'].W, net['conv2/3x3'].b]
        weight_types += net['conv2/3x3'].weight_types
        output_shape = net['conv2/3x3'].get_output_shape_for()
        all_output['conv2/3x3'] = net['conv2/3x3'].output

        net['conv2/norm2'] = LocalResponseNormalization2DLayer(net['conv2/3x3'].output, output_shape, alpha=0.0001, k=1)
        output_shape = net['conv2/norm2'].get_output_shape_for()

        net['pool2/3x3_s2'] = MaxPool2DLayer('pool2',net['conv2/norm2'].output, output_shape,  pool_size=3, stride=2, ignore_border=False)
        output_shape = net['pool2/3x3_s2'].get_output_shape_for()
        all_output['pool2/3x3_s2'] = net['pool2/3x3_s2'].output

        net['inception_3a'], pre_params, pre_weight_types = build_inception_module('inception_3a',net['pool2/3x3_s2'].output, output_shape, [32, 64, 96, 128, 16, 32])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_3a'].get_output_shape_for()
        all_output['inception_3a/output'] = net['inception_3a'].output

        net['inception_3b'], pre_params, pre_weight_types = build_inception_module('inception_3b',net['inception_3a'].output, output_shape, [64, 128, 128, 192, 32, 96])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_3b'].get_output_shape_for()
        all_output['inception_3b/output'] = net['inception_3b'].output

        net['pool3/3x3_s2'] = MaxPool2DLayer('pool3',net['inception_3b'].output, output_shape, pool_size=3, stride=2, ignore_border=False)
        output_shape = net['pool3/3x3_s2'].get_output_shape_for()
        all_output['pool3/3x3_s2'] = net['pool3/3x3_s2'].output

        net['inception_4a'], pre_params, pre_weight_types = build_inception_module('inception_4a',net['pool3/3x3_s2'].output, output_shape, [64, 192, 96, 208, 16, 48])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4a'].get_output_shape_for()
        aux_tower_shape1 = output_shape
        all_output['inception_4a/output'] = net['inception_4a'].output

        pre_params, pre_weight_types, p_y_given_x_1, negative_log_likelihood_1, classifier2 = aux_tower('loss1',net['inception_4a'].output, aux_tower_shape1, drop_flag=drop_flag)
        params += pre_params
        weight_types += pre_weight_types
        all_output['loss1/classifier'] = classifier2.output

        net['inception_4b'], pre_params, pre_weight_types = build_inception_module_4in('inception_4b',net['inception_4a'].output,net['inception_4a'].output,net['inception_4a'].output,net['inception_4a'].output, output_shape, [64, 160, 112, 224, 24, 64])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4b'].get_output_shape_for()
        all_output['inception_4b/output'] = net['inception_4b'].output

        net['inception_4c'], pre_params, pre_weight_types = build_inception_module('inception_4c',net['inception_4b'].output, output_shape, [64, 128, 128, 256, 24, 64])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4c'].get_output_shape_for()
        all_output['inception_4c/output'] = net['inception_4c'].output

        net['inception_4d'], pre_params, pre_weight_types = build_inception_module('inception_4d',net['inception_4c'].output, output_shape, [64, 112, 144, 288, 32, 64])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4d'].get_output_shape_for()
        aux_tower_shape2 = output_shape
        all_output['inception_4d/output'] = net['inception_4d'].output

        pre_params, pre_weight_types, p_y_given_x_2, negative_log_likelihood_2, classifier3 = aux_tower('loss2',net['inception_4d'].output,aux_tower_shape2,drop_flag=drop_flag)
        params += pre_params
        weight_types += pre_weight_types
        all_output['loss2/classifier'] = classifier3.output

        net['inception_4e'], pre_params, pre_weight_types = build_inception_module_4in('inception_4e',net['inception_4d'].output,net['inception_4d'].output,net['inception_4d'].output,net['inception_4d'].output, output_shape, [128, 256, 160, 320, 32, 128])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_4e'].get_output_shape_for()
        all_output['inception_4e/output'] = net['inception_4e'].output

        net['pool4/3x3_s2'] = MaxPool2DLayer('pool4',
        net['inception_4e'].output, output_shape, pool_size=3, stride=2, ignore_border=False)
        output_shape = net['pool4/3x3_s2'].get_output_shape_for()
        all_output['pool4/3x3_s2'] = net['pool4/3x3_s2'].output

        net['inception_5a'], pre_params, pre_weight_types = build_inception_module('inception_5a',net['pool4/3x3_s2'].output, output_shape, [128, 256, 160, 320, 32, 128])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_5a'].get_output_shape_for()
        all_output['inception_5a/output'] = net['inception_5a'].output

        net['inception_5b'], pre_params, pre_weight_types = build_inception_module('inception_5b',net['inception_5a'].output, output_shape, [128, 384, 192, 384, 48, 128])
        params += pre_params
        weight_types += pre_weight_types
        output_shape = net['inception_5b'].get_output_shape_for()
        all_output['inception_5b/output'] = net['inception_5b'].output

        net['pool5/7x7_s1'] = AveragePool2DLayer('pool5',net['inception_5b'].output, output_shape, pool_size=7, stride=1, pad=0, ignore_border=False)
        output_shape = net['pool5/7x7_s1'].get_output_shape_for()
        all_output['pool5/7x7_s1'] = net['pool5/7x7_s1'].output

        if drop_flag == True:
            net['dropout'] = DropoutLayer(net['pool5/7x7_s1'].output, prob_drop=0.4)
            DenseLayerInput = net['dropout'].output
        else:
            DenseLayerInput = net['pool5/7x7_s1'].output

        net['loss3/classifier'] = DenseLayer(DenseLayerInput, output_shape, 1000)

        params += net['loss3/classifier'].params
        weight_types += net['loss3/classifier'].weight_types
        #all_output['loss3/classifier'] = net['loss3/classifier'].output
        net['loss3'] = SoftmaxLayer('loss3',net['loss3/classifier'].output)

        self.net = net
        self.params = params
        self.weight_types = weight_types

        self.cost = net['loss3'].negative_log_likelihood(y,1.) + negative_log_likelihood_1(y,0.3) + negative_log_likelihood_2(y,0.3)
        self.loss = [negative_log_likelihood_1(y,0.3), negative_log_likelihood_2(y,0.3), net['loss3'].negative_log_likelihood (y,1)]

        self.errors = net['loss3'].errors(y)
        self.errors_top_5 = net['loss3'].errors_top_x(y, 5)
        self.x = x
        self.y = y
        self.all_output = all_output

    def set_dropout_off(self):
        DropoutLayer.SetDropoutOff()

    def set_dropout_on(self):
        DropoutLayer.SetDropoutOn()

def compile_train_model(model, learning_rate=0.01, batch_size=256, image_size=(3, 224, 224), use_momentum=True, momentum=0.9, weight_decay=0.0002):

    input_shape = (batch_size,) + image_size

    x = model.x
    y = model.y
    weight_types = model.weight_types
    cost = model.cost
    params = model.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    updates = []

    learning_rate = theano.shared(np.float32(learning_rate))
    lr = T.scalar('lr', dtype=theano.config.floatX)  # symbolic learning rate

    shared_x = theano.shared(np.zeros(input_shape, dtype=theano.config.floatX),borrow=True)
    shared_y = theano.shared(np.zeros((batch_size,), dtype='int32'), borrow=True)

    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]

    if use_momentum:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in \
                zip(params, grads, vels, weight_types):
            if weight_type == 'W':
                # MLSL: collect grad data from distributed nodes
                #real_grad = grad_i + weight_decay * param_i
                real_grad = collect(grad_i,param_i) + weight_decay * param_i
                real_lr = lr
            elif weight_type == 'b':
                # MLSL: collect grad data from distributed nodes
                #real_grad = grad_i
                real_grad = collect(grad_i,param_i)
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")
            vel_i_next = momentum * vel_i + real_lr * real_grad
            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i - vel_i_next))
    else:
        for param_i, grad_i, weight_type in zip(params, grads, weight_types):
            if weight_type == 'W':
                updates.append((param_i,
                                #param_i - lr * grad_i - weight_decay * lr * param_i))
                                # MLSL: update
                                param_i - lr * collect(grad_i,param_i) - weight_decay * lr * param_i))

            elif weight_type == 'b':
                #updates.append((param_i, param_i - 2 * lr * grad_i))
                # MLSL: update
                updates.append((param_i, param_i - 2 * lr * collect(grad_i,param_i)))
            else:
                raise TypeError("Weight Type Error")

    #Define Theano Functions
    train_model = theano.function([], cost, updates=updates, givens=[(x, shared_x), (y, shared_y), (lr, learning_rate)])

    return (train_model, shared_x, shared_y, learning_rate)

####derived from caffe googlenet prototxt
def set_learning_rate(shared_lr, iters):
    lr = 0.01 * math.sqrt(1.0-float(iters)/2400000.0)
    shared_lr.set_value(np.float32(lr))

def compile_val_model(model, batch_size=50, image_size=(3, 224, 224)):

    input_shape = (batch_size,) + image_size

    cost = model.cost
    x = model.x
    y = model.y
    errors = model.errors
    errors_top_5 = model.errors_top_5

    shared_x = theano.shared(np.zeros(input_shape, dtype=theano.config.floatX),borrow=True)
    shared_y = theano.shared(np.zeros((batch_size,), dtype='int32'), borrow=True)

    validate_outputs = [cost, errors, errors_top_5]

    validate_model = theano.function([], validate_outputs,
                                     givens=[(x, shared_x), (y, shared_y)])

    return (validate_model,
            shared_x, shared_y)
