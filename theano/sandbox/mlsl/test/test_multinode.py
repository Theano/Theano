import numpy as np

import theano
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
import theano.tensor as T

from theano.sandbox.mlsl.multinode import collect

import time
from datetime import datetime
import traceback

try:
    import theano.sandbox.mlsl.multinode as distributed
    print('mlsl is imported')
except ImportError as e:
    print ('Failed to import distributed module, please double check')
    print(traceback.format_exc())


class ConvPoolLayer(object):
    def __init__(self, rng, input, image_shape, filter_shape, pool_size=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(pool_size))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(input=input,
                          filters=self.W,
                          input_shape=image_shape,
                          filter_shape=filter_shape,
                          filter_flip=False)

        conv_out = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        pool_out = pool_2d(input=conv_out,
                           ds=pool_size,
                           ignore_border=False)

        self.output = pool_out
        self.params = [self.W, self.b]
        self.weight_types = ['W', 'b']


class SoftmaxLayer(object):
    def __init__(self, input):
        self.input = input
        self.p_y_given_x = T.nnet.softmax(input)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y, loss_weight):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) * loss_weight


class SimpleCost(object):
    def __init__(self, input):
        self.input = input

    def cost(self):
        return T.mean(self.input)


class convnet(object):
    def __init__(self, batch_size=20, nkerns=[10, 20]):
        self.batch_size = batch_size
        x = T.tensor4('x', dtype=theano.config.floatX)
        y = T.vector('y', dtype='int32')
        params = []
        weight_types = []

        rng = np.random.RandomState(23455)

        input_shape = (self.batch_size, 1, 28, 28)

        layer0 = ConvPoolLayer(
            rng=rng,
            input=x,
            image_shape=input_shape,
            filter_shape=(nkerns[0], 1, 5, 5),
            pool_size=(2, 2)
        )
        params += layer0.params
        weight_types += layer0.weight_types

        layer1 = ConvPoolLayer(
            rng=rng,
            input=layer0.output,
            image_shape=(self.batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            pool_size=(2, 2)
        )
        params += layer1.params
        weight_types += layer1.weight_types

        # '''
        layer2 = SoftmaxLayer(layer1.output.flatten(2))

        self.cost = layer2.negative_log_likelihood(y, 1.)
        '''
        self.cost = SimpleCost(layer1.output).cost()
        '''

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.params = params
        self.weight_types = weight_types


def compile_model(model, learning_rate=0.01, use_momentum=True, momentum=0.9, weight_decay=0.0002, dist=None):
    x = model.x
    y = model.y
    batch_size = model.batch_size
    input_shape = model.input_shape
    cost = model.cost
    params = model.params
    weight_types = model.weight_types

    grads = T.grad(cost, params)
    updates = []

    learning_rate = theano.shared(np.float32(learning_rate))
    lr = T.scalar('lr', dtype=theano.config.floatX)

    shared_x = theano.shared(np.zeros(input_shape, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.zeros((batch_size, ), dtype='int32'), borrow=True)

    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]

    if use_momentum:
        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i, weight_type in \
                zip(params, grads, vels, weight_types):
            if weight_type == 'W':
                # real_grad = grad_i + weight_decay * param_i
                # MLSL: collect grad data from distributed nodes
                real_grad = collect(grad_i, param_i) + weight_decay * param_i
                real_lr = lr
            elif weight_type == 'b':
                # real_grad = grad_i
                # MLSL: collect grad data from distributed nodes
                real_grad = collect(grad_i, param_i)
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
                                # param_i - lr * grad_i - weight_decay * lr * param_i))
                                # MLSL: update
                                param_i - lr * collect(grad_i, param_i) - weight_decay * lr * param_i))

            elif weight_type == 'b':
                # updates.append((param_i, param_i - 2 * lr * grad_i))
                # MLSL: update
                updates.append((param_i, param_i - 2 * lr * collect(grad_i, param_i)))
            else:
                raise TypeError("Weight Type Error")

    if dist.rank == 0:
        theano.printing.pydotprint(cost, outfile="mn_train_before_compile.png", var_with_name_simple=True)
    train_model = theano.function([], cost, updates=updates, givens=[(x, shared_x), (y, shared_y), (lr, learning_rate)])
    # train_model = theano.function([], cost, updates=updates, givens=[(x, shared_x), (lr, learning_rate)])
    if dist.rank == 0:
        theano.printing.pydotprint(train_model, outfile="mn_train_after_compile.png", var_with_name_simple=True)

    return (train_model, shared_x, shared_y, learning_rate)


def time_theano_run(func, info_string):
    num_batches = 50
    num_steps_burn_in = 10
    durations = []
    for i in xrange(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = func()  # noqa
        duration = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: Iteration %d, %s, time: %.2f ms' %
                      (datetime.now(), i - num_steps_burn_in, info_string, duration * 1000))
            durations.append(duration)
    durations = np.array(durations)
    print('%s: Average %s pass: %.2f ms ' %
          (datetime.now(), info_string, durations.mean() * 1000))


def train(batch_size=32):
    model = convnet(batch_size)

    dist = distributed.Distribution()
    print ('dist.rank: ', dist.rank, 'dist.size: ', dist.size)

    distributed.set_global_batch_size(batch_size * dist.size)
    distributed.set_param_count(len(model.params))

    (train_model, shared_x, shared_y, shared_lr) = compile_model(model, use_momentum=False, dist=dist)

    images = np.random.random_integers(0, 255, model.input_shape).astype('float32')
    print ('images.shape:', images.shape)
    labels = np.random.random_integers(0, 319, model.batch_size).astype('int32')
    shared_x.set_value(images)
    shared_y.set_value(labels)

    print("Start training")
    time_theano_run(train_model, 'Forward-Backward')

    # dist.destroy()


if __name__ == '__main__':
    train(batch_size=32)
