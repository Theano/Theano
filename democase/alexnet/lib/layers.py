import numpy as np
import theano
import theano.tensor as T

from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
import theano.tensor.nnet.lrn as LRN

from theano.sandbox.mkl.basic_ops import U2IConv, I2U
from theano.sandbox.mkl.mkl_conv import Conv2D

import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState(23455)
# set a fixed number for 2 purpose:
# 1. repeatable experiments; 2. for multiple-GPU, the same initial weights
uniq_id = 10000


def convGroupWithBiasLayer(image, kernel, bias, imshp, kshp,
                           subsample, border_mode, filter_flip=False, group=1):
    global uniq_id
    uniq_id += 1

    if group == 1:
        conv_out = conv2d(input=image, filters=kernel, filter_shape=kshp,
                          input_shape=imshp, subsample=subsample,
                          border_mode=border_mode, filter_flip=filter_flip)

        conv_out = conv_out + bias.dimshuffle('x', 0, 'x', 'x')
    else:
        u2i_input = U2IConv(imshp=imshp, kshp=kshp, border_mode=border_mode,
                            subsample=subsample, uniq_id=uniq_id)(image)

        conv_out = Conv2D(imshp=imshp, kshp=kshp, border_mode=border_mode,
                          subsample=subsample, filter_flip=filter_flip,
                          uniq_id=uniq_id)(u2i_input, kernel, bias)
        conv_out = I2U(uniq_id=uniq_id)(conv_out)

    return conv_out


class Weight(object):

    def __init__(self, w_shape, mean=0, std=0.01):
        super(Weight, self).__init__()
        if std != 0:
            self.np_values = np.asarray(
                rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
        else:
            self.np_values = np.cast[theano.config.floatX](
                mean * np.ones(w_shape, dtype=theano.config.floatX))

        self.val = theano.shared(value=self.np_values)

    def save_weight(self, dir, name):
        print 'weight saved: ' + name
        np.save(dir + name + '.npy', self.val.get_value())

    def load_weight(self, dir, name):
        print 'weight loaded: ' + name
        self.np_values = np.load(dir + name + '.npy')
        self.val.set_value(self.np_values)


class DataLayer(object):

    def __init__(self, input, image_shape, cropsize, rand, mirror, flag_rand):
        '''
        The random mirroring and cropping in this function is done for the
        whole batch.
        '''

        # trick for random mirroring
        mirror = input[:, :, :, ::-1]
        input = T.concatenate([input, mirror], axis=1)

        # crop images
        center_margin = (image_shape[3] - cropsize) / 2

        if flag_rand:
            mirror_rand = T.cast(rand[2], 'int32')
            crop_xs = T.cast(rand[0] * center_margin * 2, 'int32')
            crop_ys = T.cast(rand[1] * center_margin * 2, 'int32')
        else:
            mirror_rand = 0
            crop_xs = center_margin
            crop_ys = center_margin

        self.output = input[:, mirror_rand * 3:(mirror_rand + 1) * 3, :, :]
        self.output = self.output[
            :, :, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize]

        print "data layer with shape_in: " + str(image_shape)


class ConvPoolLayer(object):

    def __init__(self, input, image_shape, filter_shape, convstride, padsize,
                 group, poolsize, poolstride, bias_init, lrn=False,
                 ):
        '''
        conv, pooling, relu and norm layers
        '''

        self.filter_size = filter_shape
        self.convstride = convstride
        self.padsize = padsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.channel = image_shape[1]
        self.lrn = lrn
        assert group in [1, 2]

        self.filter_shape = list(filter_shape)
        self.image_shape = list(image_shape)

        if group == 1:
            self.W = Weight(self.filter_shape)
            self.b = Weight(self.filter_shape[0], bias_init, std=0)
            new_filter_shape = self.filter_shape
        else:
            # making new filter shape 5D tensor, when group is more than 1
            new_filter_shape = (group, self.filter_shape[0] / group,
                                self.filter_shape[1] / group, self.filter_shape[2],
                                self.filter_shape[3])
            self.W = Weight(new_filter_shape)
            self.b = Weight(self.filter_shape[0], bias_init, std=0)

        conv_out = convGroupWithBiasLayer(input, self.W.val, self.b.val,
                                          image_shape, new_filter_shape,
                                          (convstride, convstride), padsize, False, group)

        # ReLu
        self.output = T.nnet.relu(conv_out, 0)

        # LRN
        if self.lrn:
            self.output = LRN.lrn(self.output, alpha=1e-4, beta=0.75, k=1, n=5)

        # Pooling
        if self.poolsize != 1:
            self.output = pool.Pool(ignore_border=True,
                                    mode='max')(self.output,
                                                (poolsize, poolsize),
                                                (poolstride, poolstride))
        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        print "conv layer with shape_in: {}".format(str(image_shape))


class FCLayer(object):

    def __init__(self, input, n_in, n_out):

        self.W = Weight((n_in, n_out), std=0.005)
        self.b = Weight(n_out, mean=0.1, std=0)
        self.input = input
        lin_output = T.dot(self.input, self.W.val) + self.b.val
        self.output = T.maximum(lin_output, 0)
        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']
        print 'fc layer with num_in: ' + str(n_in) + ' num_out: ' + str(n_out)


class DropoutLayer(object):
    seed_common = np.random.RandomState(0)  # for deterministic results
    layers = []

    def __init__(self, input, n_in, n_out, prob_drop=0.5):

        self.prob_drop = prob_drop
        self.prob_keep = 1.0 - prob_drop
        self.scale = 1.0 / self.prob_keep
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on

        seed_this = DropoutLayer.seed_common.randint(0, 2**31-1)
        mask_rng = theano.tensor.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=input.shape)

        self.output = \
            self.flag_on * self.scale * T.cast(self.mask, theano.config.floatX) * input + \
            self.flag_off * input

        DropoutLayer.layers.append(self)

        print 'dropout layer with P_drop: ' + str(self.prob_drop)

    @staticmethod
    def SetDropoutOn():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(1.0)

    @staticmethod
    def SetDropoutOff():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(0.0)


class SoftmaxLayer(object):

    def __init__(self, input, n_in, n_out):

        self.W = Weight((n_in, n_out))
        self.b = Weight((n_out,), std=0)

        self.p_y_given_x = T.nnet.softmax(
            T.dot(input, self.W.val) + self.b.val)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        print 'softmax layer with num_in: ' + str(n_in) + \
            ' num_out: ' + str(n_out)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def errors_top_x(self, y, num_top=5):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_pred_top_x = T.argsort(self.p_y_given_x, axis=1)[:, -num_top:]
            y_top_x = y.reshape((y.shape[0], 1)).repeat(num_top, axis=1)
            return T.mean(T.min(T.neq(y_pred_top_x, y_top_x), axis=1))
        else:
            raise NotImplementedError()
