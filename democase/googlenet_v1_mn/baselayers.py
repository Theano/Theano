import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
import theano.tensor.nnet.lrn as LRN
from theano.sandbox.mkl.mkl_conv import AbstractConvGroup
from theano.tensor.nnet import relu

import warnings
warnings.filterwarnings("ignore")
rng = np.random.RandomState(23455)    
class SoftmaxLayer(object):

    def __init__(self,name,x):
        self.classifier=x
        self.output = T.nnet.softmax(self.classifier)
        self.p_y_given_x = self.output
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y, loss_weight):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])*loss_weight
    
    def categorical_crossentropy(self, targets):
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, targets))
 
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

class DropoutLayer(object):
    seed_common = np.random.RandomState(0)  # for deterministic results
    layers = []

    def __init__(self, input, prob_drop=0.5):

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

    @staticmethod
    def SetDropoutOn():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(1.0)

    @staticmethod
    def SetDropoutOff():
        for i in range(0, len(DropoutLayer.layers)):
            DropoutLayer.layers[i].flag_on.set_value(0.0)
    
class DataLayer(object):

    def __init__(self, input, input_shape, cropsize, rand, mirror, flag_rand):
        '''
        The random mirroring and cropping in this function is done for the
        whole batch.
        '''

        # trick for random mirroring
        mirror = input[:, :, ::-1, :]
        input = T.concatenate([input, mirror], axis=0)

        # crop images
        center_margin = (input_shape[2] - cropsize) / 2

        if flag_rand:
            mirror_rand = T.cast(rand[2], 'int32')
            crop_xs = T.cast(rand[0] * center_margin * 2, 'int32')
            crop_ys = T.cast(rand[1] * center_margin * 2, 'int32')
        else:
            mirror_rand = 0
            crop_xs = center_margin
            crop_ys = center_margin

        self.output = input[mirror_rand * 3:(mirror_rand + 1) * 3, :, :, :]
        self.output = self.output[
            :, crop_xs:crop_xs + cropsize, crop_ys:crop_ys + cropsize, :]


class Conv2DLayer(object):

    def __init__(self,name, input, input_shape, filter_shape, convstride=1, padsize=0, flip_filters=False):
        assert input_shape[1] == filter_shape[1]
        self.filter_size = (filter_shape[-1],) * 2
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.convstride = (convstride,) * 2
        self.pad = (padsize,) * 2
        n1, n2 = filter_shape[:2]
        receptive_field_size = np.prod(filter_shape[2:]) 
        std = np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        self.np_values = np.asarray(
                rng.uniform(low=-std, high=std, size=filter_shape), dtype=theano.config.floatX)
        self.W = theano.shared(value=self.np_values, borrow=True)
        b_values = 0.2 * np.ones((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values)       
        self.input = input
        border_mode = self.pad

        conv_out = AbstractConvGroup(imshp=input_shape,
                                     kshp=filter_shape,
                                     subsample=(convstride, convstride),
                                     border_mode=(padsize, padsize),
                                     filter_flip=False,
                                     group=1)(input, self.W, self.b)
        self.output = T.nnet.relu(conv_out,0)
          
        self.params = [self.W, self.b]
        self.weight_types = ['W', 'b']
        
    def conv_output_length(self, input_length, filter_size, stride, pad=0):
        if input_length is None:
            return None
        if pad == 'valid':
            output_length = input_length - filter_size + 1
        elif pad == 'full':
            output_length = input_length + filter_size - 1
        elif pad == 'same':
            output_length = input_length
        elif isinstance(pad, int):
            output_length = input_length + 2 * pad - filter_size + 1
        else:
            raise ValueError('Invalid pad: {0}'.format(pad))

        output_length = (output_length + stride - 1) // stride
        return output_length
    
    def get_output_shape_for(self):
        batchsize = self.input_shape[0]
        return ((batchsize, self.filter_shape[0]) +
                tuple(self.conv_output_length(input, filter, stride, p)
                for input, filter, stride, p
                in zip(self.input_shape[2:], self.filter_size,
                       self.convstride, self.pad)))

class ReluLayer(object):
    def __init__(self, name, input, input_shape):
        self.input_shape = input_shape
        self.output = relu(input)

    def get_output_shape_for(self):
        return self.input_shape
        
class MaxPool2DLayer(object):
    
    def __init__(self,name, input, input_shape, pool_size=2, stride=None, pad=0, ignore_border=True):
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
            
        self.input_shape = input_shape
        self.input = input
        self.pool_size = (pool_size,) * 2
        self.ignore_border = ignore_border
        self.stride = (self.stride,) * 2
        self.pad = (pad,) * 2
        self.output = pool.pool_2d(input,
                         ds=self.pool_size,
                         st=self.stride,
                         ignore_border=self.ignore_border,
                         padding=self.pad,
                         mode='max',
                         )
                         
    def pool_output_length(self, input_length, pool_size, stride, pad, ignore_border):
        if input_length is None or pool_size is None:
            return None

        if ignore_border:
            output_length = input_length + 2 * pad - pool_size + 1
            output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
        else:
            assert pad == 0

            if stride >= pool_size:
                output_length = (input_length + stride - 1) // stride
            else:
                output_length = max(
                    0, (input_length - pool_size + stride - 1) // stride) + 1

        return output_length
                     
    def get_output_shape_for(self):
        output_shape = list(self.input_shape)  # copy / convert to mutable list

        output_shape[2] = self.pool_output_length(self.input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = self.pool_output_length(self.input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )
        return tuple(output_shape)
        
class AveragePool2DLayer(object):
    
    def __init__(self,name, input, input_shape, pool_size=2, stride=None, pad=0, ignore_border=True):
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride
            
        self.input_shape = input_shape
        self.input = input
        self.pool_size = (pool_size,) * 2
        self.ignore_border = ignore_border
        self.stride = (self.stride,) * 2
        self.pad = (pad,) * 2

        self.output = pool.pool_2d(input,
                         ds=self.pool_size,
                         st=self.stride,
                         ignore_border=self.ignore_border,
                         #ignore_border=True,
                         padding=self.pad,
                         mode='average_exc_pad',
                         #mode='max',
                         )                         
    def pool_output_length(self, input_length, pool_size, stride, pad, ignore_border):
        if input_length is None or pool_size is None:
            return None

        if ignore_border:
            output_length = input_length + 2 * pad - pool_size + 1
            output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
        else:
            assert pad == 0

            if stride >= pool_size:
                output_length = (input_length + stride - 1) // stride
            else:
                output_length = max(
                    0, (input_length - pool_size + stride - 1) // stride) + 1

        return output_length
                     
    def get_output_shape_for(self):
        output_shape = list(self.input_shape)  # copy / convert to mutable list

        output_shape[2] = self.pool_output_length(self.input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = self.pool_output_length(self.input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )
        return tuple(output_shape)
        
class LocalResponseNormalization2DLayer(object):
    def __init__(self, input, input_shape, alpha=1e-4, k=1, beta=0.75, n=5):         
        self.input_shape = input_shape
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        self.output = LRN.lrn(input,alpha=self.alpha, beta=self.beta, k=self.k, n=self.n)

              
    def get_output_shape_for(self):
        return self.input_shape
    
class ConcatLayer(object):
    def __init__(self,name, inputs, input_shapes, axis=1,layer_count=0):
        self.input_shapes = input_shapes
        self.axis = axis
        self.lc = layer_count
        self.output = T.concatenate(inputs, axis=self.axis)


        
    def get_output_shape_for(self):
        output_shape = [next((s for s in sizes if s is not None), None)
                        for sizes in zip(*self.input_shapes)]
                        
        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(i == self.axis or s1 is None or s2 is None or s1 == s2
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in self.input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "in the concatenation axis")
        # Infer output shape on concatenation axis and return
        sizes = [input_shape[self.axis] for input_shape in self.input_shapes]
        concat_size = None if any(s is None for s in sizes) else sum(sizes)
        output_shape[self.axis] = concat_size
        return tuple(output_shape)
    
class GlobalPoolLayer(object):
    
    def __init__(self, input, input_shape):
        self.input_shape = input_shape
        self.input = input
        self.output = T.mean(input.flatten(3), axis=2)
        
    def get_output_shape_for(self):
        return self.input_shape[:2]

class DenseLayer(object):

    def __init__(self, input, input_shape, num_units):
        self.input_shape = input_shape
        self.num_units = num_units
        num_input_channels = int(np.prod(self.input_shape[1:]))
        std = np.sqrt(2. / (num_input_channels + self.num_units))
        np_values = np.asarray(rng.uniform(low=-std, high=std, size=(num_input_channels, self.num_units)), dtype=theano.config.floatX)
        self.W = theano.shared(value=np_values)
        b_values = np.zeros((self.num_units,), dtype=theano.config.floatX)
        b_values.fill(0.2)
        self.b = theano.shared(value=b_values)

        if input.ndim > 2:
            input = input.flatten(2)
        self.output =  T.dot(input, self.W) + self.b.dimshuffle('x', 0)

        self.params = [self.W, self.b]
        self.weight_types = ['W', 'b']

    def get_output_shape_for(self):
        return (self.input_shape[0], self.num_units)

