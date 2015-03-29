import theano
import numpy as np
from theano import shared
from theano.sandbox import cuda
from pylearn2.utils import sharedX
from theano.tensor.nnet.conv3d2d import corr3d
from theano.tensor.nnet.conv3d2d import conv3d
from theano.sandbox.cuda.basic_ops import gpu_contiguous

rng = np.random.RandomState([2013, 1, 29, 45])

batch_size = 5
rows = 10
cols = 9
temp = 6
channels = 3
filter_rows = 4
filter_cols = filter_rows
filter_temp = 3
num_filters = 16
images = shared(rng.uniform(-1., 1., (batch_size, temp, channels, rows,
                             cols)).astype('float32'), name='images')
filters = shared(rng.uniform(-1., 1., (num_filters, filter_temp, channels, filter_rows,
                             filter_cols)).astype('float32'), name='filters')

gpu_images = gpu_contiguous(images)
gpu_filters = gpu_contiguous(filters)
#import pdb; pdb.set_trace()
rval1 = corr3d(images, filters, None, None, (2,2))
rval2 = conv3d(images, filters, None, None, (2,2))