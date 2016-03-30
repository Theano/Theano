from __future__ import absolute_import, print_function, division
from . import pool
import warnings

warnings.warn(
    "downsample module has been moved to the theano.tensor.signal.pool module.")
max_pool_2d_same_size = pool.max_pool_2d_same_size
max_pool_2d = pool.pool_2d
DownsampleFactorMax = pool.Pool
PoolGrad = pool.PoolGrad
MaxPoolGrad = pool.MaxPoolGrad
AveragePoolGrad = pool.AveragePoolGrad


# This is for compatibility with pickled things.  It should go away at
# some point.
class DownsampleFactorMaxGrad(object):
    def __new__(self, ds, ignore_border, st=None, padding=(0, 0), mode='max'):
        if mode == 'max':
                return MaxPoolGrad(ds=ds, ignore_border=ignore_border, st=st,
                                   padding=padding)
        else:
            return AveragePoolGrad(ds=ds, ignore_border=ignore_border, st=st,
                                   padding=padding, mode=mode)

DownsampleFactorMaxGradGrad = pool.DownsampleFactorMaxGradGrad
