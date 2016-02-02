from __future__ import print_function
from . import pool
import warnings

warnings.warn("downsample module has been moved to the pool module.")
max_pool2D = pool.max_pool2D
max_pool_2d_same_size = pool.max_pool_2d_same_size
max_pool_2d = pool.pool_2d
DownsampleFactorMax = pool.Pool
PoolGrad = pool.PoolGrad
MaxPoolGrad = pool.MaxPoolGrad
AveragePoolGrad = pool.AveragePoolGrad
DownsampleFactorMaxGradGrad = pool.DownsampleFactorMaxGradGrad
local_average_pool_grad = pool.local_average_pool_grad
