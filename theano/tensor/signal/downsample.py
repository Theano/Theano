from __future__ import print_function
import pool
import warnings


def max_pool2D(*args, **kwargs):
    warnings.warn("function 'max_pool2D' is now located "
                  "in 'pool'.")
    return pool.max_pool2D(*args, **kwargs)


def max_pool_2d_same_size(*args, **kwargs):
    warnings.warn("function 'max_pool_2d_same_size' is now located "
                  "in 'pool'.")
    return pool.max_pool_2d_same_size(*args, **kwargs)


def max_pool_2d(*args, **kwargs):
    warnings.warn("function 'max_pool_2d' has been renamed and is now located "
                  "in 'pool.pool_2d'.")
    return pool.pool_2d(*args, **kwargs)


def DownsampleFactorMax(*args, **kwargs):
    warnings.warn("Class 'DownsampleFactorMax' has been renamed and is now located "
                  "in 'pool.Pool'.")
    return pool.Pool(*args, **kwargs)


def PoolGrad(*args, **kwargs):
    warnings.warn("Class 'PoolGrad' is now located "
                  "in 'pool'.")
    return pool.PoolGrad(*args, **kwargs)


def MaxPoolGrad(*args, **kwargs):
    warnings.warn("Class 'MaxPoolGrad' is now located "
                  "in 'pool'.")
    return pool.MaxPoolGrad(*args, **kwargs)


def AveragePoolGrad(*args, **kwargs):
    warnings.warn("Class 'AveragePoolGrad' is now located "
                  "in 'pool'.")
    return pool.AveragePoolGrad(*args, **kwargs)


def DownsampleFactorMaxGradGrad(*args, **kwargs):
    warnings.warn("Class 'DownsampleFactorMaxGradGrad' is now located "
                  "in 'pool'.")
    return pool.DownsampleFactorMaxGradGrad(*args, **kwargs)


def local_average_pool_grad(*args, **kwargs):
    warnings.warn("Class 'local_average_pool_grad' is now located "
                  "in 'pool'.")
    return pool.local_average_pool_grad(*args, **kwargs)
