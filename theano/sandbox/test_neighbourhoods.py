#!/usr/bin/python

import theano
import numpy

import theano.tensor as T

from theano.sandbox.neighbourhoods import *

'''
def test_imgFromNeigh_noborder_1d():
    x = T.dtensor3()

    a = numpy.arange(2*2*6).reshape((2,2,6))

    neighs = NeighbourhoodsFromImages(2, (3,))(x)

    f = theano.function([x], neighs)

    z = f(a)
    
    cmp = numpy.asarray([[[[  0.,   1.,   2.],
        [  3.,   4.,   5.]],
        [[  6.,   7.,   8.],
        [  9.,  10.,  11.]]],
        [[[ 12.,  13.,  14.],
        [ 15.,  16.,  17.]],
        [[ 18.,  19.,  20.],
        [ 21.,  22.,  23.]]]])

    assert numpy.allclose(z, cmp)

    x2 = T.dtensor4()

    imgs = ImagesFromNeighbourhoods(2, (3,))(x2)

    f2 = theano.function([x2], imgs)
    z2 = f2(cmp)

    assert numpy.allclose(z2, a)
   
def test_imgFromNeigh_1d_stridesmaller():
    x = T.dtensor3()

    a = numpy.arange(2*4).reshape((2,4))

    #neighs = NeighbourhoodsFromImages(1, (3,), strides=(1,), ignore_border=False)(x)

    cmp = numpy.asarray([[[0.,1.,2.],[1.,2.,3.],[2.,3.,0.],[3.,0.,0.]],\
                [[4.,5.,6.],[5.,6.,7.],[6.,7.,0.],[7.,0.,0.]]])

    images = ImagesFromNeighbourhoods(1, (3,), strides=(1,), ignore_border=False)(x)

    f = theano.function([x], images)

    aprime = f(cmp)

    should_be = [[0.,  1.,  2.,  3.,  0.,  0.], [ 4.,  5.,  6.,  7.,  0.,  0.]]

    assert numpy.allclose(aprime, should_be)

def test_neighFromImg_1d():
    x = T.dtensor3()

    a = numpy.arange(2*2*6).reshape((2,2,6))

    neighs = NeighbourhoodsFromImages(2, (3,))(x)

    f = theano.function([x], neighs)

    z = f(a)
    
    cmp = numpy.asarray([[[[  0.,   1.,   2.],
        [  3.,   4.,   5.]],
        [[  6.,   7.,   8.],
        [  9.,  10.,  11.]]],
        [[[ 12.,  13.,  14.],
        [ 15.,  16.,  17.]],
        [[ 18.,  19.,  20.],
        [ 21.,  22.,  23.]]]])

    assert numpy.allclose(z, cmp)

def test_neighFromImg_1d_ignoreborder():
    x = T.dtensor3()

    a = numpy.arange(1*2*7).reshape((1,2,7))

    neighs = NeighbourhoodsFromImages(2, (3,), ignore_border=True)(x)

    f = theano.function([x], neighs)

    z = f(a)

    cmp = numpy.asarray([[[[  0.,   1.,   2.],
        [  3.,   4.,   5.]],
        [[  7.,   8.,  9.],
        [  10.,  11.,  12.]]]])

    assert numpy.allclose(z, cmp)

def test_neighFromImg_1d_stridesmaller():
    x = T.dmatrix()

    a = numpy.arange(2*4).reshape((2,4))

    neighs = NeighbourhoodsFromImages(1, (3,), strides=(1,), ignore_border=False)(x)

    f = theano.function([x], neighs)

    z = f(a)

    cmp = numpy.asarray([[[0.,1.,2.],[1.,2.,3.],[2.,3.,0.],[3.,0.,0.]],\
                [[4.,5.,6.],[5.,6.,7.],[6.,7.,0.],[7.,0.,0.]]])

    assert numpy.allclose(z, cmp)

def test_neighFromImg_1d_stridesbigger():
    x = T.dmatrix()

    a = numpy.arange(2*4).reshape((2,4))

    neighs = NeighbourhoodsFromImages(1, (2,), strides=(3,), ignore_border=False)(x)

    f = theano.function([x], neighs)

    z = f(a)

    cmp = numpy.asarray([[[0.,1.],[3.,0.]],\
                [[4.,5.],[7.,0.]]])

    assert numpy.allclose(z, cmp)

def test_neighFromImg_2d():
    x = T.dtensor3()

    a = numpy.arange(2*5*3).reshape((2,5,3))

    neighs = NeighbourhoodsFromImages(1, (2,2), ignore_border=False)(x)

    f = theano.function([x], neighs)

    z = f(a)

    cmp = numpy.asarray([[[[  0.,   1.,   3.,   4.,],
           [  2.,   0.,   5.,   0.,]],
          [[  6.,   7.,   9.,  10.,],
           [  8.,   0.,  11.,   0.,]],
          [[ 12.,  13.,   0.,   0.,],
           [ 14.,   0.,   0.,   0.,]]],
         [[[ 15.,  16.,  18.,  19.,],
           [ 17.,   0.,  20.,   0.,]],
          [[ 21.,  22.,  24.,  25.,],
           [ 23.,   0.,  26.,   0.,]],
          [[ 27.,  28.,   0.,   0.,],
           [ 29.,   0.,   0.,   0.,]]]])

    assert numpy.allclose(z, cmp)



if __name__ == '__main__':

    numpy.set_printoptions(threshold=numpy.nan)
    test_neighFromImg_1d()
    test_neighFromImg_1d_ignoreborder()
    test_neighFromImg_1d_stridesmaller()
    test_neighFromImg_1d_stridesbigger()
    test_neighFromImg_2d()
    test_imgFromNeigh_noborder_1d()
    test_imgFromNeigh_1d_stridesmaller()
'''

