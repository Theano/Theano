"""
Test for jacobian/hessian functions in Theano
"""
from __future__ import absolute_import, print_function, division
from six.moves import xrange
from theano.tests import unittest_tools as utt
import theano
from theano import tensor
import numpy as np

utt.seed_rng()


def test001_jacobian_vector():
    x = tensor.vector()
    y = x * 2
    rng = np.random.RandomState(seed=utt.fetch_seed())

    # test when the jacobian is called with a tensor as wrt
    Jx = tensor.jacobian(y, x)
    f = theano.function([x], Jx)
    vx = rng.uniform(size=(10,)).astype(theano.config.floatX)
    assert np.allclose(f(vx), np.eye(10) * 2)

    # test when the jacobian is called with a tuple as wrt
    Jx = tensor.jacobian(y, (x,))
    assert isinstance(Jx, tuple)
    f = theano.function([x], Jx[0])
    vx = rng.uniform(size=(10,)).astype(theano.config.floatX)
    assert np.allclose(f(vx), np.eye(10) * 2)

    # test when the jacobian is called with a list as wrt
    Jx = tensor.jacobian(y, [x])
    assert isinstance(Jx, list)
    f = theano.function([x], Jx[0])
    vx = rng.uniform(size=(10,)).astype(theano.config.floatX)
    assert np.allclose(f(vx), np.eye(10) * 2)

    # test when the jacobian is called with a list of two elements
    z = tensor.vector()
    y = x * z
    Js = tensor.jacobian(y, [x, z])
    f = theano.function([x, z], Js)
    vx = rng.uniform(size=(10,)).astype(theano.config.floatX)
    vz = rng.uniform(size=(10,)).astype(theano.config.floatX)
    vJs = f(vx, vz)
    evx = np.zeros((10, 10))
    evz = np.zeros((10, 10))
    np.fill_diagonal(evx, vx)
    np.fill_diagonal(evz, vz)
    assert np.allclose(vJs[0], evz)
    assert np.allclose(vJs[1], evx)


def test002_jacobian_matrix():
    x = tensor.matrix()
    y = 2 * x.sum(axis=0)
    rng = np.random.RandomState(seed=utt.fetch_seed())
    ev = np.zeros((10, 10, 10))
    for dx in xrange(10):
        ev[dx, :, dx] = 2.

    # test when the jacobian is called with a tensor as wrt
    Jx = tensor.jacobian(y, x)
    f = theano.function([x], Jx)
    vx = rng.uniform(size=(10, 10)).astype(theano.config.floatX)
    assert np.allclose(f(vx), ev)

    # test when the jacobian is called with a tuple as wrt
    Jx = tensor.jacobian(y, (x,))
    assert isinstance(Jx, tuple)
    f = theano.function([x], Jx[0])
    vx = rng.uniform(size=(10, 10)).astype(theano.config.floatX)
    assert np.allclose(f(vx), ev)

    # test when the jacobian is called with a list as wrt
    Jx = tensor.jacobian(y, [x])
    assert isinstance(Jx, list)
    f = theano.function([x], Jx[0])
    vx = rng.uniform(size=(10, 10)).astype(theano.config.floatX)
    assert np.allclose(f(vx), ev)

    # test when the jacobian is called with a list of two elements
    z = tensor.matrix()
    y = (x * z).sum(axis=1)
    Js = tensor.jacobian(y, [x, z])
    f = theano.function([x, z], Js)
    vx = rng.uniform(size=(10, 10)).astype(theano.config.floatX)
    vz = rng.uniform(size=(10, 10)).astype(theano.config.floatX)
    vJs = f(vx, vz)
    evx = np.zeros((10, 10, 10))
    evz = np.zeros((10, 10, 10))
    for dx in xrange(10):
        evx[dx, dx, :] = vx[dx, :]
        evz[dx, dx, :] = vz[dx, :]
    assert np.allclose(vJs[0], evz)
    assert np.allclose(vJs[1], evx)


def test003_jacobian_scalar():
    x = tensor.scalar()
    y = x * 2
    rng = np.random.RandomState(seed=utt.fetch_seed())

    # test when the jacobian is called with a tensor as wrt
    Jx = tensor.jacobian(y, x)
    f = theano.function([x], Jx)
    vx = np.cast[theano.config.floatX](rng.uniform())
    assert np.allclose(f(vx), 2)

    # test when the jacobian is called with a tuple as wrt
    Jx = tensor.jacobian(y, (x,))
    assert isinstance(Jx, tuple)
    f = theano.function([x], Jx[0])
    vx = np.cast[theano.config.floatX](rng.uniform())
    assert np.allclose(f(vx), 2)

    # test when the jacobian is called with a list as wrt
    Jx = tensor.jacobian(y, [x])
    assert isinstance(Jx, list)
    f = theano.function([x], Jx[0])
    vx = np.cast[theano.config.floatX](rng.uniform())
    assert np.allclose(f(vx), 2)

    # test when the jacobian is called with a list of two elements
    z = tensor.scalar()
    y = x * z
    Jx = tensor.jacobian(y, [x, z])
    f = theano.function([x, z], Jx)
    vx = np.cast[theano.config.floatX](rng.uniform())
    vz = np.cast[theano.config.floatX](rng.uniform())
    vJx = f(vx, vz)

    assert np.allclose(vJx[0], vz)
    assert np.allclose(vJx[1], vx)


def test004_hessian():
    x = tensor.vector()
    y = tensor.sum(x ** 2)
    Hx = tensor.hessian(y, x)
    f = theano.function([x], Hx)
    vx = np.arange(10).astype(theano.config.floatX)
    assert np.allclose(f(vx), np.eye(10) * 2)


def test_jacobian_disconnected_inputs():
    # Test that disconnected inputs are properly handled by jacobian.

    v1 = tensor.vector()
    v2 = tensor.vector()
    jacobian_v = theano.gradient.jacobian(1 + v1, v2, disconnected_inputs='ignore')
    func_v = theano.function([v1, v2], jacobian_v)
    val = np.arange(4.0).astype(theano.config.floatX)
    assert np.allclose(func_v(val, val), np.zeros((4, 4)))

    s1 = tensor.scalar()
    s2 = tensor.scalar()
    jacobian_s = theano.gradient.jacobian(1 + s1, s2, disconnected_inputs='ignore')
    func_s = theano.function([s2], jacobian_s)
    val = np.array(1.0).astype(theano.config.floatX)
    assert np.allclose(func_s(val), np.zeros(1))
