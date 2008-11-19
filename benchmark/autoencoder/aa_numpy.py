#!/usr/bin/env python2.5
from __future__ import absolute_import
import numpy as N
import sys
import time

neg, nout, nhid, niter = [int(a) for a in sys.argv[1:]]
lr = 0.01

rng = N.random.RandomState(342)

w = rng.rand(nout, nhid)
a = rng.randn(nhid) * 0.0
b = rng.randn(nout) * 0.0
x = (rng.rand(neg, nout)-0.5) * 1.5


t = time.time()
for i in xrange(niter):
    hid = N.tanh(N.dot(x, w) + a)

    out = N.tanh(N.dot(hid, w.T) + b)

    g_out = out - x
    err = 0.5 * N.sum(g_out**2)

    g_hidwt = g_out * (1.0 - out**2)

    b -= lr * N.sum(g_hidwt, axis=0)

    g_hid = N.dot(g_hidwt, w)
    g_hidin = g_hid * (1.0 - hid**2)

    w -= lr * (N.dot(g_hidwt.T, hid) + N.dot(x.T, g_hidin))

    a -= lr * N.sum(g_hidin, axis=0)

print 'time: ',time.time() - t, 'err: ', err

