#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import numpy as N
import sys
import time
from six.moves import xrange

# c: aa.cc

neg, nout, nhid, niter = [int(a) for a in sys.argv[1:]]
lr = 0.01

rng = N.random.RandomState(342)

w = rng.rand(nout, nhid)
a = rng.randn(nhid) * 0.0
b = rng.randn(nout) * 0.0
x = (rng.rand(neg, nout)-0.5) * 1.5

dot_time = 0.0

t = time.time()
for i in xrange(niter):
    tt = time.time()
    d = N.dot(x, w)
    dot_time += time.time() - tt

    hid = N.tanh(d + a)

    tt = time.time()
    d = N.dot(hid, w.T)
    dot_time += time.time() - tt
    out = N.tanh(d + b)

    g_out = out - x
    err = 0.5 * N.sum(g_out**2)

    g_hidwt = g_out * (1.0 - out**2)

    b -= lr * N.sum(g_hidwt, axis=0)

    tt = time.time()
    g_hid = N.dot(g_hidwt, w)
    dot_time += time.time() - tt

    g_hidin = g_hid * (1.0 - hid**2)

    tt = time.time()
    d = N.dot(g_hidwt.T, hid)
    dd = N.dot(x.T, g_hidin)
    dot_time += time.time() - tt

    gw = (d + dd)
    w -= lr * gw

    a -= lr * N.sum(g_hidin, axis=0)

total_time = time.time() - t
print('time: ',total_time, 'err: ', err)
print(' of which', dot_time, 'was spent on dot. Fraction:', dot_time / total_time)

