import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_repeats",type=int,default=5)
args = parser.parse_args()

import sys
import time

import numpy as np

import theano
floatX = theano.config.floatX = 'float32'
import theano.tensor as T



_orig_img = T.tensor4("in_img")
_filt = T.tensor4("filt")
_filt_shape = T.lvector("filt_shape")
_img_shape = T.lvector("img_shape")
_out_img = theano.tensor.nnet.conv.conv2d(input=_orig_img, filters=_filt, filter_shape=(None,None,None,None), image_shape=(None,None,None,None),subsample=(2,2))
_loss = T.square(_out_img).sum()
_gradloss = T.grad(_loss,_filt)

fconv = theano.function([_filt,_orig_img],_loss)
fgradconv = theano.function([_filt,_orig_img],_gradloss)

n_repeats = args.n_repeats

titlestr = "%10s %10s %10s %10s %10s %10s %10s"%("batchsize","imgsize","kersize","inchan","outchan","meantime","stderrtime")
fmtstr   = "%10i %10i %10i %10i %10i %10.4f %10.4f"

for (fnname, fn) in [("conv",fconv),("gradconv",fgradconv)]:
    print "************ %s *************"%fnname
    print titlestr
    for (batchsize,imgsize,kersize,inchan,outchan) in [(1,200,8,1,1),(1,200,8,4,4),(1,200,8,16,16),(10,200,8,16,16),(20,200,8,16,16)]:
        orig_img = np.zeros((batchsize,inchan,imgsize,imgsize),floatX)
        filt = np.zeros((outchan,inchan,kersize,kersize),floatX)
        times = []
        for _ in xrange(n_repeats):
            t_start = time.time()
            fn(filt,orig_img)
            t_elapsed = time.time() - t_start
            times.append(t_elapsed)

        print fmtstr % (batchsize,imgsize,kersize,inchan,outchan,np.mean(times),np.std(times)/np.sqrt(n_repeats-1))



