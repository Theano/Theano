#!/bin/bash

cd /opt/theano

export THEANO_FLAGS="floatX=float32,device=cpu,lib.cnmem=0.45"

#
# theano-nose doesn't appear to support any of the variants of "exclude"
# that modern nosetests does.  As a result, a few errors and a few failures
# will appear when running this suite in theano.sandbox.cuda.tests.*.  Largely
# they are a result of running the tests for the old sandbox.cuda backend
# with the new gpuarray backend enabled.
#
#exec theano-nose --verbose --exclude theano/sandbox/cuda -a '!slow'

exec theano-nose --verbose -a '!slow'
