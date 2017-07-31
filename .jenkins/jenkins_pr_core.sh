#!/bin/bash

# Script for Jenkins continuous integration testing of theano base

# Print commands as they are executed
set -x

# Copy cache from master
BASECOMPILEDIR=$HOME/.theano
rsync -a --delete $HOME/cache/master/ $BASECOMPILEDIR

echo "===== Testing theano core"

# Test theano core
PARTS="theano -e gpuarray"
THEANO_PARAM="${PARTS} --with-timer --timer-top-n 10 --with-xunit --xunit-file=theanocore_tests.xml"
FLAGS="mode=FAST_RUN,floatX=float32,on_opt_error=raise,on_shape_error=raise,base_compiledir=$BASECOMPILEDIR"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}
