#!/bin/bash

# Script for Jenkins continuous integration testing of theano base

# Print commands as they are executed
set -x

# To make MKL work
export MKL_THREADING_LAYER=GNU

# Copy cache from master
BASECOMPILEDIR=$HOME/.theano/pr_theano
rsync -a $HOME/cache/ $HOME/.theano/pr_theano

echo "===== Testing theano core"

# Allow subprocess created by tests to find Theano.
# Keep it in the workspace
export PYTHONPATH=$PYTHONPATH:${WORKSPACE}

# Test theano core
PARTS="theano -e gpuarray"
THEANO_PARAM="${PARTS} --with-timer --timer-top-n 10 --with-xunit --xunit-file=theanocore_tests.xml"
FLAGS="mode=FAST_RUN,floatX=float32,on_opt_error=raise,on_shape_error=raise,cmodule.age_thresh_use=604800,base_compiledir=$BASECOMPILEDIR"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}
