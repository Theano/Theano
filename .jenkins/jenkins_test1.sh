#!/bin/bash

# Script for Jenkins continuous integration testing of theano base

# Print commands as they are executed
set -x

# Anaconda python
export PATH=/usr/local/miniconda2/bin:$PATH

echo "===== Testing theano core"

# Test theano core
PARTS="theano -e cuda -e gpuarray"
THEANO_PARAM="${PARTS} --with-timer --timer-top-n 10 --with-xunit --xunit-file=theanocore_tests.xml"
FLAGS="mode=FAST_RUN,floatX=float32,on_opt_error=raise,on_shape_error=raise,cmodule.age_thresh_use=604800"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}
