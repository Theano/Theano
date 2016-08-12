#!/bin/bash

# Script for Jenkins continuous integration testing of theano base

echo "===== Testing theano core"

# Test theano core
PARTS="theano -e cuda -e gpuarray"
THEANO_PARAM="${PARTS} --with-timer --timer-top-n 10"
FLAGS="mode=FAST_RUN,floatX=float32"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}
