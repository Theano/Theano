#!/bin/bash

# Script for Jenkins continuous integration testing of theano base

echo "===== Testing theano core"

# Get environment from worker, necessary for CUDA
source ~/.bashrc

# Test theano core
PARTS="theano/compat theano/compile theano/d3viz theano/gof theano/misc theano/sandbox/linalg theano/sandbox/tests theano/scalar theano/scan_module theano/sparse theano/tensor theano/tests theano/typed_list"
THEANO_PARAM="${PARTS} --with-timer --timer-top-n 10"
FLAGS="mode=FAST_RUN,init_gpu_device=gpu,floatX=float32"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}
