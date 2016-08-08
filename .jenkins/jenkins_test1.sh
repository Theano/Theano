#!/bin/bash

# Script for Jenkins continuous integration testing of theano base

echo "===== Testing theano core"

# Get environment from worker, necessary for CUDA
source ~/.bashrc

# Test theano core
PARTS="theano -e cuda -e gpuarray"
THEANO_PARAM="${PARTS} --with-timer --timer-top-n 10"
FLAGS="mode=FAST_RUN,init_gpu_device=gpu,floatX=float32"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}
