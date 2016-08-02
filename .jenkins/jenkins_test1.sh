#!/bin/bash

# Script for Jenkins continuous integration testing of theano base and old backend

echo "===== Testing theano base"

# Get environment from worker, necessary for CUDA
source ~/.bashrc

# Test theano CPU and old GPU backend sandbox.cuda
THEANO_PARAM="theano --with-timer --timer-top-n 10"
FLAGS="mode=FAST_RUN,init_gpu_device=gpu"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}
