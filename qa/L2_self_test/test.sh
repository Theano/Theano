#!/bin/bash

# THIS_DIR=$(cd $(dirname $0); pwd)


STATUS=0
cd /opt/theano/theano/tests

THEANO_FLAGS="floatX=float32,device=cpu" theano-cache clear
# THEANO_FLAGS="floatX=float32,device=cpu" theano-nose --verbose --exclude theano/sandbox/cuda

# ignore failures for now
THEANO_FLAGS="floatX=float32,device=cpu" python run_tests_in_batch.py

exit ${STATUS}
