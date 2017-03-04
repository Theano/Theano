#!/bin/bash

STATUS=0

THEANO_FLAGS="floatX=float32,device=cpu" theano-cache clear
THEANO_FLAGS="floatX=float32,device=cpu" theano-nose --verbose --exclude theano/sandbox/cuda

exit ${STATUS}
