#!/bin/bash

cd /opt/theano

export THEANO_FLAGS="floatX=float32,device=cpu"
exec theano-nose --verbose --exclude theano/sandbox/cuda -a '!slow'
