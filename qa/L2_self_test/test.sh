#!/bin/bash

cd /opt/theano

THEANO_FLAGS="floatX=float32,device=cpu,init_gpu_device=gpu0,nvcc.fastmath=True" exec theano-nose --verbose
