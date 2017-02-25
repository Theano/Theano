#!/bin/bash

cd /opt/theano

THEANO_FLAGS="floatX=float32,device=cuda,init_gpu_device=cuda,nvcc.fastmath=True" exec theano-nose --verbose
