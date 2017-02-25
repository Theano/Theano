#!/bin/bash

cd /opt/theano

THEANO_FLAGS="floatX=float32,device=cpu" exec theano-nose --verbose
