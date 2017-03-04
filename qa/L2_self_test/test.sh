#!/bin/bash

cd /opt/theano/theano/tests

THEANO_FLAGS="floatX=float32,device=cpu" theano-cache clear

THEANO_FLAGS="floatX=float32,device=cpu" exec python run_tests_in_batch.py

