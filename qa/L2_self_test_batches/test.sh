#!/bin/bash

cd /opt/theano

export THEANO_FLAGS="floatX=float32,device=cpu,lib.cnmem=0.45"

exec python -m theano.tests.run_tests_in_batch
