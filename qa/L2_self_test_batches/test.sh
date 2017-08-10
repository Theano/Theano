#!/bin/bash

cd /opt/theano

export THEANO_FLAGS="floatX=float32,device=cpu,mode=FAST_RUN"

exec python -m theano.tests.run_tests_in_batch
