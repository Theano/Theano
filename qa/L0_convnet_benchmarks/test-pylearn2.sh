#!/bin/bash

(cd ../third_party/pylearn2 && pip install -e .)

cd ../third_party/convnet-benchmarks/theano
export THEANO_FLAGS="gpuarray.preallocate=0,lib.cnmem=1"
SKIP=legacy exec python pylearn2_benchmark.py
