#!/bin/bash

(cd ../third_party/pylearn2 && pip install -e .)
(cd ../third_party/scikits.cuda && pip install .)

cd ../third_party/convnet-benchmarks/theano
SKIP=legacy exec python pylearn2_benchmark.py
