#!/bin/bash

(cd ../third_party/lasagne && pip install --no-deps -e .)

cd ../third_party/convnet-benchmarks/theano
exec python benchmark_imagenet.py --arch=alexnet
