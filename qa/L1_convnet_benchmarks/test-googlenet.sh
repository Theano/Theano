#!/bin/bash

(cd ../third_party/lasagne && pip install --no-deps -e .)

cd ../third_party/convnet-benchmarks/theano
python benchmark_imagenet.py --arch=googlenet
